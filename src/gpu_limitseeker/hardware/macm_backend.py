"""Supervised GAO pressure reader with official MSI Afterburner MACM control.

No profiles, power settings, startup tasks, resets, or driver APIs are written.
GAO never writes clocks: only MACM set_offsets writes the scalar boost pair.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from ..models import SafetyError
from .macm import MACMError, Transport, guard_snapshot, validate_offsets


VENDOR_SHA256 = "f2390c2113329e8a72682fddca377d6f44a51d019f4f3c3366c45fcaba45ee7e"
SMI_FIELDS = ("uuid,driver_version,power.limit,power.default_limit,temperature.gpu,"
              "clocks.current.graphics,clocks.current.memory,utilization.gpu,power.draw,pstate")


class BackendError(SafetyError):
    pass


def _number(text: str, label: str, *, positive: bool = False) -> float:
    try:
        value = float(text)
    except (ValueError, TypeError) as exc:
        raise BackendError(f"Unreadable {label}: {text!r}") from exc
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise BackendError(f"Invalid {label}: {text!r}")
    return value


def parse_smi(text: str) -> dict:
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) != 1 or len(rows[0]) != 10:
        raise BackendError("NVIDIA telemetry must contain exactly one GPU and ten fields")
    uuid, driver, power, default, temp, gr, mem, util, draw, pstate = [s.strip() for s in rows[0]]
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid):
        raise BackendError("Missing or invalid GPU UUID")
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", driver):
        raise BackendError("Missing or invalid driver version")
    if not re.fullmatch(r"P(?:[0-9]|1[0-5])", pstate):
        raise BackendError("Missing or invalid NVIDIA performance state")
    values = {
        "uuid": uuid, "driver": driver,
        "power_w": _number(power, "power limit", positive=True),
        "default_power_w": _number(default, "default power limit", positive=True),
        "temp_c": _number(temp, "temperature"),
        "gr_clock": _number(gr, "graphics clock"),
        "mem_clock": _number(mem, "memory clock"),
        "utilization": _number(util, "GPU utilization"),
        "draw_w": _number(draw, "measured GPU power", positive=True),
        "pstate": pstate,
    }
    if values["temp_c"] > 125 or values["utilization"] > 100:
        raise BackendError("Implausible NVIDIA telemetry")
    return values


def parse_status(text: str) -> dict:
    applied = re.findall(
        r"^applied:\s+power (\d+) %, core ([+-]?\d+) MHz, mem ([+-]?\d+) MHz\s*$",
        text, re.MULTILINE)
    tuning = re.findall(r"^nvidia app:\s+Automatic Tuning (.+)$", text, re.MULTILINE)
    if len(applied) != 1 or len(tuning) != 1:
        raise BackendError("GAO did not report unambiguous actual offsets and tuning state")
    power, core, mem = map(int, applied[0])
    if power <= 0:
        raise BackendError("GAO reported invalid power percentage")
    setting = tuning[0].strip()
    if setting == "off":
        off = True
    elif setting == "ON" or setting.startswith("ON --"):
        off = False
    elif setting.startswith("unknown"):
        off = None
    else:
        raise BackendError("Unrecognized NVIDIA automatic tuning state")
    return {"core_mhz": core, "mem_mhz": mem, "power_pct": power, "nvidia_off": off}


def parse_probe(text: str) -> dict:
    counts = re.findall(r"^GPUs:\s*(\d+)\s*$", text, re.MULTILINE)
    ranges = re.findall(
        r"^offset ranges as the driver reports them: core ([+-]?\d+) to ([+-]?\d+) MHz, "
        r"mem ([+-]?\d+) to ([+-]?\d+) MHz\s*$", text, re.MULTILINE)
    if counts != ["1"] or len(ranges) != 1:
        raise BackendError("GAO requires one GPU with readable driver offset ranges")
    cmin, cmax, mmin, mmax = map(int, ranges[0])
    if not cmin <= 0 <= cmax or not mmin <= 0 <= mmax:
        raise BackendError("Invalid driver offset ranges")
    return {"core_min": cmin, "core_max": cmax, "mem_min": mmin, "mem_max": mmax}


def parse_curve(text: str) -> list[int]:
    counts = re.findall(r"^voltage/frequency curve: (\d+) points, \d+ to \d+ mV\s*$",
                        text, re.MULTILINE)
    rows = re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d+(?:\.\d+)?)\s+([+-]?\d+)\s*$",
                      text, re.MULTILINE)
    if len(counts) != 1 or int(counts[0]) < 2 or len(rows) != int(counts[0]):
        raise BackendError("Missing or incomplete voltage/frequency curve")
    if len({row[0] for row in rows}) != len(rows):
        raise BackendError("Duplicate voltage/frequency curve slots")
    if any(int(row[1]) <= 0 or float(row[2]) <= 0 for row in rows):
        raise BackendError("Invalid voltage/frequency curve points")
    return [int(row[3]) for row in rows]


def parse_curve_uniform(text: str) -> bool:
    return len(set(parse_curve(text))) == 1


def reconcile_controls(status: dict, mapped: dict, physical_curve: list[int]) -> dict:
    if physical_curve != mapped["curve_offsets_khz"]:
        raise BackendError("MACM cached curve disagrees with actual GAO driver curve; stale/TDR state")
    if status["power_pct"] != mapped["power_pct"]:
        raise BackendError("GAO and MACM power percentages disagree")
    return mapped | {"nvidia_off": status["nvidia_off"],
                     "pstates_core": status["core_mhz"], "pstates_mem": status["mem_mhz"]}


def guard_state(actual: dict, expected: dict, *, offsets: bool = True) -> None:
    keys = ["uuid", "driver", "power_w", "default_power_w"]
    if offsets:
        keys.extend(("core_mhz", "mem_mhz"))
    for key in keys:
        if key not in actual or key not in expected:
            raise BackendError(f"Missing state guard: {key}")
        if key in ("power_w", "default_power_w"):
            for source in (actual, expected):
                value = source[key]
                if isinstance(value, bool) or not isinstance(value, (float, int)):
                    raise BackendError(f"Invalid state guard: {key}")
                if not math.isfinite(value) or value <= 0:
                    raise BackendError(f"Invalid state guard: {key}")
        elif key in ("core_mhz", "mem_mhz"):
            if type(actual[key]) is not int or type(expected[key]) is not int:
                raise BackendError(f"Invalid offset guard: {key}")
        elif not isinstance(actual[key], str) or not actual[key] or not isinstance(expected[key], str):
            raise BackendError(f"Invalid identity guard: {key}")
        if actual[key] != expected[key]:
            raise BackendError(f"Actual {key} drifted: expected {expected[key]!r}, got {actual[key]!r}")
    if offsets:
        for key in ("macm_gpu_id", "macm_index", "macm_controls_digest",
                    "pstates_core", "pstates_mem", "curve_digest"):
            if key in actual or key in expected:
                if key not in actual or key not in expected or actual[key] != expected[key]:
                    raise BackendError(f"Control-layer {key} drifted or is missing")


def process_blockers(processes: list[dict], search: bool = False) -> list[str]:
    pressure = re.compile(
        r"(?:occt|3dmark|furmark|kombustor|prime95|"
        r"unigine|superposition|heaven|valley|gpu.?burn|memtest.?vulkan|"
        r"y.?cruncher|aida64|msi.?kombustor|benchmark|stress|gpuautooptimizer|gpu-auto-optimizer)", re.I)
    blockers = []
    controllers = 0
    for process in processes:
        name = process.get("ProcessName")
        pid = process.get("Id")
        if not isinstance(name, str) or not name or type(pid) is not int:
            raise BackendError("Unreadable process inventory")
        stem = name.lower().removesuffix(".exe")
        if stem == "msiafterburner":
            controllers += 1
        if pressure.search(stem) or stem in {"gao", "gao-gui"}:
            blockers.append(f"{name} (PID {pid})")
        elif search and re.search(
                r"nvidia.?inspector|precision.?x|gpu.?tweak|firestorm|thunder.?master|aorus.?engine|oc.?scanner",
                stem, re.I):
            blockers.append(f"{name} (PID {pid}; another clock controller must be closed manually)")
    if search and controllers != 1:
        blockers.append("Exactly one MSI Afterburner must remain running as the sole MACM controller")
    return sorted(set(blockers))


def classify_result(returncode: int, text: str, *, events: list | None,
                    state_ok: bool, forced: str | None = None, details: str = "",
                    max_temp: int | None = None) -> dict:
    verdicts = re.findall(
        r"^VERDICT: (STABLE|WRONG RESULT|DEVICE LOST|STALLED|TOO HOT|NO TELEMETRY|ABORTED)"
        r"\s+score=([0-9]+(?:\.[0-9]+)?) it/s\s+[^\r\n]+$", text, re.MULTILINE)
    device_failure = re.search(
        r"DEVICE LOST|DXGI_ERROR_DEVICE_(?:REMOVED|HUNG|RESET)|VERDICT: STALLED", text, re.I)
    score = None
    if len(verdicts) == 1:
        parsed = float(verdicts[0][1])
        if math.isfinite(parsed):
            score = parsed
    if events or device_failure:
        outcome, reason = "tdr", "GPU device loss/stall or new Windows display-driver event"
    elif forced is not None:
        outcome, reason = forced, details
    elif events is None:
        outcome, reason = "invalid", "Windows driver event query was unavailable"
    elif not state_ok:
        outcome, reason = "invalid", "Actual hardware state could not be verified unchanged"
    elif len(verdicts) != 1 or score is None:
        outcome, reason = "invalid", "Missing, malformed or ambiguous GAO verdict"
    else:
        verdict = verdicts[0][0]
        mapping = {(0, "STABLE"): "pass", (2, "WRONG RESULT"): "wrong",
                   (3, "TOO HOT"): "hot", (3, "NO TELEMETRY"): "invalid",
                   (1, "ABORTED"): "stopped"}
        outcome = mapping.get((returncode, verdict), "invalid")
        reason = f"GAO {verdict}; exit {returncode}"
        if max_temp is not None and outcome in ("pass", "wrong"):
            peaks = re.findall(r"^VERDICT: [^\r\n]+?\bpeak=(\d+(?:\.\d+)?) C\b", text, re.MULTILINE)
            if len(peaks) != 1:
                outcome, reason = "invalid", "GAO did not report a readable peak temperature"
            elif float(peaks[0]) >= max_temp:
                outcome, reason = "hot", f"GAO peak reached {peaks[0]} C"
    if details and reason != details:
        reason += "; " + details
    return {"outcome": outcome, "details": reason, "score": score}


def wait_for_idle(read, stopped, *, timeout=30, interval=1,
                  clock=time.monotonic, pause=time.sleep):
    """Require two quiet samples; distinguish low-power desktop P8 from GPU work."""
    deadline = clock() + timeout
    consecutive = 0
    while True:
        if stopped():
            return None
        sample = read()
        utilization = _number(str(sample.get("utilization")), "GPU utilization")
        if not 0 <= utilization <= 100:
            raise BackendError("Unknown GPU utilization; no load started")
        if clock() > deadline:
            raise BackendError("GPU did not become idle before the admission deadline; no load started")
        quiet = utilization <= 15
        if not quiet and sample.get("pstate") == "P8":
            draw = _number(str(sample.get("draw_w")), "measured GPU power", positive=True)
            default = _number(str(sample.get("default_power_w")), "default power", positive=True)
            memory = _number(str(sample.get("mem_clock")), "memory clock")
            quiet = draw <= min(60, default * 0.15) and memory <= 2000
        consecutive = consecutive + 1 if quiet else 0
        if consecutive >= 2:
            return sample
        if clock() >= deadline:
            raise BackendError("GPU did not become idle before the admission deadline; no load started")
        pause(min(interval, max(0, deadline - clock())))


class Backend:
    def __init__(self, root: Path, vendor: Path):
        self.root = Path(root).resolve()
        self.vendor = Path(vendor)
        self._sequence = 0
        self.macm = Transport()
        self._check = None
        self._owned_pressure = None

    @staticmethod
    def _creation_ticks(child):
        """Read creation identity from our retained process handle, never by PID."""
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(int(child._handle), ctypes.byref(created), ctypes.byref(exited),
                                      ctypes.byref(system), ctypes.byref(user)):
            raise BackendError("Owned pressure creation identity is unreadable")
        # FILETIME starts in 1601; inventory StartTicks uses .NET's year-1 epoch.
        return str((created.dwHighDateTime << 32) + created.dwLowDateTime + 504911232000000000)

    def _claim_pressure(self, child, executable):
        if self._owned_pressure is not None:
            raise BackendError("Another pressure child is already owned")
        ticks = self._creation_ticks(child)
        self._owned_pressure = (child, Path(executable).stem.lower(), child.pid, ticks)

    def _release_pressure(self, child):
        if self._owned_pressure is not None and self._owned_pressure[0] is child:
            self._owned_pressure = None

    def _pressure_inventory(self, inventory):
        """Exclude only our still-live handle's exact name/PID/creation identity."""
        owned = self._owned_pressure
        if owned is None or owned[0].poll() is not None:
            return inventory
        child, name, pid, ticks = owned
        filtered = [p for p in inventory if not (
            isinstance(p.get("ProcessName"), str) and
            p["ProcessName"].lower().removesuffix(".exe") == name and
            type(p.get("Id")) is int and p["Id"] == pid and p.get("StartTicks") == ticks)]
        return filtered if child.poll() is None else inventory

    def verify_vendor(self) -> None:
        try:
            with self.vendor.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as exc:
            raise BackendError(f"Cannot read pinned GAO executable: {exc}") from exc
        if digest != VENDOR_SHA256:
            raise BackendError("Pinned GAO SHA256 mismatch; execution refused")

    def _windows(self) -> None:
        if os.name != "nt":
            raise BackendError("This backend supports Windows only")
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():
            raise BackendError("Live NVIDIA operations require Windows administrator privileges; elevate manually")

    def _powershell(self) -> str:
        self._windows()
        return str(Path(os.environ.get("SystemRoot", r"C:\Windows")) /
                   "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")

    def _smi_path(self) -> str:
        self._windows()
        paths = [Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
                 Path(os.environ.get("ProgramFiles", r"C:\Program Files")) /
                 "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"]
        for path in paths:
            if path.is_file():
                return str(path)
        raise BackendError("Official NVIDIA nvidia-smi.exe was not found in its installed locations")

    def _log_path(self, directory: Path, label: str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._sequence += 1
        return directory / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}-{self._sequence:04d}-{label}.log"

    def _run(self, args: list[str], label: str, log_dir: Path | None = None,
             timeout: float = 8) -> str:
        self._windows()
        try:
            result = subprocess.run(args, cwd=self.root, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    timeout=timeout, shell=False,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or b"").decode("utf-8-sig", errors="replace")
            if log_dir is not None:
                self._log_path(log_dir, label).write_text(
                    f"COMMAND: {args!r}\nTIMEOUT after {timeout}s\n{output}", encoding="utf-8")
            raise BackendError(f"{label} timed out after {timeout}s") from exc
        except OSError as exc:
            raise BackendError(f"Cannot execute {label}: {exc}") from exc
        output = result.stdout.decode("utf-8-sig", errors="replace")
        if log_dir is not None:
            self._log_path(log_dir, label).write_text(
                f"COMMAND: {args!r}\nEXIT: {result.returncode}\n{output}", encoding="utf-8")
        if result.returncode != 0:
            raise BackendError(f"{label} failed (exit {result.returncode}): {output[-1500:]}")
        return output

    def blockers(self, search: bool = False) -> list[str]:
        script = ("$ErrorActionPreference='Stop'; [Console]::OutputEncoding="
                  "[Text.UTF8Encoding]::new($false); "
                  "$p=@(Get-Process | Select-Object ProcessName,Id); "
                  "ConvertTo-Json -InputObject $p -Compress")
        text = self._run([self._powershell(), "-NoProfile", "-NonInteractive", "-Command", script],
                         "processes")
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise BackendError("Could not parse Windows process inventory") from exc
        if not isinstance(data, list) or not data or any(not isinstance(p, dict) for p in data):
            raise BackendError("Empty or invalid Windows process inventory")
        return process_blockers(data, search)

    def _telemetry(self, log_dir: Path | None = None, timeout: float = 5) -> dict:
        return parse_smi(self._run([self._smi_path(), f"--query-gpu={SMI_FIELDS}",
                                    "--format=csv,noheader,nounits"], "nvidia-smi", log_dir, timeout=timeout))

    def _controls(self, log_dir: Path | None = None, timeout: float = 8) -> dict:
        deadline = time.monotonic() + timeout
        try:
            mapped = self.macm.snapshot(wait_ms=max(1, int(min(2, timeout) * 1000)))
            status = parse_status(self._run(
                [str(self.vendor), "--status"], "gao-status", log_dir,
                timeout=max(0.05, deadline - time.monotonic())))
            curve = parse_curve(self._run(
                [str(self.vendor), "--curve"], "gao-curve", log_dir,
                timeout=max(0.05, deadline - time.monotonic())))
            after = self.macm.snapshot(wait_ms=max(1, int(min(2, max(0, deadline - time.monotonic())) * 1000)))
            guard_snapshot(after, mapped)
        except MACMError as exc:
            raise BackendError(str(exc)) from exc
        result = reconcile_controls(status, after, curve)
        if log_dir is not None:
            self._log_path(log_dir, "macm-readonly").write_text(
                json.dumps(result, indent=2), encoding="utf-8")
        return result

    def inspect(self, log_dir: Path | None = None) -> dict:
        self.verify_vendor()
        before = self._telemetry(log_dir)
        controls = self._controls(log_dir)
        probe = parse_probe(self._run([str(self.vendor), "--probe"], "gao-probe", log_dir))
        after = self._telemetry(log_dir)
        guard_state(after, before, offsets=False)
        if abs(controls["power_pct"] - after["power_w"] * 100 / after["default_power_w"]) > 1:
            raise BackendError("MACM/GAO/NVIDIA power limit readbacks disagree")
        result = after | controls
        for name in ("core", "mem"):
            result[name + "_min"] = max(controls[name + "_min"], probe[name + "_min"])
            result[name + "_max"] = min(controls[name + "_max"], probe[name + "_max"])
            if not result[name + "_min"] <= result[name + "_mhz"] <= result[name + "_max"]:
                raise BackendError("Actual boost offsets are outside the jointly reported ranges")
        return result

    def curve_uniform(self, log_dir: Path) -> bool:
        self.verify_vendor()
        return parse_curve_uniform(self._run([str(self.vendor), "--curve"], "gao-curve", log_dir))

    def _require_clear(self, search: bool) -> None:
        blocked = self.blockers(search)
        if blocked:
            raise BackendError("Close competing workloads manually: " + "; ".join(blocked))

    def _stopped(self, stop_file: Path) -> bool:
        if self._check is not None:
            self._check()
        return Path(stop_file).exists() or (self.root / "STOP").exists()

    def set_offsets(self, core: int, mem: int, expected: dict, log_dir: Path) -> dict:
        self.verify_vendor()
        if (self.root / "STOP").exists():
            raise BackendError("STOP requested; no offsets written")
        self._require_clear(True)
        actual = self.inspect(log_dir)
        guard_state(actual, expected)
        if actual["nvidia_off"] is not True:
            raise BackendError("NVIDIA automatic tuning must be verifiably OFF")
        if actual["pstates_core"] != 0 or actual["pstates_mem"] != 0:
            raise BackendError("Nonzero GAO Pstates offsets would stack with Afterburner BOOST TABLE; search refused")
        try:
            validate_offsets(core, mem, actual)
            # Parent journals before calling. The official transport verifies all
            # non-offset controls under its mutex and submits exactly one FLUSH.
            applied = self.macm.set_offsets(
                core, mem, actual, lambda: (self.root / "STOP").exists())
        except MACMError as exc:
            raise BackendError(f"MACM write failed; no replay/restore: {exc}") from exc
        self._log_path(log_dir, "macm-flush-readback").write_text(
            json.dumps(applied, indent=2), encoding="utf-8")
        after = self.inspect(log_dir)
        target = actual | applied | {"core_mhz": core, "mem_mhz": mem}
        guard_state(after, target)
        if after["nvidia_off"] is not True:
            raise BackendError("NVIDIA automatic tuning changed during MACM FLUSH")
        return after

    def _driver_events(self, since: datetime, log_dir: Path, timeout: float = 8) -> list:
        stamp = since.astimezone(timezone.utc).isoformat()
        script = (
            "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
            f"$start=[DateTimeOffset]::Parse('{stamp}').LocalDateTime; "
            "try { $ev=@(Get-WinEvent -FilterHashtable @{LogName='System'; "
            "StartTime=$start; Id=153,13,14,4101} -ErrorAction Stop | "
            "Where-Object {$_.ProviderName -in @('nvlddmkm','Display')} | "
            "Select-Object Id,RecordId,ProviderName,TimeCreated,Message) } "
            "catch { if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') {$ev=@()} else {throw} }; "
            "ConvertTo-Json -InputObject @($ev) -Depth 4 -Compress")
        text = self._run([self._powershell(), "-NoProfile", "-NonInteractive", "-Command", script],
                         "driver-events", log_dir, timeout=timeout)
        try:
            events = json.loads(text)
        except ValueError as exc:
            raise BackendError("Unreadable Windows driver event query") from exc
        if not isinstance(events, list) or any(not isinstance(e, dict) or
                e.get("Id") not in (153, 13, 14, 4101) or
                e.get("ProviderName") not in ("nvlddmkm", "Display") for e in events):
            raise BackendError("Invalid Windows driver event query result")
        return events

    @staticmethod
    def _cleanup(child: subprocess.Popen | None) -> None:
        if child is None or child.poll() is not None:
            return
        child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired as exc:
                raise BackendError("Own stress child did not exit after kill") from exc

    def run_test(self, seconds: int, duty: int, max_temp: int, out_dir: Path,
                 expected: dict, stop_file: Path) -> dict:
        if type(seconds) is not int or not 1 <= seconds <= 3600:
            raise BackendError("Stress seconds must be an integer in 1..3600")
        if type(duty) is not int or not 5 <= duty <= 100:
            raise BackendError("Stress duty must be an integer in 5..100")
        if type(max_temp) is not int or not 1 <= max_temp <= 80:
            raise BackendError("Temperature stop must be an integer no higher than 80 C")
        child = None
        stress_path = None
        started = None
        forced = None
        details = ""
        state_ok = False
        events = None
        samples = []
        text = ""
        rc = -1
        elapsed = 0.0
        try:
            self.verify_vendor()
            if self._stopped(stop_file):
                return {"outcome": "stopped", "details": "STOP requested before pressure", "score": None}
            self._require_clear(False)
            # Prove event access before loading the GPU. Keep this window open
            # across all final preflight checks, so intervening resets count.
            started = datetime.now(timezone.utc)
            events = self._driver_events(started, out_dir)
            if events:
                return classify_result(-1, "", events=events, state_ok=False)
            self._require_clear(False)
            if wait_for_idle(lambda: self._telemetry(out_dir), lambda: self._stopped(stop_file)) is None:
                return {"outcome": "stopped", "details": "STOP requested during idle admission", "score": None}
            self._require_clear(False)
            before = self.inspect(out_dir)
            guard_state(before, expected)
            if before["nvidia_off"] is not True:
                raise BackendError("NVIDIA automatic tuning must be verifiably OFF")
            if before["temp_c"] >= max_temp:
                return {"outcome": "hot", "details": "Already at temperature stop before pressure", "score": None}
            if self._stopped(stop_file):
                return {"outcome": "stopped", "details": "STOP requested before pressure", "score": None}
            events = self._driver_events(started, out_dir)
            if events:
                return classify_result(-1, "", events=events, state_ok=False)
            stress_path = self._log_path(out_dir, "gao-stress")
            args = [str(self.vendor), "--stress", str(seconds), "--max-temp", str(max_temp), "--duty", str(duty)]
            deadline = time.monotonic() + seconds + 45
            load_start = time.monotonic()
            with stress_path.open("wb") as output:
                output.write((f"COMMAND: {args!r}\n").encode("utf-8"))
                output.flush()
                child = subprocess.Popen(args, cwd=self.root, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT, shell=False,
                                         creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP |
                                                        subprocess.CREATE_NO_WINDOW))
                self._claim_pressure(child, self.vendor)
                next_sample = next_event = time.monotonic()
                while child.poll() is None:
                    now = time.monotonic()
                    if self._stopped(stop_file):
                        forced, details = "stopped", "STOP requested"
                        break
                    if now >= deadline:
                        forced, details = "invalid", "Stress watchdog expired (duration +45s)"
                        break
                    if now >= next_sample:
                        sample = self._telemetry(out_dir, timeout=max(0.05, min(5, deadline - now)))
                        samples.append(sample)
                        guard_state(sample, expected, offsets=False)
                        if sample["temp_c"] >= max_temp:
                            forced, details = "hot", f"Temperature reached {sample['temp_c']} C"
                            break
                        next_sample = time.monotonic() + 2
                    now = time.monotonic()
                    if now >= deadline or self._stopped(stop_file):
                        continue
                    if now >= next_event and child.poll() is None:
                        events = self._driver_events(
                            started, out_dir, timeout=max(0.05, min(8, deadline - now)))
                        if events:
                            forced, details = "tdr", "New Windows display-driver events"
                            break
                        now = time.monotonic()
                        if now >= deadline or self._stopped(stop_file):
                            continue
                        if child.poll() is None:
                            status = self._controls(
                                out_dir, timeout=max(0.05, min(5, deadline - now)))
                            guard_state(samples[-1] | status, expected)
                            if status["nvidia_off"] is not True:
                                raise BackendError("NVIDIA automatic tuning changed during pressure")
                        next_event = time.monotonic() + 10
                    time.sleep(0.2)
                elapsed = time.monotonic() - load_start
                self._cleanup(child)
                rc = child.returncode
        except KeyboardInterrupt:
            forced, details = "stopped", "Interrupted by operator"
        except (BackendError, OSError, ValueError) as exc:
            forced, details = "invalid", str(exc)
        finally:
            try:
                self._cleanup(child)
            except (BackendError, OSError) as exc:
                forced, details = "invalid", f"Own-child cleanup failed: {exc}"
            self._release_pressure(child)
            if child is not None:
                rc = child.returncode if child.returncode is not None else -1
            if stress_path is not None:
                try:
                    text = stress_path.read_text(encoding="utf-8-sig", errors="replace")
                except OSError as exc:
                    forced, details = "invalid", f"Could not read stress output: {exc}"
            if child is not None and started is not None:
                # Both checks are attempted independently even after STOP or telemetry failure.
                try:
                    events = self._driver_events(started, out_dir)
                except (BackendError, OSError, ValueError) as exc:
                    events = None
                    if forced not in ("tdr", "stopped"):
                        forced = "invalid"
                    details += f"; post-test driver event query failed: {exc}"
                try:
                    after = self.inspect(out_dir)
                    guard_state(after, expected)
                    if after["nvidia_off"] is not True:
                        raise BackendError("NVIDIA automatic tuning no longer verifiably OFF")
                    state_ok = True
                    if after["temp_c"] >= max_temp and forced is None:
                        forced, details = "hot", f"Post-test temperature reached {after['temp_c']} C"
                except (BackendError, OSError, ValueError) as exc:
                    details += f"; post-test state verification failed: {exc}"
                if not samples and forced is None:
                    forced, details = "invalid", "No independent in-test telemetry sample"
                if forced is None and elapsed < seconds:
                    forced, details = "invalid", "Stress exited before the requested minimum wall time"
        result = classify_result(rc, text, events=events, state_ok=state_ok,
                                 forced=forced, details=details, max_temp=max_temp)
        result["telemetry"] = {"samples": len(samples),
                               "peak_temp_c": max((s["temp_c"] for s in samples), default=None),
                               "peak_mem_clock_mhz": max((s["mem_clock"] for s in samples if s["utilization"] > 15), default=None),
                               "peak_gr_clock_mhz": max((s["gr_clock"] for s in samples if s["utilization"] > 15), default=None),
                               "avg_power_w": sum(s["draw_w"] for s in samples) / len(samples) if samples else None,
                               "peak_power_w": max((s["draw_w"] for s in samples), default=None)}
        result["returncode"] = rc
        result["load_started"] = child is not None
        if stress_path is not None:
            result["log"] = str(stress_path)
        return result
