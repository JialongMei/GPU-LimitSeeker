"""Pinned GAO standalone control; no MACM transport or hardware calls on import.

Pstates base offsets and the physical V/F vector are deliberately distinct.
A failed write is never replayed. Only an explicit reset plus verified uniform
pair clears uncertainty; the caller must journal and restore before another load.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

from .macm_backend import (Backend, BackendError, guard_state, parse_probe, parse_status,
                           process_blockers, wait_for_idle)

_CONTROLLER = re.compile(
    r"msiafterburner|nvidia.?inspector|precision.?x|gpu.?tweak|firestorm|"
    r"thunder.?master|aorus.?engine|oc.?scanner|gpuautooptimizer|gpu-auto-optimizer|gao-gui", re.I)


def parse_full_curve(text: str) -> list[dict]:
    """Parse every enabled slot of v0.4.9 print_curve, never a partial table."""
    headers = list(re.finditer(
        r"^voltage/frequency curve: (\d+) points, (\d+) to (\d+) mV\s*$",
        text, re.MULTILINE))
    if len(headers) != 1:
        raise BackendError("Missing or ambiguous full V/F curve")
    header = headers[0]
    count, low, high = map(int, header.groups())
    if not 2 <= count <= 255 or not 0 < low <= high:
        raise BackendError("Invalid V/F curve dimensions")
    lines = text[header.end():].splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines or lines.pop(0).split() != ["slot", "mV", "MHz", "raw", "offset"]:
        raise BackendError("Missing V/F column header")
    points = []
    for line in lines[:count]:
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s+(\d+\.\d)\s+([+-]?\d+)\s*", line)
        if match is None:
            raise BackendError("Malformed or incomplete V/F curve row")
        slot, mv, mhz, raw = match.groups()
        point = {"index": int(slot), "mv": int(mv), "mhz": float(mhz), "raw_offset": int(raw)}
        if not (0 <= point["index"] < 255 and point["mv"] > 0 and
                0 < point["mhz"] <= 10000 and -(2**31) <= point["raw_offset"] < 2**31):
            raise BackendError("Invalid V/F point")
        points.append(point)
    if len(points) != count or len({p["index"] for p in points}) != count:
        raise BackendError("Incomplete or duplicate V/F slots")
    if any(a["mv"] > b["mv"] for a, b in zip(points, points[1:])):
        raise BackendError("V/F curve is not voltage ordered")
    if points[0]["mv"] != low or points[-1]["mv"] != high:
        raise BackendError("V/F range disagrees with full table")
    if any(line.strip() for line in lines[count:]):
        raise BackendError("Unexpected trailing data beyond full V/F table")
    return points


def curve_digest(points: list[dict]) -> str:
    # Frequency is temperature dependent; the stored shape is not.
    vector = [(p["index"], p["mv"], p["raw_offset"]) for p in points]
    return hashlib.sha256(json.dumps(vector, separators=(",", ":")).encode("ascii")).hexdigest()


def parse_bandwidth(text: str, expected_name: str, max_temp=80) -> float:
    """Only the exact final CLI measurement has bandwidth units, not score."""
    results = re.findall(r"^memory bandwidth: (\d+\.\d) GB/s \(([^\r\n]+)\)\s*$",
                         text, re.MULTILINE)
    if len(results) != 1 or len(re.findall(r"^memory bandwidth:", text, re.MULTILINE)) != 1:
        raise BackendError("Missing or ambiguous GB/s bandwidth result")
    value, adapter = results[0]
    if adapter != expected_name or not adapter.startswith("NVIDIA "):
        raise BackendError("Bandwidth ran on an unexpected adapter")
    gbps = float(value)
    if not math.isfinite(gbps) or gbps <= 0:
        raise BackendError("Invalid bandwidth measurement")
    loads = re.findall(r"^\s+load: (\w+)\s+score=\d+(?:\.\d+)? it/s\s+"
                       r"(\d+\.\d) s\s+peak=(\d+(?:\.\d+)?) C\b", text, re.MULTILINE)
    if len(loads) != 1 or loads[0][0] != "STABLE" or float(loads[0][1]) < 5:
        raise BackendError("Bandwidth warmup did not report five stable seconds")
    if float(loads[0][2]) >= max_temp:
        raise BackendError("Bandwidth warmup reached temperature stop")
    runs = re.findall(r"^\s+runs: (\d+), ([^\r\n]+)$", text, re.MULTILINE)
    each = re.findall(r"^\s+each:([^\r\n]+)$", text, re.MULTILINE)
    if len(runs) != 1 or len(each) != 1:
        raise BackendError("Missing bandwidth run diagnostics")
    values = each[0].split()
    if len(values) != int(runs[0][0]) or not values:
        raise BackendError("Incomplete bandwidth run diagnostics")
    readings = []
    for item in values:
        if item == "-":
            continue
        if not re.fullmatch(r"\d+\.\d", item) or float(item) <= 0:
            raise BackendError("Malformed bandwidth run")
        readings.append(float(item))
    if not readings or not min(readings) - 0.1 <= gbps <= max(readings) + 0.1:
        raise BackendError("Bandwidth result disagrees with measured runs")
    return gbps


def standalone_blockers(processes: list[dict], *, controllers_only: bool = False) -> list[str]:
    # process_blockers validates every inventory row, even in controller-only mode.
    pressure = process_blockers(processes, search=False)
    controllers = [f"{p['ProcessName']} (PID {p['Id']}; external clock controller)"
                   for p in processes if _CONTROLLER.search(p["ProcessName"])]
    return sorted(set(controllers if controllers_only else pressure + controllers))


class NightBackend(Backend):
    def __init__(self, root: Path, vendor: Path, identity: dict, memory_base_mhz=None,
                 power_percent=100, max_temp=80):
        super().__init__(root, vendor)
        self.uuid, self.driver = identity["uuid"], identity["driver"]
        self.identity = {key: identity[key] for key in
                         ("uuid", "driver", "power_w", "default_power_w")}
        self.name = identity["name"]
        self.power_percent = power_percent
        self.max_power_percent = power_percent
        self.max_temp = max_temp
        self.memory_base_mhz = memory_base_mhz
        self.derive_memory_reference = False
        self._uncertain = False
        self._pressure_samples = None
        self._curve_target = None
        self.thermal_latched = False
        self._raw_restore_digest = None
        self.max_board_power_w = None

    def _inventory(self, timeout=8) -> list[dict]:
        script = ("$ErrorActionPreference='Stop'; [Console]::OutputEncoding="
                  "[Text.UTF8Encoding]::new($false); "
                  "$p=@(Get-Process | Select-Object ProcessName,Id,@{Name='StartTicks';Expression={try {$_.StartTime.ToUniversalTime().Ticks.ToString()} catch {$null}}}); "
                  "ConvertTo-Json -InputObject $p -Compress")
        text = self._run([self._powershell(), "-NoProfile", "-NonInteractive", "-Command", script],
                         "processes", timeout=timeout)
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise BackendError("Unreadable Windows process inventory") from exc
        if not isinstance(data, list) or not data or any(not isinstance(p, dict) for p in data):
            raise BackendError("Empty or invalid Windows process inventory")
        return data

    def blockers(self, search: bool = False) -> list[str]:
        return standalone_blockers(self._inventory())

    def _no_controller(self, timeout=8):
        blocked = standalone_blockers(self._inventory(timeout), controllers_only=True)
        if blocked:
            raise BackendError("Afterburner and external clock controllers must stay closed: " + "; ".join(blocked))

    def _identity(self, snapshot: dict):
        guard_state(snapshot, self.identity, offsets=False)

    def _telemetry(self, log_dir: Path | None = None, timeout=None) -> dict:
        sample = super()._telemetry(log_dir, timeout=5 if timeout is None else timeout)
        self._identity(sample)
        if sample["temp_c"] >= self.max_temp:
            self.thermal_latched = True
        if self.max_board_power_w is not None and sample["draw_w"] > self.max_board_power_w:
            raise BackendError("Observed board power exceeded the explicitly approved watt cap")
        # Backend.run_test supplies an explicit bounded timeout only for its
        # in-pressure samples, not admission/inspect/postflight telemetry.
        if self._pressure_samples is not None and timeout is not None:
            self._pressure_samples.append(sample)
        return sample

    def _controls(self, log_dir: Path | None = None, timeout: float = 8) -> dict:
        deadline = time.monotonic() + timeout
        def remaining():
            left = deadline - time.monotonic()
            if left <= 0:
                raise BackendError("Standalone control readback deadline expired")
            return left
        self._no_controller(remaining())
        first = parse_status(self._run([str(self.vendor), "--status"], "gao-status", log_dir, remaining()))
        points = parse_full_curve(self._run([str(self.vendor), "--curve"], "gao-curve", log_dir, remaining()))
        second = parse_status(self._run([str(self.vendor), "--status"], "gao-status-confirm", log_dir, remaining()))
        if first != second:
            raise BackendError("Pstates/power/tuning changed across independent curve readback")
        if second["nvidia_off"] is not True or second["power_pct"] != self.power_percent:
            raise BackendError("NVIDIA automatic tuning/power disagrees with the approved envelope")
        self._no_controller(remaining())
        return second | {"curve_digest": curve_digest(points), "curve_points_data": points}

    def inspect(self, log_dir: Path | None = None) -> dict:
        self.verify_vendor()
        before = self._telemetry(log_dir)
        controls = self._controls(log_dir)
        probe = parse_probe(self._run([str(self.vendor), "--probe"], "gao-probe", log_dir))
        after = self._telemetry(log_dir)
        guard_state(after, before, offsets=False)
        result = after | controls | probe
        for kind in ("core", "mem"):
            if not result[kind + "_min"] <= result[kind + "_mhz"] <= result[kind + "_max"]:
                raise BackendError("Pstates offsets outside reported driver bounds")
        if self._curve_target is not None and result["curve_digest"] == self._curve_target["digest"]:
            result.update(curve_anchor_mv=self._curve_target["mv"], curve_anchor_mhz=self._curve_target["mhz"])
        return result

    def curve_uniform(self, log_dir: Path) -> bool:
        return len({p["raw_offset"] for p in self.inspect(log_dir)["curve_points_data"]}) == 1

    def _ready(self):
        if self._uncertain:
            raise BackendError("Uncertain control write; explicit reset plus verified uniform pair required")

    @staticmethod
    def _bounds(core, mem, snapshot):
        for kind, value in (("core", core), ("mem", mem)):
            if type(value) is not int or not 0 <= value <= snapshot[kind + "_max"]:
                raise BackendError(f"Invalid nonnegative {kind} offset or driver bound")

    def _approve_power_limit(self, percent):
        if type(percent) is not int or not 1 <= percent <= self.max_power_percent:
            raise BackendError("Power write exceeds the explicitly approved percentage envelope")
        watts = self.identity["default_power_w"] * percent / 100
        if self.max_board_power_w is not None and watts > self.max_board_power_w:
            raise BackendError("Power write exceeds the explicitly approved board-watt envelope")

    def _write_allowed(self):
        if (self.root / "STOP").exists():
            raise BackendError("STOP requested; no clocks written")
        self._require_clear(True)

    def apply_uniform(self, core: int, mem: int, log_dir: Path, reset: bool = False) -> dict:
        if type(reset) is not bool:
            raise BackendError("reset must be boolean")
        if not reset:
            self._ready()
        self._write_allowed()
        before = self.inspect(log_dir)
        self._bounds(core, mem, before)
        if not reset and any(p["raw_offset"] != before["core_mhz"] * 1000
                             for p in before["curve_points_data"]):
            raise BackendError("Custom/non-normalized V/F curve requires explicit reset")
        if reset:
            # Pinned --reset always restores default power; no preserve-power flag.
            self._approve_power_limit(100)
            self._approve_power_limit(self.power_percent)
        self._uncertain = True
        if reset:
            target_power = self.power_percent
            self.power_percent = 100
            self.identity["power_w"] = self.identity["default_power_w"]
            self._run([str(self.vendor), "--reset"], "gao-reset", log_dir)
            stock = self.inspect(log_dir)
            guard_state(stock, before | {"power_w": before["default_power_w"]}, offsets=False)
            self._same_slots(stock, before)
            if stock["core_mhz"] != 0 or stock["mem_mhz"] != 0 or any(
                    p["raw_offset"] != 0 for p in stock["curve_points_data"]):
                raise BackendError("Independent reset readback is not stock; no pair replay")
            self.set_power(target_power, log_dir)
            self._uncertain = True
        # No load can occur between reset and this verified paired readback.
        self._write_allowed()
        self._run([str(self.vendor), "--set-core", str(core)], "gao-set-core", log_dir)
        middle = self.inspect(log_dir)
        guard_state(middle, before, offsets=False)
        self._same_slots(middle, before)
        if middle["core_mhz"] != core or middle["mem_mhz"] != (0 if reset else before["mem_mhz"]):
            raise BackendError("Independent first offset readback disagrees; no subsequent write")
        if any(p["raw_offset"] != core * 1000 for p in middle["curve_points_data"]):
            raise BackendError("Core writer did not produce the independently verified uniform V/F vector")
        self._write_allowed()
        self._run([str(self.vendor), "--set-mem", str(mem)], "gao-set-mem", log_dir)
        after = self.inspect(log_dir)
        guard_state(after, before, offsets=False)
        self._same_slots(after, before)
        if after["core_mhz"] != core or after["mem_mhz"] != mem or any(
                p["raw_offset"] != core * 1000 for p in after["curve_points_data"]):
            raise BackendError("Independent paired Pstates/full-curve verification failed")
        self._uncertain = False
        self._curve_target = None
        return after

    def set_offsets(self, core: int, mem: int, expected: dict, log_dir: Path) -> dict:
        guard_state(self.inspect(log_dir), expected)
        return self.apply_uniform(core, mem, log_dir)

    def _curve_command(self, args, log_dir):
        """Retain nonzero CLI refusal text without conflating timeout with refusal."""
        self._windows()
        try:
            result = subprocess.run(args, cwd=self.root, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    timeout=25, shell=False, creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackendError(f"Curve write uncertain; no retry: {exc}") from exc
        text = result.stdout.decode("utf-8-sig", errors="replace")
        self._log_path(log_dir, "gao-curve-flatten").write_text(
            f"COMMAND: {args!r}\nEXIT: {result.returncode}\n{text}", encoding="utf-8")
        return result.returncode, text

    @staticmethod
    def _same_slots(snapshot, before):
        if [(p["index"], p["mv"]) for p in snapshot["curve_points_data"]] != [
                (p["index"], p["mv"]) for p in before["curve_points_data"]]:
            raise BackendError("Curve topology changed")

    @staticmethod
    def _curve_shape(snapshot, before, mv, mhz, base):
        NightBackend._same_slots(snapshot, before)
        points = snapshot["curve_points_data"]
        anchor = next((p for p in points if p["mv"] >= mv), None)
        if anchor is None or abs(anchor["mhz"] - mhz) > 45:
            raise BackendError("Independent curve anchor not within 45 MHz of target")
        for point in points:
            if point["mv"] < anchor["mv"] and point["raw_offset"] != base * 1000:
                raise BackendError("Lower curve raw offsets differ from base")
            if point["mv"] >= anchor["mv"] and point["mhz"] > mhz + 45:
                raise BackendError("Independent upper curve exceeds target cap")
        if len({p["raw_offset"] for p in points}) == 1:
            raise BackendError("Flat-top result is only a uniform offset, not a verified curve")
        return anchor

    @staticmethod
    def _raw_vector_write(expected, target):
        from .raw_curve import RawCurve
        with RawCurve() as bridge:
            bridge.write_exact(expected, target)

    def _write_raw_curve(self, expected, target, log_dir, bounded):
        self._ready()
        self._expected_curve(expected)
        self._expected_curve(target)
        self._write_allowed()
        before = self.inspect(log_dir)
        guard_state(before, expected)
        self._same_slots(target, before)
        self._uncertain = True
        bounded(lambda: self._raw_vector_write(before["curve_points_data"], target["curve_points_data"]))
        after = self.inspect(log_dir)
        guard_state(after, before, offsets=False)
        self._same_slots(after, before)
        if (after["curve_digest"] != target["curve_digest"] or
                (after["core_mhz"], after["mem_mhz"]) != (before["core_mhz"], before["mem_mhz"])):
            raise BackendError("Independent GAO exact raw-vector verification failed; no replay")
        self._uncertain = False
        if "curve_anchor_mv" in target:
            self._curve_target = {"digest": target["curve_digest"], "mv": target["curve_anchor_mv"],
                                  "mhz": target["curve_anchor_mhz"]}
            after.update(curve_anchor_mv=target["curve_anchor_mv"], curve_anchor_mhz=target["curve_anchor_mhz"])
        return after

    def prove_raw_restore(self, expected, log_dir, bounded):
        # A single same-vector setter call, after the untouched baseline passes.
        after = self._write_raw_curve(expected, expected, log_dir, bounded)
        guard_state(after, expected)
        self._raw_restore_digest = expected["curve_digest"]
        return after

    def restore_raw_curve(self, expected, original, log_dir, bounded):
        if self._raw_restore_digest != original["curve_digest"]:
            raise BackendError("Original raw-vector setter was not proven before destructive writes")
        return self._write_raw_curve(expected, original, log_dir, bounded)

    def apply_curve(self, mv: int, mhz: int, base_core: int, mem: int, log_dir: Path) -> dict:
        self._ready()
        if type(mv) is not int or not 900 <= mv <= 1075 or type(mhz) is not int or not 2000 <= mhz <= 4000:
            raise BackendError("Curve voltage/target outside approved limits")
        self._write_allowed()
        before = self.inspect(log_dir)
        self._bounds(base_core, mem, before)
        if mem != before["mem_mhz"]:
            raise BackendError("Curve operation cannot change memory offsets")
        anchor = next((p for p in before["curve_points_data"] if p["mv"] >= mv), None)
        if anchor is None:
            self._uncertain = True
            return {"accepted": False, "reason": "No actual V/F point at requested voltage"}
        # Let the pinned CLI enforce temperature-aware physical raise/top guards.
        self._uncertain = True
        code, text = self._curve_command([str(self.vendor), "--curve-flatten", str(mv), str(mhz),
                                          "30", str(base_core)], log_dir)
        after = self.inspect(log_dir)
        guard_state(after, before, offsets=False)
        if after["mem_mhz"] != mem or after["core_mhz"] != base_core:
            raise BackendError("Curve operation drifted Pstates base/memory offsets")
        success = re.findall(r"^flat top applied: slot (\d+) runs (\d+) MHz, in ([1-4]) write rounds -- OK\s*$",
                             text, re.MULTILINE)
        if code == 0 and len(success) == 1 and int(success[0][0]) == anchor["index"] and int(success[0][1]) == mhz:
            self._curve_shape(after, before, mv, mhz, base_core)
            self._uncertain = False
            self._curve_target = {"digest": after["curve_digest"], "mv": anchor["mv"], "mhz": mhz}
            after.update(curve_anchor_mv=anchor["mv"], curve_anchor_mhz=mhz)
            return {"accepted": True, "snapshot": after}
        refusals = re.findall(r"^not applied: ([^\r\n]+)$", text, re.MULTILINE)
        if (code == 1 and len(refusals) == 1 and not success and
                not re.search(r"FAILED|unavailable|could not|writing.*failed|device.lost|NVAPI|NVML", text, re.I) and
                ("nothing was written" in refusals[0] or "curve at stock" in refusals[0])):
            # No subsequent pressure is legal until caller explicitly restores.
            return {"accepted": False, "reason": refusals[0]}
        raise BackendError("Uncertain/unrecognized curve write result; explicit uniform reset required")

    @staticmethod
    def _expected_curve(expected):
        points = expected.get("curve_points_data")
        try:
            valid = isinstance(points, list) and bool(points) and expected.get("curve_digest") == curve_digest(points)
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError("Malformed expected full V/F vector") from exc
        if not valid:
            raise BackendError("Pressure requires independently captured full V/F vector and digest")
        if any(key.startswith("macm_") or key in ("pstates_core", "pstates_mem") for key in expected):
            raise BackendError("MACM snapshots cannot be used as standalone expectations")

    def _loaded_clocks(self, samples, expected):
        loaded = [s for s in samples if s["utilization"] > 15]
        if not loaded:
            raise BackendError("No independent loaded memory/graphics telemetry")
        mem_peak = max(s["mem_clock"] for s in loaded)
        gr_peak = max(s["gr_clock"] for s in loaded)
        if self.memory_base_mhz is None:
            if not self.derive_memory_reference:
                raise BackendError("No verified loaded baseline memory reference")
            reference = mem_peak - expected["mem_mhz"]
            if reference <= 0 or abs(reference - round(reference)) > 0.01:
                raise BackendError("Loaded baseline cannot establish an integer memory reference")
            self.memory_base_mhz = int(round(reference))
        if abs(mem_peak - (self.memory_base_mhz + expected["mem_mhz"])) > 2:
            raise BackendError("Actual loaded memory clock does not match verified driver reference + offset")
        if len({p["raw_offset"] for p in expected["curve_points_data"]}) > 1:
            cap = expected.get("curve_anchor_mhz")
            if type(cap) is not int or gr_peak > cap + 75:
                raise BackendError("Cannot prove loaded custom curve obeyed its anchor cap +75 MHz")
        return {"peak_mem_clock_mhz": mem_peak, "peak_gr_clock_mhz": gr_peak,
                "loaded_samples": len(loaded)}

    def run_test(self, seconds, duty, max_temp, out_dir, expected, stop_file):
        self._ready()
        self._expected_curve(expected)
        samples = []
        self._pressure_samples = samples
        try:
            result = super().run_test(seconds, duty, max_temp, out_dir, expected, stop_file)
        finally:
            self._pressure_samples = None
        telemetry = result.setdefault("telemetry", {})
        telemetry.update(peak_mem_clock_mhz=max((s["mem_clock"] for s in samples), default=None),
                         peak_gr_clock_mhz=max((s["gr_clock"] for s in samples), default=None))
        if result["outcome"] == "pass":
            try:
                telemetry.update(self._loaded_clocks(samples, expected))
            except BackendError as exc:
                result.update(outcome="invalid", details=str(exc))
        return result

    def bandwidth(self, log_dir: Path, expected: dict, stop_file: Path) -> dict:
        self._ready()
        self._expected_curve(expected)
        self.verify_vendor()
        self._write_allowed()
        child = None
        started = datetime.now(timezone.utc)
        samples = []
        path = self._log_path(log_dir, "gao-bandwidth")
        try:
            if self._driver_events(started, log_dir):
                raise BackendError("Driver event before bandwidth")
            if wait_for_idle(lambda: self._telemetry(log_dir), lambda: self._stopped(stop_file)) is None:
                raise BackendError("STOP before bandwidth")
            before = self.inspect(log_dir)
            guard_state(before, expected)
            if before["temp_c"] >= self.max_temp or self._stopped(stop_file):
                raise BackendError("Temperature/STOP prevents bandwidth")
            self._require_clear(False)
            args = [str(self.vendor), "--bandwidth", "--after-load", "5"]
            deadline = time.monotonic() + 60
            with path.open("wb") as output:
                output.write(f"COMMAND: {args!r}\n".encode("utf-8"))
                output.flush()
                child = subprocess.Popen(args, cwd=self.root, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT, shell=False,
                                         creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW))
                self._claim_pressure(child, self.vendor)
                next_sample = next_controls = time.monotonic()
                while child.poll() is None:
                    now = time.monotonic()
                    if self._stopped(stop_file) or now >= deadline:
                        raise BackendError("Bandwidth STOP/watchdog expired")
                    if now >= next_sample:
                        sample = self._telemetry(log_dir, timeout=max(0.05, min(3, deadline - now)))
                        samples.append(sample)
                        guard_state(sample, expected, offsets=False)
                        if sample["temp_c"] >= self.max_temp:
                            raise BackendError("Bandwidth reached temperature stop")
                        next_sample = time.monotonic() + 0.5
                    now = time.monotonic()
                    if now >= deadline or self._stopped(stop_file):
                        continue
                    if now >= next_controls and child.poll() is None:
                        controls = self._controls(log_dir, timeout=min(5, deadline - now))
                        guard_state(samples[-1] | controls, expected)
                        if self._driver_events(started, log_dir, timeout=max(0.05, min(3, deadline - time.monotonic()))):
                            raise BackendError("Driver event during bandwidth")
                        next_controls = time.monotonic() + 5
                    time.sleep(0.1)
            if child.returncode != 0:
                raise BackendError(f"Bandwidth child failed: exit {child.returncode}")
            gbps = parse_bandwidth(path.read_text(encoding="utf-8-sig", errors="replace"), self.name, self.max_temp)
        finally:
            self._cleanup(child)
            self._release_pressure(child)
            if child is not None:
                failures = []
                try:
                    after = self.inspect(log_dir)
                    guard_state(after, expected)
                    if after["temp_c"] >= self.max_temp or self._stopped(stop_file):
                        raise BackendError("Bandwidth postflight temperature/STOP")
                except (BackendError, OSError, ValueError) as exc:
                    failures.append(f"Bandwidth postflight state: {exc}")
                try:
                    if self._driver_events(started, log_dir):
                        failures.append("Driver event after bandwidth")
                except (BackendError, OSError, ValueError) as exc:
                    failures.append(f"Bandwidth postflight events: {exc}")
                if failures:
                    raise BackendError("; ".join(failures))
        clocks = self._loaded_clocks(samples, expected)
        clocks.update(peak_temp_c=max((s["temp_c"] for s in samples), default=None),
                      peak_power_w=max((s["draw_w"] for s in samples), default=None))
        return {"gbps": gbps, "snapshot": after, "telemetry": clocks, "log": str(path)}
