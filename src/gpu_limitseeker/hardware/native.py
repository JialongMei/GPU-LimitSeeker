"""Explicit Windows session: MACM baseline, permission-gated nonstacked cutover."""
from __future__ import annotations
import csv
import io
import json
import math
import re
import time
from pathlib import Path
from .macm import MACMError, guard_snapshot
from .macm_backend import Backend, BackendError, guard_state, parse_status, process_blockers
from .night_backend import NightBackend, parse_full_curve, curve_digest, standalone_blockers


def parse_identity(text: str) -> dict:
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise BackendError("Exactly one NVIDIA GPU with complete PCI identity is required")
    name, bus, device, memory = (item.strip() for item in rows[0])
    match = re.fullmatch(r"[0-9A-Fa-f]{4,8}:([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.0", bus)
    if not name.startswith("NVIDIA ") or match is None or not re.fullmatch(r"0x[0-9A-Fa-f]{8}", device):
        raise BackendError("Unsupported NVIDIA name/PCI identity")
    try:
        mib = float(memory)
        if not math.isfinite(mib) or not 0 < mib <= 1024 * 1024:
            raise ValueError("invalid memory size")
        gib = math.ceil(mib / 1024)
    except ValueError as exc:
        raise BackendError("Unreadable NVIDIA memory capacity") from exc
    if not 1 <= gib <= 1024:
        raise BackendError("Unsupported memory capacity")
    return {"name": name, "pci_bus": int(match[1], 16), "pci_device": int(match[2], 16),
            "pci_device_id": int(device[2:6], 16), "memory_gib": gib, "memory_total_mib": mib}


class NativeSession(NightBackend):
    @classmethod
    def open(cls, config: dict, root: Path):
        hardware, safety = config["hardware"], config["safety"]
        probe = Backend(root, Path(hardware["gao_path"]))
        probe.verify_vendor()
        identity = probe._telemetry()
        identity.update(parse_identity(probe._run([probe._smi_path(),
            "--query-gpu=name,pci.bus_id,pci.device_id,memory.total", "--format=csv,noheader,nounits"], "identity")))
        percent = safety["power_percent"]
        approved_w = identity["default_power_w"] * percent / 100
        if abs(identity["power_w"] - approved_w) > 0.5:
            raise BackendError("Set the approved fixed power percentage manually before inspection; baseline is never reset")
        session = cls(root, Path(hardware["gao_path"]), identity,
                      hardware.get("memory_reference_mhz"), percent, safety["max_temp_c"])
        session.device_identity = identity
        limits = probe._run([probe._smi_path(),
            "--query-gpu=power.min_limit,power.max_limit", "--format=csv,noheader,nounits"], "power-bounds")
        rows = list(csv.reader(io.StringIO(limits.strip())))
        if len(rows) != 1 or len(rows[0]) != 2:
            raise BackendError("Missing or ambiguous driver power limits")
        import math
        try:
            minimum, maximum = map(float, rows[0])
        except ValueError as exc:
            raise BackendError("Unsupported/nonadjustable driver power limits") from exc
        default = identity["default_power_w"]
        if not all(math.isfinite(v) and v > 0 for v in (minimum, maximum)) or minimum > default or maximum < default:
            raise BackendError("Invalid actual driver power range")
        session.power_min_percent = math.ceil(minimum * 100 / default)
        session.power_max_percent = math.floor(maximum * 100 / default)
        session.max_board_power_w = safety.get("max_board_power_w")
        session.max_power_percent = min(safety.get("max_power_percent", 100),
            max(config.get("search", {}).get("power_limits", [percent])))
        session.mode = "standalone"
        inventory = session._inventory()
        session.approved_handoff_processes = {
            (p["ProcessName"].lower(), p["Id"], p.get("StartTicks")) for p in inventory
            if p["ProcessName"].lower() in ("msiafterburner", "rtss")}
        if any(not isinstance(token[2], str) or not token[2].isdigit() for token in session.approved_handoff_processes):
            raise BackendError("Controller process creation identity is unreadable")
        owners = [p for p in inventory if p["ProcessName"].lower() == "msiafterburner"]
        if len(owners) > 1:
            raise BackendError("Multiple Afterburner controllers are unsupported")
        if owners:
            blocked = process_blockers(inventory, search=True)
            if blocked:
                raise BackendError("Competing controllers/workloads: " + "; ".join(blocked))
            session.mode = "macm"
        return session

    def inspect(self, log_dir=None):
        return super().inspect(log_dir) | {
            key: self.device_identity[key] for key in
            ("name", "pci_bus", "pci_device", "pci_device_id", "memory_gib", "memory_total_mib")} | {
                "power_min_percent": self.power_min_percent, "power_max_percent": self.power_max_percent}

    def set_power(self, percent, log_dir):
        self._approve_power_limit(percent)
        self._write_allowed()
        if type(percent) is not int or not self.power_min_percent <= percent <= self.power_max_percent:
            raise BackendError("Power percentage exceeds actual driver bounds")
        before = self.inspect(log_dir)
        if percent == self.power_percent:
            return before
        self._uncertain = True
        watts = self.identity["default_power_w"] * percent / 100
        self._run([self._smi_path(), "--id=" + self.uuid, "--power-limit=" + format(watts, ".6g")],
                  "set-power", log_dir)
        self.power_percent = percent
        self.identity["power_w"] = watts
        after = self.inspect(log_dir)
        guard_state(after, before | {"power_w": watts})
        if after["curve_digest"] != before["curve_digest"]:
            raise BackendError("Power write changed the raw V/F vector")
        self._uncertain = False
        return after

    def blockers(self, search=False):
        inventory = self._pressure_inventory(self._inventory())
        return (process_blockers(inventory, search=self.mode == "macm") if self.mode == "macm"
                else standalone_blockers(inventory))

    def owns_controls(self):
        inventory = self._pressure_inventory(self._inventory(timeout=2))
        if self.mode == "standalone":
            return not standalone_blockers(inventory, controllers_only=True)
        owners = {(p["ProcessName"].lower(), p["Id"], p.get("StartTicks")) for p in inventory
                  if p["ProcessName"].lower() == "msiafterburner"}
        approved = {p for p in self.approved_handoff_processes if p[0] == "msiafterburner"}
        return owners == approved and not process_blockers(inventory, search=True)

    def _controls(self, log_dir=None, timeout=8):
        if self.mode == "standalone":
            return super()._controls(log_dir, timeout)
        deadline = time.monotonic() + timeout
        def left():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BackendError("MACM independent readback deadline expired")
            return remaining
        inventory = self._pressure_inventory(self._inventory(left()))
        owners = {(p["ProcessName"].lower(), p["Id"], p.get("StartTicks")) for p in inventory
                  if p["ProcessName"].lower() == "msiafterburner"}
        approved = {p for p in self.approved_handoff_processes if p[0] == "msiafterburner"}
        if owners != approved or process_blockers(inventory, search=True):
            raise BackendError("Baseline controller identity changed or a competing writer appeared")
        try:
            mapped = self.macm.snapshot(wait_ms=max(1, int(min(2, left()) * 1000)))
            status = parse_status(self._run([str(self.vendor), "--status"], "status", log_dir, left()))
            points = parse_full_curve(self._run([str(self.vendor), "--curve"], "curve", log_dir, left()))
            after = self.macm.snapshot(wait_ms=max(1, int(min(2, left()) * 1000)))
            guard_snapshot(after, mapped)
        except MACMError as exc:
            raise BackendError(str(exc)) from exc
        if [p["raw_offset"] for p in points] != after["curve_offsets_khz"]:
            raise BackendError("MACM cached vector disagrees with full physical V/F readback")
        if status["core_mhz"] or status["mem_mhz"]:
            raise BackendError("Pstates and MACM offsets would stack")
        if status["nvidia_off"] is not True or after["power_pct"] != self.power_percent or status["power_pct"] != self.power_percent:
            raise BackendError("Unverified automatic tuning or fixed power envelope")
        return after | {"nvidia_off": True, "pstates_core": 0, "pstates_mem": 0,
                        "curve_digest": curve_digest(points), "curve_points_data": points}

    @staticmethod
    def _expected_curve(expected):
        if expected.get("curve_digest") != curve_digest(expected.get("curve_points_data", [])):
            raise BackendError("Full physical curve expectation is missing or malformed")

    def handoff(self, expected, log_dir, stopped):
        if self.mode != "macm":
            return self.inspect(log_dir)
        self._approve_power_limit(100)
        self._approve_power_limit(self.power_percent)
        self._write_allowed()
        guard_state(self.inspect(log_dir), expected)
        if expected.get("macm_curve_enabled") or expected.get("macm_curve_lock"):
            raise BackendError("Custom/locked MACM cutover is unsupported; no offsets cleared")
        try:
            mapped = self.macm.snapshot()
            guard_snapshot(mapped, expected | {"curve_digest": mapped["curve_digest"]})
            if mapped["curve_offsets_khz"] != [p["raw_offset"] for p in expected["curve_points_data"]]:
                raise MACMError("MACM full vector drifted before handoff")
            self.macm.set_offsets(0, 0, mapped, stopped)
        except MACMError as exc:
            self._uncertain = True
            raise BackendError("MACM handoff failed; no competing GAO write") from exc
        cleared = self.inspect(log_dir)
        if cleared["core_mhz"] or cleared["mem_mhz"] or any(p["raw_offset"] for p in cleared["curve_points_data"]):
            raise BackendError("MACM did not independently clear its complete offset layer")
        # Capture exact approved process identities, not game names or broad closures.
        script = "$ErrorActionPreference='Stop'; ConvertTo-Json -InputObject @((Get-Process MSIAfterburner,RTSS -ErrorAction SilentlyContinue) | ForEach-Object { @{Id=$_.Id;Name=$_.ProcessName;Ticks=$_.StartTime.ToUniversalTime().Ticks.ToString()} }) -Compress"
        selected = json.loads(self._run([self._powershell(), "-NoProfile", "-NonInteractive", "-Command", script], "handoff-owners"))
        if not isinstance(selected, list) or not any(p.get("Name") == "MSIAfterburner" for p in selected):
            raise BackendError("Afterburner ownership changed during handoff")
        for process in selected:
            if process.get("Name", "").lower() not in ("msiafterburner", "rtss") or type(process.get("Id")) is not int or not str(process.get("Ticks", "")).isdigit():
                raise BackendError("Invalid handoff process identity")
            if (process["Name"].lower(), process["Id"], process["Ticks"]) not in self.approved_handoff_processes:
                raise BackendError("A new user's controller appeared; it is not approved for closing")
            if stopped():
                raise BackendError("STOP during handoff")
            close = (f"$ErrorActionPreference='Stop'; $p=Get-Process -Id {process['Id']} -ErrorAction Stop; $null=$p.Handle; "
                     f"if ($p.ProcessName -ne '{process['Name']}' -or $p.StartTime.ToUniversalTime().Ticks.ToString() -ne '{process['Ticks']}') {{throw 'Controller replaced'}}; "
                     "if ($p.CloseMainWindow() -and $p.WaitForExit(3000)) {exit 0}; "
                     "$p.Kill(); if (-not $p.WaitForExit(3000)) {throw 'Approved controller close timed out'}")
            self._run([self._powershell(), "-NoProfile", "-NonInteractive", "-Command", close], "handoff-close", log_dir, 8)
        self.mode = "standalone"
        self._no_controller()
        return self.apply_uniform(expected["core_mhz"], expected["mem_mhz"], log_dir)
