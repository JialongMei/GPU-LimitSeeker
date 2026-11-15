"""Owned, fail-closed runner for GpuZelenograd memtest_vulkan v0.5.0.

Source contract: v0.5.0/src/{main,close,output}.rs, not upstream main.
Integer positional argv[1] disables interactive mode AND subprocess retries.
That mode installs setup_handler(false): CTRL_BREAK exits immediately with the
signature/status bits, without the interactive final PASS sentence. Reports'
'Passed ... seconds' are per-report intervals, NOT total elapsed time.
"""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from .macm_backend import Backend, BackendError, guard_state, wait_for_idle
from .icd import discover_nvidia_icd

VENDOR_SHA256 = "09e67704210762ae8d8ad70fe6d71275eefbe815cc2c0658cd5afc8e231d0df6"
MAX_OUTPUT_BYTES = 8 * 1024 ** 2
SIGNATURE = 0x40
INITED_OK = 0x01
RUNTIME_ERRORS = 0x02
RUNTIME_ABORT = 0x04
USE_GRACEFUL_HANDLER = 0x08
QUIT_JOB_REQUESTED = 0x10
SELECTED = re.compile(
    r"^Standard 5-minute test of (\d+): Bus=0x([0-9A-Fa-f]+):([0-9A-Fa-f]+) "
    r"DevId=0x([0-9A-Fa-f]+)\s+(?:API [^\n]*?\s+)?(\d+)GB (.+)$")
PROGRESS = re.compile(
    r"^\s*(\d+) iteration\. Passed\s+(\d+\.\d+) seconds\s+"
    r"written:\s*(\d+\.\d+)GB\s*(\d+\.\d+)GB/sec\s+"
    r"checked:\s*(\d+\.\d+)GB\s*(\d+\.\d+)GB/sec\s*$")
STANDARD_PASS = "Standard 5-minute test PASSed! Just press Ctrl+C unless you plan long test run."
DEVICE_LOSS = re.compile(r"(?:ERROR_DEVICE_LOST|VK_ERROR_DEVICE_LOST|device[ _-]+lost|"
                         r"device[ _-]+removed|device[ _-]+hung)", re.IGNORECASE)


def decode_exit_status(code: int | None) -> dict | None:
    """close.rs ExitProcess returns its full u8, including 0x40 signature."""
    if type(code) is not int or not 0 <= code <= 255 or code & 0xE0 != SIGNATURE:
        return None
    return {"initialized": bool(code & INITED_OK),
            "errors": bool(code & RUNTIME_ERRORS),
            "abort": bool(code & RUNTIME_ABORT),
            "graceful_handler": bool(code & USE_GRACEFUL_HANDLER),
            "quit_requested": bool(code & QUIT_JOB_REQUESTED)}


class OutputEvidence:
    """Incremental newline parser shared by live watchdog and final verdict."""
    def __init__(self, identity: dict):
        self.identity = identity
        self.pending = ""
        self.selected = 0
        self.identity_ok = False
        self.iterations = 0
        self.reports = 0
        self.checked_gib = 0.0
        self.reported_seconds = 0.0
        self.errorcount = 0
        self.errors_found = False
        self.device_lost = False
        self.invalid = ""
        self.standard_pass = False

    def feed(self, text: str, *, final: bool = False) -> None:
        lines = (self.pending + text).split("\n")
        self.pending = "" if final else lines.pop()
        for line in lines:
            if line.endswith("\r"):
                line = line[:-1]
            if "\r" in line:
                self.invalid = "Malformed live newline sequence; archive normalization is not permitted"
                continue
            self._line(line)
        if len(self.pending) > 65536:
            self.invalid = "Unbounded or malformed output line"

    def _line(self, line: str) -> None:
        if DEVICE_LOSS.search(line):
            self.device_lost = True
        match = SELECTED.fullmatch(line)
        if line.startswith("Standard 5-minute test of "):
            self.selected += 1
            if match:
                label, bus, device, devid, gib, name = match.groups()
                self.identity_ok = (int(label) == 1 and
                                    int(bus, 16) == self.identity["pci_bus"] and
                                    int(device, 16) == self.identity["pci_device"] and
                                    int(devid, 16) == self.identity["pci_device_id"] and
                                    int(gib) == self.identity["memory_gib"] and name == self.identity["name"])
            if not match or not self.identity_ok or self.selected != 1:
                self.invalid = "Unexpected selected GPU/model/PCI or duplicate selection"
        match = PROGRESS.fullmatch(line)
        if match:
            iteration = int(match[1])
            values = [float(value) for value in match.groups()[1:]]
            if (not all(math.isfinite(value) and value >= 0 for value in values) or
                    iteration <= self.iterations or self.selected != 1):
                self.invalid = "Invalid, unordered or unselected work progress"
            else:
                self.iterations = iteration
                self.reports += 1
                self.reported_seconds += values[0]
                self.checked_gib += values[3]
        elif "iteration. Passed" in line:
            self.invalid = "Unparseable work progress"
        match = re.search(r"Error found\. Mode \w+, total errors 0x([0-9A-Fa-f]+) out of", line)
        if match:
            self.errorcount += int(match[1], 16)
            self.errors_found = True
        if "ERRORS FOUND" in line or "Error found." in line:
            self.errors_found = True
        if line == STANDARD_PASS:
            self.standard_pass = True
        if ("Runtime error:" in line or "early exit during init:" in line or
                "INIT OR FIRST testing failed" in line or "Spawned child" in line or
                "Using in-process testing method" in line):
            self.invalid = "Unexpected runtime/init failure or interactive worker mode"

    @property
    def positive_work(self) -> bool:
        return (self.selected == 1 and self.identity_ok and self.iterations > 0 and
                self.reports > 0 and self.checked_gib > 0 and self.reported_seconds > 0)


def classify_result(code, evidence: OutputEvidence, *, seconds, elapsed, timed_stop,
                    forced=None, details="", events=None, state_ok=False,
                    killed=False, telemetry_count=0) -> dict:
    status = decode_exit_status(code)
    if evidence.device_lost or events or forced == "tdr":
        outcome, reason = "tdr", "Device loss or new Windows display-driver event"
    elif killed:
        outcome, reason = "invalid", "Owned process required forced termination"
    elif forced in ("hot", "stopped"):
        outcome, reason = forced, details
    elif evidence.errors_found or (status is not None and status["errors"]):
        outcome, reason = "wrong", "Memory/GPU errors were reported"
    elif forced is not None:
        outcome, reason = forced, details
    elif events is None or not state_ok or telemetry_count < 1:
        outcome, reason = "invalid", "Independent events/state/in-test telemetry unavailable" + (
            f"; {details}" if details else "")
    elif evidence.invalid:
        outcome, reason = "invalid", evidence.invalid
    elif not evidence.positive_work:
        outcome, reason = "invalid", "No positive work on the independently identified NVIDIA adapter"
    elif elapsed < seconds:
        outcome, reason = "invalid", "Workload exited before the requested minimum wall time"
    elif status is None or not status["initialized"] or status["abort"] or status["graceful_handler"]:
        outcome, reason = "invalid", "Unknown, uninitialized, aborted or unexpected interactive exit status"
    elif code != SIGNATURE | INITED_OK | QUIT_JOB_REQUESTED:
        outcome, reason = "invalid", "No clean noninteractive quit-requested status"
    elif not timed_stop and not (evidence.standard_pass and elapsed >= 360):
        outcome, reason = "invalid", "Neither owned timed stop nor six-minute standard pass"
    else:
        outcome, reason = "pass", "Clean initialized status, verified work and requested minimum duration"
    return {"outcome": outcome, "details": reason, "elapsed": elapsed,
            "returncode": code, "status": status,
            "iterations": evidence.iterations or None,
            "errorcount": (evidence.errorcount if evidence.errorcount > 0 else
                           (0 if outcome == "pass" else None)),
            "checked_gib": evidence.checked_gib, "progress_reports": evidence.reports}


class VramRunner:
    def __init__(self, root: Path, backend, identity: dict, vendor: Path, icd=None):
        self.root = Path(root).resolve()
        self.backend = backend
        self.identity = identity
        self.uuid = identity["uuid"]
        self.vendor = Path(vendor)
        self.icd_override = icd
        self.icd = None
        self.max_bytes = int(identity["memory_total_mib"] * 1024 ** 2 * 0.75)
        if self.max_bytes <= 0:
            raise BackendError("Actual VRAM capacity is required before pressure")

    def verify_vendor(self) -> None:
        try:
            with self.vendor.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != VENDOR_SHA256:
                raise BackendError("Pinned memtest_vulkan SHA256 mismatch; execution refused")
            self.icd = discover_nvidia_icd(self.icd_override)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise BackendError(f"Cannot verify pinned VRAM executable/ICD: {exc}") from exc

    def child_environment(self) -> dict:
        # A newer loader gives VK_DRIVER_FILES priority over VK_ICD_FILENAMES.
        # Remove inherited loader overrides and the source's error-emulation knob.
        result = {key: value for key, value in os.environ.items()
                  if not key.upper().startswith(("VK_", "MEMTEST_VULKAN_"))}
        if self.icd is None:
            raise BackendError("Active ICD must be verified before launch")
        result["VK_ICD_FILENAMES"] = str(self.icd)
        result["VK_DRIVER_FILES"] = str(self.icd)
        return result

    @staticmethod
    def _require_console() -> None:
        if os.name != "nt":
            raise BackendError("VRAM runner requires a Windows native console")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetConsoleProcessList.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32)
        kernel.GetConsoleProcessList.restype = ctypes.c_uint32
        process_id = ctypes.c_uint32()
        if not kernel.GetConsoleProcessList(ctypes.byref(process_id), 1):
            raise BackendError("No inherited native console; owned CTRL_BREAK cannot be guaranteed")

    def _guard(self, actual: dict, expected: dict, *, offsets=True) -> None:
        guard_state(actual, expected, offsets=offsets)
        if actual.get("uuid") != self.uuid:
            raise BackendError("VRAM GPU UUID does not match the configured identity")
        if offsets and actual.get("power_pct") != self.backend.power_percent:
            raise BackendError("VRAM control-layer power differs from the approved envelope")
        if offsets and (not actual.get("curve_digest") or
                        actual.get("curve_digest") != expected.get("curve_digest") or
                        actual.get("nvidia_off") is not True):
            raise BackendError("VRAM curve or NVIDIA automatic tuning state is unverified/changed")

    def _clear(self) -> None:
        blockers = self.backend.blockers(search=False)
        if blockers:
            raise BackendError("Pressure/controller blockers: " + ", ".join(blockers))

    def run(self, seconds: int, out_dir: Path, expected: dict, stop_file: Path) -> dict:
        if type(seconds) is not int or not 1 <= seconds <= 3600:
            raise BackendError("VRAM seconds must be an integer in 1..3600")
        out_dir = Path(out_dir).resolve()
        stop_file = Path(stop_file)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "memtest-vulkan.stdout.log"
        native_log = out_dir / "memtest_vulkan.log"
        evidence = OutputEvidence(self.identity)
        samples = []
        events = None
        child = None
        started_wall = datetime.now(timezone.utc)
        launch_time = None
        exit_time = None
        timed_stop = killed = state_ok = False
        forced = None
        details = ""
        reader = None
        bytes_read = 0

        def consume(final=False):
            nonlocal bytes_read
            if reader is not None:
                block = reader.read(MAX_OUTPUT_BYTES + 1 - bytes_read)
                bytes_read += len(block)
                if bytes_read > MAX_OUTPUT_BYTES:
                    raise BackendError("VRAM stdout exceeded its bounded log size")
                evidence.feed(block.decode("utf-8-sig", errors="replace"), final=final)

        def remaining(cap):
            # Reserve Backend._cleanup's two bounded three-second waits inside
            # the absolute requested duration +20s pressure watchdog.
            return max(0.05, min(cap, launch_time + seconds + 14 - time.monotonic()))

        try:
            self.verify_vendor()
            self._require_console()
            self.backend._ready()
            if output_path.exists() or native_log.exists():
                raise BackendError("VRAM output directory already contains a workload log; use a fresh directory")
            if self.backend._stopped(stop_file):
                forced, details = "stopped", "STOP requested before pressure"
            else:
                self._clear()
                events = self.backend._driver_events(started_wall, out_dir, timeout=2)
                if events:
                    forced, details = "tdr", "Preflight display-driver event"
                else:
                    idle = wait_for_idle(lambda: self.backend._telemetry(out_dir, timeout=2),
                                         lambda: self.backend._stopped(stop_file))
                    if idle is None:
                        forced, details = "stopped", "STOP requested during idle admission"
                    else:
                        self._guard(idle, expected, offsets=False)
                        self._clear()
                        before = self.backend.inspect(out_dir)
                        self._guard(before, expected)
                        if before["temp_c"] >= self.backend.max_temp:
                            forced, details = "hot", "Already at temperature stop"
                        elif self.backend._stopped(stop_file):
                            forced, details = "stopped", "STOP requested before pressure"
                        else:
                            events = self.backend._driver_events(started_wall, out_dir, timeout=2)
                            if events:
                                forced, details = "tdr", "Preflight display-driver event"
                if forced is None:
                    with output_path.open("xb", buffering=0) as output:
                        reader = output_path.open("rb", buffering=0)
                        launch_time = time.monotonic()
                        # Integer argv[1] means a SINGLE process in pinned main.rs;
                        # interactive retry workers cannot be created. BREAK targets
                        # this owned group, never process names or the whole console.
                        child = subprocess.Popen(
                            [str(self.vendor), "1", str(self.max_bytes)], cwd=out_dir,
                            env=self.child_environment(), stdin=subprocess.DEVNULL,
                            stdout=output, stderr=subprocess.STDOUT, shell=False,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                        self.backend._claim_pressure(child, self.vendor)
                        next_sample = next_controls = next_events = launch_time
                        stop_sent = None
                        while child.poll() is None:
                            now = time.monotonic()
                            consume()
                            if evidence.device_lost:
                                forced, details = "tdr", "Vulkan device lost"
                            elif evidence.errors_found:
                                forced, details = "wrong", "Memory errors in live output"
                            elif evidence.invalid:
                                forced, details = "invalid", evidence.invalid
                            if self.backend._stopped(stop_file) and forced not in ("tdr", "hot"):
                                forced, details = "stopped", "STOP requested"
                            if now >= launch_time + seconds + 14:
                                forced, details = "invalid", "VRAM hard wall watchdog expired"
                                killed = True
                                Backend._cleanup(child)
                                break
                            if stop_sent is None and (forced is not None or now >= launch_time + seconds):
                                timed_stop = forced is None and now >= launch_time + seconds
                                stop_sent = now
                                child.send_signal(signal.CTRL_BREAK_EVENT)
                            if stop_sent is not None and now >= stop_sent + 10:
                                killed = True
                                if forced is None:
                                    forced, details = "invalid", "Owned CTRL_BREAK did not exit within 10 seconds"
                                Backend._cleanup(child)
                                break
                            if now >= next_sample:
                                sample = self.backend._telemetry(out_dir, timeout=remaining(1))
                                samples.append(sample)
                                self._guard(sample, expected, offsets=False)
                                if sample["temp_c"] >= self.backend.max_temp:
                                    forced, details = "hot", f"Temperature reached {sample['temp_c']} C"
                                    # Reenter immediately to stop before more probes.
                                    next_sample = time.monotonic() + 1
                                    continue
                                next_sample = time.monotonic() + 1
                            if time.monotonic() >= next_events and child.poll() is None:
                                events = self.backend._driver_events(started_wall, out_dir, timeout=remaining(2))
                                next_events = time.monotonic() + 10
                                if events:
                                    forced, details = "tdr", "New Windows display-driver event"
                                    continue
                            if time.monotonic() >= next_controls and child.poll() is None:
                                controls = self.backend._controls(out_dir, timeout=remaining(2))
                                self._guard(samples[-1] | controls, expected)
                                next_controls = time.monotonic() + 10
                            time.sleep(0.1)
                        exit_time = time.monotonic()
                        consume(final=True)
        except KeyboardInterrupt:
            forced, details = "stopped", "Interrupted by operator"
        except (BackendError, OSError, ValueError, KeyError, TypeError) as exc:
            forced, details = "invalid", str(exc)
        finally:
            if child is not None:
                try:
                    if child.poll() is None:
                        # Exception paths must not wait on more telemetry before
                        # stopping pressure. Source guarantees no worker children.
                        try:
                            child.send_signal(signal.CTRL_BREAK_EVENT)
                            child.wait(timeout=2)
                        except (OSError, ValueError, subprocess.TimeoutExpired):
                            killed = True
                            Backend._cleanup(child)
                    exit_time = exit_time or time.monotonic()
                    consume(final=True)
                except (BackendError, OSError, ValueError) as exc:
                    forced, details = "invalid", f"Owned-child cleanup/output failure: {exc}"
                self.backend._release_pressure(child)
                # Post-event and post-state verification are independent attempts.
                try:
                    events = self.backend._driver_events(started_wall, out_dir, timeout=2)
                except (BackendError, OSError, ValueError) as exc:
                    events = None
                    details += f"; post-event query failed: {exc}"
                try:
                    after = self.backend.inspect(out_dir)
                    self._guard(after, expected)
                    state_ok = True
                    if after["temp_c"] >= self.backend.max_temp and forced is None:
                        forced, details = "hot", "Post-test temperature reached limit"
                except (BackendError, OSError, ValueError, KeyError, TypeError) as exc:
                    details += f"; post-state verification failed: {exc}"
            if reader is not None:
                reader.close()
        elapsed = (exit_time - launch_time) if launch_time is not None and exit_time is not None else 0.0
        result = classify_result(child.returncode if child is not None else None, evidence,
                                 seconds=seconds, elapsed=elapsed, timed_stop=timed_stop,
                                 forced=forced, details=details, events=events,
                                 state_ok=state_ok, killed=killed, telemetry_count=len(samples))
        result["log"] = str(output_path) if child is not None else None
        result["native_log"] = str(native_log) if native_log.exists() else None
        result["telemetry"] = {
            "samples": len(samples),
            "peak_temp_c": max((sample["temp_c"] for sample in samples), default=None),
            "peak_gr_clock": max((sample["gr_clock"] for sample in samples), default=None),
            "peak_mem_clock": max((sample["mem_clock"] for sample in samples), default=None),
            "peak_power_w": max((sample["draw_w"] for sample in samples), default=None)}
        result["load_started"] = child is not None
        if result["outcome"] == "pass":
            try:
                result["telemetry"].update(self.backend._loaded_clocks(samples, expected))
            except BackendError as exc:
                result.update(outcome="invalid", details=str(exc))
        return result
