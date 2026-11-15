"""Lazy public Windows/NVIDIA backend. No native work before explicit inspection."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from ..models import Bounds, Profile, SafetyError, TrialResult, TrialSpec
from .awake import AwakeLease
from .fan_control import FanError, FanLoop, NvidiaFans
from .macm_backend import BackendError, guard_state, wait_for_idle
from .native import NativeSession
from .process_scope import OwnedProcessTree
from .vram_runner import VramRunner


class WindowsNvidiaBackend:
    """One inspected device, immutable power envelope and exclusively owned pressure.

    Underscored factories are dependency-injection seams, not live fallbacks.
    restore never writes a custom vector that the pinned CLI cannot reproduce.
    """
    def __init__(self, config: dict, run_dir: Path, *, _adapter=None,
                 _fans_factory=NvidiaFans, _scope_factory=OwnedProcessTree,
                 _awake_factory=AwakeLease, _vram_factory=VramRunner):
        self.config = deepcopy(config)
        self.run_dir = Path(run_dir)
        self._adapter = _adapter
        self._fans_factory, self._scope_factory = _fans_factory, _scope_factory
        self._awake_factory, self._vram_factory = _awake_factory, _vram_factory
        self._initial = self._expected = self._initial_profile = self._bounds = None
        self._baseline_passed = False
        self._passing_profiles = set()
        self._last_failed = None
        self._thermal_latched = False
        self._recoveries = self.config.get("_runtime_recoveries_used", 0)
        if type(self._recoveries) is not int or not 0 <= self._recoveries <= self.config["safety"]["max_recoveries"]:
            raise SafetyError("Invalid cumulative original recovery count")
        self._started = None
        self._scope = self._fans = self._fan_loop = None
        self._fan_name = None
        self._cleanup_unknown = self._closed = False
        self._stop_event = threading.Event()
        self._trial_sequence = 0
        safety = self.config["safety"]
        maximum = safety.get("max_power_percent", 100)
        powers = self.config.get("search", {}).get("power_limits", [safety["power_percent"]])
        if type(maximum) is not int or maximum < 1 or any(type(p) is not int or not 1 <= p <= maximum for p in powers):
            raise SafetyError("Requested power point exceeds explicit approved maximum")
        if safety["power_percent"] not in powers:
            raise SafetyError("Approved power sweep must include the untouched original baseline")
        cap = safety.get("max_board_power_w")
        if any(p > 100 for p in powers) and (cap is None or type(cap) not in (int, float) or not 0 < cap < float("inf")):
            raise SafetyError("Above-default power requires an explicit finite board-watt cap")
        if not 0 < safety["max_temp_c"] <= 80 or not 0 < safety["max_hours"] <= 72:
            raise SafetyError("Unsupported temperature/runtime envelope")
        self._runtime_seconds = self.config.get("_runtime_remaining_seconds", safety["max_hours"] * 3600)
        if type(self._runtime_seconds) not in (int, float) or not 0 < self._runtime_seconds <= safety["max_hours"] * 3600:
            raise SafetyError("Remaining original runtime must be finite, positive and within max_hours")
        memory = self.config["hardware"].get("memory_reference_mhz")
        if memory is not None and (type(memory) is not int or not 0 < memory <= 100000):
            raise SafetyError("Memory reference must be a positive integer and is verified against loaded baseline")

    def _open(self):
        if self._closed or self._cleanup_unknown:
            raise SafetyError("Backend is closed or native cleanup is unconfirmed")
        if self._adapter is None:
            self._adapter = NativeSession.open(self.config, self.run_dir)
        return self._adapter

    def inspect(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        adapter = self._open()
        state = adapter.inspect(self.run_dir / "inspection")
        percent = adapter.power_percent
        if state["power_pct"] != percent or abs(state["power_w"] - state["default_power_w"] * percent / 100) > 0.5:
            raise SafetyError("Actual power is not the approved percentage of this device's default TDP")
        points = state["curve_points_data"]
        if state["core_mhz"] < 0 or state["mem_mhz"] < 0:
            raise SafetyError("Negative initial offsets are unsupported by the public nonnegative search model")
        if not points or state.get("nvidia_off") is not True:
            raise SafetyError("Full physical curve and disabled NVIDIA automatic tuning are required")
        if self._initial is None:
            supplied = self.config["hardware"].get("baseline_profile")
            uniform = all(p["raw_offset"] == state["core_mhz"] * 1000 for p in points)
            if uniform:
                profile = Profile("uniform", state["core_mhz"], state["mem_mhz"], power_percent=percent)
                if supplied:
                    configured = Profile.from_dict(supplied)
                    if configured.kind != "uniform" or (configured.core, configured.mem) != (profile.core, profile.mem):
                        raise SafetyError("Configured baseline does not match actual current offsets")
                    profile = configured
                    if profile.power_percent != percent:
                        raise SafetyError("Configured baseline power disagrees with inspected power")
            else:
                if not supplied or adapter.mode != "standalone":
                    raise SafetyError("Custom initial curve requires a matching standalone baseline_profile")
                profile = Profile.from_dict(supplied)
                if profile.kind != "curve" or (profile.core, profile.mem) != (state["core_mhz"], state["mem_mhz"]):
                    raise SafetyError("Configured custom baseline disagrees with measured offsets")
                adapter._curve_shape(state, state, profile.mv, profile.mhz, profile.core)
                if any(p["mv"] >= profile.mv and abs(p["mhz"] - profile.mhz) > 45 for p in points):
                    raise SafetyError("Custom baseline upper vector is not a supported flat top")
                if profile.power_percent != percent:
                    raise SafetyError("Custom baseline power disagrees with inspected power")
                state.update(curve_anchor_mv=profile.mv, curve_anchor_mhz=profile.mhz)
            low, high = max(900, min(p["mv"] for p in points)), min(1050, max(p["mv"] for p in points))
            self._bounds = Bounds(state["core_max"], state["mem_max"], low, high,
                                  power_min_percent=state["power_min_percent"],
                                  power_max_percent=state["power_max_percent"])
            self._bounds.check(profile)
            self._initial = deepcopy(state)
            self._expected = deepcopy(state)
            self._initial_profile = profile
            self._approve_power(profile)
        else:
            guard_state(state, self._expected)
        return state | {"name": adapter.name, "initial": self._initial_profile.to_dict(),
                        "bounds": self._bounds.to_dict(), "capabilities": {
                            "vendor": "nvidia", "platform": "windows", "full_curve_readback": True,
                            "mode": adapter.mode, "baseline_preserved": True,
                            "afterburner_profile_export": False,
                            "memory_reference_mhz": adapter.memory_base_mhz}}

    def _ownership(self):
        try:
            return self._adapter is not None and self._adapter.owns_controls() is True
        except Exception:
            return False

    def _admit(self, stop_file):
        adapter = self._open()
        if self._thermal_latched:
            raise SafetyError("Thermal stop is latched for this session")
        if adapter._stopped(stop_file):
            raise SafetyError("STOP requested")
        idle = wait_for_idle(lambda: adapter._telemetry(self.run_dir), lambda: adapter._stopped(stop_file))
        if idle is None:
            raise SafetyError("STOP during idle admission")
        if idle["temp_c"] >= self.config["safety"]["max_temp_c"]:
            self._thermal_latched = True
            raise SafetyError("Temperature stop before write")
        if not self._ownership():
            raise SafetyError("Clock-controller ownership changed")
        guard_state(adapter.inspect(self.run_dir), self._expected)
        self._check(None)

    def apply(self, profile: Profile) -> dict:
        if self._initial is None:
            raise SafetyError("Explicit inspect is required before applying a profile")
        adapter = self._open()
        self._check(None)
        self._bounds.check(profile)
        self._approve_power(profile)
        if not self._baseline_passed:
            raise SafetyError("The exact current baseline must pass before any clock write")
        # Even a uniform write must retain a permitted exact-original reset path.
        # Prove it before MACM clearing/closing, power changes, or offset writes.
        self._approve_reset_power()
        self._admit(self.run_dir / "STOP")
        if self._initial_profile.kind == "curve" and getattr(adapter, "_raw_restore_digest", None) is None:
            adapter.prove_raw_restore(self._initial, self.run_dir / "restore-proof", self._bounded_native)
            self._check(None)
        if adapter.mode == "macm":
            if not self.config["hardware"].get("allow_afterburner_handoff", False):
                raise SafetyError("Afterburner-to-GAO handoff requires explicit permission after baseline")
            self._expected = adapter.handoff(self._expected, self.run_dir / "handoff",
                                             lambda: adapter._stopped(self.run_dir / "STOP"))
        self._bounds.check(profile)
        self._expected = adapter.set_power(profile.power_percent, self.run_dir / "apply-power")
        if profile.kind == "uniform":
            state = adapter.apply_uniform(profile.core, profile.mem, self.run_dir / "apply",
                                          reset=any(p["raw_offset"] != self._expected["core_mhz"] * 1000
                                                    for p in self._expected["curve_points_data"]))
        else:
            if not any(p["mv"] == profile.mv for p in self._initial["curve_points_data"]):
                raise SafetyError("Requested anchor is not an actual driver voltage slot")
            self._expected = adapter.apply_uniform(profile.core, profile.mem, self.run_dir / "apply", reset=True)
            result = adapter.apply_curve(profile.mv, profile.mhz, profile.core, profile.mem, self.run_dir / "apply")
            if not result.get("accepted"):
                raise SafetyError("Driver refused the requested curve: " + result.get("reason", "unknown"))
            state = result["snapshot"]
        self._expected = deepcopy(state)
        return state

    def _approve_power(self, profile):
        safety = self.config["safety"]
        powers = self.config.get("search", {}).get("power_limits", [safety["power_percent"]])
        if profile.power_percent not in powers or profile.power_percent > safety.get("max_power_percent", 100):
            raise SafetyError("Profile power is outside the immutable approved sweep")
        cap = safety.get("max_board_power_w")
        watts = self._initial["default_power_w"] * profile.power_percent / 100
        if profile.power_percent > 100 and cap is None:
            raise SafetyError("Above-default power has no approved watt cap")
        if cap is not None and watts > cap:
            raise SafetyError("Requested power cap exceeds the approved board-watt envelope")

    def _approve_reset_power(self):
        safety = self.config["safety"]
        powers = self.config.get("search", {}).get("power_limits", [safety["power_percent"]])
        if 100 > min(safety.get("max_power_percent", 100), max(powers)):
            raise SafetyError("Pinned reset requires 100% power outside the approved percentage envelope")
        cap = safety.get("max_board_power_w")
        if cap is not None and self._initial["default_power_w"] > cap:
            raise SafetyError("Pinned reset requires default board watts outside the approved envelope")

    def _bounded_native(self, function):
        done = threading.Event()
        values, errors = [], []
        def invoke():
            try:
                values.append(function())
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()
        threading.Thread(target=invoke, daemon=True, name="bounded-native").start()
        if not done.wait(5):
            self._cleanup_unknown = True
            raise SafetyError("Native fan call exceeded five seconds; no competing calls permitted")
        if errors:
            raise errors[0]
        return values[0]

    def _check(self, external):
        try:
            if (getattr(self._adapter, "thermal_latched", False) or
                    (self._fan_loop is not None and self._fan_loop.thermal_event.is_set())):
                self._thermal_latched = True
            if self._thermal_latched:
                raise BackendError("Thermal stop is latched for this session")
            if external is not None:
                external()
            if self._fan_loop is not None:
                self._fan_loop.check()
            if self._started is not None and time.monotonic() - self._started >= self._runtime_seconds:
                raise BackendError("Cumulative backend runtime exhausted")
        except Exception as exc:
            raise BackendError(str(exc)) from exc

    def _end_fans(self, relinquish=False):
        loop = self._fan_loop
        if loop is None:
            if self._fans is not None and not self._cleanup_unknown:
                self._bounded_native(self._fans.close)
                self._fans = None
            return
        if relinquish or not self._ownership():
            loop.relinquish()
        try:
            loop.__exit__(None, None, None)
        finally:
            if loop.thermal_event.is_set():
                self._thermal_latched = True
            if loop._thread is not None and loop._thread.is_alive():
                self._cleanup_unknown = True
            else:
                self._fan_loop = None
                self._bounded_native(self._fans.close)
                self._fans = None
                self._fan_name = None

    def _start_fans(self, profile, stop_file):
        if self._adapter.mode != "standalone":
            return  # Existing Afterburner owns its fan controller during untouched baseline.
        if self._fan_name == profile.fan and self._fan_loop is not None:
            self._fan_loop.check()
            return
        self._end_fans()
        points = self.config["fan_profiles"].get(profile.fan)
        if points is None:
            raise SafetyError("Unknown fan profile")
        if not self._ownership():
            raise SafetyError("Cannot take fan ownership from an external controller")
        self._fans = self._bounded_native(lambda: self._fans_factory(self._adapter.uuid))
        self._fan_loop = FanLoop(self._fans, points, self._stop_event, stop_file,
                                 self.run_dir / "fans.jsonl", self.config["safety"]["max_temp_c"],
                                 ownership=self._ownership)
        self._fan_name = profile.fan
        try:
            self._fan_loop.__enter__()
        except BaseException:
            if self._fan_loop._thread is not None and self._fan_loop._thread.is_alive():
                self._cleanup_unknown = True
            raise

    def run_trial(self, spec: TrialSpec, stop_file: Path, check=None) -> TrialResult:
        if self._initial is None:
            raise SafetyError("Explicit inspect is required before pressure")
        adapter = self._open()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._trial_sequence += 1
        directory = self.run_dir / f"trial-{self._trial_sequence:06d}"
        directory.mkdir()
        workloads, results = [], []
        bandwidth = None
        adapter._check = lambda: self._check(check)
        try:
            self._check(check)
            if self._thermal_latched or self._stop_event.is_set() or adapter._stopped(stop_file):
                return TrialResult(spec.id, "hot" if self._thermal_latched else "stopped", load_started=False,
                                   details="Latched thermal/STOP interlock")
            if self._scope is None:
                import math
                self._scope = self._scope_factory(math.ceil(self._runtime_seconds) + 300)
                self._started = time.monotonic()
            if not self._baseline_passed:
                if spec.profile != self._initial_profile:
                    raise SafetyError("First trial must preserve the exact inspected baseline")
                self._admit(stop_file)
            else:
                self.apply(spec.profile)
            self._start_fans(spec.profile, stop_file)
            adapter.derive_memory_reference = not self._baseline_passed
            with self._awake_factory():
                for kind, seconds, duty in (("full", spec.full_seconds, 100), ("half", spec.half_seconds, 50),
                                             ("vram", spec.vram_seconds, None)):
                    if not seconds:
                        continue
                    self._check(check)
                    if kind == "vram":
                        runner = self._vram_factory(self.run_dir, adapter,
                            self._initial | {"name": adapter.name}, Path(self.config["hardware"]["memtest_path"]),
                            self.config["hardware"].get("nvidia_icd"))
                        result = runner.run(seconds, directory / kind, self._expected, stop_file)
                    else:
                        result = adapter.run_test(seconds, duty, self.config["safety"]["max_temp_c"],
                                                  directory / kind, self._expected, stop_file)
                    results.append(result)
                    telemetry = result.get("telemetry", {})
                    peak = telemetry.get("peak_temp_c")
                    if result["outcome"] == "hot" or (peak is not None and peak >= self.config["safety"]["max_temp_c"]):
                        self._thermal_latched = True
                    passed = result["outcome"] == "pass"
                    workloads.append({"kind": kind, "seconds": seconds if passed else 0, "completed": passed,
                        "achieved_core_mhz": telemetry.get("peak_gr_clock_mhz", telemetry.get("peak_gr_clock")),
                        "achieved_mem_mhz": telemetry.get("peak_mem_clock_mhz", telemetry.get("peak_mem_clock"))})
                    if not passed:
                        self._last_failed = spec.profile
                        break
                self._check(check)
                if results and all(r["outcome"] == "pass" for r in results) and spec.bandwidth:
                    measured = adapter.bandwidth(directory / "bandwidth", self._expected, stop_file)
                    bandwidth = measured["gbps"]
                    results.append({"outcome": "pass", "load_started": True, "details": "Measured bandwidth",
                                    "telemetry": measured["telemetry"]})
            self._check(check)
            outcome = results[-1]["outcome"] if results else "invalid"
            if self._thermal_latched and outcome != "tdr":
                outcome = "hot"
            peaks = [r.get("telemetry", {}).get("peak_temp_c") for r in results]
            result = TrialResult(spec.id, outcome, score=results[0].get("score") if results else None,
                peak_temp_c=max((p for p in peaks if p is not None), default=None), bandwidth_gbps=bandwidth,
                load_started=any(r.get("load_started", bool(r.get("log"))) for r in results),
                details="; ".join(r.get("details", "") for r in results), workloads=tuple(workloads),
                full_avg_power_w=results[0].get("telemetry", {}).get("avg_power_w") if results else None,
                peak_power_w=max((r.get("telemetry", {}).get("peak_power_w") for r in results
                                  if r.get("telemetry", {}).get("peak_power_w") is not None), default=None))
            if outcome == "pass":
                self._baseline_passed = True
                self._passing_profiles.add(spec.profile.key)
            return result
        except (SafetyError, BackendError, FanError, OSError, ValueError) as exc:
            if getattr(adapter, "thermal_latched", False):
                self._thermal_latched = True
            self._last_failed = spec.profile
            return TrialResult(spec.id, "hot" if self._thermal_latched else "invalid",
                load_started=any(r.get("load_started", bool(r.get("log"))) for r in results),
                details=str(exc), workloads=tuple(workloads))
        finally:
            adapter.derive_memory_reference = False
            adapter._check = None

    def recover_tdr(self, fallback: Profile, stop_file: Path) -> dict:
        adapter = self._open()
        if self._thermal_latched or self._stop_event.is_set():
            raise SafetyError("Thermal/fan stop remains latched; recovery is forbidden")
        if self._recoveries >= self.config["safety"]["max_recoveries"]:
            raise SafetyError("Cumulative recovery limit reached")
        if fallback == self._last_failed or self._last_failed is None:
            raise SafetyError("Recovery requires an independently safe fallback, never the failed current profile")
        if fallback.key not in self._passing_profiles:
            raise SafetyError("Recovery fallback has never independently passed in this session")
        self._bounds.check(fallback)
        self._recoveries += 1
        self._end_fans()
        if not self._ownership() or adapter._uncertain:
            raise SafetyError("Ownership or native write state is uncertain; no recovery writes")
        deadline = time.monotonic() + 120
        consecutive = 0
        while time.monotonic() < deadline:
            self._check(None)
            if adapter._stopped(stop_file):
                raise SafetyError("STOP during TDR cooldown")
            sample = adapter._telemetry(self.run_dir / "recovery", timeout=3)
            if sample["temp_c"] >= self.config["safety"]["max_temp_c"]:
                self._thermal_latched = True
                raise SafetyError("Thermal stop during TDR cooldown")
            consecutive = consecutive + 1 if sample["temp_c"] <= self.config["safety"]["max_temp_c"] - 10 else 0
            if consecutive >= 3:
                break
            time.sleep(2)
        else:
            raise SafetyError("TDR cooldown deadline expired")
        since = datetime.now(timezone.utc)
        if adapter._driver_events(since, self.run_dir / "recovery"):
            raise SafetyError("Driver is still reporting failures")
        current = adapter.inspect(self.run_dir / "recovery")
        guard_state(current, self._expected, offsets=False)
        self._expected = current
        state = self.apply(fallback)
        if adapter._driver_events(since, self.run_dir / "recovery"):
            raise SafetyError("Driver health failed after fallback")
        return {"recovered": True, "recoveries": self._recoveries, "snapshot": state,
                "thermal_latched": self._thermal_latched}

    def restore(self) -> dict:
        if self._initial is None:
            return {"restored": False, "reason": "No inspected baseline; nothing owned"}
        adapter = self._open()
        if not self._ownership():
            self._end_fans(relinquish=True)
            return {"restored": False, "reason": "External controller detected; no clocks/fans overwritten"}
        self._end_fans()
        if adapter._uncertain:
            raise SafetyError("Unknown native write completion; restoration is unconfirmed, not replayed")
        current = adapter.inspect(self.run_dir / "restore")
        guard_state(current, self._expected)
        if not self._baseline_passed or adapter.mode == "macm":
            guard_state(current, self._initial)
            return {"restored": True, "snapshot": current, "reason": "Original controller/settings untouched"}
        if self._initial_profile.kind == "curve" and getattr(adapter, "_raw_restore_digest", None) is None:
            guard_state(current, self._initial)
            return {"restored": True, "snapshot": current, "reason": "Original custom vector untouched"}
        profile = self._initial_profile
        self._approve_power(profile)
        self._approve_reset_power()
        # Restoration bypasses STOP/thermal pressure admission, but never ownership checks.
        # The pinned writer itself refuses STOP rather than circumventing operator intent.
        self._expected = adapter.set_power(profile.power_percent, self.run_dir / "restore-power")
        current = adapter.apply_uniform(profile.core, profile.mem, self.run_dir / "restore", reset=True)
        if profile.kind == "curve":
            current = adapter.restore_raw_curve(current, self._initial, self.run_dir / "restore",
                                                self._bounded_native)
        if [p["raw_offset"] for p in current["curve_points_data"]] != [p["raw_offset"] for p in self._initial["curve_points_data"]]:
            self._expected = current
            raise SafetyError("Restored curve differs from exact original raw vector")
        self._expected = current
        return {"restored": True, "snapshot": current, "fans": "driver automatic verified by fan cleanup"}

    def close(self, relinquish: bool = False) -> None:
        if self._closed:
            return
        try:
            self._end_fans(relinquish=relinquish)
        finally:
            self._closed = True
            if self._scope is not None and not self._cleanup_unknown:
                self._scope.close()
        if self._cleanup_unknown:
            raise SafetyError("Native fan cleanup is unconfirmed; no competing native calls issued")
