"""Synthetic offline facade tests: no vendor executable, GPU, registry or NVML calls."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from gpu_limitseeker.hardware import WindowsNvidiaBackend
from gpu_limitseeker.hardware.macm_backend import BackendError, parse_smi, wait_for_idle
from gpu_limitseeker.hardware.native import NativeSession, parse_identity
from gpu_limitseeker.hardware.night_backend import NightBackend, curve_digest, parse_full_curve
from gpu_limitseeker.models import Profile, SafetyError, TrialSpec


UUID = "GPU-00000000-0000-0000-0000-000000000001"


def config():
    return {"hardware": {"gao_path": "synthetic-gao.exe", "memtest_path": "synthetic-memtest.exe",
                          "allow_afterburner_handoff": False, "memory_reference_mhz": None},
            "safety": {"power_percent": 100, "max_power_percent": 100, "max_board_power_w": None,
                       "max_temp_c": 80, "max_hours": 1, "max_recoveries": 2},
            "search": {"power_limits": [100]},
            "fan_profiles": {"performance": [[30, 30], [78, 100]]}}


class FakeFans:
    minimum, maximum, count = 25, 100, 1
    def __init__(self, uuid):
        self.uuid, self.target, self.policy, self.closed = uuid, 25, 0, False
    def read(self):
        return {"targets": [self.target], "policies": [self.policy], "speeds": [self.target], "temp_c": 40}
    def set_percent(self, value):
        self.target, self.policy = value, 1
    def restore_auto(self):
        self.policy = 0
    def close(self):
        self.closed = True


class FakeSession:
    _curve_shape = staticmethod(NightBackend._curve_shape)
    def __init__(self):
        self.uuid, self.driver, self.name = UUID, "600.01", "NVIDIA Synthetic Adapter"
        self.mode, self.memory_base_mhz = "standalone", None
        self.power_percent, self.max_temp = 100, 80
        self._uncertain, self._check, self.derive_memory_reference = False, None, False
        self.thermal_latched, self.owned, self.calls = False, True, []
        self.state = {"uuid": UUID, "driver": self.driver, "name": self.name,
            "power_w": 240.0, "default_power_w": 240.0, "power_pct": 100,
            "power_min_percent": 50, "power_max_percent": 130,
            "temp_c": 40, "utilization": 0, "draw_w": 15, "pstate": "P8", "mem_clock": 400,
            "gr_clock": 300, "core_mhz": 60, "mem_mhz": 200, "core_min": -500,
            "core_max": 800, "mem_min": -1000, "mem_max": 3000, "nvidia_off": True,
            "pci_bus": 3, "pci_device": 0, "pci_device_id": 0x1234, "memory_gib": 12}
        self._curve(60)
    def _curve(self, core):
        self.state["curve_points_data"] = [
            {"index": 0, "mv": 900, "mhz": 2000 + core, "raw_offset": core * 1000},
            {"index": 1, "mv": 1000, "mhz": 2300 + core, "raw_offset": core * 1000},
            {"index": 2, "mv": 1050, "mhz": 2400 + core, "raw_offset": core * 1000}]
        self.state["curve_digest"] = curve_digest(self.state["curve_points_data"])
    def inspect(self, directory=None):
        self.calls.append("inspect")
        return deepcopy(self.state)
    def _telemetry(self, directory=None, timeout=None):
        return deepcopy(self.state)
    def _stopped(self, path):
        if self._check:
            self._check()
        return Path(path).exists()
    def owns_controls(self):
        return self.owned
    def set_power(self, percent, directory):
        self.calls.append(("power", percent))
        self.power_percent = percent
        self.state.update(power_pct=percent, power_w=240 * percent / 100)
        return self.inspect()
    def apply_uniform(self, core, mem, directory, reset=False):
        self.calls.append(("uniform", core, mem, reset))
        self.state.update(core_mhz=core, mem_mhz=mem)
        self._curve(core)
        return self.inspect()
    def handoff(self, expected, directory, stopped):
        self.calls.append("handoff")
        self.mode = "standalone"
        return self.inspect()
    def run_test(self, seconds, duty, max_temp, directory, expected, stop_file):
        self.calls.append(("pressure", seconds, duty, expected["core_mhz"], expected["mem_mhz"]))
        sample = {"utilization": 100, "mem_clock": 11000 + expected["mem_mhz"], "gr_clock": 2500}
        clocks = NightBackend._loaded_clocks(self, [sample], expected)
        return {"outcome": "pass", "score": 123.0, "load_started": True, "details": "synthetic evidence",
                "telemetry": clocks | {"peak_temp_c": 60, "avg_power_w": 180, "peak_power_w": 200}}
    def _driver_events(self, *args, **kwargs):
        return []


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.adapter = FakeSession()
        self.scope = Mock()
        self.backend = WindowsNvidiaBackend(config(), self.root, _adapter=self.adapter,
            _fans_factory=FakeFans, _scope_factory=lambda seconds: self.scope,
            _awake_factory=nullcontext)
        self.addCleanup(self.backend.close)
    def baseline(self):
        inspected = self.backend.inspect()
        profile = Profile.from_dict(inspected["initial"])
        with patch("gpu_limitseeker.hardware.backend.wait_for_idle", side_effect=lambda read, stopped: read()):
            result = self.backend.run_trial(TrialSpec("baseline", "baseline", "screen", profile, 2, 1, 0), self.root / "STOP")
        self.assertEqual(result.outcome, "pass", result.details)
        return profile, result
    def test_constructor_is_lazy(self):
        with patch.object(NativeSession, "open", side_effect=AssertionError("hardware touched")):
            WindowsNvidiaBackend(config(), self.root)
        self.assertEqual(self.adapter.calls, [])
    def test_exact_baseline_before_any_write_and_loaded_reference(self):
        profile, result = self.baseline()
        self.assertEqual(profile.core, 60)
        self.assertEqual(profile.mem, 200)
        self.assertEqual(self.adapter.memory_base_mhz, 11000)
        self.assertFalse(any(isinstance(c, tuple) and c[0] in ("uniform", "power") for c in self.adapter.calls))
        self.assertEqual([w["kind"] for w in result.workloads], ["full", "half"])
        self.assertTrue(all(w["completed"] for w in result.workloads))
        self.assertEqual(result.full_avg_power_w, 180)
        self.assertEqual(result.peak_power_w, 200)
    def test_nonbaseline_first_trial_refused_without_write(self):
        self.backend.inspect()
        result = self.backend.run_trial(TrialSpec("wrong-first", "screen", "screen", Profile("uniform", 0, 0), 1, 0, 0), self.root / "STOP")
        self.assertEqual(result.outcome, "invalid")
        self.assertFalse(any(isinstance(c, tuple) for c in self.adapter.calls))
    def test_afterburner_requires_passing_baseline_and_permission(self):
        self.adapter.mode = "macm"
        self.baseline()
        with self.assertRaises(SafetyError):
            self.backend.apply(Profile("uniform", 90, 200))
        self.assertNotIn("handoff", self.adapter.calls)
    def test_restoration_does_not_overwrite_new_user_controller(self):
        self.baseline()
        self.adapter.owned = False
        result = self.backend.restore()
        self.assertFalse(result["restored"])
        self.assertFalse(any(isinstance(c, tuple) and c[0] == "uniform" for c in self.adapter.calls))
    def test_thermal_latch_blocks_recovery(self):
        profile, _ = self.baseline()
        self.backend._thermal_latched = True
        with self.assertRaisesRegex(SafetyError, "latched"):
            self.backend.recover_tdr(profile, self.root / "STOP")
        self.assertTrue(self.backend._thermal_latched)
    def test_default_tdp_drives_power_not_a_card_constant(self):
        inspected = self.backend.inspect()
        self.assertEqual(inspected["default_power_w"], 240)
        self.assertEqual(inspected["bounds"]["power_max_percent"], 130)
        self.adapter.state["power_w"] = 360
        with self.assertRaises(SafetyError):
            self.backend.inspect()
    def test_higher_power_requires_explicit_watt_cap(self):
        conf = config()
        conf["search"]["power_limits"] = [100, 125]
        conf["safety"]["max_power_percent"] = 125
        with self.assertRaises(SafetyError):
            WindowsNvidiaBackend(conf, self.root)
    def test_approved_power_write_uses_actual_tdp(self):
        conf = config()
        conf["search"]["power_limits"] = [100, 125]
        conf["safety"].update(max_power_percent=125, max_board_power_w=300)
        self.backend.config = conf
        self.baseline()
        with patch("gpu_limitseeker.hardware.backend.wait_for_idle", side_effect=lambda read, stopped: read()):
            result = self.backend.apply(Profile("uniform", 90, 200, power_percent=125))
        self.assertEqual(result["power_w"], 300)
    def test_explicit_memory_reference_must_match_real_loaded_clock(self):
        self.adapter.memory_base_mhz = 12000
        inspected = self.backend.inspect()
        with patch("gpu_limitseeker.hardware.backend.wait_for_idle", side_effect=lambda read, stopped: read()):
            result = self.backend.run_trial(TrialSpec("bad-reference", "baseline", "screen", Profile.from_dict(inspected["initial"]), 1, 0, 0), self.root / "STOP")
        self.assertEqual(result.outcome, "invalid")
        self.assertFalse(self.backend._baseline_passed)
    def test_sticky_native_thermal_latch_stops_remaining_loads_and_bandwidth(self):
        initial = Profile.from_dict(self.backend.inspect()["initial"])
        def pressure(*args):
            self.adapter.thermal_latched = True
            return {"outcome": "pass", "score": 123, "load_started": True,
                    "telemetry": {"peak_temp_c": 60}}
        self.adapter.run_test = Mock(side_effect=pressure)
        self.adapter.bandwidth = Mock(side_effect=AssertionError("bandwidth after thermal latch"))
        runner = Mock(side_effect=AssertionError("VRAM after thermal latch"))
        self.backend._vram_factory = runner
        with patch("gpu_limitseeker.hardware.backend.wait_for_idle", side_effect=lambda read, stopped: read()):
            result = self.backend.run_trial(TrialSpec("thermal-between", "baseline", "screen",
                initial, 1, 1, 1, bandwidth=True), self.root / "STOP")
        self.assertEqual(result.outcome, "hot", result.details)
        self.assertTrue(result.load_started)
        self.assertEqual(self.adapter.run_test.call_count, 1)
        self.adapter.bandwidth.assert_not_called()
        runner.assert_not_called()
        self.assertFalse(self.backend._baseline_passed)

    def test_thermal_latch_before_first_load_never_admits_pressure(self):
        initial = Profile.from_dict(self.backend.inspect()["initial"])
        self.adapter.thermal_latched = True
        self.adapter.run_test = Mock()
        result = self.backend.run_trial(TrialSpec("thermal-preflight", "baseline", "screen",
            initial, 1, 0, 0), self.root / "STOP")
        self.assertEqual(result.outcome, "hot")
        self.assertFalse(result.load_started)
        self.adapter.run_test.assert_not_called()
        self.assertIsNone(self.backend._scope)

    def test_fan_thermal_latch_aborts_before_external_or_pressure_admission(self):
        self.backend.inspect()
        loop = Mock()
        loop.thermal_event.is_set.return_value = True
        external = Mock()
        self.backend._fan_loop = loop
        try:
            with self.assertRaisesRegex(SafetyError, "latched"):
                self.backend._check(external)
            external.assert_not_called()
            loop.check.assert_not_called()
            self.assertTrue(self.backend._thermal_latched)
        finally:
            self.backend._fan_loop = None

    def test_thermal_latch_during_idle_blocks_write_after_cooling(self):
        self.baseline()
        self.adapter.calls.clear()
        def quiet_after_hot(read, stopped):
            self.adapter.thermal_latched = True
            return read()
        with patch("gpu_limitseeker.hardware.backend.wait_for_idle", side_effect=quiet_after_hot):
            with self.assertRaisesRegex(SafetyError, "latched"):
                self.backend.apply(Profile("uniform", 90, 200))
        self.assertFalse(any(isinstance(c, tuple) for c in self.adapter.calls))

    def test_reset_envelope_refused_before_macm_handoff_or_any_write(self):
        for limit, cap in ((80, None), (100, 200)):
            with self.subTest(limit=limit, cap=cap):
                self.adapter.mode = "macm"
                self.backend.config["hardware"]["allow_afterburner_handoff"] = True
                self.backend.config["safety"].update(power_percent=80, max_power_percent=limit,
                                                      max_board_power_w=cap)
                self.backend.config["search"]["power_limits"] = [80] if limit == 80 else [80, 100]
                self.adapter.power_percent = 80
                self.adapter.state.update(power_pct=80, power_w=192)
                self.backend._initial = self.backend._expected = self.backend._initial_profile = None
                self.backend.inspect()
                self.backend._baseline_passed = True
                self.adapter.calls.clear()
                with self.assertRaisesRegex(SafetyError, "reset"):
                    self.backend.apply(Profile("uniform", 90, 200, power_percent=80))
                self.assertNotIn("handoff", self.adapter.calls)
                self.assertFalse(any(isinstance(c, tuple) for c in self.adapter.calls))

    def test_restore_refuses_unsafe_temporary_default_power_before_writes(self):
        self.backend.config["safety"].update(power_percent=80, max_power_percent=80)
        self.backend.config["search"]["power_limits"] = [80]
        self.adapter.power_percent = 80
        self.adapter.state.update(power_pct=80, power_w=192)
        self.backend.inspect()
        self.backend._baseline_passed = True
        self.adapter.calls.clear()
        with self.assertRaisesRegex(SafetyError, "reset"):
            self.backend.restore()
        self.assertFalse(any(isinstance(c, tuple) for c in self.adapter.calls))

    def test_nonflat_custom_baseline_is_rejected_without_round_trip(self):
        points = self.adapter.state["curve_points_data"]
        points[1].update(mhz=2400, raw_offset=100000)
        points[2].update(mhz=2300, raw_offset=-40000)
        self.adapter.state["curve_digest"] = curve_digest(points)
        profile = Profile("curve", 60, 200, mv=1000, mhz=2400)
        self.backend.config["hardware"]["baseline_profile"] = profile.to_dict()
        NightBackend._curve_shape(self.adapter.state, self.adapter.state, 1000, 2400, 60)
        with self.assertRaisesRegex(SafetyError, "supported flat top"):
            self.backend.inspect()
        self.assertIsNone(self.backend._initial)
        self.assertFalse(any(isinstance(c, tuple) for c in self.adapter.calls))

    def test_flat_custom_baseline_requires_proven_raw_restore_before_writes(self):
        points = self.adapter.state["curve_points_data"]
        points[1].update(mhz=2500, raw_offset=100000)
        points[2].update(mhz=2500, raw_offset=-40000)
        self.adapter.state["curve_digest"] = curve_digest(points)
        self.backend.config["hardware"]["baseline_profile"] = Profile(
            "curve", 60, 200, mv=1000, mhz=2500).to_dict()
        self.adapter.prove_raw_restore = Mock(side_effect=BackendError("raw setter unavailable"))
        self.baseline()
        self.adapter.prove_raw_restore.assert_not_called()
        self.adapter.calls.clear()
        with patch("gpu_limitseeker.hardware.backend.wait_for_idle", side_effect=lambda read, stopped: read()):
            with self.assertRaisesRegex(SafetyError, "raw setter unavailable"):
                self.backend.apply(Profile("uniform", 90, 200))
        self.assertFalse(any(isinstance(c, tuple) for c in self.adapter.calls))
        self.adapter.prove_raw_restore.assert_called_once()

    def test_custom_baseline_restores_captured_vector_not_scalar_flatten(self):
        points = self.adapter.state["curve_points_data"]
        points[1].update(mhz=2500, raw_offset=100000)
        points[2].update(mhz=2500, raw_offset=-40000)
        self.adapter.state["curve_digest"] = curve_digest(points)
        original = deepcopy(self.adapter.state)
        self.backend.config["hardware"]["baseline_profile"] = Profile(
            "curve", 60, 200, mv=1000, mhz=2500).to_dict()
        def prove(expected, directory, bounded):
            self.assertEqual(expected["curve_digest"], original["curve_digest"])
            self.adapter._raw_restore_digest = expected["curve_digest"]
            self.adapter.calls.append("proof")
        def restore(expected, target, directory, bounded):
            self.adapter.calls.append("raw-restore")
            self.adapter.state["curve_points_data"] = deepcopy(target["curve_points_data"])
            self.adapter.state["curve_digest"] = target["curve_digest"]
            return self.adapter.inspect()
        self.adapter.prove_raw_restore = Mock(side_effect=prove)
        self.adapter.restore_raw_curve = Mock(side_effect=restore)
        self.baseline()
        with patch("gpu_limitseeker.hardware.backend.wait_for_idle", side_effect=lambda read, stopped: read()):
            self.backend.apply(Profile("uniform", 90, 200))
        self.assertLess(self.adapter.calls.index("proof"), self.adapter.calls.index(("power", 100)))
        result = self.backend.restore()
        self.assertTrue(result["restored"])
        self.assertEqual(result["snapshot"]["curve_digest"], original["curve_digest"])
        self.adapter.restore_raw_curve.assert_called_once()

    def test_exact_original_raw_vector_still_required_for_restore(self):
        self.baseline()
        original = self.adapter.apply_uniform
        def drifted_restore(*args, **kwargs):
            state = original(*args, **kwargs)
            state["curve_points_data"][-1]["raw_offset"] += 1000
            state["curve_digest"] = curve_digest(state["curve_points_data"])
            return state
        self.adapter.apply_uniform = drifted_restore
        with self.assertRaisesRegex(SafetyError, "exact original"):
            self.backend.restore()


class ResumeAndAdmissionTests(unittest.TestCase):
    def test_resume_uses_only_original_remaining_native_lifetime(self):
        with tempfile.TemporaryDirectory() as directory:
            conf = config()
            conf["_runtime_remaining_seconds"] = 75
            scope_factory = Mock(return_value=Mock())
            adapter = FakeSession()
            backend = WindowsNvidiaBackend(conf, Path(directory), _adapter=adapter,
                _fans_factory=FakeFans, _scope_factory=scope_factory, _awake_factory=nullcontext)
            try:
                initial = Profile.from_dict(backend.inspect()["initial"])
                with patch("gpu_limitseeker.hardware.backend.wait_for_idle", side_effect=lambda read, stopped: read()):
                    result = backend.run_trial(TrialSpec("resume-baseline-synthetic", "baseline", "screen",
                                                       initial, 2, 1, 0), Path(directory) / "STOP")
                self.assertEqual(result.outcome, "pass", result.details)
                scope_factory.assert_called_once_with(375)
            finally:
                backend.close()

    def test_unconfirmed_positive_score_does_not_unlock_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeSession()
            adapter.run_test = Mock(return_value={"outcome": "pass", "score": 0, "load_started": True,
                "telemetry": {"peak_temp_c": 60}})
            backend = WindowsNvidiaBackend(config(), Path(directory), _adapter=adapter,
                _fans_factory=FakeFans, _scope_factory=lambda seconds: Mock(), _awake_factory=nullcontext)
            try:
                initial = Profile.from_dict(backend.inspect()["initial"])
                with patch("gpu_limitseeker.hardware.backend.wait_for_idle", side_effect=lambda read, stopped: read()):
                    result = backend.run_trial(TrialSpec("zero-work", "baseline", "screen", initial, 1, 0, 0),
                                               Path(directory) / "STOP")
                self.assertEqual(result.outcome, "invalid")
                self.assertFalse(backend._baseline_passed)
            finally:
                backend.close()


class ParserTests(unittest.TestCase):
    def test_generic_pci_identity(self):
        identity = parse_identity("NVIDIA Synthetic Adapter, 00000000:03:00.0, 0x123410DE, 12000")
        self.assertEqual(identity["pci_bus"], 3)
        self.assertEqual(identity["pci_device_id"], 0x1234)
        self.assertEqual(identity["memory_gib"], 12)
    def test_multiple_devices_are_explicitly_unsupported(self):
        row = f"{UUID}, 600.01, 240, 240, 40, 300, 400, 0, 15, P8"
        with self.assertRaises(BackendError):
            parse_smi(row + "\n" + row)
    def test_full_raw_curve_preserved_and_truncation_refused(self):
        text = "voltage/frequency curve: 2 points, 900 to 1000 mV\n slot mV MHz raw offset\n 0 900 2000.0 60000\n 1 1000 2300.0 -10000\n"
        points = parse_full_curve(text)
        self.assertEqual([p["raw_offset"] for p in points], [60000, -10000])
        with self.assertRaises(BackendError):
            parse_full_curve(text.rsplit(" 1 ", 1)[0])
    def test_real_p8_low_power_desktop_admission(self):
        sample = {"utilization": 90, "pstate": "P8", "draw_w": 15, "default_power_w": 240, "mem_clock": 400}
        clock = lambda: 0
        self.assertEqual(wait_for_idle(lambda: sample, lambda: False, clock=clock, pause=lambda seconds: None), sample)
    def test_high_memory_desktop_is_not_admitted_as_p8_idle(self):
        sample = {"utilization": 90, "pstate": "P8", "draw_w": 15, "default_power_w": 240, "mem_clock": 11000}
        clock = Mock(side_effect=[0, 0, 31])
        with self.assertRaises(BackendError):
            wait_for_idle(lambda: sample, lambda: False, clock=clock, pause=lambda seconds: None)


if __name__ == "__main__":
    unittest.main()
