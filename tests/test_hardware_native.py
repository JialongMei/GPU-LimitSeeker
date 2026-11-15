"""Offline native protocol and ownership regressions, using synthetic fixtures only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from gpu_limitseeker.hardware.backend import WindowsNvidiaBackend
from gpu_limitseeker.hardware.fan_control import FanError, FanLoop
from gpu_limitseeker.hardware.icd import discover_nvidia_icd, validate_manifest
from gpu_limitseeker.hardware.macm_backend import BackendError
from gpu_limitseeker.hardware.native import NativeSession
from gpu_limitseeker.hardware.night_backend import NightBackend, curve_digest
from gpu_limitseeker.hardware.vram_runner import OutputEvidence, classify_result, decode_exit_status, VramRunner
from gpu_limitseeker.models import SafetyError


IDENTITY = {"uuid": "GPU-00000000-0000-0000-0000-000000000002", "name": "NVIDIA Synthetic Device",
            "pci_bus": 7, "pci_device": 0, "pci_device_id": 0x5678, "memory_gib": 12,
            "driver": "600.02", "power_w": 300.0, "default_power_w": 240.0, "memory_total_mib": 12000}


def evidence():
    result = OutputEvidence(IDENTITY)
    result.feed("Standard 5-minute test of 1: Bus=0x7:0 DevId=0x5678 12GB NVIDIA Synthetic Device\n"
                "1 iteration. Passed 10.0 seconds written: 1.0GB 1.0GB/sec checked: 1.0GB 1.0GB/sec\n", final=True)
    return result


class VramEvidenceTests(unittest.TestCase):
    def test_exact_noninteractive_0x51_and_positive_work(self):
        parsed = evidence()
        self.assertTrue(parsed.positive_work)
        result = classify_result(0x51, parsed, seconds=10, elapsed=10.1, timed_stop=True,
                                 events=[], state_ok=True, telemetry_count=1)
        self.assertEqual(result["outcome"], "pass")
        self.assertTrue(decode_exit_status(0x51)["quit_requested"])
    def test_clean_signature_without_actual_work_never_passes(self):
        result = classify_result(0x51, OutputEvidence(IDENTITY), seconds=10, elapsed=11, timed_stop=True,
                                 events=[], state_ok=True, telemetry_count=1)
        self.assertEqual(result["outcome"], "invalid")
    def test_unexpected_pci_identity_fails_closed(self):
        parsed = OutputEvidence(IDENTITY)
        parsed.feed("Standard 5-minute test of 1: Bus=0x8:0 DevId=0x5678 12GB NVIDIA Synthetic Device\n", final=True)
        self.assertFalse(parsed.identity_ok)
        self.assertTrue(parsed.invalid)
    def test_interactive_handler_or_zero_exit_never_pass(self):
        for code in (0, 0x49, 0x41):
            result = classify_result(code, evidence(), seconds=10, elapsed=11, timed_stop=True,
                                     events=[], state_ok=True, telemetry_count=1)
            self.assertEqual(result["outcome"], "invalid")
    def test_no_archival_crcrlf_weakening_in_live_progress(self):
        parsed = OutputEvidence(IDENTITY)
        parsed.feed("Standard 5-minute test of 1: Bus=0x7:0 DevId=0x5678 12GB NVIDIA Synthetic Device\n"
                    "1 iteration. Passed bad seconds written: 1GB 1GB/sec checked: 1GB 1GB/sec\n", final=True)
        self.assertTrue(parsed.invalid)
        self.assertFalse(parsed.positive_work)
    def test_vram_respects_root_stop_before_pressure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = Mock()
            backend._stopped.side_effect = lambda marker: Path(marker).exists() or (root / "STOP").exists()
            runner = VramRunner(root, backend, IDENTITY, root / "synthetic.exe")
            (root / "STOP").touch()
            with patch.object(runner, "verify_vendor"), patch.object(runner, "_require_console"), patch("gpu_limitseeker.hardware.vram_runner.subprocess.Popen", side_effect=AssertionError("pressure launched")):
                result = runner.run(1, root / "trial", {}, root / "elsewhere" / "STOP")
            self.assertEqual(result["outcome"], "stopped")
            self.assertFalse(result["load_started"])
    def test_loader_overrides_are_removed_and_verified_icd_selected(self):
        runner = VramRunner(Path("."), Mock(), IDENTITY, Path("synthetic.exe"))
        runner.icd = Path("synthetic-driver.json")
        with patch.dict("os.environ", {"VK_DRIVER_FILES": "wrong.json", "MEMTEST_VULKAN_EMULATE_ERRORS": "1", "VK_INSTANCE_LAYERS": "foreign"}):
            environment = runner.child_environment()
        self.assertEqual(environment["VK_DRIVER_FILES"], str(runner.icd))
        self.assertNotIn("MEMTEST_VULKAN_EMULATE_ERRORS", environment)
        self.assertNotIn("VK_INSTANCE_LAYERS", environment)


class _Key:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        return self.name
    def __exit__(self, *args):
        return False


class FakeRegistry:
    HKEY_LOCAL_MACHINE = object()
    def __init__(self, library, manifest):
        self.values = {r"HARDWARE\DEVICEMAP\VIDEO": [(r"\Device\Video0", r"\Registry\Machine\SYSTEM\SyntheticDriver", 1)],
                       r"SOFTWARE\Khronos\Vulkan\Drivers": [(str(manifest), 0, 4)]}
        self.library = library
    def OpenKey(self, root, name):
        if name not in self.values and name != r"SYSTEM\SyntheticDriver":
            raise OSError("missing synthetic registry key")
        return _Key(name)
    def EnumValue(self, key, index):
        try:
            return self.values[key][index]
        except (KeyError, IndexError):
            raise OSError("end")
    def QueryValueEx(self, key, value):
        return [str(self.library)], 7


class IcdTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.library = self.root / "nvoglv64.dll"
        self.library.write_bytes(b"synthetic fixture, not a vendor library")
        self.manifest = self.root / "nv-vk64.json"
        self.manifest.write_text(json.dumps({"file_format_version": "1.0.1", "ICD": {"library_path": "nvoglv64.dll"}}), encoding="utf-8")
    def test_active_registry_binding_not_driver_store_pin(self):
        result = discover_nvidia_icd(registry=FakeRegistry(self.library, self.manifest))
        self.assertEqual(result, self.manifest.resolve())
    def test_explicit_manifest_must_still_match_active_driver(self):
        other = self.root / "old" / "nvoglv64.dll"
        other.parent.mkdir()
        other.write_bytes(b"old synthetic driver")
        with self.assertRaises(BackendError):
            validate_manifest(self.manifest, {other.resolve()})
    def test_absent_active_binding_never_scans_stale_installations(self):
        registry = FakeRegistry(self.library, self.manifest)
        registry.values[r"HARDWARE\DEVICEMAP\VIDEO"] = []
        with self.assertRaises(BackendError):
            discover_nvidia_icd(self.manifest, registry=registry)


class ResetAdapter(NightBackend):
    def __init__(self, root):
        super().__init__(root, Path("synthetic.exe"), IDENTITY, 11000, 125)
        self.commands = []
        self.state = dict(IDENTITY, core_mhz=60, mem_mhz=200, core_max=800, mem_max=3000, power_pct=125,
                          nvidia_off=True)
        self._vector(60)
    def _vector(self, core):
        self.state["curve_points_data"] = [
            {"index": 0, "mv": 900, "mhz": 2000 + core, "raw_offset": core * 1000},
            {"index": 1, "mv": 1000, "mhz": 2300 + core, "raw_offset": core * 1000}]
        self.state["curve_digest"] = curve_digest(self.state["curve_points_data"])
    def _write_allowed(self):
        pass
    def inspect(self, log_dir=None):
        from copy import deepcopy
        return deepcopy(self.state)
    def _run(self, args, label, *rest):
        self.commands.append(args[1:])
        if args[1] == "--reset":
            self.state.update(core_mhz=0, mem_mhz=0, power_pct=100, power_w=240.0)
            self._vector(0)
        elif args[1] == "--set-core":
            self.state["core_mhz"] = int(args[2])
            self._vector(int(args[2]))
        elif args[1] == "--set-mem":
            self.state["mem_mhz"] = int(args[2])
        else:
            raise AssertionError("Unexpected synthetic command")
        return ""
    def set_power(self, percent, log_dir):
        self.commands.append(["power", percent])
        self.power_percent = percent
        self.identity["power_w"] = 240 * percent / 100
        self.state.update(power_pct=percent, power_w=self.identity["power_w"])
        return self.inspect()


class NativeOwnershipTests(unittest.TestCase):
    def test_reset_temporarily_100_then_approved_power_before_paired_write(self):
        adapter = ResetAdapter(Path("."))
        result = adapter.apply_uniform(90, 300, Path("."), reset=True)
        self.assertEqual(adapter.commands, [["--reset"], ["power", 125], ["--set-core", "90"], ["--set-mem", "300"]])
        self.assertEqual(result["power_pct"], 125)
        self.assertEqual(result["power_w"], 300)
        self.assertFalse(adapter._uncertain)
    def test_reset_refuses_temporary_default_outside_percent_or_watt_approval(self):
        for percent, maximum, cap in ((80, 80, None), (80, 100, 200), (125, 125, 250)):
            with self.subTest(percent=percent, maximum=maximum, cap=cap):
                adapter = ResetAdapter(Path("."))
                adapter.power_percent, adapter.max_power_percent = percent, maximum
                adapter.max_board_power_w = cap
                adapter.identity["power_w"] = 240 * percent / 100
                adapter.state.update(power_pct=percent, power_w=adapter.identity["power_w"])
                with self.assertRaisesRegex(BackendError, "envelope"):
                    adapter.apply_uniform(90, 300, Path("."), reset=True)
                self.assertEqual(adapter.commands, [])
                self.assertEqual(adapter.power_percent, percent)
                self.assertFalse(adapter._uncertain)

    def test_direct_handoff_refuses_unsafe_reset_before_clearing_macm(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = NativeSession(Path(directory), Path("gao.exe"),
                                    IDENTITY | {"power_w": 192}, 11000, 80)
            adapter.mode = "macm"
            adapter.macm = Mock()
            adapter._write_allowed = Mock()
            with self.assertRaisesRegex(BackendError, "envelope"):
                adapter.handoff({}, Path(directory), lambda: False)
            adapter.macm.set_offsets.assert_not_called()
            adapter._write_allowed.assert_not_called()

    def test_power_writer_refuses_unapproved_watts_before_native_write(self):
        adapter = NativeSession(Path("."), Path("gao.exe"), IDENTITY, 11000, 125)
        adapter.power_min_percent, adapter.power_max_percent = 50, 150
        adapter.max_board_power_w = 250
        adapter._run, adapter._write_allowed = Mock(), Mock()
        with self.assertRaisesRegex(BackendError, "board-watt"):
            adapter.set_power(125, Path("."))
        adapter._run.assert_not_called()
        adapter._write_allowed.assert_not_called()
    def test_loaded_clock_reference_is_derived_not_card_specific(self):
        adapter = ResetAdapter(Path("."))
        adapter.memory_base_mhz = None
        adapter.derive_memory_reference = True
        result = adapter._loaded_clocks([{"utilization": 100, "mem_clock": 11400, "gr_clock": 2500}], adapter.state)
        self.assertEqual(adapter.memory_base_mhz, 11200)
        self.assertEqual(result["peak_mem_clock_mhz"], 11400)
    def test_fan_worker_unknown_cleanup_never_invokes_competing_close(self):
        config = {"hardware": {}, "safety": {"power_percent": 100, "max_temp_c": 80, "max_hours": 1, "max_recoveries": 2}}
        adapter = Mock()
        adapter.owns_controls.return_value = True
        backend = WindowsNvidiaBackend(config, Path("."), _adapter=adapter)
        loop = Mock()
        loop.thermal_event.is_set.return_value = False
        loop._thread.is_alive.return_value = True
        loop.__exit__ = Mock(side_effect=FanError("native call still running"))
        fans = Mock()
        backend._fan_loop, backend._fans = loop, fans
        with self.assertRaises(FanError):
            backend.close()
        fans.close.assert_not_called()
        self.assertTrue(backend._cleanup_unknown)
    def test_fan_relinquishes_without_auto_write_to_new_controller(self):
        fans = Mock(minimum=25, maximum=100, count=1)
        import threading
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loop = FanLoop(fans, [[30, 30], [78, 100]], threading.Event(), root / "STOP", root / "fans.jsonl", ownership=lambda: False)
            loop._run()
            fans.restore_auto.assert_not_called()
            fans.set_percent.assert_not_called()
            self.assertTrue(loop._relinquished.is_set())

    def test_fan_retained_target_and_policy_checked_even_at_unchanged_target(self):
        import threading
        for targets, policies in (([35, 40], [1, 1]), ([40, 40], [1, 0])):
            with self.subTest(targets=targets, policies=policies), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fans = Mock(minimum=25, maximum=100, count=2)
                fans.read.return_value = {"targets": targets, "policies": policies,
                                          "speeds": [40, 40], "temp_c": 40}
                loop = FanLoop(fans, [[30, 40], [78, 40]], threading.Event(),
                               root / "STOP", root / "fans.jsonl", ownership=lambda: True)
                loop._applied = 40
                loop._run()
                with self.assertRaisesRegex(FanError, "drifted"):
                    loop.check()
                fans.set_percent.assert_not_called()
                fans.restore_auto.assert_not_called()
                self.assertTrue(loop._relinquished.is_set())
                self.assertTrue(loop.stop_event.is_set())
                self.assertTrue((root / "STOP").exists())

    def test_fan_drift_is_not_overwritten_when_next_target_changes(self):
        import threading
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fans = Mock(minimum=25, maximum=100, count=1)
            fans.read.return_value = {"targets": [20], "policies": [1], "speeds": [20], "temp_c": 75}
            loop = FanLoop(fans, [[30, 30], [78, 100]], threading.Event(),
                           root / "STOP", root / "fans.jsonl", ownership=lambda: True)
            loop._applied = 40
            loop._run()
            with self.assertRaisesRegex(FanError, "drifted"):
                loop.check()
            fans.set_percent.assert_not_called()
            fans.restore_auto.assert_not_called()

    def test_fan_checks_each_sample_after_verified_first_command(self):
        import threading
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fans = Mock(minimum=25, maximum=100, count=1)
            fans.read.side_effect = [
                {"targets": [25], "policies": [0], "speeds": [25], "temp_c": 40},
                {"targets": [35], "policies": [1], "speeds": [35], "temp_c": 40}]
            loop = FanLoop(fans, [[30, 40], [78, 40]], threading.Event(),
                           root / "STOP", root / "fans.jsonl", ownership=lambda: True)
            loop._stop.wait = Mock(return_value=False)
            loop._run()
            with self.assertRaisesRegex(FanError, "drifted"):
                loop.check()
            fans.set_percent.assert_called_once_with(40)
            self.assertEqual(fans.read.call_count, 2)
            fans.restore_auto.assert_not_called()


class OwnedPressureReadbackTests(unittest.TestCase):
    def session(self, directory, executable="gao.exe"):
        adapter = NativeSession(Path(directory), Path(executable), IDENTITY, 11000, 100)
        adapter.mode = "macm"
        adapter.approved_handoff_processes = {("msiafterburner", 10, "100")}
        self.owner = {"ProcessName": "MSIAfterburner", "Id": 10, "StartTicks": "100"}
        self.child = Mock(pid=20)
        self.child.poll.return_value = None
        adapter._creation_ticks = Mock(return_value="200")
        adapter._claim_pressure(self.child, executable)
        self.pressure = {"ProcessName": Path(executable).stem, "Id": 20, "StartTicks": "200"}
        adapter._inventory = Mock(return_value=[self.owner, self.pressure])
        adapter.macm = Mock()
        adapter.macm.snapshot.return_value = {"power_pct": 100, "curve_offsets_khz": [60000, 60000]}
        adapter._run = Mock(return_value="synthetic")
        return adapter

    def controls(self, adapter):
        points = [{"index": 0, "mv": 900, "mhz": 2060, "raw_offset": 60000},
                  {"index": 1, "mv": 1000, "mhz": 2360, "raw_offset": 60000}]
        with patch("gpu_limitseeker.hardware.native.parse_status", return_value={
                "power_pct": 100, "core_mhz": 0, "mem_mhz": 0, "nvidia_off": True}), \
             patch("gpu_limitseeker.hardware.native.parse_full_curve", return_value=points), \
             patch("gpu_limitseeker.hardware.native.guard_snapshot"):
            return adapter._controls()

    def test_macm_live_gao_and_vram_child_do_not_invalidate_readback(self):
        for executable in ("gao.exe", "memtest_vulkan-v0.5.0.exe"):
            with self.subTest(executable=executable), tempfile.TemporaryDirectory() as directory:
                adapter = self.session(directory, executable)
                result = self.controls(adapter)
                self.assertEqual(result["curve_points_data"][0]["raw_offset"], 60000)
                self.assertTrue(adapter.owns_controls())
                self.assertEqual(adapter.blockers(), [])

    def test_unrelated_same_named_pressure_is_never_excluded(self):
        for executable in ("gao.exe", "memtest_vulkan-v0.5.0.exe"):
            with self.subTest(executable=executable), tempfile.TemporaryDirectory() as directory:
                adapter = self.session(directory, executable)
                adapter._inventory.return_value.append(self.pressure | {"Id": 21, "StartTicks": "201"})
                with self.assertRaisesRegex(BackendError, "competing"):
                    self.controls(adapter)
                self.assertFalse(adapter.owns_controls())

    def test_pid_reuse_or_missing_creation_identity_cannot_be_excluded(self):
        for replacement in ({"StartTicks": "201"}, {"StartTicks": None}, {"ProcessName": "memtest_vulkan"}):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                adapter = self.session(directory)
                adapter._inventory.return_value = [self.owner, self.pressure | replacement]
                with self.assertRaisesRegex(BackendError, "competing"):
                    self.controls(adapter)

    def test_exited_child_or_exit_during_inventory_cannot_hide_pressure(self):
        for polls in ([1], [None, 1]):
            with self.subTest(polls=polls), tempfile.TemporaryDirectory() as directory:
                adapter = self.session(directory)
                self.child.poll.side_effect = polls
                with self.assertRaisesRegex(BackendError, "competing"):
                    self.controls(adapter)

    def test_released_child_no_longer_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.session(directory)
            adapter._release_pressure(self.child)
            with self.assertRaisesRegex(BackendError, "competing"):
                self.controls(adapter)


if __name__ == "__main__":
    unittest.main()
