"""Synthetic pinned-ABI and exact-original restoration regressions; no DLL/GPU access."""
from copy import deepcopy
import ctypes
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

from gpu_limitseeker.hardware.macm_backend import BackendError
from gpu_limitseeker.hardware.night_backend import NightBackend, curve_digest
from gpu_limitseeker.hardware.raw_curve import RawCurve


POINTS = [{"index": 0, "mv": 900, "mhz": 2060, "raw_offset": 60000},
          {"index": 2, "mv": 1000, "mhz": 2500, "raw_offset": 100000},
          {"index": 5, "mv": 1050, "mhz": 2500, "raw_offset": -40000}]
IDENTITY = {"uuid": "GPU-synthetic", "driver": "600.01", "name": "NVIDIA Synthetic",
            "power_w": 240, "default_power_w": 240}


class RawAbiTests(unittest.TestCase):
    def bridge(self):
        bridge = RawCurve.__new__(RawCurve)
        bridge.handle = 123
        self.offsets = {p["index"]: p["raw_offset"] for p in POINTS}
        self.calls, self.set_buffers = [], []
        def call(identifier, buffer):
            self.calls.append(identifier)
            size = {bridge.MASK: 6188, bridge.STATUS: 7208, bridge.GET: 9248, bridge.SET: 9248}[identifier]
            self.assertEqual(len(buffer), size)
            self.assertEqual(struct.unpack_from("<I", buffer)[0], size | 0x10000)
            if identifier == bridge.MASK:
                self.assertEqual(buffer.raw[4:36], b"\xff" * 32)
                for point in POINTS:
                    offset = 68 + point["index"] * 24
                    struct.pack_into("<I", buffer, offset, 0)
                    buffer[offset + 4] = b"\x01"
            elif identifier == bridge.STATUS:
                for point in POINTS:
                    struct.pack_into("<II", buffer, 68 + point["index"] * 28 + 4,
                                     int(point["mhz"] * 1000), point["mv"] * 1000)
            elif identifier == bridge.GET:
                # Unrelated/reserved bytes must survive the read-modify-write.
                buffer[42] = b"\xa5"
                for index, raw in self.offsets.items():
                    struct.pack_into("<i", buffer, 68 + index * 36 + 20, raw)
            elif identifier == bridge.SET:
                self.set_buffers.append(buffer.raw)
                self.assertEqual(buffer.raw[42], 0xa5)
                for point in POINTS:
                    self.offsets[point["index"]] = struct.unpack_from(
                        "<i", buffer, 68 + point["index"] * 36 + 20)[0]
        bridge._vf_call = call
        return bridge

    def test_pinned_sizes_ids_and_exact_complete_vector_write(self):
        bridge = self.bridge()
        target = deepcopy(POINTS)
        target[-1]["raw_offset"] = -55000
        with patch("gpu_limitseeker.hardware.raw_curve.time.sleep"):
            bridge.write_exact(POINTS, target)
        self.assertEqual(self.calls, [0x507B4B59, 0x21537AD4, 0x23F1B133,
                                     0x0733E009, 0x507B4B59, 0x21537AD4, 0x23F1B133])
        self.assertEqual(self.offsets, {p["index"]: p["raw_offset"] for p in target})
        self.assertEqual(len(self.set_buffers), 1)

    def test_same_vector_proof_uses_one_real_setter_without_intended_change(self):
        bridge = self.bridge()
        with patch("gpu_limitseeker.hardware.raw_curve.time.sleep"):
            bridge.write_exact(POINTS, POINTS)
        self.assertEqual(self.offsets, {p["index"]: p["raw_offset"] for p in POINTS})
        self.assertEqual(self.calls.count(bridge.SET), 1)

    def test_current_native_disagreement_refuses_before_set(self):
        bridge = self.bridge()
        self.offsets[5] += 1
        with self.assertRaisesRegex(BackendError, "no write"):
            bridge.write_exact(POINTS, POINTS)
        self.assertNotIn(bridge.SET, self.calls)

    def test_incomplete_or_duplicate_topology_refuses_before_set(self):
        for target in (POINTS[:-1], POINTS + [POINTS[0]], [POINTS[0] | {"index": 255}]):
            with self.subTest(target=target):
                bridge = self.bridge()
                with self.assertRaises(BackendError):
                    bridge.write_exact(POINTS, target)
                self.assertNotIn(bridge.SET, self.calls)

    def test_silent_setter_mismatch_is_not_replayed(self):
        bridge = self.bridge()
        original = bridge._vf_call
        def ignored(identifier, buffer):
            if identifier == bridge.SET:
                self.calls.append(identifier)
            else:
                original(identifier, buffer)
        bridge._vf_call = ignored
        target = deepcopy(POINTS)
        target[1]["raw_offset"] += 1000
        with patch("gpu_limitseeker.hardware.raw_curve.time.sleep"):
            with self.assertRaisesRegex(BackendError, "no replay"):
                bridge.write_exact(POINTS, target)
        self.assertEqual(self.calls.count(bridge.SET), 1)

    def test_initialization_requires_single_physical_gpu_and_unloads(self):
        class InventoryBridge(RawCurve):
            def _call(inner, identifier, signature, *args):
                if identifier == inner.ENUM:
                    args[0][0] = 123
                    ctypes.cast(args[1], ctypes.POINTER(ctypes.c_uint32))[0] = 2
                if identifier == inner.UNLOAD:
                    inner.unloaded = True
        library = Mock()
        with patch("gpu_limitseeker.hardware.raw_curve.os.name", "nt"), \
             patch("gpu_limitseeker.hardware.raw_curve.ctypes.CDLL", return_value=library) as load:
            with self.assertRaisesRegex(BackendError, "exactly one"):
                InventoryBridge()
        load.assert_called_once_with("nvapi64.dll", winmode=0x00000800)


class RawRestoreIntegrationTests(unittest.TestCase):
    def adapter(self, root):
        adapter = NightBackend(root, Path("synthetic.exe"), IDENTITY, 11000)
        adapter.state = dict(IDENTITY, core_mhz=60, mem_mhz=200, power_pct=100,
                             nvidia_off=True, curve_points_data=deepcopy(POINTS),
                             curve_digest=curve_digest(POINTS), curve_anchor_mv=1000,
                             curve_anchor_mhz=2500)
        adapter._write_allowed = Mock()
        adapter.inspect = Mock(side_effect=lambda directory: deepcopy(adapter.state))
        return adapter

    def test_same_vector_proof_requires_independent_gao_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(Path(directory))
            expected = deepcopy(adapter.state)
            def drift(expected_points, target_points):
                self.assertEqual(expected_points, target_points)
                adapter.state["curve_points_data"][-1]["raw_offset"] += 1
                adapter.state["curve_digest"] = curve_digest(adapter.state["curve_points_data"])
            adapter._raw_vector_write = Mock(side_effect=drift)
            with self.assertRaisesRegex(BackendError, "Independent GAO"):
                adapter.prove_raw_restore(expected, Path(directory), lambda call: call())
            self.assertTrue(adapter._uncertain)
            self.assertIsNone(adapter._raw_restore_digest)
            adapter._raw_vector_write.assert_called_once()

    def test_proven_original_vector_restores_exactly_without_frequency_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(Path(directory))
            original = deepcopy(adapter.state)
            def write(expected, target):
                self.assertEqual(expected, adapter.state["curve_points_data"])
                adapter.state["curve_points_data"] = deepcopy(target)
                adapter.state["curve_digest"] = curve_digest(target)
            adapter._raw_vector_write = Mock(side_effect=write)
            adapter.prove_raw_restore(original, Path(directory), lambda call: call())
            self.assertEqual(adapter._raw_restore_digest, original["curve_digest"])
            for point in adapter.state["curve_points_data"]:
                point["raw_offset"] = 60000
                point["mhz"] += 30  # Temperature-sensitive MHz is never used to rebuild offsets.
            adapter.state["curve_digest"] = curve_digest(adapter.state["curve_points_data"])
            current = deepcopy(adapter.state)
            restored = adapter.restore_raw_curve(current, original, Path(directory), lambda call: call())
            self.assertEqual(restored["curve_digest"], original["curve_digest"])
            self.assertEqual([p["raw_offset"] for p in restored["curve_points_data"]], [60000, 100000, -40000])
            self.assertFalse(adapter._uncertain)
            self.assertEqual(adapter._raw_vector_write.call_count, 2)

    def test_identity_or_unproven_original_refuses_before_raw_write(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(Path(directory))
            original = deepcopy(adapter.state)
            adapter._raw_vector_write = Mock()
            with self.assertRaisesRegex(BackendError, "not proven"):
                adapter.restore_raw_curve(original, original, Path(directory), lambda call: call())
            adapter.state["uuid"] = "GPU-another"
            with self.assertRaisesRegex(BackendError, "uuid drifted"):
                adapter.prove_raw_restore(original, Path(directory), lambda call: call())
            adapter._raw_vector_write.assert_not_called()

    def test_native_failure_latches_uncertainty_and_prevents_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(Path(directory))
            expected = deepcopy(adapter.state)
            adapter._raw_vector_write = Mock(side_effect=BackendError("native uncertain"))
            with self.assertRaisesRegex(BackendError, "native uncertain"):
                adapter.prove_raw_restore(expected, Path(directory), lambda call: call())
            with self.assertRaisesRegex(BackendError, "Uncertain control write"):
                adapter.prove_raw_restore(expected, Path(directory), lambda call: call())
            adapter._raw_vector_write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
