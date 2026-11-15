"""Synthetic official MACM v2.3 byte-layout and preservation tests; no native calls."""
import struct
import unittest
from gpu_limitseeker.hardware.macm import (ENTRY_SIZE, FLUSH, HEADER_SIZE, MACMError,
    SIGNATURE, VERSION, guard_snapshot, parse_snapshot, validate_offsets)

BASE = HEADER_SIZE + ENTRY_SIZE
GPU_ID = "VEN_10DE&DEV_1234&SUBSYS_00000000&REV_00&BUS_7&DEV_0&FN_0"


def fixture(core=60, mem=200, fan_auto=False):
    data = bytearray(HEADER_SIZE + 2 * ENTRY_SIZE)
    struct.pack_into("<9I", data, 0, SIGNATURE, VERSION, HEADER_SIZE, 2, ENTRY_SIZE, 1, 7, 1000, 0)
    other = b"VEN_1002&DEV_0001&SUBSYS_00000000&REV_00&BUS_0&DEV_0&FN_0\0"
    data[HEADER_SIZE + 3500:HEADER_SIZE + 3500 + len(other)] = other
    identity = GPU_ID.encode("ascii") + b"\0"
    data[BASE + 3500:BASE + 3500 + len(identity)] = identity
    struct.pack_into("<I", data, BASE, 0x80021C88)
    for offset, value in ((52, 50), (56, int(fan_auto)), (172, 100), (176, 50), (180, 130),
                          (184, 100), (188, core * 1000), (192, -500000), (196, 800000),
                          (204, mem * 1000), (208, -1000000), (212, 3000000), (284, 3)):
        struct.pack_into("<i", data, BASE + offset, value)
    for index in range(3):
        struct.pack_into("<IIi", data, BASE + 288 + index * 12,
                         900000 + index * 50000, 2000000 + index * 100000, core * 1000)
    return data


def put(data, offset, value):
    result = bytearray(data)
    struct.pack_into("<I", result, offset, value)
    return result


class MacmTests(unittest.TestCase):
    def test_selects_single_nvidia_not_entry_zero(self):
        state = parse_snapshot(fixture())
        self.assertEqual(state["macm_gpu_id"], GPU_ID)
        self.assertEqual(state["macm_index"], 1)
        self.assertEqual(state["curve_offsets_khz"], [60000] * 3)
        self.assertEqual((state["core_mhz"], state["mem_mhz"]), (60, 200))
    def test_incomplete_or_unexpected_abi_refused(self):
        data = fixture()
        for invalid in (b"", data[:35], data[:-1], data + b"x", put(data, 0, 0xDEAD),
                        put(data, 4, 0x20004), put(data, 16, ENTRY_SIZE - 4), put(data, 20, 2)):
            with self.subTest(size=len(invalid)), self.assertRaises(MACMError):
                parse_snapshot(invalid)
    def test_pending_unknown_commands_not_replayed(self):
        with self.assertRaises(MACMError):
            parse_snapshot(put(fixture(), 32, FLUSH))
        self.assertEqual(parse_snapshot(put(fixture(), 32, FLUSH), allow_pending=True)["macm_command"], FLUSH)
        with self.assertRaises(MACMError):
            parse_snapshot(put(fixture(), 32, 0x1234), allow_pending=True)
    def test_fractional_driver_bounds_round_inward(self):
        data = fixture()
        struct.pack_into("<i", data, BASE + 192, -502296)
        struct.pack_into("<i", data, BASE + 196, 800999)
        parsed = parse_snapshot(data)
        self.assertEqual((parsed["core_min"], parsed["core_max"]), (-502, 800))
        with self.assertRaises(MACMError):
            validate_offsets(801, 200, parsed)
    def test_entire_preserved_control_fingerprint_is_guarded(self):
        original = parse_snapshot(fixture())
        for offset, value in ((52, 51), (56, 1), (80, 1000), (128, 5), (172, 101), (3360, 1)):
            changed = parse_snapshot(put(fixture(), BASE + offset, value))
            with self.subTest(offset=offset), self.assertRaises(MACMError):
                guard_snapshot(changed, original)
    def test_dynamic_frequency_and_auto_fan_not_false_drift(self):
        data = fixture(fan_auto=True)
        original = parse_snapshot(data)
        changed = put(data, BASE + 52, 75)
        changed = put(changed, BASE + 4, 2500000)
        changed = put(changed, BASE + 288 + 4, 2100000)
        guard_snapshot(parse_snapshot(changed), original)
    def test_another_writable_card_or_duplicate_nvidia_fails_closed(self):
        data = fixture()
        with self.assertRaises(MACMError):
            parse_snapshot(put(data, HEADER_SIZE, 1))
        identifier = GPU_ID.encode() + b"\0"
        data[HEADER_SIZE + 3500:HEADER_SIZE + 3500 + len(identifier)] = identifier
        with self.assertRaises(MACMError):
            parse_snapshot(data)
    def test_custom_or_locked_curve_never_scalar_flushed(self):
        data = fixture()
        struct.pack_into("<i", data, BASE + 288 + 8, 61000)
        with self.assertRaises(MACMError):
            validate_offsets(90, 200, parse_snapshot(data))
        with self.assertRaises(MACMError):
            validate_offsets(90, 200, parse_snapshot(put(fixture(), BASE + 3360, 1)))


if __name__ == "__main__":
    unittest.main()
