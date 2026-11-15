"""Bounded MSI Afterburner official MACM v2.3 shared-memory transport.

Only the explicit set_offsets method opens a writable view. It writes exactly
core boost, memory boost, and the SDK FLUSH command; never INIT or entire entries.
"""
from contextlib import contextmanager
import hashlib
import os
import re
import struct
import time


SIGNATURE = 0x4D41434D
VERSION = 0x00020003
HEADER_SIZE = 36
ENTRY_SIZE = 3760
FLUSH = 0x00AB0001
BOOST_FLAGS = 0x1800


class MACMError(RuntimeError):
    pass


def header_size(data: bytes) -> int:
    if len(data) < HEADER_SIZE:
        raise MACMError("Truncated MACM header")
    signature, version, size, count, stride, master, flags, timestamp, command = struct.unpack_from("<9I", data)
    if signature != SIGNATURE or version != VERSION or size != HEADER_SIZE or stride != ENTRY_SIZE:
        raise MACMError("Unsupported/uninitialized MACM layout (requires official v2.3)")
    if not 1 <= count <= 16 or master >= count:
        raise MACMError("Invalid MACM GPU count/master")
    return size + count * stride


def _i(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _u(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _mhz(value: int) -> int:
    if value % 1000:
        raise MACMError("MACM current boost is not an exact integer MHz")
    return value // 1000


def parse_snapshot(data: bytes, *, allow_pending: bool = False) -> dict:
    total = header_size(data)
    if len(data) != total:
        raise MACMError("Truncated/oversized MACM snapshot")
    command = _u(data, 32)
    if command != 0 and not (allow_pending and command == FLUSH):
        raise MACMError("MACM has an outstanding/unknown command; no replay permitted")
    entries = []
    for index in range(_u(data, 12)):
        base = HEADER_SIZE + index * ENTRY_SIZE
        raw_id = data[base + 3500:base + 3760]
        if b"\0" not in raw_id:
            raise MACMError("Unterminated MACM GPU identifier")
        try:
            gpu_id = raw_id.split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError as exc:
            raise MACMError("Unreadable MACM GPU identifier") from exc
        entries.append((index, base, gpu_id, _u(data, base)))
    nvidia = [entry for entry in entries if entry[2].upper().startswith("VEN_10DE&")]
    if len(nvidia) != 1:
        raise MACMError("MACM requires exactly one NVIDIA entry")
    index, base, gpu_id, flags = nvidia[0]
    if not re.fullmatch(r"VEN_10DE&DEV_[0-9A-F]{4}&SUBSYS_[0-9A-F]{8}&REV_[0-9A-F]{2}&BUS_\d+&DEV_\d+&FN_\d+", gpu_id, re.I):
        raise MACMError("Invalid MACM NVIDIA hardware identity")
    if any(entry[3] != 0 for entry in entries if entry[0] != index):
        raise MACMError("Another MACM GPU has writable controls; linked FLUSH refused")
    if _u(data, 20) != index or flags & BOOST_FLAGS != BOOST_FLAGS:
        raise MACMError("NVIDIA must be MACM master with both boost controls supported")
    count = _u(data, base + 284)
    if not 2 <= count <= 256:
        raise MACMError("MACM voltage-frequency curve is missing or invalid")
    points = [struct.unpack_from("<IIi", data, base + 288 + point * 12) for point in range(count)]
    if any(voltage == 0 or frequency == 0 for voltage, frequency, offset in points):
        raise MACMError("MACM curve has invalid voltage/frequency points")
    core, mem = _mhz(_i(data, base + 188)), _mhz(_i(data, base + 204))
    # Driver range endpoints can be fractional MHz; round the admissible integer range inward.
    cmin, cmax = (_i(data, base + 192) + 999) // 1000, _i(data, base + 196) // 1000
    mmin, mmax = (_i(data, base + 208) + 999) // 1000, _i(data, base + 212) // 1000
    if not cmin <= core <= cmax or not mmin <= mem <= mmax or not cmin <= 0 <= cmax or not mmin <= 0 <= mmax:
        raise MACMError("MACM offsets/ranges are invalid")
    raw_offsets = [offset for voltage, frequency, offset in points]
    uniform = len(set(raw_offsets)) == 1
    if uniform and raw_offsets[0] != core * 1000:
        raise MACMError("MACM curve disagrees with scalar core boost (stale state)")
    power = _i(data, base + 172)
    if not 0 < _i(data, base + 176) <= power <= _i(data, base + 180):
        raise MACMError("Invalid MACM power setting")
    # Fingerprint everything FLUSH could apply except our two offsets, dynamic
    # clocks/frequency, and the fan's dynamic current percentage in AUTO mode.
    preserved = bytearray(data)
    preserved[28:36] = bytes(8)  # timestamp and transient command
    for offset in (4, 20, 36, 188, 204):
        preserved[base + offset:base + offset + 4] = bytes(4)
    if _u(data, base + 56) & 1:
        preserved[base + 52:base + 56] = bytes(4)
    for point in range(count):
        point_base = base + 288 + point * 12
        preserved[point_base + 4:point_base + 12] = bytes(8)
    return {"macm_gpu_id": gpu_id, "macm_index": index, "macm_timestamp": _u(data, 28),
            "macm_command": command, "macm_flags": flags, "macm_header_flags": _u(data, 24),
            "macm_controls_digest": hashlib.sha256(preserved).hexdigest(),
            "core_mhz": core, "mem_mhz": mem, "core_min": cmin, "core_max": cmax,
            "mem_min": mmin, "mem_max": mmax, "power_pct": power,
            "curve_offset_khz": raw_offsets[0] if uniform else None, "curve_points": count,
            "curve_offsets_khz": raw_offsets, "curve_uniform": uniform,
            "curve_digest": hashlib.sha256(struct.pack(f"<{count}i", *raw_offsets)).hexdigest(),
            "macm_curve_enabled": bool(flags & 0x40000), "macm_curve_lock": _u(data, base + 3360)}


def guard_snapshot(actual: dict, expected: dict, *, offsets: bool = True) -> None:
    keys = ["macm_gpu_id", "macm_index", "macm_controls_digest", "power_pct"]
    if offsets:
        keys += ["core_mhz", "mem_mhz", "curve_digest"]
    for key in keys:
        if key not in actual or key not in expected or actual[key] != expected[key]:
            raise MACMError(f"MACM {key} changed or is missing; no command replay")


def validate_offsets(core: int, mem: int, snapshot: dict) -> None:
    if snapshot["macm_curve_enabled"] or snapshot["macm_curve_lock"] or not snapshot["curve_uniform"]:
        raise MACMError("Search requires an unlocked scalar boost curve, not a custom VF curve")
    for name, value in (("core", core), ("mem", mem)):
        maximum = snapshot[name + "_max"]
        if type(value) is not int or not 0 <= value <= maximum:
            raise MACMError(f"Requested {name} offset must be an integer in 0..{maximum} MHz")


class Transport:
    def _api(self):
        if os.name != "nt":
            raise MACMError("MACM transport requires Windows")
        import ctypes
        from ctypes import wintypes as w
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "OpenFileMappingW": ([w.DWORD, w.BOOL, w.LPCWSTR], w.HANDLE),
            "MapViewOfFile": ([w.HANDLE, w.DWORD, w.DWORD, w.DWORD, ctypes.c_size_t], ctypes.c_void_p),
            "UnmapViewOfFile": ([ctypes.c_void_p], w.BOOL),
            "CreateMutexW": ([ctypes.c_void_p, w.BOOL, w.LPCWSTR], w.HANDLE),
            "WaitForSingleObject": ([w.HANDLE, w.DWORD], w.DWORD),
            "ReleaseMutex": ([w.HANDLE], w.BOOL), "CloseHandle": ([w.HANDLE], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(api, name)
            function.argtypes, function.restype = args, result
        return ctypes, api

    @contextmanager
    def _mapping(self, writable: bool = False, wait_ms: int = 2000):
        ctypes, api = self._api()
        mutex = mapping = view = None
        acquired = False
        try:
            mutex = api.CreateMutexW(None, False, "Global\\Access_MACMSharedMemory")
            if not mutex:
                raise MACMError(f"Cannot open official MACM mutex (Windows error {ctypes.get_last_error()})")
            wait = api.WaitForSingleObject(mutex, max(1, min(wait_ms, 2000)))
            acquired = wait in (0, 0x80)
            if wait != 0:
                raise MACMError("MACM mutex timed out/was abandoned; refusing shared state")
            access = 4 | (2 if writable else 0)
            mapping = api.OpenFileMappingW(access, False, "MACMSharedMemory")
            if not mapping:
                raise MACMError("MSI Afterburner must be running with accessible MACM control memory")
            view = api.MapViewOfFile(mapping, access, 0, 0, HEADER_SIZE)
            if not view:
                raise MACMError("Cannot map MACM header")
            total = header_size(ctypes.string_at(view, HEADER_SIZE))
            api.UnmapViewOfFile(view)
            view = None
            view = api.MapViewOfFile(mapping, access, 0, 0, total)
            if not view:
                raise MACMError("MACM advertised size exceeds accessible shared memory")
            yield ctypes, view, total
        finally:
            if view:
                api.UnmapViewOfFile(view)
            if mapping:
                api.CloseHandle(mapping)
            if acquired:
                api.ReleaseMutex(mutex)
            if mutex:
                api.CloseHandle(mutex)

    def snapshot(self, *, allow_pending: bool = False, wait_ms: int = 2000) -> dict:
        with self._mapping(wait_ms=wait_ms) as (ctypes, view, total):
            return parse_snapshot(ctypes.string_at(view, total), allow_pending=allow_pending)

    def set_offsets(self, core: int, mem: int, expected: dict, stopped) -> dict:
        # Re-read the ENTIRE mapping under the official mutex immediately before
        # writing. There is no stale snapshot memcpy and no queued retry.
        with self._mapping(writable=True) as (ctypes, view, total):
            before = parse_snapshot(ctypes.string_at(view, total))
            guard_snapshot(before, expected)
            validate_offsets(core, mem, before)
            if stopped():
                raise MACMError("STOP requested before MACM FLUSH")
            base = HEADER_SIZE + before["macm_index"] * ENTRY_SIZE
            ctypes.memmove(view + base + 188, struct.pack("<i", core * 1000), 4)
            ctypes.memmove(view + base + 204, struct.pack("<i", mem * 1000), 4)
            ctypes.memmove(view + 32, struct.pack("<I", FLUSH), 4)
        deadline = time.monotonic() + 5
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MACMError("MACM FLUSH completion timed out; state uncertain, no replay")
            if stopped():
                raise MACMError("STOP during MACM FLUSH; command may still complete, no replay")
            # During FLUSH the curve can still contain the previous core offset;
            # only parse the completed state, not a partially processed table.
            with self._mapping(wait_ms=max(1, int(min(2, remaining) * 1000))) as (ctypes, view, total):
                data = ctypes.string_at(view, total)
                header_size(data)
                command = _u(data, 32)
                if command == 0:
                    after = parse_snapshot(data)
                    guard_snapshot(after, before, offsets=False)
                    if after["core_mhz"] != core or after["mem_mhz"] != mem:
                        raise MACMError("MACM completed with offset readback mismatch; no replay")
                    return after
                if command != FLUSH:
                    raise MACMError("MACM command changed unexpectedly; no replay")
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
