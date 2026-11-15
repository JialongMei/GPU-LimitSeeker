"""Lazy exact-vector NVAPI bridge; no hardware work at import.

ABI/layout adapted from MIT Rovey/gpu-auto-optimizer commit
638de2a8b20fd9a2e9f15eea74a7029a86cb59b0, src/hw/nvapi.cpp and
src/core/vf_curve.hpp. No inferred offsets, frequency reconstruction or retries.
Caller must bind the sole GPU identity and bound the entire native operation.
"""
from __future__ import annotations

import ctypes
import os
import struct
import time

from .macm_backend import BackendError


class RawCurve:
    MASK_SIZE, STATUS_SIZE, CONTROL_SIZE = 6188, 7208, 9248
    INITIALIZE, UNLOAD, ENUM = 0x0150E828, 0xD22BDD7E, 0xE5AC921F
    MASK, STATUS, GET, SET = 0x507B4B59, 0x21537AD4, 0x23F1B133, 0x0733E009

    def __init__(self):
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise BackendError("Raw V/F restoration requires 64-bit Windows")
        # LOAD_LIBRARY_SEARCH_SYSTEM32: never resolve a DLL from cwd/PATH.
        self.library = ctypes.CDLL("nvapi64.dll", winmode=0x00000800)
        self.query = self.library.nvapi_QueryInterface
        self.query.argtypes, self.query.restype = [ctypes.c_uint32], ctypes.c_void_p
        self._initialized = False
        try:
            self._call(self.INITIALIZE, ())
            self._initialized = True
            handles = (ctypes.c_void_p * 64)()
            count = ctypes.c_uint32()
            self._call(self.ENUM, (ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32)),
                       handles, ctypes.byref(count))
            if count.value != 1 or not handles[0]:
                raise BackendError("Exact raw V/F restoration requires exactly one physical NVIDIA GPU")
            self.handle = handles[0]
        except BaseException:
            self.close()
            raise

    def _call(self, identifier, signature, *args):
        address = self.query(identifier)
        if not address:
            raise BackendError(f"Raw V/F interface 0x{identifier:08x} is unavailable")
        function = ctypes.CFUNCTYPE(ctypes.c_int32, *signature)(address)
        code = function(*args)
        if code != 0:
            raise BackendError(f"Raw V/F interface 0x{identifier:08x} failed ({code}); no retry")

    def _buffer(self, size):
        result = ctypes.create_string_buffer(size)
        struct.pack_into("<I", result, 0, size | (1 << 16))
        return result

    def _vf_call(self, identifier, buffer):
        self._call(identifier, (ctypes.c_void_p, ctypes.c_void_p), self.handle, buffer)

    def _read(self):
        mask = self._buffer(self.MASK_SIZE)
        # Upstream's all-slots query, selected from the outset; no fallback retry.
        mask[4:36] = b"\xff" * 32
        self._vf_call(self.MASK, mask)
        status, control = self._buffer(self.STATUS_SIZE), self._buffer(self.CONTROL_SIZE)
        status[4:68], control[4:68] = mask[4:68], mask[4:68]
        self._vf_call(self.STATUS, status)
        self._vf_call(self.GET, control)
        points = []
        for index in range(255):
            m, s, c = 68 + index * 24, 68 + index * 28, 68 + index * 36
            if struct.unpack_from("<I", mask, m)[0] != 0 or mask.raw[m + 4] == 0:
                continue
            khz, uv = struct.unpack_from("<II", status, s + 4)
            if khz == 0 or uv == 0:
                continue
            points.append({"index": index, "mv": uv // 1000,
                           "raw_offset": struct.unpack_from("<i", control, c + 20)[0]})
        points.sort(key=lambda point: point["mv"])
        if not points:
            raise BackendError("Raw V/F API returned no enabled graphics slots")
        return points, control

    @staticmethod
    def _vector(points):
        if not isinstance(points, list) or not points:
            raise BackendError("Complete original raw V/F vector is required")
        vector = []
        for point in points:
            index, mv, raw = (point.get(key) for key in ("index", "mv", "raw_offset"))
            if (type(index) is not int or not 0 <= index < 255 or type(mv) is not int or mv <= 0 or
                    type(raw) is not int or not -(2 ** 31) <= raw < 2 ** 31):
                raise BackendError("Invalid original raw V/F slot")
            vector.append((index, mv, raw))
        if len({p[0] for p in vector}) != len(vector):
            raise BackendError("Duplicate original raw V/F slots")
        return vector

    def write_exact(self, expected, target):
        expected_vector, target_vector = self._vector(expected), self._vector(target)
        if [(p[0], p[1]) for p in expected_vector] != [(p[0], p[1]) for p in target_vector]:
            raise BackendError("Original raw V/F topology differs from current topology")
        current, control = self._read()
        if self._vector(current) != expected_vector:
            raise BackendError("Raw V/F API disagrees with independent current GAO readback; no write")
        for index, _, raw in target_vector:
            struct.pack_into("<i", control, 68 + index * 36 + 20, raw)
        struct.pack_into("<I", control, 0, self.CONTROL_SIZE | (1 << 16))
        self._vf_call(self.SET, control)
        time.sleep(0.1)  # Pinned upstream's settling interval; not a retry.
        after, _ = self._read()
        if self._vector(after) != target_vector:
            raise BackendError("Exact raw V/F setter readback disagrees; no replay")

    def close(self):
        if self._initialized:
            self._initialized = False
            self._call(self.UNLOAD, ())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
