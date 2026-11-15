"""Windows-owned job: controller death kills its children, including GPU workloads."""
import ctypes
from ctypes import wintypes
import os
import threading


class _BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", _BasicLimits), ("io", _IoCounters),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class OwnedProcessTree:
    """Assign ONLY this process and future descendants, never existing apps.

    Keep the non-inherited job handle until process exit. Explicitly closing it
    while still assigned would kill this controller too. A native wall watchdog
    also covers a Python/NVML shutdown hang; it never reboots the computer.
    """
    def __init__(self, seconds):
        if os.name != "nt" or type(seconds) is not int or not 1 <= seconds <= 72 * 3600 + 300:
            raise ValueError("Windows process scope requires a lifetime within 72 hours plus bounded cleanup")
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        api.CreateJobObjectW.restype = wintypes.HANDLE
        api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        api.SetInformationJobObject.restype = wintypes.BOOL
        api.GetCurrentProcess.argtypes = []
        api.GetCurrentProcess.restype = wintypes.HANDLE
        api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        api.AssignProcessToJobObject.restype = wintypes.BOOL
        api.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        api.TerminateProcess.restype = wintypes.BOOL
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        api.CloseHandle.restype = wintypes.BOOL
        handle = api.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            api.CloseHandle(handle)
            raise ctypes.WinError(error)
        if not api.AssignProcessToJobObject(handle, api.GetCurrentProcess()):
            error = ctypes.get_last_error()
            api.CloseHandle(handle)
            raise ctypes.WinError(error)
        self.api, self.handle = api, handle
        self.watchdog = threading.Timer(seconds, self._terminate)
        self.watchdog.daemon = True
        self.watchdog.start()

    def _terminate(self):
        # TerminateProcess avoids deadlock in DLL detach of a failed driver.
        self.api.TerminateProcess(self.api.GetCurrentProcess(), 238)

    def close(self):
        """Disarm only the wall watchdog; keep death containment until process exit."""
        self.watchdog.cancel()
