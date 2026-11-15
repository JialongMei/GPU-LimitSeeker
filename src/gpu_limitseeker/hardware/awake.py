"""Keep only the test thread's system/display requests active; no policy changes."""
import ctypes

CONTINUOUS = 0x80000000
REQUIRED = CONTINUOUS | 0x00000001 | 0x00000002


class AwakeLease:
    def __init__(self, set_state=None, *, display=True):
        if set_state is None:
            function = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
            function.argtypes = [ctypes.c_uint32]
            function.restype = ctypes.c_uint32
            set_state = function
        self._set_state = set_state
        self._closed = False
        self.required = CONTINUOUS | 0x00000001 | (0x00000002 if display else 0)
        self.previous = set_state(self.required)
        if not self.previous:
            raise OSError("Cannot acquire temporary system/display wake request; test not started")
        combined = self.previous | self.required
        if combined != self.required and not set_state(combined):
            set_state(self.previous)
            raise OSError("Cannot preserve existing thread wake requests; test not started")

    def close(self):
        if not self._closed:
            if not self._set_state(self.previous):
                raise OSError("Could not restore test thread's previous wake request state")
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False
