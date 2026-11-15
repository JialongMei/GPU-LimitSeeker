"""Verified NVML-only fan control; importing this module never loads a DLL.

The caller MUST close Afterburner's software fan controller before entering
FanLoop, and keep it closed until exit. This module does not manage processes.
NVML ABI matches GAO v0.4.9 src/hw/nvml.cpp: opaque device pointer, unsigned
fan indices/percentages, int return codes; automatic policy=0, manual=1.
Only fan settings are written. Driver automatic control, not cached settings,
is restored on exit. A hung native driver call cannot be cancelled by Python;
bounded joining reports that failure rather than claiming successful cleanup.
"""

import ctypes
import json
import math
import os
from pathlib import Path
import threading
import time


class FanError(RuntimeError):
    """Fan control failed; NVML failures retain their operation and return code."""

    def __init__(self, message, *, operation=None, code=None):
        super().__init__(message)
        self.operation = operation
        self.code = code


_LIBRARY = None
_LIBRARY_LOCK = threading.Lock()


def _system_library():
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            if os.name != "nt":
                raise FanError("NVML fan control requires Windows")
            # The loader resolves only System32, never the cwd or application dir.
            # NVML uses the C calling convention, including its int enum outputs.
            try:
                lib = ctypes.CDLL("nvml.dll", winmode=0x00000800)
            except OSError as exc:
                raise FanError(f"Could not load System32 NVML: {exc}") from exc
            u = ctypes.c_uint
            p = ctypes.POINTER(u)
            h = ctypes.c_void_p
            signatures = {
                "nvmlInit_v2": [],
                "nvmlShutdown": [],
                "nvmlDeviceGetHandleByUUID": [ctypes.c_char_p, ctypes.POINTER(h)],
                "nvmlDeviceGetNumFans": [h, p],
                "nvmlDeviceGetMinMaxFanSpeed": [h, p, p],
                "nvmlDeviceGetTemperature": [h, ctypes.c_int, p],
                "nvmlDeviceSetFanSpeed_v2": [h, u, u],
                "nvmlDeviceSetDefaultFanSpeed_v2": [h, u],
                "nvmlDeviceGetTargetFanSpeed": [h, u, p],
                "nvmlDeviceGetFanControlPolicy_v2": [h, u, p],
                "nvmlDeviceGetFanSpeed_v2": [h, u, p],
            }
            for name, args in signatures.items():
                try:
                    function = getattr(lib, name)
                except AttributeError as exc:
                    raise FanError(f"Required NVML API unavailable: {name}", operation=name) from exc
                function.argtypes = args
                function.restype = ctypes.c_int
            _LIBRARY = lib
        return _LIBRARY


class NvidiaFans:
    """One NVML init reference, targeting only the supplied GPU UUID.

    minimum/maximum are the driver's supported percentage bounds; count is the
    actual fan count. _library is an isolated fake-call seam for unit tests.
    NVML documents reference-counted init/shutdown: close releases only this
    object's successful init, not another caller's reference. close does not
    restore settings; FanLoop owns restoration and must exit before close.
    """

    def __init__(self, uuid: str, *, _library=None):
        if not isinstance(uuid, str) or not uuid.startswith("GPU-") or "\x00" in uuid:
            raise FanError("An explicit GPU UUID is required")
        try:
            encoded = uuid.encode("ascii")
        except UnicodeEncodeError as exc:
            raise FanError("GPU UUID must be ASCII") from exc
        self.uuid = uuid
        self._closed = True
        self._lib = _system_library() if _library is None else _library
        self._lock = threading.RLock()
        self._call("nvmlInit_v2")
        self._closed = False
        self._handle = ctypes.c_void_p()
        try:
            self._call("nvmlDeviceGetHandleByUUID", encoded, ctypes.byref(self._handle))
            if not self._handle.value:
                raise FanError("NVML returned a null GPU handle")
            count = ctypes.c_uint()
            self._call("nvmlDeviceGetNumFans", self._handle, ctypes.byref(count))
            if not 1 <= count.value <= 8:
                raise FanError(f"Unsupported fan count: {count.value}; expected 1..8")
            self.count = count.value
            minimum, maximum = ctypes.c_uint(), ctypes.c_uint()
            self._call("nvmlDeviceGetMinMaxFanSpeed", self._handle,
                       ctypes.byref(minimum), ctypes.byref(maximum))
            self.minimum, self.maximum = minimum.value, maximum.value
            _bounds(self.minimum, self.maximum)
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                exc.add_note(f"NVML init cleanup failed: {cleanup}")
            raise

    def _call(self, name, *args):
        try:
            function = getattr(self._lib, name)
        except AttributeError as exc:
            raise FanError(f"Required NVML API unavailable: {name}", operation=name) from exc
        try:
            code = function(*args)
        except Exception as exc:
            raise FanError(f"{name} call failed: {exc}", operation=name) from exc
        if code != 0:
            raise FanError(f"{name} returned NVML error {code}", operation=name, code=code)

    def _open(self):
        if self._closed:
            raise FanError("NVML fan handle is closed")

    def _get(self, name, *args):
        value = ctypes.c_uint()
        self._call(name, self._handle, *args, ctypes.byref(value))
        return value.value

    def read(self) -> dict:
        with self._lock:
            self._open()
            # Driver-backed fan calls precede the temperature shared-memory read.
            reading = {
                "targets": [self._get("nvmlDeviceGetTargetFanSpeed", i) for i in range(self.count)],
                "policies": [self._get("nvmlDeviceGetFanControlPolicy_v2", i) for i in range(self.count)],
                "speeds": [self._get("nvmlDeviceGetFanSpeed_v2", i) for i in range(self.count)],
                "temp_c": float(self._get("nvmlDeviceGetTemperature", 0)),
            }
            _telemetry(reading, self.count)
            return reading

    @staticmethod
    def _failures(errors):
        if errors:
            error = FanError("; ".join(str(e) for e in errors),
                             operation=errors[0].operation, code=errors[0].code)
            raise error from errors[0]

    def set_percent(self, pct: int) -> None:
        if type(pct) is not int:
            raise FanError("Fan percentage must be an integer")
        with self._lock:
            self._open()
            target = max(self.minimum, min(self.maximum, pct))
            errors = []
            for i in range(self.count):
                try:
                    self._call("nvmlDeviceSetFanSpeed_v2", self._handle, i, target)
                except FanError as exc:
                    errors.append(exc)
            # Independent reads of every target AND policy, even after a failure.
            for i in range(self.count):
                for name, expected in (("nvmlDeviceGetTargetFanSpeed", target),
                                       ("nvmlDeviceGetFanControlPolicy_v2", 1)):
                    try:
                        actual = self._get(name, i)
                        if actual != expected:
                            raise FanError(f"{name} fan {i} read back {actual}, expected {expected}",
                                           operation=name)
                    except FanError as exc:
                        errors.append(exc)
            self._failures(errors)

    def restore_auto(self) -> None:
        with self._lock:
            self._open()
            errors = []
            for i in range(self.count):
                try:
                    self._call("nvmlDeviceSetDefaultFanSpeed_v2", self._handle, i)
                except FanError as exc:
                    errors.append(exc)
            for i in range(self.count):
                try:
                    actual = self._get("nvmlDeviceGetFanControlPolicy_v2", i)
                    if actual != 0:
                        raise FanError(f"Fan {i} remains in policy {actual}; expected automatic (0)",
                                       operation="nvmlDeviceGetFanControlPolicy_v2")
                except FanError as exc:
                    errors.append(exc)
            self._failures(errors)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                # Never double-decrement the process-wide NVML reference count.
                self._closed = True
                self._call("nvmlShutdown")


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FanError(f"{label} must be a finite number")
    return value


def _bounds(minimum, maximum):
    if type(minimum) is not int or type(maximum) is not int or not 0 <= minimum <= maximum <= 100:
        raise FanError("Driver fan bounds must be integer percentages 0 <= min <= max <= 100")


def _points(points):
    try:
        result = tuple(tuple(point) for point in points)
    except (TypeError, ValueError) as exc:
        raise FanError("Curve must contain temperature/percentage pairs") from exc
    if len(result) < 2:
        raise FanError("Curve needs at least two points")
    previous = None
    for point in result:
        if len(point) != 2:
            raise FanError("Curve must contain temperature/percentage pairs")
        temp, duty = point
        _number(temp, "Curve temperature")
        _number(duty, "Curve duty")
        if not 0 <= duty <= 100:
            raise FanError("Curve duty must be between 0 and 100")
        if previous and (temp <= previous[0] or duty < previous[1]):
            raise FanError("Curve temperatures must increase strictly and duty must not decrease")
        previous = point
    return result


def _interpolate(points, temp, minimum, maximum):
    duty = points[-1][1]
    if temp <= points[0][0]:
        duty = points[0][1]
    else:
        for (left_t, left_p), (right_t, right_p) in zip(points, points[1:]):
            if temp <= right_t:
                duty = left_p + (right_p - left_p) * ((temp - left_t) / (right_t - left_t))
                break
    # Round toward cooling, then clamp to the actual driver's supported range.
    return max(minimum, min(maximum, math.ceil(duty)))


def curve_percent(points, temp, minimum, maximum) -> int:
    """Monotone piecewise-linear curve with endpoint hold and upward rounding."""
    _number(temp, "Temperature")
    _bounds(minimum, maximum)
    return _interpolate(_points(points), temp, minimum, maximum)


class _Hysteresis:
    def __init__(self, points, minimum, maximum):
        self.points = _points(points)
        _bounds(minimum, maximum)
        self.minimum, self.maximum = minimum, maximum
        self.percent = None
        self.peak_temp = None

    def choose(self, temp):
        _number(temp, "Temperature")
        target = _interpolate(self.points, temp, self.minimum, self.maximum)
        if (self.percent is None or target > self.percent or
                (temp <= self.peak_temp - 2 and target <= self.percent - 3)):
            self.percent = target
            self.peak_temp = temp
        else:
            self.peak_temp = max(self.peak_temp, temp)
        return self.percent


def _telemetry(reading, count):
    if not isinstance(reading, dict) or "temp_c" not in reading:
        raise FanError("Missing temperature telemetry")
    temp = _number(reading["temp_c"], "Temperature telemetry")
    if temp < 0:
        raise FanError("Invalid negative temperature telemetry")
    for name in ("targets", "policies", "speeds"):
        values = reading.get(name)
        if not isinstance(values, list) or len(values) != count:
            raise FanError(f"Missing or incomplete {name} telemetry")
        if any(type(v) is not int or v < 0 for v in values):
            raise FanError(f"Invalid {name} telemetry")
        if name == "targets" and any(v > 100 for v in values):
            raise FanError("Invalid fan target telemetry")
        if name == "policies" and any(v not in (0, 1) for v in values):
            raise FanError("Unknown fan policy telemetry")
    return temp


class FanLoop:
    """One explicitly scoped fan worker; caller MUST close Afterburner first.

    Enter waits for the first verified fan command before permitting a test.
    check() must be called by the test supervisor as well as checking its shared
    stop event. Errors and >=max_temp latch that event and touch stop_file;
    operator stop markers are never removed. Normal exit stops only this worker.
    Exiting always attempts driver-auto restoration on every fan; NvidiaFans
    remains open for the caller to close. FanLoop instances are single-use.
    """

    def __init__(self, fans, points, stop_event, stop_file, log_path,
                 max_temp=80, interval=2, ownership=None):
        self.fans = fans
        self.ownership = ownership
        self._policy = _Hysteresis(points, fans.minimum, fans.maximum)
        _number(max_temp, "Maximum temperature")
        _number(interval, "Interval")
        if not 0 < max_temp <= 80 or interval <= 0:
            raise FanError("Require 0 < max_temp <= 80 and interval > 0")
        self.max_temp, self.interval = max_temp, interval
        self.stop_event = stop_event
        self.stop_file = Path(stop_file)
        self.log_path = Path(log_path)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._error = None
        self.thermal_event = threading.Event()
        self._relinquished = threading.Event()
        self._error_lock = threading.Lock()
        self._applied = None
        self._join_timeout = 5.0

    def _log(self, event, **fields):
        record = {"time": time.time(), "event": event, **fields}
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _latch(self, error):
        if not isinstance(error, FanError):
            error = FanError(str(error))
        with self._error_lock:
            if self._error is None:
                self._error = error
            elif self._error is not error:
                self._error.add_note(str(error))
        # Latch first: even a disk/logging failure must stop the supervisor.
        self.stop_event.set()
        self._stop.set()
        try:
            self.stop_file.touch(exist_ok=True)
        except Exception as exc:
            error.add_note(f"Could not create stop marker: {exc}")
        try:
            self._log("error", error=str(error), operation=error.operation, code=error.code)
        except Exception as exc:
            error.add_note(f"Could not log fan error: {exc}")

    def _external_stop(self):
        if self.stop_file.exists():
            self.stop_event.set()
        return self._stop.is_set() or self.stop_event.is_set()

    def _owns(self):
        if self._relinquished.is_set():
            return False
        if self.ownership is not None:
            try:
                confirmed = self.ownership() is True
            except Exception:
                confirmed = False
            if not confirmed:
                self.relinquish()
                return False
        return True

    def _run(self):
        try:
            while not self._external_stop():
                if not self._owns():
                    raise FanError("Fan ownership lost; external controller is not overwritten")
                reading = self.fans.read()
                temp = _telemetry(reading, self.fans.count)
                if temp >= self.max_temp:
                    self.thermal_event.set()
                if self._applied is not None and (
                        reading["targets"] != [self._applied] * self.fans.count or
                        reading["policies"] != [1] * self.fans.count):
                    self.relinquish()
                    raise FanError("Fan target/policy drifted; external owner is not overwritten")
                if temp >= self.max_temp:
                    # Exactly one emergency attempt; never retry a failing write.
                    self._latch(FanError(f"Emergency temperature {temp} C >= {self.max_temp} C"))
                    if not self._owns():
                        break
                    self.fans.set_percent(100)
                    self._log("emergency", requested_percent=100, **reading)
                    break
                if self._external_stop():
                    break
                target = self._policy.choose(temp)
                if not self._owns():
                    raise FanError("Fan ownership lost before command")
                if target != self._applied:
                    self.fans.set_percent(target)
                    self._applied = target  # Only verified readback is an applied change.
                    self._log("change", applied_percent=target, **reading)
                self._log("telemetry", applied_percent=self._applied, **reading)
                self._ready.set()
                if self._stop.wait(self.interval):
                    break
        except Exception as exc:
            self._latch(exc)
        finally:
            try:
                if not self._owns():
                    self._log("relinquished_external_controller")
                else:
                    self.fans.restore_auto()
                    self._log("restored_auto")
            except Exception as exc:
                self._latch(exc)
            finally:
                self._ready.set()

    def relinquish(self):
        """Stop our worker without overwriting a newly active external controller."""
        self._relinquished.set()
        self._stop.set()

    def check(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise error

    def __enter__(self):
        if self._thread is not None:
            raise FanError("FanLoop instances are single-use")
        if self._external_stop():
            raise FanError("Fan service cannot start while the operator stop is latched")
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log("starting", uuid=getattr(self.fans, "uuid", None))
            self._thread = threading.Thread(target=self._run, name="nvml-fans", daemon=True)
            self._thread.start()
            if not self._ready.wait(self._join_timeout):
                self._latch(FanError("Fan worker did not complete initial telemetry within 5 seconds"))
            self.check()
            if self._external_stop() or self._applied is None:
                raise FanError("Fan service stopped before admission")
            return self
        except BaseException as exc:
            if isinstance(exc, Exception) and self._error is None:
                self._latch(exc)
            self._stop.set()
            if self._thread is not None and self._thread.ident is not None:
                self._thread.join(self._join_timeout)
            else:
                try:
                    self.fans.restore_auto()
                except Exception as cleanup:
                    self._latch(cleanup)
            raise

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(self._join_timeout)
        if self._thread.is_alive():
            self._latch(FanError("Fan worker did not stop within 5 seconds; driver-auto restoration is unconfirmed"))
        try:
            self.check()
        except FanError as cleanup:
            if exc is None:
                raise
            exc.add_note(f"Fan service failure during context exit: {cleanup}")
        return False
