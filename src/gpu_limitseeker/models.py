"""Versioned, non-executable contracts shared by policy, AI and hardware layers."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import math
from typing import Any


class SafetyError(RuntimeError):
    """A safety interlock or unreliable observation forbids further GPU work."""


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(value: Any, name: str, minimum: float = 0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


@dataclass(frozen=True)
class Profile:
    kind: str
    core: int
    mem: int
    mv: int | None = None
    mhz: int | None = None
    fan: str = "performance"
    power_percent: int = 100

    def __post_init__(self):
        if self.kind not in ("uniform", "curve"):
            raise ValueError("Unsupported profile kind")
        integer(self.core, "core")
        integer(self.mem, "mem")
        integer(self.power_percent, "power_percent", 1)
        if self.kind == "curve":
            integer(self.mv, "mv", 1)
            integer(self.mhz, "mhz", 1)
        elif self.mv is not None or self.mhz is not None:
            raise ValueError("Uniform profiles cannot carry curve anchors")
        if type(self.fan) is not str or not self.fan or len(self.fan) > 48 or not all(c.isalnum() or c in "_-" for c in self.fan):
            raise ValueError("Invalid fan profile identifier")

    @property
    def key(self) -> tuple:
        return self.kind, self.core, self.mem, self.mv, self.mhz, self.fan, self.power_percent

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> Profile:
        if type(value) is not dict or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown or non-object profile data")
        return cls(**value)


@dataclass(frozen=True)
class Bounds:
    core_max: int
    mem_max: int
    voltage_min: int = 900
    voltage_max: int = 1050
    frequency_min: int = 2000
    frequency_max: int = 4000
    power_min_percent: int = 1
    power_max_percent: int = 100

    def __post_init__(self):
        for item in fields(self):
            integer(getattr(self, item.name), item.name)
        if (self.voltage_min > self.voltage_max or self.frequency_min > self.frequency_max
                or self.power_min_percent < 1 or self.power_min_percent > self.power_max_percent):
            raise ValueError("Inverted or invalid device bounds")

    def check(self, profile: Profile) -> None:
        if profile.core > self.core_max or profile.mem > self.mem_max:
            raise SafetyError("Profile exceeds independently reported driver bounds")
        if profile.kind == "curve" and not (self.voltage_min <= profile.mv <= self.voltage_max and self.frequency_min <= profile.mhz <= self.frequency_max):
            raise SafetyError("Curve exceeds the approved voltage/frequency envelope")
        if not self.power_min_percent <= profile.power_percent <= self.power_max_percent:
            raise SafetyError("Power limit exceeds independently reported device bounds")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrialSpec:
    id: str
    phase: str
    quality: str
    profile: Profile
    full_seconds: int
    half_seconds: int
    vram_seconds: int
    bandwidth: bool = False

    def __post_init__(self):
        if not self.id or not self.phase or self.quality not in ("screen", "long"):
            raise ValueError("Invalid trial identity or quality")
        for name in ("full_seconds", "half_seconds", "vram_seconds"):
            value = integer(getattr(self, name), name)
            if value > 3600:
                raise ValueError("Each owned workload is bounded to one hour")
        if self.full_seconds == 0 or type(self.bandwidth) is not bool:
            raise ValueError("Every scored trial requires full-load work")

    @property
    def seconds(self) -> int:
        return self.full_seconds + self.half_seconds + self.vram_seconds

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> TrialSpec:
        data = dict(value)
        data["profile"] = Profile.from_dict(data["profile"])
        return cls(**data)


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    outcome: str
    score: float | None = None
    peak_temp_c: float | None = None
    bandwidth_gbps: float | None = None
    load_started: bool = True
    details: str = ""
    workloads: tuple[dict, ...] = ()
    full_avg_power_w: float | None = None
    peak_power_w: float | None = None

    def __post_init__(self):
        if not self.trial_id or self.outcome not in ("pass", "wrong", "tdr", "hot", "invalid", "stopped"):
            raise ValueError("Invalid trial result")
        if type(self.load_started) is not bool or type(self.details) is not str:
            raise ValueError("Invalid trial evidence metadata")
        for name in ("score", "peak_temp_c", "bandwidth_gbps", "full_avg_power_w", "peak_power_w"):
            if getattr(self, name) is not None:
                finite(getattr(self, name), name)
        if self.outcome == "pass" and (not self.load_started or self.score is None or self.score <= 0 or self.peak_temp_c is None or self.peak_temp_c <= 0):
            raise ValueError("Passing results require positive measured work and temperature")
        for name in ("full_avg_power_w", "peak_power_w"):
            if getattr(self, name) is not None and getattr(self, name) <= 0:
                raise ValueError("Measured board power must be positive")
        if self.full_avg_power_w is not None and self.peak_power_w is not None and self.peak_power_w + 1e-6 < self.full_avg_power_w:
            raise ValueError("Peak board power cannot be below the scored full-load average")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> TrialResult:
        data = dict(value)
        data["workloads"] = tuple(data.get("workloads", ()))
        return cls(**data)


@dataclass(frozen=True)
class Advice:
    action: str
    reason: str
    core_step: int | None = None
    mem_step: int | None = None

    def __post_init__(self):
        if self.action not in ("continue", "refine", "validate", "advance", "backoff", "stop"):
            raise ValueError("Unrecognized non-executable AI action")
        if type(self.reason) is not str or not self.reason.strip() or len(self.reason) > 1000:
            raise ValueError("AI reason must be a bounded nonempty string")
        for name in ("core_step", "mem_step"):
            if getattr(self, name) is not None:
                integer(getattr(self, name), name, 1)

    def to_dict(self) -> dict:
        return asdict(self)
