"""Strict operator-owned configuration. Importing this module has no native effects."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import tomllib
from urllib.parse import urlsplit

from .models import Profile, finite, integer

DEFAULTS = {
    "hardware": {
        "gao_path": None, "memtest_path": None, "nvidia_icd": None,
        "allow_afterburner_handoff": False, "baseline_profile": None,
        "memory_reference_mhz": None,
    },
    "safety": {
        "max_temp_c": 80, "power_percent": 100, "max_power_percent": 100,
        "max_board_power_w": None, "max_recoveries": 2, "max_hours": 24,
    },
    "search": {
        "core_step": 30, "core_resolution": 15, "mem_step": 100,
        "mem_resolution": 50, "curve_step": 30, "curve_resolution": 15,
        "anchors_mv": [1000, 1025, 1050], "power_limits": [100],
        "objective": "clock_limit", "max_long_attempts": 8,
    },
    "durations": {
        "baseline": {"full_seconds": 120, "half_seconds": 60, "vram_seconds": 300},
        "screen": {"full_seconds": 120, "half_seconds": 60, "vram_seconds": 120},
        "long": {"full_seconds": 3600, "half_seconds": 600, "vram_seconds": 3600},
        "fan": {"full_seconds": 300, "half_seconds": 60, "vram_seconds": 0},
    },
    "fan_profiles": {
        "performance": [[30, 35], [45, 50], [60, 75], [70, 90], [78, 100]],
        "cool": [[30, 30], [45, 45], [60, 70], [70, 85], [78, 100]],
        "balanced": [[30, 30], [45, 40], [60, 60], [70, 80], [78, 100]],
    },
    "ai": {
        "provider": None, "model": None, "base_url": None, "api_key_env": None,
        "command": None, "timeout_seconds": 120, "max_attempts": 2,
    },
}


def defaults() -> dict:
    return deepcopy(DEFAULTS)


def _merge(base: dict, changes: dict, name: str = "config") -> dict:
    if type(changes) is not dict:
        raise ValueError(f"{name} must be a table")
    unknown = set(changes) - set(base)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")
    for key, value in changes.items():
        if key == "fan_profiles":
            base[key] = deepcopy(value)
        elif type(base[key]) is dict:
            base[key] = _merge(base[key], value, f"{name}.{key}")
        else:
            base[key] = deepcopy(value)
    return base


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def json_loads(text: str):
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=nonfinite)


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve(strict=True)
    text = path.read_text(encoding="utf-8-sig")
    raw = json_loads(text) if path.suffix.lower() == ".json" else tomllib.loads(text)
    config = validate_config(raw)
    for key in ("gao_path", "memtest_path", "nvidia_icd"):
        if config["hardware"][key]:
            candidate = Path(config["hardware"][key]).expanduser()
            config["hardware"][key] = str((path.parent / candidate).resolve())
    return config


def validate_config(raw: dict) -> dict:
    cfg = _merge(defaults(), raw)
    hardware, safety, search, ai = (cfg[name] for name in ("hardware", "safety", "search", "ai"))
    for name in ("gao_path", "memtest_path", "nvidia_icd"):
        value = hardware[name]
        if value is not None and (type(value) is not str or not value.strip() or "\x00" in value):
            raise ValueError(f"Invalid hardware.{name}")
    if type(hardware["allow_afterburner_handoff"]) is not bool:
        raise ValueError("Afterburner handoff must be explicitly boolean")
    if hardware["memory_reference_mhz"] is not None:
        integer(hardware["memory_reference_mhz"], "memory_reference_mhz", 1)
    if hardware["baseline_profile"] is not None:
        baseline = Profile.from_dict(hardware["baseline_profile"])
        hardware["baseline_profile"] = baseline.to_dict()
    else:
        baseline = None
    temperature = finite(safety["max_temp_c"], "max_temp_c", 30)
    if temperature > 80:
        raise ValueError("max_temp_c may not exceed the backend's supported 80 C ceiling")
    integer(safety["power_percent"], "power_percent", 1)
    integer(safety["max_power_percent"], "max_power_percent", 1)
    if safety["power_percent"] > safety["max_power_percent"]:
        raise ValueError("Original power exceeds the explicitly approved ceiling")
    if safety["max_board_power_w"] is not None:
        finite(safety["max_board_power_w"], "max_board_power_w", 1)
    integer(safety["max_recoveries"], "max_recoveries")
    if safety["max_recoveries"] > 3:
        raise ValueError("At most three cumulative TDR recoveries are permitted")
    hours = finite(safety["max_hours"], "max_hours", 0.1)
    if hours > 72:
        raise ValueError("max_hours may not exceed the explicit three-day ceiling")
    for axis in ("core", "mem", "curve"):
        step = integer(search[f"{axis}_step"], f"{axis}_step", 1)
        resolution = integer(search[f"{axis}_resolution"], f"{axis}_resolution", 1)
        if resolution > step or step % resolution:
            raise ValueError(f"{axis} step must be a multiple of its resolution")
    for key in ("anchors_mv", "power_limits"):
        values = search[key]
        if type(values) is not list or not values or len(values) > 16:
            raise ValueError(f"{key} must contain 1..16 explicit points")
        for value in values:
            integer(value, key, 1)
        if values != sorted(set(values)):
            raise ValueError(f"{key} must be unique and increasing")
    if not all(900 <= value <= 1050 for value in search["anchors_mv"]):
        raise ValueError("Curve anchors must stay within the supported 900..1050 mV search envelope")
    if search["power_limits"][0] != safety["power_percent"]:
        raise ValueError("The first power point must equal the original baseline power")
    if max(search["power_limits"]) > safety["max_power_percent"]:
        raise ValueError("Requested power points exceed the operator's approved ceiling")
    if safety["max_power_percent"] > 100 and safety["max_board_power_w"] is None:
        raise ValueError("Above-default power requires an explicit max_board_power_w")
    if baseline and baseline.power_percent != safety["power_percent"]:
        raise ValueError("Explicit baseline power disagrees with original power")
    if search["objective"] not in ("clock_limit", "performance"):
        raise ValueError("Unsupported search objective")
    integer(search["max_long_attempts"], "max_long_attempts", 1)
    if search["max_long_attempts"] > 32:
        raise ValueError("Search attempt budgets are too large")
    for phase, durations in cfg["durations"].items():
        for name, value in durations.items():
            integer(value, f"{phase}.{name}")
            if value > 3600:
                raise ValueError("An individual workload may not exceed one hour")
        if durations["full_seconds"] == 0:
            raise ValueError("Every trial requires a scored full workload")
    for name, minimum in (("full_seconds", 300), ("half_seconds", 60), ("vram_seconds", 300)):
        if cfg["durations"]["long"][name] < minimum:
            raise ValueError(f"Live long.{name} must be >= {minimum}")
    profiles = cfg["fan_profiles"]
    if type(profiles) is not dict or not profiles or len(profiles) > 8 or "performance" not in profiles:
        raise ValueError("Provide 1..8 fan profiles including 'performance'")
    for name, points in profiles.items():
        Profile("uniform", 0, 0, fan=name)
        if type(points) is not list or not 2 <= len(points) <= 16:
            raise ValueError("Fan curves need 2..16 points")
        previous_temp, previous_speed = -1.0, -1.0
        for point in points:
            if type(point) is not list or len(point) != 2:
                raise ValueError("Fan points are [temperature C, speed percent]")
            temp, speed = finite(point[0], "fan temperature"), finite(point[1], "fan speed")
            if temp > 95 or speed > 100 or temp <= previous_temp or speed < previous_speed:
                raise ValueError("Fan curves must increase temperature and never reduce cooling")
            previous_temp, previous_speed = temp, speed
        if previous_temp > safety["max_temp_c"] or previous_speed != 100:
            raise ValueError("Fan curve must reach 100% before the thermal stop")
    if baseline and baseline.fan not in profiles:
        raise ValueError("Baseline references an unknown fan profile")
    if ai["provider"] not in (None, "openai", "omp"):
        raise ValueError("AI provider must be openai or omp")
    for name in ("model", "base_url", "api_key_env", "command"):
        if ai[name] is not None and (type(ai[name]) is not str or not ai[name].strip() or "\x00" in ai[name]):
            raise ValueError(f"Invalid ai.{name}")
    if ai["base_url"]:
        url = urlsplit(ai["base_url"])
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("API URL may not contain credentials, query or fragment")
        if url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")):
            raise ValueError("AI endpoint must use HTTPS, except an explicit loopback endpoint")
    integer(ai["timeout_seconds"], "AI timeout", 1)
    integer(ai["max_attempts"], "AI attempts", 1)
    if ai["timeout_seconds"] > 120 or ai["max_attempts"] > 2:
        raise ValueError("AI call budget exceeds the supported bound")
    return cfg
