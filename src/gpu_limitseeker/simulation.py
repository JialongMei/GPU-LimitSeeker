"""Synthetic physics fixture, NOT hardware measurements or a live AI fallback."""
from __future__ import annotations

from .config import defaults
from .engine import SearchEngine
from .models import Advice, Bounds, Profile, TrialResult, integer


def run_demo(powers: list[int]) -> dict:
    if type(powers) is not list or not powers or len(powers) > 4 or powers[0] != 100 or powers != sorted(set(powers)):
        raise ValueError("Synthetic power points must be unique, increasing, begin at 100, and contain at most four points")
    for power in powers:
        integer(power, "synthetic power", 100)
        if power > 125:
            raise ValueError("Synthetic fixture supports at most 125 percent")
    cfg = defaults()
    cfg["search"].update(power_limits=powers, anchors_mv=[1000, 1050])
    cfg["safety"].update(max_power_percent=max(powers), max_board_power_w=375)
    # Pure engine injection: accelerated *simulated* durations are not live config.
    cfg["durations"] = {
        "baseline": {"full_seconds": 3, "half_seconds": 2, "vram_seconds": 3},
        "screen": {"full_seconds": 3, "half_seconds": 2, "vram_seconds": 3},
        "long": {"full_seconds": 12, "half_seconds": 6, "vram_seconds": 12},
        "fan": {"full_seconds": 4, "half_seconds": 2, "vram_seconds": 0},
    }
    initial = Profile("uniform", 60, 200)
    cfg["_device_curve"] = [
        {"index": index, "mv": mv, "mhz": 2700 + (mv - 1000) * 2 + initial.core,
         "raw_offset": initial.core * 1000}
        for index, mv in enumerate((1000, 1025, 1050))
    ]
    policy = SearchEngine(initial, Bounds(240, 700, 1000, 1050, 2000, 4000, 100, 125), cfg)
    simulated_seconds = 0
    trials = []
    for _ in range(2000):
        spec = policy.next_trial(43200 - simulated_seconds)
        if spec is None:
            break
        p = spec.profile
        extra = (p.power_percent - 100) // 5 * 15
        core_ceiling = 150 + extra - (15 if spec.quality == "long" else 0)
        mem_ceiling = 550 - (50 if spec.quality == "long" else 0)
        curve_ceiling = 2940 + extra + ((p.mv or 1000) - 1000) * 2 - (15 if spec.quality == "long" else 0)
        passed = p.core <= core_ceiling and p.mem <= mem_ceiling and (p.kind != "curve" or p.mhz <= curve_ceiling)
        outcome = "pass" if passed else "wrong"
        clock = p.mhz if p.kind == "curve" else 2700 + p.core
        score = 10000 + (clock - 2700) * 3 + p.mem * 0.2
        watts = 230 + (p.power_percent - 100) * 2
        workloads = tuple(
            {"kind": kind, "seconds": seconds, "completed": passed,
             "achieved_core_mhz": clock, "achieved_mem_mhz": 5000 + p.mem}
            for kind, seconds in (("full", spec.full_seconds), ("half", spec.half_seconds), ("vram", spec.vram_seconds)) if seconds
        )
        observed = TrialResult(spec.id, outcome, score=score, peak_temp_c=60 + (p.power_percent - 100) / 3,
                               workloads=workloads, full_avg_power_w=watts, peak_power_w=watts + 20)
        policy.observe(observed, Advice("continue", "Scripted offline fixture; not a real model decision"))
        simulated_seconds += spec.seconds
        trials.append({"id": spec.id, "phase": spec.phase, "quality": spec.quality,
                       "profile": p.to_dict(), "outcome": outcome})
    else:
        raise RuntimeError("Synthetic scenario exceeded its transition bound")
    return {
        "synthetic": True, "hardware_access": False, "network_access": False,
        "reviewer": "scripted fixture, NOT a live AI provider",
        "notice": "All clocks, watts, scores and durations below are invented fixture data; no GPU stability claim",
        "trial_count": len(trials), "simulated_seconds": simulated_seconds,
        "summary": policy.summary(), "trials": trials,
    }
