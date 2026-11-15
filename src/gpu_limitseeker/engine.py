"""Deterministic, hardware-free staged search with auditable qualification evidence.

A requested setting is not a measured clock or a physical maximum. All failure
exclusions include the fan, and a later failure revokes earlier qualifications.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

from .models import Advice, Bounds, Profile, SafetyError, TrialResult, TrialSpec


_DEFAULT_DURATIONS = {
    "baseline": (120, 60, 300), "screen": (120, 60, 120),
    "long": (3600, 600, 3600), "fan": (300, 60, 0),
}


class SearchEngine:
    """One pending operation at a time; construction and restore perform no I/O."""

    VERSION = 1

    def __init__(self, initial: Profile, bounds: Bounds, config: dict):
        bounds.check(initial)
        self.initial, self.bounds, self.config = initial, bounds, deepcopy(config)
        search = config.get("search", {})
        self.settings = {}
        for key, default in (("core_step", 30), ("core_resolution", 15),
                             ("mem_step", 100), ("mem_resolution", 50),
                             ("curve_step", 30), ("curve_resolution", 15),
                             ("max_long_attempts", 8)):
            value = search.get(key, default)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
            self.settings[key] = value
        self.objective = search.get("objective", "clock_limit")
        if self.objective not in ("clock_limit", "performance"):
            raise ValueError("Unknown search objective")
        safety = config.get("safety", {})
        self.power_points = list(dict.fromkeys(search.get("power_limits", [100])))
        if not self.power_points or any(type(p) is not int or p <= 0 for p in self.power_points):
            raise ValueError("power_limits must contain positive integer percentages")
        if self.power_points[0] != initial.power_percent or safety.get("power_percent", initial.power_percent) != initial.power_percent:
            raise SafetyError("Power sweep must start with the exact inspected original power point")
        approved = safety.get("max_power_percent", 100)
        self.board_cap = safety.get("max_board_power_w")
        if type(approved) is not int or approved <= 0:
            raise ValueError("Invalid approved power envelope")
        if self.board_cap is not None and (type(self.board_cap) not in (int, float)
                or not math.isfinite(self.board_cap) or self.board_cap <= 0):
            raise ValueError("Board power cap must be a positive measured-watt ceiling")
        for power in [initial.power_percent, *self.power_points]:
            if power > approved or (power > 100 and self.board_cap is None):
                raise SafetyError("Power point requires an explicit percentage approval and board-watt cap")
            bounds.check(replace(initial, power_percent=power))
        self.durations = {}
        for name, defaults in _DEFAULT_DURATIONS.items():
            raw = config.get("durations", {}).get(name, {})
            values = tuple(raw.get(key, default) for key, default in zip(
                ("full_seconds", "half_seconds", "vram_seconds"), defaults))
            # TrialSpec is the single duration validation convention.
            TrialSpec("duration-check", name, "long" if name == "long" else "screen", initial, *values)
            self.durations[name] = values
        hours = config.get("safety", {}).get("max_hours", 12)
        if type(hours) not in (int, float) or not math.isfinite(hours) or hours <= 0:
            raise ValueError("max_hours must be positive and finite")
        self.state = {
            "version": self.VERSION, "phase": "baseline", "status": "running",
            "reason": "Exact inspected baseline awaits measured evidence",
            "pending": None, "sequence": 0, "spent_seconds": 0,
            "remaining_cap": hours * 3600, "budget_seconds": hours * 3600,
            "trials": [], "audit": [], "excluded": [], "axes": [], "axis": None,
            "base": initial.to_dict(), "anchors": [], "anchor_index": 0,
            "fan_candidates": [], "fan_index": 0, "fan_results": [],
            "selected": None, "final_attempts": 0, "recoveries": 0,
            "final_qualified": False, "final_research": False,
            "power_index": 0, "power_results": [
                {"power_percent": power, "status": "untested", "reason": None, "profile": None}
                for power in self.power_points],
        }
        self._event("created", initial=initial.to_dict())

    def _event(self, event, **data):
        self.state["audit"].append({"version": self.VERSION,
                                    "index": len(self.state["audit"]), "event": event, **deepcopy(data)})

    def _stop(self, reason):
        self.state.update(status="stopped", reason=reason)
        axis = self.state["axis"]
        if axis and axis["mode"] != "done":
            axis["limited_by"] = reason
        index = self.state["power_index"]
        if index < len(self.state["power_results"]) and self.state["power_results"][index]["status"] == "running":
            self.state["power_results"][index].update(status="stopped", reason=reason)
        self._event("stopped", reason=reason)

    def _excluded(self, profile):
        return any(Profile.from_dict(item["profile"]).key == profile.key for item in self.state["excluded"])

    def _exclude(self, profile, reason):
        if not self._excluded(profile):
            self.state["excluded"].append({"profile": profile.to_dict(), "reason": reason})
        self._event("qualification_revoked", profile=profile.to_dict(), reason=reason)

    def _records(self, quality=None):
        return [r for r in self.state["trials"] if r["accepted"]
                and (quality is None or r["spec"]["quality"] == quality)
                and not self._excluded(Profile.from_dict(r["spec"]["profile"]))]

    def _known(self, profile, quality=None):
        return any(Profile.from_dict(r["spec"]["profile"]).key == profile.key
                   for r in self._records(quality))

    @staticmethod
    def _measured(result):
        evidence = result["workloads"]
        measured = {}
        for key in ("achieved_core_mhz", "achieved_mem_mhz"):
            values = [w[key] for w in evidence if w.get("kind") == "full"
                      and type(w.get(key)) in (int, float) and math.isfinite(w[key]) and w[key] > 0]
            measured[key] = min(values) if values else None
        return measured

    def _rank(self, record):
        result = record["result"]
        if self.objective == "performance":
            return (result["score"],)
        measured = self._measured(result)
        # Missing telemetry never becomes a requested-clock performance claim.
        return (measured["achieved_core_mhz"] or 0,
                measured["achieved_mem_mhz"] or 0, result["score"])

    @property
    def best_profile(self):
        selected = self.state["selected"]
        if self.state["final_qualified"] and selected is not None:
            profile = Profile.from_dict(selected)
            if self._known(profile, "long"):
                return profile
        records = self._records("long")
        return Profile.from_dict(max(records, key=self._rank)["spec"]["profile"]) if records else None
    def _current_best(self):
        power = self.power_points[self.state["power_index"]]
        records = [r for r in self._records("long") if r["spec"]["profile"]["power_percent"] == power]
        return Profile.from_dict(max(records, key=self._rank)["spec"]["profile"]) if records else None

    def _begin_power(self):
        index = self.state["power_index"]
        if index >= len(self.power_points):
            finals = [item for item in self.state["power_results"] if item["status"] == "complete"
                      and not self._excluded(Profile.from_dict(item["profile"]))]
            records = [r for r in self._records("long") if any(
                r["spec"]["profile"] == item["profile"] for item in finals)]
            if not records:
                self._stop("no_final_long_qualified_power_point")
                return
            winner = max(records, key=self._rank)
            self.state.update(status="complete", reason="Approved power sweep completed with full final qualification",
                              selected=deepcopy(winner["spec"]["profile"]), final_qualified=True, axis=None)
            self._event("completed", profile=winner["spec"]["profile"])
            return
        profile = replace(self.initial, power_percent=self.power_points[index])
        self.state["power_results"][index]["status"] = "running"
        self.state.update(base=profile.to_dict(), selected=None, final_qualified=False, final_attempts=0,
                          final_research=False, axis=None, phase="power")
        if self._known(profile):
            self._start_axis("core", profile)

    def _finish_power(self, profile=None, reason=None):
        point = self.state["power_results"][self.state["power_index"]]
        point.update(status="complete" if profile is not None else "unqualified",
                     reason=reason, profile=profile.to_dict() if profile else None)
        self._event("power_finished", **point)
        self.state["power_index"] += 1
        self._begin_power()


    def _start_axis(self, name, base, start=None):
        prefix = "mem" if name == "memory" else "curve" if name == "curve" else "core"
        value = base.mem if name == "memory" else base.mhz if name == "curve" else base.core
        maximum = self.bounds.mem_max if name == "memory" else self.bounds.frequency_max if name == "curve" else self.bounds.core_max
        minimum = self.bounds.frequency_min if name == "curve" else 0
        axis = {"name": name, "base": base.to_dict(), "floor": minimum,
                "maximum": maximum, "resolution": self.settings[prefix + "_resolution"],
                "step": self.settings[prefix + "_step"], "low": value if self._known(base) else None,
                "high": None, "candidate": value if start is None else start,
                "mode": "coarse" if self._known(base) else "first",
                "driver_limit": False, "physical_failure": None, "limited_by": None,
                "long_attempts": 0, "qualified": None}
        self.state.update(phase=name, axis=axis)
        self.state["axes"].append(axis)

    def _profile(self, axis, value):
        base = Profile.from_dict(axis["base"])
        return replace(base, **{ {"core": "core", "memory": "mem", "curve": "mhz"}[axis["name"]]: value})

    def _axis_value(self, profile, axis):
        return getattr(profile, {"core": "core", "memory": "mem", "curve": "mhz"}[axis["name"]])
    @staticmethod
    def _matches_axis(profile, axis):
        field = {"core": "core", "memory": "mem", "curve": "mhz"}[axis["name"]]
        return all(profile[key] == value for key, value in axis["base"].items() if key != field)


    def _lower_value(self, axis, value):
        adjacent = max(axis["floor"], value - axis["resolution"]) if value > axis["floor"] else value - axis["resolution"]
        known = [self._axis_value(Profile.from_dict(r["spec"]["profile"]), axis)
                 for r in self._records() if self._matches_axis(r["spec"]["profile"], axis)
                 and self._axis_value(Profile.from_dict(r["spec"]["profile"]), axis) < value]
        # A non-grid driver endpoint must not make us skip an already passing
        # point between that endpoint and the next resolution decrement.
        return max([adjacent, *known])

    def _axis_next(self):
        axis = self.state["axis"]
        while axis["mode"] != "done":
            if axis["long_attempts"] >= self.settings["max_long_attempts"]:
                axis.update(mode="done", limited_by="long_attempt_limit")
                continue
            mode, low, high = axis["mode"], axis["low"], axis["high"]
            if mode == "coarse":
                if low >= axis["maximum"]:
                    axis.update(driver_limit=True, mode="long")
                    continue
                value = min(low + axis["step"], axis["maximum"])
            elif mode == "refine":
                if low is None:
                    axis.update(mode="fallback", candidate=self._lower_value(axis, high))
                    continue
                gap = high - low
                if gap <= axis["resolution"]:
                    axis["mode"] = "long"
                    continue
                value = low + max(1, (gap // axis["resolution"]) // 2) * axis["resolution"]
            elif mode == "long":
                if low is None:
                    axis["mode"] = "done"
                    continue
                value = low
            else:  # first or descending adjacent-resolution fallback
                value = axis["candidate"]
                if value < axis["floor"]:
                    axis.update(mode="done", limited_by=axis["limited_by"] or "no_passing_candidate")
                    continue
            profile = self._profile(axis, value)
            if self._excluded(profile):
                axis.update(high=value, mode="fallback", candidate=self._lower_value(axis, value))
                continue
            if mode == "fallback" and self._known(profile):
                axis.update(low=value, mode="long")
                continue
            quality = "long" if mode == "long" else "screen"
            return profile, quality
        return None

    def _curve_anchors(self, base):
        points = self.config.get("_device_curve", [])
        anchors = []
        for requested in self.config.get("search", {}).get("anchors_mv", [1000, 1025, 1050]):
            if type(requested) is not int:
                raise ValueError("Curve anchors must be integer millivolts")
            slots = []
            for point in points:
                if not isinstance(point, dict) or not all(type(point.get(k)) in (int, float)
                        and math.isfinite(point[k]) for k in ("mv", "mhz", "raw_offset")):
                    raise ValueError("Malformed measured device curve")
                if requested <= point["mv"] <= self.bounds.voltage_max and point["mv"] >= self.bounds.voltage_min:
                    slots.append(point)
            if not slots:
                continue
            slot = min(slots, key=lambda p: p["mv"])
            mv = int(slot["mv"])
            if any(p["mv"] == mv for p in anchors):
                continue
            resolution = self.settings["curve_resolution"]
            frequency = math.floor((slot["mhz"] - slot["raw_offset"] / 1000 + base.core) / resolution) * resolution
            if self.bounds.frequency_min <= frequency <= self.bounds.frequency_max:
                anchors.append(Profile("curve", base.core, base.mem, mv, frequency, base.fan,
                                       power_percent=base.power_percent).to_dict())
        return anchors

    def _advance_axis(self):
        axis = self.state["axis"]
        qualified = axis["qualified"]
        if self.state["final_research"]:
            self.state["final_research"] = False
            profile = Profile.from_dict(qualified) if qualified is not None else self._current_best()
            if profile is None:
                self._finish_power(reason="final_failed_no_qualified_fallback")
            else:
                self._begin_fans(profile)
            return
        if axis["name"] == "core":
            if qualified is None:
                self._finish_power(reason="core_not_long_qualified")
            else:
                self.state["base"] = qualified
                self._start_axis("memory", Profile.from_dict(qualified))
        elif axis["name"] == "memory":
            if qualified is not None:
                self.state["base"] = qualified
            base = Profile.from_dict(self.state["base"])
            if self._excluded(base):
                # A later joint test can revoke the core anchor itself.
                self._start_axis("core", base)
                current = self.state["axis"]
                current.update(mode="fallback", candidate=self._lower_value(current, base.core))
                return
            self.state["anchors"] = self._curve_anchors(base)
            self.state["anchor_index"] = 0
            self._next_anchor()
        else:
            self.state["anchor_index"] += 1
            self._next_anchor()

    def _next_anchor(self):
        index = self.state["anchor_index"]
        if index < len(self.state["anchors"]):
            self._start_axis("curve", Profile.from_dict(self.state["anchors"][index]))
            return
        profile = self._current_best()
        if profile is None:
            self._finish_power(reason="no_long_qualified_profile")
            return
        self._begin_fans(profile)

    def _begin_fans(self, profile):
        fans = self.config.get("fan_profiles", {})
        names = list(dict.fromkeys([profile.fan, *fans]))
        self.state.update(phase="fan", axis=None, selected=profile.to_dict(),
                          fan_candidates=names, fan_index=0, fan_results=[])

    def _fan_next(self):
        names = self.state["fan_candidates"]
        while self.state["fan_index"] < len(names):
            profile = replace(Profile.from_dict(self.state["selected"]), fan=names[self.state["fan_index"]])
            if not self._excluded(profile):
                return profile
            self.state["fan_index"] += 1
        records = [r for r in self.state["fan_results"] if not self._excluded(Profile.from_dict(r["spec"]["profile"]))]
        if not records:
            self._stop("no_passing_matched_fan_comparison")
            return None
        winner = max(records, key=lambda r: (r["result"]["score"], -r["result"]["peak_temp_c"]))
        self.state.update(phase="final", selected=deepcopy(winner["spec"]["profile"]))
        return None

    def next_trial(self, remaining_seconds: float) -> TrialSpec | None:
        if type(remaining_seconds) not in (int, float) or not math.isfinite(remaining_seconds) or remaining_seconds < 0:
            raise ValueError("Remaining budget must be finite and nonnegative")
        if self.state["status"] != "running":
            return None
        if self.state["pending"] is not None:
            return TrialSpec.from_dict(self.state["pending"])
        self.state["remaining_cap"] = min(self.state["remaining_cap"], remaining_seconds,
                                           self.state["budget_seconds"] - self.state["spent_seconds"])
        while self.state["status"] == "running":
            phase = self.state["phase"]
            if phase == "baseline":
                profile, quality, duration = self.initial, "screen", "baseline"
            elif phase == "power":
                profile, quality, duration = Profile.from_dict(self.state["base"]), "screen", "baseline"
            elif phase in ("core", "memory", "curve"):
                candidate = self._axis_next()
                if candidate is None:
                    self._advance_axis()
                    continue
                profile, quality = candidate
                duration = quality
            elif phase == "fan":
                profile = self._fan_next()
                if profile is None:
                    continue
                quality, duration = "screen", "fan"
            else:
                profile = Profile.from_dict(self.state["selected"])
                quality, duration = "long", "long"
                if self.state["final_attempts"] >= self.settings["max_long_attempts"]:
                    self._stop("final_long_attempt_limit")
                    continue
            self.bounds.check(profile)
            if self._excluded(profile):
                self._stop("excluded_configuration_requested")
                continue
            seconds = sum(self.durations[duration])
            if seconds > self.state["remaining_cap"]:
                self._stop("budget_exhausted")
                continue
            self.state["sequence"] += 1
            spec = TrialSpec(f"trial-{self.state['sequence']:06d}", phase, quality, profile,
                             *self.durations[duration], bandwidth=phase == "memory")
            self.state["pending"] = spec.to_dict()
            if phase == "baseline":
                self.state["power_results"][0]["status"] = "running"
            self.state["spent_seconds"] += spec.seconds
            self.state["remaining_cap"] -= spec.seconds
            self._event("trial_issued", trial=spec.to_dict())
            return spec
        return None

    def _complete_evidence(self, spec, result):
        if not result.load_started or any(not isinstance(w, dict) or w.get("kind") not in ("full", "half", "vram")
                                          for w in result.workloads):
            return False
        for workload in result.workloads:
            for key in ("achieved_core_mhz", "achieved_mem_mhz"):
                if key in workload and (type(workload[key]) not in (int, float)
                        or not math.isfinite(workload[key]) or workload[key] <= 0):
                    return False
        for kind, required in zip(("full", "half", "vram"),
                                  (spec.full_seconds, spec.half_seconds, spec.vram_seconds)):
            if not required:
                continue
            matches = [w for w in result.workloads if w.get("kind") == kind]
            if len(matches) != 1:
                return False
            workload = matches[0]
            if (workload.get("completed") is not True or type(workload.get("seconds")) not in (int, float)
                    or not math.isfinite(workload["seconds"]) or workload["seconds"] < required):
                return False
        return True

    def _axis_observe(self, spec, accepted):
        axis = self.state["axis"]
        value = self._axis_value(spec.profile, axis)
        if spec.quality == "long":
            axis["long_attempts"] += 1
            if accepted:
                axis.update(qualified=spec.profile.to_dict(), mode="done")
            else:
                axis.update(high=value, low=None, mode="fallback", candidate=self._lower_value(axis, value),
                            physical_failure=value)
        elif accepted:
            axis["low"] = value
            if value == axis["maximum"]:
                axis["driver_limit"] = True
            if axis["mode"] == "fallback":
                axis["mode"] = "long"
            elif axis["mode"] == "first":
                axis["mode"] = "coarse"
        else:
            axis.update(high=value, physical_failure=value)
            if axis["mode"] in ("fallback", "first"):
                axis.update(mode="fallback", candidate=self._lower_value(axis, value))
            else:
                axis["mode"] = "refine"

    def _advice(self, advice, phase):
        if advice.action == "stop":
            self._stop("ai_stop")
            return
        axis = self.state["axis"]
        if not axis or axis["name"] != phase or axis["mode"] == "done":
            return
        step = advice.mem_step if phase == "memory" else advice.core_step
        if step is not None:
            prefix = "mem" if phase == "memory" else phase
            resolution = axis["resolution"]
            axis["step"] = max(resolution, min(step, self.settings[prefix + "_step"]))
        if advice.action in ("validate", "advance") and axis["low"] is not None:
            if axis["mode"] not in ("long", "fallback"):
                axis["limited_by"] = "ai_" + advice.action
            axis["mode"] = "long"
        elif advice.action == "refine" and axis["low"] is not None:
            if axis["high"] is None:
                axis["high"] = min(axis["maximum"] + axis["resolution"], axis["low"] + axis["step"])
                axis["limited_by"] = "ai_refine"
            axis["mode"] = "refine"
        elif advice.action == "backoff" and axis["low"] is not None:
            axis.update(mode="fallback", candidate=self._lower_value(axis, axis["low"]), limited_by="ai_backoff")

    def observe(self, result: TrialResult, advice: Advice) -> None:
        pending = self.state["pending"]
        if pending is None or result.trial_id != pending["id"]:
            raise SafetyError("Result identity does not match the pending trial")
        spec = TrialSpec.from_dict(pending)
        if self.state["status"] != "running":
            raise SafetyError("A stopped search cannot consume trial results")
        complete = self._complete_evidence(spec, result)
        max_temp = self.config.get("safety", {}).get("max_temp_c", 80)
        thermal = result.outcome == "hot" or (result.peak_temp_c is not None and result.peak_temp_c >= max_temp)
        power_violation = self.board_cap is not None and result.peak_power_w is not None and result.peak_power_w > self.board_cap
        power_missing = (spec.profile.power_percent > 100 or self.board_cap is not None) and (
            result.full_avg_power_w is None or result.peak_power_w is None)
        accepted = result.outcome == "pass" and complete and not thermal and not power_violation and not power_missing
        record = {"spec": spec.to_dict(), "result": result.to_dict(), "accepted": accepted,
                  "advice": advice.to_dict()}
        self.state["trials"].append(record)
        self.state["pending"] = None
        self._event("trial_observed", trial_id=spec.id, outcome=result.outcome,
                    accepted=accepted, complete_evidence=complete, advice=advice.to_dict())
        if not accepted:
            self._exclude(spec.profile, "hot" if thermal else "power_interlock" if power_violation
                          else "missing_power_evidence" if power_missing else result.outcome
                          if result.outcome != "pass" else "incomplete_evidence")
        if result.outcome == "tdr":
            self.state["recoveries"] += 1
        if thermal or power_violation or power_missing or result.outcome in ("invalid", "stopped") or not result.load_started or (result.outcome == "pass" and not complete):
            self._stop("thermal_interlock" if thermal else "power_interlock" if power_violation
                       else "no_load_admission" if not result.load_started
                       else result.outcome if result.outcome in ("invalid", "stopped")
                       else "missing_power_evidence" if power_missing else "incomplete_evidence")
            return
        if self.state["recoveries"] > self.config.get("safety", {}).get("max_recoveries", 2):
            self._stop("recovery_limit")
            return
        if spec.phase == "baseline":
            if not accepted:
                self._stop("initial_baseline_failed")
                return
            self._begin_power()
        elif spec.phase == "power":
            if accepted:
                self._start_axis("core", spec.profile)
            else:
                self._finish_power(reason="power_baseline_failed")
        elif spec.phase in ("core", "memory", "curve"):
            self._axis_observe(spec, accepted)
        elif spec.phase == "fan":
            if accepted:
                self.state["fan_results"].append(record)
            self.state["fan_index"] += 1
        else:
            self.state["final_attempts"] += 1
            if accepted:
                self._finish_power(spec.profile)
            else:
                # A late failure retires the exact profile, including its old
                # LONG pass. Resolve the adjacent lower point, not a fixed margin.
                name = "curve" if spec.profile.kind == "curve" else "memory" if spec.profile.mem else "core"
                self._start_axis(name, spec.profile)
                axis = self.state["axis"]
                axis.update(mode="fallback", candidate=self._lower_value(axis, self._axis_value(spec.profile, axis)))
                self.state["final_research"] = True
        if self.state["status"] == "running":
            self._advice(advice, self.state["phase"] if spec.phase in ("baseline", "power") else spec.phase)

    def snapshot(self) -> dict:
        return {"version": self.VERSION, "initial": self.initial.to_dict(), "bounds": self.bounds.to_dict(),
                "config": deepcopy(self.config), "state": deepcopy(self.state)}

    @classmethod
    def restore(cls, state: dict, *, pending_result: TrialResult | None = None) -> SearchEngine:
        saved = deepcopy(state)
        if saved.get("version") != cls.VERSION or saved.get("state", {}).get("version") != cls.VERSION:
            raise ValueError("Unsupported search snapshot version")
        engine = cls(Profile.from_dict(saved["initial"]), Bounds(**saved["bounds"]), saved["config"])
        engine.state = saved["state"]
        # JSON round trips lose aliasing between active axis and the history list.
        if engine.state["axis"] is not None:
            engine.state["axis"] = engine.state["axes"][-1]
        if engine.state["spent_seconds"] < 0 or engine.state["spent_seconds"] > engine.state["budget_seconds"]:
            raise ValueError("Invalid cumulative budget")
        for record in engine.state["trials"]:
            engine.bounds.check(TrialSpec.from_dict(record["spec"]).profile)
            TrialResult.from_dict(record["result"])
        pending = engine.state["pending"]
        if pending is not None:
            spec = TrialSpec.from_dict(pending)
            engine.bounds.check(spec.profile)
            if pending_result is not None:
                if pending_result.trial_id != spec.id:
                    raise SafetyError("Persisted evidence does not match the pending checkpoint trial")
            else:
                engine._exclude(spec.profile, "pending_unknown")
                engine.state["pending"] = None
                engine._stop("pending_unknown_requires_operator_resolution")
        elif pending_result is not None:
            raise SafetyError("Persisted evidence has no pending checkpoint trial")
        return engine

    def summary(self) -> dict:
        records = self._records("long")
        best = self.best_profile
        performance = max(records, key=lambda r: r["result"]["score"]) if records else None
        global_highest = max(records, key=lambda r: (
            self._measured(r["result"])["achieved_core_mhz"] or 0,
            self._measured(r["result"])["achieved_mem_mhz"] or 0,
            r["result"]["score"]), default=None)
        axes = []
        for axis in self.state["axes"]:
            base = Profile.from_dict(axis["base"])
            matching = [r for r in self._records() if self._matches_axis(r["spec"]["profile"], axis)]
            screened = max(matching, key=lambda r: self._axis_value(Profile.from_dict(r["spec"]["profile"]), axis), default=None)
            longs = [r for r in matching if r["spec"]["quality"] == "long"]
            highest = max(longs, key=lambda r: self._axis_value(Profile.from_dict(r["spec"]["profile"]), axis), default=None)
            fastest = max(longs, key=lambda r: r["result"]["score"], default=None)
            axes.append({"axis": axis["name"], "anchor_mv": base.mv if axis["name"] == "curve" else None,
                         "power_percent": base.power_percent,
                         "screen_pass_ceiling": screened["spec"]["profile"] if screened else None,
                         "highest_long_validated": highest["spec"]["profile"] if highest else None,
                         "best_long_performance": fastest["spec"]["profile"] if fastest else None,
                         "achieved_clocks": self._measured(highest["result"]) if highest else None,
                         "driver_limit": axis["driver_limit"], "physical_failure": axis["physical_failure"],
                         "limited_by": axis["limited_by"], "resolution": axis["resolution"],
                         "absolute_maximum_proven": False})
        return {"version": self.VERSION, "status": self.state["status"], "phase": self.state["phase"],
                "reason": self.state["reason"], "best_profile": best.to_dict() if best else None,
                "highest_long_validated": deepcopy(global_highest["spec"]["profile"]) if global_highest else None,
                "selected_profile": deepcopy(self.state["selected"]),
                "best_long_performance": deepcopy(performance["spec"]["profile"]) if performance else None,
                "final_qualified": self.state["final_qualified"], "axes": deepcopy(axes),
                "power_results": deepcopy(self.state["power_results"]),
                "configured_long_seconds": dict(zip(("full_seconds", "half_seconds", "vram_seconds"), self.durations["long"])),
                "long_measurements": [
                    {"profile": deepcopy(r["spec"]["profile"]), "score": r["result"]["score"],
                     "full_avg_power_w": r["result"].get("full_avg_power_w"),
                     "peak_power_w": r["result"].get("peak_power_w"),
                     "score_per_watt": r["result"]["score"] / r["result"]["full_avg_power_w"]
                     if r["result"].get("full_avg_power_w") else None} for r in records],
                "spent_seconds": self.state["spent_seconds"], "remaining_seconds": self.state["remaining_cap"],
                "excluded": deepcopy(self.state["excluded"]), "trials": len(self.state["trials"]),
                "pending": deepcopy(self.state["pending"]), "absolute_maximum_proven": False}
