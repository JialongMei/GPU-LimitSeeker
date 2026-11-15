"""Synthetic policy-only scenarios. No GPU, vendor process or network is used."""
from dataclasses import replace
import json
import unittest

from gpu_limitseeker.engine import SearchEngine
from gpu_limitseeker.models import Advice, Bounds, Profile, SafetyError, TrialResult


CONTINUE = Advice("continue", "Synthetic evidence permits the bounded next step")


def configuration(**search):
    return {
        "search": {"core_step": 30, "core_resolution": 15, "mem_step": 100,
                   "mem_resolution": 50, "curve_step": 30, "curve_resolution": 15,
                   "anchors_mv": [], "max_long_attempts": 8, **search},
        "safety": {"max_hours": 12, "max_temp_c": 80, "max_recoveries": 2},
        "durations": {
            "baseline": {"full_seconds": 2, "half_seconds": 1, "vram_seconds": 2},
            "screen": {"full_seconds": 2, "half_seconds": 1, "vram_seconds": 2},
            "long": {"full_seconds": 10, "half_seconds": 3, "vram_seconds": 10},
            "fan": {"full_seconds": 3, "half_seconds": 1, "vram_seconds": 0},
        },
        "fan_profiles": {"performance": [], "cool": [], "balanced": []},
    }


def engine(core=90, mem=200, config=None, initial=None, **bounds):
    return SearchEngine(initial or Profile("uniform", 0, 0), Bounds(core, mem, **bounds),
                        config or configuration())


def evidence(spec, outcome="pass", **overrides):
    clock = spec.profile.mhz if spec.profile.kind == "curve" else 2200 + spec.profile.core
    workloads = tuple({"kind": kind, "seconds": seconds, "completed": outcome == "pass",
                       "achieved_core_mhz": clock, "achieved_mem_mhz": 5000 + spec.profile.mem}
                      for kind, seconds in (("full", spec.full_seconds), ("half", spec.half_seconds),
                                            ("vram", spec.vram_seconds)) if seconds)
    values = {"score": 100 + spec.profile.core + spec.profile.mem / 10,
              "peak_temp_c": 65, "workloads": workloads,
              "full_avg_power_w": 200 + spec.profile.power_percent - 100,
              "peak_power_w": 240 + spec.profile.power_percent - 100}
    values.update(overrides)
    return TrialResult(spec.id, outcome, **values)


def consume(policy, outcome="pass", advice=CONTINUE, **overrides):
    spec = policy.next_trial(1_000_000)
    if spec is None:
        raise AssertionError(policy.summary())
    policy.observe(evidence(spec, outcome, **overrides), advice)
    return spec


def until(policy, predicate, decide=None):
    seen = []
    for _ in range(500):
        spec = policy.next_trial(1_000_000)
        if spec is None:
            raise AssertionError(("Search ended before requested trial", policy.summary()))
        if predicate(spec):
            return spec, seen
        outcome = decide(spec) if decide else "pass"
        policy.observe(evidence(spec, outcome), CONTINUE)
        seen.append(spec)
    raise AssertionError("Unbounded policy transition")


def finish(policy, decide=None):
    seen = []
    for _ in range(1000):
        spec = policy.next_trial(1_000_000)
        if spec is None:
            return seen
        outcome = decide(spec) if decide else "pass"
        policy.observe(evidence(spec, outcome), CONTINUE)
        seen.append(spec)
    raise AssertionError("Unbounded policy search")


class EngineTests(unittest.TestCase):
    def test_baseline_is_exact_inspected_profile_not_stock(self):
        initial = Profile("uniform", 17, 40, fan="balanced", power_percent=95)
        cfg = configuration(power_limits=[95])
        cfg["safety"]["power_percent"] = 95
        policy = engine(initial=initial, config=cfg)
        first = policy.next_trial(1000)
        self.assertEqual(first.profile, initial)
        self.assertEqual(first.phase, "baseline")
        self.assertEqual(first.quality, "screen")
        self.assertIsNone(policy.best_profile)

    def test_pending_trial_is_reused_without_budget_charge(self):
        policy = engine()
        spec = policy.next_trial(1000)
        before = policy.snapshot()["state"]["spent_seconds"]
        self.assertEqual(policy.next_trial(999), spec)
        self.assertEqual(policy.snapshot()["state"]["spent_seconds"], before)

    def test_identity_mismatch_cannot_consume_pending(self):
        policy = engine()
        pending = policy.next_trial(1000)
        with self.assertRaises(SafetyError):
            policy.observe(TrialResult("another-trial", "invalid"), CONTINUE)
        self.assertEqual(policy.next_trial(1000), pending)

    def test_highest_core_gets_long_before_any_lower_selection(self):
        policy = engine()
        spec, seen = until(policy, lambda s: s.phase == "core" and s.quality == "long")
        self.assertEqual([s.profile.core for s in seen if s.phase == "core"], [30, 60, 90])
        self.assertEqual(spec.profile.core, 90)
        self.assertEqual((spec.full_seconds, spec.half_seconds, spec.vram_seconds), (10, 3, 10))
        self.assertIsNone(policy.best_profile)
        policy.observe(evidence(spec), CONTINUE)
        self.assertEqual(policy.best_profile.core, 90)

    def test_coarse_failure_bracket_is_refined_before_highest_long(self):
        policy = engine(core=120, config=configuration(core_step=60))
        spec, seen = until(policy, lambda s: s.phase == "core" and s.quality == "long",
                           lambda s: "wrong" if s.profile.core == 120 else "pass")
        self.assertEqual([s.profile.core for s in seen if s.phase == "core"], [60, 120, 90, 105])
        self.assertEqual(spec.profile.core, 105)
        axis = policy.summary()["axes"][0]
        self.assertEqual(axis["physical_failure"], 120)
        self.assertFalse(axis["absolute_maximum_proven"])

    def test_long_failure_resolves_omitted_adjacent_point_not_coarse_retreat(self):
        policy = engine(config=configuration(core_step=60))
        high, _ = until(policy, lambda s: s.phase == "core" and s.quality == "long")
        self.assertEqual(high.profile.core, 90)
        policy.observe(evidence(high, "wrong"), CONTINUE)
        adjacent = consume(policy)
        self.assertEqual((adjacent.quality, adjacent.profile.core), ("screen", 75))
        next_long = policy.next_trial(10000)
        self.assertEqual((next_long.quality, next_long.profile.core), ("long", 75))
        self.assertIsNone(policy.best_profile)
        self.assertEqual(policy.summary()["axes"][0]["physical_failure"], 90)

    def test_multiple_long_failures_exclude_all_failed_points(self):
        policy = engine(config=configuration(core_step=60))
        high, _ = until(policy, lambda s: s.phase == "core" and s.quality == "long")
        policy.observe(evidence(high, "wrong"), CONTINUE)
        consume(policy)  # adjacent 75 screen
        middle = consume(policy, "wrong")  # 75 long
        low = policy.next_trial(10000)
        self.assertEqual(middle.profile.core, 75)
        self.assertEqual((low.profile.core, low.quality), (60, "long"))
        self.assertEqual({p["profile"]["core"] for p in policy.summary()["excluded"]}, {75, 90})

    def test_memory_has_its_own_highest_long_and_resolution_fallback(self):
        policy = engine(core=0, mem=300, config=configuration(mem_step=200))
        spec, seen = until(policy, lambda s: s.phase == "memory" and s.quality == "long")
        self.assertEqual([s.profile.mem for s in seen if s.phase == "memory"], [200, 300])
        self.assertEqual(spec.profile.mem, 300)
        policy.observe(evidence(spec, "wrong"), CONTINUE)
        adjacent = consume(policy)
        self.assertEqual((adjacent.phase, adjacent.quality, adjacent.profile.mem), ("memory", "screen", 250))
        qualified = consume(policy)
        self.assertEqual((qualified.quality, qualified.profile.mem), ("long", 250))

    def test_failed_points_persist_across_json_resume(self):
        policy = engine(config=configuration(core_step=60))
        high, _ = until(policy, lambda s: s.phase == "core" and s.quality == "long")
        policy.observe(evidence(high, "wrong"), CONTINUE)
        saved = json.loads(json.dumps(policy.snapshot()))
        resumed = SearchEngine.restore(saved)
        issued = finish(resumed)
        self.assertTrue(issued)
        self.assertNotIn(high.profile.key, [s.profile.key for s in issued])
        self.assertEqual(resumed.summary()["status"], "complete")
        self.assertGreater(resumed.summary()["spent_seconds"], saved["state"]["spent_seconds"])

    def test_unknown_pending_after_crash_is_latched_and_never_replayed(self):
        policy = engine()
        pending = policy.next_trial(1000)
        resumed = SearchEngine.restore(json.loads(json.dumps(policy.snapshot())))
        self.assertIsNone(resumed.next_trial(1_000_000))
        self.assertIn("pending_unknown", resumed.summary()["reason"])
        self.assertEqual(resumed.summary()["excluded"][0]["profile"], pending.profile.to_dict())
        self.assertEqual(resumed.summary()["spent_seconds"], pending.seconds)

    def test_late_long_failure_revokes_earlier_long_not_only_short_pass(self):
        policy = engine(core=90, mem=0)
        memory, _ = until(policy, lambda s: s.phase == "memory" and s.quality == "long")
        self.assertEqual(policy.best_profile.core, 90)
        policy.observe(evidence(memory, "wrong"), CONTINUE)
        self.assertIsNone(policy.best_profile)
        next_spec = policy.next_trial(10000)
        self.assertEqual((next_spec.phase, next_spec.profile.core), ("core", 75))
        self.assertNotEqual(next_spec.profile.key, memory.profile.key)

    def test_final_long_failure_re_refines_adjacent_configuration(self):
        policy = engine(core=90, mem=0)
        final, _ = until(policy, lambda s: s.phase == "final")
        self.assertIsNotNone(policy.best_profile)
        policy.observe(evidence(final, "wrong"), CONTINUE)
        later = finish(policy)
        self.assertEqual((later[0].phase, later[0].quality, later[0].profile.core), ("core", "screen", 75))
        self.assertNotIn(final.profile.key, [s.profile.key for s in later])
        self.assertEqual(policy.summary()["status"], "complete")
        self.assertEqual(policy.best_profile.core, 75)

    def test_full_long_must_include_complete_half_and_vram_evidence(self):
        for missing in ("full", "half", "vram"):
            with self.subTest(missing=missing):
                policy = engine()
                spec, _ = until(policy, lambda s: s.quality == "long")
                good = evidence(spec)
                partial = tuple(w for w in good.workloads if w["kind"] != missing)
                policy.observe(replace(good, workloads=partial), Advice("advance", "Do not override missing evidence"))
                self.assertIsNone(policy.best_profile)
                self.assertEqual(policy.summary()["reason"], "incomplete_evidence")
                self.assertIsNone(policy.next_trial(10000))

    def test_shortened_or_uncompleted_work_is_not_long_qualified(self):
        for mutation in ({"seconds": 9}, {"completed": False}):
            policy = engine()
            spec, _ = until(policy, lambda s: s.quality == "long")
            good = evidence(spec)
            workloads = [dict(w) for w in good.workloads]
            workloads[0].update(mutation)
            policy.observe(replace(good, workloads=tuple(workloads)), CONTINUE)
            self.assertIsNone(policy.best_profile)
            self.assertEqual(policy.summary()["reason"], "incomplete_evidence")

    def test_budget_does_not_shorten_long_or_invent_best(self):
        policy = engine(core=0, mem=0)
        consume(policy)
        self.assertIsNone(policy.next_trial(22))
        self.assertIsNone(policy.best_profile)
        self.assertEqual(policy.summary()["reason"], "budget_exhausted")
        self.assertFalse(policy.summary()["final_qualified"])
        self.assertEqual(policy.summary()["configured_long_seconds"],
                         {"full_seconds": 10, "half_seconds": 3, "vram_seconds": 10})

    def test_resume_cannot_reset_original_total_budget(self):
        cfg = configuration()
        cfg["safety"]["max_hours"] = 30 / 3600
        policy = engine(core=0, mem=0, config=cfg)
        consume(policy)
        consume(policy)  # 5 + 23 seconds reserved cumulatively
        resumed = SearchEngine.restore(json.loads(json.dumps(policy.snapshot())))
        self.assertIsNone(resumed.next_trial(1_000_000))
        self.assertEqual(resumed.summary()["spent_seconds"], 28)
        self.assertIsNotNone(resumed.best_profile)
        self.assertFalse(resumed.summary()["final_qualified"])

    def test_non_grid_driver_endpoint_does_not_skip_higher_screen_pass(self):
        policy = engine(core=65, mem=0)
        high, _ = until(policy, lambda s: s.phase == "core" and s.quality == "long")
        self.assertEqual(high.profile.core, 65)
        policy.observe(evidence(high, "wrong"), CONTINUE)
        lower = policy.next_trial(10000)
        self.assertEqual((lower.profile.core, lower.quality), (60, "long"))

    def test_driver_ceiling_is_not_instability_or_physical_maximum(self):
        policy = engine(core=65, mem=0)
        spec, seen = until(policy, lambda s: s.phase == "core" and s.quality == "long")
        self.assertEqual([s.profile.core for s in seen if s.phase == "core"], [30, 60, 65])
        self.assertEqual(spec.profile.core, 65)
        axis = policy.summary()["axes"][0]
        self.assertTrue(axis["driver_limit"])
        self.assertIsNone(axis["physical_failure"])
        self.assertFalse(axis["absolute_maximum_proven"])

    def test_invalid_no_load_is_not_a_physical_oc_failure(self):
        policy = engine()
        consume(policy)
        spec = consume(policy, "invalid", load_started=False, workloads=())
        self.assertEqual(spec.phase, "core")
        self.assertEqual(policy.summary()["reason"], "no_load_admission")
        self.assertIsNone(policy.summary()["axes"][0]["physical_failure"])
        self.assertIsNone(policy.next_trial(10000))

    def test_thermal_and_operator_stop_override_ai_advance(self):
        for outcome, overrides, reason in (("pass", {"peak_temp_c": 80}, "thermal_interlock"),
                                            ("hot", {}, "thermal_interlock"),
                                            ("stopped", {}, "stopped")):
            policy = engine()
            consume(policy, outcome, Advice("advance", "Synthetic premature advance"), **overrides)
            self.assertIsNone(policy.next_trial(10000))
            self.assertIsNone(policy.best_profile)
            self.assertEqual(policy.summary()["reason"], reason)

    def test_ai_steps_are_bounded_and_early_qualification_is_labelled(self):
        policy = engine()
        consume(policy, advice=Advice("continue", "Use a smaller bounded step", core_step=1))
        first = consume(policy, advice=Advice("advance", "Qualify early, do not claim the maximum"))
        self.assertEqual(first.profile.core, 15)
        long_trial = policy.next_trial(10000)
        self.assertEqual((long_trial.quality, long_trial.profile.core), ("long", 15))
        self.assertEqual(policy.summary()["axes"][0]["limited_by"], "ai_advance")
        self.assertIsNone(policy.best_profile)

    def test_ai_validate_refine_backoff_stop_all_steer_without_fake_pass(self):
        for action, expected_quality, expected_core in (("validate", "long", 30),
                                                         ("refine", "screen", 45),
                                                         ("backoff", "screen", 15)):
            policy = engine()
            consume(policy)
            consume(policy, advice=Advice(action, "Synthetic bounded steering"))
            spec = policy.next_trial(10000)
            self.assertEqual((spec.quality, spec.profile.core), (expected_quality, expected_core))
            self.assertIsNone(policy.best_profile)
        policy = engine()
        consume(policy, advice=Advice("stop", "End this synthetic search"))
        self.assertIsNone(policy.next_trial(10000))
        self.assertEqual(policy.summary()["reason"], "ai_stop")

    def test_measured_clock_and_performance_winners_are_distinct(self):
        policy = engine(core=30, mem=100)
        first, _ = until(policy, lambda s: s.phase == "core" and s.quality == "long")
        good = evidence(first, score=500)
        policy.observe(good, CONTINUE)
        second, _ = until(policy, lambda s: s.phase == "memory" and s.quality == "long")
        faster_clocks = tuple({**w, "achieved_core_mhz": 2400} for w in evidence(second).workloads)
        policy.observe(evidence(second, score=400, workloads=faster_clocks), CONTINUE)
        summary = policy.summary()
        self.assertEqual(summary["highest_long_validated"]["mem"], 100)
        self.assertEqual(summary["best_long_performance"]["mem"], 0)
        self.assertEqual(summary["axes"][1]["achieved_clocks"]["achieved_core_mhz"], 2400)
        self.assertNotEqual(summary["highest_long_validated"], summary["best_long_performance"])

    def test_missing_achieved_clock_is_not_filled_with_requested_target(self):
        policy = engine(core=30, mem=0)
        spec, _ = until(policy, lambda s: s.phase == "core" and s.quality == "long")
        workloads = tuple({k: v for k, v in w.items() if not k.startswith("achieved_")}
                          for w in evidence(spec).workloads)
        policy.observe(evidence(spec, workloads=workloads), CONTINUE)
        self.assertEqual(policy.summary()["axes"][0]["achieved_clocks"],
                         {"achieved_core_mhz": None, "achieved_mem_mhz": None})

    def test_curve_reference_uses_measured_stock_component_and_available_slot(self):
        cfg = configuration(anchors_mv=[1000, 1001])
        cfg["_device_curve"] = [{"index": 0, "mv": 1005, "mhz": 2415, "raw_offset": 30000}]
        policy = engine(core=30, mem=0, config=cfg, frequency_max=2475)
        first, _ = until(policy, lambda s: s.phase == "curve")
        self.assertEqual((first.profile.mv, first.profile.mhz), (1005, 2415))
        policy.observe(evidence(first), CONTINUE)
        high, seen = until(policy, lambda s: s.phase == "curve" and s.quality == "long")
        self.assertEqual(high.profile.mhz, 2475)
        self.assertEqual([s.profile.mhz for s in seen], [2445, 2475])
        policy.observe(evidence(high, "wrong"), CONTINUE)
        adjacent = consume(policy)
        self.assertEqual((adjacent.quality, adjacent.profile.mhz), ("screen", 2460))
        self.assertEqual(consume(policy).quality, "long")
        self.assertEqual(len(policy.snapshot()["state"]["anchors"]), 1)

    def test_no_measured_curve_never_fabricates_reference_clock(self):
        policy = engine(core=0, mem=0, config=configuration(anchors_mv=[1000]))
        issued = finish(policy)
        self.assertNotIn("curve", [s.phase for s in issued])
        self.assertTrue(policy.summary()["final_qualified"])

    def test_matched_fan_comparison_then_full_long_with_winning_fan(self):
        policy = engine(core=0, mem=0)
        seen = []
        for _ in range(100):
            spec = policy.next_trial(10000)
            if spec is None:
                break
            seen.append(spec)
            score = {"performance": 100, "cool": 110, "balanced": 120}[spec.profile.fan]
            policy.observe(evidence(spec, score=score), CONTINUE)
        fans = [s for s in seen if s.phase == "fan"]
        self.assertEqual([s.profile.fan for s in fans], ["performance", "cool", "balanced"])
        self.assertEqual({(s.full_seconds, s.half_seconds, s.vram_seconds) for s in fans}, {(3, 1, 0)})
        final = [s for s in seen if s.phase == "final"]
        self.assertEqual(len(final), 1)
        self.assertEqual((final[0].profile.fan, final[0].quality, final[0].seconds), ("balanced", "long", 23))
        self.assertEqual(policy.best_profile.fan, "balanced")
        self.assertEqual(policy.summary()["status"], "complete")

    def test_snapshot_and_summary_are_detached_copies(self):
        policy = engine()
        consume(policy)
        saved = policy.snapshot()
        saved["state"]["axes"][0]["low"] = 99999
        summary = policy.summary()
        summary["axes"][0]["resolution"] = 99999
        self.assertEqual(policy.next_trial(10000).profile.core, 30)

    def test_long_attempt_limit_is_honest_not_a_fake_qualification(self):
        policy = engine(core=30, config=configuration(max_long_attempts=1))
        high, _ = until(policy, lambda s: s.quality == "long")
        policy.observe(evidence(high, "wrong"), CONTINUE)
        self.assertIsNone(policy.next_trial(10000))
        self.assertIsNone(policy.best_profile)
        self.assertEqual(policy.summary()["axes"][0]["limited_by"], "long_attempt_limit")

    def test_power_sweep_requires_explicit_percent_watt_and_driver_envelopes(self):
        for allow_percent, cap, driver_max in ((100, 400, 125), (125, None, 125), (125, 400, 100)):
            cfg = configuration(power_limits=[100, 110, 125])
            cfg["safety"].update(max_power_percent=allow_percent, max_board_power_w=cap)
            with self.assertRaises(SafetyError):
                engine(config=cfg, power_max_percent=driver_max)
        with self.assertRaises(SafetyError):
            engine(config=configuration(power_limits=[110, 100]), power_max_percent=125)

    def test_each_approved_power_gets_independent_high_candidate_long(self):
        cfg = configuration(power_limits=[100, 110, 125])
        cfg["safety"].update(max_power_percent=125, max_board_power_w=400)
        policy = engine(core=30, mem=100, config=cfg, power_max_percent=125)
        seen = finish(policy)
        for power in (100, 110, 125):
            longs = [s for s in seen if s.profile.power_percent == power and s.quality == "long"]
            self.assertIn(("core", 30, 0), [(s.phase, s.profile.core, s.profile.mem) for s in longs])
            self.assertIn(("memory", 30, 100), [(s.phase, s.profile.core, s.profile.mem) for s in longs])
            self.assertIn("final", [s.phase for s in longs])
        self.assertEqual([p["status"] for p in policy.summary()["power_results"]], ["complete"] * 3)
        self.assertEqual({a["power_percent"] for a in policy.summary()["axes"]}, {100, 110, 125})

    def test_power_specific_exclusions_do_not_poison_other_power_points(self):
        cfg = configuration(power_limits=[100, 110])
        cfg["safety"].update(max_power_percent=110, max_board_power_w=400)
        policy = engine(core=30, mem=0, config=cfg, power_max_percent=110)
        seen = finish(policy, lambda s: "wrong" if s.phase == "core" and s.quality == "long"
                      and s.profile.core == 30 and s.profile.power_percent == 100 else "pass")
        self.assertTrue(any(s.profile.core == 30 and s.profile.power_percent == 110 and s.quality == "long" for s in seen))
        self.assertEqual(policy.summary()["power_results"][1]["profile"]["core"], 30)
        self.assertEqual({p["profile"]["power_percent"] for p in policy.summary()["excluded"]}, {100})

    def test_budget_leaves_future_power_points_explicitly_untested(self):
        cfg = configuration(power_limits=[100, 110, 125])
        cfg["safety"].update(max_power_percent=125, max_board_power_w=400)
        policy = engine(core=0, mem=0, config=cfg, power_max_percent=125)
        consume(policy)
        self.assertIsNone(policy.next_trial(0))
        points = policy.summary()["power_results"]
        self.assertEqual([p["status"] for p in points], ["stopped", "untested", "untested"])
        self.assertIsNone(policy.best_profile)

    def test_watts_efficiency_uses_measurement_not_power_percentage(self):
        policy = engine(core=0, mem=0)
        spec, _ = until(policy, lambda s: s.quality == "long")
        policy.observe(evidence(spec, score=600, full_avg_power_w=240, peak_power_w=260), CONTINUE)
        measurement = policy.summary()["long_measurements"][0]
        self.assertEqual(measurement["score_per_watt"], 2.5)
        self.assertEqual(measurement["full_avg_power_w"], 240)
        self.assertEqual(measurement["profile"]["power_percent"], 100)

    def test_missing_watts_remain_unknown_not_a_configured_cap(self):
        policy = engine(core=0, mem=0)
        spec, _ = until(policy, lambda s: s.quality == "long")
        policy.observe(evidence(spec, full_avg_power_w=None, peak_power_w=None), CONTINUE)
        self.assertIsNone(policy.summary()["long_measurements"][0]["score_per_watt"])
        self.assertIsNone(policy.summary()["long_measurements"][0]["full_avg_power_w"])

    def test_watt_cap_is_hard_stop_even_with_ai_continue(self):
        cfg = configuration()
        cfg["safety"]["max_board_power_w"] = 250
        policy = engine(config=cfg)
        consume(policy, peak_power_w=251)
        self.assertEqual(policy.summary()["reason"], "power_interlock")
        self.assertIsNone(policy.next_trial(10000))

    def test_recovery_count_and_budget_are_global_across_power_resume(self):
        cfg = configuration(power_limits=[100, 110])
        cfg["safety"].update(max_power_percent=110, max_board_power_w=400, max_recoveries=1)
        policy = engine(core=30, mem=0, config=cfg, power_max_percent=110)
        consume(policy)
        consume(policy, "tdr")
        saved = policy.snapshot()
        policy = SearchEngine.restore(json.loads(json.dumps(saved)))
        next_high, _ = until(policy, lambda s: s.profile.power_percent == 110 and s.phase == "core")
        policy.observe(evidence(next_high, "tdr"), CONTINUE)
        self.assertEqual(policy.summary()["reason"], "recovery_limit")
        self.assertEqual(policy.snapshot()["state"]["recoveries"], 2)
        self.assertGreater(policy.summary()["spent_seconds"], saved["state"]["spent_seconds"])

    def test_duplicate_or_malformed_loaded_clock_evidence_halts(self):
        for bad in ("duplicate", "nonfinite", "nonobject"):
            policy = engine(core=0, mem=0)
            spec, _ = until(policy, lambda s: s.quality == "long")
            workloads = [dict(w) for w in evidence(spec).workloads]
            if bad == "duplicate":
                workloads.append(dict(workloads[0]))
            elif bad == "nonfinite":
                workloads[0]["achieved_core_mhz"] = float("nan")
            else:
                workloads.append("not telemetry")
            policy.observe(evidence(spec, workloads=tuple(workloads)), CONTINUE)
            self.assertEqual(policy.summary()["reason"], "incomplete_evidence")
            self.assertIsNone(policy.best_profile)

    def test_curve_profiles_preserve_current_approved_power(self):
        cfg = configuration(power_limits=[100, 110], anchors_mv=[1000])
        cfg["safety"].update(max_power_percent=110, max_board_power_w=400)
        cfg["_device_curve"] = [{"index": 0, "mv": 1000, "mhz": 2400, "raw_offset": 0}]
        policy = engine(core=0, mem=0, config=cfg, power_max_percent=110, frequency_max=2430)
        seen = finish(policy)
        for power in (100, 110):
            curves = [s for s in seen if s.phase == "curve" and s.profile.power_percent == power]
            self.assertEqual([s.profile.mhz for s in curves if s.quality == "long"], [2430])

    def test_explicit_board_cap_requires_observed_watts(self):
        cfg = configuration()
        cfg["safety"]["max_board_power_w"] = 400
        policy = engine(config=cfg)
        consume(policy, full_avg_power_w=None, peak_power_w=None)
        self.assertEqual(policy.summary()["reason"], "missing_power_evidence")
        self.assertIsNone(policy.next_trial(10000))


if __name__ == "__main__":
    unittest.main()
