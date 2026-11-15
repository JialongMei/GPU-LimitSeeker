"""Consumer-visible lifecycle regressions using synthetic hardware only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_limitseeker.config import defaults, json_loads, validate_config
from gpu_limitseeker.controller import run
from gpu_limitseeker.engine import SearchEngine
from gpu_limitseeker.models import Advice, Bounds, Profile, SafetyError, TrialResult
from gpu_limitseeker.storage import load_json, run_lock, save_json
from gpu_limitseeker.tools import _publish


def evidence(spec):
    return TrialResult(spec.id, "pass", score=100, peak_temp_c=55,
        workloads=tuple({"kind": kind, "seconds": seconds, "completed": True,
                         "achieved_core_mhz": 2000, "achieved_mem_mhz": 5000}
                        for kind, seconds in (("full", spec.full_seconds), ("half", spec.half_seconds), ("vram", spec.vram_seconds)) if seconds),
        full_avg_power_w=200, peak_power_w=230)


def configuration():
    cfg = defaults()
    cfg["durations"]["long"] = {"full_seconds": 300, "half_seconds": 60, "vram_seconds": 300}
    return cfg


class ContinueReviewer:
    def __init__(self, config, work_dir):
        pass

    def review(self, context):
        return Advice("continue", "Synthetic fixture permits the bounded next step")


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.calls = []
        self.instances = []
        outer = self

        class Backend:
            def __init__(self, config, run_dir):
                self.config = config
                self.run_dir = Path(run_dir)
                self.sequence = 0
                self.abort_trial = False
                self.fail_restore = False
                outer.instances.append(self)

            def inspect(self):
                return {"uuid": "SYNTHETIC-DEVICE-NOT-A-REAL-UUID", "driver": "synthetic-driver",
                        "power_w": 300, "default_power_w": 300,
                        "initial": Profile("uniform", 0, 0).to_dict(),
                        "bounds": Bounds(0, 0, 1000, 1050, 2000, 2000, 100, 125).to_dict(),
                        "curve_points_data": [{"index": i, "mv": mv, "mhz": 2000, "raw_offset": 0}
                                              for i, mv in enumerate((1000, 1025, 1050))]}

            def run_trial(self, spec, stop_file, check=None):
                self.sequence += 1
                (self.run_dir / f"trial-{self.sequence:06d}").mkdir()
                outer.calls.append(spec)
                if check:
                    check()
                if self.abort_trial:
                    raise KeyboardInterrupt()
                return evidence(spec)

            def recover_tdr(self, fallback, stop_file):
                raise AssertionError("This fixture never injects a TDR")

            def restore(self):
                if self.fail_restore:
                    raise RuntimeError("Synthetic unconfirmed native restoration")
                return {"restored": True, "snapshot": self.inspect()}

            def close(self, relinquish=False):
                self.closed = True

        self.backend = Backend
        self.lock_patch = patch("gpu_limitseeker.controller.hardware_writer_lock",
                                lambda _: run_lock(self.root / "hardware.lock"))
        self.lock_patch.start()

    def tearDown(self):
        self.lock_patch.stop()
        self.temporary.cleanup()

    def execute(self, cfg=None, **kwargs):
        return run(cfg or configuration(), self.root / "run", backend_factory=self.backend,
                   reviewer_factory=kwargs.pop("reviewer_factory", ContinueReviewer),
                   tools_resolver=deepcopy, **kwargs)

    def test_each_approved_power_requires_its_own_final_long_evidence(self):
        cfg = configuration()
        cfg["safety"].update(max_power_percent=125, max_board_power_w=375)
        cfg["search"]["power_limits"] = [100, 110, 125]
        report = self.execute(cfg)
        self.assertEqual(report["status"], "completed")
        finals = [spec for spec in self.calls if spec.phase == "final" and spec.quality == "long"]
        self.assertEqual({spec.profile.power_percent for spec in finals}, {100, 110, 125})
        self.assertTrue(all(point["status"] == "complete" for point in report["summary"]["power_results"]))
        self.assertTrue(report["restore_receipt"]["restored"])

    def test_ai_failure_before_native_construction_cannot_run_pressure(self):
        class Refuse(ContinueReviewer):
            def review(self, context):
                raise RuntimeError("Synthetic provider refusal")
        with self.assertRaises(RuntimeError):
            self.execute(reviewer_factory=Refuse)
        self.assertEqual(self.instances, [])
        self.assertEqual(self.calls, [])
        self.assertEqual(load_json(self.root / "run" / "report.json")["status"], "failed")

    def test_budget_exhaustion_is_not_success_or_qualification(self):
        cfg = configuration()
        cfg["safety"]["max_hours"] = 0.1
        report = self.execute(cfg)
        self.assertEqual(report["status"], "stopped")
        self.assertFalse(report["summary"]["final_qualified"])
        self.assertEqual(self.calls, [])
        self.assertEqual(report["summary"]["power_results"][0]["status"], "untested")

    def test_known_result_resume_reviews_without_replaying_original_pressure(self):
        class PauseReview(ContinueReviewer):
            def review(self, context):
                if context["pending_trial"] is not None:
                    raise KeyboardInterrupt()
                return super().review(context)
        with self.assertRaises(RuntimeError):
            self.execute(reviewer_factory=PauseReview)
        original = self.calls[0].id
        state_path = self.root / "run" / "state.json"
        state = load_json(state_path)
        self.assertEqual(state["phase"], "awaiting_review")
        state["elapsed_seconds"] += 3600
        state["recoveries"] = 1
        save_json(state_path, state)
        report = self.execute(resume=True)
        self.assertEqual(sum(spec.id == original for spec in self.calls), 1)
        self.assertEqual(sum(spec.id.startswith("resume-") for spec in self.calls), 1)
        self.assertGreaterEqual(report["elapsed_seconds"], 3600)
        self.assertIsNone(report["error"])
        self.assertEqual(report["recoveries"], 1)
        self.assertLessEqual(self.instances[-1].config["_runtime_remaining_seconds"],
                             state["config"]["safety"]["max_hours"] * 3600 - state["elapsed_seconds"])
        self.assertEqual(self.instances[-1].config["_runtime_recoveries_used"], 1)
        self.assertTrue(report["summary"]["final_qualified"])

    def test_unknown_pressure_checkpoint_never_launches_another_backend(self):
        base = self.backend
        class Interrupt(base):
            def __init__(self, config, work_dir):
                super().__init__(config, work_dir)
                self.abort_trial = True
        self.backend = Interrupt
        with self.assertRaises(RuntimeError):
            self.execute()
        previous_count = len(self.instances)
        with self.assertRaises(SafetyError):
            self.execute(resume=True)
        self.assertEqual(len(self.instances), previous_count)
        self.assertEqual(len(self.calls), 1)

    def test_uncertain_restore_is_failure_and_still_closes_children(self):
        base = self.backend
        class Uncertain(base):
            def __init__(self, config, work_dir):
                super().__init__(config, work_dir)
                self.fail_restore = True
        self.backend = Uncertain
        with self.assertRaises(RuntimeError):
            self.execute()
        report = load_json(self.root / "run" / "report.json")
        self.assertEqual(report["status"], "failed")
        self.assertTrue(self.instances[-1].closed)

    def test_resumed_tdr_uses_rechecked_original_not_historical_long_profile(self):
        base, outer = self.backend, self

        class SessionAdmission(base):
            def __init__(self, config, work_dir):
                super().__init__(config, work_dir)
                self.passing = set()
                self.recovered = []
                self.injected = False
                self.resumed = False

            def inspect(self):
                device = super().inspect()
                device["bounds"] = Bounds(30, 0, 1000, 1050, 2000, 2060, 100, 125).to_dict()
                return device

            def run_trial(self, spec, stop_file, check=None):
                if self.resumed and not self.injected and spec.phase == "curve":
                    self.injected = True
                    outer.calls.append(spec)
                    return TrialResult(spec.id, "tdr", peak_temp_c=55)
                observed = super().run_trial(spec, stop_file, check)
                self.passing.add(spec.profile.key)
                if spec.id.startswith("resume-"):
                    self.resumed = True
                return observed

            def recover_tdr(self, fallback, stop_file):
                if fallback.key not in self.passing:
                    raise SafetyError("Historical profile has no current-session recovery admission")
                self.recovered.append(fallback)
                return {"recovered": True}

        class PauseAtCurve(ContinueReviewer):
            def review(self, context):
                trial = context["pending_trial"]
                if trial is not None and trial["phase"] == "curve":
                    raise KeyboardInterrupt()
                return super().review(context)

        self.backend = SessionAdmission
        with self.assertRaises(RuntimeError):
            self.execute(reviewer_factory=PauseAtCurve)
        prior = load_json(self.root / "run" / "state.json")
        self.assertEqual(prior["engine"]["state"]["trials"][-1]["spec"]["profile"]["core"], 30)
        report = self.execute(resume=True)
        self.assertTrue(report["summary"]["final_qualified"])
        self.assertEqual(self.instances[-1].recovered, [Profile("uniform", 0, 0)])

    def test_changed_raw_curve_cannot_reuse_old_qualification_or_run_resume_pressure(self):
        base = self.backend
        initial = Profile("curve", 300, 0, 1000, 2300)

        class CurveOriginal(base):
            changed = False

            def inspect(self):
                device = super().inspect()
                device["initial"] = initial.to_dict()
                device["bounds"] = Bounds(300, 0, 1000, 1050, 2000, 2300, 100, 125).to_dict()
                device["curve_points_data"] = [
                    {"index": i, "mv": mv, "mhz": 2300, "raw_offset": offset}
                    for i, (mv, offset) in enumerate(((1000, 300000), (1025, 250000), (1050, 200000)))
                ]
                if self.changed:
                    device["curve_points_data"][1]["raw_offset"] += 1
                return device

        class PauseReview(ContinueReviewer):
            def review(self, context):
                if context["pending_trial"] is not None:
                    raise KeyboardInterrupt()
                return super().review(context)

        cfg = configuration()
        cfg["hardware"]["baseline_profile"] = initial.to_dict()
        self.backend = CurveOriginal
        with self.assertRaises(RuntimeError):
            self.execute(cfg, reviewer_factory=PauseReview)
        CurveOriginal.changed = True
        with self.assertRaises(RuntimeError):
            self.execute(cfg, resume=True)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(load_json(self.root / "run" / "report.json")["status"], "failed")



class EnvelopeTests(unittest.TestCase):
    def test_raised_power_needs_both_explicit_ceilings(self):
        for changes in ({"search": {"power_limits": [100, 125]}},
                        {"safety": {"max_power_percent": 125}, "search": {"power_limits": [100, 125]}}):
            with self.assertRaises(ValueError):
                validate_config(changes)
        cfg = validate_config({"safety": {"max_power_percent": 125, "max_board_power_w": 375},
                               "search": {"power_limits": [100, 125]}})
        self.assertEqual(cfg["search"]["power_limits"], [100, 125])
        with self.assertRaises(SafetyError):
            Bounds(500, 2000).check(Profile("uniform", 0, 0, power_percent=125))

    def test_power_is_part_of_exact_failure_identity(self):
        self.assertNotEqual(Profile("uniform", 30, 100).key,
                            Profile("uniform", 30, 100, power_percent=110).key)

    def test_duplicate_or_nonfinite_json_cannot_reinterpret_safety(self):
        for raw in ('{"max_power_percent":100,"max_power_percent":125}', '{"max_hours":NaN}'):
            with self.assertRaises(ValueError):
                json_loads(raw)

    def test_unsupported_thermal_threshold_refuses_before_live_setup(self):
        with self.assertRaises(ValueError):
            validate_config({"safety": {"max_temp_c": 81}})

    def test_unknown_pending_cannot_be_certified_by_other_trial_evidence(self):
        cfg = configuration()
        policy = SearchEngine(Profile("uniform", 0, 0), Bounds(30, 100), cfg)
        pending = policy.next_trial(10000)
        wrong = TrialResult("unrelated", "invalid")
        with self.assertRaises(SafetyError):
            SearchEngine.restore(policy.snapshot(), pending_result=wrong)
        unknown = SearchEngine.restore(policy.snapshot())
        self.assertIsNone(unknown.next_trial(10000))
        restored = SearchEngine.restore(policy.snapshot(), pending_result=evidence(pending))
        restored.observe(evidence(pending), Advice("continue", "Synthetic known completed result"))
        self.assertEqual(restored.summary()["trials"], 1)

    def test_download_collision_never_deletes_another_writers_file(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "source", Path(root) / "target"
            source.write_bytes(b"pinned-new-tool")
            target.write_bytes(b"another-writers-file")
            original = Path.exists
            with patch.object(Path, "exists", lambda path: False if path == target else original(path)):
                with self.assertRaises(FileExistsError):
                    _publish(source, target)
            self.assertEqual(target.read_bytes(), b"another-writers-file")


class HomeRelativeCLITests(unittest.TestCase):
    def test_resume_reads_the_home_relative_checkpoint(self):
        import io
        import os
        from contextlib import redirect_stdout
        from gpu_limitseeker.cli import main
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "gpu-run"
            save_json(run_dir / "state.json", {"config": configuration()})
            output = io.StringIO()
            with patch.dict(os.environ, {"HOME": root, "USERPROFILE": root}), \
                    patch("gpu_limitseeker.cli._approval"), \
                    patch("gpu_limitseeker.controller.run", return_value={"status": "synthetic-no-control"}), \
                    redirect_stdout(output):
                code = main(["resume", "--run-dir", str(Path("~") / "gpu-run"), "--i-accept-gpu-risk"])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
