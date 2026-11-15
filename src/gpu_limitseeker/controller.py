"""Single-writer experiment lifecycle; no hardware work at import time."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

from .config import validate_config
from .models import Advice, Bounds, Profile, SafetyError, TrialResult, TrialSpec, finite, integer
from .storage import hardware_writer_lock, load_json, run_lock, save_json
from .tools import data_root, resolve_tools

STATE_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _context(engine, spec, result, remaining):
    summary = deepcopy(engine.summary()) if engine else {"phase": "baseline"}
    # Full evidence stays local. Bound review complexity as sweeps accumulate.
    for name, count in (("axes", 2), ("long_measurements", 4), ("excluded", 4)):
        if isinstance(summary.get(name), list):
            summary[name] = summary[name][-count:]
    return {
        "pending_trial": spec.to_dict() if spec else None,
        "result": result.to_dict() if result else None,
        "search_summary": summary,
        "remaining_seconds": max(0, remaining),
    }


def _device_envelope(device: dict, config: dict) -> tuple[Profile, Bounds]:
    for key in ("uuid", "driver"):
        if type(device.get(key)) is not str or not device[key]:
            raise SafetyError("Device identity is missing")
    initial = Profile.from_dict(device["initial"])
    bounds = Bounds(**device["bounds"])
    bounds.check(initial)
    safety = config["safety"]
    if initial.power_percent != safety["power_percent"]:
        raise SafetyError("Current power differs from the explicitly expected original baseline")
    default_w = finite(device["default_power_w"], "default board power", 1)
    current_w = finite(device["power_w"], "current board power cap", 1)
    if abs(current_w - default_w * initial.power_percent / 100) > 0.6:
        raise SafetyError("Independent current power readback disagrees with baseline")
    ceiling = safety["max_board_power_w"]
    for power in config["search"]["power_limits"]:
        if not bounds.power_min_percent <= power <= bounds.power_max_percent:
            raise SafetyError("Requested power is outside actual driver limits")
        if power > safety["max_power_percent"]:
            raise SafetyError("Requested power exceeds explicit operator approval")
        if ceiling is not None and default_w * power / 100 > ceiling + 1e-6:
            raise SafetyError("Requested power exceeds explicit board-watt ceiling")
    if initial.fan not in config["fan_profiles"]:
        raise SafetyError("Initial profile references an unknown software fan curve")
    return initial, bounds


def run(config: dict, run_dir: Path, *, resume: bool = False,
        backend_factory=None, reviewer_factory=None, engine_factory=None,
        tools_resolver=resolve_tools, clock=time.monotonic) -> dict:
    """Explicit live entrypoint; injected factories are for isolated offline tests.

    A checkpoint interrupted inside pressure or native recovery is ambiguous and is
    never automatically replayed. Resume handles known round boundaries only.
    """
    config = validate_config(config)
    run_dir = Path(run_dir).expanduser().resolve()
    # Runtime evidence must not accidentally enter the source checkout.
    source_root = Path(__file__).resolve().parents[2]
    if run_dir == source_root or source_root in run_dir.parents:
        raise ValueError("Store runtime evidence outside the package/source directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "state.json"
    stop_file = run_dir / "STOP"
    if stop_file.exists():
        raise SafetyError("STOP exists; remove it deliberately before an explicitly requested restart")
    with run_lock(run_dir / "run.lock"), hardware_writer_lock(data_root() / "hardware-writer.lock"):
        if checkpoint.exists() != resume:
            raise ValueError("Use a new run directory, or explicitly resume an existing checkpoint")
        state = load_json(checkpoint) if resume else {
            "version": STATE_VERSION, "run_id": uuid.uuid4().hex,
            "created_at": utc_now(), "config": deepcopy(config),
            "elapsed_seconds": 0.0, "recoveries": 0, "phase": "ready", "status": "starting",
        }
        if state.get("version") != STATE_VERSION or state.get("config") != config:
            raise SafetyError("Checkpoint version/configuration differs; do not reinterpret prior safety approval")
        base_elapsed = finite(state["elapsed_seconds"], "cumulative elapsed")
        recoveries = integer(state["recoveries"], "cumulative recoveries")
        if recoveries > config["safety"]["max_recoveries"]:
            raise SafetyError("Checkpoint recovery budget is invalid")
        if resume and state.get("status") == "completed":
            raise SafetyError("This run already ended; do not silently restart its experiments")
        if resume and state.get("phase") not in ("ready", "awaiting_review"):
            raise SafetyError("Interrupted pressure/recovery has unknown evidence; automatic replay is forbidden")
        if resume and state.get("phase") == "awaiting_review" and state.get("result", {}).get("outcome") == "tdr":
            raise SafetyError("Interrupted TDR handling requires manual recovery review, not automatic replay")
        budget = config["safety"]["max_hours"] * 3600
        started = clock()
        backend = engine = None
        failure = None
        result_to_review = None

        def elapsed():
            return base_elapsed + max(0, clock() - started)

        def remaining():
            return max(0, budget - elapsed())

        def commit():
            state["elapsed_seconds"] = elapsed()
            state["updated_at"] = utc_now()
            if engine is not None:
                state["engine"] = engine.snapshot()
            save_json(checkpoint, state)

        def check():
            if stop_file.exists():
                raise SafetyError("Operator STOP requested")
            if remaining() <= 0:
                raise SafetyError("Original cumulative wall-clock budget exhausted")

        try:
            commit()
            check()
            resolved = tools_resolver(config)
            if reviewer_factory is None:
                from .ai import AIReviewer
                reviewer_factory = AIReviewer
            reviewer = reviewer_factory(resolved, run_dir)
            # Validate a real, fresh AI response BEFORE creating a native backend.
            preflight = reviewer.review(_context(None, None, None, remaining()))
            if preflight.action != "continue":
                raise SafetyError("AI preflight did not explicitly permit continuation")
            check()
            if backend_factory is None:
                from .hardware import WindowsNvidiaBackend
                backend_factory = WindowsNvidiaBackend
            if engine_factory is None:
                from .engine import SearchEngine
                engine_factory = SearchEngine
            resolved["_runtime_remaining_seconds"] = remaining()
            resolved["_runtime_recoveries_used"] = state["recoveries"]
            session_dir = run_dir / "sessions" / uuid.uuid4().hex
            session_dir.mkdir(parents=True, exist_ok=False)
            state["backend_session"] = str(session_dir.relative_to(run_dir))
            backend = backend_factory(resolved, session_dir)
            device = backend.inspect()
            initial, bounds = _device_envelope(device, config)
            if resume and "engine" in state:
                previous = state["device"]
                if (device["uuid"] != previous["uuid"] or device["driver"] != previous["driver"]
                        or device["bounds"] != previous["bounds"]
                        or initial.to_dict() != previous["initial"]):
                    raise SafetyError("Resume requires the same driver/device and independently restored original profile")
                if initial.kind == "curve" and _curve_signature(device) != _curve_signature(previous):
                    raise SafetyError("Original custom curve raw vector differs; prior qualification cannot be reused")
                if state["phase"] == "awaiting_review":
                    result_to_review = TrialResult.from_dict(state["result"])
                engine = engine_factory.restore(state["engine"], pending_result=result_to_review)
                baseline = TrialSpec("resume-" + uuid.uuid4().hex, "baseline", "screen", initial,
                                     **config["durations"]["baseline"])
                if baseline.seconds > remaining():
                    raise SafetyError("Insufficient ORIGINAL budget to recheck the starting profile")
                old_phase = state["phase"]
                state["phase"] = "running_resume_baseline"
                commit()
                observed = backend.run_trial(baseline, stop_file, check=check)
                save_json(run_dir / f"{baseline.id}.json", observed.to_dict())
                if observed.outcome != "pass":
                    raise SafetyError("Original-profile resume recheck did not pass")
                decision = reviewer.review(_context(engine, baseline, observed, remaining()))
                if decision.action != "continue":
                    raise SafetyError("AI did not approve continuation after resume recheck")
                state["phase"] = old_phase
                # Known completed evidence remains pending for its fresh AI review;
                # it is not physically replayed after the resume recheck.
            else:
                state["device"] = device
                resolved["_device_curve"] = device.get("curve_points_data", [])
                engine = engine_factory(initial, bounds, resolved)
            state["status"] = "running"
            prior_errors = {key: state.pop(key) for key in ("error", "restore_error", "cleanup_error") if key in state}
            if prior_errors:
                state.setdefault("prior_attempt_errors", []).append(prior_errors)
            commit()
            while True:
                check()
                if result_to_review is None:
                    spec = engine.next_trial(remaining())
                    if spec is None:
                        summary = engine.summary()
                        state["status"] = "completed" if summary.get("status") == "complete" and summary.get("final_qualified") else "stopped"
                        break
                    state.update(phase="running_trial", pending_trial=spec.to_dict())
                    state.pop("result", None)
                    commit()
                    observed = backend.run_trial(spec, stop_file, check=check)
                    if observed.trial_id != spec.id:
                        raise SafetyError("Backend evidence does not belong to the pending trial")
                    state.update(phase="awaiting_review", result=observed.to_dict())
                    save_json(run_dir / f"{spec.id}.json", observed.to_dict())
                    commit()
                else:
                    spec = TrialSpec.from_dict(state["pending_trial"])
                    observed = result_to_review
                    result_to_review = None
                if observed.outcome in ("hot", "invalid", "stopped"):
                    engine.observe(observed, Advice("stop", "Hard safety gate: no AI override"))
                    state.update(phase="ready", status="stopped")
                    commit()
                    break
                decision = reviewer.review(_context(engine, spec, observed, remaining()))
                engine.observe(observed, decision)
                state.update(phase="ready", advice=decision.to_dict())
                if observed.outcome == "tdr" and decision.action != "stop":
                    # Historical LONG evidence is not a fresh backend's recovery
                    # admission. The exact original baseline passed this session.
                    fallback = initial
                    retired = {Profile.from_dict(item["profile"]).key for item in engine.summary().get("excluded", [])}
                    if (fallback.key == spec.profile.key or fallback.key in retired
                            or state["recoveries"] >= config["safety"]["max_recoveries"]):
                        raise SafetyError("No current-session nonfailed original fallback or cumulative recovery budget remains")
                    check()
                    state["recoveries"] += 1
                    state["phase"] = "recovering"
                    commit()
                    receipt = backend.recover_tdr(fallback, stop_file)
                    state["recovery_receipt"] = receipt
                    state["phase"] = "ready"
                commit()
                if decision.action == "stop":
                    state["status"] = "stopped"
                    break
        except (Exception, KeyboardInterrupt) as exc:
            failure = exc
            state["status"] = "paused" if isinstance(exc, KeyboardInterrupt) or stop_file.exists() else "failed"
            state["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        finally:
            if backend is not None:
                try:
                    state["restore_receipt"] = backend.restore()
                except Exception as exc:
                    state["restore_error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                    state["status"] = "failed"
                    failure = failure or exc
                try:
                    backend.close()
                except Exception as exc:
                    state["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                    state["status"] = "failed"
                    failure = failure or exc
            commit()
            report = {
                "status": state["status"], "elapsed_seconds": state["elapsed_seconds"],
                "recoveries": state["recoveries"], "summary": engine.summary() if engine else None,
                "restore_receipt": state.get("restore_receipt"),
                "restore_error": state.get("restore_error"), "cleanup_error": state.get("cleanup_error"),
                "error": state.get("error"),
                "qualification": "Configured workloads only; not a guarantee of every game or PSU/transient stability",
            }
            save_json(run_dir / "report.json", report)
        if failure:
            raise RuntimeError(f"Run {state['status']}; local evidence: {run_dir / 'report.json'}") from failure
        return report


def _curve_signature(device: dict) -> tuple:
    points = device.get("curve_points_data")
    if type(points) is not list or not points:
        raise SafetyError("A custom baseline requires a complete raw curve snapshot")
    values = []
    for point in points:
        if type(point) is not dict or type(point.get("raw_offset")) is not int:
            raise SafetyError("Malformed raw baseline curve")
        values.append((integer(point["index"], "curve index"),
                       finite(point["mv"], "curve voltage", 1), point["raw_offset"]))
    if len({point[0] for point in values}) != len(values):
        raise SafetyError("Duplicate baseline curve indices")
    return tuple(values)
