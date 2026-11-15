"""CLI imports are portable; live native dependencies are loaded only on request."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from . import __version__
from .config import defaults, load_config, validate_config
from .storage import load_json, save_json
from .tools import data_root, fetch_tools, resolve_tools


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _require_admin():
    if os.name != "nt":
        raise RuntimeError("Live hardware control supports Windows only; demo is portable")
    import ctypes
    if not ctypes.windll.shell32.IsUserAnAdmin():
        raise RuntimeError("Open an Administrator terminal yourself. This tool does not launch or answer UAC")


def _approval(args, config):
    if not args.i_accept_gpu_risk:
        raise ValueError("Live control requires --i-accept-gpu-risk; crashes, data loss and hardware damage are possible")
    if config["safety"]["max_power_percent"] > 100 and not args.allow_above_default_power:
        raise ValueError("Above-default power also requires --allow-above-default-power and explicit config watt/percent ceilings")
    _require_admin()


def _background(args, config, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "gpu_limitseeker", args.command, "--run-dir", str(run_dir), "--i-accept-gpu-risk"]
    if args.command == "run":
        command += ["--config", str(Path(args.config).resolve())]
    if args.allow_above_default_power:
        command.append("--allow-above-default-power")
    output = run_dir / "controller.log"
    with output.open("ab", buffering=0) as log:
        child = subprocess.Popen(
            command, cwd=run_dir, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            close_fds=True,
            creationflags=(subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
                           | subprocess.CREATE_BREAKAWAY_FROM_JOB),
        )
    receipt = {"status": "launched_not_yet_verified", "pid": child.pid,
               "run_dir": str(run_dir), "log": str(output),
               "note": "An independent console/process was created, not a scheduled task. Check status for actual progress."}
    save_json(run_dir / "launch.json", receipt)
    return receipt


def parser():
    root = argparse.ArgumentParser(prog="gpu-limitseeker", description="AI-reviewed GPU limit exploration; alpha, explicit hardware opt-in")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Clearly labelled offline synthetic experiment; no GPU or AI provider calls")
    demo.add_argument("--powers", default="100,110,125", help="Synthetic approved power points, comma-separated")
    init = commands.add_parser("init-config", help="Write a new JSON configuration; never overwrite an existing file")
    init.add_argument("path", type=Path)
    inspect = commands.add_parser("inspect", help="Explicit read-only local NVIDIA inspection; does not run pressure or persist OC")
    inspect.add_argument("--config", required=True, type=Path)
    for name in ("run", "resume"):
        command = commands.add_parser(name, help="Explicit live AI-reviewed control" if name == "run" else "Resume a known round-boundary checkpoint, never unknown pressure")
        if name == "run":
            command.add_argument("--config", required=True, type=Path)
        command.add_argument("--run-dir", type=Path, required=name == "resume")
        command.add_argument("--i-accept-gpu-risk", action="store_true")
        command.add_argument("--allow-above-default-power", action="store_true")
        command.add_argument("--background", action="store_true", help="Windows: create an independent native console and log; no auto-restart or reboot task")
    for name in ("stop", "status"):
        command = commands.add_parser(name, help="Create local STOP signal" if name == "stop" else "Read local checkpoint/report, without GPU calls")
        command.add_argument("--run-dir", required=True, type=Path)
    tools = commands.add_parser("tools", help="Opt-in upstream tool download; never executes downloaded files")
    tool_commands = tools.add_subparsers(dest="tool_command", required=True)
    fetch = tool_commands.add_parser("fetch")
    fetch.add_argument("--accept-downloads", action="store_true")
    fetch.add_argument("--cache", type=Path)
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            from .simulation import run_demo
            powers = [int(value) for value in args.powers.split(",")]
            _print(run_demo(powers))
        elif args.command == "init-config":
            args.path.parent.mkdir(parents=True, exist_ok=True)
            with args.path.open("x", encoding="utf-8") as stream:
                json.dump(defaults(), stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            _print({"created": str(args.path), "note": "Set an explicit AI provider/model/credentials environment; live execution is not enabled automatically"})
        elif args.command == "tools":
            if not args.accept_downloads:
                raise ValueError("Downloads require --accept-downloads; see THIRD_PARTY_NOTICES.md for upstream licenses and hashes")
            _print(fetch_tools(args.cache))
        elif args.command == "inspect":
            config = resolve_tools(load_config(args.config))
            from .hardware import WindowsNvidiaBackend
            # inspect must remain read-only even if handoff is enabled in a run config.
            config["hardware"]["allow_afterburner_handoff"] = False
            backend = WindowsNvidiaBackend(config, data_root() / "inspection")
            try:
                _print(backend.inspect())
            finally:
                backend.close()
        elif args.command in ("run", "resume"):
            if args.command == "run":
                config = load_config(args.config)
                run_dir = args.run_dir or data_root() / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            else:
                run_dir = args.run_dir.expanduser().resolve()
                config = validate_config(load_json(run_dir / "state.json")["config"])
            run_dir = run_dir.expanduser().resolve()
            _approval(args, config)
            if args.background:
                _print(_background(args, config, run_dir))
            else:
                from .controller import run
                _print(run(config, run_dir, resume=args.command == "resume"))
        elif args.command == "stop":
            run_dir = args.run_dir.expanduser().resolve(strict=True)
            stop = run_dir / "STOP"
            try:
                with stop.open("x", encoding="ascii") as stream:
                    stream.write("Operator stop requested\n")
            except FileExistsError:
                pass
            _print({"stop_requested": True, "run_dir": str(run_dir), "note": "No GPU query/write occurred in this command; controller owns bounded child shutdown"})
        elif args.command == "status":
            run_dir = args.run_dir.expanduser().resolve(strict=True)
            if (run_dir / "state.json").exists():
                state = load_json(run_dir / "state.json")
                _print({key: state.get(key) for key in ("status", "phase", "elapsed_seconds", "recoveries", "pending_trial", "updated_at", "error")})
            else:
                _print(load_json(run_dir / "launch.json"))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
