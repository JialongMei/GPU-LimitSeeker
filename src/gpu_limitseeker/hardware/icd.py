"""Read only active display-driver registry bindings, never scan old driver stores."""
from __future__ import annotations
import json
import os
from pathlib import Path
from .macm_backend import BackendError


def validate_manifest(path: Path, active_libraries: set[Path]) -> Path:
    path = Path(path).resolve(strict=True)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        library = data["ICD"]["library_path"]
        if not isinstance(library, str) or not library or "\x00" in library:
            raise ValueError("invalid library")
        target = Path(library)
        if not target.is_absolute():
            target = path.parent / target
        target = target.resolve(strict=True)
        if target.name.lower() != "nvoglv64.dll" or target not in active_libraries:
            raise ValueError("not the active NVIDIA display library")
        if data.get("file_format_version") not in ("1.0.0", "1.0.1"):
            raise ValueError("unsupported manifest version")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise BackendError(f"Invalid active NVIDIA Vulkan ICD: {exc}") from exc
    return path


def discover_nvidia_icd(explicit=None, *, registry=None) -> Path:
    if registry is None:
        if os.name != "nt":
            raise BackendError("NVIDIA ICD discovery requires Windows")
        import winreg as registry
    libraries = set()
    candidates = set()
    try:
        with registry.OpenKey(registry.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\VIDEO") as key:
            index = 0
            bindings = []
            while True:
                try:
                    name, value, _ = registry.EnumValue(key, index)
                except OSError:
                    break
                index += 1
                if name.startswith(r"\Device\Video") and isinstance(value, str):
                    bindings.append(value)
        prefix = "\\Registry\\Machine\\"
        for binding in bindings:
            if not binding.lower().startswith(prefix.lower()):
                continue
            with registry.OpenKey(registry.HKEY_LOCAL_MACHINE, binding[len(prefix):]) as key:
                try:
                    values, _ = registry.QueryValueEx(key, "OpenGLDriverName")
                except OSError:
                    continue
            for value in values if isinstance(values, list) else [values]:
                if not isinstance(value, str):
                    continue
                library = Path(os.path.expandvars(value))
                if library.name.lower() not in ("nvoglv64", "nvoglv64.dll"):
                    continue
                if library.suffix == "":
                    library = library.with_suffix(".dll")
                if not library.is_absolute():
                    library = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / library
                library = library.resolve(strict=True)
                libraries.add(library)
                candidates.add(library.parent / "nv-vk64.json")
        if not libraries:
            raise BackendError("Active NVIDIA OpenGL/Vulkan driver binding is unavailable")
        try:
            with registry.OpenKey(registry.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\Vulkan\Drivers") as key:
                index = 0
                while True:
                    try:
                        name, enabled, _ = registry.EnumValue(key, index)
                    except OSError:
                        break
                    index += 1
                    if enabled == 0:
                        candidates.add(Path(name))
        except OSError:
            pass
        if explicit:
            return validate_manifest(Path(explicit), libraries)
        valid = set()
        for candidate in candidates:
            try:
                valid.add(validate_manifest(candidate, libraries))
            except (BackendError, OSError):
                continue
        if len(valid) != 1:
            raise BackendError("Active NVIDIA Vulkan ICD is missing or ambiguous; specify its installed manifest")
        return valid.pop()
    except OSError as exc:
        raise BackendError(f"Cannot read active NVIDIA driver registration: {exc}") from exc
