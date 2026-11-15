"""Opt-in pinned upstream downloads. Resolving a path never executes a tool."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile

GAO_VERSION = "0.4.9"
MEMTEST_VERSION = "0.5.0"
GAO_URL = "https://github.com/Rovey/gpu-auto-optimizer/releases/download/v0.4.9/GpuAutoOptimizer-v0.4.9-win-x64.zip"
GAO_ARCHIVE_SHA256 = "ec276d6c870cbd225d43d90591ab1b5e829d05925503c4c71a4d92de2ddcccb8"
GAO_SHA256 = "f2390c2113329e8a72682fddca377d6f44a51d019f4f3c3366c45fcaba45ee7e"
MEMTEST_URL = "https://github.com/GpuZelenograd/memtest_vulkan/releases/download/v0.5.0/memtest_vulkan-v0.5.0.exe"
MEMTEST_SHA256 = "09e67704210762ae8d8ad70fe6d71275eefbe815cc2c0658cd5afc8e231d0df6"
MEMTEST_LICENSE_URL = "https://raw.githubusercontent.com/GpuZelenograd/memtest_vulkan/v0.5.0/LICENSE"
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024


def data_root() -> Path:
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        if not root:
            raise RuntimeError("LOCALAPPDATA is required for local-only runtime storage")
        return Path(root) / "GPU-LimitSeeker"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "gpu-limitseeker"


def tool_paths(cache: Path | None = None) -> dict[str, Path]:
    root = Path(cache) if cache else data_root() / "tools"
    return {
        "gao_path": root / f"gao-{GAO_VERSION}" / "gao.exe",
        "memtest_path": root / f"memtest-vulkan-{MEMTEST_VERSION}" / f"memtest_vulkan-v{MEMTEST_VERSION}.exe",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> Path:
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink() or sha256(path) != expected:
        raise ValueError(f"Pinned executable hash mismatch: {path.name}")
    return path


def resolve_tools(config: dict, cache: Path | None = None) -> dict:
    cfg = deepcopy(config)
    cached = tool_paths(cache)
    for key, digest in (("gao_path", GAO_SHA256), ("memtest_path", MEMTEST_SHA256)):
        path = Path(cfg["hardware"][key]) if cfg["hardware"][key] else cached[key]
        if not path.is_file():
            raise ValueError(f"Missing {key}; supply an explicit path or run 'gpu-limitseeker tools fetch --accept-downloads'")
        cfg["hardware"][key] = str(verify(path, digest))
    return cfg


def _download(url: str, target: Path, expected: str | None = None) -> None:
    request = Request(url, headers={"User-Agent": "GPU-LimitSeeker/0.1"})
    with urlopen(request, timeout=60) as response, target.open("wb") as output:
        final = urlsplit(response.geturl())
        if final.scheme != "https" or not (final.hostname == "github.com" or (final.hostname or "").endswith(".githubusercontent.com")):
            raise ValueError("Refusing a download redirected outside HTTPS GitHub hosting")
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError("Upstream download exceeds size limit")
            output.write(chunk)
    if expected and sha256(target) != expected:
        raise ValueError("Upstream download did not match the pinned SHA-256")


def _publish(source: Path, destination: Path, expected: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected:
            verify(destination, expected)
        elif destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Refusing to overwrite a different cached notice: {destination.name}")
        return
    # Exclusive creation avoids replacing an operator's unrelated file.
    created = False
    try:
        with source.open("rb") as incoming:
            with destination.open("xb") as outgoing:
                created = True
                for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                    outgoing.write(chunk)
    except BaseException:
        # Only this call's newly created partial file is eligible for deletion.
        if created and destination.exists() and (not expected or sha256(destination) != expected):
            destination.unlink()
        raise
    if expected:
        verify(destination, expected)


def fetch_tools(cache: Path | None = None) -> dict:
    """Download only. Caller must have obtained explicit download authorization."""
    paths = tool_paths(cache)
    root = paths["gao_path"].parent.parent
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="download-", dir=root) as temporary:
        temp = Path(temporary)
        archive = temp / "gao.zip"
        _download(GAO_URL, archive, GAO_ARCHIVE_SHA256)
        with zipfile.ZipFile(archive) as bundle:
            for leaf in ("gao.exe", "LICENSE", "THIRD-PARTY-NOTICES.md"):
                entries = [item for item in bundle.infolist() if not item.is_dir() and item.filename.replace("\\", "/").split("/")[-1] == leaf]
                if len(entries) != 1 or entries[0].file_size > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"Ambiguous or oversized upstream archive entry: {leaf}")
                info = entries[0]
                if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("Archive symlinks are not accepted")
                local = temp / leaf
                with bundle.open(info) as incoming, local.open("wb") as outgoing:
                    total = 0
                    while chunk := incoming.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            raise ValueError("Expanded archive entry exceeds size limit")
                        outgoing.write(chunk)
                if leaf == "gao.exe":
                    verify(local, GAO_SHA256)
                _publish(local, paths["gao_path"].parent / leaf, GAO_SHA256 if leaf == "gao.exe" else None)
        memory = temp / "memtest.exe"
        _download(MEMTEST_URL, memory, MEMTEST_SHA256)
        _publish(memory, paths["memtest_path"], MEMTEST_SHA256)
        license_path = temp / "MEMTEST-LICENSE"
        _download(MEMTEST_LICENSE_URL, license_path)
        _publish(license_path, paths["memtest_path"].parent / "LICENSE")
    return {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
