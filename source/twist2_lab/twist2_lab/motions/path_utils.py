from __future__ import annotations

import os
from pathlib import Path


def _resolve_configured_motion_root(manifest_path: Path, root_path_value: str) -> Path:
    root_path = Path(os.path.expandvars(root_path_value)).expanduser()
    if root_path.is_absolute():
        return root_path
    return (manifest_path.parent / root_path).resolve()


def _iter_motion_root_candidates(manifest_path: Path, root_path_value: str):
    manifest_path = manifest_path.expanduser().resolve()
    configured_root = _resolve_configured_motion_root(manifest_path, root_path_value)

    seen: set[Path] = set()

    def _yield_unique(path: Path):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved

    yield from _yield_unique(configured_root)

    bundled_motion_root = (manifest_path.parent.parent / "motions" / configured_root.name).resolve()
    yield from _yield_unique(bundled_motion_root)


def get_motion_root_candidates(manifest_path: Path, root_path_value: str) -> list[Path]:
    return list(_iter_motion_root_candidates(manifest_path, root_path_value))


def resolve_motion_file_path(manifest_path: Path, root_path_value: str, motion_file: str) -> Path:
    motion_relative_path = Path(motion_file)
    candidates = [root_path / motion_relative_path for root_path in _iter_motion_root_candidates(manifest_path, root_path_value)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted_paths = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find motion file '{motion_file}' for manifest '{manifest_path}'. "
        f"Tried the following paths:\n{attempted_paths}"
    )
