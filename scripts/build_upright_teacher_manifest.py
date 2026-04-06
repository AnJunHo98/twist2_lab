from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import yaml

try:
    from twist2_lab.motions.path_utils import resolve_motion_file_path
except ModuleNotFoundError:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(_PROJECT_ROOT / "source" / "twist2_lab"))
    from twist2_lab.motions.path_utils import resolve_motion_file_path


class _LegacyNumpyUnpickler(pickle.Unpickler):
    MODULE_ALIASES = {
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.umath": "numpy.core.umath",
    }

    def find_class(self, module: str, name: str):
        return super().find_class(self.MODULE_ALIASES.get(module, module), name)


def _load_motion_pickle(path: Path) -> dict:
    with path.open("rb") as file_obj:
        return _LegacyNumpyUnpickler(file_obj).load()


def _classify_motion(file_name: str, root_xy_delta: float, root_z_span: float, max_root_speed: float) -> tuple[str, int]:
    lower_name = file_name.lower()
    if re.search(r"run|jog", lower_name):
        return "upright-run", 4
    if re.search(r"jump|dance|kick|punch|cartwheel|flip", lower_name):
        return "upright-dynamic", 4
    if re.search(r"walk|stair|step", lower_name):
        return "upright-walk", 3
    if root_xy_delta >= 3.0 or max_root_speed >= 2.5:
        return "upright-travel", 4
    if root_xy_delta >= 0.6 or max_root_speed >= 1.0:
        return "upright-step", 2
    if root_z_span >= 0.2:
        return "upright-dynamic", 3
    return "upright-static", 1


def build_manifest(
    source_manifest: Path,
    output_manifest: Path,
    *,
    upright_min_root_z: float,
    exclude_pattern: str,
) -> tuple[int, int, int]:
    manifest_data = yaml.safe_load(source_manifest.read_text(encoding="utf-8"))
    motions = manifest_data["motions"]
    root_path_value = str(manifest_data["root_path"])
    exclude_regex = re.compile(exclude_pattern, re.IGNORECASE)

    kept_motions: list[dict] = []
    removed_by_name = 0
    removed_by_height = 0

    for motion_entry in motions:
        file_name = str(motion_entry["file"])
        if exclude_regex.search(file_name):
            removed_by_name += 1
            continue

        motion_data = _load_motion_pickle(resolve_motion_file_path(source_manifest, root_path_value, file_name))
        root_pos = motion_data["root_pos"]
        fps = float(motion_data["fps"])

        min_root_z = float(root_pos[:, 2].min())
        if min_root_z < upright_min_root_z:
            removed_by_height += 1
            continue

        root_xy_delta = float(((root_pos[-1, :2] - root_pos[0, :2]) ** 2).sum() ** 0.5)
        root_z_span = float(root_pos[:, 2].max() - root_pos[:, 2].min())
        max_root_speed = 0.0
        if len(root_pos) >= 2:
            dt = max(1.0 / fps, 1.0e-6)
            max_root_speed = float((((root_pos[1:, :2] - root_pos[:-1, :2]) / dt) ** 2).sum(axis=1).max() ** 0.5)

        motion_group, curriculum_level = _classify_motion(file_name, root_xy_delta, root_z_span, max_root_speed)
        difficulty_score = min(10.0, 1.0 + 0.35 * root_xy_delta + 1.5 * root_z_span + 0.2 * max_root_speed)

        kept_entry = dict(motion_entry)
        kept_entry["motion_group"] = motion_group
        kept_entry["curriculum_level"] = curriculum_level
        kept_entry["difficulty_score"] = round(float(difficulty_score), 4)
        kept_motions.append(kept_entry)

    output_data = {
        "root_path": root_path_value,
        "motions": kept_motions,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        yaml.safe_dump(output_data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return len(kept_motions), removed_by_name, removed_by_height


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an upright-only motion manifest for the base teacher.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("source/twist2_lab/data/motion_data_configs/twist2_local_dataset.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("source/twist2_lab/data/motion_data_configs/twist2_teacher_upright_dataset.yaml"),
    )
    parser.add_argument("--upright-min-root-z", type=float, default=0.7)
    parser.add_argument("--exclude-pattern", type=str, default=r"lay|sit|floor|kneel|crawl")
    args = parser.parse_args()

    kept, removed_by_name, removed_by_height = build_manifest(
        args.source,
        args.output,
        upright_min_root_z=args.upright_min_root_z,
        exclude_pattern=args.exclude_pattern,
    )
    print(
        f"[upright-manifest] kept={kept} removed_by_name={removed_by_name} "
        f"removed_by_height={removed_by_height} output={args.output}"
    )


if __name__ == "__main__":
    main()
