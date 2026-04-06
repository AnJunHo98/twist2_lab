#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from twist2_lab.motions.path_utils import resolve_motion_file_path
except ModuleNotFoundError:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(_PROJECT_ROOT / "source" / "twist2_lab"))
    from twist2_lab.motions.path_utils import resolve_motion_file_path


class _LegacyNumpyUnpickler(pickle.Unpickler):
    """Compat loader for motion PKLs saved with alternate NumPy module paths."""

    MODULE_ALIASES = {
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.umath": "numpy.core.umath",
    }

    def find_class(self, module: str, name: str):
        return super().find_class(self.MODULE_ALIASES.get(module, module), name)


def _load_motion_pickle(file_obj) -> dict[str, Any]:
    return _LegacyNumpyUnpickler(file_obj).load()


def _normalize_quat_rows(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.clip(norm, 1.0e-8, None)
    return quat / norm


def _score_quaternion_order(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    local_body_pos: np.ndarray,
    link_body_list: list[str],
    order: str,
) -> float:
    sample_count = min(root_rot.shape[0], 16)
    sample_idx = np.linspace(0, root_rot.shape[0] - 1, num=sample_count, dtype=np.int64)

    try:
        neck_idx = link_body_list.index("Link_Neck_Pitch")
        left_ankle_idx = link_body_list.index("Link_Ankle_Roll_Left")
        right_ankle_idx = link_body_list.index("Link_Ankle_Roll_Right")
    except ValueError:
        return float("-inf")

    sampled_root_pos = root_pos[sample_idx]
    sampled_root_rot = root_rot[sample_idx]
    sampled_local_body_pos = local_body_pos[sample_idx]

    if order == "xyzw":
        x, y, z, w = sampled_root_rot.T
    else:
        w, x, y, z = sampled_root_rot.T

    def world_z(body_idx: int) -> np.ndarray:
        vx, vy, vz = sampled_local_body_pos[:, body_idx].T
        return (
            2.0 * (x * z - y * w) * vx
            + 2.0 * (y * z + x * w) * vy
            + (1.0 - 2.0 * (x * x + y * y)) * vz
            + sampled_root_pos[:, 2]
        )

    neck_z = world_z(neck_idx)
    left_ankle_z = world_z(left_ankle_idx)
    right_ankle_z = world_z(right_ankle_idx)
    return float(np.mean(neck_z - 0.5 * (left_ankle_z + right_ankle_z)))


def _convert_root_rot_to_wxyz(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    local_body_pos: np.ndarray,
    link_body_list: list[str],
) -> np.ndarray:
    score_xyzw = _score_quaternion_order(root_pos, root_rot, local_body_pos, link_body_list, "xyzw")
    score_wxyz = _score_quaternion_order(root_pos, root_rot, local_body_pos, link_body_list, "wxyz")
    if score_xyzw + 5.0e-2 >= score_wxyz:
        root_rot = root_rot[:, [3, 0, 1, 2]]
    return _normalize_quat_rows(root_rot)


def _roll_pitch_yaw_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat.T

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(sin_yaw, cos_yaw)

    return np.stack((roll, pitch, yaw), axis=-1)


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _compute_motion_metrics(path: Path, description: str) -> dict[str, Any]:
    with path.open("rb") as f:
        motion_data = _load_motion_pickle(f)

    fps = float(motion_data["fps"])
    dt = 1.0 / fps
    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float64)
    local_body_pos = np.asarray(motion_data["local_body_pos"], dtype=np.float64)
    link_body_list = list(motion_data["link_body_list"])

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Invalid root_pos shape: {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"Invalid root_rot shape: {root_rot.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"Invalid dof_pos shape: {dof_pos.shape}")
    if local_body_pos.ndim != 3 or local_body_pos.shape[2] != 3:
        raise ValueError(f"Invalid local_body_pos shape: {local_body_pos.shape}")
    if root_pos.shape[0] < 2:
        raise ValueError("Motion file needs at least 2 frames.")

    root_rot = _convert_root_rot_to_wxyz(root_pos, root_rot, local_body_pos, link_body_list)
    rpy = _roll_pitch_yaw_from_quat_wxyz(root_rot)
    yaw = np.unwrap(rpy[:, 2])

    root_delta = np.diff(root_pos, axis=0)
    root_vel = root_delta / dt
    root_speed_xy = np.linalg.norm(root_vel[:, :2], axis=1)
    root_speed_z = np.abs(root_vel[:, 2])
    root_path_xy = float(np.sum(np.linalg.norm(root_delta[:, :2], axis=1)))
    root_disp_xy = float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]))

    yaw_delta = np.diff(yaw)
    yaw_rate = np.abs(yaw_delta / dt)
    yaw_span_abs = float(np.max(yaw) - np.min(yaw))

    dof_vel = np.diff(dof_pos, axis=0) / dt
    dof_speed = np.abs(dof_vel)
    dof_acc = np.diff(dof_vel, axis=0) / dt if dof_vel.shape[0] > 1 else np.zeros((0, dof_pos.shape[1]))
    dof_acc_abs = np.abs(dof_acc)
    joint_range = np.max(dof_pos, axis=0) - np.min(dof_pos, axis=0)

    local_body_vel = np.diff(local_body_pos, axis=0) / dt
    local_body_speed = np.linalg.norm(local_body_vel, axis=-1)

    return {
        "file": str(path),
        "description": description,
        "num_frames": int(root_pos.shape[0]),
        "fps": fps,
        "duration_s": float((root_pos.shape[0] - 1) * dt),
        "root_height_min": float(np.min(root_pos[:, 2])),
        "root_height_median": float(np.median(root_pos[:, 2])),
        "root_height_max": float(np.max(root_pos[:, 2])),
        "root_height_range": float(np.max(root_pos[:, 2]) - np.min(root_pos[:, 2])),
        "root_path_xy": root_path_xy,
        "root_disp_xy": root_disp_xy,
        "root_speed_xy_mean": _safe_mean(root_speed_xy),
        "root_speed_xy_p95": _safe_percentile(root_speed_xy, 95.0),
        "root_speed_z_p95": _safe_percentile(root_speed_z, 95.0),
        "yaw_rate_p95": _safe_percentile(yaw_rate, 95.0),
        "yaw_span_abs": yaw_span_abs,
        "joint_vel_p95": _safe_percentile(dof_speed, 95.0),
        "joint_acc_p95": _safe_percentile(dof_acc_abs, 95.0),
        "joint_range_mean": _safe_mean(joint_range),
        "joint_range_p95": _safe_percentile(joint_range, 95.0),
        "body_motion_p95": _safe_percentile(local_body_speed, 95.0),
    }


def _robust_normalize(values: np.ndarray, low_q: float = 5.0, high_q: float = 95.0) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.percentile(values, low_q))
    high = float(np.percentile(values, high_q))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low + 1.0e-8:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _assign_levels(scores: np.ndarray, num_levels: int) -> np.ndarray:
    if scores.size == 0:
        return np.zeros(0, dtype=np.int64)
    if num_levels <= 1:
        return np.ones(scores.shape[0], dtype=np.int64)

    quantiles = np.linspace(0.0, 1.0, num_levels + 1)[1:-1]
    edges = np.quantile(scores, quantiles)
    levels = np.searchsorted(edges, scores, side="right") + 1
    return levels.astype(np.int64)


def _derive_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def _contains_any(name: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in name for keyword in keywords)


def _classify_motion_group(manifest_file: str, row: dict[str, Any]) -> str:
    name = manifest_file.lower()
    root_disp_xy = float(row["root_disp_xy"])
    root_speed_xy_p95 = float(row["root_speed_xy_p95"])
    yaw_span_abs = float(row["yaw_span_abs"])
    yaw_rate_p95 = float(row["yaw_rate_p95"])
    joint_vel_p95 = float(row["joint_vel_p95"])
    body_motion_p95 = float(row["body_motion_p95"])
    crouch_depth = float(row["crouch_depth"])
    root_height_min = float(row["root_height_min"])
    locomotion_score = float(row["locomotion_score"])
    rotation_score = float(row["rotation_score"])
    articulation_score = float(row["articulation_score"])
    posture_score = float(row["posture_score"])

    floor_keywords = (
        "sit",
        "sitting",
        "chair",
        "sofa",
        "lie",
        "lying",
        "lay",
        "floor",
        "crawl",
        "prone",
        "supine",
        "rollover",
        "roll_over",
        "getup",
        "get_up",
        "get-up",
    )
    recovery_keywords = (
        "recovery",
        "push_recovery",
        "balance",
        "beam",
        "gap",
        "step_over",
        "stepover",
        "stone",
        "obstacle",
        "perturb",
        "stumble",
        "trip",
    )
    jump_keywords = (
        "jump",
        "jumping",
        "hop",
        "hopping",
        "skip",
        "leap",
        "longjump",
        "cartwheel",
        "kick",
        "punch",
        "boxing",
        "parkour",
    )
    crouch_keywords = (
        "crouch",
        "squat",
        "kneel",
        "kneeling",
        "bend",
        "stoop",
        "duck",
    )
    run_keywords = ("run", "running", "jog", "sprint")
    turn_keywords = (
        "turn",
        "circle",
        "clockwise",
        "counterclockwise",
        "anticlockwise",
        "fig8",
        "s_shape",
        "sshape",
    )
    walk_keywords = ("walk", "walking", "stroll", "forward", "backward", "backwards", "straight")
    upper_body_keywords = (
        "wave",
        "wipe",
        "guitar",
        "shower",
        "dry_",
        "stir",
        "golf",
        "putting",
        "hair",
        "book",
        "cards",
        "football",
        "talking",
        "whispering",
        "physicalexamination",
        "shake_arms",
        "washing",
        "brush",
        "clean",
    )

    if _contains_any(name, ("running_on_spot", "on_the_spot", "treadmill")):
        return "in_place"
    if _contains_any(name, floor_keywords):
        return "floor"
    if _contains_any(name, recovery_keywords):
        return "recovery"
    if _contains_any(name, jump_keywords):
        return "jump"
    if _contains_any(name, crouch_keywords) or crouch_depth > 0.18 or root_height_min < 0.78:
        return "crouch"
    if _contains_any(name, run_keywords) or root_speed_xy_p95 > 2.5:
        if root_disp_xy < 0.3 and joint_vel_p95 > 0.8:
            return "in_place"
        return "run"
    if _contains_any(name, turn_keywords) or (yaw_span_abs > 1.2 and root_disp_xy > 0.2) or rotation_score > 0.75:
        return "turn"
    if _contains_any(name, walk_keywords) or (root_disp_xy > 0.75 and root_speed_xy_p95 > 0.25 and locomotion_score > 0.2):
        return "walk"
    if _contains_any(name, upper_body_keywords):
        return "upper_body"
    if root_disp_xy < 0.2:
        if articulation_score > 0.45 or body_motion_p95 > 1.2 or joint_vel_p95 > 2.0:
            return "in_place"
        return "upper_body"
    if posture_score > 0.6:
        return "crouch"
    if locomotion_score > 0.5 and rotation_score > 0.5:
        return "turn"
    if articulation_score > 0.6 and root_speed_xy_p95 > 1.0:
        return "mixed_dynamic"
    return "mixed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate a motion manifest with curriculum levels inferred from PKL motion statistics.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("source/twist2_lab/data/motion_data_configs/twist2_local_dataset.yaml"),
        help="Path to the input motion manifest YAML.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Path to the annotated YAML output. Defaults to '<manifest>_curriculum.yaml'.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to the metrics CSV output. Defaults to '<manifest>_curriculum_metrics.csv'.",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=5,
        help="Number of curriculum levels to assign. Level 1 is easiest.",
    )
    parser.add_argument(
        "--max-motions",
        type=int,
        default=None,
        help="Optional cap on the number of motions to process for a quick smoke run.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=250,
        help="Print a progress line every N processed motions.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve() if args.output_manifest else _derive_output_path(
        manifest_path, "_curriculum"
    )
    output_csv = args.output_csv.expanduser().resolve() if args.output_csv else manifest_path.with_name(
        f"{manifest_path.stem}_curriculum_metrics.csv"
    )

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    motion_entries = list(manifest["motions"])
    if args.max_motions is not None:
        motion_entries = motion_entries[: args.max_motions]

    metrics_rows: list[dict[str, Any]] = []
    failed_files: list[tuple[str, str]] = []

    total = len(motion_entries)
    for index, entry in enumerate(motion_entries, start=1):
        motion_path = resolve_motion_file_path(manifest_path, str(manifest["root_path"]), str(entry["file"]))
        description = str(entry.get("description", ""))
        try:
            row = _compute_motion_metrics(motion_path, description)
            row["manifest_file"] = str(entry["file"])
            row["base_weight"] = float(entry.get("weight", 1.0))
            metrics_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            failed_files.append((str(entry["file"]), str(exc)))

        if args.progress_interval > 0 and (index % args.progress_interval == 0 or index == total):
            print(f"[{index}/{total}] processed motions")

    if not metrics_rows:
        raise RuntimeError("No valid motions were processed.")

    root_height_reference = float(np.percentile([row["root_height_median"] for row in metrics_rows], 90.0))
    for row in metrics_rows:
        row["crouch_depth"] = max(root_height_reference - row["root_height_min"], 0.0)
        row["path_rate_xy"] = row["root_path_xy"] / max(row["duration_s"], 1.0e-6)

    locomotion_raw = np.array(
        [
            0.45 * row["root_speed_xy_p95"] + 0.15 * row["root_speed_z_p95"] + 0.25 * row["path_rate_xy"] + 0.15 * row["root_disp_xy"]
            for row in metrics_rows
        ],
        dtype=np.float64,
    )
    rotation_raw = np.array(
        [0.60 * row["yaw_rate_p95"] + 0.40 * row["yaw_span_abs"] for row in metrics_rows],
        dtype=np.float64,
    )
    articulation_raw = np.array(
        [0.45 * row["joint_vel_p95"] + 0.35 * row["joint_acc_p95"] + 0.20 * row["body_motion_p95"] for row in metrics_rows],
        dtype=np.float64,
    )
    posture_raw = np.array(
        [0.65 * row["crouch_depth"] + 0.35 * row["root_height_range"] for row in metrics_rows],
        dtype=np.float64,
    )

    locomotion_score = _robust_normalize(locomotion_raw)
    rotation_score = _robust_normalize(rotation_raw)
    articulation_score = _robust_normalize(articulation_raw)
    posture_score = _robust_normalize(posture_raw)
    difficulty_score = (
        0.32 * locomotion_score
        + 0.18 * rotation_score
        + 0.32 * articulation_score
        + 0.18 * posture_score
    )
    levels = _assign_levels(difficulty_score, args.num_levels)
    score_percentile = np.argsort(np.argsort(difficulty_score)).astype(np.float64)
    if difficulty_score.shape[0] > 1:
        score_percentile /= difficulty_score.shape[0] - 1
    else:
        score_percentile[:] = 0.0

    for index, row in enumerate(metrics_rows):
        row["locomotion_score"] = float(locomotion_score[index])
        row["rotation_score"] = float(rotation_score[index])
        row["articulation_score"] = float(articulation_score[index])
        row["posture_score"] = float(posture_score[index])
        row["difficulty_score"] = float(difficulty_score[index])
        row["difficulty_percentile"] = float(score_percentile[index])
        row["curriculum_level"] = int(levels[index])
        row["motion_group"] = _classify_motion_group(row["manifest_file"], row)

    metrics_by_file = {row["manifest_file"]: row for row in metrics_rows}

    annotated_manifest = dict(manifest)
    annotated_motions = []
    for entry in manifest["motions"]:
        annotated_entry = dict(entry)
        metrics = metrics_by_file.get(str(entry["file"]))
        if metrics is not None:
            annotated_entry["difficulty_score"] = round(metrics["difficulty_score"], 6)
            annotated_entry["curriculum_level"] = int(metrics["curriculum_level"])
            annotated_entry["motion_group"] = str(metrics["motion_group"])
        annotated_motions.append(annotated_entry)
    annotated_manifest["motions"] = annotated_motions

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_manifest.open("w", encoding="utf-8") as f:
        yaml.safe_dump(annotated_manifest, f, sort_keys=False, allow_unicode=False)

    csv_fieldnames = [
        "manifest_file",
        "description",
        "base_weight",
        "num_frames",
        "fps",
        "duration_s",
        "root_height_min",
        "root_height_median",
        "root_height_max",
        "root_height_range",
        "crouch_depth",
        "root_path_xy",
        "root_disp_xy",
        "path_rate_xy",
        "root_speed_xy_mean",
        "root_speed_xy_p95",
        "root_speed_z_p95",
        "yaw_rate_p95",
        "yaw_span_abs",
        "joint_vel_p95",
        "joint_acc_p95",
        "joint_range_mean",
        "joint_range_p95",
        "body_motion_p95",
        "locomotion_score",
        "rotation_score",
        "articulation_score",
        "posture_score",
        "difficulty_score",
        "difficulty_percentile",
        "curriculum_level",
        "motion_group",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in sorted(metrics_rows, key=lambda item: item["difficulty_score"]):
            writer.writerow({key: row[key] for key in csv_fieldnames})

    print(f"Annotated manifest written to: {output_manifest}")
    print(f"Metrics CSV written to: {output_csv}")
    print(f"Root height reference: {root_height_reference:.4f} m")

    level_counts: dict[int, int] = {}
    for level in levels.tolist():
        level_counts[level] = level_counts.get(level, 0) + 1
    level_summary = ", ".join(f"L{level}: {count}" for level, count in sorted(level_counts.items()))
    print(f"Level counts: {level_summary}")

    group_counts = Counter(str(row["motion_group"]) for row in metrics_rows)
    group_summary = ", ".join(f"{group}: {count}" for group, count in sorted(group_counts.items()))
    print(f"Group counts: {group_summary}")

    easiest = sorted(metrics_rows, key=lambda item: item["difficulty_score"])[:5]
    hardest = sorted(metrics_rows, key=lambda item: item["difficulty_score"], reverse=True)[:5]
    print("Easiest motions:")
    for row in easiest:
        print(f"  L{row['curriculum_level']} score={row['difficulty_score']:.3f} {row['manifest_file']}")
    print("Hardest motions:")
    for row in hardest:
        print(f"  L{row['curriculum_level']} score={row['difficulty_score']:.3f} {row['manifest_file']}")

    if failed_files:
        print(f"Skipped {len(failed_files)} invalid motions:")
        for file_name, error in failed_files[:20]:
            print(f"  {file_name}: {error}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")


if __name__ == "__main__":
    main()
