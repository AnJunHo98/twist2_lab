from __future__ import annotations

import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse, quat_conjugate, quat_mul

from .path_utils import resolve_motion_file_path


class _LegacyNumpyUnpickler(pickle.Unpickler):
    """Compat loader for motion PKLs saved with alternate NumPy module paths."""

    MODULE_ALIASES = {
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.umath": "numpy.core.umath",
    }

    def find_class(self, module: str, name: str):
        return super().find_class(self.MODULE_ALIASES.get(module, module), name)


def _load_motion_pickle(file_obj) -> dict:
    return _LegacyNumpyUnpickler(file_obj).load()


def _resolve_storage_dtype(storage_dtype: str | torch.dtype | None) -> torch.dtype:
    if storage_dtype is None:
        return torch.float32
    if isinstance(storage_dtype, torch.dtype):
        return storage_dtype
    if not hasattr(torch, storage_dtype):
        raise ValueError(f"Unsupported motion storage dtype: {storage_dtype}")
    resolved = getattr(torch, storage_dtype)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unsupported motion storage dtype: {storage_dtype}")
    return resolved


def _batch_slerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)
    dot = torch.sum(q0 * q1, dim=-1)
    q1 = q1.clone()
    neg_mask = dot < 0.0
    q1[neg_mask] = -q1[neg_mask]
    dot = torch.abs(dot).clamp(max=1.0)

    blend = blend.unsqueeze(-1)
    linear_mask = dot > 0.9995
    linear_out = F.normalize((1.0 - blend) * q0 + blend * q1, dim=-1)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp(min=1.0e-6)
    theta = theta_0.unsqueeze(-1) * blend
    sin_theta = torch.sin(theta)

    s0 = torch.cos(theta) - dot.unsqueeze(-1) * sin_theta / sin_theta_0.unsqueeze(-1)
    s1 = sin_theta / sin_theta_0.unsqueeze(-1)
    slerp_out = s0 * q0 + s1 * q1
    slerp_out = F.normalize(slerp_out, dim=-1)

    return torch.where(linear_mask.unsqueeze(-1), linear_out, slerp_out)


def _roll_pitch_yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sin_pitch, min=-1.0, max=1.0))

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    return torch.stack((roll, pitch, yaw), dim=-1)


class PklMotionLibrary:
    _QUAT_ORDER_SCORE_MARGIN = 5.0e-2

    def __init__(
        self,
        motion_file: str,
        device: str | torch.device,
        storage_device: str | torch.device | None = None,
        storage_dtype: str | torch.dtype | None = None,
    ):
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device is not None else self.device
        self.storage_dtype = _resolve_storage_dtype(storage_dtype)
        self._load_motions(motion_file)

    def _fetch_motion_files(self, motion_file: str) -> tuple[list[str], list[float], list[int], list[float], list[str]]:
        if motion_file.endswith(".yaml"):
            manifest_path = Path(motion_file).expanduser().resolve()
            with manifest_path.open("r", encoding="utf-8") as f:
                motion_config = yaml.safe_load(f)

            motion_files = [
                str(resolve_motion_file_path(manifest_path, str(motion_config["root_path"]), str(entry["file"])))
                for entry in motion_config["motions"]
            ]
            motion_weights = [float(entry.get("weight", 1.0)) for entry in motion_config["motions"]]
            motion_curriculum_levels = [max(1, int(entry.get("curriculum_level", 1))) for entry in motion_config["motions"]]
            motion_difficulty_scores = [float(entry.get("difficulty_score", 0.0)) for entry in motion_config["motions"]]
            motion_groups = [str(entry.get("motion_group", "default")) for entry in motion_config["motions"]]
            return motion_files, motion_weights, motion_curriculum_levels, motion_difficulty_scores, motion_groups
        return [motion_file], [1.0], [1], [0.0], ["default"]

    def _compute_angular_velocity(self, root_rot: torch.Tensor, dt: float) -> torch.Tensor:
        if root_rot.shape[0] < 2:
            return torch.zeros(root_rot.shape[0], 3, device=root_rot.device, dtype=root_rot.dtype)
        delta_rot = quat_mul(root_rot[1:], quat_conjugate(root_rot[:-1]))
        angular_velocity = axis_angle_from_quat(delta_rot) / dt
        return torch.cat((angular_velocity, angular_velocity[-1:]), dim=0)

    def _score_quaternion_order(
        self,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        local_body_pos: torch.Tensor,
        link_body_list: list[str],
        order: str,
    ) -> float:
        sample_count = min(root_rot.shape[0], 16)
        sample_idx = torch.linspace(0, root_rot.shape[0] - 1, steps=sample_count, device=root_rot.device).long()

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
            x, y, z, w = sampled_root_rot.unbind(dim=-1)
        else:
            w, x, y, z = sampled_root_rot.unbind(dim=-1)

        def _world_z(body_idx: int) -> torch.Tensor:
            vx, vy, vz = sampled_local_body_pos[:, body_idx].unbind(dim=-1)
            return (
                2.0 * (x * z - y * w) * vx
                + 2.0 * (y * z + x * w) * vy
                + (1.0 - 2.0 * (x * x + y * y)) * vz
                + sampled_root_pos[:, 2]
            )

        neck_z = _world_z(neck_idx)
        left_ankle_z = _world_z(left_ankle_idx)
        right_ankle_z = _world_z(right_ankle_idx)
        upright_score = torch.mean(neck_z - 0.5 * (left_ankle_z + right_ankle_z))
        return float(upright_score.item())

    def _convert_root_rot_to_wxyz(
        self,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        local_body_pos: torch.Tensor,
        link_body_list: list[str],
    ) -> tuple[torch.Tensor, str]:
        score_xyzw = self._score_quaternion_order(root_pos, root_rot, local_body_pos, link_body_list, "xyzw")
        score_wxyz = self._score_quaternion_order(root_pos, root_rot, local_body_pos, link_body_list, "wxyz")

        if score_xyzw + self._QUAT_ORDER_SCORE_MARGIN >= score_wxyz:
            return root_rot[:, [3, 0, 1, 2]], "xyzw"
        return root_rot, "wxyz"

    def _load_motions(self, motion_file: str) -> None:
        motion_files, motion_weights, motion_curriculum_levels, motion_difficulty_scores, motion_groups = self._fetch_motion_files(
            motion_file
        )
        skipped_motions: list[tuple[str, str]] = []
        quat_order_counts = {"xyzw": 0, "wxyz": 0}

        self._motion_names = []
        self._motion_weights = []
        self._motion_curriculum_levels = []
        self._motion_manifest_difficulty_scores = []
        self._motion_group_names = []
        self._motion_lengths = []
        self._motion_num_frames = []
        self._motion_root_delta = []
        self._motion_root_pos_delta_local = []
        self._motion_root_rot_delta_local = []
        root_pos_list = []
        root_rot_list = []
        root_vel_list = []
        root_ang_vel_list = []
        dof_pos_list = []
        dof_vel_list = []
        local_body_pos_list = []
        self._body_link_list: list[str] | None = None

        for curr_file, curr_weight, curr_level, curr_score, curr_group in zip(
            motion_files, motion_weights, motion_curriculum_levels, motion_difficulty_scores, motion_groups, strict=True
        ):
            path = Path(curr_file)
            try:
                if not path.is_file():
                    raise FileNotFoundError(f"Missing motion file: {path}")

                with path.open("rb") as f:
                    motion_data = _load_motion_pickle(f)

                fps = float(motion_data["fps"])
                root_pos = torch.tensor(motion_data["root_pos"], dtype=torch.float32, device=self.storage_device)
                root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float32, device=self.storage_device)
                dof_pos = torch.tensor(motion_data["dof_pos"], dtype=torch.float32, device=self.storage_device)
                local_body_pos = torch.tensor(motion_data["local_body_pos"], dtype=torch.float32, device=self.storage_device)
                link_body_list = list(motion_data["link_body_list"])

                if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
                    raise ValueError(f"Invalid root_pos shape in {path}: {root_pos.shape}")
                if root_rot.ndim != 2 or root_rot.shape[-1] != 4:
                    raise ValueError(f"Invalid root_rot shape in {path}: {root_rot.shape}")
                if dof_pos.ndim != 2:
                    raise ValueError(f"Invalid dof_pos shape in {path}: {dof_pos.shape}")
                if local_body_pos.ndim != 3 or local_body_pos.shape[-1] != 3:
                    raise ValueError(f"Invalid local_body_pos shape in {path}: {local_body_pos.shape}")
                if root_pos.shape[0] != root_rot.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
                    raise ValueError(f"Frame count mismatch in {path}")
                if root_pos.shape[0] != local_body_pos.shape[0]:
                    raise ValueError(f"Frame count mismatch between root_pos and local_body_pos in {path}")
                if root_pos.shape[0] < 2:
                    raise ValueError(f"Motion file needs at least 2 frames: {path}")
                if self._body_link_list is None:
                    self._body_link_list = link_body_list
                elif self._body_link_list != link_body_list:
                    raise ValueError(f"Inconsistent link_body_list in {path}")

                root_rot, quat_order = self._convert_root_rot_to_wxyz(root_pos, root_rot, local_body_pos, link_body_list)
                quat_order_counts[quat_order] += 1

                dt = 1.0 / fps
                motion_len = dt * (root_pos.shape[0] - 1)

                root_rot = F.normalize(root_rot, dim=-1)
                root_vel = torch.gradient(root_pos, spacing=dt, dim=0)[0]
                dof_vel = torch.gradient(dof_pos, spacing=dt, dim=0)[0]
                root_ang_vel = self._compute_angular_velocity(root_rot, dt)
                root_delta = root_pos[-1] - root_pos[0]
                root_delta[-1] = 0.0
                root_pos_delta_local = torch.zeros_like(root_pos)
                root_pos_delta_local[1:] = root_pos[1:] - root_pos[:-1]
                root_pos_delta_local[1:] = quat_apply_inverse(root_rot[:-1], root_pos_delta_local[1:])

                root_rot_delta_local = torch.zeros_like(root_pos)
                delta_rot = quat_mul(root_rot[1:], quat_conjugate(root_rot[:-1]))
                delta_rot_euler = _roll_pitch_yaw_from_quat(delta_rot)
                root_rot_delta_local[1:] = quat_apply_inverse(root_rot[:-1], delta_rot_euler)

                self._motion_names.append(path.name)
                self._motion_weights.append(curr_weight)
                self._motion_curriculum_levels.append(curr_level)
                self._motion_manifest_difficulty_scores.append(curr_score)
                self._motion_group_names.append(curr_group)
                self._motion_lengths.append(motion_len)
                self._motion_num_frames.append(root_pos.shape[0])
                self._motion_root_delta.append(root_delta.to(dtype=self.storage_dtype))
                self._motion_root_pos_delta_local.append(root_pos_delta_local.to(dtype=self.storage_dtype))
                self._motion_root_rot_delta_local.append(root_rot_delta_local.to(dtype=self.storage_dtype))
                root_pos_list.append(root_pos.to(dtype=self.storage_dtype))
                root_rot_list.append(root_rot.to(dtype=self.storage_dtype))
                root_vel_list.append(root_vel.to(dtype=self.storage_dtype))
                root_ang_vel_list.append(root_ang_vel.to(dtype=self.storage_dtype))
                dof_pos_list.append(dof_pos.to(dtype=self.storage_dtype))
                dof_vel_list.append(dof_vel.to(dtype=self.storage_dtype))
                local_body_pos_list.append(local_body_pos.to(dtype=self.storage_dtype))
            except Exception as exc:
                skipped_motions.append((str(path), str(exc)))
                continue

        if not self._motion_weights:
            raise RuntimeError(f"No valid motion files were loaded from {motion_file}.")

        if skipped_motions:
            print(f"[MotionLib] Skipped {len(skipped_motions)} invalid motion files.")
            for skipped_path, reason in skipped_motions[:5]:
                print(f"[MotionLib] {skipped_path}: {reason}")
            if len(skipped_motions) > 5:
                print(f"[MotionLib] ... and {len(skipped_motions) - 5} more.")
        print(
            "[MotionLib] Quaternion order detection:"
            f" xyzw->wxyz={quat_order_counts['xyzw']}, kept_wxyz={quat_order_counts['wxyz']}"
        )

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self.device)
        self._motion_weights = self._motion_weights / torch.sum(self._motion_weights)
        self._motion_curriculum_levels = torch.tensor(self._motion_curriculum_levels, dtype=torch.long, device=self.device)
        self._motion_manifest_difficulty_scores = torch.tensor(
            self._motion_manifest_difficulty_scores, dtype=torch.float32, device=self.device
        )
        unique_group_names = sorted(set(self._motion_group_names))
        self._group_name_to_id = {group_name: index for index, group_name in enumerate(unique_group_names)}
        self._motion_group_ids = torch.tensor(
            [self._group_name_to_id[group_name] for group_name in self._motion_group_names],
            dtype=torch.long,
            device=self.device,
        )
        self._group_names = unique_group_names
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float32, device=self.device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, dtype=torch.long, device=self.device)
        self._motion_root_delta = torch.stack(self._motion_root_delta, dim=0)
        self._motion_root_delta = self._motion_root_delta.to(device=self.storage_device, dtype=self.storage_dtype)

        self._motion_root_pos = torch.cat(root_pos_list, dim=0)
        self._motion_root_rot = torch.cat(root_rot_list, dim=0)
        self._motion_root_vel = torch.cat(root_vel_list, dim=0)
        self._motion_root_ang_vel = torch.cat(root_ang_vel_list, dim=0)
        self._motion_dof_pos = torch.cat(dof_pos_list, dim=0)
        self._motion_dof_vel = torch.cat(dof_vel_list, dim=0)
        self._motion_local_body_pos = torch.cat(local_body_pos_list, dim=0)
        self._motion_root_pos_delta_local = torch.cat(self._motion_root_pos_delta_local, dim=0)
        self._motion_root_rot_delta_local = torch.cat(self._motion_root_rot_delta_local, dim=0)

        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)
        self._motion_ids = torch.arange(self.num_motions(), device=self.device, dtype=torch.long)

    @property
    def num_dofs(self) -> int:
        return self._motion_dof_pos.shape[-1]

    def num_motions(self) -> int:
        return int(self._motion_weights.shape[0])

    def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_lengths[motion_ids]

    def get_motion_names(self) -> list[str]:
        return self._motion_names

    def max_curriculum_level(self) -> int:
        return int(self._motion_curriculum_levels.max().item())

    def get_motion_curriculum_levels(self, motion_ids: torch.Tensor | None = None) -> torch.Tensor:
        if motion_ids is None:
            return self._motion_curriculum_levels
        return self._motion_curriculum_levels[motion_ids]

    def get_motion_group_ids(self, motion_ids: torch.Tensor | None = None) -> torch.Tensor:
        if motion_ids is None:
            return self._motion_group_ids
        return self._motion_group_ids[motion_ids]

    def get_motion_group_names(self) -> list[str]:
        return list(self._group_names)

    def get_body_link_names(self) -> list[str]:
        if self._body_link_list is None:
            raise RuntimeError("Motion library body-link names are not available.")
        return self._body_link_list

    def get_key_body_idx(self, key_body_names: list[str] | tuple[str, ...]) -> list[int]:
        body_link_names = self.get_body_link_names()
        return [body_link_names.index(name) for name in key_body_names]

    def sample_motions(
        self,
        n: int,
        motion_difficulty: torch.Tensor | None = None,
        max_curriculum_level: int | None = None,
        frontier_curriculum_level: int | None = None,
        frontier_level_boost: float = 1.0,
        balance_motion_groups: bool = False,
        motion_group_balance_power: float = 1.0,
        max_key_body_error: torch.Tensor | None = None,
        use_error_aware_sampling: bool = False,
        error_sampling_power: float = 5.0,
        error_sampling_threshold: float = 0.15,
    ) -> torch.Tensor:
        motion_prob = self._motion_weights
        allowed_motion_mask = torch.ones_like(motion_prob, dtype=torch.bool)
        if max_curriculum_level is not None:
            allowed_motion_mask = self._motion_curriculum_levels <= int(max_curriculum_level)
            if not torch.any(allowed_motion_mask):
                min_level = int(self._motion_curriculum_levels.min().item())
                allowed_motion_mask = self._motion_curriculum_levels <= min_level
            motion_prob = motion_prob * allowed_motion_mask.float()
        if balance_motion_groups and len(self._group_names) > 1:
            allowed_group_ids = self._motion_group_ids[allowed_motion_mask]
            if allowed_group_ids.numel() > 0:
                allowed_group_weights = self._motion_weights[allowed_motion_mask]
                group_mass = torch.bincount(
                    allowed_group_ids,
                    weights=allowed_group_weights,
                    minlength=len(self._group_names),
                )
                group_balance = torch.ones_like(group_mass)
                positive_mass = group_mass > 0
                group_balance[positive_mass] = torch.pow(group_mass[positive_mass], -float(motion_group_balance_power))
                motion_prob = motion_prob * group_balance[self._motion_group_ids]
        if frontier_curriculum_level is not None and float(frontier_level_boost) > 1.0:
            frontier_mask = allowed_motion_mask & (self._motion_curriculum_levels == int(frontier_curriculum_level))
            if torch.any(frontier_mask):
                motion_prob = motion_prob.clone()
                motion_prob[frontier_mask] *= float(frontier_level_boost)
        if motion_difficulty is not None:
            motion_prob = motion_prob * motion_difficulty
        if use_error_aware_sampling and max_key_body_error is not None:
            error_threshold = max(float(error_sampling_threshold), 1.0e-6)
            error_ratio = torch.clamp(max_key_body_error / error_threshold, min=1.0, max=2.0)
            error_boost = torch.ones_like(motion_prob)
            high_error_mask = max_key_body_error > error_threshold
            error_boost[high_error_mask] = error_ratio[high_error_mask].pow(float(error_sampling_power))
            motion_prob = motion_prob * error_boost

        if torch.sum(motion_prob).item() <= 0.0:
            motion_prob = self._motion_weights
        return torch.multinomial(motion_prob, num_samples=n, replacement=True)

    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        phase = torch.rand(motion_ids.shape, device=self.device)
        return phase * self._motion_lengths[motion_ids]

    def _calc_frame_blend(self, motion_ids: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        phase = torch.clip(times / lengths, 0.0, 1.0)
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0.float()

        frame_start_idx = self._motion_start_idx[motion_ids]
        return frame_idx0 + frame_start_idx, frame_idx1 + frame_start_idx, blend

    def calc_motion_frame(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_loop_num = torch.floor(motion_times / self._motion_lengths[motion_ids])
        motion_times = motion_times - motion_loop_num * self._motion_lengths[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
        motion_ids_storage = motion_ids.to(device=self.storage_device, dtype=torch.long)
        frame_idx0_storage = frame_idx0.to(device=self.storage_device, dtype=torch.long)
        frame_idx1_storage = frame_idx1.to(device=self.storage_device, dtype=torch.long)

        root_pos0 = self._motion_root_pos[frame_idx0_storage].to(device=self.device, dtype=torch.float32)
        root_pos1 = self._motion_root_pos[frame_idx1_storage].to(device=self.device, dtype=torch.float32)
        root_rot0 = self._motion_root_rot[frame_idx0_storage].to(device=self.device, dtype=torch.float32)
        root_rot1 = self._motion_root_rot[frame_idx1_storage].to(device=self.device, dtype=torch.float32)
        dof_pos0 = self._motion_dof_pos[frame_idx0_storage].to(device=self.device, dtype=torch.float32)
        dof_pos1 = self._motion_dof_pos[frame_idx1_storage].to(device=self.device, dtype=torch.float32)

        blend_unsqueezed = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsqueezed) * root_pos0 + blend_unsqueezed * root_pos1
        root_pos = root_pos + motion_loop_num.unsqueeze(-1) * self._motion_root_delta[motion_ids_storage].to(
            device=self.device, dtype=torch.float32
        )
        root_rot = _batch_slerp(root_rot0, root_rot1, blend)
        dof_pos = (1.0 - blend_unsqueezed) * dof_pos0 + blend_unsqueezed * dof_pos1

        return (
            root_pos,
            root_rot,
            self._motion_root_vel[frame_idx0_storage].to(device=self.device, dtype=torch.float32),
            self._motion_root_ang_vel[frame_idx0_storage].to(device=self.device, dtype=torch.float32),
            dof_pos,
            self._motion_dof_vel[frame_idx0_storage].to(device=self.device, dtype=torch.float32),
        )

    def calc_motion_frame_with_local_info(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        motion_loop_num = torch.floor(motion_times / self._motion_lengths[motion_ids])
        motion_times = motion_times - motion_loop_num * self._motion_lengths[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
        motion_ids_storage = motion_ids.to(device=self.storage_device, dtype=torch.long)
        frame_idx0_storage = frame_idx0.to(device=self.storage_device, dtype=torch.long)
        frame_idx1_storage = frame_idx1.to(device=self.storage_device, dtype=torch.long)

        root_pos0 = self._motion_root_pos[frame_idx0_storage].to(device=self.device, dtype=torch.float32)
        root_pos1 = self._motion_root_pos[frame_idx1_storage].to(device=self.device, dtype=torch.float32)
        root_rot0 = self._motion_root_rot[frame_idx0_storage].to(device=self.device, dtype=torch.float32)
        root_rot1 = self._motion_root_rot[frame_idx1_storage].to(device=self.device, dtype=torch.float32)
        dof_pos0 = self._motion_dof_pos[frame_idx0_storage].to(device=self.device, dtype=torch.float32)
        dof_pos1 = self._motion_dof_pos[frame_idx1_storage].to(device=self.device, dtype=torch.float32)
        local_body_pos0 = self._motion_local_body_pos[frame_idx0_storage].to(device=self.device, dtype=torch.float32)
        local_body_pos1 = self._motion_local_body_pos[frame_idx1_storage].to(device=self.device, dtype=torch.float32)
        root_pos_delta_local0 = self._motion_root_pos_delta_local[frame_idx0_storage].to(
            device=self.device, dtype=torch.float32
        )
        root_pos_delta_local1 = self._motion_root_pos_delta_local[frame_idx1_storage].to(
            device=self.device, dtype=torch.float32
        )
        root_rot_delta_local0 = self._motion_root_rot_delta_local[frame_idx0_storage].to(
            device=self.device, dtype=torch.float32
        )
        root_rot_delta_local1 = self._motion_root_rot_delta_local[frame_idx1_storage].to(
            device=self.device, dtype=torch.float32
        )

        blend_unsqueezed = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsqueezed) * root_pos0 + blend_unsqueezed * root_pos1
        root_pos = root_pos + motion_loop_num.unsqueeze(-1) * self._motion_root_delta[motion_ids_storage].to(
            device=self.device, dtype=torch.float32
        )
        root_rot = _batch_slerp(root_rot0, root_rot1, blend)
        dof_pos = (1.0 - blend_unsqueezed) * dof_pos0 + blend_unsqueezed * dof_pos1
        local_body_pos = (1.0 - blend_unsqueezed.unsqueeze(1)) * local_body_pos0
        local_body_pos += blend_unsqueezed.unsqueeze(1) * local_body_pos1
        root_pos_delta_local = (1.0 - blend_unsqueezed) * root_pos_delta_local0 + blend_unsqueezed * root_pos_delta_local1
        root_rot_delta_local = (1.0 - blend_unsqueezed) * root_rot_delta_local0 + blend_unsqueezed * root_rot_delta_local1

        return (
            root_pos,
            root_rot,
            self._motion_root_vel[frame_idx0_storage].to(device=self.device, dtype=torch.float32),
            self._motion_root_ang_vel[frame_idx0_storage].to(device=self.device, dtype=torch.float32),
            dof_pos,
            self._motion_dof_vel[frame_idx0_storage].to(device=self.device, dtype=torch.float32),
            local_body_pos,
            root_pos_delta_local,
            root_rot_delta_local,
        )

    def calc_motion_frame_batch(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if motion_ids.shape != motion_times.shape:
            raise ValueError(
                f"motion_ids shape {motion_ids.shape} must match motion_times shape {motion_times.shape}"
            )

        batch_shape = motion_ids.shape
        outputs = self.calc_motion_frame(motion_ids.reshape(-1), motion_times.reshape(-1))
        return tuple(output.reshape(*batch_shape, output.shape[-1]) for output in outputs)

    def calc_motion_frame_batch_with_local_info(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if motion_ids.shape != motion_times.shape:
            raise ValueError(
                f"motion_ids shape {motion_ids.shape} must match motion_times shape {motion_times.shape}"
            )

        batch_shape = motion_ids.shape
        outputs = self.calc_motion_frame_with_local_info(motion_ids.reshape(-1), motion_times.reshape(-1))
        reshaped_outputs = []
        for output in outputs:
            reshaped_outputs.append(output.reshape(*batch_shape, *output.shape[1:]))
        return tuple(reshaped_outputs)
