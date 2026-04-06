# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--reference_vis",
    action="store_true",
    default=False,
    help="Visualize reference motion key bodies as red spheres for the first environment.",
)
parser.add_argument(
    "--torso_cam",
    action="store_true",
    default=False,
    help="Follow env 0 robot root with a torso-focused viewer camera.",
)
parser.add_argument(
    "--viewer_origin",
    type=str,
    default=None,
    choices=("world", "env", "asset_root", "asset_body"),
    help="Viewer origin type override applied without Hydra syntax.",
)
parser.add_argument("--viewer_asset", type=str, default=None, help="Viewer asset name override.")
parser.add_argument("--viewer_body", type=str, default=None, help="Viewer body name override.")
parser.add_argument("--viewer_env_index", type=int, default=None, help="Viewer environment index override.")
parser.add_argument(
    "--viewer_eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Viewer eye override as three floats.",
)
parser.add_argument(
    "--viewer_lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Viewer lookat override as three floats.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import twist2_lab.tasks  # noqa: F401


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(sin_yaw, cos_yaw)


def _set_torso_camera(env: DirectRLEnv, eye_local: tuple[float, float, float], lookat_local: tuple[float, float, float]):
    root_pos = env.robot.data.root_link_pos_w[0]
    root_quat = env.robot.data.root_link_quat_w[0]
    yaw = _yaw_from_quat(root_quat)

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotation = torch.stack(
        (
            torch.stack((cos_yaw, -sin_yaw, torch.zeros_like(cos_yaw))),
            torch.stack((sin_yaw, cos_yaw, torch.zeros_like(cos_yaw))),
            torch.tensor((0.0, 0.0, 1.0), device=root_pos.device, dtype=root_pos.dtype),
        )
    )

    eye_local_tensor = torch.tensor(eye_local, device=root_pos.device, dtype=root_pos.dtype)
    lookat_local_tensor = torch.tensor(lookat_local, device=root_pos.device, dtype=root_pos.dtype)
    eye_world = root_pos + rotation @ eye_local_tensor
    lookat_world = root_pos + rotation @ lookat_local_tensor
    env.sim.set_camera_view(eye=eye_world.tolist(), target=lookat_world.tolist())


class StreamingVideoRecorder:
    def __init__(self, video_folder: str, fps: int, video_name: str = "rl-video-step-0"):
        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)
        self.path = os.path.join(self.video_folder, f"{video_name}.mp4")
        self.writer = imageio.get_writer(self.path, fps=fps)
        self.frame_count = 0

    def capture(self, render_output):
        if isinstance(render_output, list):
            for frame in render_output:
                self._append_frame(frame)
        else:
            self._append_frame(render_output)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _append_frame(self, frame):
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray frame, got {type(frame)}")
        self.writer.append_data(frame)
        self.frame_count += 1


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if hasattr(env_cfg, "enable_reference_markers"):
        env_cfg.enable_reference_markers = args_cli.reference_vis
    if args_cli.torso_cam:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = None
        env_cfg.viewer.body_name = None
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-2.2, 0.0, 1.1)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.9)
    if args_cli.viewer_origin is not None:
        env_cfg.viewer.origin_type = args_cli.viewer_origin
    if args_cli.viewer_asset is not None:
        env_cfg.viewer.asset_name = args_cli.viewer_asset
    if args_cli.viewer_body is not None:
        env_cfg.viewer.body_name = args_cli.viewer_body
    if args_cli.viewer_env_index is not None:
        env_cfg.viewer.env_index = args_cli.viewer_env_index
    if args_cli.viewer_eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.viewer_eye)
    if args_cli.viewer_lookat is not None:
        env_cfg.viewer.lookat = tuple(args_cli.viewer_lookat)

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_recorder = None
    render_env = env

    # prepare streaming video recording
    if args_cli.video:
        video_fps = int(round(1.0 / env.unwrapped.step_dt))
        if isinstance(getattr(render_env, "metadata", None), dict):
            video_fps = render_env.metadata.get("render_fps", video_fps)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "video_length": args_cli.video_length,
            "video_fps": video_fps,
            "video_name": "rl-video-step-0",
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        video_recorder = StreamingVideoRecorder(
            video_folder=video_kwargs["video_folder"],
            fps=video_kwargs["video_fps"],
            video_name=video_kwargs["video_name"],
        )

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if args_cli.torso_cam:
        torso_cam_eye = tuple(args_cli.viewer_eye) if args_cli.viewer_eye is not None else (3.0, 0.0, 1.5)
        torso_cam_lookat = tuple(args_cli.viewer_lookat) if args_cli.viewer_lookat is not None else (0.0, 0.0, 0.0)
        _set_torso_camera(env.unwrapped, torso_cam_eye, torso_cam_lookat)
    timestep = 0
    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                if args_cli.torso_cam:
                    _set_torso_camera(env.unwrapped, torso_cam_eye, torso_cam_lookat)
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, dones, _ = env.step(actions)
                # reset recurrent states for episodes that have terminated
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)
            if video_recorder is not None:
                video_recorder.capture(render_env.render())
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if video_recorder is not None:
            video_recorder.close()
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
