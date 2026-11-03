# TWIST2 Lab

Isaac Lab extension for TWIST2-style humanoid motion-imitation (teacher/student) control on the IGRIS robot.

It migrates TWIST2 humanoid control work into a standalone Isaac Lab project.

Current scope:

- IGRIS robot configuration in `twist2_lab.robots.igris_c` (USD assets must be provided locally, see [Required data](#required-data-not-included))
- PKL motion loading driven by YAML motion manifests under `source/twist2_lab/data/motion_data_configs` (e.g. `twist2_local_dataset.yaml`, `twist2_teacher_upright_dataset.yaml`)
- direct RL tasks for the canonical base teacher and future-conditioned student control

Commonly used tasks:

- `TWIST2-IGRIS-Base-Teacher-Direct-v0`
- `TWIST2-IGRIS-Student-Future-Direct-v0`
- `TWIST2-IGRIS-Student-Future-Priv-Distill-Direct-v0`
- `TWIST2-IGRIS-Student-Future-Priv-Dagger-Direct-v0`

## Installation

Install the package into a Python environment that already has Isaac Lab available:

```bash
python -m pip install -e source/twist2_lab
```

### Required data (not included)

Of `source/twist2_lab/data`, only the motion manifests in `motion_data_configs/` are tracked in git. USD files and `source/twist2_lab/data/motions/` are listed in `.gitignore`, so the following must be provided locally:

- **IGRIS USD assets**: `source/twist2_lab/data/robots/igris_c_v2/igris_c_v2.usd`, plus optionally `igris_c_v2_floating.usd` (the non-floating file is used when it is absent). Importing `twist2_lab.robots.igris_c` raises `FileNotFoundError` if the USD is missing.
- **PKL motion files**: the manifests set `root_path` to `GMR/motion_data/standing_transition_full` in a `GMR` directory next to this repository. If a file is not found there, the loader also looks under `source/twist2_lab/data/motions/standing_transition_full`.

## Docker

This repository now includes a `docker/` setup for running the extension on top of an Isaac Lab base image.

Prerequisites:

- Docker Engine with `docker compose`
- NVIDIA Container Toolkit for GPU passthrough
- an Isaac Lab base image available locally as `isaac-lab-base`, or a matching prebuilt Isaac Lab image set through `docker/.env.base`
- an optional external motion dataset available under `../GMR` from the repository root

The default `docker/.env.base` expects the common sibling layout below and preserves the existing manifest relative paths unchanged:

```text
<workspace>/
├── GMR/
└── twist2_lab/
```

Build and start the container:

```bash
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml build twist2-lab
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml up -d
docker exec -it twist2-lab bash
```

Inside the container:

```bash
source ~/.bashrc
python scripts/list_envs.py
python scripts/rsl_rl/train.py --task TWIST2-IGRIS-Base-Teacher-Direct-v0 --headless
```

Stop the container:

```bash
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml down
```

Notes:

- The repository is bind-mounted to `/workspace/twist2_lab` for live editing.
- Motion loading now tries the external `GMR` tree first and falls back to a local copy under `source/twist2_lab/data/motions` (not tracked in git) when a file is not present externally.
- The optional external `GMR` tree is mounted to `/workspace/GMR`, so the existing `../../../../../GMR/...` motion manifest paths continue to work without modifying YAML files.
- If you only need headless training, you can point `ISAACLAB_BASE_IMAGE` in `docker/.env.base` at a matching `nvcr.io/nvidia/isaac-lab:<tag>` image. For GUI or X11 workflows, prefer a locally built Isaac Lab base image that matches your Isaac Sim version.

## Usage

List the tasks exposed by this extension:

```bash
python scripts/list_envs.py
```

Train the canonical base teacher:

```bash
python scripts/rsl_rl/train.py --task TWIST2-IGRIS-Base-Teacher-Direct-v0
```

Train the future-conditioned student:

```bash
python scripts/rsl_rl/train.py --task TWIST2-IGRIS-Student-Future-Direct-v0
```

Run a trained checkpoint:

```bash
python scripts/rsl_rl/play.py --task TWIST2-IGRIS-Base-Teacher-Direct-v0
python scripts/rsl_rl/play.py --task TWIST2-IGRIS-Student-Future-Direct-v0
```

## Notes

- `TWIST2-IGRIS-Base-Teacher-Direct-v0` is the recommended single starting point for teacher tuning.
- The base teacher is the closest Isaac Lab reproduction of the old TWIST2 privileged mimic teacher and defaults to the upright-only manifest `twist2_teacher_upright_dataset.yaml`.
- The base teacher stores motion data on CPU so the full TWIST2 local dataset can be used without exhausting GPU memory during startup.
- `twist2_lab.robots.igris_c` no longer depends on an external IGRIS ROS checkout.
- `twist2_local_dataset.yaml` still prefers `../GMR/motion_data/standing_transition_full`, but falls back to `source/twist2_lab/data/motions/standing_transition_full` when needed.
- Alternate teacher task registrations were removed so future changes happen from the single base teacher.
- `TWIST2-IGRIS-Student-Future-Direct-v0` is a first-pass student env with current target motion, proprioceptive state, observation history, and future reference frames.
- Step-by-step teacher/student training commands (including distillation and DAgger from the base teacher) are in [`train_readme.md`](train_readme.md); the base teacher tuning log is in [`BASE_TEACHER.md`](BASE_TEACHER.md).

## Status

Experimental research code. The base teacher is still being tuned (see `BASE_TEACHER.md`); the last code change was in April 2026.

## License

Released under the BSD 3-Clause License. See [LICENSE](LICENSE).
