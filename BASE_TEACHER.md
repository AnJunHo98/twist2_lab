# Base Teacher Notes

## Purpose

`TWIST2-IGRIS-Base-Teacher-Direct-v0` is the single teacher task used for teacher tuning.

Goal:

- stay upright for a long time
- follow reference motion
- follow root path
- remain robust under moderate randomization

This task is intended to be the closest Isaac Lab reproduction of the old TWIST2 privileged mimic teacher.

## Current Summary

- task: `TWIST2-IGRIS-Base-Teacher-Direct-v0`
- robot asset: `source/twist2_lab/data/robots/igris_c_v2`
- default motion manifest: `source/twist2_lab/data/motion_data_configs/twist2_teacher_upright_dataset.yaml`
- teacher policy: future-conditioned actor/critic
- teacher variants: removed, tuning should start from this base only

## Quick Train

Short test:

```bash
source ~/miniconda3/etc/profile.d/conda.sh  # adjust to your conda install
conda activate isaac
# run from the repository root; ISAACLAB_PATH points to your Isaac Lab checkout
TERM=xterm "${ISAACLAB_PATH}"/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task TWIST2-IGRIS-Base-Teacher-Direct-v0 \
  --num_envs 128 \
  --max_iterations 20 \
  --headless
```

Long run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh  # adjust to your conda install
conda activate isaac
# run from the repository root; ISAACLAB_PATH points to your Isaac Lab checkout
TERM=xterm "${ISAACLAB_PATH}"/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task TWIST2-IGRIS-Base-Teacher-Direct-v0 \
  --num_envs 1024 \
  --max_iterations 1000000 \
  --headless
```

## Update Rule

When this teacher is changed, append a short note below.

Recommended format:

```text
## YYYY-MM-DD HH:MM KST
- Why: what problem was observed in training
- Change: what was modified
- Test: what run was executed
- Result: short outcome
```

## Change Log

## 2026-04-07 12:55 KST
- Why: batch-based promotion worked, but the previous `0.2` threshold advanced the curriculum too aggressively.
- Change: raised the base teacher batch promotion threshold from `0.2` to `0.35`.
- Test: not run by request; intended for the next long training run.
- Result: pending long-run validation.

## 2026-04-07 12:50 KST
- Why: cumulative success rate kept the teacher at visible level `2` even when recent batches were already performing much better.
- Change: made curriculum promotion metric configurable and switched the base teacher to `batch`-based promotion.
- Test: `128` env, `20` iteration diagnostic run at `logs/rsl_rl/igris_base_teacher_direct/2026-04-07_12-51-17`.
- Result: promotion worked immediately and the curriculum reached visible level `4` within the short test, which confirms the logic works but also suggests the current `0.2` threshold is likely too aggressive for batch-based promotion.

## 2026-04-07 11:12 KST
- Why: after 500 iterations, local imitation improved a lot, but global root drift and global key-body drift were still visible.
- Change: added explicit drift penalties for absolute root XY error and global key-body error in the base teacher reward.
- Test: `128` env, `20` iteration diagnostic run at `logs/rsl_rl/igris_base_teacher_direct/2026-04-07_11-12-14`.
- Result: no training regression; root/global eval stayed slightly better than the previous 20-iteration reference, but the effect is still small and needs a longer run to judge fully.

## 2026-04-07
- Why: many teacher variants existed, but none reliably satisfied long standing, reference tracking, root-path tracking, and robustness at the same time.
- Change: unified teacher tuning around `TWIST2-IGRIS-Base-Teacher-Direct-v0` and removed alternate teacher registrations.
- Test: smoke test and repeated short training runs on the base teacher.
- Result: base teacher became the only teacher task and is now the single starting point for teacher tuning.

## 2026-04-07
- Why: the full mixed motion set contained non-upright motions, while the teacher termination logic assumed upright humanoid behavior.
- Change: switched the base teacher to an upright-only manifest and reworked the base task toward the old TWIST2 privileged mimic structure.
- Test: 1-iteration smoke test and 20-iteration diagnostic training.
- Result: training became stable enough to proceed without immediate startup failure.

## 2026-04-07
- Why: early diagnostic runs showed many resets before the policy could learn stable tracking.
- Change: relaxed curriculum gates, disabled robot push, and loosened `termination_pose_tracking_dist` and `termination_root_height_diff`.
- Test: 20-iteration diagnostic run at `128` envs.
- Result: episode length improved and curriculum reached visible level `2`, which was better than the stricter setup.

## 2026-04-07
- Why: after long training, it was still unclear which termination condition dominated failures.
- Change: added per-reason termination logging for `pose_tracking`, `root_height_diff`, `max_root_lin_vel`, `motion_end`, and related signals.
- Test: 20-iteration diagnostic run at `128` envs.
- Result: the dominant failure reason is currently `pose_tracking`, with smaller contribution from `root_height_diff`; `contact`, `root_xy_tracking`, `tilt`, and `invalid_state` are near zero.

## 2026-04-07
- Why: a long run was interrupted when overlapping training processes were started during code edits.
- Change: verified surviving logs and re-ran a single non-overlapping diagnostic test after the termination-reason logging patch.
- Test: surviving long run `logs/rsl_rl/igris_base_teacher_direct/2026-04-07_07-57-55` and new test run `logs/rsl_rl/igris_base_teacher_direct/2026-04-07_08-49-55`.
- Result: the surviving long run actually reached iteration `219`; the latest short test kept the same learning direction, so the current modification path is still reasonable.
