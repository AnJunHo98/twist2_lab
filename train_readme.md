# TWIST2 Lab Training Commands

## 0. Environment

```bash
cd /path/to/twist2_lab
python scripts/list_envs.py
```

Expected teacher task:

- `TWIST2-IGRIS-Base-Teacher-Direct-v0`

## 1. Train the canonical base teacher first

Short verification run:

```bash
python scripts/rsl_rl/train.py \
  --task TWIST2-IGRIS-Base-Teacher-Direct-v0 \
  --num_envs 128 \
  --max_iterations 1000 \
  --headless
```

Long old-TWIST2 style run:

```bash
python scripts/rsl_rl/train.py \
  --task TWIST2-IGRIS-Base-Teacher-Direct-v0 \
  --num_envs 4096 \
  --max_iterations 30000 \
  --headless \
  --logger wandb \
  --log_project_name twist2_lab \
  --run_name base_teacher_twist2_repro \
  agent.save_interval=500 \
  env.use_error_aware_sampling=true
```

## 2. Pick the base teacher checkpoint

Latest run:

```bash
ls -td logs/rsl_rl/igris_base_teacher_direct/* | head
```

Available checkpoints:

```bash
ls logs/rsl_rl/igris_base_teacher_direct/<BASE_RUN>/*.pt
```

Set shell variables:

```bash
export BASE_TEACHER_RUN=<BASE_RUN>
export BASE_TEACHER_CKPT=model_999.pt
```

## 3. Student PPO baseline

```bash
python scripts/rsl_rl/train.py \
  --task TWIST2-IGRIS-Student-Future-Direct-v0 \
  --num_envs 128 \
  --max_iterations 1000 \
  --headless
```

## 4. Student distillation from the single base teacher

```bash
python scripts/rsl_rl/train.py \
  --task TWIST2-IGRIS-Student-Future-Priv-Distill-Direct-v0 \
  --num_envs 128 \
  --max_iterations 1000 \
  --headless \
  agent.load_run=$BASE_TEACHER_RUN \
  agent.load_checkpoint=$BASE_TEACHER_CKPT
```

## 5. Student DAgger from the single base teacher

```bash
python scripts/rsl_rl/train.py \
  --task TWIST2-IGRIS-Student-Future-Priv-Dagger-Direct-v0 \
  --num_envs 128 \
  --max_iterations 1000 \
  --headless \
  agent.teacher_load_run=$BASE_TEACHER_RUN \
  agent.teacher_load_checkpoint=$BASE_TEACHER_CKPT
```

## 6. Preview the base teacher with reference markers

```bash
python scripts/rsl_rl/play.py \
  --task TWIST2-IGRIS-Base-Teacher-Direct-v0 \
  --num_envs 32 \
  --checkpoint logs/rsl_rl/igris_base_teacher_direct/$BASE_TEACHER_RUN/$BASE_TEACHER_CKPT \
  --reference_vis
```

## 7. Compare logs

```bash
tensorboard --logdir logs/rsl_rl
```

Primary metrics:

- `Mean reward`
- `Mean episode length`
- `Eval/root_pos_error`
- `Eval/dof_pos_error`
- `Eval/key_body_local_error`
- `Eval/key_body_global_error`
- `Curriculum/mean_motion_difficulty`
