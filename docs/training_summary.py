"""Generate training environment summary PDF (English only)."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.enums import TA_CENTER

W, H = A4
MARGIN = 18 * mm

C_HEADER  = colors.HexColor("#1a2740")
C_SECTION = colors.HexColor("#2563eb")
C_ALT     = colors.HexColor("#f0f4ff")
C_BORDER  = colors.HexColor("#cbd5e1")
C_WHITE   = colors.white


def styles():
    s = {}
    s["title"]    = ParagraphStyle("title",    fontSize=20, leading=26, textColor=C_WHITE,
                                    alignment=TA_CENTER, fontName="Helvetica-Bold")
    s["subtitle"] = ParagraphStyle("subtitle", fontSize=9,  leading=14,
                                    textColor=colors.HexColor("#cbd5e1"),
                                    alignment=TA_CENTER, fontName="Helvetica")
    s["section"]  = ParagraphStyle("section",  fontSize=12, leading=16, textColor=C_SECTION,
                                    fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=5)
    s["label"]    = ParagraphStyle("label",    fontSize=8,  leading=11,
                                    textColor=colors.HexColor("#475569"), fontName="Helvetica-Bold")
    s["note"]     = ParagraphStyle("note",     fontSize=7,  leading=10,
                                    textColor=colors.HexColor("#64748b"), fontName="Helvetica-Oblique")
    return s


def hr():
    return HRFlowable(width="100%", thickness=0.4, color=C_BORDER, spaceAfter=5)


def sec(title, s):
    return Paragraph(f"| {title}", s["section"])


def tbl(data, col_widths, note_rows=None):
    note_rows = note_rows or []
    style = [
        ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("LEADING",        (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_ALT]),
        ("GRID",           (0, 0), (-1, -1), 0.3, C_BORDER),
        ("LEFTPADDING",    (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 5),
        ("TOPPADDING",     (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",     (0, 0), (-1, 0),  C_HEADER),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
    ]
    for r in note_rows:
        style += [
            ("TEXTCOLOR", (0, r), (-1, r), colors.HexColor("#64748b")),
            ("FONTNAME",  (0, r), (-1, r), "Helvetica-Oblique"),
        ]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


def build_pdf(path):
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="IGRIS Teacher Training Summary",
    )
    s = styles()
    story = []
    W_CONTENT = W - 2 * MARGIN

    # ── Header ────────────────────────────────────────────────────────────────
    title_tbl = Table(
        [[Paragraph("IGRIS Teacher Policy — Training Environment Summary", s["title"])],
         [Paragraph(
             "Task: TWIST2-IGRIS-Base-Teacher-Direct-v0  |  Canonical base teacher  |  2026-04-07",
             s["subtitle"])]],
        colWidths=[W_CONTENT],
    )
    title_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_HEADER),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    story += [title_tbl, Spacer(1, 10)]

    # ── 1. Overview ───────────────────────────────────────────────────────────
    story.append(sec("1. Training Overview", s))
    story.append(tbl([
        ["Item",              "Value"],
        ["Task",              "TWIST2-IGRIS-Base-Teacher-Direct-v0"],
        ["Algorithm",         "PPO  (OnPolicyRunner, RSL-RL)"],
        ["Num Envs",          "4,096"],
        ["Max Iterations",    "100,000  (v3 resumed from iter 7,600)"],
        ["Sim / Step dt",     "0.002 s / 0.02 s  (decimation=10)"],
        ["Episode Length",    "500 steps = 10 s"],
        ["Run name (v3)",     "v3_motion_curriculum — motion_curriculum=True, level_curriculum=False"],
    ], [55*mm, W_CONTENT - 55*mm]))
    story.append(Spacer(1, 4))

    # ── 2. PPO ────────────────────────────────────────────────────────────────
    story.append(sec("2. PPO Hyperparameters", s))
    story.append(tbl([
        ["Parameter",           "Value",    "Note"],
        ["num_steps_per_env",   "64",       "rollout buffer length per env"],
        ["num_learning_epochs", "5",        ""],
        ["num_mini_batches",    "4",        "~67K samples per mini-batch"],
        ["learning_rate",       "1e-4",     "adaptive (desired_kl=0.008)"],
        ["clip_param",          "0.2",      ""],
        ["entropy_coef",        "0.01",     "* modified: 0.0 -> 0.01  (prevent entropy collapse)"],
        ["value_loss_coef",     "1.0",      ""],
        ["gamma / lambda",      "0.99 / 0.95", ""],
        ["max_grad_norm",       "1.0",      ""],
    ], [55*mm, 30*mm, W_CONTENT - 85*mm], note_rows=[6]))
    story.append(Spacer(1, 4))

    # ── 3. Network ────────────────────────────────────────────────────────────
    story.append(sec("3. Network Architecture", s))
    story.append(tbl([
        ["Module",       "Hidden Dims",           "Activation",   "Note"],
        ["Actor",        "[512, 512, 256, 128]",  "SiLU",         "* modified: ELU -> SiLU"],
        ["Critic",       "[512, 512, 256, 128]",  "SiLU",         "obs_normalization=True"],
        ["Obs Norm",     "EmpiricalNormalization", "—",           "running mean/std"],
        ["Distribution", "GaussianDistribution",  "init_std=1.0", ""],
    ], [28*mm, 54*mm, 28*mm, W_CONTENT - 110*mm], note_rows=[1]))
    story.append(Spacer(1, 4))

    # ── 4. Observations ───────────────────────────────────────────────────────
    story.append(sec("4. Observation Space  (total 2,764 dim)", s))
    story.append(tbl([
        ["Group",                "Dim",  "Contents",                                           "Noise"],
        ["(1) Motion Future Obs","1,580","20 future timesteps x 79 dim\n"
                                          "  ref_root_pos(3) / pos_delta(3) / rpy(3) / vel_local(3)\n"
                                          "  ang_vel_local(3) / pos_delta_local(3) / rot_delta_local(3)\n"
                                          "  dof_pos(31) / key_body_local(27)",                  "Yes"],
        ["(2) Proprio Obs",      "98",   "ang_vel_b x0.25(3) / roll-pitch(2)\n"
                                          "  joint_pos_offset x1.0(31) / joint_vel x0.05(31)\n"
                                          "  actions(31)  [raw_actions if action_delay active]",  "Yes"],
        ["(3) Privileged Info",  "106",  "root_lin_vel_b(3) / root_pos_w(3) / root_quat_w(4)\n"
                                          "  key_body_local_cur(27) / foot_contacts(2)\n"
                                          "  mass_feature(4) / friction_feature(1) / motor_offset(62)","No"],
        ["(4) Proprio History",  "980",  "* NEW: 10 timesteps x 98 dim  (sliding window of proprio)\n"
                                          "  reset to zero on episode start",                     "Yes"],
    ], [38*mm, 13*mm, W_CONTENT - 66*mm, 15*mm], note_rows=[4]))
    story.append(Spacer(1, 4))

    # ── 5. Reward ─────────────────────────────────────────────────────────────
    story.append(sec("5. Reward Structure", s))
    story.append(Paragraph("Tracking rewards (positive)", s["label"]))
    story.append(tbl([
        ["Term",                 "Scale", "Formula"],
        ["track_root_pos",       "1.0",   "exp(-4 * ||pos_err||^2) * dt"],
        ["track_root_rot",       "0.75",  "exp(-2 * rot_err^2) * dt"],
        ["track_root_vel",       "0.5",   "exp(-2 * ||vel_err||^2) * dt"],
        ["track_root_ang_vel",   "0.5",   "exp(-0.5 * ||ang_vel_err||^2) * dt"],
        ["track_dof_pos",        "1.0",   "exp(-0.25 * ||dof_err||^2) * dt"],
        ["track_dof_vel",        "0.35",  "exp(-0.05 * ||dof_vel_err||^2) * dt"],
        ["track_key_body",       "1.5",   "* modified: 0.75->1.5   exp(-2 * key_body_local_err) * dt"],
        ["track_key_body_global","1.5",   "* modified: 0.75->1.5   exp(-2 * key_body_global_err) * dt"],
        ["alive",                "0.25",  "constant * dt"],
    ], [45*mm, 18*mm, W_CONTENT - 63*mm], note_rows=[8, 9]))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Penalty terms (negative)", s["label"]))
    story.append(tbl([
        ["Term",                  "Scale",  "Formula"],
        ["action_rate",           "-0.01",  "||a - a_prev||^2 * dt"],
        ["torque_penalty",        "-1e-6",  "cmd override (cfg default: -1e-5)  ||tau||^2 * dt"],
        ["feet_slip",             "-0.1",   "slip_velocity * dt"],
        ["feet_contact_forces",   "-5e-4",  "excess_contact_force * dt"],
        ["feet_air_time",         "+5.0",   "(air_time - 0.5s).clamp(max=0) * dt  [always <= 0]"],
        ["feet_stumble",          "-1.25",  "* dt"],
        ["lin_vel_z",             "-1.0",   "vz^2 * dt"],
        ["orientation",           "-1.0",   "projected_gravity_xy^2 * dt"],
        ["ang_vel_xy",            "-0.01",  "||omega_xy||^2 * dt"],
        ["dof_vel",               "-1e-4",  "||joint_vel||^2 * dt"],
        ["ankle_dof_vel",         "-2e-4",  "||ankle_vel||^2 * dt"],
        ["joint_acc",             "-5e-8",  "||joint_acc||^2 * dt"],
        ["dof_pos_limits",        "-5.0",   "limit_violation * dt"],
        ["dof_torque_limits",     "-1.0",   "torque_limit_violation * dt"],
        ["collision",             "-10.0",  "penalized_contact * dt"],
        ["terminated",            "-2.0",   "terminated * dt"],
    ], [45*mm, 18*mm, W_CONTENT - 63*mm]))
    story.append(Spacer(1, 4))

    # ── 6. Domain Randomization ───────────────────────────────────────────────
    story.append(sec("6. Domain Randomization", s))
    story.append(tbl([
        ["Item",                 "Setting",              "Note"],
        ["Mass scale",           "x(0.9 ~ 1.1)",        "resampled on reset"],
        ["Joint friction",       "x(0.75 ~ 1.25)",      "resampled on reset"],
        ["Stiffness / Damping",  "x(0.85 ~ 1.15)",      "resampled on reset"],
        ["Root CoM offset",      "+/- 0.05 m",          "resampled on reset"],
        ["Gravity XY",           "+/- 0.1 m/s^2",       "resampled every 4 s"],
        ["Robot push",           "disabled",             "--enable_teacher_robot_push=false"],
        ["End-effector push",    "disabled",             "--enable_teacher_end_effector_push=false"],
        ["Action delay",         "0~1 step",             "activated after obs_noise ramp"],
        ["Obs noise",            "0 -> max (ramp)",      "* modified: ramp over 10,000x24 steps (~iter 11,355 = 100%)"],
    ], [42*mm, 38*mm, W_CONTENT - 80*mm], note_rows=[9]))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Obs noise max values (at noise_progress=1.0):", s["label"]))
    story.append(tbl([
        ["Obs term",                                     "Max noise"],
        ["root_pos / key_body",                          "0.02"],
        ["orientation / IMU",                            "0.05"],
        ["root_vel / root_ang_vel / dof_vel / ang_vel",  "0.1"],
        ["dof_pos",                                      "0.01"],
    ], [100*mm, W_CONTENT - 100*mm]))
    story.append(Spacer(1, 4))

    # ── 7. Curriculum ─────────────────────────────────────────────────────────
    story.append(sec("7. Curriculum", s))
    story.append(tbl([
        ["Item",                         "Setting",              "Note"],
        ["motion_curriculum",            "True  * v3 new",       "per-motion adaptive difficulty"],
        ["motion_curriculum_gamma",      "0.01",                 "difficulty update rate"],
        ["difficulty update rule",       "completion <= 50%  -> x(1+gamma)\n"
                                          "completion >= 95%  -> x(1-gamma)\n"
                                          "completion >= 99%  -> x(1-20*gamma)", ""],
        ["difficulty range",             "1.0 ~ 10.0",           "initialized to 10.0 for all motions"],
        ["enable_motion_level_curriculum","False  * v3 changed", "level gating disabled -> levels 1-5 all open"],
        ["use_error_aware_sampling",     "True",                 "key body error weighted sampling"],
    ], [52*mm, 50*mm, W_CONTENT - 102*mm], note_rows=[1, 5]))
    story.append(Spacer(1, 4))

    # ── 8. Termination ────────────────────────────────────────────────────────
    story.append(sec("8. Termination Conditions", s))
    story.append(tbl([
        ["Condition",      "Threshold"],
        ["Root height",    "< 0.65 m"],
        ["Body tilt",      "projected_gravity_xy^2 > 0.8"],
        ["Torso contact",  "contact on body_link"],
        ["NaN / Inf state","joint pos/vel/acc, contact forces, etc."],
        ["Timeout",        "500 steps (10 s)"],
    ], [70*mm, W_CONTENT - 70*mm]))
    story.append(Spacer(1, 4))

    # ── 9. Changes ────────────────────────────────────────────────────────────
    story.append(sec("9. Changes from v1 (original)", s))
    story.append(tbl([
        ["Item",                    "v1 (original)", "v3 (current)",  "Effect"],
        ["entropy_coef",            "0.0",           "0.01",          "prevent entropy collapse"],
        ["activation",              "ELU",           "SiLU",          "smoother gradient"],
        ["num_steps_per_env",       "32",            "64",            "better credit assignment"],
        ["obs_noise_ramp_steps",    "3,000",         "10,000",        "gradual noise introduction"],
        ["proprio_history_len",     "0",             "10",            "noise/delay compensation"],
        ["track_key_body scale",    "0.75",          "1.5",           "stronger keypoint tracking"],
        ["motion_curriculum",       "False",         "True",          "focus on hard motions"],
        ["level_curriculum",        "True",          "False",         "remove cumulative counter issue"],
    ], [45*mm, 25*mm, 25*mm, W_CONTENT - 95*mm]))
    story.append(Spacer(1, 8))

    story.append(hr())
    story.append(Paragraph("Generated from twist2_lab codebase — 2026-03-24", s["note"]))

    doc.build(story)
    print(f"PDF saved: {path}")


if __name__ == "__main__":
    docs_dir = Path(__file__).resolve().parent
    docs_dir.mkdir(parents=True, exist_ok=True)
    build_pdf(str(docs_dir / "training_summary.pdf"))
