# Worklog

## 2026-09-11

- Read the event page and Intel's 5-page brief; captured both in [brief.md](brief.md).
- Checked the laptop: i7-1165G7 (11th gen, not Core Ultra, no NPU), Iris Xe graphics, 16 GB RAM.
  The brief accepts "Intel CPU and iGPU", so we can enter, but we lose some of the 20 Core Ultra points.
- Decided to build for learning. Folder created and plan written in [plan.md](plan.md).
- Day 1 setup done on plain Windows:
  - Python 3.12 environment via `uv` at `C:\Users\USER\.venvs\ai-infra` (outside the synced folder).
  - MuJoCo 3.13 + OpenVINO 2026.3 installed. OpenVINO sees CPU and Iris Xe GPU.
  - SO-101 arm model copied from TheRobotStudio SO-ARM100 (Apache-2.0).
  - `sim/day1_wave.py` makes one arm wave and saves a GIF. Joints follow commands closely.
  - Steps written up in [setup.md](setup.md).
- **Day 2 done (same evening):**
  - `sim/scene.py` builds the table: two SO-101 arms facing each other, rimmed plate, mug,
    fork, spoon, bottle, four place-setting spots. Seed controls positions, yaw, mass,
    friction, lighting and table colour.
  - `sim/robot.py`: inverse kinematics, straight-line moves, gripper, contact-based grasp check.
  - `sim/skills.py`: pick, place, hand-off (giver holds one end, taker grabs the other), home.
  - `sim/day2_demo.py`: plate (A), mug (B), fork B→A hand-off, spoon A→B hand-off.
    **22/30 seeds pass.** ~3 s per run without video, ~2½ min with video.
  - Fixes that got it from 0/10 to 8/10, worth remembering:
    1. Jaw collision meshes are convex hulls that fill the finger gap → replaced by box pads.
    2. The moving jaw swings in an arc, so it can't pinch wide flat things → plate has a rim, gripped by the rim.
    3. Cutlery 8 mm tall only got a sliver of fingertip → handles now 14 mm.
    4. IK orientation error read zero at 180° → fingers closed from the wrong side. Now uses quaternion difference.
    5. Taker lifted during hand-off while giver still held → fingers slid off. Taker now grips, giver lets go, then lift.
    6. Wrist can't turn a full circle → cutlery placed at yaw or yaw+180°, whichever reaches.
    7. Stock grip ~60 N pushed objects through the pads → ~20 N, stiffer pads, less finger damping.
  - Remaining failures (seeds 8, 9, 13, 19, 20, 24, 28, 29): plate nudged 2.5–6 cm after
    placing (3×), spoon/fork grasp or hand-off misses (4×), mug grasp miss (1×).
- **Audit of Day 1–2:**
  - Fixed: plate was being bumped when cutlery was placed (release opening now sized to the
    object, and the arm picks the grip direction that swings the finger into free space).
  - Fixed: success check ignored cutlery direction — now fails if >20° off. (All passing runs
    were already correct.)
  - Fixed: pick now confirms the object is still held after lifting.
  - Fixed docs: finger gap is ~5.4 cm, not 8 cm; setup steps no longer tied to this laptop's
    paths; pad comment no longer points at a doc section that didn't exist.
  - Added: MIT `LICENSE` (lablab requires it) with the Apache-2.0 note for the arm model.
  - Checked: same seed gives the same result every run.
  - **Result: 24/30 seeds (80%)**, up from 22/30. Remaining: hand-off/start grasp misses
    (seeds 13, 19, 28, 29) and cutlery tipped on placing (9, 26).
  - Not fixed yet, bigger (added to plan): skills read exact object positions from the
    simulator instead of seeing them; shape isn't randomised; start positions only vary ±2 cm.
## 2026-09-11 (continued) — Day 3

- **Eyes** (`brain/perception.py`): overhead colour + depth camera → height map → blobs →
  classified by height/size/shape, colour only for fork vs spoon. Robot masked out using its own
  known shape (self-filter). **20/20 seeds: every object found, mean error 0.5 mm, 0.0° on cutlery.**
  - Lighting was washing colours to white from above → toned down lights and shine.
  - MuJoCo renderer bug: closing one renderer blacks out the others → `robot.close_renderer()`.
- **Skills no longer use simulator positions.** Table objects come from the last camera look;
  held objects from the hand's own pose plus how they sat in the fingers (`sim.held`).
  Simulator truth is now only used for scoring.
- **Brain** (`brain/language.py`): Qwen3-VL-4B INT4 on OpenVINO GenAI, Iris Xe GPU. Loads in
  20–70 s, reads a command in 18–31 s. Understands "the red thing" → mug, "the grey utensil" →
  fork from the picture. Keyword reader as fallback. Dropped the model's free-text scene
  comment — it made things up.
- **Planner** (`brain/planner.py`): goals → which arm, hand-off for cutlery, set-down swap for
  round items at the clearest spot both arms reach; puts the plate in the middle first if a
  command is relative to it; checks with the camera after every step, retries up to 3×, then a
  repair pass for anything nudged.
- **Voice** (`brain/voice.py`): Speechmatics batch API. **Untested — no API key yet.**
  Test clip: `brain/test_command.wav` ("Please set the table", made with Windows speech).
- Fixes found along the way: wrist can hit its limit turning the short way round (now plans
  both directions); cutlery taller than wide rolled over on release (now 16–18 mm wide, 12 mm
  tall); bottle clipped by the open finger (wider clearance for the moving finger); hand-off grip
  now deep, and release stops at fingertip height.
- **Scores (30 random tables):** `brain/evaluate.py` "set the table" **29/30, 0 retries**;
  Day 2 fixed script **29/30**. Only failure: seed 12, fork tipped.
- **Next (Day 4):** OpenVINO speed work (INT4/INT8, CPU vs GPU benchmark), and the training
  element (ACT policy for one skill on Kaggle).
- **Still on the user:** lablab + Discord sign-up, team, Kaggle account, GitHub repo,
  Speechmatics API key.

## 2026-09-12 — Day 4

- Audit of Day 3 + competitive plan agreed: training shortcut and speed (Day 4); recovery
  demo, harder randomisation, both arms at once (Day 5); video + GitHub (Day 6).
- Installed LeRobot 0.6.1 (+ dataset, training extras; PyTorch 2.11 CPU) and NNCF. Removed
  torchcodec (its Windows DLLs don't load; we store no video, so it isn't needed).
- **`train/record_demos.py`**: the scripted skills are the teacher. Random table, random mug
  start, random target; arm B picks and places; sampled at 20 Hz as (6 joint angles, camera
  mug xy + target xy) -> 6 motor commands. Misses are discarded using simulator truth.
  **200 episodes, 32,400 frames, 93% of teacher runs kept**, ~10 min. Standard LeRobot dataset.
- **`train/train_act.py`**: LeRobot's `lerobot-train`, ACT, 5 M parameters (256-wide,
  1-second action chunks), CPU. Windows fix: LeRobot's `checkpoints/last` symlink needs admin
  rights, so it's replaced with a copy.
- **`train/eval_policy.py`**: the policy alone drives arm B on unseen tables (seeds 50,000+).
- **`bench/export_act.py`**: ACT -> OpenVINO FP32/FP16/INT8 (NNCF, calibrated on the demos).
- **`bench/benchmark.py`**: VLM (device × picture size, with a 5-command accuracy check),
  ACT (PyTorch vs OpenVINO precisions × devices), perception timing -> `bench/RESULTS.md`.
- VLM now gets a 480-px-wide picture instead of 960 (fewer image tokens); benchmark will confirm.

### Training the policy — what worked and what didn't (honest record for the write-up)

Task: arm B picks the mug and puts it on a target. Success = mug upright within 1.5 cm,
tested on unseen tables (seeds 50,000+). Teacher succeeds ~90–95%.

| Version | Action the policy outputs | Inputs | Data | Result |
|---|---|---|---|---|
| v1 | joint angles | mug + target, table coords | 200 | 1/20 — fingertips ~5 mm off, mug knocked over |
| v2 | joint angles | mug + target relative to fingertip | 500 | 1/10 @5k steps |
| v3 | small hand moves | relative | 500 | 1/10 @2.5k — teacher's grip direction looked random (12 options) |
| v4 | small hand moves | relative, fixed "radial" grip rule | 409 | 0/10 — 0.3 mm/tick errors add up over 160 ticks |
| v5 | hand target position (abs) | table coords | 409 | 1/10 — aimed at mug centre on new tables |
| v6 | hand target position (abs) | table coords | 1,219 | 2/10 @3k — still biased to "average mug" |
| v7 | hand target **measured from the mug** | mug-relative | 1,219 | **16/20 within 1.5 cm, 17/20 within 2.5 cm @6k steps**; no tip-overs |

v7 is wired into the full system: `run.py --policy` lets the ACT policy (OpenVINO FP16) move
the mug whenever the target is inside the area it was trained on; elsewhere the scripted
skills take over. First try: "put the cup to the right of the plate" — plate by skills, mug by
the policy, both correct.

INT8 accuracy (open-loop drift vs PyTorch, normalised units): FP16 0.0002; INT8 default 0.067,
MIXED 0.10, TRANSFORMER/SmoothQuant 0.16–0.29. Also exporting INT8 weights-only. Task-level
success per precision is in the benchmark (`act_task`).

Laptop slept 02:40–09:54 on 13 Sep and paused everything; now kept awake during long jobs
(app keep-awake + a SetThreadExecutionState helper — no settings changed).

Lessons: (1) a teacher that picks the best of many options looks random to a student —
give it one smooth rule; (2) incremental actions drift, absolute ones don't; (3) express
the problem in the object's frame so "pick" is a fixed offset and "place" is given.
Tools built for this: `check_fit.py` (open-loop error), teacher-replay tests per action
format, `convert_to_absolute.py` / `convert_to_mug_frame.py` (re-express data without
re-simulating), `merge_datasets.py`, parallel recording (`--seed-start`), resume after the
Windows ~300-render-window limit.

### VLM speed (Iris Xe GPU, 5-command accuracy suite)

| Setting | s / command | correct |
|---|---|---|
| 960 px picture, JSON answer | 14.1 | 5/5 |
| 480 px picture, JSON answer | 10.2 | 5/5 |
| 480 px picture, `object: place` lines | **5.2** | **5/5** |

## 2026-09-13 — Day 5 (in progress)

- **Recovery demo** (`run.py --bump plate`): after each item the planner re-checks
  everything already placed and fixes anything moved. Works: plate knocked, noticed, put back.
- **Hard randomisation** (`--hard`): starts ±4 cm, sizes ±10% (plate ±5%), floor colour,
  overlap-free starts. The robot still uses the standard sizes.
  - First hard run: 14/20. Fork tipped 4×: the taker's closing finger twisted the fork in the
    giver's grip. Tried "giver lets go as taker closes" → forks dropped (0/20!), reverted.
    Fix: taker lines up 1.5 mm away instead of 4 mm → **hard 18/20, normal 20/20**.
  - Cutlery starting very near an arm's base: the near end can't be reached pointing down;
    now falls back to the far end.
- **Still on the user:** lablab + Discord sign-up, team, teammate, Kaggle account, GitHub repo.
