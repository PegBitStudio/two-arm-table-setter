# Project Plan

**Last updated:** 2026-09-11
**Deadline:** Wed 16 Sep 2026, 7:30 PM WAT. **Our own target: submit by 3:00 PM** (4½ hours of buffer).

## Goal

Two simulated robot arms that hear or read a dinner-table command, look at the scene, plan the
steps, and carry them out. Show it working across 10 random scene setups.

Main aim is **learning** (robot simulation, vision-language models, OpenVINO). Winning is a bonus.

## Rules we set ourselves

- **Zero spend.** Free tools only. Training on free GPUs (Kaggle — weekly quota, check the
  current hours when signing up — or Google Colab). No paid cloud.
- **AssemblyAI build (due 30 Sep) comes first** if time gets squeezed.
- **Something working every night.** Each day ends with a runnable version saved to GitHub.

## How it works (the simple version)

```
  voice / text command
          │
          ▼
  ┌──────────────────┐   camera image    ┌──────────────┐
  │  Brain            │ ◄──────────────── │  MuJoCo scene │
  │  small vision-    │                   │  2 × SO-101   │
  │  language model   │ ── step list ───► │  table, plate,│
  │  (OpenVINO)       │                   │  mug, cutlery │
  └──────────────────┘                   └──────▲───────┘
          │ one step at a time                   │
          ▼                                      │
  ┌──────────────────┐        arm movements      │
  │  Hands            │ ──────────────────────────┘
  │  pick / place /   │
  │  hand-off skills  │
  └──────────────────┘
```

- **Brain** (earns the 20 "reasoning" points): a small vision-language model reads the command and
  the camera image, then outputs the next step, e.g. `{arm: "A", action: "pick", object: "plate"}`.
  After each step it looks again and checks the step worked.
- **Hands** (earns the 30 "task completion" points): reliable pick, place and hand-off moves.
  Start with scripted moves so the demo definitely works. Then replace one skill with a trained
  policy (ACT via LeRobot) to earn the "training" points.
- **Speed** (earns the 20 "OpenVINO" points): convert the models to OpenVINO, shrink them (INT8),
  and measure speed on the laptop's CPU vs. its Iris Xe graphics chip.
- **Voice** (bonus prize): Speechmatics turns spoken commands into text.

## Day by day

### Day 1 · Fri 11 Sep — Set up (evening)
- [ ] Register on lablab.ai + lablab Discord; create the team on lablab
- [ ] Find a teammate (see Team below)
- [x] Choose Windows or WSL2 Ubuntu → **plain Windows** (see Decision 1)
- [x] Install Python 3.12, MuJoCo, OpenVINO — see [setup.md](setup.md)
- [ ] Open a free Kaggle account (needed Day 4)
- [x] Find a ready-made SO-101 model file → TheRobotStudio SO-ARM100 (Apache-2.0)

**End of day:** one SO-101 arm moving on screen. ✅ `sim/day1_wave.py`

### Day 2 · Sat 12 Sep — Scene and hands
- [x] Two arms facing each other across a table
- [x] Add plate, mug, fork, spoon, bottle — drawer and pouring cut by choice (the gate was
  passed, but they'd eat Day 3–4 time; bottle stays as a distractor object)
- [x] Scripted skills: move to a point, grab, release, hand an object from arm A to arm B
- [x] Random scene setup from a seed number (object positions, lighting, friction, weight, table colour)

**End of day:** one fixed sequence runs start to finish with no AI. ✅ `sim/day2_demo.py` —
**24/30 seeds set the whole table** (80%) after the audit fixes. Failure list is in the worklog for Day 5.
**Gate:** if the arms can't reliably pick up a plate by Saturday night, cut the drawer and the pouring.

### Day 3 · Sun 13 Sep — Brain and voice
- [x] Vision-language model: **Qwen3-VL-4B INT4** (Intel's own OpenVINO conversion), runs on the Iris Xe GPU
- [x] Prompt: command + camera image → goals as JSON (object → place); keyword reader as fallback
- [x] Planner picks arms, hand-offs, and a set-down swap for round things; loop: act → look again → retry
- [x] Speechmatics voice input — **written, untested: needs a free API key (user)**
- [x] **Eyes for the hands:** skills now use the overhead camera (colour + depth), never
  simulator positions. Held objects are tracked from the hand's own pose.

**End of day:** type a command, the arms do it. ✅ `brain/run.py` — **29/30 random tables set
("set the table"), 0 retries needed**; `brain/evaluate.py` scores it.

### Day 4 · Mon 14 Sep — Training and speed
*Actual: Sat 12 – Sun 13 Sep (started early; ran long).*
- [x] Record demos automatically — scripted skills as teacher, 1,219 mug pick-and-place episodes
- [x] Train ACT with LeRobot — **on the laptop CPU, no Kaggle needed**. Took 7 versions to get
  right (see worklog table). v7 (mug-relative hand targets): picks and places upright every time;
  placement ~2 cm off at step 3k, still training.
- [ ] Convert the policy to OpenVINO FP32/FP16/INT8 (`bench/export_act.py` ready, waiting on final v7)
- [x] VLM benchmark on Iris Xe: 30 s → **5.2 s per command**, 5/5 correct
- [ ] ACT + CPU-vs-GPU benchmark rows (after v7)

### Day 5 · Sun 13 Sep — Test and fix
- [x] Recovery demo: knock an item, robot sees and fixes it (`run.py --bump plate`)
- [x] Harder randomness (`--hard`): ±4 cm starts, sizes ±5–10%, floor colour. Robot uses standard sizes.
- [x] Fixes: gentle hand-off (taker lines up 1.5 mm away), far-end grip near an arm's base
- [x] Both arms move together during hand-offs
- [x] Scores (30 tables each): **normal 30/30, hard 29/30**
- [ ] Freeze the code at 10 PM Tue — no new features after this

### Day 6 · Wed 16 Sep — Package and submit
- [ ] Record the demo video (command → 10 seeds → results → benchmark), about 3 minutes
- [ ] Slides (5–7), cover image, README with architecture and setup steps
- [ ] Clean-machine test of the setup steps if possible
- [ ] **Submit on lablab by 3:00 PM WAT**

## Decisions still open

1. **Windows or WSL2 Ubuntu?** Intel's install scripts are Ubuntu-only. MuJoCo and OpenVINO also
   run on plain Windows. **Decided 11 Sep: plain Windows.** MuJoCo 3.13 and OpenVINO 2026.3 both
   work, and OpenVINO sees CPU + Iris Xe GPU. Switch to WSL2 only if LeRobot training breaks.
2. **Which vision-language model?** Needs to run on 16 GB RAM with no graphics card. Decide on Day 3
   after a quick speed test.
3. **Demo "app URL".** A simulation can't really be hosted for free. Likely a GitHub Pages page with
   the video and results. Confirm on Discord.

## Where we'll lose points (and accept it)

- **Core Ultra part of the 20 OpenVINO points.** Our laptop is an 11th-gen i7 with no AI chip (NPU).
  The brief does allow "Intel CPU and iGPU", so we still show CPU vs. iGPU results. Say this openly
  in the README.
- **Pouring and drawer.** Hard to simulate well. Stretch goals only.

## Team

Useful split for two people:
- **Person 1 — sim and arms:** MuJoCo scene, skills, random seeds, test runs.
- **Person 2 — AI and speed:** vision-language model, training, OpenVINO, benchmark, voice.
- **Both:** video, slides, README.

A teammate with a Core Ultra laptop would win back most of the lost points. Worth asking for on Discord.
