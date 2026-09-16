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
- [x] Convert the policy to OpenVINO FP32/FP16/INT8/INT8-weights (`bench/export_act.py`)
- [x] VLM benchmark: 30 s → **~4 s per command** (CPU or iGPU), 5/5 correct
- [x] ACT benchmark: OpenVINO CPU **2.5× faster** than PyTorch, task success unchanged
- [x] Final policy v7 @ 9k steps: **19/20 within 1.5 cm (95%)**; wired in as `run.py --policy`

### Day 5 · Sun 13 Sep — Test and fix
- [x] Recovery demo: knock an item, robot sees and fixes it (`run.py --bump plate`)
- [x] Harder randomness (`--hard`): ±4 cm starts, sizes ±5–10%, floor colour. Robot uses standard sizes.
- [x] Fixes: gentle hand-off (taker lines up 1.5 mm away), far-end grip near an arm's base
- [x] Both arms move together during hand-offs
- [x] Scores (30 tables each): **normal 30/30, hard 29/30**
- [ ] Freeze the code at 10 PM Tue — no new features after this

### Day 6 · Package and submit (*kit built Sun 13 Sep*)
- [x] Demo video: 10 hard tables, AI model reading the command → **10/10** (`brain/out/demo_10_tables.mp4`)
- [x] Slides (7, narration in notes), cover image, README with architecture, results and setup
- [x] Public GitHub repo: https://github.com/PegBitStudio/two-arm-table-setter
- [x] lablab form text: `submission/lablab_form.md`
- [x] Close the final-audit gaps below (all but narration and the user's lablab actions)
- [x] **Core Ultra 7 155H run done (15 Sep)** — CPU + Arc iGPU + AI Boost NPU benchmarked, task
      checked on all three, final video rebuilt; logs in `bench/core_ultra/`
- [ ] Clean-machine test of the setup steps if possible
- [ ] **User: submit on lablab by 3:00 PM WAT Wed 16 Sep** (deadline 7:30 PM)

## Final audit — requirements vs what we have (13 Sep)

| Requirement (source) | Status |
|---|---|
| Intel 1 — reproducible GitHub repo (setup, deps, scene, training, eval, inference, commands) | ✓ |
| Intel 2 — reproducible MuJoCo dual-arm dinner-table sim, randomisation, eval config | ✓ `scene.py --hard`, `evaluate.py` |
| Intel 3 — benchmark script: latency, throughput, device, precision, on Core Ultra 2/3 | ✓ latency + throughput on **Core Ultra 7 155H**, CPU / Arc iGPU / NPU, at FP32, FP16, INT8, INT8-weights |
| Intel 4 — video, 10 randomised seeds, command + scene variation + outcome clear | ✓ `submission/two_arm_table_setter_demo.mp4`: 10/10 tables → recovery → benchmark cards (5:41) |
| Intel 5 — technical README: architecture, model choice, bimanual strategy, training, robustness, OpenVINO, hardware mapping | ✓ all sections added |
| Intel — "run final simulation on Core Ultra 2/3" | ✓ full robot on 10 hard tables on a Core Ultra 7 155H (9/10, 8 s/table), policy on the NPU end to end |
| Intel — preserve task success after optimisation | ✓ `act_task` table |
| Intel scenario — drawer, pouring | ✗ cut (optional "challenge option") |
| lablab — title, short/long description, tags | ✓ `lablab_form.md` |
| lablab — cover image, slides | ✓ |
| lablab — video presentation | ✓ demo video; a narrated pitch video would be better |
| lablab — demo application platform + application URL | ✓ GitHub Pages: https://pegbitstudio.github.io/two-arm-table-setter/ |
| lablab judging — business value | ✓ slide 3, README, form text |
| lablab — MIT licence, original work | ✓ |
| lablab — team on lablab, Discord | ✗ user |
| Speechmatics bonus — voice agent | ✓ batch API tested: "Please set the table." in 2.6 s → table set 4/4; live listening not built yet |

## Decisions still open

1. **Windows or WSL2 Ubuntu?** Intel's install scripts are Ubuntu-only. MuJoCo and OpenVINO also
   run on plain Windows. **Decided 11 Sep: plain Windows.** MuJoCo 3.13 and OpenVINO 2026.3 both
   work, and OpenVINO sees CPU + Iris Xe GPU. Switch to WSL2 only if LeRobot training breaks.
2. **Which vision-language model?** Decided Day 3: Qwen3-VL-4B INT4 (Intel's OpenVINO build).
3. **Demo "app URL".** A simulation can't really be hosted for free. Plan: a GitHub Pages page
   with the video and results. **Still open** — see Final audit.

## Where we'll lose points (and accept it)

- ~~**Core Ultra part of the 20 OpenVINO points.**~~ **Closed 15 Sep** — a Core Ultra 7 155H was
  borrowed and every benchmark re-run on its CPU, Arc iGPU and AI Boost NPU. One honest caveat
  stays in the README: OpenVINO 2026.3's NPU compiler cannot build Qwen3-VL-4B's vision tower, so
  the language model runs on the iGPU. The policy runs on the NPU.
- **Pouring and drawer.** Hard to simulate well. Stretch goals only.

## Team

Useful split for two people:
- **Person 1 — sim and arms:** MuJoCo scene, skills, random seeds, test runs.
- **Person 2 — AI and speed:** vision-language model, training, OpenVINO, benchmark, voice.
- **Both:** video, slides, README.

A teammate with a Core Ultra laptop would win back most of the lost points. **Done 15 Sep** — the
user's brother lent a Core Ultra 7 155H and the full suite was re-run on it.
