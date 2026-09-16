# Two-Arm Table Setter

**Two simulated SO-101 robot arms that set a dinner table from a spoken or typed command —
seeing with a camera, understanding with a vision-language model, handing things to each other,
and fixing their own mistakes. Everything runs on an ordinary Intel laptop with OpenVINO —
measured end to end on an **Intel Core Ultra 7 155H** (CPU + Arc iGPU + AI Boost NPU) and on a
2020 i7-1165G7 with Iris Xe. No GPU card, no cloud.**

Built for the AI Infra Summit Hackathon (lablab.ai) — Intel online track: *Bimanual VLA
Manipulation with Multi-Modal Reasoning*.

> 🎬 **Demo page with the 10-table video:** https://pegbitstudio.github.io/two-arm-table-setter/

## What it does

```
 "put the red thing top left of the plate      overhead camera (colour + depth)
  and the grey utensil on the right"                        │
          │ (voice: Speechmatics, 2.6 s)                     ▼
          ▼                                   ┌──────────────────────────┐
 ┌───────────────────────────┐   picture      │ EYES  perception.py      │
 │ UNDERSTAND  Qwen3-VL-4B    │ ◄───────────── │ finds every object from  │
 │ INT4 on OpenVINO, iGPU     │                │ height map + shape;      │
 │ "red thing" = mug          │                │ 0.5 mm mean error        │
 └────────────┬──────────────┘                └────────────┬─────────────┘
              │ goals: mug → top_left, fork → right         │ positions
              ▼                                             ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ PLAN  planner.py — which arm, hand-off or set-down swap, what first │
 │ (plate goes to the middle first if everything is placed relative to it)
 └────────────┬────────────────────────────────────────────────────────┘
              ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ ACT   two SO-101 arms in MuJoCo — pick, place, hand-off (both arms  │
 │ moving at once); a trained ACT policy (LeRobot → OpenVINO) for the  │
 │ mug skill                                                           │
 └────────────┬────────────────────────────────────────────────────────┘
              ▼
 LOOK AGAIN after every item: anything knocked out of place is put back first
```

## Results

All speed numbers are measured on an **Intel Core Ultra 7 155H** (CPU + Arc iGPU + AI Boost NPU,
Windows 11, OpenVINO 2026.3.1) — raw logs in [bench/core_ultra/](bench/core_ultra/). The 20- and
30-table task scores were run on a 2020 i7-1165G7 with Iris Xe; the 10 hard tables were re-run on
the Core Ultra and gave the same result, including the same single failure, so hardware does not
change behaviour. Success is always judged against simulator ground truth, never the robot's own
camera; each row says which machine it came from where it matters.

| What | Result |
|---|---|
| Full task, "set the table", 30 random tables | **30/30** |
| Harder tables (±4 cm starts, sizes ±5–10%, floor colour), 30 tables | **29/30** |
| Recovery: an item knocked out of place mid-task | noticed and fixed (demo) |
| Camera perception, 20 tables | every object found, 0.5 mm mean / 1.0 mm worst error |
| Spoken command → text (Speechmatics batch API) | "Please set the table." transcribed in 2.6 s; table then set 4/4 |
| Command understanding (Qwen3-VL-4B INT4, OpenVINO GenAI) | 5/5 test commands, **1.3 s** per command on the Arc iGPU (3.6 s on CPU) |
| Trained ACT policy (LeRobot), mug pick-and-place on 20 unseen tables | **19/20 within 1.5 cm** |
| ACT on OpenVINO vs PyTorch, CPU | **2.9× faster** (0.72 ms vs 2.07 ms per call), task success unchanged |
| ACT policy on the **NPU** (AI Boost) | 0.90 ms per call at FP32; at FP16 (1.06 ms) it places **10/10** mugs |
| Full task on the Core Ultra, 10 hard tables | **9/10**, 8 s per table |

### Intel optimisation, measured ([full table](bench/RESULTS.md), `python bench/benchmark.py`)

**ACT policy (5 M parameters) on the Core Ultra 7 155H — ms per call on all three Intel chips,
and task success on 10 unseen tables:**

| Backend | Precision | CPU ms | iGPU ms | NPU ms | Mug within 1.5 cm |
|---|---|---|---|---|---|
| PyTorch | FP32 | 2.07 | – | – | 9/10 |
| OpenVINO | FP32 | 0.83 | 1.49 | 0.90 | 9/10 (CPU) |
| OpenVINO | FP16 | 0.81 | 1.52 | 1.06 | 9/10 CPU · 10/10 iGPU · 10/10 NPU |
| OpenVINO | INT8 (NNCF, calibrated on demos) | **0.72** | 1.42 | 1.10 | 9/10 CPU · 8/10 iGPU · 8/10 NPU |
| OpenVINO | INT8 weights only | 0.74 | 1.93 | 1.04 | 10/10 (CPU) |

Findings: OpenVINO gives a 2.5–2.9× speed-up over PyTorch with no loss of task success, and the
whole policy runs on the NPU at 0.90 ms per call (2.3× PyTorch) — freeing the CPU and iGPU for
perception and the language model. For a model this small the CPU still wins outright: dispatch
to the iGPU or NPU costs more than the maths. INT8 accuracy was tuned — NNCF's default preset
drifted least (MIXED and SmoothQuant were worse). The same table on the older i7-1165G7 (no NPU)
is in [docs/worklog.md](docs/worklog.md): 4.34 ms PyTorch → 1.73 ms OpenVINO, the same 2.5× ratio.

**Qwen3-VL-4B INT4 (Intel's OpenVINO build) on the Core Ultra, seconds per command,
5/5 correct in every row:**

| Picture width | CPU | iGPU (Arc) |
|---|---|---|
| 320 px | 3.0 | 1.1 |
| 480 px (used) | 3.6 | **1.3** |
| 960 px | 10.9 | 2.8 |

The Arc iGPU is 2.8× faster than the CPU at the size we use, and its advantage is reading the
picture (time to first token 0.70 s vs 2.7 s at 480 px; 2.2 s vs 9.9 s at 960 px). Two prompt
changes cut a command from ~30 s to a few seconds: a 480-px picture instead of 960, and plain
`object: place` lines instead of JSON (fewer tokens to write). **The NPU cannot run this model:**
the OpenVINO 2026.3 NPU compiler aborts on Qwen3-VL-4B's vision tower, so `brain/language.py`
detects an NPU request and falls back to the iGPU — see [Honest limitations](#honest-limitations).

## Why it matters (business value)

Service robots in homes, hospitals, hotels and restaurants need three things this project
demonstrates together: **plain-language instructions** anyone can give, **two-handed work**
(passing, holding, placing), and **checking their own work** so a bump doesn't ruin the job.
Two further points lower the cost of getting there:

- **No expensive computer.** Everything, including the vision-language model, runs on an
  ordinary Intel laptop with OpenVINO — the kind of computer that can sit inside a low-cost,
  open-source robot like the SO-101.
- **No hand-recorded training data.** The scripted skills record their own demonstrations
  (1,219 here, overnight on a laptop), and a policy learns from them. Data collection is usually
  the most expensive part of robot learning.

## Intel hardware mapping

| Part | Runs on | Format |
|---|---|---|
| Command understanding — Qwen3-VL-4B | **iGPU** (Arc) by default, 1.3 s/command; CPU also works; the NPU cannot compile it | OpenVINO GenAI, INT4 |
| Mug skill — ACT policy (5 M parameters) | **CPU** by default (0.72 ms, 2.9× PyTorch); also runs on the **NPU** (0.90 ms, 10/10 task) and the iGPU — `run.py --policy-device NPU` | OpenVINO IR, FP16 (FP32/INT8 also exported) |
| Perception — colour + depth analysis | CPU — 155 ms per look at 960×720 colour + depth | NumPy/SciPy (no neural network) |
| Physics simulation | CPU | MuJoCo 3.13 |

All three Intel chips are used and measured: CPU for the policy and physics, Arc iGPU for the
vision-language model, NPU verified end to end on the policy (`bench/core_ultra/npu_policy_run.log`).

## Training approach

1. **Teacher:** the scripted skills (camera-driven pick, place, hand-off) perform the mug task
   on random tables; each run is sampled 20×/s as (joint angles, task inputs) → (hand target).
2. **Filter:** runs that miss (judged by simulator truth) are thrown away — 1,219 kept.
3. **Student:** LeRobot's ACT (action chunking transformer), 5 M parameters, 1-second action
   chunks, trained 9,000 steps on a laptop CPU.
4. **What made it work** (7 versions, all in [docs/worklog.md](docs/worklog.md)): one consistent
   grip rule for the teacher; absolute hand targets instead of small moves (which drift);
   everything measured from the mug, so picking is a fixed offset and placing is given.
5. **Deployment:** exported to OpenVINO; used by the full robot with `run.py --policy` where the
   target lies in the area it was trained on.

## Robustness

- **Randomised every run (seeded):** start positions, object mass (0.7–1.4×), friction
  (0.8–1.2×), light brightness and direction, table colour. `--hard` adds ±4 cm starts, object
  sizes ±5–10% and floor colour — the robot still assumes standard sizes.
- **Closed loop:** after every item the robot looks again; failed grasps are retried (up to 3×)
  and anything knocked out of place is put back first.
- **Measured, not claimed:** `brain/evaluate.py` scores 30 tables against simulator truth;
  `brain/make_video.py` films 10 with result cards.

## Tools used — and Intel resources we didn't use

Used: **MuJoCo**, **Hugging Face LeRobot** (ACT), **OpenVINO** (runtime, GenAI, model conversion),
**NNCF** (INT8), Intel's **Qwen3-VL-4B INT4** OpenVINO build, SO-101 model from TheRobotStudio.
Not used: **Intel Physical AI Studio** and **Edge AI Suites** (their install scripts are
Ubuntu-only; our laptop runs Windows), **Intel Geti** (for training camera models; our
perception is geometric and needs no training). The brief lists these as optional resources.

## How it maps to Intel's scoring

| Criterion | Where |
|---|---|
| End-to-end task + bimanual manipulation (30) | `brain/run.py`, `brain/evaluate.py`; hand-offs in `sim/skills.py` |
| VLA / multi-modal reasoning (20) | `brain/language.py` (sees the picture), `brain/planner.py` (re-plans after every look) |
| Robustness across 10 seeds (15) | `brain/evaluate.py --hard`, recovery demo `run.py --bump plate` |
| OpenVINO & Intel Core Ultra optimisation (20) | `bench/export_act.py`, `bench/benchmark.py`; policy on CPU / Arc iGPU / **NPU**, VLM on the Arc iGPU, all measured on a Core Ultra 7 155H — logs in `bench/core_ultra/` |
| Technical quality & reproducibility (10) | seeded scenes, one-command scripts, [docs/setup.md](docs/setup.md) |
| Innovation (5) | teacher-generated training data, closed-loop recovery, laptop-only |

## Run it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt
python brain/get_model.py                               # 3.1 GB, once
python brain/run.py --seed 3 "set the table"
python brain/run.py --seed 7 "put the red thing top left of the plate and the grey utensil on the right"
python brain/run.py --seed 3 --bump plate --video "set the table"   # recovery demo + GIF
python brain/evaluate.py --seeds 10 --hard              # score over 10 random tables
python bench/benchmark.py                               # Intel inference benchmark
```

Have an Intel Core Ultra? [docs/core_ultra_guide.md](docs/core_ultra_guide.md) runs every test in one
command (`python bench/volunteer.py`). The trained policy downloads with `python bench/get_policy.py`.

Training the policy (CPU is enough): `train/record_demos.py` → `train/train_act.py` →
`train/eval_policy.py`. Details in [docs/setup.md](docs/setup.md).

## Honest limitations

- **The NPU cannot run the vision-language model.** On a Core Ultra 7 155H with OpenVINO
  2026.3, compiling Qwen3-VL-4B INT4 for the NPU aborts the process inside Intel's graphics
  compiler ("Channels count of input tensor shape and filter shape must be the same"), so it
  cannot even be caught and retried. `brain/language.py` therefore picks the iGPU up front when
  the NPU is asked for. The ACT policy does run on the NPU, and is benchmarked there.
- **The NPU needs static shapes.** `bench/export_act.py` reshapes the policy to the batch of one
  the robot always uses before compiling for the NPU; dynamic batches are an OpenVINO NPU
  limitation, not a model one.
- **Development hardware.** Most of the build and all the training happened on a 2020 i7-1165G7
  with Iris Xe and no NPU; the Core Ultra 7 155H was borrowed for the final measured run.
- **Scripted hands for most skills.** Pick, place and hand-off are geometric skills driven by
  the camera. One skill (mug pick-and-place) is a trained ACT policy (`run.py --policy`), used
  when the target lies in the area it was trained on; the rest are the teacher that generated
  its training data. Getting the policy to 95% took seven versions — see
  [docs/worklog.md](docs/worklog.md) for what failed and why.
- **Simulation choices.** Toy-scale tableware sized to the SO-101 gripper (max ~5.4 cm): a rimmed
  plate gripped by its rim, chunky cutlery. Box finger pads replace the jaw meshes for contact.
- **Not done:** the drawer and pouring from the example task.

## Credits & licence

MIT (see [LICENSE](LICENSE)). SO-101 model from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0).
Qwen3-VL-4B INT4 by Intel/OpenVINO on Hugging Face. LeRobot by Hugging Face.
Team notes: [docs/working-notes.md](docs/working-notes.md).
