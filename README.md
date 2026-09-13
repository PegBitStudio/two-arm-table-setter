# Two-Arm Table Setter

**Two simulated SO-101 robot arms that set a dinner table from a spoken or typed command —
seeing with a camera, understanding with a vision-language model, handing things to each other,
and fixing their own mistakes. Everything runs on a 2020 Intel laptop (i7-1165G7, Iris Xe) with
OpenVINO. No GPU card, no cloud.**

Built for the AI Infra Summit Hackathon (lablab.ai) — Intel online track: *Bimanual VLA
Manipulation with Multi-Modal Reasoning*.

> 🎬 Demo video: *(link)* · Recovery demo: `brain/out/run_seed3.gif`

## What it does

```
 "put the red thing top left of the plate      overhead camera (colour + depth)
  and the grey utensil on the right"                        │
          │ (voice: Speechmatics)                            ▼
          ▼                                   ┌──────────────────────────┐
 ┌───────────────────────────┐   picture      │ EYES  perception.py      │
 │ UNDERSTAND  Qwen3-VL-4B    │ ◄───────────── │ finds every object from  │
 │ INT4 on OpenVINO, Iris Xe  │                │ height map + shape;      │
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

All numbers from this laptop (Intel Core i7-1165G7, Iris Xe iGPU, 16 GB, no NPU).
Success is always judged against simulator ground truth, never the robot's own camera.

| What | Result |
|---|---|
| Full task, "set the table", 30 random tables | **30/30** |
| Harder tables (±4 cm starts, sizes ±5–10%, floor colour), 30 tables | **29/30** |
| Recovery: an item knocked out of place mid-task | noticed and fixed (demo) |
| Camera perception, 20 tables | every object found, 0.5 mm mean / 1.0 mm worst error |
| Command understanding (Qwen3-VL-4B INT4, OpenVINO GenAI) | 5/5 test commands, **~4 s** per command (CPU or iGPU) |
| Trained ACT policy (LeRobot), mug pick-and-place on 20 unseen tables | **19/20 within 1.5 cm** |
| ACT on OpenVINO vs PyTorch, CPU | **2.5× faster** (1.7 ms vs 4.3 ms per call), task success unchanged |

### Intel optimisation, measured ([full table](bench/RESULTS.md), `python bench/benchmark.py`)

**ACT policy (5 M parameters), ms per call and task success on 10 unseen tables:**

| Backend | Precision | CPU ms | iGPU ms | Mug within 1.5 cm |
|---|---|---|---|---|
| PyTorch | FP32 | 4.34 | – | 9/10 |
| OpenVINO | FP32 | **1.73** | 12.4 | 9/10 |
| OpenVINO | FP16 | 1.88 | 14.6 | 9/10 |
| OpenVINO | INT8 (NNCF, calibrated on demos) | 3.68 | 12.9 | 9/10 |
| OpenVINO | INT8 weights only | 3.53 | 23.5 | 10/10 |

Findings: OpenVINO gives a 2.5× speed-up with identical behaviour. For a model this small,
INT8 and the iGPU don't pay off: quantise/dequantise and GPU dispatch cost more than the maths.
Optimisation never hurt the task. INT8 accuracy was tuned: NNCF's default preset drifted least
(MIXED and SmoothQuant were worse).

**Qwen3-VL-4B INT4 (Intel's OpenVINO build), seconds per command, 5/5 correct in every row:**

| Picture width | CPU | iGPU (Iris Xe) |
|---|---|---|
| 320 px | 3.8 | 4.0 |
| 480 px (used) | 4.3 | 4.2 |
| 960 px | 14.9 | 7.5 |

The iGPU's advantage is reading the picture (time to first token 1.6 s vs 2.7 s at 480 px;
4.8 s vs 13.2 s at 960 px). Two prompt changes cut a command from ~30 s to ~4 s: a 480-px
picture instead of 960, and plain `object: place` lines instead of JSON (fewer tokens to write).

## How it maps to Intel's scoring

| Criterion | Where |
|---|---|
| End-to-end task + bimanual manipulation (30) | `brain/run.py`, `brain/evaluate.py`; hand-offs in `sim/skills.py` |
| VLA / multi-modal reasoning (20) | `brain/language.py` (sees the picture), `brain/planner.py` (re-plans after every look) |
| Robustness across 10 seeds (15) | `brain/evaluate.py --hard`, recovery demo `run.py --bump plate` |
| OpenVINO & Intel optimisation (20) | `bench/export_act.py`, `bench/benchmark.py`, VLM on iGPU |
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

Training the policy (CPU is enough): `train/record_demos.py` → `train/train_act.py` →
`train/eval_policy.py`. Details in [docs/setup.md](docs/setup.md).

## Honest limitations

- **Hardware.** Tested on an 11th-gen Core i7 with Iris Xe — not the Core Ultra the brief
  prefers, and no NPU. The brief allows "Intel CPU and iGPU"; the NPU path is untested.
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
