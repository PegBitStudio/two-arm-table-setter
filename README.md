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
| Command understanding (Qwen3-VL-4B INT4, Iris Xe) | 5/5 test commands, **5.2 s** per command |
| Trained ACT policy (mug pick-and-place, unseen tables) | *TBD* |
| ACT on OpenVINO vs PyTorch (CPU / iGPU, FP32 / FP16 / INT8) | see [bench/RESULTS.md](bench/RESULTS.md) |

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
  the camera. One skill (mug pick-and-place) is a trained ACT policy; the rest are the teacher
  that generated its training data.
- **Simulation choices.** Toy-scale tableware sized to the SO-101 gripper (max ~5.4 cm): a rimmed
  plate gripped by its rim, chunky cutlery. Box finger pads replace the jaw meshes for contact.
- **Not done:** the drawer and pouring from the example task.

## Credits & licence

MIT (see [LICENSE](LICENSE)). SO-101 model from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0).
Qwen3-VL-4B INT4 by Intel/OpenVINO on Hugging Face. LeRobot by Hugging Face.
Team notes: [docs/working-notes.md](docs/working-notes.md).
