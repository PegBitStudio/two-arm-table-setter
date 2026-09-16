# Setup

Tested on Windows 11 on two machines: an Intel Core Ultra 7 155H (CPU + Arc iGPU + AI Boost NPU,
the machine the published benchmarks come from) and a 2020 i7-1165G7 with Iris Xe, 16 GB RAM.
Plain Windows, no WSL needed.

## 1. Python 3.12 environment

Any machine (Windows, Linux, macOS) with Python 3.10–3.12. Python 3.13+ is too new for some
of the AI libraries. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
```

Or plain pip: `python3.12 -m venv .venv` then `.venv/bin/pip install -r requirements.txt`
(`.venv\Scripts\pip` on Windows). Commands below write `python` for the environment's Python.

*On the team laptop* the environment lives at `C:\Users\USER\.venvs\ai-infra` instead of
`.venv`, because the project folder is cloud-synced and a venv inside it would upload
thousands of files.

## 2. Check it works

```bash
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

The old laptop prints `['CPU', 'GPU']` (GPU = Iris Xe; no NPU). The Core Ultra 7 155H prints
`['CPU', 'GPU', 'NPU']` — GPU = Arc Graphics, NPU = Intel AI Boost. The ACT policy runs on all
three (`run.py --policy --policy-device NPU`); the vision-language model cannot compile for the
NPU on OpenVINO 2026.3 and falls back to the GPU automatically.

```bash
python sim/day1_wave.py
```

Saves `sim/out/day1_wave.gif` — one SO-101 arm waving. Add `--view` for a live 3D window.

## Robot model

`sim/assets/so101/` is copied from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(`Simulation/SO101`, new calibration), Apache-2.0 — licence kept alongside.

Joint notes learned on Day 1:
- 6 motors: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
- Zero = middle of each joint's range. `shoulder_lift` negative raises the arm;
  `shoulder_lift=-1.2, elbow_flex=-0.3` stands it upright.
- Position control tracks targets closely; the gripper lags on fast open/close.

## Running the Day 2 table-setting script

```bash
python sim/day2_demo.py --seed 3
python sim/day2_demo.py --seeds 30 --no-video
```

The first saves `sim/out/day2_seed3.gif` (~2½ min, rendering is the slow part). The second
prints pass/fail per seed with a reason for each misplaced item (~3 s per seed).

Gripper facts learned on Day 2 (gripper-site frame: x points out of the fingers, z is the
closing direction, fixed finger's inner face sits on the site):
- Gap between the finger pads: ~0.3 cm closed (`-0.17`), 1.6 cm at `0`, 3.6 cm at `0.5`,
  5.4 cm fully open (`1.0`). `robot.grip_for_gap()` converts. The gripper opens only ~1 cm
  wider than the object to pick and ~6 mm wider to let go, because the moving finger swings
  in an arc and bumps neighbours.
- Top-down reach: table level out to ~27 cm from the base; straight-down pointing is only
  possible up to ~8 cm above the table (wrist limit). Carry height is 6 cm.

## 3. The AI model (Day 3)

Qwen3-VL-4B, already converted to OpenVINO INT4 by Intel — 3.1 GB, free:

```bash
python brain/get_model.py
```

`brain/language.py` looks for it in `~/.models/Qwen3-VL-4B-Instruct-int4-ov`. On the team laptop
(Iris Xe GPU) it loads in 20–70 s and reads a command in ~18–30 s; on the Core Ultra's Arc iGPU
it loads in 13 s and reads a command in 1.3 s.

## 4. Voice (optional, Speechmatics bonus prize)

Get a free API key at speechmatics.com, then set it before running:

```bash
set SPEECHMATICS_API_KEY=your-key        # Windows cmd
export SPEECHMATICS_API_KEY=your-key     # Linux/macOS
```

## Running the robot

```bash
python brain/run.py --seed 3 "set the table"
python brain/run.py --seed 7 "put the red thing top left of the plate and the grey utensil on the right"
python brain/run.py --seed 3 --audio command.wav       # spoken command
python brain/run.py --seed 3 --rules "set the table"   # keyword reader, no AI model needed
python brain/evaluate.py --seeds 10                    # score over 10 random tables
python brain/test_perception.py --seeds 10             # how accurate the camera is
```

Add `--video` to `run.py` to save `brain/out/run_seed<N>.gif` (a few minutes — rendering is slow).

## Training the ACT policy (CPU, no GPU needed)

```bash
python train/record_demos.py --episodes 270 --mode abs --seed-start 100000 --out train/data/mug_abs_p1
python train/merge_datasets.py --out train/data/mug_abs_all train/data/mug_abs_p1 ...
python train/convert_to_mug_frame.py --src train/data/mug_abs_all --dst train/data/mug_rel_all
python train/train_act.py --steps 9000 --data train/data/mug_rel_all   # ~2.5 h on 4 cores
python train/eval_policy.py --episodes 20                              # unseen tables
python bench/export_act.py                                             # OpenVINO FP32/FP16/INT8
```

Runs and checkpoints go to `~/.models/runs/act_mug` (outside the project: they're large).
Windows allows one process ~300 render windows, so record in batches of ≤270 episodes
(run several recorders in parallel with different `--seed-start`).

## Benchmark and video

```bash
python bench/benchmark.py                      # -> bench/RESULTS.md, bench/results.json (~40 min)
python brain/make_video.py --seeds 10 --start 100 --hard --model   # -> brain/out/demo.mp4 (~1 h)
python submission/make_cover.py                # cover image
```

Long jobs stall if the PC sleeps — keep it awake while they run.

## Renderer gotcha (Windows)

`mujoco.Renderer.close()` frees its GPU objects in the wrong order, which makes every other
renderer in the process draw black. Always close renderers with `robot.close_renderer()` (the
`Sim.close()` / `Eyes.close()` / `day2_demo.close_sim()` helpers do this).
