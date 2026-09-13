"""Day 1 check: load one SO-101 arm in MuJoCo, make it wave, and save a GIF.

    python sim/day1_wave.py          # saves sim/out/day1_wave.gif
    python sim/day1_wave.py --view   # opens a live 3D window instead
"""
import argparse
import math
import time
from pathlib import Path

import imageio.v3 as iio
import mujoco

HERE = Path(__file__).parent
SCENE = HERE / "assets" / "so101" / "scene_one_arm.xml"
OUT = HERE / "out" / "day1_wave.gif"

SECONDS = 4.0
FPS = 25


def wave_targets(t: float) -> dict[str, float]:
    """Joint angles (radians) for a simple wave at time t."""
    return {
        "shoulder_pan": 0.6 * math.sin(2 * math.pi * 0.5 * t),
        "shoulder_lift": -1.1,  # negative lifts the arm up
        "elbow_flex": -0.4 + 0.3 * math.sin(2 * math.pi * 1.0 * t),
        "wrist_flex": 0.5 * math.sin(2 * math.pi * 1.0 * t),
        "wrist_roll": 0.0,
        "gripper": 0.8 + 0.8 * math.sin(2 * math.pi * 1.5 * t),  # open/close
    }


def apply(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name, value in wave_targets(data.time).items():
        data.actuator(name).ctrl = value


def record(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    frames = []
    steps_per_frame = int(1 / (FPS * model.opt.timestep))
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.2]
    cam.distance, cam.azimuth, cam.elevation = 0.9, 150, -15
    with mujoco.Renderer(model, 480, 640) as renderer:
        while data.time < SECONDS:
            for _ in range(steps_per_frame):
                apply(model, data)
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
    OUT.parent.mkdir(exist_ok=True)
    iio.imwrite(OUT, frames, duration=1000 / FPS, loop=0)
    print(f"Saved {len(frames)} frames to {OUT}")


def view(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            start = time.time()
            apply(model, data)
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="open a live window")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    print(f"Loaded arm: {model.nu} motors, {model.njnt} joints, timestep {model.opt.timestep}s")
    view(model, data) if args.view else record(model, data)


if __name__ == "__main__":
    main()
