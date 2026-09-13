"""Re-express a hand-target ("abs") dataset relative to the mug's starting position ("mug" mode).
Pure arithmetic on the stored numbers — no simulation.

    python train/convert_to_mug_frame.py --src train/data/mug_abs_all --dst train/data/mug_rel_all
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

import record_demos as R

ARM_B_BASE = np.array([0.24, 0.0])  # scene.BASE_X, arm B


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=R.DATA.parent / "mug_abs_all")
    p.add_argument("--dst", type=Path, default=R.DATA.parent / "mug_rel_all")
    args = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset(f"local/{args.src.name}", root=args.src)
    if args.dst.exists():
        shutil.rmtree(args.dst)
    dst = LeRobotDataset.create(repo_id=f"local/{args.dst.name}", fps=R.FPS, features=R.features("mug"),
                                root=args.dst, robot_type="so101_bimanual_sim", use_videos=False)
    eps = src.meta.episodes
    for ep in range(src.num_episodes):
        lo, hi = int(eps["dataset_from_index"][ep]), int(eps["dataset_to_index"][ep])
        for i in range(lo, hi):
            f = src[i]
            mx, my, tx, ty, px, py, pz = f["observation.environment_state"].numpy()
            act = f["action"].numpy().copy()
            act[0] -= mx
            act[1] -= my
            env = np.array([tx - mx, ty - my, px - mx, py - my, pz,
                            mx - ARM_B_BASE[0], my - ARM_B_BASE[1]], np.float32)
            dst.add_frame({"observation.state": f["observation.state"].numpy(),
                           "observation.environment_state": env, "action": act, "task": R.TASK})
        dst.save_episode()
        if (ep + 1) % 200 == 0:
            print(f"{ep + 1}/{src.num_episodes} episodes converted")
    dst.finalize()
    print(f"done: {dst.num_episodes} episodes -> {args.dst}")


if __name__ == "__main__":
    main()
