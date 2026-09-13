"""Rebuild a small-moves ("ee") dataset as a hand-target ("abs") dataset — no re-simulation.

Every small move was the difference between two consecutive commanded hand poses, so adding
them back up from the start pose recovers each commanded pose exactly. The mug/target inputs
were stored relative to the fingertip; the fingertip comes from the recorded joint angles
(forward kinematics), so absolute positions are recovered too.

    python train/convert_to_absolute.py --src train/data/mug_pick_place_ee_v4 --dst train/data/mug_pick_place_abs

Check built in: every episode ends with the arm back home, so the rebuilt pose at the last
frame must equal the home pose. The worst mismatch is printed.
"""
import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import record_demos as R
import day2_demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=R.DATA.parent / "mug_pick_place_ee_v4")
    p.add_argument("--dst", type=Path, default=R.DATA.parent / "mug_pick_place_abs")
    args = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset(f"local/{args.src.name}", root=args.src)
    sim = day2_demo.start_sim(0, video=False, eyes="truth")  # only for the arm's kinematics
    arm = sim.arms[R.ARM]
    fk = mujoco.MjData(sim.m)

    def tip_pose(q5):
        fk.qpos[:] = sim.d.qpos
        fk.qpos[arm.qadr] = q5
        mujoco.mj_kinematics(sim.m, fk)
        return fk.site_xpos[arm.site].copy(), fk.site_xmat[arm.site].reshape(3, 3).copy()

    home_p, home_R = tip_pose(sim.home_q[R.ARM])
    if args.dst.exists():
        shutil.rmtree(args.dst)
    dst = LeRobotDataset.create(repo_id=f"local/{args.dst.name}", fps=R.FPS, features=R.features("abs"),
                                root=args.dst, robot_type="so101_bimanual_sim", use_videos=False)
    eps = src.meta.episodes
    worst = 0.0
    for ep in range(src.num_episodes):
        lo, hi = int(eps["dataset_from_index"][ep]), int(eps["dataset_to_index"][ep])
        cmd_p, cmd_R = home_p.copy(), home_R.copy()
        for i in range(lo, hi):
            f = src[i]
            state = f["observation.state"].numpy()
            rel = f["observation.environment_state"].numpy()
            d = f["action"].numpy()
            tip, _ = tip_pose(state[:5])
            cmd_p = cmd_p + d[:3]
            cmd_R = Rotation.from_rotvec(d[3:6]).as_matrix() @ cmd_R
            dst.add_frame({
                "observation.state": state,
                "observation.environment_state": np.array(
                    [rel[0] + tip[0], rel[1] + tip[1], rel[2] + tip[0], rel[3] + tip[1], *tip], np.float32),
                "action": np.array([*cmd_p, *cmd_R[:, 0], *cmd_R[:, 1], d[6]], np.float32),
                "task": R.TASK,
            })
        dst.save_episode()
        worst = max(worst, float(np.linalg.norm(cmd_p - home_p)))
        if (ep + 1) % 50 == 0:
            print(f"{ep + 1}/{src.num_episodes} episodes converted")
    dst.finalize()
    day2_demo.close_sim(sim)
    print(f"done: {dst.num_episodes} episodes -> {args.dst}")
    print(f"check: rebuilt final hand pose vs home, worst {1000 * worst:.2f} mm (should be ~0)")


if __name__ == "__main__":
    main()
