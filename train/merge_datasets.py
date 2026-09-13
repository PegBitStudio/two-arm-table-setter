"""Merge several recorded LeRobot datasets (same features) into one.

    python train/merge_datasets.py --out train/data/mug_all train/data/a train/data/b ...

Used to combine demonstrations recorded by parallel recorder processes.
"""
import argparse
import shutil
from pathlib import Path

import record_demos as R


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sources", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    srcs = [LeRobotDataset(f"local/{s.name}", root=s) for s in args.sources]
    features = {k: v for k, v in srcs[0].meta.features.items() if k in R.features("abs")}
    if args.out.exists():
        shutil.rmtree(args.out)
    out = LeRobotDataset.create(repo_id=f"local/{args.out.name}", fps=R.FPS, features=features,
                                root=args.out, robot_type="so101_bimanual_sim", use_videos=False)
    for src in srcs:
        eps = src.meta.episodes
        for ep in range(src.num_episodes):
            lo, hi = int(eps["dataset_from_index"][ep]), int(eps["dataset_to_index"][ep])
            for i in range(lo, hi):
                f = src[i]
                out.add_frame({k: f[k].numpy() for k in features} | {"task": R.TASK})
            out.save_episode()
        print(f"added {src.num_episodes} episodes from {src.root}")
    out.finalize()
    print(f"done: {out.num_episodes} episodes, {out.num_frames} frames -> {args.out}")


if __name__ == "__main__":
    main()
