"""How closely does the trained policy copy the teacher? Feeds recorded states in (open loop)
and compares the policy's first predicted action with what the teacher actually did.

    python train/check_fit.py
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import record_demos as R
from eval_policy import RUN, load_torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    from export_act import training_data

    data = training_data(args.run)
    ds = LeRobotDataset(f"local/{data.name}", root=data)
    policy, pre, post = load_torch(args.run / "checkpoints" / "last" / "pretrained_model")
    errs = []
    idx = ds.meta.episodes  # per-episode start/end indices
    with torch.no_grad():
        for ep in range(args.episodes):
            lo, hi = int(idx["dataset_from_index"][ep]), int(idx["dataset_to_index"][ep])
            for i in range(lo, hi, 5):
                f = ds[i]
                b = pre({"observation.state": f["observation.state"],
                         "observation.environment_state": f["observation.environment_state"], "task": R.TASK})
                policy.reset()
                a = post(policy.select_action(b)).squeeze(0).numpy()
                errs.append(np.abs(a - f["action"].numpy()))
    e = np.array(errs)
    if e.shape[1] == len(R.ABS_NAMES):
        names, unit = R.ABS_NAMES, "metres / rotation-matrix entries"
    elif e.shape[1] == len(R.EE_NAMES):
        names, unit = R.EE_NAMES, "metres / radians per tick"
    else:
        names, unit = [n.replace("b_", "") for n in R.FEATURES["action"]["names"]], "radians"
    print(f"mean |policy - teacher| per action ({unit}):")
    for n, m, q in zip(names, e.mean(0), np.percentile(e, 90, axis=0)):
        print(f"  {n:14s} mean {m:.4f}   90th percentile {q:.4f}")
    if e.shape[1] == 6:  # joint-angle policy
        # Rough fingertip error: ~0.3 m lever on the shoulder joints.
        print(f"=> fingertip error roughly {1000 * 0.3 * e[:, :3].mean():.1f} mm (mean)")


if __name__ == "__main__":
    main()
