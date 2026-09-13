"""How accurate are the eyes? Compares what the camera finds with the simulator's truth.

    python brain/test_perception.py --seeds 10
"""
import argparse

import numpy as np

import __init__  # noqa: F401  (adds sim/ to the path)
import day2_demo
import skills


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()
    errs, yaw_errs, missing = [], [], []
    for seed in range(args.seeds):
        sim = day2_demo.start_sim(seed, video=False, eyes="camera")
        seen = sim.eyes.look(sim.d)
        day2_demo.close_sim(sim)
        row = []
        for name in ("plate", "mug", "fork", "spoon", "bottle"):
            if name not in seen:
                missing.append((seed, name))
                row.append(f"{name}:MISSING")
                continue
            true_p, R = skills.truth_pose(sim, name)
            e = np.hypot(*(seen[name].xy - true_p[:2]))
            errs.append(e)
            txt = f"{name}:{e * 1000:.1f}mm"
            if name in ("fork", "spoon"):
                d = (seen[name].yaw - skills._yaw_of(R) + np.pi / 2) % np.pi - np.pi / 2
                yaw_errs.append(abs(d))
                txt += f"/{np.degrees(abs(d)):.1f}°"
            row.append(txt)
        extra = set(seen) - {"plate", "mug", "fork", "spoon", "bottle"}
        print(f"seed {seed:2d}  " + "  ".join(row) + (f"  extra:{extra}" if extra else ""))
    print(f"\nposition error: mean {np.mean(errs) * 1000:.1f} mm, worst {np.max(errs) * 1000:.1f} mm")
    if yaw_errs:
        print(f"cutlery angle error: mean {np.degrees(np.mean(yaw_errs)):.1f}°, worst {np.degrees(np.max(yaw_errs)):.1f}°")
    print(f"missed: {missing or 'none'}")


if __name__ == "__main__":
    main()
