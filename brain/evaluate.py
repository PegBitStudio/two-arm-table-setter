"""Score the whole robot (camera -> brain -> planner -> arms) over many random tables.

    python brain/evaluate.py --seeds 10                       # "set the table", keyword reader
    python brain/evaluate.py --seeds 10 --model               # command read by Qwen3-VL each time
    python brain/evaluate.py --seeds 10 --command "fork right, spoon left"

Success is judged from simulator truth (not the robot's own camera), per object and per table.
"""
import argparse
import time

import numpy as np

import __init__  # noqa: F401
import day2_demo
import planner
import skills
from language import VisionLanguage, read_rules


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--command", default="set the table")
    p.add_argument("--model", action="store_true", help="read the command with Qwen3-VL every run")
    p.add_argument("--device", default="GPU")
    p.add_argument("--hard", action="store_true",
                   help="wider randomisation: ±4 cm starts, sizes ±5-10%%, floor colour")
    args = p.parse_args()

    brain = VisionLanguage(args.device) if args.model else None
    tables, per_obj, retries, secs = 0, {}, 0, []
    for seed in range(args.start, args.start + args.seeds):
        t = time.time()
        sim = day2_demo.start_sim(seed, video=False, eyes="camera", hard=args.hard)
        try:
            skills.observe(sim)
            if brain:
                rgb, *_ = sim.eyes.capture(sim.d)
                goals, _ = brain.read(args.command, (rgb * 255).astype("uint8"))
            else:
                goals = read_rules(args.command)
            report = planner.execute(sim, goals, say=lambda s: None)
            sim.step(0.5)
            plate = planner.plate_xy(sim, report.goals)
            why = {g.obj: skills.why_not(sim, g.obj, g.xy(plate)) for g in report.goals}
        finally:
            day2_demo.close_sim(sim)
        ok = all(v == "" for v in why.values())
        tables += ok
        extra = sum(max(0, n - 1) for n in report.attempts.values())
        retries += extra
        secs.append(time.time() - t)
        for o, v in why.items():
            per_obj.setdefault(o, []).append(v == "")
        fails = "  ".join(f"{o}: {v}" for o, v in why.items() if v)
        print(f"seed {seed:3d}  {'PASS' if ok else 'FAIL'}  retries {extra}  {secs[-1]:.0f}s  {fails}")
    n = args.seeds
    print(f"\n{tables}/{n} tables fully set ({100 * tables / n:.0f}%), {retries} retries in total, "
          f"{np.mean(secs):.0f}s per table")
    print("per object: " + ", ".join(f"{o} {sum(v)}/{len(v)}" for o, v in per_obj.items()))


if __name__ == "__main__":
    main()
