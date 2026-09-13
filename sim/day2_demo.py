"""Day 2: set the table with a fixed script (no AI yet).

    python sim/day2_demo.py --seed 3            # one run, saves sim/out/day2_seed3.gif
    python sim/day2_demo.py --seeds 10 --no-video   # success rate over seeds 0-9
    python sim/day2_demo.py --seeds 10 --no-video --eyes truth   # same, cheating eyes (comparison)

Sequence: A moves the plate, B moves the mug, B hands the fork to A, A hands the spoon to B.
"""
import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

import robot
import scene
import skills

OUT = Path(__file__).parent / "out"


def start_sim(seed: int, video: bool, eyes: str = "camera", hard: bool = False) -> robot.Sim:
    """eyes: "camera" (overhead colour + depth, the honest mode) or "truth" (simulator
    positions, only for comparison). hard: wider randomisation (see scene.build)."""
    model, _ = scene.build(seed, hard=hard)
    sim = robot.Sim(model, record_cameras=("front",) if video else (), size=(360, 480))
    homes = {}
    for arm, x in (("A", -0.12), ("B", 0.12)):
        homes[arm], _ = sim.arms[arm].ik(sim.d, np.array([x, 0, skills.CARRY_Z]), np.pi / 2,
                                          q0=np.array([0, 0, 0, 1.5, 0]))
    sim.set_home(homes)
    sim.step(0.4)
    if eyes == "camera":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "brain"))
        from perception import Eyes
        sim.eyes = Eyes(model)
    else:
        sim.eyes = skills.TruthEyes(sim)
    return sim


def close_sim(sim: robot.Sim) -> None:
    sim.close()
    if hasattr(sim.eyes, "close"):
        sim.eyes.close()


def set_table(sim: robot.Sim) -> None:
    """Fixed script. Looks with the camera whenever both arms are home."""
    S = scene.SPOTS
    skills.observe(sim)
    skills.pick(sim, "A", "plate")
    skills.place(sim, "A", "plate", S["plate"])
    skills.home(sim, ["A"])
    skills.observe(sim)
    skills.pick(sim, "B", "mug")
    skills.place(sim, "B", "mug", S["mug"])
    skills.home(sim, ["B"])
    skills.observe(sim)
    skills.handoff(sim, "B", "A", "fork")
    skills.place(sim, "A", "fork", S["fork"], obj_yaw=np.pi / 2)
    skills.home(sim)
    skills.observe(sim)
    skills.handoff(sim, "A", "B", "spoon")
    skills.place(sim, "B", "spoon", S["spoon"], obj_yaw=np.pi / 2)
    skills.home(sim)


def run(seed: int, video: bool = True, eyes: str = "camera") -> dict:
    sim = start_sim(seed, video, eyes)
    error = None
    try:
        set_table(sim)
    except skills.SkillError as e:
        error = str(e)
    sim.step(0.5)
    result = skills.check(sim, scene.SPOTS)
    reasons = {n: skills.why_not(sim, n, xy) for n, xy in scene.SPOTS.items() if not result[n]}
    close_sim(sim)
    if video:
        OUT.mkdir(exist_ok=True)
        # Played at 2x speed: frames are captured at 12 fps of sim time.
        iio.imwrite(OUT / f"day2_seed{seed}.gif", sim.frames["front"], duration=1000 / 24, loop=0)
    return {"seed": seed, "placed": result, "reasons": reasons, "error": error, "log": sim.log,
            "sim_time": sim.d.time}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, help="run seeds 0..N-1 and report the success rate")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--eyes", choices=["camera", "truth"], default="camera")
    args = p.parse_args()

    seeds = range(args.seeds) if args.seeds else [args.seed]
    wins = 0
    for s in seeds:
        t = time.time()
        r = run(s, video=not args.no_video, eyes=args.eyes)
        ok = all(r["placed"].values())
        wins += ok
        items = " ".join(f"{k}:{'ok' if v else 'x'}" for k, v in r["placed"].items())
        print(f"seed {s:2d}  {'PASS' if ok else 'FAIL'}  {items}  ({time.time() - t:.0f}s)"
              + (f"  error: {r['error']}" if r["error"] else ""))
        for name, why in r["reasons"].items():
            print(f"          {name}: {why}")
        for line in r["log"]:
            if "off by" in line:
                print("         ", line)
    print(f"\n{wins}/{len(seeds)} seeds set the whole table")


if __name__ == "__main__":
    main()
