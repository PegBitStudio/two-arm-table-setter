"""Tell the robots what to do.

    python brain/run.py --seed 3 "set the table"
    python brain/run.py --seed 3 "put the fork on the right and the spoon on the left"
    python brain/run.py --seed 3 --audio command.wav          # spoken (needs SPEECHMATICS_API_KEY)
    python brain/run.py --seed 3 --rules "set the table"      # keyword reader, no AI model
    python brain/run.py --seed 3 --video "set the table"      # also saves brain/out/run_seed3.gif

Observe (camera) -> Understand (Qwen3-VL on OpenVINO, or keywords) -> Plan (planner) ->
Act (skills) -> Observe again, retrying anything the camera says didn't land.
"""
import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

import __init__  # noqa: F401
import day2_demo
import planner
import scene
import skills
from language import VisionLanguage, read_rules

OUT = Path(__file__).parent / "out"


def bumper(obj: str, distance: float = 0.06):
    """after_goal hook: once `obj` has been placed and one more item is done, flick `obj`
    sideways (towards the front of the table) so it slides ~`distance`. Happens once."""
    state = {"placed": False, "done": False}

    def hook(sim, goal):
        if state["done"]:
            return
        if goal.obj == obj:
            state["placed"] = True
            return
        if state["placed"]:
            adr = sim.m.joint(f"{obj}_free").dofadr[0]
            mu = sim.m.geom_friction[sim.m.geom(obj).id][0]
            speed = np.sqrt(2 * mu * 9.81 * distance)  # slides `distance` before friction stops it
            sim.d.qvel[adr:adr + 3] = [0.0, -speed, 0.0]
            sim.step(0.8)
            print(f"  >>> BUMP: someone knocks the {obj} out of place")
            state["done"] = True

    return hook


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", nargs="?", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--audio", type=Path, help="spoken command (wav/mp3) instead of text")
    p.add_argument("--rules", action="store_true", help="keyword reader instead of the AI model")
    p.add_argument("--device", default="GPU", help="OpenVINO device for the AI model (GPU/CPU/NPU)")
    p.add_argument("--video", action="store_true")
    p.add_argument("--hard", action="store_true", help="wider randomisation of the table")
    p.add_argument("--policy", action="store_true",
                   help="let the trained ACT policy (OpenVINO) move the mug where it was trained to")
    p.add_argument("--policy-device", default="CPU", help="OpenVINO device for the policy (CPU/GPU/NPU)")
    p.add_argument("--policy-precision", default="fp16", choices=["fp32", "fp16", "int8", "int8w"])
    p.add_argument("--bump", metavar="OBJECT",
                   help="recovery demo: knock this object ~6 cm sideways once it has been placed "
                        "and the next item is done, to show the robot noticing and fixing it")
    args = p.parse_args()

    command = args.command
    if args.audio:
        from voice import transcribe
        t = time.time()
        command = transcribe(args.audio)
        print(f'Heard ({time.time() - t:.1f}s, Speechmatics): "{command}"')
    if not command:
        p.error("give a command (text) or --audio")

    sim = day2_demo.start_sim(args.seed, video=args.video, eyes="camera", hard=args.hard)
    if args.policy:
        import policy_skill
        sim.policy = policy_skill.load(device=args.policy_device, precision=args.policy_precision)
        print(f"Trained ACT policy loaded (OpenVINO {args.policy_precision.upper()}, {args.policy_device})")
    try:
        skills.observe(sim)
        print(f"Camera sees: {', '.join(sorted(sim.world))}")
        if args.rules:
            goals, info = read_rules(command), {"reader": "keyword reader"}
        else:
            brain = VisionLanguage(device=args.device)
            print(f"AI model loaded on {brain.device} in {brain.load_seconds:.0f}s")
            rgb, *_ = sim.eyes.capture(sim.d)
            goals, info = brain.read(command, (rgb * 255).astype("uint8"))
        print(f"Understood ({info['reader']}"
              + (f", {info['seconds']}s" if "seconds" in info else "") + "):")
        for g in goals:
            print(f"  {g.obj} -> {g.place}")
        if not goals:
            print("  nothing I can do with that command.")
            return

        report = planner.execute(sim, goals, after_goal=bumper(args.bump) if args.bump else None)
        goals = report.goals
        sim.step(0.5)
        truth = {g.obj: skills.why_not(sim, g.obj, g.xy(planner.plate_xy(sim, goals)))
                 for g in goals}
        print("\nResult:")
        for g in goals:
            cam = "yes" if report.done.get(g.obj) else "no"
            real = "yes" if truth[g.obj] == "" else f"no ({truth[g.obj]})"
            print(f"  {g.obj:6s} camera says done: {cam:3s} | actually done: {real}"
                  + (f" | tries: {report.attempts[g.obj]}" if report.attempts.get(g.obj, 1) > 1 else ""))
    finally:
        day2_demo.close_sim(sim)
    if args.video:
        OUT.mkdir(exist_ok=True)
        iio.imwrite(OUT / f"run_seed{args.seed}.gif", sim.frames["front"], duration=1000 / 24, loop=0)
        print(f"saved {OUT / f'run_seed{args.seed}.gif'}")


if __name__ == "__main__":
    main()
