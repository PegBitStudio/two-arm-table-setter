"""Make the demo video: one command, several random tables, every result checked.

    python brain/make_video.py --seeds 10 --hard --model          # the 10-seed video Intel asks for
    python brain/make_video.py --seeds 1 --start 3 --bump plate   # recovery clip

Each table: front camera with the robot's own overhead view in the corner, captions of what
the robot is doing, then a result card (per item, checked against simulator truth). The film
ends with a scoreboard. Played at 2x speed. Writes brain/out/demo.mp4.
"""
import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import __init__  # noqa: F401
import day2_demo
import planner
import skills
from language import VisionLanguage, read_rules
from run import bumper

OUT = Path(__file__).parent / "out"
W, H = 960, 540          # output video size
FPS_SIM, FPS_OUT = 12, 24
FONT = "C:/Windows/Fonts/segoeui.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def card(lines, sub=None, seconds=2.5, colour=(22, 27, 34)):
    img = Image.new("RGB", (W, H), colour)
    d = ImageDraw.Draw(img)
    y = H // 2 - 40 * len(lines) // 2 - (30 if sub else 0)
    for i, line in enumerate(lines):
        f = font(40 if i == 0 else 30)
        d.text((W // 2, y), line, font=f, fill=(240, 240, 240), anchor="mm")
        y += 56 if i == 0 else 44
    if sub:
        d.text((W // 2, y + 20), sub, font=font(22), fill=(170, 180, 190), anchor="mm")
    return [np.asarray(img)] * int(seconds * FPS_OUT)


def compose(front, top, caption, header):
    img = Image.fromarray(front).resize((W, H))
    inset = Image.fromarray(top).resize((W // 4, H // 4))
    img.paste(inset, (W - W // 4 - 12, 12))
    d = ImageDraw.Draw(img)
    d.rectangle([W - W // 4 - 12, 12, W - 12, 12 + H // 4], outline=(255, 255, 255), width=2)
    d.text((W - W // 8 - 12, 16 + H // 4), "robot's camera", font=font(16), fill=(255, 255, 255), anchor="mt")
    d.rectangle([0, 0, W - W // 4 - 24, 40], fill=(0, 0, 0))
    d.text((12, 20), header, font=font(20), fill=(255, 255, 255), anchor="lm")
    if caption:
        d.rectangle([0, H - 44, W, H], fill=(0, 0, 0))
        d.text((12, H - 22), caption, font=font(20), fill=(255, 230, 120), anchor="lm")
    return np.asarray(img)


def film_table(seed, command, brain, hard, bump, policy, policy_device="CPU"):
    sim = day2_demo.start_sim(seed, video=False, eyes="camera", hard=hard)
    import mujoco
    import robot

    sim._renderer = mujoco.Renderer(sim.m, 360, 640)
    sim.cams, sim.fps = ("front", "top"), FPS_SIM
    sim.frames = {"front": [], "top": []}
    sim._next_frame = sim.d.time
    if policy:
        import policy_skill
        sim.policy = policy_skill.load(device=policy_device)
    captions = []  # (frame index, text)

    def say(text):
        captions.append((len(sim.frames["front"]), text.strip()))

    try:
        skills.observe(sim)
        if brain:
            rgb, *_ = sim.eyes.capture(sim.d)
            goals, info = brain.read(command, (rgb * 255).astype("uint8"))
            understood = f"Qwen3-VL on OpenVINO, {info['seconds']} s"
        else:
            goals, understood = read_rules(command), "keyword reader"
        report = planner.execute(sim, goals, say=say, after_goal=bumper(bump) if bump else None)
        sim.step(0.5)
        plate = planner.plate_xy(sim, report.goals)
        why = {g.obj: skills.why_not(sim, g.obj, g.xy(plate)) for g in report.goals}
    finally:
        robot.close_renderer(sim._renderer)
        sim._renderer = None
        day2_demo.close_sim(sim)
    goals_txt = ", ".join(f"{g.obj} -> {g.place.replace('_', ' ')}" for g in report.goals)
    return sim.frames, captions, why, goals_txt, understood


def write_table(writer, seed, hard, command, frames, captions):
    header = f"Table {seed}{' (hard)' if hard else ''}  |  \"{command}\""
    ci, caption = 0, ""
    for k, (f, t) in enumerate(zip(frames["front"], frames["top"])):
        while ci < len(captions) and captions[ci][0] <= k:
            caption = captions[ci][1]
            ci += 1
        writer.write_frame(compose(f, t, caption, header))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--start", type=int, default=100)
    p.add_argument("--command", default="set the table")
    p.add_argument("--hard", action="store_true")
    p.add_argument("--model", action="store_true", help="read the command with Qwen3-VL")
    p.add_argument("--bump", metavar="OBJECT")
    p.add_argument("--policy", action="store_true")
    p.add_argument("--device", default="GPU", help="OpenVINO device for Qwen3-VL (CPU/GPU/NPU)")
    p.add_argument("--policy-device", default="CPU", help="OpenVINO device for the ACT policy")
    p.add_argument("--out", type=Path, default=OUT / "demo.mp4")
    args = p.parse_args()

    from hardware import label
    hw = label()
    brain = VisionLanguage(args.device) if args.model else None
    args.out.parent.mkdir(exist_ok=True)
    score = []
    with iio.imopen(args.out, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=FPS_OUT)

        def put(frames):
            for f in frames:
                writer.write_frame(f)

        put(card(["Two-Arm Table Setter", f"\"{args.command}\"", f"running on {hw}"],
                 sub=f"{args.seeds} random table{'s' if args.seeds > 1 else ''}"
                     f"{' (hard: positions, sizes, colours)' if args.hard else ''}"
                     " - played at 2x speed", seconds=3))
        for seed in range(args.start, args.start + args.seeds):
            t = time.time()
            frames, captions, why, goals_txt, understood = film_table(
                seed, args.command, brain, args.hard, args.bump, args.policy, args.policy_device)
            ok = all(v == "" for v in why.values())
            score.append(ok)
            put(card([f"Table {seed}", f"Understood: {goals_txt}"], sub=understood, seconds=2))
            write_table(writer, seed, args.hard, args.command, frames, captions)
            del frames
            lines = [f"Table {seed}: {'ALL CORRECT' if ok else 'NOT ALL CORRECT'}"]
            lines += [f"{o}: {'OK' if v == '' else v}" for o, v in why.items()]
            put(card(lines, sub="checked against simulator truth, not the robot's camera", seconds=2.5,
                     colour=(20, 60, 30) if ok else (80, 30, 20)))
            print(f"table {seed}: {'PASS' if ok else 'FAIL'} {why} ({time.time() - t:.0f}s)", flush=True)
        put(card([f"{sum(score)} / {len(score)} tables set correctly", f"{hw}, OpenVINO"], seconds=4))
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
