"""Cover image for the lablab submission: a finished table, rendered sharp, with the title.

    python submission/make_cover.py    # -> submission/cover.png (1600x900)
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "brain"), str(ROOT / "sim"), str(ROOT / "train")]
import day2_demo  # noqa: E402
import planner  # noqa: E402
import robot  # noqa: E402
from language import read_rules  # noqa: E402

FONT = "C:/Windows/Fonts/segoeuib.ttf"
FONT_REG = "C:/Windows/Fonts/segoeui.ttf"


def main():
    sim = day2_demo.start_sim(3, video=False, eyes="camera")
    planner.execute(sim, read_rules("set the table"), say=lambda s: None)
    sim.m.vis.global_.offwidth, sim.m.vis.global_.offheight = 1600, 900
    r = mujoco.Renderer(sim.m, 900, 1600)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0.03, 0.04]
    cam.distance, cam.azimuth, cam.elevation = 0.72, 90, -38  # from the front (-y side)
    r.update_scene(sim.d, camera=cam)
    img = Image.fromarray(r.render())
    robot.close_renderer(r)
    day2_demo.close_sim(sim)

    # Dark band at the bottom for the title.
    band = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(band)
    for y in range(560, 900):
        d.line([(0, y), (1600, y)], fill=(10, 14, 20, int(235 * (y - 560) / 340)))
    img = Image.alpha_composite(img.convert("RGBA"), band)
    d = ImageDraw.Draw(img)
    d.text((70, 690), "Two-Arm Table Setter", font=ImageFont.truetype(FONT, 78), fill=(255, 255, 255))
    d.text((74, 790), "See · understand · hand over · fix mistakes  —  on an Intel laptop with OpenVINO",
           font=ImageFont.truetype(FONT_REG, 34), fill=(200, 215, 230))
    out = Path(__file__).parent / "cover.png"
    img.convert("RGB").save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
