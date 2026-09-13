"""Clean renders for the slides (no captions): hand-off close-up, the robot's camera view with
what it detected, and the recovery (knocked plate, then fixed).

    python submission/make_slide_images.py   # -> submission/img/*.png
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "brain"), str(ROOT / "sim"), str(ROOT / "train")]
import day2_demo  # noqa: E402
import planner  # noqa: E402
import robot  # noqa: E402
import skills  # noqa: E402
from language import read_rules  # noqa: E402
from run import bumper  # noqa: E402

OUT = Path(__file__).parent / "img"
FONT = "C:/Windows/Fonts/segoeuib.ttf"


def shot(sim, lookat, distance, azimuth, elevation, size=(900, 1600)):
    sim.m.vis.global_.offwidth, sim.m.vis.global_.offheight = size[1], size[0]
    r = mujoco.Renderer(sim.m, *size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    r.update_scene(sim.d, camera=cam)
    img = Image.fromarray(r.render())
    robot.close_renderer(r)
    return img


def handoff():
    sim = day2_demo.start_sim(3, video=False, eyes="camera")
    S = skills
    S.observe(sim)
    end = 0.6 * 0.055
    S.pick(sim, "B", "fork", along=S._near_end(sim, "B", "fork", end))
    side = np.sign(sim.arms["B"].base_xy[0])
    opts = [S._site_target_for(sim, "B", "fork", S.HANDOFF_POINT[:2], oy, S.HANDOFF_POINT[2]) for oy in (0.0, np.pi)]
    g, y = max(opts, key=lambda o: side * o[0][0])
    sim.move({"B": (g, y)}, 1.5)
    S.pick(sim, "A", "fork", along=S._near_end(sim, "A", "fork", end), lift_to=None, margin=S.HANDOFF_MARGIN)
    img = shot(sim, [0, -0.11, 0.07], 0.38, 90, -12)  # from the front, both hands on the fork
    day2_demo.close_sim(sim)
    img.save(OUT / "handoff.png")


def eyes():
    sim = day2_demo.start_sim(7, video=False, eyes="camera")
    rgb, height, wx, wy, robot_mask = sim.eyes.capture(sim.d)
    seen = sim.eyes.look(sim.d)
    img = Image.fromarray((rgb * 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 26)
    cam_z, fpx = sim.eyes.cam_pos[2], sim.eyes.f
    for name, s in seen.items():
        u = s.x * fpx / cam_z + img.width / 2
        v = -s.y * fpx / cam_z + img.height / 2
        d.ellipse([u - 9, v - 9, u + 9, v + 9], outline=(80, 255, 120), width=4)
        d.text((u + 14, v - 16), name, font=f, fill=(80, 255, 120))
    day2_demo.close_sim(sim)
    img.save(OUT / "eyes.png")


def recovery():
    sim = day2_demo.start_sim(3, video=False, eyes="camera")
    shots = {}
    hook = bumper("plate")

    def after_goal(s, goal):
        hook(s, goal)
        if "bumped" not in shots and s.d.body("plate").xpos[1] < -0.04:
            shots["bumped"] = shot(s, [0, -0.02, 0.03], 0.55, 90, -45, size=(720, 960))

    planner.execute(sim, read_rules("set the table"), say=lambda s: None, after_goal=after_goal)
    shots["fixed"] = shot(sim, [0, -0.02, 0.03], 0.55, 90, -45, size=(720, 960))
    day2_demo.close_sim(sim)
    for k, im in shots.items():
        im.save(OUT / f"recovery_{k}.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    only = sys.argv[1:] or ["handoff", "eyes", "recovery"]
    for name in only:
        globals()[name]()
    print("saved", sorted(p.name for p in OUT.glob("*.png")))
