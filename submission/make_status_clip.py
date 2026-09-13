"""Vertical 30-second clip for a WhatsApp/Instagram status: hook -> table being set ->
knocked plate fixed -> call for Core Ultra volunteers.

    python submission/make_status_clip.py   # -> submission/status_clip.mp4 (1080x1920, <=30 s)
"""
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "brain" / "out"
OUT = Path(__file__).parent / "status_clip.mp4"
W, H, FPS = 1080, 1920, 24
BG = (22, 27, 34)
YELLOW, WHITE, MUTE = (232, 185, 35), (245, 245, 245), (170, 180, 190)
BOLD, REG = "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf"


def f(size, bold=True):
    return ImageFont.truetype(BOLD if bold else REG, size)


def text_block(d, y, lines):
    """lines: [(text, size, colour, bold)] centred, stacked from y."""
    for text, size, colour, bold in lines:
        d.text((W // 2, y), text, font=f(size, bold), fill=colour, anchor="mt")
        y += int(size * 1.3)


def still(top, bottom=(), seconds=2.5):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    text_block(d, 700, top)
    text_block(d, 1150, bottom)
    return [np.asarray(img)] * int(seconds * FPS)


def footage(path, start, end, speed, top, bottom):
    meta = iio.immeta(path, plugin="pyav")
    fps = meta["fps"]
    frames = []
    for i, fr in enumerate(iio.imiter(path, plugin="pyav")):
        t = i / fps
        if t < start:
            continue
        if t > end:
            break
        if (i - int(start * fps)) % speed:  # keep every `speed`-th frame -> plays `speed`x faster
            continue
        img = Image.new("RGB", (W, H), BG)
        src = Image.fromarray(fr)
        cut = int(src.width * 0.1)  # trim 10% each side so the arms look bigger on a phone
        src = src.crop((cut, 0, src.width - cut, src.height))
        vh = int(W * src.height / src.width)
        img.paste(src.resize((W, vh)), (0, 960 - vh // 2))
        d = ImageDraw.Draw(img)
        text_block(d, 330, top)
        text_block(d, 1400, bottom)
        frames.append(np.asarray(img))
    return frames


def main():
    clip = []
    clip += still([("I taught two robot arms", 72, WHITE, True), ("to set a dinner table", 72, WHITE, True)],
                  [("they see, think, and fix their own mistakes", 40, MUTE, False)], seconds=2.5)
    clip += footage(SRC / "demo_10_tables.mp4", 5.0, 28.5, 3,
                    [("\"Set the table\"", 70, YELLOW, True), ("AI reads the command", 44, WHITE, False)],
                    [("they work out who does what", 44, WHITE, False), ("and pass the fork across", 44, WHITE, False)])
    clip += footage(SRC / "recovery.mp4", 9.0, 33.0, 3,
                    [("Now knock the plate...", 70, YELLOW, True)],
                    [("it notices and puts it back", 44, WHITE, False), ("before carrying on", 44, WHITE, False)])
    clip += still([("Got an Intel Core Ultra laptop?", 64, YELLOW, True),
                   ("Help me test it this week", 60, WHITE, True)],
                  [("about an hour, and I'll guide you", 42, MUTE, False),
                   ("message me", 50, WHITE, True)], seconds=4.5)
    clip = clip[: 30 * FPS]
    with iio.imopen(OUT, "w", plugin="pyav") as w:
        w.init_video_stream("libx264", fps=FPS)
        for fr in clip:
            w.write_frame(fr)
    print(f"saved {OUT} ({len(clip) / FPS:.1f} s, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
