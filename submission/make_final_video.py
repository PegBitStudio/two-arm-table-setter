"""Join the finished pieces into the submission video, following Intel's recommended demo
sequence: 10 random tables (with command, scene variation and result cards) -> recovery ->
OpenVINO benchmark results -> links.

    python submission/make_final_video.py    # -> submission/two_arm_table_setter_demo.mp4

Inputs: brain/out/demo_10_tables.mp4, brain/out/recovery.mp4 (from brain/make_video.py) and
bench/results.json (from bench/benchmark.py).
"""
import json
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "brain"), str(ROOT / "sim"), str(ROOT / "train")]
from make_video import FPS_OUT, H, W, card, font  # noqa: E402

OUT = Path(__file__).resolve().parent
TMP = ROOT / "brain" / "out"


def table_card(title, header, rows, note, seconds=6):
    img = Image.new("RGB", (W, H), (22, 27, 34))
    d = ImageDraw.Draw(img)
    d.text((W // 2, 50), title, font=font(34), fill=(255, 255, 255), anchor="mm")
    # First column holds long labels: give it 32% of the width, share the rest.
    first = int((W - 140) * 0.32)
    rest = (W - 140 - first) // (len(header) - 1)
    xs = [70] + [70 + first + i * rest for i in range(len(header) - 1)]
    y = 110
    for x, h in zip(xs, header):
        d.text((x, y), h, font=font(20), fill=(232, 185, 35))
    for r, row in enumerate(rows):
        yy = y + 46 + r * 44
        for x, v in zip(xs, row):
            d.text((x, yy), str(v), font=font(22), fill=(240, 240, 240))
    d.text((W // 2, H - 50), note, font=font(20), fill=(170, 180, 190), anchor="mm")
    return [np.asarray(img)] * int(seconds * FPS_OUT)


def write(frames, path):
    with iio.imopen(path, "w", plugin="pyav") as w:
        w.init_video_stream("libx264", fps=FPS_OUT)
        for f in frames:
            w.write_frame(f)


def main():
    res = json.loads((ROOT / "bench" / "results.json").read_text())
    act = {(r["backend"], r["precision"], r["device"]): r for r in res["parts"]["act"]}
    task = {r["policy"]: r for r in res["parts"]["act_task"]}
    vlm = {(r["device"], r["image_width"]): r for r in res["parts"]["vlm"]}

    rows = []
    for label, key, tkey in [("PyTorch FP32", ("PyTorch", "FP32", "CPU"), "PyTorch FP32"),
                             ("OpenVINO FP32", ("OpenVINO", "FP32", "CPU"), "OpenVINO FP32 (CPU)"),
                             ("OpenVINO FP16", ("OpenVINO", "FP16", "CPU"), "OpenVINO FP16 (CPU)"),
                             ("OpenVINO INT8", ("OpenVINO", "INT8", "CPU"), "OpenVINO INT8 (CPU)"),
                             ("OpenVINO INT8 weights", ("OpenVINO", "INT8W", "CPU"), "OpenVINO INT8W (CPU)")]:
        a = act[key]
        rows.append([label, f"{a['ms_per_call']:.2f} ms", f"{a['calls_per_s']:.0f}/s",
                     f"{a['speedup_vs_pytorch']:.1f}x", task[tkey]["within_1_5cm"]])
    act_card = table_card("Trained ACT policy on OpenVINO (CPU)",
                          ["backend", "latency", "throughput", "speed-up", "mug placed"], rows,
                          "Intel Core i7-1165G7 - task success unchanged by optimisation (10 unseen tables)")
    vrows = [[f"{w} px", f"{vlm[('CPU', w)]['s_per_command']} s", f"{vlm[('GPU', w)]['s_per_command']} s",
              f"{vlm[('GPU', w)]['first_token_ms'] / 1000:.1f} s", vlm[("GPU", w)]["correct"]]
             for w in (320, 480, 960)]
    vlm_card = table_card("Qwen3-VL-4B INT4 on OpenVINO GenAI",
                          ["picture width", "CPU", "iGPU", "iGPU 1st word", "correct"], vrows,
                          "seconds per command - the Iris Xe iGPU halves it for large pictures; tuning took it from 30 s to 4 s")

    pieces = [
        TMP / "demo_10_tables.mp4",
        ("rec_intro", card(["Recovery", "Mid-task, someone knocks the plate out of place"],
                           sub="the robot looks again after every item", seconds=3)),
        ("recovery_trim", None),
        ("bench", card(["Intel optimisation", "measured on this laptop"], sub="bench/benchmark.py", seconds=2)
         + act_card + vlm_card),
        ("end", card(["Two-Arm Table Setter", "github.com/PegBitStudio/two-arm-table-setter"],
                     sub="MuJoCo - LeRobot ACT - OpenVINO - Qwen3-VL - SO-101", seconds=4)),
    ]
    files = []
    for p in pieces:
        if isinstance(p, Path):
            files.append(p)
            continue
        name, frames = p
        path = TMP / f"_{name}.mp4"
        if name == "recovery_trim":  # drop the clip's own title card (3 s) and scoreboard (4 s)
            src = TMP / "recovery.mp4"
            keep = duration(src) - 3 - 4
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "3", "-i", str(src), "-t", f"{keep:.2f}",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS_OUT), str(path)], check=True)
        else:
            write(frames, path)
        files.append(path)
    listing = TMP / "_concat.txt"
    listing.write_text("".join(f"file '{f.as_posix()}'\n" for f in files))
    out = OUT / "two_arm_table_setter_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS_OUT), str(out)], check=True)
    print(f"saved {out} ({duration(out):.0f} s, {out.stat().st_size / 1e6:.1f} MB)")


def duration(path: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                 "csv=p=0", str(path)], capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
