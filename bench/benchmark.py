"""Intel inference benchmark: every AI part of the robot, on every device this PC has.

    python bench/benchmark.py              # all parts (~15 min on the team laptop)
    python bench/benchmark.py --only act   # just the trained policy

Writes bench/results.json and bench/RESULTS.md (a table for the README).
Parts:
  vlm         Qwen3-VL-4B INT4 (OpenVINO GenAI): load time, time to first token, tokens/s,
              seconds per command, and whether it still reads a 5-command test set correctly —
              per device and per picture size.
  act         ACT policy: PyTorch vs OpenVINO FP32 / FP16 / INT8, per device, ms per call.
  perception  camera look (render + depth + find objects), ms per look.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "brain"), str(ROOT / "sim"), str(ROOT / "train"), str(ROOT / "bench")]
# Results go next to this script, unless BENCH_OUT says otherwise (volunteer runs keep their
# own file so they never mix with the team laptop's numbers).
OUT = Path(os.environ.get("BENCH_OUT", Path(__file__).resolve().parent))

SUITE = [  # command -> expected {object: place}
    ("set the table", {"plate": "center", "fork": "left_of_plate", "spoon": "right_of_plate",
                       "mug": "top_right_of_plate"}),
    ("put the red thing to the top left of the plate and the grey utensil on the right",
     {"mug": "top_left_of_plate", "fork": "right_of_plate"}),
    ("move the spoon to the left of the plate", {"spoon": "left_of_plate"}),
    ("the cup goes top right and the fork on the left", {"mug": "top_right_of_plate", "fork": "left_of_plate"}),
    ("put the white dish in the middle", {"plate": "center"}),
]


def hardware():
    core = ov.Core()
    return {"cpu": core.get_property("CPU", "FULL_DEVICE_NAME"),
            "devices": {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices},
            "openvino": ov.__version__, "os": platform.platform()}


def scene_image():
    import day2_demo
    import skills

    sim = day2_demo.start_sim(7, video=False, eyes="camera")
    skills.observe(sim)
    rgb, *_ = sim.eyes.capture(sim.d)
    day2_demo.close_sim(sim)
    return (rgb * 255).astype("uint8")


def bench_vlm(devices, widths):
    from language import VisionLanguage

    img = scene_image()
    rows = []
    for dev in devices:
        brain = VisionLanguage(device=dev)
        if brain.device != dev:
            print(f"  {dev}: not usable for the VLM, skipped")
            continue
        brain.read("set the table", img, width=widths[-1])  # warm-up
        for w in widths:
            secs, ttft, tps, correct = [], [], [], 0
            for cmd, want in SUITE:
                goals, info = brain.read(cmd, img, width=w)
                got = {g.obj: g.place for g in goals}
                correct += got == want and "fell back" not in info["reader"]
                secs.append(info["seconds"])
                ttft.append(info.get("first_token_ms", np.nan))
                tps.append(info.get("tokens_per_s", np.nan))
            row = {"device": dev, "image_width": w, "load_s": round(brain.load_seconds, 1),
                   "s_per_command": round(float(np.mean(secs)), 1),
                   "first_token_ms": round(float(np.nanmean(ttft))),
                   "tokens_per_s": round(float(np.nanmean(tps)), 1),
                   "correct": f"{correct}/{len(SUITE)}"}
            print("  ", row)
            rows.append(row)
        del brain
    return rows


def bench_act(devices, calls=500):
    import torch
    from export_act import ACTCore, RUN, ckpt_of
    from eval_policy import io_dims, load_torch

    policy, *_ = load_torch(ckpt_of(RUN))
    core = ACTCore(policy.model).eval()
    s = np.random.randn(1, 6).astype(np.float32)
    e = np.random.randn(1, io_dims(policy)[0]).astype(np.float32)
    rows = []
    with torch.no_grad():
        ts, te = torch.from_numpy(s), torch.from_numpy(e)
        for _ in range(20):
            core(ts, te)
        t = time.perf_counter()
        for _ in range(calls):
            core(ts, te)
        rows.append({"backend": "PyTorch", "precision": "FP32", "device": "CPU",
                     "ms_per_call": round(1000 * (time.perf_counter() - t) / calls, 3)})
    ovc = ov.Core()
    for prec in ("fp32", "fp16", "int8", "int8w"):
        xml = RUN / "openvino" / f"act_{prec}.xml"
        for dev in devices:
            try:
                net = ovc.compile_model(xml, dev, {"PERFORMANCE_HINT": "LATENCY"})
            except RuntimeError as ex:
                print(f"  {prec} on {dev}: {ex}")
                continue
            for _ in range(20):
                net([s, e])
            t = time.perf_counter()
            for _ in range(calls):
                net([s, e])
            rows.append({"backend": "OpenVINO", "precision": prec.upper(), "device": dev,
                         "ms_per_call": round(1000 * (time.perf_counter() - t) / calls, 3)})
    add_throughput(rows)
    for r in rows:
        print("  ", r)
    return rows


def add_throughput(rows):
    """Calls per second, action steps per second (each call yields a 1-second, 20-step chunk,
    run open-loop), and speed-up over PyTorch."""
    base = rows[0]["ms_per_call"]
    for r in rows:
        r["calls_per_s"] = round(1000 / r["ms_per_call"], 1)
        r["actions_per_s"] = round(20 * 1000 / r["ms_per_call"])
        r["speedup_vs_pytorch"] = round(base / r["ms_per_call"], 2)


def bench_act_task(episodes=10):
    """Does optimisation keep the robot working? Same unseen tables for every precision."""
    import eval_policy
    from export_act import OpenVINORunner, RUN, ckpt_of

    runners = {"PyTorch FP32": eval_policy.TorchRunner(ckpt_of(RUN))}
    for prec in ("fp32", "fp16", "int8", "int8w"):
        runners[f"OpenVINO {prec.upper()} (CPU)"] = OpenVINORunner(ckpt_of(RUN), "CPU", prec)
    rows = []
    for name, runner in runners.items():
        results = [eval_policy.run_episode(runner, eval_policy.EVAL_SEED0 + i) for i in range(episodes)]
        row = {"policy": name,
               "within_1_5cm": f"{sum(r[0] for r in results)}/{episodes}",
               "within_2_5cm": f"{sum(r[4] for r in results)}/{episodes}"}
        print("  ", row)
        rows.append(row)
    return rows


def bench_perception(looks=30):
    import day2_demo

    sim = day2_demo.start_sim(7, video=False, eyes="camera")
    sim.eyes.look(sim.d)
    t = time.perf_counter()
    for _ in range(looks):
        sim.eyes.look(sim.d)
    ms = 1000 * (time.perf_counter() - t) / looks
    day2_demo.close_sim(sim)
    row = {"ms_per_look": round(ms, 1), "resolution": "960x720 colour + depth"}
    print("  ", row)
    return [row]


def to_markdown(res) -> str:
    lines = [f"# Benchmark results\n", f"Hardware: {res['hardware']['cpu']}; devices: "
             + ", ".join(f"{k} = {v}" for k, v in res["hardware"]["devices"].items())
             + f"; OpenVINO {res['hardware']['openvino']}.\n"]
    for part, rows in res["parts"].items():
        if not rows:
            continue
        cols = list(rows[0])
        lines += [f"\n## {part}\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["vlm", "act", "act_task", "perception"])
    p.add_argument("--widths", default="320,480,960")
    p.add_argument("--devices", help="comma list, default: every CPU/GPU/NPU found")
    p.add_argument("--report-only", action="store_true", help="rebuild RESULTS.md from results.json")
    args = p.parse_args()
    if args.report_only:
        res = json.loads((OUT / "results.json").read_text())
        add_throughput(res["parts"]["act"])
        (OUT / "results.json").write_text(json.dumps(res, indent=2))
        (OUT / "RESULTS.md").write_text(to_markdown(res))
        print("rebuilt RESULTS.md")
        return
    devices = [d for d in ov.Core().available_devices if d in ("CPU", "GPU", "NPU")]
    if args.devices:
        devices = [d for d in args.devices.split(",") if d in devices]
    path = OUT / "results.json"
    res = json.loads(path.read_text()) if path.exists() else {"parts": {}}
    res["hardware"] = hardware()
    parts = [args.only] if args.only else ["perception", "act", "act_task", "vlm"]
    for part in parts:
        print(f"{part}:")
        if part == "vlm":
            rows = bench_vlm(devices, [int(w) for w in args.widths.split(",")])
            keep = [r for r in res["parts"].get("vlm", []) if r["device"] not in devices]
            res["parts"]["vlm"] = keep + rows
        elif part == "act":
            res["parts"]["act"] = bench_act(devices)
        elif part == "act_task":
            res["parts"]["act_task"] = bench_act_task()
        else:
            res["parts"]["perception"] = bench_perception()
    path.write_text(json.dumps(res, indent=2))
    (OUT / "RESULTS.md").write_text(to_markdown(res))
    print(f"wrote {path} and {OUT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
