"""Everything a Core Ultra volunteer needs to run, in one go. Takes ~45-60 minutes.

    python bench/volunteer.py            # full run
    python bench/volunteer.py --quick    # ~10 min smoke test (fewer tables, no VLM)

Steps: show the Intel devices OpenVINO sees (CPU / GPU / NPU) -> download the AI model (3.1 GB)
and the trained policy (45 MB) if missing -> run the benchmark on every device -> set 10 random
tables with the full robot -> put everything in volunteer_results.zip to send back.
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import openvino as ov

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OUT = ROOT / "volunteer_results"


def step(title):
    print(f"\n=== {title} ===", flush=True)


def run(args, log_name):
    """Run a project script, show its output live, and keep a copy in the results folder."""
    t = time.time()
    env = dict(os.environ, BENCH_OUT=str(OUT), PYTHONIOENCODING="utf-8")
    with open(OUT / log_name, "w", encoding="utf-8") as log:
        p = subprocess.Popen([PY, *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", env=env)
        for line in p.stdout:
            log.write(line)
            if "torchcodec" not in line and "Map:" not in line:
                print("   " + line.rstrip(), flush=True)
        p.wait()
    print(f"   ({time.time() - t:.0f} s, exit {p.returncode})", flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short smoke test")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    step("1/5  Your Intel hardware")
    core = ov.Core()
    devices = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}
    for d, name in devices.items():
        print(f"   {d:4s} {name}")
    if "NPU" not in devices:
        print("   (no NPU found - fine, CPU and GPU still count; on a Core Ultra, updating the Intel NPU driver may add it)")
    info = {"when": datetime.datetime.now().isoformat(timespec="seconds"), "os": platform.platform(),
            "python": platform.python_version(), "openvino": ov.__version__, "devices": devices,
            "quick": args.quick}
    (OUT / "machine.json").write_text(json.dumps(info, indent=2))

    step("2/5  Downloading models (skipped if already there)")
    run(["bench/get_policy.py"], "get_policy.log")
    if not args.quick:
        run(["brain/get_model.py"], "get_model.log")

    step("3/5  OpenVINO benchmark (trained policy + camera)")
    run(["bench/benchmark.py", "--only", "perception"], "bench_perception.log")
    run(["bench/benchmark.py", "--only", "act"], "bench_act.log")
    if not args.quick:
        run(["bench/benchmark.py", "--only", "act_task"], "bench_act_task.log")
        step("4/5  OpenVINO benchmark (vision-language model, every device)")
        run(["bench/benchmark.py", "--only", "vlm"], "bench_vlm.log")

    step("5/5  The full robot on random tables")
    tables = ["--seeds", "3"] if args.quick else ["--seeds", "10", "--hard", "--model"]
    run(["brain/evaluate.py", *tables], "evaluate.log")

    zpath = ROOT / "volunteer_results.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in OUT.iterdir():
            z.write(f, f.name)
    print(f"\nDone. Please send this file back:\n   {zpath}")


if __name__ == "__main__":
    main()
