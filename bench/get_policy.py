"""Download the trained ACT policy (45 MB) from the GitHub release into ~/.models/runs/act_mug.

    python bench/get_policy.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/PegBitStudio/two-arm-table-setter/releases/download/policy-v7/act_mug_v7.zip"
RUNS = Path.home() / ".models" / "runs"


def main():
    target = RUNS / "act_mug"
    if (target / "openvino" / "act_fp16.xml").exists():
        print(f"policy already in {target}")
        return
    print(f"downloading {URL} ...")
    data = urllib.request.urlopen(URL, timeout=300).read()
    RUNS.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(RUNS)
    print(f"policy ready in {target}")


if __name__ == "__main__":
    main()
