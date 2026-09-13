"""Ears: turn a spoken command (audio file) into text with Speechmatics.

Needs a free Speechmatics API key in the SPEECHMATICS_API_KEY environment variable.
Uses the batch API (upload a file, wait, fetch the transcript) — plain HTTPS, no SDK.

    python brain/voice.py command.wav
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://asr.api.speechmatics.com/v2"


class VoiceError(RuntimeError):
    pass


def transcribe(audio: Path, language: str = "en", timeout: float = 120) -> str:
    key = os.environ.get("SPEECHMATICS_API_KEY")
    if not key:
        raise VoiceError("set SPEECHMATICS_API_KEY to use voice commands")
    headers = {"Authorization": f"Bearer {key}"}
    config = {"type": "transcription",
              "transcription_config": {"language": language, "operating_point": "enhanced"}}
    with open(audio, "rb") as f:
        r = requests.post(f"{API}/jobs", headers=headers,
                          files={"data_file": f}, data={"config": json.dumps(config)}, timeout=60)
    if r.status_code >= 300:
        raise VoiceError(f"upload failed: {r.status_code} {r.text[:200]}")
    job = r.json()["id"]
    start = time.time()
    while time.time() - start < timeout:
        status = requests.get(f"{API}/jobs/{job}", headers=headers, timeout=30).json()["job"]["status"]
        if status == "done":
            t = requests.get(f"{API}/jobs/{job}/transcript", headers=headers,
                             params={"format": "txt"}, timeout=30)
            return t.text.strip()
        if status in ("rejected", "deleted", "expired"):
            raise VoiceError(f"transcription {status}")
        time.sleep(1.0)
    raise VoiceError("transcription timed out")


if __name__ == "__main__":
    print(transcribe(Path(sys.argv[1])))
