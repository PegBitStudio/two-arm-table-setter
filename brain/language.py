"""Understanding the command: text (+ the camera picture) -> a list of goals.

Two readers with the same output:
  - VisionLanguage: Qwen3-VL-4B running locally on OpenVINO. Sees the overhead picture, so it
    can resolve "the red thing" or "whatever is lying next to the bottle".
  - read_rules: a keyword reader. Used when the model isn't available, and as a safety net
    when the model's answer doesn't parse.
Either way the result is checked against the objects and places the robot actually knows.
"""
import re
import time
from pathlib import Path

import numpy as np

from planner import DEFAULT_SETTING, PLACES, Goal

OBJECT_WORDS = {
    "plate": ["plate", "dish"],
    "mug": ["mug", "cup", "red"],
    "fork": ["fork"],
    "spoon": ["spoon"],
}
PLACE_WORDS = {
    "center": ["middle", "center", "centre"],
    "left_of_plate": ["left"],
    "right_of_plate": ["right"],
    "top_right_of_plate": ["top right", "upper right", "behind right", "top-right"],
    "top_left_of_plate": ["top left", "upper left", "behind left", "top-left"],
}
MODEL_DIR = Path.home() / ".models" / "Qwen3-VL-4B-Instruct-int4-ov"

PROMPT = """You control two robot arms setting a dinner table. The picture is the overhead camera.
Objects on the table: plate (white, with a rim), mug (red), fork (grey), spoon (tan), bottle (blue, never moved).
Places you may use: {places}.
"Set the table" means: plate center, fork left_of_plate, spoon right_of_plate, mug top_right_of_plate.

Instruction: "{command}"

Answer with one line per object to move, written as `object: place`, and nothing else.
Example answer:
fork: left_of_plate
mug: top_right_of_plate"""
# (Plain `object: place` lines instead of JSON: the model writes ~4 tokens a second on the
# laptop GPU, so every token of JSON punctuation cost time.)


def read_rules(command: str) -> list[Goal]:
    """Keyword reader. Splits on 'and'/commas and matches objects to the place named nearby."""
    text = command.lower()
    if re.search(r"\bset (up )?the table\b|\bplace setting\b|\blay the table\b", text):
        return [Goal(o, p) for o, p in DEFAULT_SETTING.items()]
    goals = []
    for clause in re.split(r",|\band\b|\bthen\b|;", text):
        # The object being moved is the first one mentioned ("the fork on the right of the plate").
        hits = [(m.start(), o) for o, ws in OBJECT_WORDS.items() for w in ws
                for m in re.finditer(rf"\b{w}\b", clause)]
        obj = min(hits)[1] if hits else None
        # Longest phrases first, so "top right" wins over "right".
        place = None
        for p, ws in sorted(PLACE_WORDS.items(), key=lambda kv: -max(len(w) for w in kv[1])):
            if any(w in clause for w in ws):
                place = p
                break
        if obj and (place or obj in DEFAULT_SETTING):
            goals.append(Goal(obj, place or DEFAULT_SETTING[obj]))
    return goals


def validate(raw: dict) -> list[Goal]:
    goals, seen = [], set()
    for g in raw.get("goals", []):
        obj, place = str(g.get("object", "")).lower(), str(g.get("place", "")).lower()
        if obj in DEFAULT_SETTING and place in PLACES and obj not in seen:
            goals.append(Goal(obj, place))
            seen.add(obj)
    return goals


class VisionLanguage:
    # Picture width sent to the model. Fewer pixels -> fewer image tokens -> faster answers;
    # bench/benchmark.py measures the trade-off.
    IMAGE_WIDTH = 480

    def __init__(self, device: str = "GPU", model_dir: Path = MODEL_DIR):
        import openvino as ov
        import openvino_genai as ov_genai

        self._ov = ov
        t = time.time()
        try:
            self.pipe = ov_genai.VLMPipeline(str(model_dir), device)
            self.device = device
        except Exception:
            self.pipe = ov_genai.VLMPipeline(str(model_dir), "CPU")
            self.device = "CPU"
        self.load_seconds = time.time() - t
        self.config = ov_genai.GenerationConfig()
        self.config.max_new_tokens = 60
        self.last = {}

    def read(self, command: str, image_rgb: np.ndarray, width: int | None = None) -> tuple[list[Goal], dict]:
        """Returns (goals, info). info has the raw answer, timing, and which reader was used."""
        from PIL import Image

        prompt = PROMPT.format(places=", ".join(PLACES), command=command)
        w = width or self.IMAGE_WIDTH
        h = round(image_rgb.shape[0] * w / image_rgb.shape[1])
        small = np.asarray(Image.fromarray(image_rgb).resize((w, h), Image.BILINEAR))
        image = self._ov.Tensor(np.ascontiguousarray(small[None], dtype=np.uint8))
        t = time.time()
        result = self.pipe.generate(prompt, images=[image], generation_config=self.config)
        seconds = time.time() - t
        text = result.texts[0] if hasattr(result, "texts") else str(result)
        info = {"reader": f"Qwen3-VL-4B ({self.device})", "raw": text.strip(), "seconds": round(seconds, 1)}
        pm = getattr(result, "perf_metrics", None)
        if pm is not None:
            info["first_token_ms"] = round(pm.get_ttft().mean, 0)
            info["tokens_per_s"] = round(pm.get_throughput().mean, 1)
            info["tokens"] = int(pm.get_num_generated_tokens())
        pairs = re.findall(r"\b(plate|mug|fork|spoon)\s*[:=-]\s*([a-z_]+)", text.lower())
        goals = validate({"goals": [{"object": o, "place": p} for o, p in pairs]})
        if not goals:
            goals = read_rules(command)
            info["reader"] += " -> fell back to keyword reader"
        self.last = info
        return goals, info
