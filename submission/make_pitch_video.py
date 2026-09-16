"""Build the narrated pitch video from the deck's own speaker notes.

    python submission/make_pitch_video.py              # -> submission/pitch_video.mp4
    python submission/make_pitch_video.py --voice "Microsoft David Desktop"
    python submission/make_pitch_video.py --keep-audio # reuse an existing voice track

The script is the single source of truth for the pitch: the words come straight out of
`build_slides.py`'s speaker notes, so the deck and the video can never drift apart, and each
card is held on screen for exactly as long as its own line of narration.

The voice is Windows' built-in speech synthesiser, which is free and offline but plainly
robotic. To use a real voice instead, record one wav per narration block into
`submission/voice/narration_01.wav` ... `_08.wav` and re-run with --keep-audio; the script
prints the script for each block so they can be read aloud. Needs ffmpeg on PATH.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submission"
VOICE = SUB / "voice"
FRAMES = SUB / "_pitch_frames"
W, H = 1280, 720

# The deck's palette, so the video and the slides look like one piece of work.
BG = (0x16, 0x1B, 0x22)
PANEL = (0x22, 0x2A, 0x35)
YELLOW = (0xE8, 0xB9, 0x23)
WHITE = (0xF0, 0xF2, 0xF5)
MUTE = (0x9A, 0xA5, 0xB1)
GREEN = (0x4A, 0xDE, 0x80)

BOLD = "C:/Windows/Fonts/segoeuib.ttf"
REG = "C:/Windows/Fonts/segoeui.ttf"

NOTE_RE = re.compile(r'addNotes\(\s*"((?:[^"\\]|\\.)*)"\s*\)', re.S)


def font(size, bold=True):
    return ImageFont.truetype(BOLD if bold else REG, size)


def narration():
    """The speaker notes from the deck, in order."""
    src = (SUB / "build_slides.js").read_text(encoding="utf-8")
    notes = [m.group(1) for m in NOTE_RE.finditer(src)]
    if not notes:
        sys.exit("no speaker notes found in build_slides.js")
    return [n.replace('\\"', '"').replace("\\'", "'").replace("\\n", " ") for n in notes]


def speak(notes, voice, rate):
    """One wav per narration block, using the built-in Windows voice."""
    VOICE.mkdir(exist_ok=True)
    script = VOICE / "_say.ps1"
    script.write_text(
        "param([string]$Json,[string]$Out,[string]$Voice,[int]$Rate)\n"
        "Add-Type -AssemblyName System.Speech\n"
        "$notes = Get-Content -Raw -Encoding UTF8 $Json | ConvertFrom-Json\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "$s.SelectVoice($Voice); $s.Rate = $Rate\n"
        "$i = 0\n"
        "foreach ($n in $notes) {\n"
        "  $i++\n"
        "  $s.SetOutputToWaveFile((Join-Path $Out ('narration_{0:d2}.wav' -f $i)))\n"
        "  $s.Speak($n)\n"
        "}\n"
        "$s.SetOutputToNull(); $s.Dispose()\n", encoding="utf-8")
    payload = VOICE / "_notes.json"
    payload.write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(script), "-Json", str(payload), "-Out", str(VOICE),
                    "-Voice", voice, "-Rate", str(rate)], check=True)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip())


def card(title, sub=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill=YELLOW)
    if title:
        d.text((64, 54), title, font=font(40), fill=WHITE)
    if sub:
        d.text((64, 112), sub, font=font(19, False), fill=MUTE)
    return img, d


def bullets(d, items, top=200, gap=74):
    y = top
    for label, detail in items:
        d.ellipse([64, y + 12, 78, y + 26], fill=YELLOW)
        d.text((100, y), label, font=font(27), fill=WHITE)
        if detail:
            d.text((100, y + 36), detail, font=font(18, False), fill=MUTE)
        y += gap


def stat_row(d, stats, y=250):
    box_w = (W - 128 - 30 * (len(stats) - 1)) // len(stats)
    x = 64
    for value, label in stats:
        d.rounded_rectangle([x, y, x + box_w, y + 190], radius=12, fill=PANEL)
        tw = d.textlength(value, font=font(46))
        d.text((x + (box_w - tw) / 2, y + 40), value, font=font(46), fill=YELLOW)
        line, lines = "", []
        for word in label.split():
            trial = (line + " " + word).strip()
            if d.textlength(trial, font=font(16, False)) > box_w - 36:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        ly = y + 108
        for ln in lines[:3]:
            lw = d.textlength(ln, font=font(16, False))
            d.text((x + (box_w - lw) / 2, ly), ln, font=font(16, False), fill=MUTE)
            ly += 22
        x += box_w + 30


def picture_card(title, sub, paths, captions, footer=None):
    """The renders are different shapes, so each is fitted into a common box and centred in it,
    and every caption sits on the same line underneath."""
    img, d = card(title, sub)
    box_w = (W - 128 - 30 * (len(paths) - 1)) // len(paths)
    box_top, box_h = 186, 330
    x = 64
    for path, cap in zip(paths, captions):
        try:
            pic = Image.open(path).convert("RGB")
        except OSError:
            x += box_w + 30
            continue
        ratio = min(box_w / pic.width, box_h / pic.height)
        pic = pic.resize((int(pic.width * ratio), int(pic.height * ratio)), Image.LANCZOS)
        img.paste(pic, (x + (box_w - pic.width) // 2, box_top + (box_h - pic.height) // 2))
        cw = d.textlength(cap, font=font(18, False))
        d.text((x + (box_w - cw) / 2, box_top + box_h + 18), cap, font=font(18, False), fill=MUTE)
        x += box_w + 30
    if footer:
        d.text((64, box_top + box_h + 76), footer, font=font(19, False), fill=MUTE)
    return img


def build_cards():
    FRAMES.mkdir(exist_ok=True)
    cards = []

    img, d = card(None)
    d.text((64, 230), "Two-Arm Table Setter", font=font(62), fill=WHITE)
    d.text((64, 320), "Say it. Two SO-101 arms set the table.", font=font(30, False), fill=YELLOW)
    d.text((64, 380), "Camera, vision-language model, two hands, and it fixes its own mistakes.",
           font=font(21, False), fill=MUTE)
    d.text((64, 430), "Intel Core Ultra 7 155H  -  CPU, Arc iGPU and AI Boost NPU  -  OpenVINO",
           font=font(21, False), fill=MUTE)
    d.text((64, 620), "AI Infra Summit Hackathon  -  Intel online track",
           font=font(18, False), fill=MUTE)
    cards.append(img)

    img, d = card("Measured, not claimed",
                  "every number checked against the simulator's own truth, never the robot's camera")
    stat_row(d, [("30/30", "random tables set correctly"),
                 ("29/30", "harder: further apart, resized, recoloured"),
                 ("95%", "trained policy places the mug within 1.5 cm"),
                 ("0.5 mm", "camera's average position error")])
    cards.append(img)

    img, d = card("Why it matters",
                  "what service robots in homes, hospitals and restaurants need")
    bullets(d, [("Anyone can ask, in plain language",
                 "no teaching, no scripting, no app - just say what you want"),
                ("Two hands that cooperate",
                 "passing, holding and placing together, not one arm at a time"),
                ("It checks its own work", "a knock mid-task does not ruin the job"),
                ("Cheap to run, and it teaches itself",
                 "an ordinary Intel laptop; skills learn from demos the robot records")])
    cards.append(img)

    img, d = card("Observe, understand, plan, act - then look again",
                  "the camera image is the only way the robot learns where anything is")
    steps = [("1", "OBSERVE", "colour + depth"), ("2", "UNDERSTAND", "Qwen3-VL-4B"),
             ("3", "PLAN", "who does what"), ("4", "ACT", "two arms at once"),
             ("5", "LOOK AGAIN", "after every item")]
    box_w = (W - 128 - 22 * 4) // 5
    x = 64
    for num, name, detail in steps:
        d.rounded_rectangle([x, 250, x + box_w, 440], radius=12, fill=PANEL)
        d.ellipse([x + box_w // 2 - 22, 278, x + box_w // 2 + 22, 322], fill=YELLOW)
        nw = d.textlength(num, font=font(22))
        d.text((x + box_w // 2 - nw / 2, 289), num, font=font(22), fill=BG)
        nw = d.textlength(name, font=font(19))
        d.text((x + box_w // 2 - nw / 2, 342), name, font=font(19), fill=WHITE)
        dw = d.textlength(detail, font=font(15, False))
        d.text((x + box_w // 2 - dw / 2, 372), detail, font=font(15, False), fill=MUTE)
        x += box_w + 22
    d.text((64, 500), "The loop is what makes it robust: every placement is re-checked, and "
                      "anything knocked out of place is put back first.",
           font=font(19, False), fill=MUTE)
    cards.append(img)

    cards.append(picture_card(
        "Two hands, and it fixes its own mistakes",
        "cutlery that starts on the wrong side is passed across; a knocked plate is noticed and put back",
        [SUB / "img/handoff_crop.png", SUB / "img/recovery_bumped.png",
         SUB / "img/recovery_fixed.png"],
        ["arm to arm", "knocked out of place", "noticed, and put back"],
        footer="Every hand-off and every recovery is driven by the camera - the robot is never "
               "told where anything really is."))

    img, d = card("Our own skills taught the policy",
                  "no hand-collected data - the most expensive part of robot learning")
    stat_row(d, [("1,219", "demonstrations recorded automatically, failures thrown away"),
                 ("7", "versions - the first six failed, each one taught us something"),
                 ("9k", "training steps, on a laptop CPU"),
                 ("95%", "mug placed within 1.5 cm on tables never seen")])
    cards.append(img)

    img, d = card("OpenVINO on all three Intel chips",
                  "Core Ultra 7 155H  -  CPU, Arc iGPU, AI Boost NPU  -  OpenVINO 2026.3.1")
    xs = [64, 300, 470, 640, 810]
    for x, h in zip(xs, ["ACT policy", "CPU", "Arc iGPU", "NPU", "mug placed"]):
        d.text((x, 190), h, font=font(19), fill=YELLOW)
    y = 232
    for row in [["PyTorch FP32", "2.07 ms", "-", "-", "9/10"],
                ["OpenVINO FP32", "0.83 ms", "1.49 ms", "0.90 ms", "9/10"],
                ["OpenVINO INT8", "0.72 ms", "1.42 ms", "1.10 ms", "9/10"],
                ["Qwen3-VL-4B", "3.6 s", "1.3 s", "will not compile", "5/5"]]:
        for x, v in zip(xs, row):
            d.text((x, y), v, font=font(21, False), fill=WHITE)
        y += 42
    d.text((64, 470), "2.9x faster than PyTorch on the CPU, and the whole policy also runs on the "
                      "NPU at 0.90 ms - 10/10 mugs placed.", font=font(19, False), fill=GREEN)
    d.text((64, 512), "Honest finding: OpenVINO's NPU compiler cannot build the vision-language "
                      "model, so it falls back to the iGPU on its own.",
           font=font(19, False), fill=MUTE)
    cards.append(img)

    img, d = card("Open, reproducible, honest")
    bullets(d, [("Every number here comes from one command",
                 "seeded scenes, scripts in the repo, raw Core Ultra logs in bench/core_ultra/"),
                ("What is missing, said plainly",
                 "the VLM will not compile for the NPU; most skills are scripted; no drawer or pouring")],
            top=190, gap=96)
    d.rounded_rectangle([64, 400, W - 64, 560], radius=12, fill=PANEL)
    d.text((100, 430), "github.com/PegBitStudio/two-arm-table-setter", font=font(26), fill=YELLOW)
    d.text((100, 478), "pegbitstudio.github.io/two-arm-table-setter", font=font(24, False),
           fill=WHITE)
    d.text((100, 518), "MIT licence  -  built for the AI Infra Summit Hackathon, Intel online track",
           font=font(17, False), fill=MUTE)
    d.text((64, 620), "Thank you.", font=font(28), fill=WHITE)
    cards.append(img)

    paths = []
    for i, img in enumerate(cards, 1):
        p = FRAMES / f"card_{i:02d}.png"
        img.save(p)
        paths.append(p)
    return paths


def ffmpeg(args):
    res = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if res.returncode:
        sys.exit(res.stderr[-1500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Microsoft Zira Desktop")
    ap.add_argument("--rate", type=int, default=-1, help="Windows speech rate, -10..10")
    ap.add_argument("--keep-audio", action="store_true",
                    help="use the wavs already in submission/voice (e.g. a real recording)")
    args = ap.parse_args()

    notes = narration()
    if not args.keep_audio:
        speak(notes, args.voice, args.rate)
    wavs = sorted(VOICE.glob("narration_*.wav"))
    cards = build_cards()
    if len(wavs) != len(cards):
        sys.exit(f"{len(wavs)} voice clips but {len(cards)} cards - "
                 f"record one wav per narration block")

    segments = []
    for i, (png, wav) in enumerate(zip(cards, wavs), 1):
        seconds = duration(wav) + 0.6          # a breath at the end of each card
        out = FRAMES / f"seg_{i:02d}.mp4"
        ffmpeg(["-loop", "1", "-i", str(png), "-i", str(wav), "-t", f"{seconds:.2f}",
                "-vf", f"scale={W}:{H},format=yuv420p", "-c:v", "libx264", "-preset", "medium",
                "-crf", "20", "-r", "25", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
                "-ac", "2", "-af", "adelay=300|300,apad", "-shortest", str(out)])
        segments.append(out)
        print(f"  card {i}: {seconds:.1f}s")

    listing = FRAMES / "segments.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in segments), encoding="utf-8")
    final = SUB / "pitch_video.mp4"
    ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", str(final)])
    print(f"\nsaved {final}  ({duration(final):.0f}s, {final.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
