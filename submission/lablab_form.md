# lablab submission — copy-paste text

## Project title
Two-Arm Table Setter

## Short description (one line)
Two simulated SO-101 arms set a dinner table from a spoken or typed command: they see with a camera, understand with Qwen3-VL on OpenVINO, hand items to each other, and fix their own mistakes — all on an Intel Core Ultra laptop, using its CPU, Arc iGPU and NPU through OpenVINO.

## Long description
Say "set the table" (Speechmatics turns speech into text in about 2.6 s) — or "put the red thing top left of the plate and the grey utensil on the right" — and two SO-101 robot arms in MuJoCo do it.

**How it works.** An overhead colour-and-depth camera finds every object from its height and shape (0.5 mm average error on 20 random tables; the simulator's positions are never used to act). Qwen3-VL-4B, running locally as Intel's INT4 OpenVINO build, reads the command together with the camera picture, so it understands "the red thing" means the mug. A planner decides which arm does what: cutlery that starts on the wrong side is passed hand to hand, and round items are set down where both arms can reach. After every item the robot looks again, and anything knocked out of place is put back first.

**Learning.** Our scripted skills became the teacher: they recorded 1,219 successful demonstrations automatically, and a LeRobot ACT policy trained on a laptop CPU learned the mug skill — 19/20 within 1.5 cm on tables it had never seen. It took seven versions; the write-up covers what failed and why (an inconsistent teacher, drifting small-step actions, the wrong reference frame).

**Intel optimisation.** Measured on an Intel Core Ultra 7 155H across all three of its chips. OpenVINO runs the ACT policy 2.9× faster than PyTorch on the CPU (0.72 ms vs 2.07 ms per call) with no loss of task success, at FP32, FP16, INT8 and INT8-weights. The same policy runs on the **AI Boost NPU** at 0.90 ms per call and places 10/10 mugs, freeing the CPU and iGPU. Command understanding runs on the **Arc iGPU** in 1.3 s per command — 2.8× faster than the CPU — after cutting it from 30 s with smaller pictures and shorter answers. We also report an honest negative result: OpenVINO 2026.3's NPU compiler cannot build Qwen3-VL-4B's vision tower, so the language model falls back to the iGPU automatically.

**Why it matters.** Service robots in homes, hospitals, hotels and restaurants need exactly this combination: instructions anyone can give in plain language, two-handed handling, and checking their own work so a bump doesn't ruin the job. It is also cheap: everything runs on an ordinary Intel laptop, and skills are taught from demonstrations the robot records itself — no hand-collected data, usually the most expensive part of robot learning.

**Results.** 30/30 random tables set correctly; 29/30 on harder tables (±4 cm starts, object sizes ±5–10%, varied colours); 9/10 hard tables on the Core Ultra at 8 s per table. Every number is checked against simulator truth, never the robot's own camera, reproducible from one command (`python bench/volunteer.py`), and the raw Core Ultra logs ship in `bench/core_ultra/`. Hardware: Intel Core Ultra 7 155H — CPU, Arc iGPU and AI Boost NPU — Windows 11, OpenVINO 2026.3.1, no cloud.

## Technology & category tags
OpenVINO, Intel Core Ultra, NPU, Intel, LeRobot, ACT, Qwen3-VL, MuJoCo, SO-101, Robotics, Physical AI, Vision-Language Model, Imitation Learning, NNCF, Speechmatics

## App hosting
- **Demo application platform:** GitHub Pages (simulation demo page — the robot itself runs locally)
- **Application URL:** https://pegbitstudio.github.io/two-arm-table-setter/

## Links
- GitHub: https://github.com/PegBitStudio/two-arm-table-setter
- Pitch video (narrated, 3:25): `submission/pitch_video.mp4` — upload to YouTube and paste the link.
  Rebuild with `python submission/make_pitch_video.py`; the words are the deck's own speaker notes.
- Demo video (5:41, the robot working): `submission/two_arm_table_setter_demo.mp4` (upload to YouTube and paste the link)
- Slides: `submission/slides.pptx` (export to PDF if the form asks)
- Cover image: `submission/cover.png`
