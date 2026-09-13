# lablab submission — copy-paste text

## Project title
Two-Arm Table Setter

## Short description (one line)
Two simulated SO-101 arms set a dinner table from a spoken or typed command: they see with a camera, understand with Qwen3-VL on OpenVINO, hand items to each other, and fix their own mistakes — all on an Intel laptop.

## Long description
Say "set the table" — or "put the red thing top left of the plate and the grey utensil on the right" — and two SO-101 robot arms in MuJoCo do it.

**How it works.** An overhead colour-and-depth camera finds every object from its height and shape (0.5 mm average error on 20 random tables; the simulator's positions are never used to act). Qwen3-VL-4B, running locally as Intel's INT4 OpenVINO build, reads the command together with the camera picture, so it understands "the red thing" means the mug. A planner decides which arm does what: cutlery that starts on the wrong side is passed hand to hand, and round items are set down where both arms can reach. After every item the robot looks again, and anything knocked out of place is put back first.

**Learning.** Our scripted skills became the teacher: they recorded 1,219 successful demonstrations automatically, and a LeRobot ACT policy trained on a laptop CPU learned the mug skill — 19/20 within 1.5 cm on tables it had never seen. It took seven versions; the write-up covers what failed and why (an inconsistent teacher, drifting small-step actions, the wrong reference frame).

**Intel optimisation.** OpenVINO runs the ACT policy 2.5× faster than PyTorch on the CPU with no loss of task success, measured at FP32, FP16, INT8 and INT8-weights. Command understanding went from 30 s to about 4 s through smaller pictures and shorter answers; the Iris Xe iGPU halves the time for large pictures.

**Results.** 30/30 random tables set correctly; 29/30 on harder tables (±4 cm starts, object sizes ±5–10%, varied colours). Every number is checked against simulator truth and reproducible from one command. Hardware: Intel Core i7-1165G7 with Iris Xe, 16 GB, no NPU, no cloud.

## Technology & category tags
OpenVINO, Intel, LeRobot, ACT, Qwen3-VL, MuJoCo, SO-101, Robotics, Physical AI, Vision-Language Model, Imitation Learning, NNCF

(Add "Speechmatics" only once voice input has been tested with a real API key — the code is in
`brain/voice.py` but it has never run.)

## Links
- GitHub: https://github.com/PegBitStudio/two-arm-table-setter
- Demo video: (upload `brain/out/demo_10_tables.mp4` or the narrated version)
- Slides: `submission/slides.pptx` (export to PDF if the form asks)
- Cover image: `submission/cover.png`
