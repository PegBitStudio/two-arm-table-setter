# Hand-over — Two-Arm Table Setter (read this first in a new session)

Written 2026-09-13 at the end of the first Claude Code session. Start a new session by asking
Claude to read this file, then `docs/worklog.md` for the full history.

## The user

- Wants **short answers in plain, simple English**, no jargon; a summary first, then an offer
  to explain more. (This is in their global CLAUDE.md too.)
- Building for **learning**; winning is a bonus. **Zero spend** — free tools and tiers only.
- Their AssemblyAI hackathon build (due 30 Sep) takes priority if time gets tight.

## The hackathon

- lablab.ai **AI Infra Summit Hackathon**, Intel online track *Bimanual VLA Manipulation with
  Multi-Modal Reasoning*. **Deadline: Wed 16 Sep 2026, 7:30 PM WAT** (our target 3 PM).
- Brief and scoring: `docs/brief.md`. Online prizes $3k/$2k/$1k + Speechmatics bonus ($500/$250).
- Rule that matters: must use **two simulated SO-101 arms** (two-finger grippers). The user asked
  about human-like hands — ruled out (breaks the rules, weeks of work).

## Where everything is

| What | Where |
|---|---|
| Project folder (cloud-synced) | `C:\DriveSync\Creative\hackathons\ai-infra-summit-2026` |
| Public repo | https://github.com/PegBitStudio/two-arm-table-setter (gh CLI logged in as PegBitStudio) |
| Demo page (lablab "application URL") | https://pegbitstudio.github.io/two-arm-table-setter/ — GitHub Pages from `docs/` |
| Trained policy download (release) | https://github.com/PegBitStudio/two-arm-table-setter/releases/tag/policy-v7 |
| Python env (outside the synced folder) | `C:\Users\USER\.venvs\ai-infra` (Python 3.12) — run scripts with `C:\Users\USER\.venvs\ai-infra\Scripts\python.exe` |
| Big files (outside the synced folder) | `C:\Users\USER\.models\` — Qwen3-VL model, `runs\act_mug` (final policy), older runs `act_mug_v1..v6` |
| Training data | `train\data\` (git-ignored); final = `mug_rel_all` (1,219 episodes) |
| Submission kit | `submission\` — `cover.png`, `slides.pptx` (8 slides, narration in notes), `lablab_form.md`, `two_arm_table_setter_demo.mp4` (5:41), `status_clip.mp4` (WhatsApp, 23 s) |
| History, decisions, what failed | `docs/worklog.md`; plan + **final audit table**: `docs/plan.md`; setup: `docs/setup.md` |
| Volunteer guide (Core Ultra) | `docs/core_ultra_guide.md`, script `bench/volunteer.py` |

## What it does (plain English)

Two robot arms in a simulator set a dinner table from a spoken or typed command. A camera finds
the items; Qwen3-VL (on OpenVINO) understands the command and picture; a planner decides which
arm does what and passes cutlery between the arms; after every item it looks again and fixes
anything knocked out of place. One skill (moving the mug) is a trained LeRobot ACT policy.

## Numbers (all on the user's laptop: i7-1165G7, Iris Xe, 16 GB, no NPU)

- Set the table: **30/30** normal, **29/30** hard tables; demo video **10/10** hard tables.
- Camera: 0.5 mm mean error. Voice: Speechmatics 2.6 s. Qwen3-VL: ~4 s/command, 5/5 correct.
- ACT policy: **19/20 within 1.5 cm** on unseen tables (v7, 9k steps).
- OpenVINO: ACT **2.5× faster** than PyTorch on CPU, task success unchanged at every precision.
  Full table: `bench/RESULTS.md`.

## Main commands

```bash
python brain/run.py --seed 3 "set the table"               # add --video, --hard, --policy, --bump plate, --audio file.wav
python brain/evaluate.py --seeds 30 [--hard] [--model]      # score over random tables
python train/eval_policy.py --episodes 20 [--backend openvino]
python bench/benchmark.py [--only vlm|act|act_task|perception] [--report-only]
python brain/make_video.py --seeds 10 --start 100 --hard --model
python submission/make_final_video.py                       # join demo + recovery + benchmark cards
python bench/volunteer.py [--quick]                         # the Core Ultra volunteer's one command
```

## Gotchas (learned the hard way)

- **The PC sleeps overnight** and pauses long jobs — request keep-awake before long background work.
- **Windows allows one process ~300 render windows** — record demos in batches of ≤270
  (`record_demos.py --seed-start` for parallel recorders, `--resume` to continue).
- **Close MuJoCo renderers with `robot.close_renderer()`** — `Renderer.close()` frees GPU objects
  in the wrong context and blacks out every other renderer.
- **PowerShell cwd resets between calls** — always use full paths to scripts.
- **Background output is buffered** when piped through grep — add `--line-buffered` or wait for the end.
- LeRobot's `checkpoints/last` symlink needs admin on Windows — `train_act.py` patches it to a copy.
- `torchcodec` DLLs don't load on Windows — harmless warnings; uninstalled in our env.
- The **Speechmatics key** is in the user's environment (`SPEECHMATICS_API_KEY`, set by them with
  setx). Never ask them to paste it in chat.
- **Benchmark numbers swing with power state** (PyTorch ACT measured 4.3 ms plugged in, 12 ms in a
  later run) — run benchmarks plugged in, nothing else running, and compare within one run.
- Don't claim anything untested (we removed "spoken" claims until voice was tested, then restored them).

## If this session is on the Core Ultra laptop

The user's **brother's Intel Core Ultra laptop** is being lent to the user, who runs everything on it
personally with Claude Code. Step-by-step guide: **`docs/core_ultra_setup.html`** (also on the demo site:
https://pegbitstudio.github.io/two-arm-table-setter/core_ultra_setup.html).

- Project at `C:\work\two-arm-table-setter`, environment in `.venv` there (not the paths above).
- Run order: `bench/volunteer.py --quick` → `bench/volunteer.py` (results in `volunteer_results/`) →
  try NPU (`run.py --device NPU`, `--policy --policy-device NPU`) → film with `make_video.py ... --device <best>`
  → `submission/make_final_video.py --results volunteer_results/results.json`.
- Video cards and reports read the hardware name automatically (`brain/hardware.py`).
- It's someone else's laptop: remind the user to sign out of Claude/GitHub, delete `C:\work\...` and
  `~/.models`, and remove the Speechmatics key before handing it back (guide step 11).

## Open items / next steps

1. **Core Ultra runs** — the user now has the laptop (see above). When results come back, add the
   numbers to `bench/RESULTS.md`, README, slides and the demo page, and swap in the Core Ultra demo
   video (this closes the biggest scoring gap: "run on Core Ultra 2/3", NPU).
2. **Live listening (optional, for the Speechmatics bonus):** microphone → Speechmatics real-time
   API → robot acts. Offered to the user, not started (~half a day).
3. **Narrated pitch video (optional):** voice-over from the slide notes.
4. **User's own steps:** lablab + Discord sign-up and team; upload the video (YouTube); submit
   with `submission/lablab_form.md` before Wed 16 Sep 3 PM WAT.
5. Known misses: hard seed 3 (spoon starts too close to an arm's base); drawer and pouring not
   built; most skills are scripted (one learned).

## Session memory

A memory file with the key facts also exists at
`C:\Users\USER\.claude\projects\C--DriveSync-Creative-hackathons\memory\ai-infra-summit-hackathon.md`.
