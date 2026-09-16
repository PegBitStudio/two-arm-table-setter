# Submission checklist — deadline Wed 16 Sep, 7:30 PM WAT

Everything in the repo is done and pushed. What is left is the part only you can do: the lablab
account, the uploads, and the form. Work top to bottom — the first three items block submitting
at all, so do them first even if you want to polish something else.

## 1. Blocking — do these first

- [ ] **Register on lablab.ai** (an account is required to submit).
- [ ] **Join the lablab Discord** — the rules require both.
- [ ] **Create the team on lablab.** A team must exist before you can submit, even solo.
      Teams are 1–5 people; your brother can be added if you want him credited for the
      Core Ultra run.

## 2. Uploads

- [ ] **Pitch video → YouTube** (`submission/pitch_video.mp4`, 3:25). This is the one judges
      watch first. Unlisted is fine.
- [ ] **Demo video → YouTube** (`submission/two_arm_table_setter_demo.mp4`, 5:41) — the robot
      actually working, 10 tables, recovery, benchmark cards.
- [ ] Paste both links into the form. If it only takes one, use the **pitch video**.

## 3. The form

Copy the text from [`lablab_form.md`](lablab_form.md) — it is written to be pasted as-is:

| Field | Where it comes from |
|---|---|
| Title | Two-Arm Table Setter |
| Short description | one line, in the form file |
| Long description | the long section, already includes the Core Ultra results |
| Tags | the tag list in the form file |
| Cover image | `submission/cover.png` |
| Slides | `submission/slides.pptx` (export to PDF if it asks) |
| Demo platform | GitHub Pages |
| Application URL | https://pegbitstudio.github.io/two-arm-table-setter/ |
| Repo | https://github.com/PegBitStudio/two-arm-table-setter |

## 4. Worth checking before you hit submit

- [ ] The demo page loads: https://pegbitstudio.github.io/two-arm-table-setter/
- [ ] The repo is public and the README shows the Core Ultra table.
- [ ] Both YouTube links actually play in a private window.

## 5. If you have time left over

- **Re-record the pitch narration in your own voice.** The video already has a proper
  ElevenLabs voice-over, so this is optional polish only. The script is the speaker notes in
  `submission/build_slides.js` — eight blocks. Record one wav each into
  `submission/voice/narration_01.wav` … `_08.wav`, then:

      python submission/make_pitch_video.py --keep-audio

  Nothing else changes; the cards re-time themselves to your recording.

## What is deliberately not in the submission

Said plainly here so nothing is oversold in the form:

- The **drawer and the pouring** from Intel's example task are not built. Pouring was attempted
  and is on the `pour-experiment` branch with the measurements that explain why it does not work
  yet — a 5-DOF SO-101 gripping top-down runs out of wrist before it can tip a jug over a mug.
- The **vision-language model cannot compile for the NPU** on OpenVINO 2026.3; it falls back to
  the Arc iGPU automatically. The ACT policy does run on the NPU and is benchmarked there.
- **Most skills are scripted**, not learned; one (the mug pick-and-place) is a trained ACT policy.
