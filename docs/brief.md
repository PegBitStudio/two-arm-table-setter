# Challenge Brief — Intel Online Track

Captured 2026-09-11 from the event page and Intel's 5-page PDF brief
("Online_Physical_AI_Challenge_Online.pdf", Google Drive link on the event page).

- Event page: https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon
- Intel setup guide: https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html

## Key dates (West Africa Time)

| | |
|---|---|
| Online build | 10–16 Sep 2026 |
| **Submission deadline** | **Wed 16 Sep, 7:30 PM WAT** |
| Onsite (invite only, not us) | 15–17 Sep, Santa Clara |

## The task

**Bimanual VLA Manipulation with Multi-Modal Reasoning.** Two simulated SO-101 robot arms in MuJoCo
take a natural-language instruction, look at camera images, coordinate both arms, and complete a
multi-step dinner-table task.

Example command: *"Open the top drawer, pick up the plate with arm A, place it on the table, pick up
the mug with arm B, pour water into the mug with arm A."*

Scenario elements: open a drawer, fetch spoons/forks, pick up a plate and cup, hand objects between
arms, one arm holds a mug while the other pours.

## Technical objectives (from the brief)

1. **Bimanual manipulation** — two-arm control, object hand-off, collision-aware sequencing.
2. **Multi-modal reasoning** — language + raw camera images → identify objects, track task state, pick next action.
3. **Robustness** — must still work when object weight, friction, shape, lighting, background and start positions change.
4. **Simulation training** — train or fine-tune a policy in MuJoCo with Hugging Face LeRobot (SmolVLA, Pi0.5, ACT, or similar).
5. **Intel edge optimisation** — convert to OpenVINO, quantise, run on Intel CPU / iGPU / NPU.

Workflow they expect: **Observe → Understand → Plan → Act → Optimise.**

## Platform

- Target system: "Intel Core Ultra AI PC / NUC, **or a local execution environment with Intel CPU and iGPU acceleration**."
- Preferred for the final demo: Core Ultra Series 2 or 3.
- Training: any local machine or cloud. **Intel does not provide training hardware.**
- No physical robot needed — simulation only.
- Intel's official install scripts are **Ubuntu 24.04 only**.

## Required deliverables (Intel)

1. Reproducible GitHub repo — setup, dependencies, MuJoCo scene, training code, eval code, inference code, run commands.
2. Reproducible MuJoCo simulation — the dinner-table scene with randomisation settings.
3. Intel inference benchmark script — latency, throughput, device, precision.
4. **Demo video — the task succeeding across 10 random seeds.**
5. Technical README — architecture, model choice, two-arm strategy, training, robustness, OpenVINO, hardware.

Plus lablab's standard submission: title, short + long description, tags, cover image, video,
slides, public GitHub repo, demo platform, app URL.

## Scoring (100 points)

| Criterion | Points |
|---|---|
| End-to-end task completion & two-arm manipulation | **30** |
| VLA / multi-modal reasoning | **20** |
| OpenVINO & Intel Core Ultra optimisation | **20** |
| Robustness across 10 random seeds | 15 |
| Technical quality & reproducibility | 10 |
| Innovation & presentation | 5 |

lablab's general criteria also apply: application of technology, presentation, business value, originality.

## Prizes

Online track: **$3,000 / $2,000 / $1,000**.
Bonus (stacks): Best Use of Speechmatics — $500 + 1,000 credits / $250 + 500 credits / 250 credits.
Prize tax rules: see lablab payout notes (30% US withholding for non-US winners unless treaty + TIN).

## Teams

1–5 people. Everyone must register on lablab.ai **and** the lablab Discord. A team must exist on
lablab to submit. Prize money is paid to named individuals, not the team.
