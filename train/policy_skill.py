"""The trained ACT policy as a skill the planner can call instead of scripted pick + place.

It is only used where it was trained: arm B, the mug, a target inside TARGET_BOX (with a
small margin). Anywhere else the planner keeps the scripted skills — see planner.plan_move.
"""
from pathlib import Path

import numpy as np

import record_demos as R
import skills

RUN = Path.home() / ".models" / "runs" / "act_mug"
MAX_SECONDS = 12.0


def load(backend: str = "openvino", device: str = "CPU", precision: str = "fp16"):
    from eval_policy import TorchRunner

    ckpt = RUN / "checkpoints" / "last" / "pretrained_model"
    if backend == "torch":
        return TorchRunner(ckpt)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    from export_act import OpenVINORunner
    return OpenVINORunner(ckpt, device=device, precision=precision)


def covers(arm: str, obj: str, target, margin: float = 0.005) -> bool:
    (x0, x1), (y0, y1) = R.TARGET_BOX
    return (arm == R.ARM and obj == "mug"
            and x0 - margin <= target[0] <= x1 + margin and y0 - margin <= target[1] <= y1 + margin)


def run(sim, runner, target) -> None:
    """Let the policy drive arm B until it has put the mug down (or time runs out)."""
    mug_xy = (sim.world["mug"].x, sim.world["mug"].y)
    runner.reset()
    start = sim.d.time
    sim.log.append(f"{R.ARM}: ACT policy moves mug to {np.round(target, 3).tolist()}")
    while sim.d.time - start < MAX_SECONDS:
        action = runner.act(R.arm_state(sim), R.task_input_mug(sim, mug_xy, target))
        R.apply_mug(sim, action, mug_xy)
        sim.step(1 / R.FPS)
    sim.held.pop(R.ARM, None)
    skills.home(sim, [R.ARM])
