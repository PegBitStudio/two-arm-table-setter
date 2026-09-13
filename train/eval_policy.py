"""Test the trained ACT policy on tables it has never seen (seeds 50 000+; training used 0–~200).

    python train/eval_policy.py --episodes 20                      # PyTorch
    python train/eval_policy.py --episodes 20 --backend openvino   # OpenVINO IR (bench/export_act.py)

Each run: random table, random mug start, random target. The policy alone drives arm B's six
motors 20 times a second for up to 14 s. Success = the mug rests upright within 1.5 cm of the
target (simulator truth).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import record_demos as R
import skills
import day2_demo

RUN = Path.home() / ".models" / "runs" / "act_mug"
EVAL_SEED0 = 50_000
MAX_SECONDS = 14.0


def load_torch(ckpt: Path, ensemble: float | None = None):
    """ensemble: ACT's temporal ensembling coefficient. When set, the policy predicts a new
    1-second chunk every tick and blends all overlapping predictions (smoother, more stable),
    instead of playing each chunk out blind."""
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    config = None
    if ensemble is not None:
        config = ACTConfig.from_pretrained(str(ckpt))
        config.temporal_ensemble_coeff = ensemble
        config.n_action_steps = 1
    policy = ACTPolicy.from_pretrained(str(ckpt), config=config)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ckpt))
    return policy, pre, post


def io_dims(policy) -> tuple[int, int]:
    """(task-input size, action size) — tells the action mode apart (joints / ee / abs)."""
    return (policy.config.input_features["observation.environment_state"].shape[0],
            policy.config.output_features["action"].shape[0])


def env_names(ckpt: Path) -> list[str]:
    """Names of the task inputs, from the dataset the policy was trained on."""
    import json

    root = Path(json.loads((ckpt / "train_config.json").read_text())["dataset"]["root"])
    info = json.loads((root / "meta" / "info.json").read_text())
    return info["features"]["observation.environment_state"]["names"]


class TorchRunner:
    def __init__(self, ckpt: Path, ensemble: float | None = None):
        self.policy, self.pre, self.post = load_torch(ckpt, ensemble)
        self.env_dim, self.act_dim = io_dims(self.policy)
        self.env_names = env_names(ckpt)

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def act(self, state, env):
        batch = self.pre({"observation.state": torch.from_numpy(state),
                          "observation.environment_state": torch.from_numpy(env),
                          "task": R.TASK})
        return self.post(self.policy.select_action(batch)).squeeze(0).numpy()


def run_episode(runner, seed: int) -> tuple[bool, str, float]:
    sim, target = R.new_episode(seed)
    try:
        skills.observe(sim)
        mug_xy = (sim.world["mug"].x, sim.world["mug"].y)
        a = sim.arms[R.ARM]
        mug_frame = runner.env_dim == len(R.MUG_ENV_NAMES) and "target_dx" in runner.env_names
        env_fn = (R.task_input_mug if mug_frame else
                  R.task_input_abs if runner.env_dim == len(R.ABS_ENV_NAMES) else R.task_input)
        runner.reset()
        infer = []
        while sim.d.time < MAX_SECONDS:
            t = time.perf_counter()
            action = runner.act(R.arm_state(sim), env_fn(sim, mug_xy, target))
            infer.append(time.perf_counter() - t)
            if mug_frame:                            # hand target measured from the mug
                R.apply_mug(sim, action, mug_xy)
            elif runner.act_dim == len(R.ABS_NAMES):  # hand-target policy: IK turns it into joints
                R.apply_abs(sim, action)
            elif runner.act_dim == len(R.EE_NAMES):  # small-move policy
                R.apply_ee(sim, action)
            else:                                    # joint-angle policy
                sim.d.ctrl[a.act] = action[:5]
                sim.d.ctrl[a.grip_act] = action[5]
            sim.step(1 / R.FPS)
        why = skills.why_not(sim, "mug", target, tol=0.015)
        offset = skills.truth_pose(sim, "mug")[0][:2] - target
        return why == "", why, float(np.mean(infer)), offset, skills.why_not(sim, "mug", target, tol=0.025) == ""
    finally:
        day2_demo.close_sim(sim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--backend", choices=["torch", "openvino"], default="torch")
    p.add_argument("--device", default="CPU", help="OpenVINO device")
    p.add_argument("--ensemble", type=float, help="temporal ensembling coefficient, e.g. 0.01")
    args = p.parse_args()
    ckpt = args.run / "checkpoints" / "last" / "pretrained_model"
    if args.backend == "torch":
        runner = TorchRunner(ckpt, args.ensemble)
    else:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
        from export_act import OpenVINORunner
        runner = OpenVINORunner(ckpt, device=args.device)
    wins, loose, times, offsets = 0, 0, [], []
    for i in range(args.episodes):
        ok, why, t, off, ok25 = run_episode(runner, EVAL_SEED0 + i)
        wins += ok
        loose += ok25
        times.append(t)
        offsets.append(off)
        print(f"episode {i:2d}  {'PASS' if ok else 'FAIL'}  {why}  (mug - target: "
              f"{100 * off[0]:+.1f}, {100 * off[1]:+.1f} cm)")
    n = args.episodes
    mean_off = 100 * np.mean(offsets, axis=0)
    print(f"\n{wins}/{n} within 1.5 cm ({100 * wins / n:.0f}%), {loose}/{n} within 2.5 cm "
          f"({100 * loose / n:.0f}%); average miss {mean_off[0]:+.1f}, {mean_off[1]:+.1f} cm; "
          f"policy call {1000 * np.mean(times):.2f} ms ({args.backend})")


if __name__ == "__main__":
    main()
