"""Train an ACT policy (LeRobot) on the recorded mug demonstrations — on a CPU, no GPU needed.

    python train/train_act.py                  # 8k steps, ~1.5 h on a 4-core laptop CPU
    python train/train_act.py --steps 200      # quick smoke test

This runs LeRobot's own `lerobot-train` with our settings, so the run is standard and
reproducible (plus one Windows fix, see _copy_not_link). The model is a smaller ACT than the default (inputs are joint angles plus the
camera's mug/target estimate, not raw images), which is what makes CPU training practical.
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "mug_pick_place"
RUNS = Path.home() / ".models" / "runs"   # outside the project: checkpoints are large

SETTINGS = {
    # 20 actions = 1 s at 20 Hz per prediction ("action chunking", ACT's key idea)
    "policy.chunk_size": 20,
    "policy.n_action_steps": 20,
    # Smaller than the default 512-wide model: 6-D state + 4-D task input don't need more
    "policy.dim_model": 256,
    "policy.dim_feedforward": 1024,
    "policy.n_heads": 4,
    "policy.n_encoder_layers": 3,
    "policy.n_vae_encoder_layers": 2,
    "policy.optimizer_lr": 1e-4,
    "policy.device": "cpu",
    "policy.push_to_hub": "false",
    "batch_size": 64,
    "num_workers": 0,
    "log_freq": 250,
    "wandb.enable": "false",
}


def _copy_not_link(checkpoint_dir: Path) -> Path:
    """LeRobot marks the newest checkpoint with a symlink `checkpoints/last`. Windows refuses
    symlinks without admin/developer mode, so keep a plain copy there instead."""
    last = checkpoint_dir.parent / "last"
    if last.is_symlink() or last.is_file():
        last.unlink()
    elif last.exists():
        shutil.rmtree(last)
    shutil.copytree(checkpoint_dir, last)
    return last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--name", default="act_mug")
    p.add_argument("--data", type=Path, default=DATA)
    args = p.parse_args()
    out = RUNS / args.name
    if out.exists():
        shutil.rmtree(out)  # LeRobot refuses to start in an existing run folder
    argv = ["lerobot-train", f"--dataset.repo_id=local/{args.data.name}", f"--dataset.root={args.data}",
            "--policy.type=act", f"--output_dir={out}", f"--job_name={args.name}",
            f"--steps={args.steps}", f"--save_freq={max(1000, args.steps // 4)}"]
    argv += [f"--{k}={v}" for k, v in SETTINGS.items()]
    print(" ".join(argv))

    import lerobot.common.train_utils as train_utils
    import lerobot.scripts.lerobot_train as lerobot_train

    if sys.platform == "win32":
        train_utils.update_last_checkpoint = lerobot_train.update_last_checkpoint = _copy_not_link
    sys.argv = argv
    lerobot_train.main()


if __name__ == "__main__":
    main()
