"""Convert the trained ACT policy to OpenVINO: FP32, FP16 and INT8 (NNCF post-training
quantisation, calibrated on the recorded demonstrations).

    python bench/export_act.py            # writes <run>/openvino/act_{fp32,fp16,int8}.xml

Normalisation stays in LeRobot's own pre/post-processors; only the network itself — the part
that costs time — runs in OpenVINO.
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import openvino as ov
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "sim"), str(ROOT / "brain")]
RUN = Path.home() / ".models" / "runs" / "act_mug"


class ACTCore(torch.nn.Module):
    """Normalised (state, task) -> normalised action chunk. The VAE branch is unused at run time."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state, env):
        return self.model({"observation.state": state, "observation.environment_state": env})[0]


def ckpt_of(run: Path) -> Path:
    return run / "checkpoints" / "last" / "pretrained_model"


def training_data(run: Path) -> Path:
    """The dataset a run was trained on (from LeRobot's saved training config)."""
    import json

    cfg = json.loads((ckpt_of(run) / "train_config.json").read_text())
    return Path(cfg["dataset"]["root"])


def calibration_batches(pre, data: Path, n=300):
    """Pre-processed (state, env) pairs from the recorded demonstrations."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import record_demos as R

    ds = LeRobotDataset(f"local/{data.name}", root=data)
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    out = []
    for i in idx:
        f = ds[int(i)]
        b = pre({"observation.state": f["observation.state"],
                 "observation.environment_state": f["observation.environment_state"], "task": R.TASK})
        out.append((b["observation.state"].numpy(), b["observation.environment_state"].numpy()))
    return out


def export(run: Path):
    import nncf
    import record_demos as R
    from eval_policy import load_torch

    policy, pre, _ = load_torch(ckpt_of(run))
    core = ACTCore(policy.model).eval()
    from eval_policy import io_dims

    state, env = torch.zeros(1, 6), torch.zeros(1, io_dims(policy)[0])
    out = run / "openvino"
    out.mkdir(exist_ok=True)
    ov_model = ov.convert_model(core, example_input=(state, env))
    ov.save_model(ov_model, out / "act_fp32.xml", compress_to_fp16=False)
    ov.save_model(ov_model, out / "act_fp16.xml", compress_to_fp16=True)
    calib = calibration_batches(pre, training_data(run))
    # Full INT8 (weights + activations). NNCF's default preset drifted least on this model
    # (0.067 normalised units; MIXED 0.10, TRANSFORMER/SmoothQuant 0.16-0.29 — measured Day 5).
    int8 = nncf.quantize(ov_model, nncf.Dataset(calib, lambda b: {0: b[0], 1: b[1]}),
                         subset_size=len(calib))
    ov.save_model(int8, out / "act_int8.xml", compress_to_fp16=False)
    # INT8 weights only: stored in 8 bits, computed in float — near-lossless.
    int8w = nncf.compress_weights(ov.convert_model(core, example_input=(state, env)))
    ov.save_model(int8w, out / "act_int8w.xml", compress_to_fp16=False)
    # Accuracy check: how far do the OpenVINO chunks drift from PyTorch's?
    core_ = ov.Core()
    with torch.no_grad():
        ref = [core(torch.from_numpy(s), torch.from_numpy(e)).numpy() for s, e in calib[:50]]
    for name in ("fp32", "fp16", "int8", "int8w"):
        c = core_.compile_model(out / f"act_{name}.xml", "CPU")
        diff = np.mean([np.abs(c([s, e])[0] - r).mean() for (s, e), r in zip(calib[:50], ref)])
        print(f"{name}: mean |OpenVINO - PyTorch| = {diff:.5f} (normalised action units)")
    print(f"saved to {out}")


class OpenVINORunner:
    """Drop-in replacement for eval_policy.TorchRunner, network in OpenVINO."""

    def __init__(self, ckpt: Path, device: str = "CPU", precision: str = "fp16"):
        from eval_policy import load_torch
        import record_demos as R

        from eval_policy import env_names, io_dims

        policy, self.pre, self.post = load_torch(ckpt)
        self.env_dim, self.act_dim = io_dims(policy)
        self.env_names = env_names(ckpt)
        self.n = policy.config.n_action_steps
        self.task = R.TASK
        xml = ckpt.parent.parent.parent / "openvino" / f"act_{precision}.xml"
        self.net = ov.Core().compile_model(xml, device)
        self.queue = deque()

    def reset(self):
        self.queue.clear()

    def act(self, state, env):
        if not self.queue:
            b = self.pre({"observation.state": torch.from_numpy(state),
                          "observation.environment_state": torch.from_numpy(env), "task": self.task})
            chunk = self.net([b["observation.state"].numpy(), b["observation.environment_state"].numpy()])[0]
            for a in torch.from_numpy(chunk[0, : self.n]):
                self.queue.append(self.post(a[None]).squeeze(0).numpy())
        return self.queue.popleft()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=RUN)
    export(p.parse_args().run)
