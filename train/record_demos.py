"""Record demonstrations automatically: the scripted skills are the teacher.

For each episode: a random table (seed), the mug moved to a random start, a random target
spot. Arm B picks the mug up and puts it on the target with the Day 2–3 skills while we sample
(joint angles, mug + target position relative to the fingertips) -> (motor commands) 20 times
a second.
Episodes where the mug does NOT end on the target (checked against simulator truth) are thrown
away, so the dataset only contains successes.

    python train/record_demos.py --episodes 150          # -> train/data/mug_pick_place
    python train/record_demos.py --episodes 250 --mode ee --out train/data/mug_pick_place_ee
    python train/record_demos.py --episodes 250 --mode ee --out train/data/mug_pick_place_ee --resume

Output is a standard LeRobot dataset, so LeRobot's own training scripts read it directly.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "brain")]
import day2_demo  # noqa: E402
import skills  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "mug_pick_place"
FPS = 20
ARM = "B"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TASK = "pick up the mug and put it on the target spot"

# Where the mug may start and where it may be sent (arm B's side of the table, clear of the
# plate and the bottle's usual area). The target is part of the policy's input.
START_BOX = ((0.06, 0.15), (0.06, 0.16))
TARGET_BOX = ((0.04, 0.12), (-0.07, 0.05))

# Task input, recomputed every tick: where the mug (as the camera saw it at the start) and the
# target are *relative to the fingertips*, plus fingertip height. The fingertip position comes
# from the arm's own joint angles (forward kinematics), so nothing here is simulator truth.
# Relative inputs are what made this learnable: v1 used absolute table positions, and 200
# episodes weren't enough to learn "table position -> exact joint angles" (5% success).
ENV_NAMES = ["mug_dx", "mug_dy", "target_dx", "target_dy", "tip_z"]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"b_{j}" for j in JOINTS]},
    "observation.environment_state": {"dtype": "float32", "shape": (len(ENV_NAMES),), "names": ENV_NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": [f"b_{j}" for j in JOINTS]},
}
# The teacher leaves 8 mm (not 4) between the fixed finger and the mug before closing, so a
# student that is a few mm off still grasps cleanly.
TEACHER_MARGIN = 0.008


def radial_yaw(sim, arm, pos):
    """Teacher's grip rule for the mug: fingers close along the line from the arm's base to
    the mug (fixed finger on the near side). One smooth rule, so the student can copy it —
    choosing the best of 12 directions made the teacher look random (up to 175° apart)."""
    d = np.asarray(pos[:2]) - sim.arms[arm].base_xy
    return float(np.arctan2(d[1], d[0]))


def task_input(sim, mug_xy, target) -> np.ndarray:
    tip = sim.arms[ARM].site_pose(sim.d)[0]
    return np.array([mug_xy[0] - tip[0], mug_xy[1] - tip[1],
                     target[0] - tip[0], target[1] - tip[1], tip[2]], np.float32)


def arm_state(sim):
    a = sim.arms[ARM]
    grip = sim.m.joint(a.prefix + "gripper").qposadr[0]
    return np.append(sim.d.qpos[a.qadr], sim.d.qpos[grip]).astype(np.float32)


def arm_command(sim):
    a = sim.arms[ARM]
    return np.append(sim.d.ctrl[a.act], sim.d.ctrl[a.grip_act]).astype(np.float32)


# --- "ee" action mode: the policy outputs small fingertip moves instead of joint angles ---
# The arm's own IK turns each move into joint targets (as a real robot's controller would).
# Small relative moves are far easier to learn precisely than absolute joint angles.
# Hand move (metres) + hand rotation (rotation vector, radians, world frame) + gripper.
# The full rotation matters: near the top of its reach the wrist can't point straight down,
# and a yaw-only action lost that tilt (replaying the teacher then drifted by centimetres).
EE_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]


# Moves are measured from where the hand was last *commanded* to be, not where it actually
# is: the hand lags its commands, and anchoring on the lagging position made small errors
# snowball (a replay of the teacher's own moves drifted by centimetres).


def commanded_pose(sim):
    """Fingertip pose the current joint commands lead to (forward kinematics)."""
    import mujoco

    a = sim.arms[ARM]
    if not hasattr(sim, "_fk"):
        sim._fk = mujoco.MjData(sim.m)  # scratch copy for forward kinematics
    fk = sim._fk
    fk.qpos[:] = sim.d.qpos
    fk.qpos[a.qadr] = sim.d.ctrl[a.act]
    mujoco.mj_kinematics(sim.m, fk)
    return fk.site_xpos[a.site].copy(), fk.site_xmat[a.site].reshape(3, 3).copy()


def ee_command(sim) -> np.ndarray:
    """Teacher's command as a hand move since the previous tick."""
    from scipy.spatial.transform import Rotation

    tip, R = commanded_pose(sim)
    last_tip, last_R = getattr(sim, "_last_cmd", (tip, R))
    sim._last_cmd = (tip, R)
    dr = Rotation.from_matrix(R @ last_R.T).as_rotvec()
    return np.array([*(tip - last_tip), *dr, sim.d.ctrl[sim.arms[ARM].grip_act]], np.float32)


def apply_ee(sim, action) -> None:
    from scipy.spatial.transform import Rotation

    a = sim.arms[ARM]
    tip, R = commanded_pose(sim)
    R_goal = Rotation.from_rotvec(action[3:6]).as_matrix() @ R
    q, _ = a.ik(sim.d, tip + action[:3], 0.0, q0=sim.d.ctrl[a.act], R_goal=R_goal)
    sim.d.ctrl[a.act] = q
    sim.d.ctrl[a.grip_act] = action[6]


# --- "abs" action mode: the policy outputs where the hand should be, not a small move ---
# v3/v4 (small moves) copied each move to ~0.3 mm, but 160 moves add up: the hand drifted
# centimetres into places the demonstrations never visited. An absolute hand target can't
# drift, and "grip here" is close to "mug position + a fixed offset" — easy to learn.
# Orientation is the first two columns of the rotation matrix (continuous, unlike angles).
ABS_NAMES = ["x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32", "gripper"]
ABS_ENV_NAMES = ["mug_x", "mug_y", "target_x", "target_y", "tip_x", "tip_y", "tip_z"]


def task_input_abs(sim, mug_xy, target) -> np.ndarray:
    tip = sim.arms[ARM].site_pose(sim.d)[0]
    return np.array([*mug_xy, *target, *tip], np.float32)


def abs_command(sim) -> np.ndarray:
    tip, R = commanded_pose(sim)
    return np.array([*tip, *R[:, 0], *R[:, 1], sim.d.ctrl[sim.arms[ARM].grip_act]], np.float32)


def rotation_from_6d(v) -> np.ndarray:
    """Two (roughly) orthogonal columns -> a proper rotation matrix (Gram-Schmidt)."""
    a = np.asarray(v[:3], float)
    a /= np.linalg.norm(a)
    b = np.asarray(v[3:6], float)
    b -= a * (a @ b)
    b /= np.linalg.norm(b)
    return np.column_stack([a, b, np.cross(a, b)])


def apply_abs(sim, action) -> None:
    a = sim.arms[ARM]
    q, _ = a.ik(sim.d, np.asarray(action[:3], float), 0.0, q0=sim.d.ctrl[a.act],
                R_goal=rotation_from_6d(action[3:9]))
    sim.d.ctrl[a.act] = q
    sim.d.ctrl[a.grip_act] = action[9]


# --- "mug" mode: the same hand targets, but measured from the mug's starting position ---
# In "abs" mode the student leaned towards where mugs usually are (~1.5 cm off on new tables).
# Measured from the mug, picking is "go just beside the mug" (a small, nearly fixed offset)
# and placing is "move by (target - mug)", which is given as an input. Nothing left to guess.
MUG_ENV_NAMES = ["target_dx", "target_dy", "tip_dx", "tip_dy", "tip_z", "mug_from_base_x", "mug_from_base_y"]


def task_input_mug(sim, mug_xy, target) -> np.ndarray:
    tip = sim.arms[ARM].site_pose(sim.d)[0]
    base = sim.arms[ARM].base_xy
    return np.array([target[0] - mug_xy[0], target[1] - mug_xy[1], tip[0] - mug_xy[0], tip[1] - mug_xy[1],
                     tip[2], mug_xy[0] - base[0], mug_xy[1] - base[1]], np.float32)


def apply_mug(sim, action, mug_xy) -> None:
    a = np.array(action, float)
    a[0] += mug_xy[0]
    a[1] += mug_xy[1]
    apply_abs(sim, a)


def features(mode: str) -> dict:
    f = dict(FEATURES)
    if mode == "mug":
        f["action"] = {"dtype": "float32", "shape": (len(ABS_NAMES),), "names": ABS_NAMES}
        f["observation.environment_state"] = {"dtype": "float32", "shape": (len(MUG_ENV_NAMES),),
                                              "names": MUG_ENV_NAMES}
    if mode == "ee":
        f["action"] = {"dtype": "float32", "shape": (len(EE_NAMES),), "names": EE_NAMES}
    elif mode == "abs":
        f["action"] = {"dtype": "float32", "shape": (len(ABS_NAMES),), "names": ABS_NAMES}
        f["observation.environment_state"] = {"dtype": "float32", "shape": (len(ABS_ENV_NAMES),),
                                              "names": ABS_ENV_NAMES}
    return f


def new_episode(seed: int):
    """Random table + random mug start + random target. Returns (sim, target)."""
    rng = np.random.default_rng(10_000 + seed)
    sim = day2_demo.start_sim(seed, video=False, eyes="camera")
    start = np.array([rng.uniform(*START_BOX[0]), rng.uniform(*START_BOX[1])])
    target = np.array([rng.uniform(*TARGET_BOX[0]), rng.uniform(*TARGET_BOX[1])])
    # Move the mug to its start, keeping it clear of the bottle.
    bottle = sim.d.body("bottle").xpos[:2]
    if np.hypot(*(start - bottle)) < 0.05:
        start = bottle + 0.05 * (start - bottle) / max(np.hypot(*(start - bottle)), 1e-6)
    adr = sim.m.joint("mug_free").qposadr[0]
    sim.d.qpos[adr:adr + 3] = [*start, skills.rest_height("mug") + 0.001]
    sim.d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    sim.step(0.3)
    return sim, target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--out", type=Path, default=DATA)
    p.add_argument("--mode", choices=["joints", "ee", "abs"], default="abs",
                   help="action = joint targets, fingertip moves (ee), or hand targets (abs)")
    p.add_argument("--resume", action="store_true",
                   help="add episodes to an existing dataset (seeds continue after its last one)")
    p.add_argument("--seed-start", type=int, default=0,
                   help="first table seed; give parallel recorders far-apart ranges")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    skills.MARGIN = TEACHER_MARGIN
    skills.CYLINDER_YAW = radial_yaw
    command = {"ee": ee_command, "abs": abs_command, "joints": arm_command}[args.mode]
    env_input = task_input_abs if args.mode == "abs" else task_input
    repo = f"local/{args.out.name}"
    kept = tried = 0
    if args.resume and args.out.exists():
        # Windows' graphics driver stops creating new (hidden) render windows after ~300 in
        # one process, so long recordings are done in several runs.
        ds = LeRobotDataset.resume(repo, root=args.out)
        kept = ds.num_episodes
        tried = int(kept * 1.1) + 20  # fresh seeds, clear of the ones already used
        args.episodes += kept
        print(f"resuming: {kept} episodes already recorded")
    else:
        if args.out.exists():
            shutil.rmtree(args.out)
        ds = LeRobotDataset.create(repo_id=repo, fps=FPS, features=features(args.mode),
                                   root=args.out, robot_type="so101_bimanual_sim", use_videos=False)
    t0 = time.time()
    while kept < args.episodes:
        seed = args.seed_start + tried
        tried += 1
        sim, target = new_episode(seed)
        try:
            skills.observe(sim)
            if "mug" not in sim.world:
                continue
            mug_xy = (sim.world["mug"].x, sim.world["mug"].y)
            frames = []
            sim.on_tick = lambda s: frames.append({
                "observation.state": arm_state(s),
                "observation.environment_state": env_input(s, mug_xy, target),
                "action": command(s),
                "task": TASK,
            })
            sim._next_tick = sim.d.time
            try:
                skills.pick(sim, ARM, "mug")
                skills.place(sim, ARM, "mug", target)
                skills.home(sim, [ARM])
            except skills.SkillError:
                continue
            sim.on_tick = None
            if skills.why_not(sim, "mug", target, tol=0.015):
                continue  # teacher missed: don't learn from it
            for f in frames:
                ds.add_frame(f)
            ds.save_episode()
            kept += 1
            if kept % 10 == 0:
                print(f"{kept}/{args.episodes} episodes kept ({tried} tried, {time.time() - t0:.0f}s)")
        finally:
            day2_demo.close_sim(sim)
    ds.finalize()
    print(f"done: {kept} episodes, {ds.num_frames} frames, kept {kept}/{tried} "
          f"({100 * kept / tried:.0f}%) -> {args.out}")


if __name__ == "__main__":
    main()
