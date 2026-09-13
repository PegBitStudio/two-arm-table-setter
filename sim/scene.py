"""Builds the dinner-table scene: two SO-101 arms facing each other across a table, plus
tableware. Every random choice comes from one seed, so any run can be replayed exactly.

    from scene import build
    model, info = build(seed=3)
"""
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ARM_DIR = Path(__file__).parent.resolve() / "assets" / "so101"
ARM_XML = ARM_DIR / "so101_new_calib.xml"
MESH_DIR = ARM_DIR / "assets"

BASE_X = 0.24  # arms sit at x = -BASE_X (arm A) and +BASE_X (arm B), facing each other
TABLE_HALF = (0.42, 0.30, 0.02)

# Finger pads replace the jaw meshes for contact. The mesh collision shapes are convex hulls
# that fill the gap between the jaws, which makes real grasps impossible. Each pad is a 1 cm
# thick box on the inner face of a fingertip, measured from the jaw mesh vertices with the arm
# at zero pose: body name -> (pos, half-size) in that body's frame.
PADS = {
    "gripper": ([-0.0124, -0.0004, -0.0833], [0.0046, 0.0082, 0.0215]),
    "moving_jaw_so101_v1": ([-0.0076, -0.0589, 0.018], [0.0046, 0.0205, 0.0082]),
}
HULLS_OFF = {"wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"}

# Tableware, toy-scale to fit the SO-101 gripper (max ~5.4 cm between the pads).
# name: (shape, size, rgba, mass kg, nominal start xy)
OBJECTS = {
    # The plate is gripped by its rim: the jaw swings in an arc, so it cannot pinch anything
    # wide and flat near the table. size = [outer radius, total height].
    "plate": ("rim", [0.045, 0.017], [0.95, 0.95, 0.92, 1], 0.030, (-0.10, 0.13)),
    "mug": ("cylinder", [0.018, 0.028], [0.80, 0.20, 0.18, 1], 0.030, (0.10, 0.13)),
    # Fork starts on arm B's side but belongs left of the plate (arm A's side), and the spoon
    # the other way round, so both need a hand-off between the arms.
    # Chunky 12 mm tall handles: the fingertips stop ~2 mm above the table, so thinner cutlery
    # only gets a sliver of contact and slips out on lift. Wider than tall (16-18 mm), or the
    # releasing finger rolls them onto their side.
    "fork": ("box", [0.055, 0.008, 0.006], [0.70, 0.72, 0.75, 1], 0.015, (0.12, -0.13)),
    "spoon": ("box", [0.050, 0.009, 0.006], [0.75, 0.70, 0.55, 1], 0.015, (-0.12, -0.13)),
    "bottle": ("cylinder", [0.015, 0.045], [0.20, 0.45, 0.85, 1], 0.040, (0.03, 0.19)),
}

# Where each item belongs in the finished place setting (xy on the table).
SPOTS = {
    "plate": (0.0, -0.02),
    "mug": (0.065, 0.075),
    "fork": (-0.075, -0.02),
    "spoon": (0.075, -0.02),
}
RIM_WALL = 0.004   # plate rim thickness
RIM_BASE = 0.003   # plate floor thickness
RIM_SEGMENTS = 20

TABLE_COLOURS = [(0.55, 0.38, 0.24), (0.35, 0.36, 0.40), (0.72, 0.66, 0.55), (0.20, 0.28, 0.22)]


SIZE_RANGE = {"plate": (0.95, 1.05), "mug": (0.9, 1.1), "fork": (0.9, 1.1),
              "spoon": (0.9, 1.1), "bottle": (0.9, 1.1)}


@dataclass
class SceneInfo:
    seed: int
    start: dict = field(default_factory=dict)  # object -> (x, y, yaw)
    scale: dict = field(default_factory=dict)  # object -> size factor (1.0 unless hard)
    mass_scale: float = 1.0
    friction: float = 1.0
    light: float = 1.0


def rest_height(name: str) -> float:
    """Height of the body origin when the object sits on the table."""
    shape, size, *_ = OBJECTS[name]
    return {"cylinder": lambda: size[1], "box": lambda: size[2], "rim": lambda: 0.0}[shape]()


def _add_rimmed_plate(body, name, size, mass, common) -> None:
    """Flat base plus a ring of short wall segments. Body origin is at the base's underside."""
    radius, height = size
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, RIM_BASE / 2, 0],
                  pos=[0, 0, RIM_BASE / 2], mass=mass * 0.6, **common)
    r_mid = radius - RIM_WALL / 2
    seg_half = np.pi * r_mid / RIM_SEGMENTS * 1.05
    for i in range(RIM_SEGMENTS):
        a = 2 * np.pi * i / RIM_SEGMENTS
        body.add_geom(name=f"{name}_rim{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[RIM_WALL / 2, seg_half, (height - RIM_BASE) / 2],
                      pos=[r_mid * np.cos(a), r_mid * np.sin(a), RIM_BASE + (height - RIM_BASE) / 2],
                      quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)],
                      mass=mass * 0.4 / RIM_SEGMENTS, **common)


def _add_arm(spec: mujoco.MjSpec, prefix: str, x: float, facing_quat) -> None:
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    arm.meshdir = str(MESH_DIR)
    for g in arm.geoms:
        if g.meshname in HULLS_OFF:
            g.contype = g.conaffinity = 0
    for pad_name, (body_name, (pos, size)) in zip(("pad_fixed", "pad_moving"), PADS.items()):
        arm.body(body_name).add_geom(
            name=pad_name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
            friction=[1.5, 0.02, 0.002], condim=4, group=3, rgba=[0.2, 0.9, 0.2, 0.5],
            solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],  # stiff: objects can't sink in
        )
    # The stock gripper squeezes with ~60 N at the fingertip, enough to push light objects
    # through the pads. ~20 N holds everything here with a wide margin. Less joint damping
    # keeps the weaker motor closing in well under a second.
    arm.actuator("gripper").forcerange = [-1.0, 1.0]
    arm.joint("gripper").damping = [0.15, 0, 0]
    frame = spec.worldbody.add_frame(pos=[x, 0, 0], quat=facing_quat)
    spec.attach(arm, prefix=prefix, frame=frame)


def build(seed: int = 0, hard: bool = False) -> tuple[mujoco.MjModel, SceneInfo]:
    """hard=True: starts vary ±4 cm (not ±2), object sizes vary, and so does the floor colour."""
    rng = np.random.default_rng(seed)
    info = SceneInfo(
        seed=seed,
        mass_scale=float(rng.uniform(0.7, 1.4)),
        friction=float(rng.uniform(0.8, 1.2)),
        light=float(rng.uniform(0.5, 1.0)),
    )

    spec = mujoco.MjSpec()
    spec.meshdir = str(MESH_DIR)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 960, 720
    # Lighting is kept below the point where colours wash out to white in the overhead camera
    # (light and camera both straight above make flat tops glare), and specular is low.
    spec.visual.headlight.ambient = [0.2, 0.2, 0.2]
    spec.visual.headlight.diffuse = [0.25 * info.light] * 3
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

    world = spec.worldbody
    L = world.add_light(pos=[rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 1.2], dir=[0, 0, -1])
    L.diffuse = [0.6 * info.light] * 3
    L.specular = [0.05, 0.05, 0.05]
    floor = rng.uniform(0.05, 0.45, 3) if hard else (0.15, 0.17, 0.2)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05],
                   pos=[0, 0, -0.75], rgba=[*floor, 1])
    colour = TABLE_COLOURS[rng.integers(len(TABLE_COLOURS))]
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=TABLE_HALF,
                   pos=[0, 0, -TABLE_HALF[2]], rgba=[*colour, 1], friction=[1.0, 0.01, 0.001])

    _add_arm(spec, "a_", -BASE_X, [1, 0, 0, 0])
    _add_arm(spec, "b_", BASE_X, [0, 0, 0, 1])  # rotated 180° about z

    # Place-setting markers: flat, visual only.
    for name, (sx, sy) in SPOTS.items():
        world.add_geom(name=f"spot_{name}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[0.022, 0.0005, 0], pos=[sx, sy, 0.0005],
                       contype=0, conaffinity=0, rgba=[1, 1, 1, 0.25])

    jitter = 0.04 if hard else 0.02
    placed = []  # (x, y, footprint radius) — hard mode rejects overlapping starts
    for name, (shape, size, rgba, mass, (nx, ny)) in OBJECTS.items():
        # Hard mode: sizes vary (the robot still works from the standard sizes, so this tests
        # coping with objects that don't match its model). The plate varies least: its rim
        # grip is the most size-sensitive.
        scale = float(rng.uniform(*SIZE_RANGE[name])) if hard else 1.0
        size = [s * scale for s in size]
        info.scale[name] = scale
        footprint = size[0]  # radius, or half-length for cutlery
        for _ in range(50):
            x, y = nx + rng.uniform(-jitter, jitter), ny + rng.uniform(-jitter, jitter)
            if all(np.hypot(x - px, y - py) > footprint + pr + 0.01 for px, py, pr in placed):
                break
        placed.append((x, y, footprint))
        yaw = rng.uniform(-0.5, 0.5) if shape == "box" else 0.0
        info.start[name] = (x, y, yaw)
        body = world.add_body(name=name, pos=[x, y, rest_height(name) * scale + 0.001],
                              quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        body.add_freejoint(name=f"{name}_free")
        common = dict(rgba=rgba, condim=4, friction=[1.2 * info.friction, 0.02, 0.002])
        if shape == "rim":
            _add_rimmed_plate(body, name, size, mass * info.mass_scale, common)
            continue
        geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER if shape == "cylinder" else mujoco.mjtGeom.mjGEOM_BOX
        body.add_geom(name=name, type=geom_type, size=size + [0] * (3 - len(size)),
                      mass=mass * info.mass_scale, **common)

    world.add_camera(name="top", pos=[0, 0, 0.85], quat=[1, 0, 0, 0], fovy=45)
    world.add_camera(name="front", pos=[0, -0.62, 0.42], xyaxes=[1, 0, 0, 0, 0.55, 0.83], fovy=50)

    return spec.compile(), info
