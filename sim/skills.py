"""Scripted skills built on robot.Sim: pick, place, and hand an object from one arm to the other.

Skills act on what the robot *believes* — never on simulator truth:
  - objects on the table: the last camera snapshot (`observe`)
  - an object in a gripper: the hand's own pose plus where the object sat in the fingers when
    grasped (`sim.held`), like a real robot using its joint encoders
Simulator truth is used only to *score* the result (`check`).
"""
from types import SimpleNamespace

import numpy as np

from robot import Sim, grip_for_gap, top_down_rotation
from scene import OBJECTS, RIM_BASE, RIM_WALL, rest_height

CARRY_Z = 0.06       # safe travel height for the gripper tip (see reach map in docs/setup.md)
APPROACH_UP = 0.035  # hover this far above a grasp before descending
MARGIN = 0.004       # gap between the fixed finger and the object before closing
TIP_CLEARANCE = 0.009  # lowest gripper-site height: finger tips stay ~2 mm above the table
# Optional fixed rule for which way to grip round objects: f(sim, arm, position) -> yaw.
# Normally the best of 12 directions is chosen, which is great for a scripted robot but looks
# random to a student policy (it can't see why), so demonstrations use a fixed rule.
CYLINDER_YAW = None


class SkillError(RuntimeError):
    pass


def _yaw_of(mat: np.ndarray) -> float:
    """Heading of an upright object's long (x) axis."""
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def _grip_yaw_of(site_mat: np.ndarray) -> float:
    """Heading of the gripper's finger-closing (z) axis — the `yaw` in top_down_rotation."""
    return float(np.arctan2(site_mat[1, 2], site_mat[0, 2]))


def grasp_width(name: str) -> float:
    """Width of the part of `name` that sits between the fingers."""
    shape, size, *_ = OBJECTS[name]
    return {"rim": lambda: RIM_WALL, "cylinder": lambda: 2 * size[0], "box": lambda: 2 * size[1]}[shape]()


def release_opening(name: str) -> float:
    """Open just ~6 mm wider than the object: the moving finger swings in an arc, and opening
    it further knocks neighbouring items (cutlery sits ~2 cm from the plate rim)."""
    return grip_for_gap(grasp_width(name) + 0.006)


def _nearest_other(sim: Sim, name: str, xy) -> float:
    """Distance from `xy` to the edge of the closest other object the robot knows about."""
    best = np.inf
    for other, seen in sim.world.items():
        if other != name and other in OBJECTS and not _in_hand(sim, other):
            shape, size, *_ = OBJECTS[other]
            radius = size[0] if shape != "box" else size[1]
            best = min(best, np.hypot(seen.x - xy[0], seen.y - xy[1]) - radius)
    return best


class TruthEyes:
    """Stand-in eyes that read the simulator directly. Only for comparing against the camera."""

    def __init__(self, sim: Sim):
        self.sim = sim

    def look(self, data):
        out = {}
        for name in OBJECTS:
            p, R = truth_pose(self.sim, name)
            out[name] = SimpleNamespace(x=p[0], y=p[1], yaw=_yaw_of(R))
        return out


def observe(sim: Sim) -> dict:
    """Take a fresh camera snapshot. Call with the arms at home so nothing is hidden."""
    sim.world = sim.eyes.look(sim.d)
    return sim.world


def _in_hand(sim: Sim, name: str) -> str | None:
    return next((arm for arm, h in sim.held.items() if h[0] == name), None)


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def locate(sim: Sim, name: str):
    """Where the robot believes `name` is: (position, rotation matrix)."""
    arm = _in_hand(sim, name)
    if arm:
        _, rel_p, rel_yaw = sim.held[arm]
        site_p, site_R = sim.arms[arm].site_pose(sim.d)
        return site_p + site_R @ rel_p, _yaw_matrix(_grip_yaw_of(site_R) + rel_yaw)
    if name not in sim.world:
        raise SkillError(f"can't see the {name}")
    s = sim.world[name]
    return np.array([s.x, s.y, rest_height(name)]), _yaw_matrix(s.yaw)


def _remember_grasp(sim: Sim, arm: str, name: str, obj_p, obj_R):
    """Store where the object sits in the fingers. The moving finger pushes it MARGIN towards
    the fixed finger while closing, so shift the pre-grasp estimate by that much."""
    site_p, site_R = sim.arms[arm].site_pose(sim.d)
    snapped = obj_p - MARGIN * site_R[:, 2]
    sim.held[arm] = (name, site_R.T @ (snapped - site_p), _yaw_of(obj_R) - _grip_yaw_of(site_R))


def truth_pose(sim: Sim, name: str):
    """Simulator ground truth — for scoring only."""
    b = sim.d.body(name)
    return b.xpos.copy(), b.xmat.reshape(3, 3).copy()


def grasp_target(sim: Sim, arm: str, name: str, along: float = 0.0):
    """Gripper-site position and yaw for a top-down grasp of `name`.
    `along` shifts the grasp point along a long object's axis (used for hand-offs)."""
    shape, size, *_ = OBJECTS[name]
    pos, mat = locate(sim, name)
    a = sim.arms[arm]
    centre = pos
    # Grip 16 mm below the object's top: deep enough for a firm hold. On the table the
    # fingertips can't go lower than TIP_CLEARANCE; in the air (hand-offs) they can, and the
    # deeper grip is what keeps the object from slipping when the giver lets go.
    if shape == "rim":
        # Fixed finger inside the plate, moving finger outside: `offset` puts the fixed
        # finger just inside the rim wall, along the closing direction.
        offset, grip_z = size[0] - RIM_WALL - MARGIN, pos[2] + RIM_BASE + 0.009
        yaws = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    elif shape == "cylinder":
        offset = -(size[0] + MARGIN)
        grip_z = max(TIP_CLEARANCE, pos[2] + size[1] - 0.016)
        yaws = np.linspace(-np.pi, np.pi, 12, endpoint=False)
        if CYLINDER_YAW is not None:
            yaws = [CYLINDER_YAW(sim, arm, pos)]
    else:  # long thin box: close across its width
        offset = -(size[1] + MARGIN)
        grip_z = max(TIP_CLEARANCE, pos[2] + size[2] - 0.016)
        centre = pos + along * mat[:, 0]
        base = _yaw_of(mat) + np.pi / 2
        yaws = [base, base + np.pi]
    best = None
    for yaw in yaws:
        z_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        target = np.array([*(centre[:2] + offset * z_axis[:2]), grip_z])
        q, err = a.ik(sim.d, target, yaw, q0=sim.d.ctrl[a.act])
        hover_q, hover_err = a.ik(sim.d, target + [0, 0, APPROACH_UP], yaw, q0=q)
        # Keep both fingers clear of neighbours: the fixed finger sits just behind `target`,
        # the moving one opens out past the object.
        # The open moving finger also rides high, so it can clip tall things like the bottle:
        # give it a wider berth.
        fingers = [(target[:2] - 0.01 * z_axis[:2], 0.01),
                   (target[:2] + (grasp_width(name) + MARGIN + 0.015) * z_axis[:2], 0.03)]
        crowded = sum(max(0.0, need - _nearest_other(sim, name, f)) for f, need in fingers)
        score = err + hover_err + 0.002 * abs(q[4]) + crowded  # prefer little wrist twist
        if best is None or score < best[0]:
            best = (score, target, float(yaw))
    return best[1], best[2]


def pick(sim: Sim, arm: str, name: str, along: float = 0.0, lift_to: float | None = CARRY_Z,
         margin: float | None = None):
    """Top-down grasp. lift_to=None grips without lifting (hand-offs: the other arm still holds it).
    margin: gap left between the fixed finger and the object before closing (default MARGIN)."""
    global MARGIN
    saved, MARGIN = MARGIN, (MARGIN if margin is None else margin)
    try:
        _pick(sim, arm, name, along, lift_to)
    finally:
        MARGIN = saved


def _pick(sim: Sim, arm: str, name: str, along: float, lift_to: float | None):
    target, yaw = grasp_target(sim, arm, name, along)
    obj_p, obj_R = locate(sim, name)
    sim.log.append(f"{arm}: pick {name}")
    # Open only as wide as needed: a wide-open finger swings out and sweeps into neighbours
    # (or the other arm, during a hand-off).
    sim.grip(arm, closed=False, seconds=0.3, opening=grip_for_gap(grasp_width(name) + MARGIN + 0.012))
    sim.move({arm: (target + [0, 0, APPROACH_UP], yaw)}, duration=1.2)
    sim.move({arm: (target, yaw)}, duration=0.6)
    sim.grip(arm, closed=True)
    if not sim.arms[arm].holding(sim.d, sim.m.body(name).id):
        raise SkillError(f"{arm} failed to grasp {name}")
    _remember_grasp(sim, arm, name, obj_p, obj_R)
    if lift_to is not None:
        sim.move({arm: (np.array([*target[:2], lift_to]), yaw)}, duration=0.6)
        if not sim.arms[arm].holding(sim.d, sim.m.body(name).id):
            raise SkillError(f"{arm} dropped {name} while lifting")


def _site_target_for(sim: Sim, arm: str, name: str, xy, obj_yaw, obj_z):
    """Where the gripper must go so the held object ends at (xy, obj_z) with yaw `obj_yaw`
    (None = keep the current gripper yaw). Uses the object's current pose relative to the
    gripper, so any slip during the grasp is absorbed."""
    held_name, rel_p, rel_yaw = sim.held[arm]
    assert held_name == name, f"{arm} holds {held_name}, not {name}"
    site_yaw = _grip_yaw_of(sim.arms[arm].site_pose(sim.d)[1])
    grip_yaw = site_yaw if obj_yaw is None else obj_yaw - rel_yaw
    R_goal = top_down_rotation(grip_yaw)
    site_goal = np.array([*xy, obj_z]) - R_goal @ rel_p
    return site_goal, grip_yaw


def place(sim: Sim, arm: str, name: str, xy, obj_yaw=None):
    rest_z = rest_height(name) + 0.002
    if obj_yaw is None:
        site_goal, yaw = _site_target_for(sim, arm, name, xy, None, rest_z)
    else:
        # Cutlery is symmetric end-to-end, so yaw and yaw+180° look the same. The wrist can't
        # turn a full circle, and one choice puts the moving finger on the plate's side;
        # take the one the arm can reach with the moving finger opening into free space.
        a, options = sim.arms[arm], []
        for oy in (obj_yaw, obj_yaw + np.pi):
            goal, gy = _site_target_for(sim, arm, name, xy, oy, rest_z)
            _, err = a.ik(sim.d, goal, gy, q0=sim.d.ctrl[a.act])
            finger_xy = goal[:2] + 0.03 * np.array([np.cos(gy), np.sin(gy)])
            crowded = max(0.0, 0.015 - _nearest_other(sim, name, finger_xy))
            options.append((err + 0.05 * a.last_rot_err + crowded, goal, gy))
        _, site_goal, yaw = min(options, key=lambda o: o[0])
    # If the object sits high in the fingers (taken in a hand-off), the fingertips would reach
    # the table before it does. Stop at fingertip height and let it drop the last few mm.
    site_goal[2] = max(site_goal[2], TIP_CLEARANCE)
    sim.log.append(f"{arm}: place {name} at {np.round(xy, 3).tolist()}")
    above = site_goal + [0, 0, max(0.0, CARRY_Z - site_goal[2])]
    sim.move({arm: (above, yaw)}, duration=1.4)
    sim.move({arm: (site_goal, yaw)}, duration=0.6)
    sim.grip(arm, closed=False, seconds=0.4, opening=release_opening(name))
    sim.held.pop(arm, None)
    sim.move({arm: (site_goal + [0, 0, APPROACH_UP], yaw)}, duration=0.5)


HANDOFF_POINT = (0.0, -0.11, 0.05)  # clear air in front of the place setting, in reach of both arms
HANDOFF_MARGIN = 0.0015


def _near_end(sim: Sim, arm: str, name: str, end: float) -> float:
    """`along` value (+end or -end) for the end of a long object nearest this arm's base —
    unless that end is too close to the base to reach (the arm can't fold that far while
    pointing down), in which case the far end. Works whichever way the object's axis points."""
    pos, R = locate(sim, name)
    a = sim.arms[arm]
    ends = sorted((end, -end), key=lambda e: np.hypot(*(pos[:2] + e * R[:2, 0] - a.base_xy)))
    for e in ends:
        target, yaw = grasp_target(sim, arm, name, e)
        _, err = a.ik(sim.d, target, yaw, q0=sim.d.ctrl[a.act])
        if err < 0.003 and a.last_rot_err < 0.05:
            return e
    return ends[0]


def handoff(sim: Sim, giver: str, taker: str, name: str, point=HANDOFF_POINT):
    """Giver holds one end of a long object at `point`; taker grabs the other end; giver lets go."""
    shape, size, *_ = OBJECTS[name]
    if shape != "box":
        raise SkillError("hand-off is only defined for long objects")
    sim.log.append(f"{giver} -> {taker}: hand off {name}")
    end = 0.6 * size[0]
    pick(sim, giver, name, along=_near_end(sim, giver, name, end))
    # Carry so the object lies along x, centred on the hand-off point, with the giver's
    # fingers on the giver's side (either 0° or 180° does that — try both).
    side = np.sign(sim.arms[giver].base_xy[0])
    options = [_site_target_for(sim, giver, name, point[:2], oy, point[2]) for oy in (0.0, np.pi)]
    site_goal, yaw = max(options, key=lambda o: side * o[0][0])
    # Both arms move at once: while the giver carries the object in, the taker comes to wait
    # above its own end of the hand-off point.
    wait = np.array([point[0] - side * (end + 0.01), point[1], point[2] + APPROACH_UP])
    sim.move({giver: (site_goal, yaw), taker: (wait, np.pi / 2)}, duration=1.5)
    # The giver still holds the object while the taker closes, so the taker's closing finger
    # would push it sideways and twist it in the giver's fingers. Line up only 1.5 mm away
    # (not MARGIN) so there's barely any push. (Letting the giver open at the same moment
    # was tried: the object dropped.)
    pick(sim, taker, name, along=_near_end(sim, taker, name, end), lift_to=None, margin=HANDOFF_MARGIN)
    sim.grip(giver, closed=False, seconds=0.4)
    sim.held.pop(giver, None)
    # Giver backs away upward first, so its open fingers clear the object.
    g_pos, _ = sim.arms[giver].site_pose(sim.d)
    sim.move({giver: (g_pos + [0, 0, 0.03], yaw)}, duration=0.4)
    sim.move({giver: (np.array([np.sign(g_pos[0]) * 0.13, g_pos[1], CARRY_Z]), yaw)}, duration=0.6)
    if not sim.arms[taker].holding(sim.d, sim.m.body(name).id):
        raise SkillError(f"{taker} dropped {name} during the hand-off")


def home(sim: Sim, arms=("A", "B"), duration=1.0):
    """Back to the start pose. Lift straight up first so nothing on the table gets dragged."""
    lifts = {}
    for arm in arms:
        p, R = sim.arms[arm].site_pose(sim.d)
        if p[2] < CARRY_Z - 0.005:
            lifts[arm] = (np.array([p[0], p[1], CARRY_Z]), _grip_yaw_of(R))
    if lifts:
        sim.move(lifts, duration=0.4)
    sim.move_joints({arm: sim.home_q[arm] for arm in arms}, duration=duration)


def check(sim: Sim, spots: dict, tol=0.025) -> dict:
    """Which items ended on their spot (within `tol` metres), resting on the table, upright,
    and — for cutlery — lying the right way (along y, like a real place setting)."""
    return {name: why_not(sim, name, xy, tol) == "" for name, xy in spots.items()}


CUTLERY_YAW_TOL = np.radians(20)


def why_not(sim: Sim, name: str, xy, tol=0.025) -> str:
    """Empty if `name` sits correctly on its spot; otherwise a short reason."""
    p, R = truth_pose(sim, name)
    reasons = []
    dist = np.hypot(p[0] - xy[0], p[1] - xy[1])
    if dist >= tol:
        reasons.append(f"{dist * 100:.1f}cm off")
    if abs(p[2] - rest_height(name)) >= 0.01:
        reasons.append(f"height {p[2] * 100:.1f}cm")
    if R[2, 2] <= 0.9:
        reasons.append("tipped over")
    elif OBJECTS[name][0] == "box":
        # Symmetric end-to-end: 90° and -90° both count.
        off = abs(_yaw_of(R) % np.pi - np.pi / 2)
        if off > CUTLERY_YAW_TOL:
            reasons.append(f"turned {np.degrees(off):.0f}°")
    return ", ".join(reasons)
