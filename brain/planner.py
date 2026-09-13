"""The planner: turns goals ("fork -> left of the plate") into arm actions, runs them, and
checks with the camera after every step — retrying from wherever things actually ended up.

Observe -> Plan -> Act -> Observe again. The language model (language.py) only decides the
*goals*; which arm does what, and when a hand-off is needed, is decided here from what the
camera sees and what each arm can reach.
"""
from dataclasses import dataclass, field

import numpy as np

import __init__  # noqa: F401  (adds sim/ to the path)
import skills
from robot import Sim
from scene import OBJECTS, SPOTS

# Named places, relative to where the plate is (or is going). Cutlery lies along y there.
# Taken from the scene's standard place setting so the two can't drift apart.
PLATE_HOME = np.array(SPOTS["plate"])   # "center" of the table setting
_rel = {k: np.array(v) - PLATE_HOME for k, v in SPOTS.items()}
PLACES = {
    "center": (0.0, 0.0),          # the plate's own spot
    "left_of_plate": tuple(_rel["fork"]),
    "right_of_plate": tuple(_rel["spoon"]),
    "top_right_of_plate": tuple(_rel["mug"]),
    "top_left_of_plate": (-_rel["mug"][0], _rel["mug"][1]),
}
RELAY = np.array([0.0, -0.11])        # where a round object is set down to swap arms
DEFAULT_SETTING = {"plate": "center", "fork": "left_of_plate", "spoon": "right_of_plate",
                   "mug": "top_right_of_plate"}
AT_GOAL_TOL = 0.02


@dataclass
class Goal:
    obj: str
    place: str

    def xy(self, plate_xy) -> np.ndarray:
        return (PLATE_HOME if self.obj == "plate" and self.place == "center"
                else np.asarray(plate_xy) + PLACES[self.place])

    @property
    def yaw(self):
        return np.pi / 2 if OBJECTS[self.obj][0] == "box" else None


@dataclass
class Report:
    goals: list
    done: dict = field(default_factory=dict)       # obj -> True/False (by the camera)
    attempts: dict = field(default_factory=dict)   # obj -> number of tries
    events: list = field(default_factory=list)     # human-readable story of the run


def order(goals: list[Goal]) -> list[Goal]:
    """Plate first (everything else is placed relative to it), then cutlery, then the mug."""
    rank = {"plate": 0, "fork": 1, "spoon": 2, "mug": 3, "bottle": 4}
    return sorted(goals, key=lambda g: rank.get(g.obj, 9))


REACH_MARGIN = 0.02  # must also reach this much further out, so grasp offsets still fit


def can_reach(sim: Sim, arm: str, xy) -> bool:
    """Can this arm point straight down at `xy` (with a little room to spare)?"""
    a = sim.arms[arm]
    xy = np.asarray(xy, float)
    out = xy - a.base_xy
    far = xy + REACH_MARGIN * out / max(np.linalg.norm(out), 1e-6)
    for yaw in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        _, err = a.ik(sim.d, np.array([far[0], far[1], 0.02]), yaw, q0=sim.home_q[arm])
        if err < 0.003 and a.last_rot_err < 0.05:
            return True
    return False


def nearest_arm(sim: Sim, xy) -> str:
    return min(sim.arms, key=lambda n: np.hypot(*(np.asarray(xy) - sim.arms[n].base_xy)))


def other(arm: str) -> str:
    return "B" if arm == "A" else "A"


def at_goal(sim: Sim, goal: Goal, target) -> bool:
    s = sim.world.get(goal.obj)
    if s is None:
        return False
    ok = np.hypot(s.x - target[0], s.y - target[1]) < AT_GOAL_TOL
    if goal.yaw is not None:
        ok &= abs(s.yaw % np.pi - goal.yaw % np.pi) < np.radians(20)
    return bool(ok)


def plate_xy(sim: Sim, goals: list[Goal]) -> np.ndarray:
    """Where the plate is — or will be, if moving it is one of the goals."""
    if any(g.obj == "plate" for g in goals):
        return PLATE_HOME
    return np.array([sim.world["plate"].x, sim.world["plate"].y]) if "plate" in sim.world else PLATE_HOME


def plan_move(sim: Sim, goal: Goal, target) -> list[tuple]:
    """Steps to bring one object from where the camera sees it to `target`."""
    here = np.array([sim.world[goal.obj].x, sim.world[goal.obj].y])
    first = nearest_arm(sim, here)
    if not can_reach(sim, first, here):
        first = other(first)
    policy = getattr(sim, "policy", None)
    if policy is not None and can_reach(sim, first, target):
        import policy_skill
        if policy_skill.covers(first, goal.obj, target):
            return [("policy", first, goal.obj, target)]
    if can_reach(sim, first, target):
        return [("pick", first, goal.obj), ("place", first, goal.obj, target, goal.yaw)]
    second = other(first)
    if OBJECTS[goal.obj][0] == "box":
        return [("handoff", first, second, goal.obj), ("place", second, goal.obj, target, goal.yaw)]
    # Round things can't be passed hand to hand; set it down where both arms can reach.
    relay = find_relay(sim, goal.obj)
    return [("pick", first, goal.obj), ("place", first, goal.obj, relay, None),
            ("home", first), ("observe",),
            ("pick", second, goal.obj), ("place", second, goal.obj, target, goal.yaw)]


def find_relay(sim: Sim, obj: str) -> np.ndarray:
    """The clearest spot on the table that both arms can reach, for swapping hands."""
    shape, size, *_ = OBJECTS[obj]
    radius = size[0] if shape != "box" else size[1]
    best, best_gap = RELAY, -np.inf
    for y in np.arange(-0.14, 0.15, 0.02):
        for x in (-0.02, 0.0, 0.02):
            xy = np.array([x, y])
            gap = skills._nearest_other(sim, obj, xy) - radius
            if gap > best_gap + 0.005 and all(can_reach(sim, a, xy) for a in sim.arms):
                best, best_gap = xy, gap
    return best


def run_step(sim: Sim, step: tuple) -> None:
    kind = step[0]
    if kind == "pick":
        skills.pick(sim, step[1], step[2])
    elif kind == "place":
        _, arm, obj, xy, yaw = step
        skills.place(sim, arm, obj, xy, obj_yaw=yaw)
    elif kind == "handoff":
        skills.handoff(sim, step[1], step[2], step[3])
    elif kind == "policy":
        import policy_skill
        policy_skill.run(sim, sim.policy, step[3])
    elif kind == "home":
        skills.home(sim, [step[1]])
    elif kind == "observe":
        skills.observe(sim)


def describe(step: tuple) -> str:
    kind = step[0]
    if kind == "pick":
        return f"arm {step[1]} picks up the {step[2]}"
    if kind == "place":
        return f"arm {step[1]} puts the {step[2]} at ({step[3][0]:+.2f}, {step[3][1]:+.2f})"
    if kind == "handoff":
        return f"arm {step[1]} hands the {step[3]} to arm {step[2]}"
    if kind == "policy":
        return (f"arm {step[1]} moves the {step[2]} to ({step[3][0]:+.2f}, {step[3][1]:+.2f}) "
                f"with the trained ACT policy (OpenVINO)")
    return kind


def complete(sim: Sim, goals: list[Goal], say=print) -> list[Goal]:
    """Add goals the command implies. Places are relative to the plate, so if the plate isn't
    in the middle of the setting yet, put it there first."""
    relative = any(g.obj != "plate" and g.place != "center" for g in goals)
    has_plate = any(g.obj == "plate" for g in goals)
    if relative and not has_plate and "plate" in sim.world:
        p = sim.world["plate"]
        if np.hypot(p.x - PLATE_HOME[0], p.y - PLATE_HOME[1]) > AT_GOAL_TOL:
            say("  (the plate isn't in the middle yet - I'll put it there first)")
            goals = [Goal("plate", "center")] + goals
    return goals


MAX_REPAIRS = 2  # per object: how often to put back something that got moved


def execute(sim: Sim, goals: list[Goal], max_tries: int = 3, say=print, after_goal=None) -> Report:
    """Work through the goals. After each one, look at everything already finished: if
    something has moved (knocked by an arm, or by a person), put it back before going on.
    `after_goal(sim, goal)` is a hook for demos that disturb the scene on purpose."""
    skills.observe(sim)
    goals = order(complete(sim, goals, say))
    report = Report(goals=goals)
    queue, finished, repairs = list(goals), [], {}
    while queue:
        goal = queue.pop(0)
        _achieve(sim, goal, goals, report, max_tries, say)
        if goal not in finished:
            finished.append(goal)
        if after_goal:
            after_goal(sim, goal)
        skills.observe(sim)
        for g in finished:
            moved = g.obj in sim.world and not at_goal(sim, g, g.xy(plate_xy(sim, goals)))
            if moved and g not in queue and repairs.get(g.obj, 0) < MAX_REPAIRS:
                repairs[g.obj] = repairs.get(g.obj, 0) + 1
                say(f"  I can see the {g.obj} has moved - putting it back first")
                report.events.append(f"{g.obj} moved; repaired")
                queue.insert(0, g)
    skills.observe(sim)
    for goal in goals:
        report.done[goal.obj] = goal.obj in sim.world and at_goal(sim, goal, goal.xy(plate_xy(sim, goals)))
    return report


def _achieve(sim, goal, all_goals, report, max_tries, say):
    """Bring one object to its goal: plan from what the camera sees, act, look, retry."""
    for attempt in range(1, max_tries + 1):
        target = goal.xy(plate_xy(sim, all_goals))
        if goal.obj not in sim.world:
            report.events.append(f"can't see the {goal.obj}")
            say(f"  I can't see the {goal.obj}.")
            return
        if at_goal(sim, goal, target):
            return
        report.attempts[goal.obj] = report.attempts.get(goal.obj, 0) + 1
        steps = plan_move(sim, goal, target)
        say(f"  {goal.obj} -> {goal.place}" + (f" (try {attempt})" if attempt > 1 else "")
            + ": " + "; ".join(describe(s) for s in steps if s[0] in ("pick", "place", "handoff", "policy")))
        try:
            for s in steps:
                run_step(sim, s)
        except skills.SkillError as e:
            report.events.append(f"{goal.obj}: {e}")
            say(f"  ! {e} - looking again")
            for arm in sim.arms:  # let go of anything half-held before re-planning
                sim.grip(arm, closed=False, seconds=0.3, opening=0.4)
        sim.held.clear()
        skills.home(sim)
        skills.observe(sim)
