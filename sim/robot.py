"""Arm control: inverse kinematics (where should the joints go so the gripper is *here*?),
smooth motion, gripper open/close, and a recorder that turns the run into video frames.

Grasps are top-down: the gripper points straight at the table and `yaw` sets the direction
the fingers close along.
"""
import numpy as np
import mujoco

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIP_OPEN = 1.0     # ~5.4 cm between the finger pads
GRIP_CLOSED = -0.17

# Gripper command -> gap between the pads' inner faces (m), measured on Day 2.
_GRIP_GAP = ([-0.17, 0.0, 0.5, 1.0], [0.003, 0.016, 0.036, 0.054])


def grip_for_gap(gap: float) -> float:
    """Gripper command that opens the pads to `gap` metres."""
    return float(np.interp(gap, _GRIP_GAP[1], _GRIP_GAP[0]))


def close_renderer(r: mujoco.Renderer | None) -> None:
    """Free a renderer without breaking the others. mujoco.Renderer.close() destroys its GL
    context *before* freeing its GPU objects, so they get freed in whichever context is
    current — another renderer's — which then draws black. Always close renderers with this."""
    if r is None:
        return
    if r._gl_context:
        r._gl_context.make_current()
    if r._mjr_context:
        r._mjr_context.free()
        r._mjr_context = None
    if r._gl_context:
        r._gl_context.free()
        r._gl_context = None


def top_down_rotation(yaw: float) -> np.ndarray:
    """Gripper-site rotation: x (approach) points down, z (finger closing) is horizontal at `yaw`."""
    x = np.array([0.0, 0.0, -1.0])
    z = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


class Arm:
    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.m, self.prefix = model, prefix
        self.site = model.site(f"{prefix}gripperframe").id
        self.qadr = np.array([model.joint(prefix + j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(prefix + j).dofadr[0] for j in ARM_JOINTS])
        self.act = np.array([model.actuator(prefix + j).id for j in ARM_JOINTS])
        self.grip_act = model.actuator(prefix + "gripper").id
        self.lo = model.jnt_range[[model.joint(prefix + j).id for j in ARM_JOINTS], 0]
        self.hi = model.jnt_range[[model.joint(prefix + j).id for j in ARM_JOINTS], 1]
        self.pads = {model.geom(prefix + "pad_fixed").id, model.geom(prefix + "pad_moving").id}
        self.base_xy = model.body(prefix + "base").pos[:2].copy()
        self._scratch = mujoco.MjData(model)

    def site_pose(self, data):
        return data.site_xpos[self.site].copy(), data.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, data, pos, yaw, q0=None, iters=300, tol=1e-3, R_goal=None):
        """Joint angles that put the gripper at `pos` pointing down with fingers at `yaw`
        (or at orientation `R_goal`, if given). Returns (q, position error in metres)."""
        s = self._scratch
        s.qpos[:] = data.qpos
        if q0 is not None:
            s.qpos[self.qadr] = q0
        q_goal = np.zeros(4)
        R_goal = top_down_rotation(yaw) if R_goal is None else np.asarray(R_goal)
        mujoco.mju_mat2Quat(q_goal, R_goal.flatten())
        q_cur, e_local = np.zeros(4), np.zeros(3)
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        err_p = np.inf
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)
            p, R = s.site_xpos[self.site], s.site_xmat[self.site].reshape(3, 3)
            e_p = pos - p
            # Rotation error as a world-frame rotation vector. (A cross-product error would
            # read zero when the gripper is turned exactly 180°, closing from the wrong side.)
            mujoco.mju_mat2Quat(q_cur, R.flatten())
            mujoco.mju_subQuat(e_local, q_goal, q_cur)
            e_r = R @ e_local
            err_p = np.linalg.norm(e_p)
            if err_p < tol and np.linalg.norm(e_r) < 0.02:
                break
            mujoco.mj_jacSite(self.m, s, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], 0.5 * jacr[:, self.dadr]])
            e = np.concatenate([e_p, 0.5 * e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            s.qpos[self.qadr] = np.clip(s.qpos[self.qadr] + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        self.last_rot_err = float(np.linalg.norm(e_r))  # radians
        return s.qpos[self.qadr].copy(), err_p

    def holding(self, data, body_id: int) -> bool:
        """True when both finger pads touch the given object (any of its geoms)."""
        touched = set()
        for c in data.contact[: data.ncon]:
            bodies = {self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]}
            if body_id in bodies:
                touched |= {c.geom1, c.geom2} & self.pads
        return touched == self.pads


class Sim:
    """Steps physics, drives both arms, and records frames."""

    def __init__(self, model, record_cameras=("front",), fps=12, size=(480, 640)):
        self.m, self.d = model, mujoco.MjData(model)
        self.arms = {"A": Arm(model, "a_"), "B": Arm(model, "b_")}
        self.fps, self.cams = fps, record_cameras
        self.frames = {c: [] for c in record_cameras}
        self._renderer = mujoco.Renderer(model, *size) if record_cameras else None
        self._next_frame = 0.0
        self.log: list[str] = []
        # What the robot believes, as opposed to simulator truth:
        self.eyes = None       # anything with .look(data) -> {name: obj with .x .y .yaw}
        self.world: dict = {}  # last thing the eyes saw
        self.held: dict = {}   # arm -> (object name, position in gripper frame, yaw offset)
        # Optional sampler for recording demonstrations: called every `tick_period` s of sim time.
        self.on_tick = None
        self.tick_period = 0.05
        self._next_tick = 0.0

    def close(self):
        close_renderer(self._renderer)
        self._renderer = None

    def step(self, seconds: float):
        for _ in range(int(seconds / self.m.opt.timestep)):
            if self.on_tick and self.d.time >= self._next_tick:
                self.on_tick(self)
                self._next_tick += self.tick_period
            mujoco.mj_step(self.m, self.d)
            if self._renderer and self.d.time >= self._next_frame:
                for c in self.cams:
                    self._renderer.update_scene(self.d, camera=c)
                    self.frames[c].append(self._renderer.render())
                self._next_frame += 1 / self.fps

    def set_home(self, poses: dict):
        """Teleport arms to a joint pose (used once at start) and remember it as home."""
        self.home_q = {name: np.asarray(q).copy() for name, q in poses.items()}
        for name, q in poses.items():
            arm = self.arms[name]
            self.d.qpos[arm.qadr] = q
            self.d.ctrl[arm.act] = q
            self.d.ctrl[arm.grip_act] = GRIP_OPEN
            self.d.qpos[self.m.joint(arm.prefix + "gripper").qposadr[0]] = GRIP_OPEN
        mujoco.mj_forward(self.m, self.d)

    def move(self, targets: dict, duration=1.0):
        """Move one or both arms at the same time. targets: arm -> (pos, yaw).
        The gripper travels in a straight line, solved as a chain of IK waypoints."""
        n = max(2, int(duration * 20))
        plans = {}
        for name, (pos, yaw) in targets.items():
            arm = self.arms[name]
            start, R = arm.site_pose(self.d)
            yaw0 = np.arctan2(R[1, 2], R[0, 2])
            short = (yaw - yaw0 + np.pi) % (2 * np.pi) - np.pi
            # The wrist can't spin freely, so turning the short way round can run into its
            # limit. Plan both directions and keep the one that stays on target.
            best = None
            for turn in (short, short - np.sign(short) * 2 * np.pi):
                q, path, worst = self.d.ctrl[arm.act].copy(), [], 0.0
                for i in range(1, n + 1):
                    a = 0.5 - 0.5 * np.cos(np.pi * i / n)  # ease in/out
                    q, err = arm.ik(self.d, start + a * (np.asarray(pos) - start), yaw0 + a * turn, q0=q)
                    worst = max(worst, err + 0.02 * arm.last_rot_err)
                    path.append(q)
                if best is None or worst < best[0]:
                    best = (worst, path, err, arm.last_rot_err)
                if worst < 0.005:
                    break
            _, path, err, rot = best
            if err > 0.01 or rot > 0.15:
                self.log.append(f"arm {name}: target {np.round(pos, 3)} off by {err * 100:.1f} cm,"
                                f" {np.degrees(rot):.0f}°")
            plans[name] = path
        for i in range(n):
            for name, path in plans.items():
                self.d.ctrl[self.arms[name].act] = path[i]
            self.step(duration / n)
        self.step(0.15)  # settle

    def move_joints(self, targets: dict, duration=1.0):
        """Move arms to joint poses directly (no IK). targets: arm -> joint angles."""
        n = max(2, int(duration * 20))
        starts = {a: self.d.ctrl[self.arms[a].act].copy() for a in targets}
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / n)
            for a, q in targets.items():
                self.d.ctrl[self.arms[a].act] = starts[a] + s * (np.asarray(q) - starts[a])
            self.step(duration / n)
        self.step(0.15)

    def grip(self, arm: str, closed: bool, seconds=0.6, opening: float = GRIP_OPEN):
        self.d.ctrl[self.arms[arm].grip_act] = GRIP_CLOSED if closed else opening
        self.step(seconds)
