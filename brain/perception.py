"""Eyes: find the tableware in the overhead camera's colour + depth image.

No simulator ground truth is used for objects. The one thing taken from the simulator is the
robot's own outline (a "self-filter" — real robots mask themselves out using their known
shape and joint angles), so the arms aren't mistaken for tableware.

How it works:
  1. depth image -> height of every pixel above the table
  2. pixels higher than 2 mm, minus the robot, grouped into blobs
  3. each blob classified by height, size and shape; colour only separates fork from spoon
  4. position from the blob's geometry: circle fit for the plate rim, top-face centre for
     mug/bottle, principal axis for cutlery (gives its direction too)
"""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import ndimage

from robot import close_renderer

CAMERA = "top"
H, W = 720, 960          # ~0.7 mm per pixel on the table
MIN_HEIGHT = 0.002       # anything lower is table
MIN_AREA = 0.0002        # m² — ignore specks (2 cm²)


@dataclass
class Seen:
    name: str
    x: float
    y: float
    yaw: float           # cutlery: direction of the long axis; 0 for round things
    height: float        # tallest point above the table (m)
    area: float          # top-view area (m²)
    confidence: str = "high"

    @property
    def xy(self):
        return np.array([self.x, self.y])


class Eyes:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.r = mujoco.Renderer(model, H, W)
        cam = model.camera(CAMERA)
        self.cam_pos = model.cam_pos[cam.id].copy()
        self.f = (H / 2) / np.tan(np.radians(model.cam_fovy[cam.id]) / 2)
        self.robot_geoms = np.array([model.geom(i).name.startswith(("a_", "b_")) or
                                     model.body(model.geom_bodyid[i]).name.startswith(("a_", "b_"))
                                     for i in range(model.ngeom)])
        # Pixel grid -> world xy per unit depth (camera looks straight down, image up = world +y).
        v, u = np.mgrid[0:H, 0:W]
        self._kx = (u + 0.5 - W / 2) / self.f
        self._ky = -(v + 0.5 - H / 2) / self.f

    def close(self):
        close_renderer(self.r)
        self.r = None

    def capture(self, data: mujoco.MjData):
        """Colour image, height map, and robot mask from the overhead camera."""
        self.r.update_scene(data, camera=CAMERA)
        rgb = self.r.render().astype(np.float32) / 255
        self.r.enable_depth_rendering()
        depth = self.r.render().copy()
        self.r.disable_depth_rendering()
        self.r.enable_segmentation_rendering()
        seg = self.r.render()[..., 0].copy()
        self.r.disable_segmentation_rendering()
        robot = np.zeros((H, W), bool)
        valid = seg >= 0
        robot[valid] = self.robot_geoms[seg[valid]]
        height = self.cam_pos[2] - depth
        wx = self.cam_pos[0] + self._kx * depth
        wy = self.cam_pos[1] + self._ky * depth
        return rgb, height, wx, wy, robot

    def look(self, data: mujoco.MjData) -> dict[str, Seen]:
        rgb, height, wx, wy, robot = self.capture(data)
        px_area = (self.cam_pos[2] / self.f) ** 2  # m² per pixel at table height
        mask = (height > MIN_HEIGHT) & ~robot & (np.abs(wx) < 0.42) & (np.abs(wy) < 0.30)
        # Robot pixels count as "unknown", so objects next to an arm are not cut in two.
        labels, n = ndimage.label(mask | (robot & (height > MIN_HEIGHT)))
        found: list[Seen] = []
        for i in range(1, n + 1):
            blob = (labels == i) & mask
            if blob.sum() * px_area < MIN_AREA:
                continue
            found.append(self._classify(blob, rgb, height, wx, wy, px_area))
        # One of each object. If two blobs claim a name, keep the larger.
        out: dict[str, Seen] = {}
        for s in sorted(found, key=lambda s: -s.area):
            if s.name != "unknown" and s.name not in out:
                out[s.name] = s
        return out

    def _classify(self, blob, rgb, height, wx, wy, px_area) -> Seen:
        h = height[blob]
        xs, ys = wx[blob], wy[blob]
        top = h.max()
        area = blob.sum() * px_area
        pts = np.column_stack([xs, ys])
        centre = pts.mean(0)
        cov = np.cov((pts - centre).T)
        evals, evecs = np.linalg.eigh(cov)
        elong = np.sqrt(evals[1] / max(evals[0], 1e-12))

        if top > 0.07:
            name = "bottle"
        elif top > 0.035:
            name = "mug"
        elif elong > 3.0:
            colour = rgb[blob].mean(0)
            name = "fork" if colour[2] / max(colour[0], 1e-6) > 0.88 else "spoon"
        elif area > 0.003:
            name = "plate"
        else:
            name = "unknown"

        yaw = 0.0
        if name in ("mug", "bottle"):
            topface = blob & (height > top - 0.003)
            centre = np.array([wx[topface].mean(), wy[topface].mean()])
        elif name == "plate":
            rim = blob & (height > 0.010)
            centre = _fit_circle(wx[rim], wy[rim]) if rim.sum() > 30 else centre
        elif name in ("fork", "spoon"):
            axis = evecs[:, 1]
            yaw = float(np.arctan2(axis[1], axis[0]))
            along = (pts - centre) @ axis
            centre = centre + axis * (along.min() + along.max()) / 2
        return Seen(name, float(centre[0]), float(centre[1]), yaw, float(top), float(area))


def _fit_circle(x, y):
    """Least-squares circle centre (algebraic fit) — works on a partly hidden rim."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([c[0] / 2, c[1] / 2])
