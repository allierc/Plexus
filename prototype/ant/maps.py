"""Load Sebastian Lague's saved ant maps (papers/Ant-Simulation/Assets/SavedMap/*.txt)
into a Plexus obstacle list -- the *actual* environment, reused as the one MPM+field
boundary-condition mask (paper/plexus.tex: "obstacles are boundary conditions").

The map is a {width,height,map:[...]} scalar grid (the marching-squares editor field);
cells below `thr` are rock/wall. We max-pool to a coarse grid (so thin walls survive),
run-length-encode each row into horizontal rectangles, and return them in world
coordinates over [0,W]x[0,1] with square cells (W = width/height).
"""
import json, os
import numpy as np

MAPDIR = "/workspace/Plexus/papers/Ant-Simulation/Assets/SavedMap"


def load_walls(name="Map.txt", factor=4, thr=0.81):
    d = json.load(open(os.path.join(MAPDIR, name)))
    w, h = d["width"], d["height"]
    m = np.array(d["map"]).reshape(h, w)
    H, W = h // factor, w // factor
    wall = (m < thr)[:H * factor, :W * factor].reshape(H, factor, W, factor).max(axis=(1, 3))
    return wall, W, H


def rects_and_world(name="Map.txt", factor=4, thr=0.81):
    """Return (rect_list, world_width). Each rect is [x0,y0,x1,y1] in world coords."""
    wall, W, H = load_walls(name, factor, thr)
    s = 1.0 / H
    rects = []
    for r in range(H):
        c = 0
        while c < W:
            if wall[r, c]:
                c0 = c
                while c < W and wall[r, c]:
                    c += 1
                rects.append([round(c0 * s, 4), round(r * s, 4), round(c * s, 4), round((r + 1) * s, 4)])
            else:
                c += 1
    return rects, W / H, wall


def open_disc(wall, W, H, x_frac, y_frac, want_r):
    """Find an open spot near (x_frac,y_frac) of the world and the largest clear radius
    up to want_r, so a nest/food disc sits in free space. Returns (cx,cy,r) in world."""
    s = 1.0 / H
    # search outward from the requested cell for an open cell with clearance
    cx0, cy0 = int(x_frac * W), int(y_frac * H)
    best = None
    for rad in range(0, max(W, H)):
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                cx, cy = cx0 + dx, cy0 + dy
                if not (0 <= cx < W and 0 <= cy < H) or wall[cy, cx]:
                    continue
                # clearance: how far to nearest wall (in cells), capped
                k = 1
                while k < int(want_r / s) + 2:
                    lo_y, hi_y = max(0, cy - k), min(H, cy + k + 1)
                    lo_x, hi_x = max(0, cx - k), min(W, cx + k + 1)
                    if wall[lo_y:hi_y, lo_x:hi_x].any():
                        break
                    k += 1
                clear = k * s
                if best is None or clear > best[3]:
                    best = (cx, cy, clear, clear)
        if best is not None and best[2] >= want_r:
            break
    cx, cy, clear, _ = best
    return round((cx + 0.5) * s, 4), round((cy + 0.5) * s, 4), round(min(clear * 0.8, want_r), 4)
