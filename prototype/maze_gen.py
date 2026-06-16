"""maze_gen.py -- generate a real (perfect) maze as obstacle rectangles for a spec.

Recursive-backtracker over a cols x rows grid of rooms. A perfect maze has exactly
one path between any two rooms, hence many genuine DEAD ENDS. The maze is laid out in
the world rectangle [0,W]x[0,1] (W>1 -> longitudinal), inset into x in [x0, W] so that
x < x0 is an open entry corridor (cells spawn there, clear of walls). Rooms are made
roughly square by choosing cols ~ (W-x0)*rows. Prints a YAML `obstacles:` block plus
a `world:` line, and BFS-verifies that the right edge is reachable from the left.

    python maze_gen.py 15 4 1 4.0 0.30      # cols rows seed W x0
"""
from __future__ import annotations
import sys, random
from collections import deque


def generate(cols, rows, seed, W=4.0, x0=0.30, t=0.012):
    rng = random.Random(seed)
    vwall = {(i, j): True for i in range(cols - 1) for j in range(rows)}
    hwall = {(i, j): True for i in range(cols) for j in range(rows - 1)}
    seen = [[False] * rows for _ in range(cols)]
    stack = [(0, 0)]; seen[0][0] = True
    while stack:
        i, j = stack[-1]
        nb = []
        if i > 0 and not seen[i - 1][j]: nb.append(("W", i - 1, j))
        if i < cols - 1 and not seen[i + 1][j]: nb.append(("E", i + 1, j))
        if j > 0 and not seen[i][j - 1]: nb.append(("S", i, j - 1))
        if j < rows - 1 and not seen[i][j + 1]: nb.append(("N", i, j + 1))
        if not nb:
            stack.pop(); continue
        d, ni, nj = rng.choice(nb)
        if d == "E": vwall[(i, j)] = False
        if d == "W": vwall[(ni, nj)] = False
        if d == "N": hwall[(i, j)] = False
        if d == "S": hwall[(i, nj)] = False
        seen[ni][nj] = True; stack.append((ni, nj))

    dxr = (W - x0) / cols                                   # room width (world units)
    dyr = 1.0 / rows
    sx = lambda i: x0 + i * dxr
    rects = []
    for (i, j), on in vwall.items():                        # vertical segments at x=x0+(i+1)*dxr
        if on:
            x = sx(i + 1)
            rects.append([round(x - t, 4), round(j * dyr, 4), round(x + t, 4), round((j + 1) * dyr, 4)])
    for (i, j), on in hwall.items():                        # horizontal segments at y=(j+1)*dyr
        if on:
            y = (j + 1) * dyr
            rects.append([round(sx(i), 4), round(y - t, 4), round(sx(i + 1), 4), round(y + t, 4)])
    return rects


def bfs_ok(rects, W, ny=160):
    import numpy as np
    sys.path.insert(0, ".")
    import engine2
    walls = engine2._raster(rects, ny, W, "cpu").numpy()    # [nx, ny]
    nx = walls.shape[0]
    seen = np.zeros((nx, ny), bool); dq = deque()
    for i in range(int(0.04 * nx)):
        for j in range(ny):
            if not walls[i, j]: seen[i, j] = True; dq.append((i, j))
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and not walls[a, b] and not seen[a, b]:
                seen[a, b] = True; dq.append((a, b))
    return bool(seen[int(0.97 * nx):, :].any()), float(walls.mean())


if __name__ == "__main__":
    a = sys.argv[1:]
    cols, rows, seed = int(a[0]), int(a[1]), int(a[2])
    W = float(a[3]) if len(a) > 3 else 4.0
    x0 = float(a[4]) if len(a) > 4 else 0.30
    rects = generate(cols, rows, seed, W=W, x0=x0)
    ok, frac = bfs_ok(rects, W)
    print(f"# maze {cols}x{rows} seed {seed} W={W} x0={x0}: {len(rects)} walls, passable={ok}, wallfrac={frac:.3f}")
    print(f"world: {W}")
    print("obstacles:")
    for r in rects:
        print(f"  - [{r[0]}, {r[1]}, {r[2]}, {r[3]}]")
