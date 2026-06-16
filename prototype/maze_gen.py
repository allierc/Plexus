"""maze_gen.py -- generate a real (perfect) maze as obstacle rectangles for a spec.

Recursive-backtracker over a W x H grid of rooms (W>H -> longitudinal). A perfect
maze has exactly one path between any two rooms, hence many genuine DEAD ENDS. We
emit the remaining interior walls as thin rectangles [x0,y0,x1,y1] in [0,1]^2; the
left edge (entry) and right edge (exit at x>0.9) are open, the domain border is the
MPM/field wall. Prints a YAML `obstacles:` block and BFS-verifies left->right.

    python maze_gen.py 9 4 1        # W H seed
"""
from __future__ import annotations
import sys, random
from collections import deque


def generate(W, H, seed, t=0.010, x0=0.0):
    """x0 > 0 insets the maze into x in [x0, 1] and leaves x < x0 as an open entry
    corridor (the column-0 rooms open onto it), so cells spawn clear of any wall."""
    rng = random.Random(seed)
    # carve a spanning tree; vwall[(i,j)] = wall between rooms (i,j)|(i+1,j); hwall[(i,j)] = (i,j)|(i,j+1)
    vwall = {(i, j): True for i in range(W - 1) for j in range(H)}
    hwall = {(i, j): True for i in range(W) for j in range(H - 1)}
    seen = [[False] * H for _ in range(W)]
    stack = [(0, 0)]; seen[0][0] = True
    while stack:
        i, j = stack[-1]
        nb = []
        if i > 0 and not seen[i - 1][j]: nb.append(("W", i - 1, j))
        if i < W - 1 and not seen[i + 1][j]: nb.append(("E", i + 1, j))
        if j > 0 and not seen[i][j - 1]: nb.append(("S", i, j - 1))
        if j < H - 1 and not seen[i][j + 1]: nb.append(("N", i, j + 1))
        if not nb:
            stack.pop(); continue
        d, ni, nj = rng.choice(nb)
        if d == "E": vwall[(i, j)] = False
        if d == "W": vwall[(ni, nj)] = False
        if d == "N": hwall[(i, j)] = False
        if d == "S": hwall[(i, nj)] = False
        seen[ni][nj] = True; stack.append((ni, nj))

    dx, dy = 1.0 / W, 1.0 / H
    sx = lambda x: x0 + x * (1.0 - x0)                     # inset x into [x0, 1]
    rects = []
    for (i, j), on in vwall.items():                       # vertical segments at x=(i+1)/W
        if on:
            x = sx((i + 1) * dx)
            rects.append([round(x - t, 4), round(j * dy, 4), round(x + t, 4), round((j + 1) * dy, 4)])
    for (i, j), on in hwall.items():                       # horizontal segments at y=(j+1)/H
        if on:
            y = (j + 1) * dy
            rects.append([round(sx(i * dx), 4), round(y - t, 4), round(sx((i + 1) * dx), 4), round(y + t, 4)])
    return rects


def bfs_ok(rects, res=160, xf=0.90):
    import numpy as np
    sys.path.insert(0, ".")
    import engine2
    walls = engine2._raster(rects, res, "cpu").numpy()
    seen = np.zeros((res, res), bool); dq = deque()
    for i in range(int(0.06 * res)):
        for j in range(res):
            if not walls[i, j]: seen[i, j] = True; dq.append((i, j))
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < res and 0 <= b < res and not walls[a, b] and not seen[a, b]:
                seen[a, b] = True; dq.append((a, b))
    return bool(seen[int(xf * res):, :].any()), float(walls.mean())


if __name__ == "__main__":
    W, H, seed = (int(x) for x in (sys.argv[1:4] if len(sys.argv) >= 4 else (9, 4, 1)))
    x0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.14
    rects = generate(W, H, seed, x0=x0)
    ok, frac = bfs_ok(rects)
    print(f"# maze {W}x{H} seed {seed}: {len(rects)} walls, passable={ok}, wallfrac={frac:.3f}")
    print("obstacles:")
    for r in rects:
        print(f"  - [{r[0]}, {r[1]}, {r[2]}, {r[3]}]")
