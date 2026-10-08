"""exp17: THE VISUAL STIMULUS OF EACH BLOCK AS A SMALL MOVIE (Cedric, 2026-10-07: "small movies of the visual input to
illustrate the block slides, top right, square as described in the paper"). Local, no model.

ZAPBench (Lueckmann et al. 2025, A.3-A.4): the stimuli are projected UNDERNEATH the head-fixed fish, in a square chamber.
Each movie is that square seen from above, the fish at its centre head up, drawn from the RECORDED stimulus columns
(B.6's encoding, 0-based):
    0 gain (-1 low, +1 high)            2 dots (+1 the coherent rightward motion, -1 flicker)   4 flash (-1 dark, +1 light)
    6, 7 taxis left / right (-1 dark, +1 light)                9 turning speed, 10, 11 its direction (sin, cos)
    13-15 position, one-hot: 13 open-loop forward, 14 still (delay or rest), 15 the 1-s pulse   19 rotation (+1 left, -1 right)
open loop: a forward grating throughout; dark: nothing. A forward grating drifts from the tail to the head (the
optomotor stimulus the fish swims against). The drift and the dots move at a fixed illustrative speed per movie frame;
the STATE (which grating, which direction, light or dark) follows the recording at SPEED x real time.

    PYTHONPATH=src:tools python tools/exp17_stim_movies.py
-> presentation/Movies/stim_<block>.mp4 (+ .png) and stim_all.mp4 (the nine blocks in order), DUR_S each
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
MOV = os.path.join(EXP, "presentation", "Movies")
DT, SPEED, FPS, PX, PERIOD = 0.914, 30.0, 25, 256, 48.0
DUR_S, LEAD_S = 2.0, 30.0      # each movie 2 s, played once (Cedric, 2026-10-08: "they draw too much attention"),
                               # from 30 s of recording before the block's first stimulus change


def fish_mask(px=PX):
    """A small fish from above, head up, at the centre: two eyes, a head, a tapering body."""
    y, x = np.mgrid[0:px, 0:px].astype(np.float32)
    cx, cy = px / 2, px / 2
    s = px / 256.0
    head = ((x - cx) / (11 * s)) ** 2 + ((y - (cy - 10 * s)) / (15 * s)) ** 2 < 1
    body_t = (y - (cy - 2 * s)) / (60 * s)                         # 0 at the head's back, 1 at the tail tip
    body = (body_t >= 0) & (body_t <= 1) & (np.abs(x - cx) < (7 * s) * (1 - body_t) + 1.0 * s)
    eyes = (((x - (cx - 8 * s)) / (5 * s)) ** 2 + ((y - (cy - 18 * s)) / (6 * s)) ** 2 < 1) | \
           (((x - (cx + 8 * s)) / (5 * s)) ** 2 + ((y - (cy - 18 * s)) / (6 * s)) ** 2 < 1)
    return head | body, eyes


def grating(theta, phase, px=PX):
    """A sine grating drifting along direction theta (radians, 0 = toward the fish's head, i.e. up), in [0, 1]."""
    y, x = np.mgrid[0:px, 0:px].astype(np.float32)
    u = -np.sin(theta) * (x - px / 2) * -1 + np.cos(theta) * (y - px / 2)   # coordinate along the motion
    return 0.5 + 0.5 * np.sin(2 * np.pi * (u + phase) / PERIOD)


def windmill(rot, px=PX, n=8):
    """A radial (windmill) sine grating turned by `rot` radians."""
    y, x = np.mgrid[0:px, 0:px].astype(np.float32)
    a = np.arctan2(y - px / 2, x - px / 2)
    return 0.5 + 0.5 * np.sin(n * a + rot)


class Dots:
    """Flickering dots, 200-ms lifetime; with `coherent` every dot moves right."""
    def __init__(self, n=220, px=PX, seed=0):
        self.g = np.random.default_rng(seed)
        self.p = self.g.uniform(0, px, (n, 2))
        self.age = self.g.uniform(0, 1, n)
        self.px = px

    def draw(self, coherent, dt_s):
        life = 0.2 * SPEED / SPEED                                  # the dots' 200 ms, illustrative (one per frame)
        self.age += dt_s / life * 0.25
        dead = self.age > 1
        self.p[dead] = self.g.uniform(0, self.px, (int(dead.sum()), 2))
        self.age[dead] = 0
        if coherent:
            self.p[:, 0] = (self.p[:, 0] + 4.0) % self.px
        im = np.zeros((self.px, self.px), np.float32)
        q = self.p.astype(int)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                im[np.clip(q[:, 1] + dy, 0, self.px - 1), np.clip(q[:, 0] + dx, 0, self.px - 1)] = 1.0
        return im


def frame_of(block, u, phase, dots, t_block):
    """One [PX, PX] grey image of the screen and its state label, from the stimulus row u (22 columns)."""
    if block == "gain":
        return grating(0.0, phase), ("high gain" if u[0] > 0 else "low gain")
    if block == "dots":
        coh = u[2] > 0
        return dots.draw(coh, 1.0), ("dots moving right" if coh else "flickering dots")
    if block == "flash":
        return np.full((PX, PX), 1.0 if u[4] > 0 else 0.0, np.float32), ("light" if u[4] > 0 else "dark")
    if block == "taxis":
        im = np.zeros((PX, PX), np.float32)
        im[:, :PX // 2] = 1.0 if u[6] > 0 else 0.0
        im[:, PX // 2:] = 1.0 if u[7] > 0 else 0.0
        return im, f"left {'light' if u[6] > 0 else 'dark'}, right {'light' if u[7] > 0 else 'dark'}"
    if block == "turning":
        th = float(np.arctan2(u[10], u[11]))                         # the direction, sin / cos
        moving = abs(u[9]) > 1e-3
        name = {0: "forward", 1: "left", 2: "back", 3: "right"}[int(round(((th % (2 * np.pi)) / (np.pi / 2)))) % 4]
        return grating(th, phase if moving else 0.0), (name if moving else "still")
    if block == "position":
        # read from the recording (2026-10-07): 15 = the 1-s pulse, 14 = a still grating (the delay while 16 > 0, the
        # 30-s rest between trials while 16 = 0), 13 = the 30-s open-loop forward grating
        k = int(np.argmax(u[13:16])) if np.abs(u[13:16]).max() > 0 else 1
        lab = ["open loop forward", ("delay (still)" if u[16] > 0 else "rest (still)"), "forward pulse"][k]
        return grating(0.0, phase if k != 1 else 0.0), lab
    if block == "open loop":
        return grating(0.0, phase), "forward, open loop"
    if block == "rotation":
        return windmill(phase / PERIOD * 2 * np.pi * (1 if u[19] > 0 else -1)), ("turning left" if u[19] > 0 else "turning right")
    return np.zeros((PX, PX), np.float32), "dark"


def render(seq, stem, label_block=False):
    """seq: one (block, stimulus row [22]) per movie frame, at FPS -> <stem>.mp4, its poster the first frame."""
    from PIL import Image, ImageDraw
    from plexus.tasks.trace_recording import _ffmpeg
    body, eyes = fish_mask()
    dots = Dots()
    tmp = tempfile.mkdtemp(prefix="stim_")
    phase = 0.0
    for i, (blk, r) in enumerate(seq):
        phase += 3.0                                                # the drift, px per movie frame
        im, lab = frame_of(blk, r, phase, dots, i / FPS)
        g = (0.12 + 0.76 * im)                                      # the projector's grey range
        rgb = np.stack([g, g, g], -1)
        rgb[body] = (1.0, 0.55, 0.15)                               # the fish, orange
        rgb[eyes] = (0.05, 0.05, 0.05)
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        d.rectangle([0, PX - 22, PX, PX], fill=(0, 0, 0))
        d.text((6, PX - 18), (f"{blk}: " if label_block else "") + lab, fill=(255, 255, 255))
        img.save(os.path.join(tmp, f"{i:05d}.png"))
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%05d.png"),
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", stem + ".mp4"], check=True)
    shutil.copy(os.path.join(tmp, "00000.png"), stem + ".png")
    shutil.rmtree(tmp)
    print(f"[stim] {stem}.mp4: {len(seq)} frames, {len(seq) / FPS:.1f} s")


def window(rows, n):
    """n movie frames of the block at SPEED x, starting LEAD_S of recording before its first stimulus change (both
    states shown); a block that never changes from its first frame."""
    ch = np.flatnonzero(np.abs(np.diff(rows, axis=0)).max(1) > 1e-6)
    t0 = max(0.0, (ch[0] + 1) * DT - LEAD_S) if len(ch) else 0.0
    return [rows[min(int((t0 + i * SPEED / FPS) / DT), len(rows) - 1)] for i in range(n)]


def main():
    from plexus.paths import graphs_data_path
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
    U, off, names = np.asarray(z["stimulus"], np.float32), z["offsets"], [str(x) for x in z["names"]]
    os.makedirs(MOV, exist_ok=True)
    n = int(round(DUR_S * FPS))
    for k, b in enumerate(names):
        render([(b, r) for r in window(U[off[k]:off[k + 1]], n)], os.path.join(MOV, f"stim_{b.replace(' ', '_')}"))
    # the atlas slide: the nine blocks in order within the same DUR_S, each its share of the frames from its first change
    seq = []
    for k, idx in enumerate(np.array_split(np.arange(n), len(names))):
        seq += [(names[k], r) for r in window(U[off[k]:off[k + 1]], len(idx))]
    render(seq, os.path.join(MOV, "stim_all"), label_block=True)


if __name__ == "__main__":
    main()
