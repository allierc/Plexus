"""06 -- THE THREE BODIES ON ONE REPLAYED TISSUE, IN ONE 2x2.

    python test_06_panels.py --device cuda:0                 # solve the sheet, then draw
    python test_06_panels.py --reuse                          # draw from a solved sheet, no GPU

WHAT THIS IS, AND WHAT IT IS NOT. The epithelium is a REPLAY -- `cellfix_B_new_f401`, 396 vertices at
frame 0 and 12,756 at the last -- and both other bodies are solved against it, each by its own certified
rig, neither one seeing the other:

    the matrix    `06_spheroid_ecm`: 200,000 MPM particles on 10,000 fibres, loaded through
                  `mesh_contact` (03's ICFEMP on a curved moving surface). Already solved; this
                  script does not re-run it, it re-draws it.
    the sheet     `Rig06c` -- 05b's rig with ONE variable changed, the driver -- run here, 401 frames,
                  2,562 nodes on 5,120 faces held by plaques to the tissue's own faces.

THERE IS NO BM-ECM INTERACTION, AND THAT IS THE DESIGN. Nothing gives the matrix the sheet's reaction
or the sheet the matrix's; `fibril_pull`, the sheet->stroma arrow, does not exist anywhere yet. So this
is a picture of three bodies sharing one tissue, not of three bodies in contact, and the panel says so.
The two halves ARE tied to the same frame axis: `mesh_contact` steps the matrix once per kept mesh
(`mesh_stride` pass-2 frames each), so a pass-2 frame is mapped back to its TISSUE frame and the sheet
is drawn at that tissue frame -- the same pairing `run_ecm.render` already makes for the epithelium
itself, rather than the two clocks being assumed to agree.

WHY THE BOTTOM-LEFT WAS EMPTY UNTIL NOW. `run_ecm.render` reserves it for the basement membrane and
draws nothing there when the run has no membrane SET -- the interface runs have none, because the BM
here is a vertex-model sheet in another rig rather than a third MPM set. `bm_draw` is that panel handed
in from outside; `bm_panel.draw_bm` draws it.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bm_panel                                                        # noqa: E402
import test_05b_plaque as B                                            # noqa: E402
import test_06c_real_driver as R6                                      # noqa: E402

SRC = "06_spheroid_ecm"                    # the matrix half, solved and certified, re-drawn not re-run
# NOT `06_three_bodies`: that name belongs to the run that failed, and a folder holding the picture of
# a working 06 must not carry the name of the experiment it replaced.
NAME = "06_spheroid_bm_ecm"
STORE = "bm_frames.npz"
CAM = dict(elev=18, azim=30)               # ecm_render.CAM_SIDE -- the SAME camera as the other panels


# =============================================================================================
#  the sheet half
# =============================================================================================
def solve_sheet(path, dev, frames, stride, break_load=None, kn=5.0, xi=0.0):
    """Run 06c and keep every `stride`-th TISSUE frame. The kept tuple is 05b's own -- positions, the
    per-face geometric stretch, the epithelium, the bound plaques' nodes and their attachment points --
    so nothing is recomputed here that the rig did not already record.

    `break_load` GIVES THE PLAQUES A BREAKING STRENGTH, which is 05b's own `Plaques.rupture` and is off
    (None) in 06. With it set, the count of bound plaques falls frame by frame -- so the store is
    RAGGED, like the tearing and the refining ones. It was stacked here, with an assert to stop a rig
    that changes size from being drawn at frame 0's shape; the assert did its job and the answer to it
    is the ragged layout, not a rig that is forbidden to detach.
    """
    T = 2.0e-3
    P = dict(subdiv=4, subdiv_epi=3, E=400.0, thickness=T, nu=0.3, kn=kn, xi=xi,
             l0=0.3 * T, zeta=20.0, s_target=1.0, k_drive=50.0, dev=dev,
             break_load=break_load)
    keep = set(range(0, frames, stride)) | {frames - 1}
    rig = R6.Rig06c(**P)
    t0 = time.time()
    kept, reached = B.run(rig, frames, keep=keep, label="06c")
    if reached != frames:
        raise SystemExit(f"[06] the sheet DIVERGED at frame {reached} of {frames} -- no panel")

    from spec_06 import write_spec
    write_spec(os.path.dirname(path), rig, name=os.path.basename(os.path.dirname(path)),
               frames=frames, matrix_src=SRC,
               extra=dict(kind="mechanical", break_load=break_load, stride=stride,
                          lam_geo_end=rig.res["lam_geo"][-1],
                          momentum_max=max(rig.res["momentum"]),
                          bound_end=rig.res["bound"][-1], standoff_end=rig.res["standoff"][-1]))
    F = rig.sheet.Fc.cpu().numpy().astype(np.int32)
    store = {}
    for i, (t, X, L, XE, nod, PP) in enumerate(kept):
        store[f"t{i}"] = np.int32(t)
        store[f"x{i}"] = X.astype(np.float32)
        store[f"f{i}"] = F
        store[f"v{i}"] = L.astype(np.float32)
        store[f"e{i}"] = XE.astype(np.float32)
        store[f"n{i}"] = nod.astype(np.int32)
        store[f"p{i}"] = PP.astype(np.float32)
    np.savez_compressed(
        path, n_kept=np.int32(len(kept)),
        FE=rig.F_epi.cpu().numpy().astype(np.int32),
        # THE MAP BACK TO TISSUE UNITS, which is what makes this panel comparable with the other three.
        # 06c solves in box units, at its own scale `R0_box / r_apical(0)`; the panels draw in the
        # cache's own tissue coordinates. Storing the scale means the panel converts rather than
        # rescales -- the same refusal `run_ecm.render` states for the matrix.
        centre=rig.c.float().cpu().numpy(),
        scale=np.float64(rig.scale),
        lam_geo=np.asarray(rig.res["lam_geo"], np.float64),
        momentum=np.asarray(rig.res["momentum"], np.float64),
        bound=np.asarray(rig.res["bound"], np.float64),
        load_p50=np.asarray(rig.res["load_p50"], np.float64),
        load_p99=np.asarray(rig.res["load_p99"], np.float64),
        standoff=np.asarray(rig.res["standoff"], np.float64), **store)
    print(f"[06] sheet: {len(kept)} frames kept of {frames} in {time.time()-t0:.0f}s -> "
          f"{path}\n[06]   lam_geo {rig.res['lam_geo'][-1]:.4f}, momentum "
          f"{max(rig.res['momentum']):.2e}, bound {rig.res['bound'][-1]*100:.1f}%, standoff "
          f"{rig.res['standoff'][-1]:+.3e}, plaque load p50 {rig.res['load_p50'][0]:.3e} -> "
          f"{rig.res['load_p50'][-1]:.3e} (p99 {rig.res['load_p99'][-1]:.3e})", flush=True)


# =============================================================================================
#  the panel
# =============================================================================================
class BMPanel:
    """`bm_draw` for `run_ecm.render`: called with the PASS-2 frame and the box its neighbours use,
    draws the sheet at the tissue frame that pass-2 frame acts on, in TISSUE UNITS.

    ONE FIXED COLOUR SCALE, and the box comes from the caller. Both are the convention the other three
    panels were already fixed to: a scale or a camera recomputed per frame turns the growth this run is
    about into a constant picture, and a panel that frames itself puts one tissue at two sizes in one
    figure."""

    FIELD = {"lam": "$\\lambda^{\\rm geo}$", "mt1": "MT1-MMP"}

    def __init__(self, path, mesh_frames, stride, mode="lam", name="basement membrane (06c)"):
        z = np.load(path)
        self.c, self.scale = np.asarray(z["centre"], float), float(z["scale"])
        self.FE = z["FE"]
        # TWO STORE LAYOUTS, ONE PANEL. A sheet that only stretches keeps one face list and one plaque
        # set for the whole run, so 06 stacks them; a sheet that TEARS loses faces and plaques as it
        # goes, and stacking those would need padding -- which puts dead faces back in the picture, the
        # exact defect `sheet.live` produced when it was read as a torn count. The breach store is
        # therefore ragged, one key per kept frame, and the panel reads whichever it is given.
        if "n_kept" in z.files:
            n = int(z["n_kept"])
            self.t = np.array([int(z[f"t{i}"]) for i in range(n)])
            self.XE = [self._tis(z[f"e{i}"]) for i in range(n)]
            self.PP = [self._tis(z[f"p{i}"]) for i in range(n)]
            self.L = [z[f"v{i}"] for i in range(n)]
            # THE RESERVE IS NOT THE SHEET. `bm_secrete` carries an unsecreted pool inside the same node
            # array, parked at the box origin, and `sheet.Fc` simply does not index it. Those nodes draw
            # nothing -- but they are in every statistic taken over `x`, and they put frame 0's mean
            # sheet radius at 95.5 tissue units against a real 4.7, which is the number the inset's limb
            # and its window are found from. So the panel keeps only what a live face or a live plaque
            # refers to, and renumbers to match. It is the same lesson as `(~live).sum()` counting a
            # reservoir as torn faces: a mask over a pool is not a measurement of the thing in it.
            self.X, self.F, self.nod = [], [], []
            for i in range(n):
                x, f, nd = self._tis(z[f"x{i}"]), z[f"f{i}"], z[f"n{i}"]
                used = np.unique(np.concatenate([f.reshape(-1), nd]) if nd.size else f.reshape(-1))
                remap = np.full(x.shape[0], -1, np.int64)
                remap[used] = np.arange(used.size)
                self.X.append(x[used]); self.F.append(remap[f]); self.nod.append(remap[nd])
        else:
            n = len(z["frames"])
            self.t = np.asarray(z["frames"])             # TISSUE frames the sheet was kept at
            self.X, self.XE = list(self._tis(z["X"])), list(self._tis(z["XE"]))
            self.PP, self.L = list(self._tis(z["PP"])), list(z["L"])
            self.nod, self.F = list(z["nod"]), [z["F"]] * n
        self.mesh_frames, self.stride = np.asarray(mesh_frames), max(1, int(stride))
        self.mode, self.name, self.field = mode, name, self.FIELD[mode]
        # THE FULL SCALE IS THE RUN'S MAXIMUM, NOT ITS p99. lam_geo here is not a fluctuating field, it
        # is a ramp: it climbs from 1 to 4.4 over 401 frames, so the p99 of every face at every frame
        # (3.69) is a MID-RUN value and every frame past about 350 renders saturated -- a flat white
        # ball for the last fifth of the movie, which is where the run is most stretched. The price is
        # that the first frames sit at the dark end of the ramp, which is what they are.
        self.vmax = max(float(np.max(v)) for v in self.L if v.size)
        # THE INSET'S WINDOW, MEASURED ONCE FROM THE THING THE INSET IS ABOUT. The gap a plaque spans is
        # the standoff; at 20x it the two bodies are a legible distance apart and the window still holds
        # several nodes of the sheet. Measured over the WHOLE run and then fixed, so the window is a
        # constant length and the surface grows across it -- a window re-measured per frame would hold
        # the standoff at a constant apparent size, which is the one thing G46 is arguing about.
        gap = [np.linalg.norm(x[nd] - pp, axis=-1) if len(nd) else np.zeros(1)
               for x, nd, pp in zip(self.X, self.nod, self.PP)]
        # OFF THE LAST FRAME, not off the run. The gap is not constant -- the sheet is dragged outward
        # as the tissue grows -- so the run's median is a window sized for the middle of the run, and by
        # the end the two bodies are further apart than it and the section runs off the top of it. The
        # final frame's gap is the largest the window has to hold, so a window sized on it holds every
        # frame; the earlier ones simply show a tighter pair.
        self.win = 3.0 * float(np.median(gap[-1]))
        print(f"[06] BM panel: {len(self.t)} sheet frames, {self.F[0].shape[0]} -> "
              f"{self.F[-1].shape[0]} faces, {len(self.nod[0])} -> {len(self.nod[-1])} plaques; "
              f"{self.mode} colour full-scale {self.vmax:.4g} (the run's max, fixed); sheet radius "
              f"{np.linalg.norm(self.X[0], axis=-1).mean():.2f} -> "
              f"{np.linalg.norm(self.X[-1], axis=-1).mean():.2f} tissue units; plaque gap median "
              f"{np.median(gap[0]):.4f} -> {np.median(gap[-1]):.4f} -> inset window "
              f"+-{self.win:.3f} tissue units (fixed, off the last frame)",
              flush=True)

    def _tis(self, A):
        """box units -> the cache's own tissue coordinates, which is what the panels draw in."""
        return (np.asarray(A, np.float64) - self.c) / self.scale

    def tissue_frame(self, t):
        k = min(int(t) // self.stride, len(self.mesh_frames) - 1)
        return int(self.mesh_frames[k])

    def __call__(self, ax, t, L):
        f = self.tissue_frame(t)
        k = int(np.argmin(np.abs(self.t - f)))
        # THE LABEL IS THE PANEL'S OWN NUMBERS, PER FRAME -- what is drawn (faces, plaques) and the
        # range of the field it is coloured by. The colour scale is fixed over the run, so the range
        # printed here is the only thing that says where in that scale THIS frame sits; without it a
        # panel whose every face has saturated looks the same as one where none has.
        v, F, nd = self.L[k], self.F[k], self.nod[k]
        rng = f"{v.min():.2f}-{v.max():.2f}" if v.size else "gone"
        st = bm_panel.plaque_stride(len(nd))
        lab = (f"{self.name}   {F.shape[0]} triangles   {len(nd)} plaques"
               f"{f' (1 in {st} drawn)' if st > 1 else ''}   {self.field} {rng}")
        if not F.shape[0]:
            # NOTHING LEFT IS ALSO A FRAME. The tear run ends with no faces at all, and a drawer that
            # returned early would leave the previous frame's sheet on the axis -- a movie in which the
            # membrane survives its own destruction.
            ax.set_facecolor("black"); ax.axis("off")
            ax.text2D(0.98, 0.96, lab, transform=ax.transAxes, color="white", fontsize=9,
                      ha="right", va="top")
            return
        bm_panel.draw_bm(ax, self.X[k], F, v, self.PP[k], np.zeros(3), float(L),
                         mode=self.mode, vmax=self.vmax, XE=self.XE[k], FE=self.FE,
                         label=lab, x_node=self.X[k][nd], win=self.win, **CAM)


# =============================================================================================
def main():
    def arg(flag, cast, default):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 401)
    stride = arg("--stride", int, 2)                     # every 2nd TISSUE frame: the mesh cadence
    name = arg("--name", str, NAME)
    src = os.path.join(B.LOG, arg("--src", str, SRC))
    d = os.path.join(B.LOG, name)
    os.makedirs(d, exist_ok=True)
    store = os.path.join(d, STORE)

    if not os.path.exists(os.path.join(src, "traj.npz")):
        raise SystemExit(f"[06] {src} has no traj.npz -- the matrix half has to be solved first")
    if "--reuse" in sys.argv and os.path.exists(store):
        print(f"[06] reusing the solved sheet in {store}", flush=True)
    else:
        solve_sheet(store, dev, frames, stride, break_load=arg("--break-load", float, None),
                    kn=arg("--kn", float, 5.0), xi=arg("--xi", float, 0.0))

    # the pass-2 -> tissue frame map, read from the SAME cache and the SAME `mesh_stride` the matrix ran
    # against, so the two halves cannot drift apart in the picture without drifting apart in the spec.
    import yaml
    spec = yaml.safe_load(open(os.path.join(src, "spec_run.yaml")))
    op = next(o for o in spec["operators"] if o["op"] == "mesh_contact")
    mf = np.asarray(np.load(op["tissue"].replace("/groups/saalfeld/home/allierc/Graph", "/workspace"),
                            mmap_mode="r")["mesh_frames"])
    st = int(op.get("mesh_stride", 1))
    # THE ABSENCE OF BM-ECM COUPLING IS NOT IN THE PANEL'S LABEL. It is a property of the RUN, not of
    # the frame, so it belongs in this file's docstring and in the folder's `what.yaml` -- a caption
    # repeated on all 200 frames spends the panel's only line on a constant.
    panel = BMPanel(store, mf, st, name=name)

    # `--frame-limit` CLIPS BOTH HALVES, not one. It exists for the smoke run, where the sheet is solved
    # for a few frames only; without it the still would be the matrix at frame 398 beside the sheet at
    # the last frame it happens to have, which is two different times in one picture.
    kw = {}
    if "--frame-limit" in sys.argv:
        kw["frame_limit"] = arg("--frame-limit", int, 0)
    if "--no-movie" in sys.argv:
        kw["movie"] = False
    import run_ecm
    run_ecm.rerender(src, dest=d, movie_frames=arg("--movie-frames", int, 200),
                     fps=arg("--fps", int, 20), bm_draw=panel, **kw)
    print(f"[06] -> {d}", flush=True)


if __name__ == "__main__":
    main()
