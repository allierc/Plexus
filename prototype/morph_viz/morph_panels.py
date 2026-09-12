"""morph_panels -- a 2x2 movie of a finished morph: the glass render, the learned rest length, the
realised deformation and the stress, frame by frame, from the same camera.

  A  the glass render already in movie.mp4
  B  REST LENGTH LEARNED: the accumulated growth instruction per material point, det(expm(-t A_p)) =
     exp(-t tr A_p) -- the factor by which the point's rest volume has been re-labelled by frame t
     (1 = untouched; > 1 grown; < 1 shrunk). Constant field, switched on progressively.
  C  DEFORMATION REALISED: J = det F per point, the ratio of current to rest volume the elastic solve
     actually produced (1 = at rest).
  D  STRESS: the Frobenius norm of the Cauchy stress per point (spec stress units), what the
     mismatch between B and C costs.

The rollout is re-run from the saved control and the saved t = 0 positions (morph.npz), with
`store_stress: true` on the torch body of mpm_scatter (the warp body does not cache stress).
Nothing under src/plexus is edited; plexus.morph is imported for its spec, weights and camera.

    python morph_panels.py /groups/.../si_material/morph_armadillo [--every 2]
"""
import argparse, json, os, sys, time
import numpy as np, torch

ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--every", type=int, default=2)
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--px", type=int, default=550); ap.add_argument("--fps", type=int, default=20)
args = ap.parse_args()
from plexus import morph as MO, engine, operators  # noqa: F401
from plexus.schema import load
import yaml, tempfile, pyvista as pv, imageio_ffmpeg
pv.OFF_SCREEN = True

z = np.load(os.path.join(args.folder, "morph.npz")); meta = json.loads(str(z["meta"]))
frames_saved, control, box = z["frames"], z["control"], float(z["box"])
T, N = frames_saved.shape[:2]; C0, R0 = meta["extent"][0], meta["extent"][3]; K = meta["ctrl_K"]; DOF = meta["control_dof"]
dev = torch.device(args.device)
X0n = frames_saved[0].astype(np.float64); c = np.array([C0, C0, C0])
dt = meta["dt"] * meta["opt_frames"] / meta["render_frames"]          # morph's final-rollout frame dt

# ---- the per-point growth instruction A_p, exactly as morph derives it ------------------------
idx, wts = MO.trilinear(X0n, c, R0, K, dev, torch)
theta = torch.as_tensor(control, device=dev)
a = sum(w[:, None] * theta[i] for i, w in zip(idx, wts))
if DOF == 9:
    A = a.reshape(-1, 3, 3)
else:
    A = torch.zeros(N, 3, 3, device=dev) + torch.diag_embed(a[:, :3])
    A[:, 0, 1] = A[:, 1, 0] = a[:, 3]; A[:, 0, 2] = A[:, 2, 0] = a[:, 4]; A[:, 1, 2] = A[:, 2, 1] = a[:, 5]
trA = torch.diagonal(A, dim1=1, dim2=2).sum(1)                        # tr A_p per point
eye = torch.eye(3, device=dev); Adt = -A * dt; G = eye + Adt + 0.5 * (Adt @ Adt)

# ---- the rollout, recording J and |sigma| per frame ------------------------------------------
raw = MO.spec(N, box, meta["n_grid"], T - 1, meta["dt"], 3.4e-4, differentiable=False)
for o in raw["operators"]:
    if o["op"].startswith("mpm_"):
        o["implementation"] = "default"      # ONE family of bodies: the warp grid step reads the warp
    if o["op"] == "mpm_scatter":             # scatter's buffers, so mixing warp and torch blows up
        o["store_stress"] = True             # (measured: J nan, particles across the box); the torch
for st in raw["schedule"]:                   # scatter is the one that caches the stress
    if isinstance(st, dict):
        st["compile"] = False
f = os.path.join(tempfile.mkdtemp(prefix="morph_panels_"), "spec.yaml"); yaml.safe_dump(raw, open(f, "w"), sort_keys=False)
sim = load(f)                      # the ENGINE keeps the spec's frame dt (0.002, 6 substeps); only the
                                   # injected growth uses the reduced dt -- exactly morph's final rollout
X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
J_t, S_t, P_t = [], [], []
def cb(H, tick):
    q = H.level("mpm_particle")
    if tick == 0:
        q0, q1 = q.state_schema["pos"]
        with torch.no_grad():
            q.state[:, q0:q1] = X0
    else:
        q.F = torch.bmm(G, q.F)
    J_t.append(torch.linalg.det(q.F).detach().cpu().numpy())
    sig = getattr(q, "sigma", None)
    S_t.append((torch.linalg.norm(sig.reshape(N, -1), dim=1) if sig is not None else torch.zeros(N, device=dev)).detach().cpu().numpy())
    P_t.append(q.get("pos").detach().cpu().numpy().copy())
t0 = time.time()
with torch.no_grad():
    engine.run(sim, device=args.device, on_frame=cb, progress=False, grad=False)
print(f"  rollout {len(P_t)} frames in {time.time()-t0:.0f} s; max |pos - saved| {np.abs(np.stack(P_t) - frames_saved[:len(P_t)]).max():.2e}", flush=True)

# ---- the four panels --------------------------------------------------------------------------
trA_np = trA.cpu().numpy()
growth_final = np.exp(-(T - 1) * dt * trA_np)
lo_g, hi_g = np.percentile(growth_final, (5, 95)); span = min(max(abs(np.log(lo_g)), abs(np.log(hi_g))), 1.5)   # a few points are re-labelled 100x; the map is about the many
print(f"  control theta: |theta| median {np.median(np.abs(control)):.2f} max {np.abs(control).max():.2f}; final rest-volume factor p5/p50/p95 "
      f"{lo_g:.2f} / {np.median(growth_final):.2f} / {hi_g:.2f}", flush=True)
J_all = np.stack(J_t); S_all = np.stack(S_t)
spanJ = min(max(np.nanpercentile(np.abs(np.log(np.clip(J_all, 1e-3, None))), 95), 0.05), 1.0)
s_hi = np.nanpercentile(S_all, 95)
print(f"  realised J p5/p50/p95 at the end {np.percentile(J_all[-1], [5, 50, 95]).round(3)}; |sigma| p50/p95 {np.percentile(S_all[-1], [50, 95]).round(3)}", flush=True)
cam = MO.camera_for(box)
reader = imageio_ffmpeg.read_frames(os.path.join(args.folder, "movie.mp4")); vmeta = next(reader)
W, Hh = vmeta["size"]

def cloud(pts, scal, title, cmap, clim, label):
    pl = pv.Plotter(off_screen=True, window_size=(args.px, args.px)); pl.set_background("black")
    pd = pv.PolyData(pts.astype(np.float32)); pd["v"] = scal
    pl.add_mesh(pd, scalars="v", cmap=cmap, clim=clim, point_size=3, render_points_as_spheres=True,
                scalar_bar_args=dict(title=label, color="white", title_font_size=12, label_font_size=10, n_labels=3, width=0.5, position_x=0.25))
    pl.add_mesh(pv.Cube(bounds=(0, box, 0, box, 0, box)), style="wireframe", color="#2a3a4a", line_width=1)   # the same box as the glass render
    pl.add_text(title, position="upper_left", font_size=9, color="#c8d4e0"); pl.camera_position = cam
    img = pl.screenshot(return_img=True); pl.close(); return img

def fit(img, px):
    from PIL import Image
    return np.asarray(Image.fromarray(img[..., :3]).resize((px, px), Image.BILINEAR))

out = os.path.join(args.folder, "panels_2x2.mp4")
writer = imageio_ffmpeg.write_frames(out, (2 * args.px, 2 * args.px), fps=args.fps, quality=7); writer.send(None)
t0 = time.time(); n_done = 0
for i, glass in enumerate(reader):
    if i >= len(P_t):
        break
    if i % args.every:
        continue
    glass = np.frombuffer(glass, np.uint8).reshape(Hh, W, 3)
    growth = np.exp(-i * dt * trA_np)
    A_img = fit(glass, args.px)
    B_img = cloud(P_t[i], np.log(growth), f"B  rest length learned: log growth factor exp(-t tr A), frame {i}", "PuOr", (-span, span), "log rest-volume factor")
    C_img = cloud(P_t[i], np.log(np.clip(J_all[i], 1e-3, None)), f"C  deformation realised: log J = log det F, frame {i}", "PuOr", (-spanJ, spanJ), "log J")
    D_img = cloud(P_t[i], S_all[i], f"D  Cauchy stress |sigma|, frame {i}", "magma", (0, s_hi), "|sigma| (spec units)")
    top = np.concatenate([A_img, B_img[..., :3]], 1); bot = np.concatenate([C_img[..., :3], D_img[..., :3]], 1)
    writer.send(np.ascontiguousarray(np.concatenate([top, bot], 0))); n_done += 1
writer.close()
print(f"  {n_done} frames -> {out}  ({(time.time()-t0)/max(n_done,1):.1f} s a frame)")
