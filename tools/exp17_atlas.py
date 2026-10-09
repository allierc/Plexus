"""exp17: THE DESTRIPED NEURONS IN THE Z-BRAIN ATLAS (Cedric, 2026-10-07: "I have made the BigWarp registration,
experiments/exp17_zapbench_graphcast/landmarks.csv; for convenience, head left, I rotated the reference tif to the left").
Z-Brain: Randlett et al. 2015, Nature Methods 12, 1039 ("Whole-brain activity mapping onto a zebrafish brain atlas").

THE CHAIN. The destriped neurons' positions (zapbench_destripe_recording pos_um) are in the anatomy volume's frame, um,
x y z -- BigWarp's MOVING frame (anatomy_t0_v2.ome.zarr, 0.406 x 0.406 x 1 um). The landmarks' FIXED side is the Z-Brain
reference (0.798 x 0.798 x 2 um, 621 x 1406 x 138 voxels, head up) after Fiji's "Rotate 90 Degrees Left": a rotated
pixel (x', y') is the original (W - 1 - y', x'), W = 621. The thin-plate spline of tools/exp20_bigwarp.py (BigWarp's own
kernel, r^2 log r) carries a neuron to the rotated reference; the rotation is undone; each neuron then gets the Z-Brain
regions its voxel lies in (MaskDatabase.mat, 294 overlapping masks).
CHECKS: the leave-one-out landmark error, and the share of the neurons that land inside the Z-Brain brain (the union of
the masks), for the left rotation and, as a control, the right one.

    PYTHONPATH=src:tools python tools/exp17_atlas.py                       # the mapping
    PYTHONPATH=src:tools python tools/exp17_atlas.py summary [--iso|--merged]   # the per-region figure
-> data/atlas_destripe.npz (atlas um, inside, regions [N, 294] bool, names), data/atlas_destripe.json,
   presentation/figs/atlas_regions{,_iso,_merged}.png, data/atlas_regions.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
ZB = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast", "data", "atlas", "zbrain", "Additional_mat_files")
ZVOX = (0.798, 0.798, 2.0)


def unrotate(A, W, rot):
    """Rotated-reference um -> original Z-Brain um. rot 'left': x = (W-1) vox - y', y = x'; 'right': x = y', y = (H-1) - x'."""
    out = A.copy()
    if rot == "left":
        out[:, 0] = (W - 1) * ZVOX[0] - A[:, 1]
        out[:, 1] = A[:, 0]
    else:
        out[:, 0] = A[:, 1]
        out[:, 1] = (1406 - 1) * ZVOX[1] - A[:, 0]
    return out


def main():
    import h5py
    from scipy.sparse import csc_matrix
    from exp20_bigwarp import read_landmarks, tps_apply, tps_fit
    from plexus.paths import graphs_data_path
    mv, fx = read_landmarks(os.path.join(EXP, "landmarks.csv"))
    ok = np.isfinite(mv).all(1) & np.isfinite(fx).all(1)
    mv, fx = mv[ok], fx[ok]
    model = tps_fit(mv, fx)
    loo = [float(np.linalg.norm(tps_apply(tps_fit(np.delete(mv, k, 0), np.delete(fx, k, 0)), mv[k:k + 1])[0] - fx[k]))
           for k in range(len(mv))]
    P = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))["pos_um"].astype(np.float64)
    Ar = tps_apply(model, P)
    with h5py.File(os.path.join(ZB, "MaskDatabase.mat")) as h:
        H, W, Z = int(h["height"][0, 0]), int(h["width"][0, 0]), int(h["Zs"][0, 0])
        names = ["".join(chr(c) for c in h[r][:].ravel()) for r in h["MaskDatabaseNames"][:, 0]]
        g = h["MaskDatabase"]
        S = csc_matrix((g["data"][:], g["ir"][:], g["jc"][:]), shape=(H * W * Z, len(names))).tocsr()
    res = {}
    for rot in ("left", "right"):
        A = unrotate(Ar, W, rot)
        ix = np.round(A[:, 0] / ZVOX[0]).astype(int)
        iy = np.round(A[:, 1] / ZVOX[1]).astype(int)
        iz = np.round(A[:, 2] / ZVOX[2]).astype(int)
        box = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (iz >= 0) & (iz < Z)
        lin = np.clip(iy, 0, H - 1) + H * np.clip(ix, 0, W - 1) + H * W * np.clip(iz, 0, Z - 1)   # column-major (y, x, z)
        reg = S[lin].astype(bool).toarray() & box[:, None]
        inside = reg.any(1)
        res[rot] = (A, inside, reg)
        print(f"[atlas] rotation {rot}: {100 * box.mean():.1f} % in the atlas box, {100 * inside.mean():.1f} % inside the brain")
    rot = max(res, key=lambda r: res[r][1].mean())
    A, inside, reg = res[rot]
    np.savez_compressed(os.path.join(EXP, "data", "atlas_destripe.npz"), atlas_um=A.astype(np.float32), inside=inside,
                        regions=reg, names=np.array(names))
    cnt = reg.sum(0)
    doc = {"landmarks": int(len(mv)), "loo_um_median": float(np.median(loo)), "loo_um_max": float(np.max(loo)),
           "rotation_used": rot, "inside_share": {r: float(res[r][1].mean()) for r in res}, "neurons": int(len(P)),
           "regions_nonempty": int((cnt > 0).sum()),
           "top_regions": [(names[i], int(cnt[i])) for i in np.argsort(cnt)[::-1][:25]]}
    json.dump(doc, open(os.path.join(EXP, "data", "atlas_destripe.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "top_regions"}, indent=1))
    for n, c in doc["top_regions"][:15]:
        print(f"   {c:7,d}  {n}")


# the regions of the table, anatomical (no transgene masks), head to tail (Cedric, 2026-10-07: "work with the atlas now")
REGIONS = [("Telencephalon - Olfactory Bulb", "olfactory bulb"), ("Telencephalon - Pallium", "pallium"),
           ("Telencephalon - Subpallium", "subpallium"), ("Diencephalon - Habenula", "habenula"),
           ("Diencephalon - Dorsal Thalamus", "dorsal thalamus"), ("Diencephalon - Ventral Thalamus", "ventral thalamus"),
           ("Diencephalon - Pretectum", "pretectum"), ("Diencephalon - Preoptic Area", "preoptic area"),
           ("Diencephalon - Intermediate Hypothalamus", "hypothalamus (intermediate)"),
           ("Diencephalon - Posterior Tuberculum", "posterior tuberculum"),
           ("Mesencephalon - Tectum Stratum Periventriculare", "tectum (periventricular)"),
           ("Mesencephalon - Torus Semicircularis", "torus semicircularis"), ("Mesencephalon - Tegmentum", "tegmentum"),
           ("Rhombencephalon - Cerebellum", "cerebellum"), ("Rhombencephalon - Rhombomere 1", "rhombomere 1"),
           ("Rhombencephalon - Rhombomere 2", "rhombomere 2"), ("Rhombencephalon - Rhombomere 3", "rhombomere 3"),
           ("Rhombencephalon - Rhombomere 4", "rhombomere 4"), ("Rhombencephalon - Rhombomere 5", "rhombomere 5"),
           ("Rhombencephalon - Rhombomere 6", "rhombomere 6"), ("Rhombencephalon - Rhombomere 7", "rhombomere 7"),
           ("Rhombencephalon - Noradrendergic neurons of the Interfascicular and Vagal areas", "noradrenergic (IFN, vagal)"),
           ("Rhombencephalon - X Vagus motorneuron cluster", "vagus motor neurons"), ("Spinal Cord", "spinal cord")]


# THE TABLE REGIONS IN PLAIN WORDS (Cedric, 2026-10-08: "explain each region", then "expand a bit, + a reference in grey
# small font"): what each region holds and does in the larval zebrafish, one reference each (REGION_REF). The atlas itself
# is Randlett et al. 2015 (Nature Methods 12:1039). Kawashima 2016, Mu 2019 and Daie 2015 are in the experiment's papers/;
# the others are cited from memory, not yet checked against their PDFs.
REGION_ROLE = {
    "olfactory bulb": "the first relay of smell: its mitral cells receive the nose's sensory neurons and project to the "
                      "pallium (Dp, the olfactory cortex), the habenula and the hypothalamus",
    "pallium": "dorsal forebrain, everted in fish: homologue of the cortex -- Dl of the hippocampus, Dm of the amygdala, "
               "Dp of the olfactory cortex: learning, memory, valence",
    "subpallium": "ventral forebrain, mostly GABAergic: homologue of the striatum and septum (basal ganglia), fed "
                  "dopamine by the posterior tuberculum: action selection",
    "habenula": "dorsal diencephalon, left / right asymmetric: relays the forebrain, smell and light to the "
                "interpeduncular nucleus and the raphe: fear, aversion, passive coping",
    "dorsal thalamus": "the thalamus proper: relays sensory input (vision, light level) to the pallium and the habenula",
    "ventral thalamus": "the prethalamus, GABAergic, homologue of the reticular thalamic nucleus and the zona incerta: "
                        "gates the thalamic relay",
    "pretectum": "between thalamus and tectum, fed directly by the retina: direction-selective optic-flow neurons drive "
                 "the optomotor and optokinetic responses; one retinal target (AF7) detects prey",
    "preoptic area": "in front of the optic chiasm; neurosecretory cells (oxytocin, vasopressin, CRH) driving the "
                     "pituitary: stress axis, water and temperature balance",
    "hypothalamus (intermediate)": "the hypothalamus's middle: feeding and hunger (prey in view activates it), arousal; "
                                   "neuroendocrine output to the pituitary",
    "posterior tuberculum": "ventral diencephalon: the dopaminergic neurons (A11-like) that project down to the hindbrain "
                            "and spinal cord and up to the subpallium",
    "tectum (periventricular)": "the optic tectum's cell layer, homologue of the superior colliculus and the retina's main "
                                "target: a visual map of space; detects prey and looming threats, drives hunting and escape",
    "torus semicircularis": "below the tectum, homologue of the inferior colliculus: hearing and the lateral line "
                            "(water flow)",
    "tegmentum": "ventral midbrain: the nMLF (nucleus of the medial longitudinal fasciculus), which sets swim speed and "
                 "posture, and the oculomotor nucleus (III, the eye muscles)",
    "cerebellum": "Purkinje and granule cells, climbing fibres from the inferior olive: motor coordination and learning, "
                  "e.g. adapting the swim to its visual feedback",
    "rhombomere 1": "the hindbrain's first segment, under the cerebellum: the locus coeruleus (noradrenaline) and the "
                    "dorsal raphe (serotonin), which tracks the outcome of swims for short-term motor learning",
    "rhombomere 2": "the ARTR's front (anterior rhombencephalic turning region): left and right populations in "
                    "antiphase that bias the turn direction and alternate every 10-20 s; reticulospinal neurons",
    "rhombomere 3": "the ARTR's back; the trigeminal motor nucleus (jaw) and reticulospinal neurons",
    "rhombomere 4": "the two Mauthner cells, one per side: a single spike launches the fast escape (the C-start)",
    "rhombomere 5": "the abducens motor nucleus (eye, with r6) and the saccade generator; the vestibular tangential "
                    "nucleus: gravity-driven eye movements",
    "rhombomere 6": "the abducens nucleus's back; reticulospinal neurons of the escape network (MiD3, a Mauthner "
                    "homologue); facial motor neurons",
    "rhombomere 7": "caudal hindbrain: the oculomotor integrator that holds the eyes still between saccades (with r8); "
                    "V2a reticulospinal neurons of the swim command",
    "noradrenergic (IFN, vagal)": "the noradrenaline cluster of the medulla (NE-MO, like the mammalian A2): it signals "
                                  "failed swims to radial astrocytes, which switch the fish to passivity",
    "vagus motor neurons": "the vagal motor nucleus (X), the last branchiomotor nucleus: muscles of the gill arches and "
                           "the pharynx, heart rate, the gut",
    "spinal cord": "motor neurons and the interneurons of the swim rhythm, recruited in order of swim speed; the "
                   "mechanosensory Rohon-Beard neurons"}
REGION_REF = {"olfactory bulb": "Yaksi et al. 2009, Nat Neurosci",
              "pallium": "Wullimann & Mueller 2004, J Comp Neurol",
              "subpallium": "Wullimann & Mueller 2004, J Comp Neurol; Rink & Wullimann 2001, Brain Res",
              "habenula": "Agetsuma et al. 2010, Nat Neurosci; Andalman et al. 2019, Cell",
              "dorsal thalamus": "Mueller 2012, Front Neural Circuits",
              "ventral thalamus": "Mueller 2012, Front Neural Circuits",
              "pretectum": "Kubo et al. 2014, Neuron; Naumann et al. 2016, Cell; Semmelhack et al. 2014, eLife",
              "preoptic area": "Herget et al. 2014, J Comp Neurol",
              "hypothalamus (intermediate)": "Muto et al. 2017, Nat Commun; Wee et al. 2019, eLife",
              "posterior tuberculum": "Tay et al. 2011, Nat Commun; Rink & Wullimann 2001, Brain Res",
              "tectum (periventricular)": "Bianco & Engert 2015, Curr Biol; Temizer et al. 2015, Curr Biol",
              "torus semicircularis": "Privat et al. 2019, Curr Biol",
              "tegmentum": "Severi et al. 2014, Neuron; Thiele et al. 2014, Neuron",
              "cerebellum": "Ahrens et al. 2012, Nature",
              "rhombomere 1": "Kawashima et al. 2016, Cell",
              "rhombomere 2": "Dunn et al. 2016, eLife; Wolf et al. 2017, Nat Commun",
              "rhombomere 3": "Dunn et al. 2016, eLife; Higashijima et al. 2000, J Neurosci",
              "rhombomere 4": "Korn & Faber 2005, Neuron",
              "rhombomere 5": "Schoonheim et al. 2010, J Neurosci; Bianco et al. 2012, Curr Biol",
              "rhombomere 6": "Kinkhabwala et al. 2011, PNAS",
              "rhombomere 7": "Daie et al. 2015, Neuron; Kinkhabwala et al. 2011, PNAS",
              "noradrenergic (IFN, vagal)": "Mu et al. 2019, Cell",
              "vagus motor neurons": "Higashijima et al. 2000, J Neurosci",
              "spinal cord": "McLean et al. 2007, Nature"}


_SUB_CACHE = {}


def subregions(full, names, reg, frac=0.8, top=6):
    """The Z-Brain masks lying (>= frac of their voxels) inside the table region `full`: its sub-regions (Randlett et al.
    2015's 294 masks nest: anatomical subdivisions, nuclei and transgene-labelled clusters). Each with its neurons among
    the destriped ones; the `top` largest by neurons. Voxels from MaskDatabase.mat, read once."""
    import h5py
    from scipy.sparse import csc_matrix
    if "S" not in _SUB_CACHE:
        with h5py.File(os.path.join(ZB, "MaskDatabase.mat")) as h:
            H, W, Z = int(h["height"][0, 0]), int(h["width"][0, 0]), int(h["Zs"][0, 0])
            g = h["MaskDatabase"]
            _SUB_CACHE["S"] = csc_matrix((g["data"][:], g["ir"][:], g["jc"][:]), shape=(H * W * Z, len(names)))
            _SUB_CACHE["n"] = H * W * Z
    S = _SUB_CACHE["S"]
    c = names.index(full)
    inside = np.zeros(_SUB_CACHE["n"], bool)
    inside[S.indices[S.indptr[c]:S.indptr[c + 1]]] = True
    table = {f for f, _ in REGIONS}
    out = []
    for j, nm in enumerate(names):
        if j == c or nm in table or nm.count(" - ") == 0:
            continue
        v = S.indices[S.indptr[j]:S.indptr[j + 1]]
        if len(v) and inside[v].mean() >= frac and len(v) < S.indptr[c + 1] - S.indptr[c]:
            out.append({"name": nm.split(" - ", 1)[1], "neurons": int(reg[:, j].sum()), "voxels": int(len(v))})
    out.sort(key=lambda d: -d["neurons"])
    return {"count": len(out), "top": out[:top]}


def iso_images(rlist, rfull, rcol, names, sizes=((1640, 1000), (1800, 620))):
    """THE REGIONS AS SURFACES (Cedric, 2026-10-07: "a twin with VTK isosurfaces, as the Plexus watcher"): each table
    region's Z-Brain mask, halved in x and y, contoured at 0.5 and smoothed, in its table colour, semi-opaque; the brain
    (the union of the division masks) a faint grey shell. Shown head left as the scatter (x' = y, y' = (W-1) dx - x), from
    above and from the side, parallel projection on black, each render the shape of its panel. Returns the two images."""
    import h5py
    import pyvista as pv
    from scipy.sparse import csc_matrix
    pv.OFF_SCREEN = True
    with h5py.File(os.path.join(ZB, "MaskDatabase.mat")) as h:
        H, W, Z = int(h["height"][0, 0]), int(h["width"][0, 0]), int(h["Zs"][0, 0])
        g = h["MaskDatabase"]
        S = csc_matrix((g["data"][:], g["ir"][:], g["jc"][:]), shape=(H * W * Z, len(names)))

    def volume(col):
        v = np.zeros(H * W * Z, np.float32)
        a, b = S.indptr[col], S.indptr[col + 1]
        v[S.indices[a:b]] = 1.0
        return v.reshape((Z, W, H)).transpose(2, 1, 0)[::2, ::2, :]       # column-major (y, x, z) -> [y, x, z], halved

    def surface(vol):
        img = pv.ImageData(dimensions=vol.shape, spacing=(2 * ZVOX[1], 2 * ZVOX[0], ZVOX[2]))
        img.point_data["m"] = vol.ravel(order="F")
        srf = img.contour([0.5], scalars="m")
        if srf.n_points == 0:
            return None
        srf = srf.smooth(n_iter=30)
        P = np.asarray(srf.points)                                      # (y, x, z) um -> head left (x', y', z)
        srf.points = np.column_stack([P[:, 0], (W - 1) * ZVOX[0] - P[:, 1], P[:, 2]])
        return srf
    brain = None
    for key in ("Telencephalon -", "Diencephalon -", "Mesencephalon -", "Rhombencephalon -", "Spinal Cord"):
        if key in names:
            v_ = volume(names.index(key))
            brain = v_ if brain is None else np.maximum(brain, v_)
    out = []
    for view, size in zip(("top", "side"), sizes):
        pl = pv.Plotter(off_screen=True, window_size=size)
        pl.set_background("black")
        sb = surface(brain)
        if sb is not None:
            pl.add_mesh(sb, color="#8c8c8c", opacity=0.10, smooth_shading=True)
        for r_ in rlist:
            sr = surface(volume(names.index(rfull[r_])))
            if sr is not None:
                pl.add_mesh(sr, color=rcol[r_][:3], opacity=0.75, smooth_shading=True, specular=0.3)
        pl.enable_parallel_projection()
        b = sb.bounds if sb is not None else (0, 1, 0, 1, 0, 1)
        c = ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
        if view == "top":
            pl.camera.position = (c[0], c[1], c[2] + 3000.0)
            pl.camera.up = (0.0, 1.0, 0.0)
            pl.camera.parallel_scale = max((b[3] - b[2]) / 2, (b[1] - b[0]) / 2 * size[1] / size[0]) * 1.06
        else:
            pl.camera.position = (c[0], c[1] - 3000.0, c[2])
            pl.camera.up = (0.0, 0.0, 1.0)
            pl.camera.parallel_scale = max((b[5] - b[4]) / 2, (b[1] - b[0]) / 2 * size[1] / size[0]) * 1.06
        pl.camera.focal_point = c
        out.append(pl.screenshot(return_img=True))
        pl.close()
    return out


def summary(iso=False):
    """THE ANALYSES PER REGION: per Z-Brain region, its neurons, their lead over the brain mean, the share of input neurons
    (the balanced 20 % mask), of swim- or turn-tuned neurons, of neurons whose response repeats and of neurons that ramp
    up in open loop (tools/exp17_classic.py). iso False: the neurons as dots; True: the regions as surfaces; "merged":
    one column of four panels, all at ONE scale. -> figs/atlas_regions{,_iso,_merged}.png, data/atlas_regions.json"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    A, reg, names = za["atlas_um"], za["regions"], [str(x) for x in za["names"]]
    zl = np.load(os.path.join(EXP, "data", "brain_mean_lag_zapbench_destripe.npz"))
    lead, wl = -zl["lag_s"], zl["with_lag"]
    zc = np.load(os.path.join(EXP, "data", "classic_per_neuron.npz"))
    gn = [str(x) for x in zc["groups"]]
    motor = zc["tuned"] & np.isin(zc["best"], [gn.index(g) for g in ("swim", "turn left", "turn right")])
    relb = zc["reliable_block"] >= 0
    rampup = zc["ramp"] > 0.5
    inp = np.load(graphs_data_path("zebrafish", "input_mask_destripe_bal20.npz"))["mask"] > 0
    cols = [("neurons", None), ("median lead, s", None), ("lead > 1 s, %", None), ("input neurons, %", None),
            ("swim / turn tuned, %", None), ("response repeats, %", None), ("ramps up in open loop, %", None)]
    rows, out = [], {}
    for full, short in REGIONS:
        if full not in names:
            continue
        m = reg[:, names.index(full)]
        if m.sum() < 50:
            continue
        ml = m & wl
        v = [int(m.sum()), float(np.median(lead[ml])) if ml.sum() > 20 else np.nan,
             100 * float((lead[ml] > 1.0).mean()) if ml.sum() > 20 else np.nan, 100 * float(inp[m].mean()),
             100 * float(motor[m].mean()), 100 * float(relb[m].mean()), 100 * float(rampup[m].mean())]
        rows.append((short, v))
        out[short] = dict(zip([c for c, _ in cols], v))
    json.dump(out, open(os.path.join(EXP, "data", "atlas_regions.json"), "w"), indent=1)
    plt.style.use("dark_background")
    blkv = str(iso).startswith("block:")                 # a block raster beside the fish (Cedric, 2026-10-07)
    mg = iso in ("merged", "merged_raster", "fish") or blkv
    FH = 12.4 if mg else 8.6                  # taller: the slide's free height filled (Cedric, 2026-10-07)
    FW = 21.0 if (iso in ("merged_raster", "fish") or str(iso).startswith("block:")) else 14.0      # wide: fish, a large raster, three panels (Cedric, 2026-10-07)
    fig = plt.figure(figsize=(FW, FH), facecolor="black")
    # the neurons in the atlas, head left (the reference turned as Cedric turned it), each coloured by its TABLE region
    # (Cedric, 2026-10-07: "only 5 regions in the fish, many more in the table"): of the table's regions it lies in, the
    # most specific (fewest neurons); grey the neurons in none
    W = 621
    xd, yd = A[:, 1], (W - 1) * ZVOX[0] - A[:, 0]
    rlist = [short for short, _ in rows]
    rfull = {short: full for full, short in REGIONS}
    pal = list(plt.get_cmap("tab20")(np.arange(20))) + list(plt.get_cmap("Set1")(np.arange(9)))
    rcol = {r_: pal[k % len(pal)] for k, r_ in enumerate(rlist)}
    lab_ = np.full(len(A), -1)
    best_n = np.full(len(A), np.inf)
    for k, r_ in enumerate(rlist):
        m = reg[:, names.index(rfull[r_])]
        n_ = m.sum()
        upd = m & (n_ < best_n)
        lab_[upd], best_n[upd] = k, n_
    ins_ = lab_ >= 0
    if mg:
        # ONE SCALE for the four panels (Cedric, 2026-10-07: "top and side views are not to scale"): um per inch the
        # same everywhere, each panel's height its own extent at that scale, stacked from the top
        ext = lambda v: float(np.percentile(v[ins_], 99.9) - np.percentile(v[ins_], 0.1))     # noqa: E731
        ex, ey, ez = ext(xd), ext(yd), ext(A[:, 2]) + 40.0
        TT = 0.28                                          # a title line, inches
        sc = min((0.30 if (iso in ("merged_raster", "fish") or blkv) else 0.60) * FW / ex,
                 (0.97 * FH - 4 * TT) / (2 * ey + 2 * ez))
        wf = ex * sc / FW
        tot_ = (4 * TT + (2 * ey + 2 * ez) * sc) / FH          # the stack's height, figure fraction
        # centred on the raster's height (Cedric, 2026-10-07: "the fish a bit lower, centred with the raster")
        panels, ycur = [], (min(0.985, 0.54 + tot_ / 2) if (iso in ("merged_raster", "fish") or blkv) else 0.985)
        for kind, view, ttl in (("dots", "top", "from above, head left: the neurons"),
                                ("iso", "top", "from above: the regions as surfaces (atlas)"),
                                ("dots", "side", "from the side: the neurons"),
                                ("iso", "side", "from the side: the regions as surfaces (atlas)")):
            hf = (ey if view == "top" else ez) * sc / FH
            ycur -= TT / FH + hf
            panels.append((kind, [0.01, ycur, wf, hf], view, ttl))
    elif iso:
        panels = [("iso", [0.0, 0.40, 0.58, 0.54], "top", "the regions as surfaces, from above, head left"),
                  ("iso", [0.0, 0.05, 0.58, 0.30], "side", "from the side")]
    else:
        panels = [("dots", [-0.03, 0.37, 0.62, 0.62], "top", "in Z-Brain, from above, head left"),
                  ("dots", [-0.03, 0.02, 0.62, 0.34], "side", "from the side")]

    def trim(im, pad=6):                                  # a render cut to the brain
        on = np.argwhere(im[:, :, :3].max(2) > 12)
        (y0, x0), (y1, x1) = on.min(0), on.max(0)
        return im[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
    imgs = ({k: trim(v) for k, v in zip(("top", "side"), iso_images(rlist, rfull, rcol, names))}
            if any(k == "iso" for k, *_ in panels) else {})
    for kind, rect, view, ttl in panels:
        ax = fig.add_axes(rect)
        ax.axis("off")
        ax.set_title(ttl + ("" if mg else " -- the colours of the table's names"), fontsize=9.5, loc="left",
                     x=0.06, pad=2)
        if kind == "iso":
            ax.imshow(imgs[view], aspect="auto" if mg else "equal")
            continue
        Y = yd if view == "top" else A[:, 2]
        ax.scatter(xd[~ins_], Y[~ins_], s=0.15, color="0.25", lw=0, rasterized=True)
        if view == "side":                                # a 100-um bar under the side view
            yb_ = np.percentile(Y, 0.5) - 25.0
            xb_ = np.percentile(xd, 99.5) - 100.0
            ax.plot([xb_, xb_ + 100.0], [yb_, yb_], color="white", lw=2.0)
            ax.text(xb_ + 50.0, yb_ - 8.0, "100 µm", color="white", fontsize=9, ha="center", va="top")
        for k, r_ in enumerate(rlist):
            m = lab_ == k
            ax.scatter(xd[m], Y[m], s=0.35 if not mg else 0.25, color=rcol[r_], lw=0, rasterized=True)
        if mg:                                                               # the brain's own extent, the panel's
            ax.set_xlim(np.percentile(xd[ins_], 0.1), np.percentile(xd[ins_], 99.9))
            ax.set_ylim(np.percentile(Y[ins_], 0.1) - (40.0 if view == "side" else 0.0), np.percentile(Y[ins_], 99.9))
            ax.set_aspect("auto")
        else:
            ax.set_aspect("equal")
    if isinstance(iso, str) and iso.startswith("block:"):
        _, blk_, lr_ = iso.split(":")
        draw_raster(fig, raster_data(block=blk_, split_lr=lr_ == "lr"), wf + 0.025, 0.985, y0=0.07, y1=0.975,
                    name_w=0.085, fs=7.5)
    elif iso == "merged_raster":
        # right: THE RASTER BY REGION next to the fish (Cedric, 2026-10-07: "put the raster by region next to the fish")
        imr = draw_raster(fig, raster_data(), wf + 0.025, 0.735, y0=0.105, y1=0.975, name_w=0.085, fs=7.5)
        cbx = fig.add_axes([wf + 0.025 + 0.102, 0.045, 0.735 - wf - 0.025 - 0.102, 0.012])      # the grey LUT below
        cbr = fig.colorbar(imr, cax=cbx, orientation="horizontal")
        cbr.set_label("dF/F z-scored over the recording (bin mean)", fontsize=8.5)
        cbr.ax.tick_params(labelsize=7.5)
        import matplotlib.image as mpimg               # beside the raster, the three panels of the brain-mean slide
        pan = [("raster", "every neuron, sorted by correlation with the brain mean"),
               ("spectrum", "the share of the power per frequency"),
               ("swings", "correlation with the brain mean against swings")]
        hs = [0.355, 0.30, 0.30]
        ycur = 0.985
        for (tag_, ttl_), h_ in zip(pan, hs):
            f_ = os.path.join(EXP, "presentation", "figs", f"brain_mean_spread_zapbench_destripe_{tag_}.png")
            ycur -= h_ + 0.01
            if not os.path.exists(f_):
                continue
            a_ = fig.add_axes([0.75, ycur, 0.245, h_ - 0.02])
            a_.imshow(mpimg.imread(f_))
            a_.axis("off")
            a_.set_title(ttl_, fontsize=8.5, loc="left", pad=2)
    elif iso == "fish":
        # THE FISH ALONE (Cedric, 2026-10-08: "slide 3's fishes, and each region's sub-regions of the atlas explained"):
        # the four panels, and each table region's colour and its Z-Brain sub-masks for the slide's table
        SUB = {r_: subregions(rfull[r_], names, reg) for r_ in rlist}
        json.dump({r_: {"colour": [float(c_) for c_ in rcol[r_][:3]], "full": rfull[r_], "neurons": int((lab_ == k).sum()),
                        "subregions": SUB[r_], "role": REGION_ROLE.get(r_, ""), "ref": REGION_REF.get(r_, "")}
                   for k, r_ in enumerate(rlist)},
                  open(os.path.join(EXP, "data", "atlas_subregions.json"), "w"), indent=1)
        # THE TABLE ON SLIDE 3'S CANVAS (Cedric, 2026-10-08: "the fishes aligned to slide 3, the text larger"): where slide 3
        # has its raster, the regions head to tail in two columns, each a block -- its name in its colour, its role, and
        # its Z-Brain sub-masks in grey -- with a blank line between blocks
        import textwrap
        x0_, cw_ = wf + 0.035, (0.985 - wf - 0.035) / 2
        h_ = (len(rlist) + 1) // 2
        blocks_ = []
        for k, r_ in enumerate(rlist):
            tops = [x["name"].strip() for x in SUB[r_]["top"][:2] if x["neurons"] > 0]   # one line (2026-10-08: room for the refs)
            more = SUB[r_]["count"] - len(tops)
            sub = ("atlas: " + ", ".join(tops) + (f" (+{more} more)" if more > 0 else "")) if tops else ""
            blocks_.append((r_, f"{r_} ({int((lab_ == k).sum()):,})", textwrap.fill(REGION_ROLE.get(r_, ""), 66),
                            textwrap.fill(sub, 82) if sub else "", textwrap.fill(REGION_REF.get(r_, ""), 82)))
        # FLOWED top-down from the lines each block holds (no overlap), one font scale so the fuller column fits
        lh = lambda pt: pt * 1.22 / 72 / FH                                  # noqa: E731   one line, figure fraction
        F0 = (16.0, 13.5, 11.5)
        gap = 0.012

        def col_h(bl, f):
            return sum(lh(f * F0[0]) + lh(f * F0[1]) * (r.count("\n") + 1) + (lh(f * F0[2]) * (sb.count("\n") + 1)
                       if sb else 0) + (lh(f * F0[2]) * (rf.count("\n") + 1) if rf else 0) + gap for _, _, r, sb, rf in bl)
        f_ = min(1.0, 0.95 / max(col_h(blocks_[:h_], 1.0), col_h(blocks_[h_:], 1.0)))
        print(f"[atlas] the table's font scale {f_:.3f} of its nominal sizes {F0} pt")
        for c_, bl in enumerate((blocks_[:h_], blocks_[h_:])):
            y = 0.5 + col_h(bl, f_) / 2                                      # centred in height
            cx = x0_ + c_ * cw_
            for r_, nm, role, sb, rf in bl:
                fig.text(cx, y, nm, color=rcol[r_], fontsize=f_ * F0[0], weight="bold", va="top")
                y -= lh(f_ * F0[0])
                fig.text(cx, y, role, color="white", fontsize=f_ * F0[1], va="top", linespacing=1.1)
                y -= lh(f_ * F0[1]) * (role.count("\n") + 1)
                if rf:                                   # the reference: grey, small, italic (Cedric, 2026-10-08)
                    fig.text(cx, y, rf, color="0.55", fontsize=f_ * F0[2], style="italic", va="top", linespacing=1.1)
                    y -= lh(f_ * F0[2]) * (rf.count("\n") + 1)
                if sb:
                    fig.text(cx, y, sb, color="0.6", fontsize=f_ * F0[2], va="top", linespacing=1.1)
                    y -= lh(f_ * F0[2]) * (sb.count("\n") + 1)
                y -= gap
    elif iso == "merged":
        # right: THE REGIONS' MEAN TRACES (Cedric, 2026-10-07: "instead of the table, average the traces of each region,
        # compared to the brain mean in green; the region's name at the right with its neurons in parentheses"): per table
        # region, the mean dF/F of its neurons over the whole recording, z-scored, in its colour, the brain mean z-scored
        # over it in green; the stimulus blocks as bands
        z_ = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
        X_ = z_["dff"]
        off_, nm_b = z_["offsets"], [str(x) for x in z_["names"]]
        T_ = X_.shape[0]
        tm_ = np.arange(T_) * 0.914 / 60
        # Cedric, 2026-10-07: dF/F itself, ONE scale for every row (no z-scoring), the band 90 % of the region's neurons
        # fall in (5th-95th percentile per frame) in transparent grey, the brain mean in green over each
        Xa = np.asarray(X_)
        bm_ = Xa.mean(1, dtype=np.float64)
        STEP = 0.05                       # dF/F between two rows: narrow (Cedric, 2026-10-07), each row about its median
        x0t = wf + 0.07                       # a gap between the fish and the traces (Cedric, 2026-10-07)
        ax = fig.add_axes([x0t, 0.03, 0.985 - x0t - 0.17, 0.90])
        nr = len(rows)
        for k_ in range(len(off_) - 1):
            ax.axvspan(tm_[off_[k_]], tm_[min(off_[k_ + 1], T_ - 1)], color=("0.16" if k_ % 2 else "0.08"), lw=0, zorder=0)
            ab_ = {"turning": "turn", "position": "pos", "open loop": "open", "rotation": "rot"}.get(nm_b[k_], nm_b[k_])
            ax.text((tm_[off_[k_]] + tm_[min(off_[k_ + 1], T_ - 1)]) / 2, (nr + 1.3) * STEP + 0.016 * (k_ % 2), ab_,
                    color="0.75", fontsize=7.5, ha="center", va="bottom")
        trace_doc = {}
        for i, (short, v) in enumerate(rows):
            m = np.flatnonzero(reg[:, names.index(rfull[short])])
            tr = Xa[:, m].mean(1, dtype=np.float64)
            o_ = (nr - 1 - i) * STEP
            ax.plot(tm_, o_ + tr - np.median(tr), color=rcol[short], lw=0.6)    # no band, no per-row brain mean
            ax.text(tm_[-1] + 1.0, o_, f"{short} ({len(m):,})", color=rcol[short], fontsize=8.5,
                    va="center", ha="left", weight="bold")
            trace_doc[short] = {"r_with_brain_mean": float(np.corrcoef(tr, bm_)[0, 1])}
        json.dump(trace_doc, open(os.path.join(EXP, "data", "atlas_region_traces.json"), "w"), indent=1)
        ob_ = nr * STEP + 0.3 * STEP                                     # the brain mean once, in green, at the top
        ax.plot(tm_, ob_ + bm_ - np.median(bm_), color="#2ca02c", lw=0.8)
        ax.text(tm_[-1] + 1.0, ob_, "brain mean (all)", color="#2ca02c", fontsize=8.5, va="center", ha="left", weight="bold")
        ax.plot([2.0, 2.0], [-1.2 * STEP, -0.8 * STEP], color="white", lw=2.0)     # a 0.02 dF/F bar
        ax.text(3.0, -1.0 * STEP, "0.02 dF/F", color="white", fontsize=8.5, va="center")
        ax.set_ylim(-1.4 * STEP, (nr + 1.9) * STEP)
        ax.set_xlim(0, tm_[-1])
        ax.set_yticks([])
        for sp_ in ("top", "right", "left"):
            ax.spines[sp_].set_visible(False)
        ax.set_xlabel("time since the recording's start, min", fontsize=10)
        ax.tick_params(labelsize=8.5)
        ax.set_facecolor("black")
        ax.set_title("each region's mean dF/F about its median, one scale; the brain mean in green on top", fontsize=9.5,
                     loc="left")
    else:
        # right: the table, each column shaded on its own scale
        ax = fig.add_axes([0.60, 0.05, 0.395, 0.88])           # the table right, its names in its own left part
        ax.axis("off")
        M = np.array([v for _, v in rows], float)
        nr, nc = M.shape
        for j in range(nc):
            col = M[:, j]
            lo, hi = np.nanmin(col), np.nanmax(col)
            for i in range(nr):
                x = col[i]
                if np.isfinite(x) and j > 0:
                    t = (x - lo) / max(hi - lo, 1e-9) if j != 1 else 0.5 + 0.5 * x / max(abs(lo), abs(hi), 1e-9)
                    fc = plt.get_cmap("coolwarm" if j == 1 else "magma")(0.15 + 0.7 * t)
                    ax.add_patch(plt.Rectangle((j, nr - 1 - i), 1, 1, color=fc, alpha=0.75))
                s_ = "--" if not np.isfinite(x) else (f"{int(x):,}" if j == 0 else (f"{x:+.2f}" if j == 1 else f"{x:.0f}"))
                ax.text(j + 0.5, nr - 0.5 - i, s_, ha="center", va="center", fontsize=8, color="white")
        for i, (short, _) in enumerate(rows):
            ax.text(-0.1, nr - 0.5 - i, short, ha="right", va="center", fontsize=8.5, color=rcol[short], weight="bold")
        for j, (c, _) in enumerate(cols):
            ax.text(j + 0.5, nr + 0.2, c, ha="left", va="bottom", fontsize=8.5, rotation=35)
        ax.set_xlim(-4.6, nc + 1.2)                            # room for the names (left) and the last column title
        ax.set_ylim(0, nr + 2.6)
    fout = (f"atlas_regions_block_{iso.split(':')[1].replace(' ', '_')}{'_lr' if iso.endswith(':lr') else ''}.png" if blkv
            else None)
    fig.savefig(os.path.join(EXP, "presentation", "figs", fout or {"merged": "atlas_regions_merged.png", "merged_raster": "atlas_regions_raster.png", "fish": "atlas_regions_fish.png", True: "atlas_regions_iso.png",
                                                           False: "atlas_regions.png"}[iso]), dpi=130, facecolor="black")
    plt.close(fig)


def raster_data(block=None, split_lr=False, NB=1000):
    """The raster's rows (raster() below): binned z-scored dF/F [NB, T], the region bounds in rows, names, colours.
    block: one stimulus block's frames only, z-scored over it, every frame shown (Cedric, 2026-10-07: "a twin focused
    on the gain block, so we see all frames"); split_lr: within each region the left neurons, then the right (Z-Brain's
    midline), each sorted by correlation (Cedric: "left / right, to see a chess-board pattern")."""
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    reg, names = za["regions"], [str(x) for x in za["names"]]
    z_ = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
    X = np.asarray(z_["dff"], np.float32)
    off_, nm_b = z_["offsets"], [str(x) for x in z_["names"]]
    U_ = np.asarray(z_["stimulus"], np.float32)
    f0 = 0
    if block is not None:
        kb = nm_b.index(block)
        f0, f1 = int(off_[kb]), int(off_[kb + 1])
        X, U_ = X[f0:f1], U_[f0:f1]
        off_, nm_b = np.array([0, f1 - f0]), [block]
    T = X.shape[0]
    left = za["atlas_um"][:, 0] < (621 - 1) * ZVOX[0] / 2          # Z-Brain's midline (x, head up)
    rlist = [short for full, short in REGIONS if full in names and reg[:, names.index(full)].sum() >= 50]
    rfull = {short: full for full, short in REGIONS}
    pal = list(plt.get_cmap("tab20")(np.arange(20))) + list(plt.get_cmap("Set1")(np.arange(9)))
    rcol = {r_: pal[k % len(pal)] for k, r_ in enumerate(rlist)}
    lab = np.full(X.shape[1], -1)
    best_n = np.full(X.shape[1], np.inf)
    for k, r_ in enumerate(rlist):
        m = reg[:, names.index(rfull[r_])]
        upd = m & (m.sum() < best_n)
        lab[upd], best_n[upd] = k, m.sum()
    b = X.mean(1, dtype=np.float64)
    bc = (b - b.mean()) / b.std()
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-9)
    r = ((X - mu) / sd * bc[:, None].astype(np.float32)).mean(0)
    order, bounds, lr_cut = [], [], []
    for k in range(len(rlist)):
        a0 = len(order)
        for side in ((True, False) if split_lr else (None,)):
            ids = np.flatnonzero((lab == k) & (True if side is None else (left == side)))
            ids = ids[np.argsort(-r[ids])]
            order.extend(ids.tolist())
            if side is True:
                lr_cut.append(len(order))
        bounds.append((a0, len(order)))
    order = np.array(order)
    NB = 1000
    edges = np.linspace(0, len(order), NB + 1).astype(int)
    img = np.empty((NB, T), np.float32)
    for q in range(NB):
        ids = order[edges[q]:edges[q + 1]]
        img[q] = ((X[:, ids] - mu[ids]) / sd[ids]).mean(1)
    fl = [j for j in range(U_.shape[1]) if U_[:, j].std() > 1e-6] if block is not None else []
    return {"img": img, "bounds": bounds, "rlist": rlist, "rcol": rcol, "n": len(order), "b": b, "off": off_,
            "names": nm_b, "T": T, "lr_cut": lr_cut, "U": U_[:, fl] if fl else None, "block": block}


def draw_raster(fig, D, x0, x1, y0=0.07, y1=0.95, name_w=0.15, fs=7.5):
    """The raster in the figure's [x0, x1] x [y0, y1]: the names (name_w wide) and the region bar at its left, the
    brain mean (green) and the stimulus blocks above it."""
    import matplotlib.pyplot as plt
    img, bounds, rlist, rcol, n, b, off_, nm_b, T = (D[k] for k in ("img", "bounds", "rlist", "rcol", "n", "b", "off",
                                                                   "names", "T"))
    tmin = T * 0.914 / 60
    blk = D.get("block")
    xa = x0 + name_w + 0.017
    hb = 0.09 * (y1 - y0) / 0.88
    yr1 = y1 - hb - 0.01
    ab = fig.add_axes([xa, yr1 + 0.01, x1 - xa, hb])
    for k_ in range(len(off_) - 1):
        ab.axvspan(off_[k_] * 0.914 / 60, min(off_[k_ + 1], T - 1) * 0.914 / 60, color=("0.16" if k_ % 2 else "0.08"), lw=0)
        abb = {"turning": "turn", "position": "pos", "open loop": "open", "rotation": "rot"}.get(nm_b[k_], nm_b[k_])
        ab.text((off_[k_] + min(off_[k_ + 1], T - 1)) / 2 * 0.914 / 60, 1.02, abb, color="0.75", fontsize=fs,
                ha="center", va="bottom", transform=ab.get_xaxis_transform())
    if blk is not None and D.get("U") is not None:       # the block's changing stimulus features, faint orange
        Uu = D["U"]
        Uz = (Uu - Uu.min(0)) / np.maximum(np.ptp(Uu, 0), 1e-9)
        ab2 = ab.twinx()
        for j in range(Uz.shape[1]):
            ab2.plot(np.arange(T) * 0.914 / 60, Uz[:, j] + 1.2 * j, color="#ff9f1c", lw=0.6, alpha=0.8)
        ab2.set_yticks([])
        ab2.set_ylim(-0.2, 1.2 * Uz.shape[1])
    ab.plot(np.arange(T) * 0.914 / 60, b, color="#2ca02c", lw=0.6)
    ab.set_xlim(0, tmin)
    ab.set_xticklabels([])
    ab.set_yticks([])
    ab.set_ylabel("brain\nmean", fontsize=fs + 0.5, color="#2ca02c")
    ab.set_facecolor("black")
    ax = fig.add_axes([xa, y0, x1 - xa, yr1 - y0])
    vm = float(np.percentile(np.abs(img), 99))
    im_out = ax.imshow(img, aspect="auto", cmap="gray", vmin=-0.3 * vm, vmax=vm, extent=(0, tmin, n, 0),
                       interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel(f"time in the {blk} block, min" if blk else "time since the recording's start, min", fontsize=fs + 1.5)
    if blk is not None and D.get("U") is not None:       # the stimulus's front edges, dashed (Cedric, 2026-10-07): a frame
        Uu = np.asarray(D["U"])                           # where the features change to a state that is not all off
        ed = np.flatnonzero((np.abs(np.diff(Uu, axis=0)) > 1e-6).any(1) & (np.abs(Uu[1:]) > 1e-6).any(1)) + 1
        for e_ in ed:
            for a_ in (ax, ab):
                a_.axvline(e_ * 0.914 / 60, color="#ff9f1c", lw=0.6, ls=(0, (4, 3)), alpha=0.85)
    for c_ in D.get("lr_cut", []):                      # left / right within each region
        ax.axhline(c_, color="#4a7bff", lw=0.5, ls="--")
    ax.tick_params(labelsize=fs + 0.5)
    bar = fig.add_axes([xa - 0.014, y0, 0.011, yr1 - y0])
    bar.set_ylim(n, 0)
    bar.set_xlim(0, 1)
    bar.axis("off")
    hgt = yr1 - y0
    want = np.array([yr1 - hgt * (a_ + b_) / 2 / n for a_, b_ in bounds])
    gap = 0.0215 * (fs / 7.5)
    ys = want.copy()
    for _ in range(300):                                  # the names spread where small regions crowd
        for k in range(1, len(ys)):
            if ys[k - 1] - ys[k] < gap:
                mid = (ys[k - 1] + ys[k]) / 2
                ys[k - 1], ys[k] = mid + gap / 2, mid - gap / 2
    for k, (a_, b_) in enumerate(bounds):
        bar.add_patch(plt.Rectangle((0, a_), 1, b_ - a_, color=rcol[rlist[k]]))
        ax.axhline(a_, color="0.5", lw=0.3)
        fig.text(xa - 0.03, ys[k], f"{rlist[k]} ({b_ - a_:,})", color=rcol[rlist[k]], fontsize=fs, ha="right",
                 va="center", weight="bold")
        fig.add_artist(plt.Line2D([xa - 0.028, xa - 0.016], [ys[k], want[k]], color=rcol[rlist[k]], lw=0.6,
                                  transform=fig.transFigure))
    return im_out


def raster():
    """THE RASTER BY REGION (Cedric, 2026-10-07: "a raster sorted by region first, correlation second, coloured per
    region"): every neuron in a table region (its most specific one), z-scored over the recording; rows by region (head
    to tail), then by correlation with the brain mean (highest first); 1,000 bins of consecutive rows, so every neuron
    counts. -> figs/atlas_raster.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = raster_data()
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8.6), facecolor="black")
    draw_raster(fig, D, 0.02, 0.94)
    fig.text(0.20, 0.975, f"{D['n']:,} neurons, z-scored: by region (head to tail), then by correlation with the brain "
             "mean (highest first)", fontsize=10, color="white")
    fig.savefig(os.path.join(EXP, "presentation", "figs", "atlas_raster.png"), dpi=130, facecolor="black")
    plt.close(fig)
    print("[atlas raster]", D["n"], "neurons")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "raster":
    raster()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        if "--block" in sys.argv:                          # summary --block gain [--lr]
            summary(iso=f"block:{sys.argv[sys.argv.index('--block') + 1]}:{'lr' if '--lr' in sys.argv else ''}")
        else:
            summary(iso="merged_raster" if "--raster" in sys.argv else ("merged" if "--merged" in sys.argv else
                    ("fish" if "--fish" in sys.argv else ("--iso" in sys.argv))))
    elif len(sys.argv) == 1:
        main()
