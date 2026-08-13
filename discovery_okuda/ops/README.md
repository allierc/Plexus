# Tyssue — a true vertex model (AVM) for epithelial mechanics in plexus2

Rebuild of the epithelium simulator [`tyssue`](https://github.com/DamCB/tyssue) (DamCB) as
registered plexus2 operators. Two goals:

1. **Complete the Turing_vertex prototype.** That prototype modelled epithelial mechanics as a
   *Self-Propelled Voronoi* (SPV) tissue and found that clean tubulation was gated on two
   mechanics ingredients the Voronoi route lacks: **force-balance iteration** and an
   **explicit T1 reconnection** operator. The true vertex model here supplies both natively —
   same shape-energy contract, a genuinely different implementation (plexus2 §5).
2. **Extract the tyssue operator atlas.** Decompose tyssue's full mechanism vocabulary into
   Plexus operators (sets / states / fields / maps / operators), per plexus2 App B & H —
   `tyssue_atlas.yaml`.

## Representation

- **Set** `vertex` — the mechanical DOF (`pos`). Faces are *not* an integrated set; they are a
  half-edge table `(srce, trgt, face)` stashed on the vertex Level (like the SPV prototype
  stashed Voronoi rings). Area/perimeter are aggregate readouts (vertices → face).
- Bounded honeycomb bootstrapped from a Voronoi of a triangular lattice (once); border pinned.

## Operators (`ops_2d.py`)

| op | family / kind | role |
|---|---|---|
| `seed_mesh` | growth / structural | build honeycomb half-edge mesh; stash edge table + per-face A0/P0 |
| `shape_energy` | mechanics / lateral | AVM shape-energy force on vertices (vectorised scatter-add + autograd), inner relax loop = force balance |
| `t1_transition` | topology / rewire | *(coming)* explicit T1: collapse short edge → split vertex |
| `face_divide` | growth / structural | *(coming)* split a face along an axis |
| `face_extrude` | growth / structural | *(coming)* remove a face (apoptosis / delamination) |

## Run

```bash
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
PYTHONPATH=../../src $PY run_tyssue2d.py            # p0 rigidity-transition sweep -> archive/
PYTHONPATH=../../src $PY run_tyssue2d.py --montage  # montage + transition curve
```

Each test writes `archive/<name>/`: `spec.yaml`, `traj.npz`, `strip.png`, `movie.mp4`, `diag.json`.
Cluster (L4) generation: `python cluster_gen.py <preset>` (reused from Turing_vertex).

## Findings (see `tyssue_report.pdf` for the full living log)

- **Stage 1 — rigidity transition from true force balance.** Sweeping p0 ∈ [3.6, 4.1], residual
  energy/cell collapses ~4 orders across p0*≈3.81; relaxed ⟨q⟩ saturates at the hexagon floor
  (~3.72) below, tracks p0 above. No timestep tuning — the SPV's "master knob" problem is gone.
