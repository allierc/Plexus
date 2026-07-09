# SMG2 — budding epithelium (Shaohe Wang) — prototype

3D + time movie of a mouse **submandibular gland (SMG)** epithelium undergoing
branching morphogenesis / **budding**. Tracked cell centroids, ParticleGraph
format.

## Source
* Raw (Wang lab, not mounted here): `/groups/wang/wanglab/GNN/240408-LVpD80-E10-IAI/SMG2-processed/`
  (`masks_smooth2_label_props/`, `masks_smooth2_mesh_csv/`).
* Processed tensors (used here):
  `/workspace/ParticleGraph/graphs_data/cell/cell_gland_SMG2_smooth{2,10}/x_list_0.pt`
* Config: `ParticleGraph/config/cell/archive/cell_gland_SMG2_smooth2.yaml`
  (`PDE_Cell`, dimension 3, n_particles 12774, n_frames 553,
  has_cell_division/death = True).

## Data layout (`x_list_0.pt`)
`list` of 553 frames; frame `t` is an `(N_t, 16)` float tensor. N grows
**9,626 → 12,755** over time (cell divisions = the budding).
* col 0, 15 — per-frame row index (no stable lineage id in this tensor)
* **cols 1,2,3 — centroid x, y, z in isotropic microns** (gland ≈ 738 × 683 × 146 µm,
  aspect 1 : 0.92 : 0.20 — a wide, flat branched gland). Verified isotropic: nearest-
  neighbour displacement is equal per axis (|dx|,|dy|,|dz| ≈ 3.4, 3.6, 3.6 µm) and 3D
  cell spacing ≈ 7.3 µm (nuclear spacing). **z-slice = same µm as the xy pixel** — no
  anisotropy correction needed; the 3D view uses true aspect (no z exaggeration).
* cols 4–9 — zero (velocity/accel placeholders, filled during GNN training)
* cols 10–14 — morphology features (size, sphericity-like col12 ∈ [0.25,1.7], intensities)
* `smooth2` vs `smooth10` — segmentation-mask smoothing level; centroids nearly identical.

## Files
* `make_movie.py` — 3D rotating point cloud + xy top view, colored by depth.
  Run: `python make_movie.py [--smooth 2|10] [--fps 25] [--stride N]`
* `SMG2_smooth2_budding.mp4` — the rendered movie (553 frames).

## Notes for modeling (ties to the field+pairwise discussion)
* Clear branching/budding: buds elongate and split — a strong tissue-scale
  **field** (morphogenetic flow) on top of local cell–cell **pairwise** rules.
* Boundary/basement-membrane effects (per Shaohe): residual velocity fails at
  the tissue boundary → candidate for **virtual ECM nodes**.
* No stable tracks in this tensor; motile-package tracking lives with the raw
  data (`tracking_motile/`, `centroid_trackid.tif`) on the Wang-lab path.
