
## Round 4 — 2026-08-01 14:41

- batch: 3 mechanism hypotheses (100% confirmatory / 0% adversarial) + 1 control
- **surprise rate: 0.00**
- supervisor: surprise 0.00 < 0.1: the batch is confirming what we already believe (drifting to 100/0, near-zero information). Push ADVERSARIAL.

### Validated
- **removing vesicle_growth removes the driver of the inflation, collapsing the body to a static mechanical mesh with no protrusion**
  - edit `('remove_op', 'vesicle_growth0')` on `Ce8e350fdb89` · intent *confirmatory*
  - predicted `protr_peak <= 1.5` on `protr_peak` → observed `{"saturated": false, "inert_operators": [], "retention": 1.0, "valid_evidence": true, "protr_final": 1.003, "protr_peak": 1.003, "elongation_at_end": 1.003, "elongation_peak": 1.003, "horizon_frame": 900, "horizon_why": "broken_n never sustained damage to the end (max 0)", "first_damage_frame": null, "valid_frac": 1.0, "protr_peak_untruncated": 1.003, "protr_final_untruncated": 1.003, "n_cells_final": 2000, "red_frac_final": 0.0, "act_max_final": 0.0, "frames": 901, "wall_s": 231.2, "ta_hollow_n_peak": 0, "ta_hollow_n_final": 0, "ta_area_cv_final": 0.107, "ta_vol_cv_final": 0.107, "ta_tube_diam_final": 0.0, "ta_n_tubes_final": 0, "ta_tube_len_final": 0.0, "ta_protr_final": 1.003, "ta_red_frac_final": 0.0, "ta_tip_act_final": 0.0, "morphology": "sphere", "morphology_path": "sphere", "morphology_why": "", "mech_force_mean": 0.4623, "mech_p_body": 0.2931, "mech_p_tube": 0.0, "mech_p_ratio": 0.0, "mech_tension_mean": 0.1183, "mech_migration": 0.00365, "Q_protr_after_relax": 1.003, "Q_drop": -0.0, "analyst_consensus": "sphere", "analyst_agreement": 1.0, "analyst_forced_or_grown": "unclear", "analyst_disagreement": false, "analyst_concerns": ["p95 shape index 3.887 (>3.81 fluidization line) in the stretched tail, but bulk min is 3.668 and cells in the strip look uniform with no localized stretching; metrics.png trajectory-shape classifier failed (ValueError on 'sphere' string) so I read the curves directly \u2014 all flat/converged, no exploded or pinned artefact.", "cell census labels ~40% of cells 'branch/tip' even though tube_diam=0 and n_tubes=0 (classifier artifact, not geometry); the p95-late shape index 3.887 sits just above the 3.81 fluidization line but only in the tail (min 3.668 near the ~3.5 preferred), and the cross-section rings are uniform circles with no stretched/thin cells anywhere.", "shape_idx p95_late 3.887 exceeds the 3.81 fluid-solid threshold (tail cells near-flowing, min 3.668) yet nothing deforms \u2014 cosmetic, not physics; migration spike to 0.07 at frame ~50 is just the initial relaxation transient, and metrics.png trajectory-shape classifier failed (read manually: all curves flat/converged, none pinned or exploded)."], "watcher_verdict": "supports", "watcher_seen": "two static geodesic-grid spheres, no motion", "watcher_why": "The vision model plainly reports two large spherical structures, which directly supports the claimed 'sphere' morphology.", "watcher_blocks": false}`
  - protr_peak=1.003 satisfies <=1.5

### Refuted
- _(none this round)_

### Open / inconclusive
- **the parent, unchanged -- uniform inflation gives a smooth near-spherical body with no patterned protrusion**
  - edit `control (parent unchanged)` on `C6a07112cc06` · intent *control*
  - predicted `protr_peak 1.0-1.8 and ta_n_tubes_final <= 0` on `protr_peak` → observed `{"saturated": true, "inert_operators": [], "retention": 0.762, "valid_evidence": false, "protr_final": 2.163, "protr_peak": 2.839, "elongation_at_end": 2.163, "elongation_peak": 2.839, "horizon_frame": 900, "horizon_why": "broken_n never sustained damage to the end (max 0)", "first_damage_frame": null, "valid_frac": 1.0, "protr_peak_untruncated": 2.839, "protr_final_untruncated": 2.163, "n_cells_final": 20804, "red_frac_final": 0.0, "act_max_final": 0.0, "frames": 901, "wall_s": 983.4, "ta_hollow_n_peak": 16472, "ta_hollow_n_final": 16472, "ta_area_cv_final": 0.352, "ta_vol_cv_final": 0.502, "ta_tube_diam_final": 37.128, "ta_n_tubes_final": 44, "ta_tube_len_final": 89.621, "ta_protr_final": 1.693, "ta_red_frac_final": 0.0, "ta_tip_act_final": 0.0, "ta_aspect_len_over_diam": 2.414, "morphology": "branched", "morphology_path": "sphere -> branched", "morphology_why": "", "mech_force_mean": 3312858112.0, "mech_p_body": 1028474.0625, "mech_p_tube": 1131453.375, "mech_p_ratio": 1.1, "mech_tension_mean": 3443.9893, "mech_migration": 14.49028, "Q_protr_after_relax": 2.234, "Q_drop": -0.071}`
  - NOT EVIDENCE: [<P2_BUFFER_SATURATED: n_cells=20804>]
- **swapping shape_energy_3d to the monolayer implementation buckles the uniformly-inflating shell into protrusions (undulation), contradicting the type system's 'no patterning' label**
  - edit `('set_impl', 'shape_energy_3d0', 'monolayer')` on `C60badfef692` · intent *adversarial*
  - predicted `protr_peak >= 2.0 and ta_n_tubes_final >= 1` on `protr_peak` → observed `{}`
  - no diag.json
- **removing T1 topology relief leaves the isotropic-growth shape essentially unchanged -- reconnect_t1_3d is remodeling plumbing, not a shape driver under uniform growth**
  - edit `('remove_op', 'reconnect_t1_3d0')` on `Ccf2633bc4ba` · intent *confirmatory*
  - predicted `protr_peak 1.0-1.8` on `protr_peak` → observed `{"saturated": true, "inert_operators": [], "retention": 0.83, "valid_evidence": false, "protr_final": 2.093, "protr_peak": 2.522, "elongation_at_end": 2.093, "elongation_peak": 2.522, "horizon_frame": 900, "horizon_why": "broken_n never sustained damage to the end (max 0)", "first_damage_frame": null, "valid_frac": 1.0, "protr_peak_untruncated": 2.522, "protr_final_untruncated": 2.093, "n_cells_final": 5204, "red_frac_final": 0.0, "act_max_final": 0.0, "frames": 901, "wall_s": 229.5, "ta_hollow_n_peak": 4192, "ta_hollow_n_final": 4148, "ta_area_cv_final": 0.356, "ta_vol_cv_final": 0.528, "ta_tube_diam_final": 55.874, "ta_n_tubes_final": 36, "ta_tube_len_final": 98.378, "ta_protr_final": 1.635, "ta_red_frac_final": 0.0, "ta_tip_act_final": 0.0, "ta_aspect_len_over_diam": 1.761, "morphology": "unclear", "morphology_path": "sphere -> unclear -> branched -> unclear -> tube -> sphere -> branched -> unclear -> branched -> unclear", "morphology_why": "", "mech_force_mean": 34890108928.0, "mech_p_body": 4262065.0, "mech_p_tube": 4603568.0, "mech_p_ratio": 1.08, "mech_tension_mean": 5561.4702, "mech_migration": 18.13529, "Q_protr_after_relax": 2.195, "Q_drop": -0.102}`
  - NOT EVIDENCE: [<P2_BUFFER_SATURATED: n_cells=5204>]

### Ledger movement this round

- kept: 1
- dropped: 0
