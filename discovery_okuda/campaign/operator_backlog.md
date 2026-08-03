# Operator backlog

_Mechanisms the search wanted and the language could not express._ Each entry is
one unit of atlas growth: it survives parameter changes, and it is actionable
without re-running anything.

_1 open of 1 filed._

## OR001 — Oriented junctional remodeling (planar-polarized T1 neighbor exchanges) driving convergent extension — cells intercalate along a polarity axis so the tissue narrows and elongates into a tube instead of inflating into a ball.
- **status**: open   ·   filed round 2
- **why the language cannot express it**: No operator EMITs a topology REWIRE: every operator emits velocity or force on the vertex set, so the mesh connectivity is frozen and cells can never swap neighbors. There is no edge-set rewire contract to collapse/reconnect a junction, and the cell set has no planar-polarity vector state block to orient which edge is shrunk. shape_energy_3d only relaxes the fixed graph via a scalar Lambda; it cannot change adjacency. Hence growth (uniform inflation / Okuda monolayer) can only isotropically expand — convergent extension is unreachable.
- **what it would answer**: Fills the topology×growth cell of the map that separates 'uniform inflation (no patterning)' from an actual elongating tube — the axis-breaking route none of the 4 frontier compositions can express.
- **proposed contract**: `contract=junction_remodel_t1  set=edge  kind=rewire  family=topology  EMIT=none`
  - params: `{"polarity_vec": "per-cell axis orienting which junction shrinks", "l_threshold": "edge length below which T1 fires", "l_new": "reconnected edge rest length"}`
- **acceptance test**: On a flat 10x10 hexagonal sheet with polarity_vec fixed along x, oriented T1s yield tissue aspect ratio (x/y) >=3.0 within 200 steps at constant cell count, vs 1.0+-0.1 for isotropic cell_grow.
- **motivated by**: C93160bc7edb, Cd13473bf1c3, Cf4907ea516e, C6a07112cc06
