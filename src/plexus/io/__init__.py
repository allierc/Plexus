"""Importers and adapters: the boundary between Plexus and the outside world.

NOTHING HERE IS A PLEXUS CONCEPT. A `Level`, a `Field` and an `Operator` are the language;
these modules only READ a foreign source into the shape the language expects, or WRITE a
Plexus result into the shape a foreign consumer expects. The distinction is worth keeping
sharp because it is the one that keeps a downstream file format out of `models/`:

    neuprint.py   a connectome server -> a frozen REGION MANIFEST (ids, xyz, types, meshes)
    walrus.py     a recorded Field    -> a Well-format HDF5 for the Walrus transformer

Both are one-directional and neither is imported by the engine. An importer runs ONCE,
offline, and leaves a manifest on disk; a seed operator then establishes `x_0` from that
manifest. That split is deliberate: a network query inside an operator would make a run
non-deterministic and unreproducible, and the whole point of freezing the manifest is that
the spec records WHICH region was selected rather than the procedure that selected it.
"""
