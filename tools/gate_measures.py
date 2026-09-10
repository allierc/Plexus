#!/usr/bin/env python
"""One function per gate measure, and one facade over the two trajectory layouts.

WHY A FACADE. A gate is only worth running if the SAME code grades the promoted run and the okuda
reference; two measurement functions that agree on a good run and disagree on a bad one are worse
than none. So every `fn` below takes a `Traj` and never a filename, and there are two `Traj`
implementations -- one over the core's `trajectory.npz`, one over okuda's `traj.npz` -- with exactly
the same seven accessors.

WHAT IS IN A CORE TRAJECTORY, since every function here is bounded by it:

    <set>__pos              [T, buffer, D]      positions, the live prefix given by nF/Nv
    <set>__occ              [T, buffer]         bool
    <set>__<block>          [T, buffer, width]  every recorded state block (chem, area, centroid, ...)
    <set>__mesh_offsets     [T+1]               HALF-EDGE row offsets
    <set>__mesh_face_offsets[T+1]               FACE row offsets -- a DIFFERENT ragged length
    <set>__mesh_nF/_Nv      [T]
    <set>__mesh_E_srce/_E_trgt/_E_face  concatenated int64
    <set>__mesh_<name>      concatenated float32, one entry per face per row (A0, age, ndiv, ...)

THE TWO OFFSET ARRAYS ARE NOT INTERCHANGEABLE and mixing them is the mistake this module is written
to make impossible: there are about six half-edges per face, so slicing the myosin with the half-edge
offsets returns a window six times too long, starting in the wrong frame, and the resulting number
looks entirely plausible.

EVERY FUNCTION RETURNS A SERIES, one entry per recorded row, and the row's `reduce:` collapses it.
That is deliberate: a gate that only ever looks at the last frame cannot see a tissue that tore at
frame 200 and healed, and `euler_characteristic` with `reduce: all` is exactly that case.

UNITS ARE APPLIED BY THE REDUCER, NEVER BY THE FUNCTION. An `fn` returns simulation units; a row whose
`unit:` is physical converts through the spec's `units:` block, which RAISES if none is declared.
That is the mechanism that stops a dimensionless run quoting a micrometre.
"""
from __future__ import annotations

import os

import numpy as np


# ============================================================================== the facade
# THE READERS LIVE IN THE PACKAGE NOW (`plexus.measures`), with the curve panel's and the
# fingerprints' quantities beside them: one entry point for every number read off a run.
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from plexus.measures import _Lazy, Traj, CoreTraj, ParticleTraj, OkudaTraj, _core, open_traj  # noqa: E402,F401

# THE BODIES LIVE IN THE PACKAGE (`plexus.measures_rows`); this file is the name the gate runner
# and the tests import, and it re-exports everything, `MEASURES` included.
import plexus.measures_rows as _rows  # noqa: E402
# EVERY NAME, the underscore helpers included: tests read `_cell_polyhedron_volume` and friends here.
globals().update({_k: _v for _k, _v in vars(_rows).items() if not _k.startswith("__")})
MEASURES = _rows.MEASURES
