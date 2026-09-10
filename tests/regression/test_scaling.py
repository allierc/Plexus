"""Layer B2 of tests/REGRESSION_PLAN.md: scaling laws, which no wall-clock budget can replace.

A frame-time budget is bound to one GPU; a RATIO between two problem sizes is not. Each test runs
the same operation at size n and at a multiple of n, takes the median of three timings, and asserts
the ratio stays under what a linear (or launch-bound) implementation gives, with slack for a shared
machine. A quadratic path returning -- `_edge_face_map` rebuilt per division cost 19.4 s of a
24.2 s frame and killed four gpu_l4 jobs on the wall clock -- pushes the 4x ratio to 16x and fails
on any host.
"""
import os
import sys
import time

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytestmark = pytest.mark.regression
CUDA = torch.cuda.is_available()
DEVICE = os.environ.get("PLEXUS_REGRESSION_DEVICE", "cuda:0") if CUDA else "cpu"


def _median_time(fn, reps=3):
    ts = []
    for _ in range(reps):
        if CUDA:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if CUDA:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _sphere(n, dev):
    from plexus.operators.vertex_ops import build_sphere_mesh
    pos, es, et, ef, nF = build_sphere_mesh(n, r=1.0, jitter=0.05, seed=0)
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=dev)   # noqa: E731
    return t(pos), t(es, torch.long), t(et, torch.long), t(ef, torch.long), nF


@pytest.mark.skipif(not CUDA, reason="the contact lookup is exercised on the GPU")
def test_mesh_contact_lookup_is_linear():
    """Bin table build + query for a fixed surface and 4x the particles. The query is [candidates,
    9K] per slice and the slices are independent, so it is linear in the particles -- the dimension
    that grows with the matrix around a tissue and that the chunked query exists for. (Scaling the
    surface too is NOT linear: the per-bucket depth K grows with the faces.) 4x the particles must
    cost < 8x the time (2x slack over linear on a shared host)."""
    from plexus.operators.contact_ops import MeshContact
    op = MeshContact({"at": "mpm_particle", "surface": "vertex", "k_frac": 0.9, "mu": 0.4,
                      "dt": 1.0, "n_grid": 64, "centre": [0.0, 0.0, 0.0]})
    dev = torch.device(DEVICE)

    def make(n_cells, n_pts):
        V, es, et, ef, nF = _sphere(n_cells, dev)
        g = torch.Generator(device="cpu").manual_seed(1)
        X = (torch.rand(n_pts, 3, generator=g) * 2.6 - 1.3).to(dev)       # a cloud around the unit shell
        return V, es, et, ef, X

    def once(V, es, et, ef, X):
        M = op._build_from(V, es, et, ef, torch.zeros_like(V), dev, torch.float32)
        r = X.norm(dim=1).clamp_min(1e-9)
        op._query(M, X, X / r[:, None], r)

    a = make(800, 100_000); b = make(800, 400_000)
    once(*a); once(*b)                                     # warm the kernels once
    ta, tb = _median_time(lambda: once(*a)), _median_time(lambda: once(*b))
    assert tb / ta < 8.0, f"contact lookup: 4x particles cost {tb/ta:.1f}x the time ({ta*1e3:.1f} -> {tb*1e3:.1f} ms)"


@pytest.mark.skipif(not CUDA, reason="the warp gradient is CUDA-only")
def test_warp_shape_gradient_is_launch_bound():
    """The hand-written warp gradient is 0.40 ms whether the mesh has 1,188 or 71,988 half-edges
    (vertex_ops, ShapeEnergy3D._grad). 16x the half-edges must cost < 4x the time."""
    from plexus.operators.vertex_ops import shape_energy_grad_warp, face_geometry_3d
    dev = torch.device(DEVICE)

    def make(n_cells):
        pos, es, et, ef, nF = _sphere(n_cells, dev)
        eocc = torch.ones(es.shape[0], device=dev)
        area, perim, _, vf = face_geometry_3d(pos, es, et, ef, nF, eocc)
        alive = torch.ones(nF, device=dev); vocc = torch.ones(pos.shape[0], device=dev)
        buf = {}
        return lambda: shape_energy_grad_warp(pos, es, et, ef, nF, area, perim, vf, alive, 0.85,
                                              1.0, 0.6, 2.0, 0.4, 0.5, 0.15, eocc, vocc, buffers=buf)

    fa, fb = make(200), make(3200)
    fa(); fb()
    ta, tb = _median_time(fa, reps=5), _median_time(fb, reps=5)
    assert tb / ta < 4.0, f"warp gradient: 16x half-edges cost {tb/ta:.1f}x the time ({ta*1e3:.2f} -> {tb*1e3:.2f} ms)"


def test_cell_divide_topology_is_linear():
    """`divide_face_3d` with the maintained edge->face map: dividing every third face of a 200-cell
    and an 800-cell sphere. Four times the faces are four times the divisions, each O(1) with the
    map carried, so 4x the tissue must cost < 8x the time. The per-call rebuild that was removed
    (19.4 s of a 24.2 s frame at frame 380 of mesh_mpm_spheroid_nominal) makes it 16x."""
    from plexus.models.topology import _edge_face_map, divide_face_3d, rings_from_flat_3d
    from plexus.operators.vertex_ops import build_sphere_mesh

    def make(n):
        pos, es, et, ef, nF = build_sphere_mesh(n, r=1.0, jitter=0.05, seed=0)
        return rings_from_flat_3d(np.asarray(es), np.asarray(et), np.asarray(ef), int(nF)), [p for p in np.asarray(pos, np.float64)]

    def divide_all(n):
        rings, pos = make(n)
        emap = _edge_face_map(rings)
        k = 0
        for f in range(0, len(rings), 3):
            if rings[f] is not None and len(rings[f]) >= 4:
                if divide_face_3d(rings, pos, f, emap=emap) is not None:
                    k += 1
        return k

    ka, kb = divide_all(200), divide_all(800)
    assert ka > 30 and kb > 4 * ka * 0.8, f"too few divisions to time ({ka}, {kb})"
    ta = _median_time(lambda: divide_all(200)); tb = _median_time(lambda: divide_all(800))
    assert tb / ta < 8.0, f"cell_divide topology: 4x tissue cost {tb/ta:.1f}x ({ta*1e3:.0f} -> {tb*1e3:.0f} ms, {ka} -> {kb} divisions)"
