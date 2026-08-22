"""connectome -- generate a synthetic W over a set of neurons and freeze it to disk.

The region importer (`plexus.io.neuprint`) says WHICH neurons and WHERE they are. This says
WHO TALKS TO WHOM and HOW STRONGLY, for a run whose connectivity is synthetic rather than
measured. It writes one `connectome.npz` -- `edge_index [2, E]`, `weights [E]` -- which a
spec names through `sets.<edge_set>.edges_file:`.

    neurons.npz  (xyz)        ->   connectome.npz  (edge_index, weights)   ->   spec

WHY THIS IS A SEPARATE FILE FROM THE MANIFEST. The manifest is a record of something
OBSERVED: a NeuPrint dataset, a query, a cube, and the neurons that fell inside it. The
connectome here is something INVENTED, with a seed and a rule. Keeping them apart means a
later run can hold the region fixed and swap the wiring, or hold the wiring fixed and move
the cube, and in both cases the file that changed says which of the two it was.

TWO CONNECTIVITY KERNELS, and the choice is not cosmetic:

    uniform      P(i<-j) = p, independent of distance. The classic random recurrent network.
    exponential  P(i<-j) = p * exp(-d_ij / lambda). Nearby somas are likelier to be wired.

`exponential` IS THE DEFAULT, because of what the output is for. A field rendered from a
uniformly-wired network is spatially WHITE: neighbouring voxels are as uncorrelated as
distant ones, because nothing in the dynamics knows about space. That is a legitimate neural
model and a poor continuum dataset -- and the downstream consumer (Walrus) is a model of
CONTINUUM dynamics, which learns spatial structure. With a distance kernel the activity has
a correlation length set by `lambda`, so the rendered volume is a field with a scale in it
rather than noise on a grid. The kernel is recorded in the npz so the claim is auditable.

Weights are drawn N(0, g / sqrt(<k>)) with <k> the mean in-degree -- the balanced-random
scaling that keeps the recurrent drive O(1) as the network grows, so `g` means the same thing
at 100 neurons and at 10,000.

    python -m plexus.io.connectome --region <dir> --p 0.02 --g 1.5 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def random_connectome(xyz: np.ndarray, p: float = 0.02, g: float = 1.5, seed: int = 0,
                      kernel: str = "exponential", lam: float | None = None,
                      allow_self: bool = False, max_edges: int = 4_000_000):
    """(edge_index [2, E], weights [E], meta). `edge_index[0]` is presynaptic.

    `lam` is the decay length in the SAME UNITS AS `xyz`, i.e. NANOMETRES for a region
    written by `plexus.io.neuprint`. Without one it defaults to a tenth of the cloud's diameter, which puts a few
    correlation lengths across the volume rather than one -- a field with one correlation
    length across it is a single blob, and with fifty it is noise.
    """
    rng = np.random.default_rng(seed)
    n = int(xyz.shape[0])
    if kernel == "uniform":
        prob = np.full((n, n), float(p))
        lam_used = None
    elif kernel == "exponential":
        d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)      # [n, n]
        diam = float(d.max())
        lam_used = float(lam) if lam else diam / 10.0
        k = np.exp(-d / lam_used)
        # normalise so the MEAN connection probability is exactly `p`: without this, `p`
        # would mean something different for every lambda and the two kernels would not be
        # comparable at the same `p`.
        off = ~np.eye(n, dtype=bool)
        prob = k * (float(p) / k[off].mean())
    else:
        raise ValueError(f"kernel must be 'uniform' or 'exponential', got {kernel!r}")
    np.clip(prob, 0.0, 1.0, out=prob)
    if not allow_self:
        np.fill_diagonal(prob, 0.0)
    adj = rng.random((n, n)) < prob                     # adj[i, j] = j -> i (row = postsynaptic)
    post, pre = np.nonzero(adj)
    E = int(pre.shape[0])
    if E > max_edges:
        raise ValueError(f"{E} edges exceeds max_edges={max_edges}; lower --p")
    mean_k = E / max(n, 1)                              # mean in-degree
    w = rng.normal(0.0, float(g) / np.sqrt(max(mean_k, 1.0)), size=E).astype(np.float32)
    meta = {"n_neurons": n, "n_edges": E, "p": float(p), "g": float(g), "seed": int(seed),
            "kernel": kernel, "lambda": lam_used, "mean_in_degree": float(mean_k),
            "weight_sd": float(w.std()), "allow_self": bool(allow_self)}
    return np.stack([pre, post]).astype(np.int64), w, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, help="region dir holding neurons.npz")
    ap.add_argument("--out", default=None, help="output npz (default: <region>/connectome.npz)")
    ap.add_argument("--p", type=float, default=0.02, help="mean connection probability")
    ap.add_argument("--g", type=float, default=1.5, help="coupling gain (weight sd scaling)")
    ap.add_argument("--kernel", default="exponential", choices=("exponential", "uniform"))
    ap.add_argument("--lam", type=float, default=None, help="decay length, in xyz units")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    z = np.load(os.path.join(a.region, "neurons.npz"), allow_pickle=True)
    # NANOMETRES, not voxels. The distance kernel below has a length scale, and on an
    # anisotropic dataset (fish2: 16/16/15 nm) a voxel distance is not a distance -- the same
    # lambda would reach 6.7% further along z than along x, which is a claim about the tissue
    # that nothing in the model intends to make.
    xyz = np.asarray(z["xyz_nm"], float)
    ei, w, meta = random_connectome(xyz, p=a.p, g=a.g, seed=a.seed, kernel=a.kernel, lam=a.lam)
    out = a.out or os.path.join(a.region, "connectome.npz")
    np.savez_compressed(out, edge_index=ei, weights=w, **{f"meta_{k}": v for k, v in meta.items()
                                                          if v is not None})
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[connectome] {meta['n_neurons']} neurons, {meta['n_edges']} edges "
          f"(mean in-degree {meta['mean_in_degree']:.1f}), kernel={meta['kernel']}"
          + (f" lambda={meta['lambda']:.0f}" if meta["lambda"] else "")
          + f", weight sd {meta['weight_sd']:.4f}")
    print(f"[connectome] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
