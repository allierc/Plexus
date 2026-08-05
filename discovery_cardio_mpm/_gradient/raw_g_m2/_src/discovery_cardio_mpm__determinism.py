#!/usr/bin/env python
"""determinism -- fix the dice, and prove they are fixed.

WHAT WAS MEASURED
================================================================================================
The inherited trainer never calls `torch.manual_seed`. The stiffness and fibre networks are
initialised from whatever the global generator happens to hold, so every run drew a different
model before a single gradient step. Two invocations of the SAME command, with no training at
all, differed by 83% of the amplitude of the signal being fitted. Sixty rounds of rankings were
then decided on differences of 0.003.

That is not a small defect and it is not an expensive fix. It is three lines.

THE SECOND SOURCE, WHICH A SEED DOES NOT CURE
------------------------------------------------------------------------------------------------
The particle-to-grid transfer scatters with CUDA atomics (`index_add_`), whose summation order
is not fixed. So the same seed can still give slightly different answers on the GPU. The
inherited trainer makes this worse at module scope:

    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark   = True

Those are executed on IMPORT, so anything that sets them earlier is silently overridden. `enforce`
must therefore be called AFTER every import, which is why it is a function and not a preamble.

TWO DIFFERENT NOISE FLOORS, AND THE CAMPAIGN NEEDS BOTH
------------------------------------------------------------------------------------------------
  sigma_repeat   same seed, run twice   -- pure arithmetic non-reproducibility
  sigma_seed     different seeds        -- the initialisation lottery

They answer different questions and Phase 2 measures both; the smallest difference the campaign
may call real is built from the larger. Confusing them is how "this run beat that one" survived
for sixty rounds.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

# CUBLAS needs this set BEFORE the CUDA context is created for deterministic matmul reductions.
_CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"


def enforce(seed: int, deterministic: bool = True, warn_only: bool = False):
    """Seed everything and pin the arithmetic. Call AFTER all imports -- see the module note.

    Returns a dict recording exactly what was set, for the run manifest.
    """
    if deterministic and _CUBLAS_ENV not in os.environ:
        os.environ[_CUBLAS_ENV] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(warn_only))
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)
    try:                                              # torch >= 2.9 renames these
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
    except Exception:                                 # pragma: no cover - version drift
        pass

    return {"seed": int(seed), "deterministic": bool(deterministic),
            "warn_only": bool(warn_only), _CUBLAS_ENV: os.environ.get(_CUBLAS_ENV),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None}


def state_fingerprint():
    """A cheap fingerprint of the RNG state, so a manifest can prove two runs started level."""
    import hashlib
    h = hashlib.sha256()
    h.update(torch.random.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for s in torch.cuda.get_rng_state_all():
            h.update(s.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def selftest(device="cpu", verbose=True):
    """Prove the seed controls the draw, and that it is the ONLY thing that does.

    Draws the same network initialisation twice at one seed and once at another. Same seed must
    be bit-identical; different seed must not be. A test that only checks the first half would
    pass on a constant.
    """
    def draw(seed):
        enforce(seed, deterministic=True)
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
        net.to(device)
        x = torch.randn(512, 2, device=device)
        return torch.cat([p.detach().flatten().cpu() for p in net.parameters()] + [x.flatten().cpu()])

    a1, a2, b = draw(1234), draw(1234), draw(4321)
    same = bool(torch.equal(a1, a2))
    differs = not bool(torch.equal(a1, b))
    ok = same and differs
    if verbose:
        print(f"  [determinism] same seed identical : {same}")
        print(f"  [determinism] other seed differs  : {differs}"
              f"   (max|d| = {float((a1 - b).abs().max()):.4g})")
        print(f"  [determinism] selftest {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    info = enforce(0, deterministic=True)
    print(f"  [determinism] {info}")
    ok = selftest("cpu")
    if torch.cuda.is_available():
        ok = selftest("cuda:0") and ok
    sys.exit(0 if ok else 1)
