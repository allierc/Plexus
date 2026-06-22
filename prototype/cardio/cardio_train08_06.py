#!/usr/bin/env python
"""train_08_06 -- like train_08_05 (phase + global LEARNABLE k_anchor) but the simulation
runs at the NATIVE real-data resolution 137x137 (every optical-flow control point is a
node; no _sel down-sampling), i.e. ~4.6x more points in the training loss than the 64x64
runs. Trajectory display stays 10x10 (cog crop).

This is GPU-only territory: 137^2 = 18769 nodes makes each iteration heavy, which is why
the chain runs on cuda with torch.compile (~17x over CPU eager, measured).

Sets the env flags BEFORE importing train_08_03 so its module-level config picks them up,
then reuses 08_03's machinery; archives to archive/train_08_06/.

Run:  CARDIO_DEVICE=cuda:1 CARDIO_COMPILE=1 CARDIO_MAX_SECONDS=28800 CARDIO_CKPT_EVERY=5000 \
      PYTHONPATH=src .../python prototype/cardio/cardio_train08_06.py
"""
from __future__ import annotations

import os

os.environ["CARDIO_K_ANCHOR_LEARNABLE"] = "1"
os.environ.setdefault("CARDIO_ARCH_NAME", "train_08_06")
os.environ.setdefault("CARDIO_GRID", "137")          # native real-data resolution
os.environ.setdefault("CARDIO_BWIDTH", "11")         # anchored band reaches the margin-10 display ring at 137

import cardio_train08_03 as T03   # noqa: E402  (imported AFTER env is set)

if __name__ == "__main__":
    T03.main()
