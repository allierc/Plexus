#!/usr/bin/env python
"""train_08_05 -- identical to train_08_03 (phase map + full-dataset cycle-batched RMSE,
boundary anchored in train+render) but with the SUBSTRATE SPRING ANCHOR (k_anchor) made
a GLOBAL LEARNABLE scalar instead of fixed at 0.06.

This is the "is the spring stiffness identifiable from the beat?" experiment. k_anchor is
clamped to a floor (0.01) so the tissue can't drift off its rest position entirely.

Everything else (model, data, renders, checkpoints incl. model_*.pt) is inherited from
cardio_train08_03 -- this wrapper just flips the env flags BEFORE importing it, so the
module-level config picks them up, and archives to archive/train_08_05/.

Run:  PYTHONPATH=src .../python prototype/cardio/cardio_train08_05.py
      (honours CARDIO_N_ITER / CARDIO_CKPT_EVERY like 08_03)
"""
from __future__ import annotations

import os

os.environ["CARDIO_K_ANCHOR_LEARNABLE"] = "1"
os.environ.setdefault("CARDIO_ARCH_NAME", "train_08_05")

import cardio_train08_03 as T03   # noqa: E402  (imported AFTER env is set)

if __name__ == "__main__":
    T03.main()
