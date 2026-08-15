#!/usr/bin/env python
"""07i -- adhesion arrives at a RATE, and leaves when it is overstretched.

    python test_07i_ramp.py --frames 401 --every 10 --batched --name 07i_ramp

THE TWO GATES 07h LEFT OPEN, and they are the same defect seen twice.

  G71  the plaque count grows smoothly        FAIL at 7.5% per kept interval against 5%
  G72  the length distribution is stationary  FAIL on its tail: p99 4.62 l0 against 4, while the
                                              MEDIAN was fine at 1.83 l0

07h refilled a dividing cell's shortfall in the frame the division happened: a mother holding twelve
plaques splits into six and six, and twelve fresh clusters appear at once. Divisions come in bursts,
so the count moves in steps rather than growing -- which is what G71 reads. A real focal adhesion does
not appear complete; it nucleates and matures over minutes, and twelve of them do not nucleate in the
same instant.

WHAT CHANGES, and it is two rules on the same register.

  ARRIVAL AT A RATE. The shortfall is not banked at division and replayed; it is RECOMPUTED every
  frame as `N0 - (plaques each cell holds)`, and at most `ramp` of them are seeded per cell per
  frame. Recomputing rather than accumulating is what keeps this honest: a banked deficit is counted
  again at the next division, because `PlaqueOwner.deficit()` reports the shortfall that the bank has
  not yet paid. With `ramp` = 2 and `N0` = 12 a division's shortfall fills over six frames.

  DEPARTURE ON OVERSTRETCH. 07h culled a plaque when it had no bonds AND was out of reach. That is
  the right rule for a cluster that let go, and it left the p99 tail untouched: a plaque can hold
  bonds at 4.6 rest lengths -- 3.2 um, eighty times the 40 nm an integrin spans -- and 07h kept it.
  Here a plaque is also culled at `cull_l0` rest lengths whatever its bonds, which bounds G72's tail
  by construction.

AND THE CHAIN-FAILURE RISK THIS RUNS, stated before the run rather than after. `test_07_plaque`'s own
docstring warns that cutting on length alone breaks a second population -- plaques that are not
overstretched but displaced -- and that each break moves load onto the neighbours that remain, which
can cascade. Two things make that survivable here and neither is a guarantee: the replacement is
seeded radially UNDER a sheet node within `N0/ramp` frames, so the cell does not stay short; and the
cull is the same register's other end, so a run that cut too eagerly would show up as a seeding rate
that never keeps up -- `ppc` falling away from `N0`, which G70 already measures. If G70 breaks while
G72 closes, the threshold is wrong and not the mechanism.
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_07h_bind_cull as H                                            # noqa: E402

NAME = "07i_ramp"


class Rig07i(H.Rig07d):
    """07h, with the shortfall drained at a bounded rate and a length term in the cull."""

    def __init__(self, ramp=2, cull_l0=4.0, **P):
        super().__init__(**P)
        self.ramp = max(1, int(ramp))
        self.cull_l0 = float(cull_l0)
        self._seeded = 0
        self._cull_long = 0
        print(f"[07i] up to {self.ramp} new plaques per cell per frame (a division's shortfall of "
              f"{self.N0} fills in {-(-self.N0 // self.ramp)} frames), culled beyond "
              f"{self.cull_l0} rest lengths whatever the bonds", flush=True)

    # -- arrival -------------------------------------------------------------------------------
    def _seed_into(self, cells, fresh=False):
        """Deferred, EXCEPT at t = 0.

        The first seeding is the initial condition and is not a rate: the sheet starts adhered. Every
        later request is dropped on the floor, because `_drain` recomputes the shortfall from
        `ct_face` itself and would otherwise seed it twice.
        """
        if fresh:
            return super()._seed_into(cells, fresh=True)
        return None

    def _drain(self):
        have = torch.bincount(self.ct_face, minlength=self._nF)
        owe = (self.N0 - have).clamp_min(0).clamp(max=self.ramp)
        k = int(owe.sum())
        if not k:
            return 0
        cells = torch.repeat_interleave(torch.arange(self._nF, device=self.dev), owe)
        super()._seed_into(cells.cpu().numpy())
        return k

    # -- departure -----------------------------------------------------------------------------
    def _cull(self):
        """07h's rule, OR simply too long. Both ends report their own count."""
        nb = self.clutch.Nb
        if not nb.numel():
            return 0
        att = (self.x_epi[self.ct_tri] * self.ct_w[:, :, None]).sum(1)
        sep = (self.sheet.x[self.ct_node] - att).norm(dim=1)
        let_go = (nb < self.cull_below) & (sep > self.bind_max * self.clutch.l0)
        too_long = sep > self.cull_l0 * self.clutch.l0
        gone = let_go | too_long
        if not bool(gone.any()):
            return 0
        self._cull_long += int((too_long & ~let_go).sum())
        keep = ~gone
        back = torch.zeros(self._nF, device=self.dev, dtype=self.dtype)
        back.index_add_(0, self.ct_face[gone], nb[gone])
        self.clutch.Nf = self.clutch.Nf + back            # conserved: G75 still has to hold
        self.ct_node, self.ct_face = self.ct_node[keep], self.ct_face[keep]
        self.ct_w, self.ct_tri = self.ct_w[keep], self.ct_tri[keep]
        self.clutch.Nb, self.clutch.D = nb[keep], self.clutch.D[keep]
        return int(gone.sum())

    def _epi_anchor(self, t):
        # cull and refine first (the parent), then fill: a plaque seeded this frame binds the mesh
        # this frame ends with, not the one the refinement replaced
        tgt = super()._epi_anchor(t)
        self._seeded += self._drain()
        return tgt


def main():
    H.build(Rig07i, default_name=NAME,
            add_args=lambda ap: (ap.add_argument("--ramp", type=int, default=2),
                                 ap.add_argument("--cull-l0", dest="cull_l0", type=float,
                                                 default=4.0)),
            pass_args=("ramp", "cull_l0"),
            extra=dict(seeding="staggered: at most `ramp` new plaques per cell per frame",
                       cull="no bonds and out of reach, OR beyond cull_l0 rest lengths"))


if __name__ == "__main__":
    main()
