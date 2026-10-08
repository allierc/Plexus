"""trainer.rollout's clock is a recurrent network's: h_k = h_{k-1} + dt u_k, y_k = g(h_k) (2026-10-08).

A one-cell integrator, dx/dt = u, driven by a known u: trace frame k must be dt * (u_0 + ... + u_k), whether the
observed block is integrated by the engine (the cell's voltage) or written in place by a `readout` from it
(`integration: none`). Before the fix the drive reached the dynamics one tick late (`on_frame` fires at the END of a
tick) and the readout trace one more: the HD twin with the reference's trained weights lost 0.6 deg of heading RMSE.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from plexus import engine
from plexus import trainer as T

MODEL = """
general: {name: clock, seed: 0, n_frames: 6, dt: 0.1, dim: 2, world: [1.0, 1.0], boundary: wall}
sets:
  root: {n: 1}
  drive:
    parent: root
    per_parent: 1
    state:
      signal: {width: 1, integration: none, boundary: free}
  cell:
    parent: root
    per_parent: 1
    state:
      voltage: {width: 1, role: coordinate, integration: first_order, boundary: free}
  out:
    parent: root
    per_parent: 1
    state:
      value: {width: 1, integration: none, boundary: free}
  inp: {parent: root, edge_set: true, entity: connection, pre: drive, post: cell, edges: [[0, 0, 1.0]]}
  rd: {parent: root, edge_set: true, entity: connection, pre: cell, post: out, edges: [[0, 0, 1.0]]}
fields: {}
operators:
- {op: project, at: cell, edge_set: inp, block: signal}
- {op: readout, at: out, edge_set: rd, block: voltage, into: value}
schedule: [project, readout]
"""


@pytest.mark.parametrize("observe", [("cell", "voltage"), ("out", "value")])
@pytest.mark.parametrize("batch", [1, 2])
def test_trace_k_has_seen_u_0_to_k(tmp_path, observe, batch):
    engine.quiet(True)
    path = tmp_path / "clock.yaml"
    path.write_text(MODEL)
    sim = T._model({"model": str(path)}, train=False)
    u = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])[:, None]            # [T, 1]
    sim.n_frames = len(u)
    U = u[None].expand(batch, -1, -1).clone() * torch.arange(1, batch + 1.0)[:, None, None]
    task = {"drive": {"set": "drive", "block": "signal"}, "observe": {"set": observe[0], "block": observe[1]}}
    _, y = T.rollout(sim, T.Learnables([]), U if batch > 1 else U[0], task, grad=False)
    y = (y if batch > 1 else y[None])[..., 0].numpy()                      # [B, frames]
    want = 0.1 * np.cumsum(U[..., 0].numpy(), axis=1)                        # dt * (u_0 + ... + u_k)
    np.testing.assert_allclose(y[:, :len(u)], want, rtol=1e-6)
