"""proposer -- an interchangeable SEARCH ACCELERATOR for Loop I. The methodology does not depend on it:
Loop I explores the composition edit-tree, and *how* it orders candidate edits (BFS, UCB, evolution,
this surrogate) is a replaceable implementation detail. Here a RandomForest is trained on the archive
so far -- composition structural encoding -> emergence rate -- and used to RANK the legal one-edit
children of a node, so the frontier is expanded best-first instead of breadth-first. It proposes; the
real simulator + metric decide (its predictions are never trusted as evidence).
"""
from __future__ import annotations
import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:                                              # graceful: fall back to BFS ordering
    RandomForestRegressor = None


class Surrogate:
    def __init__(self):
        self.rf = None

    def fit(self, encodings, rates):
        if RandomForestRegressor is None or len(encodings) < 6:
            self.rf = None
            return self
        self.rf = RandomForestRegressor(n_estimators=80, min_samples_leaf=1,
                                        random_state=0).fit(np.array(encodings), np.array(rates))
        return self

    def predict(self, g):
        if self.rf is None:
            return 0.0
        return float(self.rf.predict(g.encode()[None])[0])

    def rank_edits(self, g, edits_children):
        """edits_children: list of (edit_label, child_graph). Return them ordered by predicted
        emergence (descending). With no model yet, order is preserved (BFS)."""
        if self.rf is None:
            return list(edits_children)
        return sorted(edits_children, key=lambda ec: -self.predict(ec[1]))
