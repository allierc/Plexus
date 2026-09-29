"""Train a Plexus model for a task. The engine simulates; this module trains.

    python Plexus_Main.py -o train_test_analyse t1_integrator_perfect_zf285
    python -m plexus.trainer -o train test analyse config/training/neural/t1_integrator_perfect_zf285.yaml

A TRAINING SPEC IS A DECLARATION IN THREE PARTS, beside a model it never edits (plexus2.tex,
"Training differentiable models"):

    model:      the forward spec -- exactly the file `Plexus_Main.py -o generate` runs
    learnable:  what is free, and in which representation
    task:       what is observed, against what reference, driven by what, scored by which loss
    training:   how the gradient is obtained and used -- the one part that is not a claim

    name: t1_integrator_perfect_zf285
    model: config/neural/zf_circuit_285.yaml
    learnable:
      - {block: w, of: synapse}
      - {block: tau, of: neuron, bounds: [0.005, 10.0]}
    task:
      reference: {corpus: t1_integrator_perfect, n_train: 512, n_val: 60, n_test: 80}
      drive: {set: sensor, block: signal}
      observe: {set: output, block: value, channel: 0}
      loss: mse
    training: {optimizer: adam, lr: 0.005, schedule: cosine, clip: 1.0, epochs: 100, batch: 32,
               seed: 0, horizon: [0.25, 0.5, 0.75, 1.0], horizon_min: 60}

THE TRAINER OWNS THE PARAMETERS; THE ENGINE NEVER SEES A LEARNABLE. `engine.run` builds a fresh
Hierarchy -- operators included -- on every call, which is what makes a run reproducible, so a
tensor made inside one rollout does not survive into the next. The parameters therefore live
here, in `Learnables`, and are written into the seeded state by the engine's one neutral hook,
`on_seeded`, before the batch axis exists: every rollout of a run writes the SAME leaves, and
`repeat` along the batch axis carries their gradient back from all B trials. The engine attaches
no meaning to the write.

EVERY KEY IS READ OR REFUSED. A key that nothing reads -- a representation named before it is
implemented, a loss misspelt -- would be dropped, and the run would train something other than
what its spec says. `load` refuses it instead, with the list of what is read.

What is implemented, and nothing else: learnable STATE BLOCKS in the `tensor` representation (one
free value per element, starting at the model's own value), with an optional step size and
bounds per block; a CORPUS reference (`plexus.tasks`: stimulus, teacher, condition grid, disjoint
splits) driven through a state block and observed through another; the `mse` loss; Adam with a
cosine schedule, gradient clipping, a horizon curriculum and a divergence guard.

Output, one folder per run under the model's own folder, mirroring `config/training/<model>/`:

    log/training/<model>/<name>/
        config.yaml                  the training spec, copied
        model.yaml                   the forward spec it trained
        models/best.pt               the learnables, by validation error
        results/report.json          what training did
        results/<name>_test.json     the held-out rollout
        results/<name>_test.png      the analysis figure
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

import plexus.operators                                          # noqa: F401  self-registers
from plexus import engine
from plexus.models.registry import get_contract
from plexus.paths import get_repo_root, log_path, training_spec
from plexus.schema import load as load_model
from plexus.tasks.generate import task_dir
from plexus.tasks.trainer import load_split, n_condition_cells, with_context


# ============================================================================== the declaration
_KEYS = {
    "top": {"name", "model", "learnable", "task", "training"},
    "learnable": {"block", "of", "with", "lr", "bounds", "over", "K", "extent", "param", "op",
                  "prior"},
    "task": {"reference", "drive", "observe", "loss", "settle_s", "u_weight", "mask"},
    "reference": {"corpus", "n_train", "n_val", "n_test", "context", "shape", "size", "centre",
                  "recording", "beats", "points"},
    "drive": {"set", "block", "prescribe", "width"},
    "observe": {"set", "block", "channel", "unit", "measure", "grid", "of"},
    "training": {"optimizer", "lr", "lr_min", "lr_min_frac", "schedule", "clip", "epochs", "batch",
                 "seed", "horizon", "horizon_min", "snapshot_every", "guard", "stages", "render",
                 "iters", "save_every", "anneal"},
    "term": {"term", "weight", "reduction"},
    "stage": {"resolution", "iters"},
}
REPRESENTATIONS = ("tensor", "lattice")
REFERENCES = ("corpus", "shape", "recording")
_LOSS_FOR = {"corpus": ("mse",), "shape": ("log_mse",), "recording": ("affine_mse",)}
PRIORS = ("shrink", "shrink_to_mean", "smooth", "l1", "l2", "group_l1", "sign")
# THE EVIDENCE TERMS each reference kind can score, by name. `task.loss` is one name or a list of
# {term, weight, reduction}; the loop computes the quantities, `_objective` weighs and records them.
TERMS = {"corpus": ("mse",), "shape": ("log_mse", "volume", "point_mse"),
         "recording": ("affine_mse", "affine_A", "affine_u")}
REDUCTIONS = ("mean", "norm2", "huber", "relative_l2")
OPTIMIZERS = ("adam",)
SCHEDULES = ("cosine", "none")
GUARDS = ("restore_and_halve", "none")


def _refuse_unread(where, d, allowed):
    extra = sorted(set(d) - allowed)
    if extra:
        raise ValueError(f"{where}: key(s) {extra} are read by nothing; the keys read here are "
                         f"{sorted(allowed)}. An unread key would be dropped and the run would "
                         f"train something other than what the spec says.")


def _loss_terms(task, kind):
    """`task.loss` as a list of {term, weight, reduction}: one name is one term of weight 1."""
    L = task.get("loss", _LOSS_FOR[kind][0])
    if isinstance(L, str):
        return [{"term": L}]
    if not isinstance(L, list) or not all(isinstance(x, dict) and "term" in x for x in L):
        raise ValueError("task.loss is a term name or a list of {term, weight, reduction}")
    return [dict(x) for x in L]


def _anneal(spec, t):
    """1 - exp(-rate t), the factor every prior carries; 1 when nothing is annealed."""
    an = spec["training"].get("anneal")
    return 1.0 if an is None else float(1.0 - math.exp(-float(an["rate"]) * float(t)))


def _objective(spec, learn, evidence, t, parts, prior=True):
    """THE ONE PLACE A LOSS IS ASSEMBLED: the task's weighted evidence terms plus the learnables'
    priors (annealed by `t`), every term written into `parts` under its own name so the history
    shows what entered the gradient. `evidence` maps a term name to a function of its reduction."""
    tot = None
    for term in _loss_terms(spec["task"], spec["_kind"]):
        v = evidence[term["term"]](term.get("reduction", "mean"))
        w = float(term.get("weight", 1.0))
        v = v if w == 1.0 else w * v
        parts[f"loss.{term['term']}"] = float(v.detach())
        tot = v if tot is None else tot + v
    pr = learn.prior(scale=_anneal(spec, t), parts=parts) if prior else None
    return tot if pr is None else tot + pr


def _reduce(r, reduction, target=None):
    """A residual to a scalar -- connectome-gnn's fit_residual_loss reductions."""
    if reduction == "mean":
        return (r ** 2).mean()
    if reduction == "norm2":
        return r.norm(2)
    if reduction == "huber":
        return torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction="mean", delta=1.0)
    return r.norm(2) / (target.norm(2) + 1e-8)


def load(path_or_name) -> dict:
    """Read and validate a training spec. Returns the dict with `_path` and `_model_dir` added."""
    path = training_spec(path_or_name)
    with open(path) as f:
        s = yaml.safe_load(f)
    for k in ("name", "model", "learnable", "task", "training"):
        if k not in s:
            raise ValueError(f"{path}: a training spec needs `{k}:` -- the model, what is "
                             f"learnable, the task and the training scheme.")
    _refuse_unread(path, s, _KEYS["top"])
    if not os.path.isabs(s["model"]):                    # repo-relative, like every spec path
        s["model"] = os.path.join(get_repo_root(), s["model"])
    raw = yaml.safe_load(open(s["model"]))
    sets = raw.get("sets") or {}
    ops = {o.get("op") for o in (raw.get("operators") or [])}
    an = s["training"].get("anneal")
    if an is not None and (not isinstance(an, dict) or set(an) - {"rate", "per"} or "rate" not in an
                           or an.get("per", "epoch") not in ("epoch", "iter")):
        raise ValueError(f"{path}: training.anneal is {{rate: r, per: epoch|iter}} -- every prior is "
                         f"multiplied by 1 - exp(-r t) so the fit leads and the priors follow")
    for i, e in enumerate(s["learnable"]):
        _refuse_unread(f"{path}: learnable[{i}]", e, _KEYS["learnable"])
        for k in (e.get("prior") or {}):
            if k not in PRIORS:
                raise ValueError(f"{path}: learnable[{i}] prior {k!r}; implemented: {list(PRIORS)}")
        if "param" in e or "op" in e:
            # A PARAMETER OF AN ACTIVITY: the operator is stated, one of its constants is free.
            if "param" not in e or "op" not in e or "block" in e or "of" in e:
                raise ValueError(f"{path}: learnable[{i}] is EITHER {{block:, of:}} (a state of a "
                                 f"set) OR {{param:, op:}} (a parameter of an activity)")
            if e["op"] not in ops:
                raise ValueError(f"{path}: learnable[{i}] frees `{e['op']}.{e['param']}`, but no "
                                 f"operator line of the model is `{e['op']}` ({sorted(ops)})")
            continue
        for k in ("block", "of"):
            if k not in e:
                raise ValueError(f"{path}: learnable[{i}] needs `{k}:`")
        if e["of"] not in sets:
            raise ValueError(f"{path}: learnable[{i}] fits `{e['of']}.{e['block']}`, but "
                             f"`{e['of']}` is not a set of {s['model']} ({sorted(sets)})")
        rep = e.get("with", "tensor")
        if rep not in REPRESENTATIONS:
            raise ValueError(f"{path}: learnable[{i}] asks for representation {rep!r}; "
                             f"implemented: {list(REPRESENTATIONS)}.")
        if rep == "lattice":
            if e.get("over") != "material":
                raise ValueError(f"{path}: learnable[{i}] is a lattice, which is read at each "
                                 f"element's MATERIAL coordinate -- say `over: material`")
            if "K" not in e or len(e.get("extent") or []) != 4:
                raise ValueError(f"{path}: learnable[{i}] lattice needs `K:` (nodes per axis) and "
                                 f"`extent: [cx, cy, cz, half_width]` in world units")
        elif any(k in e for k in ("over", "K", "extent")):
            raise ValueError(f"{path}: learnable[{i}] sets lattice keys on a {rep!r} representation")
    t = s["task"]
    _refuse_unread(f"{path}: task", t, _KEYS["task"])
    ref = t.get("reference") or {}
    kinds = [k for k in REFERENCES if k in ref]
    if len(kinds) != 1:
        raise ValueError(f"{path}: task.reference must name exactly one of {list(REFERENCES)} -- a "
                         f"corpus under graphs_data/task/ or a shape from the library; it names {kinds}")
    kind = kinds[0]
    parts = ("drive", "observe") if kind in ("corpus", "recording") else ("observe",)
    for k in ("reference",) + parts:
        if k not in t:
            raise ValueError(f"{path}: task needs `{k}:`")
        _refuse_unread(f"{path}: task.{k}", t[k], _KEYS[k])
    for part in parts:
        # THE EDGE BAND PRESCRIBES A SET'S POSITIONS AND VELOCITIES, so it names the set only.
        for k in (("set",) if (kind == "recording" and part == "drive") else ("set", "block")):
            if k not in t[part]:
                raise ValueError(f"{path}: task.{part} needs `{k}:`")
        if t[part]["set"] not in sets:
            raise ValueError(f"{path}: task.{part}.set {t[part]['set']!r} is not a set of the model")
    for j, term in enumerate(_loss_terms(t, kind)):
        _refuse_unread(f"{path}: task.loss[{j}]", term, _KEYS["term"])
        if term["term"] not in TERMS.get(kind, _LOSS_FOR[kind]):
            raise ValueError(f"{path}: task.loss term {term['term']!r} does not score a {kind} "
                             f"reference; implemented for it: {list(TERMS.get(kind, _LOSS_FOR[kind]))}")
        if term.get("reduction", "mean") not in REDUCTIONS or (term.get("reduction", "mean") != "mean"
                                                                and term["term"] != "mse"):
            raise ValueError(f"{path}: task.loss term {term['term']!r}: reduction "
                             f"{term.get('reduction')!r}; only `mse` takes one of {list(REDUCTIONS)}")
    tr = s["training"]
    _refuse_unread(f"{path}: training", tr, _KEYS["training"])
    if kind == "shape":
        if "drive" in t:
            raise ValueError(f"{path}: a shape task has no drive -- nothing is fed in while it grows")
        if t["observe"].get("measure") != "grid_mass" or t["observe"].get("grid") not in (raw.get("fields") or {}):
            raise ValueError(f"{path}: a shape task observes `measure: grid_mass` on a `grid:` field "
                             f"of the model ({sorted(raw.get('fields') or {})})")
        if any(x["term"] == "point_mse" for x in _loss_terms(t, kind)) and "/" not in str(ref.get("points", "")):
            raise ValueError(f"{path}: the `point_mse` term scores each particle against ITS OWN target, "
                             f"so task.reference needs `points: <shape>/<part>` (same order as the seed)")
        for k in ("size", "centre"):
            if k not in ref:
                raise ValueError(f"{path}: task.reference needs `{k}:` -- where the shape sits and "
                                 f"how long its longest axis is, in world units")
        if not tr.get("stages"):
            raise ValueError(f"{path}: a shape task trains in `stages:` -- a list of {{resolution, iters}}")
        for j, st in enumerate(tr["stages"]):
            _refuse_unread(f"{path}: training.stages[{j}]", st, _KEYS["stage"])
            for name in (st.get("resolution") or {}):
                if name not in sets and name not in (raw.get("fields") or {}):
                    raise ValueError(f"{path}: training.stages[{j}].resolution names {name!r}, "
                                     f"which is neither a set nor a field of the model")
    elif kind == "recording":
        if not ref.get("beats"):
            raise ValueError(f"{path}: task.reference names the `beats:` it is fitted on")
        if t.get("drive", {}).get("prescribe") != "edge_band" or "width" not in t["drive"]:
            raise ValueError(f"{path}: a recording task drives its `edge_band` of `width:` from the "
                             f"recording -- the tissue beyond the field of view enters there")
        if t["observe"].get("measure") != "cell_affine" or t["observe"].get("of") not in sets:
            raise ValueError(f"{path}: a recording task observes `measure: cell_affine` of the "
                             f"points' parent set `of:`")
        if "iters" not in tr:
            raise ValueError(f"{path}: a recording task trains for `iters:` steps")
    elif any(k in tr for k in ("stages", "render", "lr_min")):
        raise ValueError(f"{path}: `stages`, `render` and `lr_min` belong to a shape task's scheme")
    s["_kind"] = kind
    for k, allowed in (("optimizer", OPTIMIZERS), ("schedule", SCHEDULES), ("guard", GUARDS)):
        if k in tr and tr[k] not in allowed:
            raise ValueError(f"{path}: training.{k} {tr[k]!r}; implemented: {list(allowed)}")
    s["_path"] = os.path.abspath(path)
    s["_model_dir"] = os.path.basename(os.path.dirname(s["_path"]))
    return s


def out_dir(spec, root=None) -> str:
    return os.path.join(root or log_path(), "training", spec["_model_dir"], spec["name"])


# ============================================================================== the learnables
class Learnables:
    """The free quantities of one run, held here across rollouts and written into each one.

    Each entry is a state block of a set. Its parameter is made the FIRST time a rollout is
    seeded, from the value the model put there -- a connectome's measured weights, a declared
    time constant -- so what training does is a departure from the stated model and can be read
    as one. After that, every rollout writes the same leaf back, functionally (clone, assign,
    publish), so the tape connects the parameter to every later use of the state.
    """

    def __init__(self, entries, device="cpu"):
        self.entries = [dict(e) for e in entries]
        self.device = device
        self.p: dict[str, nn.Parameter] = {}
        self._pre: dict[str, torch.Tensor] = {}   # sender index of an edge-set block, for `sign`
        self.scale = 1.0               # a DISPLAY rollout may spread one deformation over more frames

    @staticmethod
    def key(e) -> str:
        return f"{e['op']}.{e['param']}" if "param" in e else f"{e['of']}.{e['block']}"

    def ready(self, H):
        """Parameters of ACTIVITIES, set on the operator instances once they exist (`on_ready`).
        Made the first time from the value the model gave the operator -- its `fit:` file or its
        inline numbers -- and handed to every later instance, so each rollout reads the same leaf."""
        for e in self.entries:
            if "param" not in e:
                continue
            insts = [o for n, o in zip(H.operator_names, H.operators) if n == e["op"]]
            if len(insts) != 1:
                raise ValueError(f"learnable `{self.key(e)}`: {len(insts)} instances of {e['op']!r}")
            par = self.p.get(self.key(e))
            if par is None:
                v = getattr(insts[0], e["param"], None)
                if not torch.is_tensor(v):
                    raise ValueError(f"learnable `{self.key(e)}`: the operator holds no tensor "
                                     f"{e['param']!r} to start from (it has {type(v).__name__})")
                par = self.p[self.key(e)] = nn.Parameter(v.detach().clone().to(self.device))
            setattr(insts[0], e["param"], par)

    def prior(self, scale=1.0, parts=None):
        """The priors declared WITH the learnables -- claims about the unknown, not the evidence.

            shrink: l          l * mean(x^2)                        toward zero
            shrink_to_mean: l  l * mean((x - mean x)^2)             toward uniform
            smooth: l          l * sum of squared steps along the last axis
            l1: l              l * ||x||_1                          sparse (connectome-gnn coeff_*_L1)
            l2: l              l * ||x||_2                          small (coeff_*_L2) -- a NORM, not a mean
            group_l1: l        l * sum_c ||x[..., c]||_2            whole columns to zero (group lasso)
            sign: l            l * ||std_i tanh(10 x_e)||_2         Dale: one sign per sender i, over
                                                                    the edges it sends (edge-set blocks)

        `scale` is the annealing factor; `parts`, a dict, receives every term by name.
        """
        tot = None
        for e in self.entries:
            x = self.p.get(self.key(e))
            for kind, lam in (e.get("prior") or {}).items():
                lam = float(lam) * float(scale)
                if kind == "shrink":
                    term = x.pow(2).mean()
                elif kind == "shrink_to_mean":
                    term = (x - x.mean()).pow(2).mean()
                elif kind == "smooth" and e.get("with") == "lattice":
                    # IN SPACE, not across components: a lattice is stored flat as [K^3, width], so
                    # the last axis is the rate's six entries. Steps between neighbouring NODES along
                    # x, y and z of the [K, K, K, width] view (found by the exp04 session).
                    K = int(e["K"])
                    g = x.reshape(K, K, K, -1)
                    term = sum((g.narrow(ax, 1, K - 1) - g.narrow(ax, 0, K - 1)).pow(2).sum() for ax in range(3))
                elif kind == "smooth":
                    term = (x[..., 1:] - x[..., :-1]).pow(2).sum()
                elif kind == "l1":
                    term = x.abs().sum()
                elif kind == "l2":
                    term = x.norm(2)
                elif kind == "group_l1":
                    term = x.reshape(-1, x.shape[-1]).norm(2, dim=0).sum()
                else:
                    pre = self._pre.get(self.key(e))
                    if pre is None:
                        raise ValueError(f"prior `sign` on `{self.key(e)}`: it groups an edge set's "
                                         f"weights by sender, and `{e.get('of')}` is not an edge set")
                    v = torch.tanh(10.0 * x.reshape(-1))
                    n = int(pre.max()) + 1
                    cnt = torch.zeros(n, device=v.device).index_add(0, pre, torch.ones_like(v))
                    mu = torch.zeros(n, device=v.device).index_add(0, pre, v) / cnt.clamp(min=1)
                    m2 = torch.zeros(n, device=v.device).index_add(0, pre, v * v) / cnt.clamp(min=1)
                    term = ((m2 - mu * mu).clamp(min=0) * (cnt > 1)).sqrt().norm(2)
                tot = lam * term if tot is None else tot + lam * term
                if parts is not None:
                    parts[f"prior.{self.key(e)}.{kind}"] = float((lam * term).detach())
        return tot

    def inject(self, H):
        for e in self.entries:
            if "param" in e:
                continue
            lvl = H.level(e["of"])
            if e["block"] not in lvl.state_schema:
                raise ValueError(f"learnable `{self.key(e)}`: {e['of']!r} has no block "
                                 f"{e['block']!r} (it has {sorted(lvl.state_schema)})")
            a, b = lvl.state_schema[e["block"]]
            par = self.p.get(self.key(e))
            if "sign" in (e.get("prior") or {}) and self.key(e) not in self._pre \
                    and getattr(lvl, "is_edge_set", False):
                self._pre[self.key(e)] = lvl.incidence("pre").detach().long()
            if e.get("with", "tensor") == "lattice":
                if par is None:
                    # A LATTICE STARTS WHERE THE MODEL DOES ONLY WHEN THE MODEL SAYS "NOTHING": a
                    # uniform zero block is exactly the zero lattice, and anything else would need
                    # a fit to reproduce it, which is not implemented and so is refused.
                    if float(lvl.state[:, a:b].abs().max()) > 0:
                        raise ValueError(f"learnable `{self.key(e)}`: a lattice starts from a zero "
                                         f"block; the model seeds a non-zero one")
                    K = int(e["K"])
                    par = self.p[self.key(e)] = nn.Parameter(
                        torch.zeros(K ** 3, b - a, device=lvl.state.device, dtype=lvl.state.dtype))
                val = self._lattice(e, par, lvl.get("pos").detach())
            else:
                if par is None:
                    par = self.p[self.key(e)] = nn.Parameter(lvl.state[:, a:b].detach().clone())
                val = par
            st = lvl.state.clone()
            st[:, a:b] = val if self.scale == 1.0 else val * self.scale
            lvl.state = st

    @staticmethod
    def _lattice(e, par, X0):
        """The block's value at each element: trilinear in a K^3 cube of control nodes, read at the
        element's MATERIAL coordinate X0 -- its position at set-up, so the field is Lagrangian and
        means the same thing at any number of elements.

            a(X0) = sum_c w_c(X0) theta_c        w = trilinear weights, c = the 8 corners
        """
        K = int(e["K"])
        cx, cy, cz, half = (float(v) for v in e["extent"])
        c = torch.tensor([cx, cy, cz], device=X0.device, dtype=X0.dtype)
        u = (((X0 - c) / (2.0 * half) + 0.5) * (K - 1)).clamp(0.0, K - 1.001)
        b = u.floor().long()
        f = u - b.to(u.dtype)
        val = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (((1 - f[:, 0]) if dx == 0 else f[:, 0])
                         * ((1 - f[:, 1]) if dy == 0 else f[:, 1])
                         * ((1 - f[:, 2]) if dz == 0 else f[:, 2]))
                    i = (((b[:, 0] + dx).clamp(0, K - 1) * K + (b[:, 1] + dy).clamp(0, K - 1)) * K
                         + (b[:, 2] + dz).clamp(0, K - 1))
                    val = val + w[:, None] * par[i]
        return val

    def parameters(self) -> list:
        return [self.p[self.key(e)] for e in self.entries]

    def groups(self, lr) -> list:
        return [{"params": [self.p[self.key(e)]], "lr": float(e.get("lr", lr))}
                for e in self.entries]

    @torch.no_grad()
    def clamp_(self):
        for e in self.entries:
            if "bounds" in e:
                lo, hi = (None if v is None else float(v) for v in e["bounds"])
                self.p[self.key(e)].clamp_(min=lo, max=hi)

    def snapshot(self) -> dict:
        return {k: v.detach().clone() for k, v in self.p.items()}

    @torch.no_grad()
    def restore(self, values: dict):
        for k, v in values.items():
            if k in self.p:
                self.p[k].copy_(v.to(self.p[k].device))
            else:
                self.p[k] = nn.Parameter(v.to(self.device).clone())


# ============================================================================== the task
def write_drive(H, frame, u, drive_set, drive_block):
    """Write frame `frame` of the stimulus into the drive set's block.

    ONE CHANNEL PER SENSORY ELEMENT, not one value broadcast over all of them. A corpus with C
    channels -- a stimulus plus, for a teacher-varying grid, a one-hot of the condition cell --
    has to arrive on C DIFFERENT lines or the circuit cannot tell them apart. Element i takes
    channel i; a drive set wider than the corpus leaves its spare lines as they were seeded.
    """
    lvl = H.level(drive_set)
    a, b = lvl.state_schema[drive_block]
    t = min(frame, u.shape[-2] - 1)
    st = lvl.state.clone()
    C = u.shape[-1]
    if C == 1:
        st[..., a:b] = u[..., t, :].unsqueeze(-2).expand(*st.shape[:-1], b - a)
    else:
        if lvl.n < C:
            raise ValueError(
                f"the corpus has {C} input channels but `{drive_set}` has only {lvl.n} "
                f"element(s). Give the drive set at least one element per channel -- with a "
                f"context one-hot the channel count is 1 + the number of condition cells.")
        st[..., :C, a:a + 1] = u[..., t, :].unsqueeze(-1)
    lvl.state = st


def rollout(sim, learn, u, task, device="cpu", grad=True, watch=None):
    """Drive the model with `u` ([T, C] or [B, T, C]) and return (H, observed trace).

    The trace is the observed set's element 0 at the end of every tick, trial-major like the
    corpus: [T, w] or [B, T, w]. `watch(H)` is called at the same moment, for a caller that needs
    more of the state than the observable (the analysis reads the voltages).
    """
    drv, obs = task["drive"], task["observe"]
    batch = int(u.shape[0]) if u.dim() == 3 else 1
    if u.dim() == 3 and batch == 1:                        # the engine runs unbatched at B = 1
        u = u[0]
    trace = []

    def hook(H, frame):
        write_drive(H, frame, u, drv["set"], drv["block"])
        trace.append(H.level(obs["set"]).get(obs["block"])[..., 0, :].clone())
        if watch is not None:
            watch(H)

    H, _ = engine.run(sim, device=device, progress=False, grad=grad, on_frame=hook,
                      batch=batch, on_seeded=learn.inject)
    if not trace:
        return H, None
    y = torch.stack(trace)
    return H, y.transpose(0, 1) if batch > 1 else y


def _mse(y, target, ch):
    """Squared error, aligned at frame 0 and truncated to the shorter series; batch-transparent.

    The mean is over frames AND trials. Indexed from the right, so [T, w] against [T, 1] and
    [B, T, w] against [B, T, 1] are the same call.
    """
    n = min(y.shape[-2], target.shape[-2])
    return ((y[..., :n, ch] - target[..., :n, 0]) ** 2).mean()


def _log_mse(m, m_ref):
    """Squared error of log(1 + nodal count) on the grid -- the Eulerian shape loss.

    A target shape has no particle correspondence, so a position loss has nothing to match where
    the two shapes differ; comparing where material IS on a grid does. `log` because the two
    fields differ by orders of magnitude between "material" and "none".
    """
    return ((torch.log1p(m) - torch.log1p(m_ref)) ** 2).mean()


LOSSES = {"mse": _mse, "log_mse": _log_mse}


def _corpus(spec):
    return spec["task"]["reference"]["corpus"]


def _unit(spec):
    u = spec["task"]["observe"].get("unit")
    return (f" {u}", f" {u}^2") if u else ("", "")


def _data(spec, split, n_cond, device):
    U, Y, c = load_split(_corpus(spec), split, device)
    return with_context(U, c, n_cond), Y, c


def _model(spec, train=True, resolution=None, n_frames=None):
    """The model, as TRAINING or as DISPLAY runs it. Same file, same law, two numerical choices.

    TRAINING picks each operator's `differentiable` implementation where the model pins none --
    the default MPM bodies write in place, which a captured CUDA graph needs and autograd cannot
    have. DISPLAY leaves the choice to the engine, which picks its fastest bodies (warp on CUDA).

    SUBSTEP CAPTURE IS OFF IN BOTH, for two measured reasons. Under a tape, a replayed graph writes
    its static buffers in place, so a tensor autograd saved at tick 0 is a different tensor at
    backward time ("[5000] ... at version 2; expected version 1"). And with warp bodies, a captured
    substep does not see a deformation gradient written by an operator OUTSIDE the block: the same
    trained morph ended as the untouched ball (10.85 x 10.67 x 10.65 um) with capture and as the
    trained cow (14.13 x 7.00 x 4.32 um) without it, or with the torch bodies. That is an engine
    defect, not fixed here; `resolution` resizes sets (`n`) and fields (`n_grid`) for a stage.
    """
    sim = load_model(spec["model"])
    if sim.learnable:
        raise ValueError(f"{spec['model']} declares `learnable:` itself. What is learnable belongs "
                         f"to the training spec; the model file stays the forward description.")
    for name, v in (resolution or {}).items():
        if name in sim.sets:
            sim.sets[name]["n"] = int(v)
        else:
            sim.fields[name]["n_grid"] = int(v)
    if n_frames is not None:
        sim.n_frames = int(n_frames)
    if train:
        for o in sim.operators:
            if o.impl is None and "differentiable" in get_contract(o.op).implementations:
                o.impl = "differentiable"
    for blk in sim.schedule:
        if isinstance(blk, dict) and "steps" in blk:
            blk["capture"] = False
    return sim


# ============================================================================== train
def train(spec, device="cpu", root=None):
    if spec.get("_kind") == "shape":
        return _train_shape(spec, device, root)
    if spec.get("_kind") == "recording":
        return _train_recording(spec, device, root)
    tr, task = spec["training"], spec["task"]
    ref = task["reference"]
    torch.manual_seed(int(tr.get("seed", 0)))
    out = out_dir(spec, root)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    shutil.copyfile(spec["_path"], os.path.join(out, "config.yaml"))
    shutil.copyfile(spec["model"], os.path.join(out, "model.yaml"))
    print(f"[run] {spec['name']} -> {out}")

    corpus = _corpus(spec)
    n_cond = n_condition_cells(corpus) if ref.get("context", True) else 1
    U, Y, cU = _data(spec, "train", n_cond, device)
    has_val = os.path.isdir(os.path.join(task_dir(corpus), "val"))
    Uv, Yv, cV = _data(spec, "val", n_cond, device) if has_val else (U[:8], Y[:8], cU[:8])
    if n_cond > 1:
        print(f"[data] {n_cond} condition cells -> a one-hot context channel is appended "
              f"({U.shape[2] - n_cond} -> {U.shape[2]} inputs)")
    n_tr = min(int(ref.get("n_train", 32)), U.shape[0])
    n_va = min(int(ref.get("n_val", 8)), Uv.shape[0])
    print(f"[data] corpus {corpus}: using {n_tr} of {U.shape[0]} train trials, "
          f"{n_va} val  ({U.shape[1]} frames each)")

    engine.quiet(True)
    sim = _model(spec)
    learn = Learnables(spec["learnable"], device)
    # THE FIRST ROLLOUT MAKES THE PARAMETERS, from the seeded state, and proves the drive and the
    # observable name real blocks before an optimiser exists to hide a failure behind a loss.
    rollout(sim, learn, U[0], task, device, grad=True)
    params = learn.parameters()
    n_par = sum(p.numel() for p in params)
    print(f"[fit] {len(params)} tensor(s), {n_par} values: " + ", ".join(learn.p))

    epochs = int(tr.get("epochs", 30))
    batch = max(1, int(tr.get("batch", 8)))
    T_full = int(sim.n_frames)
    # TRUNCATED BPTT OVER A GROWING PREFIX: short horizons first, where the gradient still reaches
    # the start of the trajectory, then longer. Duplicates dropped, so a floor that collapses the
    # early stages costs no epochs.
    horizon = sorted({max(int(tr.get("horizon_min", 60)), int(T_full * f))
                      for f in tr.get("horizon", [0.125, 0.25, 0.5, 0.75, 1.0])})
    print(f"[fit] horizon curriculum {horizon} frames of {T_full};  "
          f"gradient averaged over {batch} trials per step, ONE rollout per step")
    lr = float(tr.get("lr", 1e-2))
    opt = torch.optim.Adam(learn.groups(lr), lr=lr)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
           if tr.get("schedule", "cosine") == "cosine" else None)
    clip = float(tr.get("clip", 1.0))
    guard = tr.get("guard", "restore_and_halve")
    loss_fn = _mse                                         # the validation METRIC; the loss is _objective
    ch = int(task["observe"].get("channel", 0))
    U1, U2 = _unit(spec)
    hist, best, best_state, t0 = [], np.inf, None, time.time()
    last_ok = learn.snapshot()

    for ep in range(epochs):
        h = horizon[min(len(horizon) - 1, int(len(horizon) * ep / max(epochs, 1)))]
        sim.n_frames = h
        perm = torch.randperm(n_tr)
        tot, n_step, n_bad, terms = 0.0, 0, 0, {}
        for k in range(0, n_tr - batch + 1, batch):
            idx = perm[k:k + batch]
            _, y = rollout(sim, learn, U[idx], task, device, grad=True)
            Yb = Y[idx]
            nf = min(y.shape[-2], Yb.shape[-2])
            r = y[..., :nf, ch] - Yb[..., :nf, 0]
            parts = {}
            loss = _objective(spec, learn, {"mse": lambda red: _reduce(r, red, Yb[..., :nf, 0])},
                              ep, parts)
            for kk, vv in parts.items():
                terms[kk] = terms.get(kk, 0.0) + vv
            if not torch.isfinite(loss) and guard == "restore_and_halve":
                # A DIVERGED ROLLOUT ENDS THE STEP, NOT THE RUN: the last finite values come back
                # and every step size halves. Without it one NaN poisons Adam's moments for good.
                learn.restore(last_ok)
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                n_bad += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            learn.clamp_()
            last_ok = learn.snapshot()
            tot += float(loss.detach()); n_step += 1
        if sch is not None:
            sch.step()
        sim.n_frames = T_full                              # always scored on the full trial
        with torch.no_grad():
            _, yv = rollout(sim, learn, Uv[:n_va], task, device, grad=False)
            v = float(loss_fn(yv, Yv[:n_va], ch))
        if int(tr.get("snapshot_every", 0)) and ep % int(tr["snapshot_every"]) == 0:
            from plexus.tasks import plot_trainer as PT
            PT.snapshot(out, ep, ep * max(n_step, 1),
                        Uv[:4, :, :1].cpu().numpy(), Yv[:4].cpu().numpy(),
                        yv[:4].detach().cpu().numpy(), dt=float(sim.dt), unit=U1.strip(),
                        title=f"{spec['name']}  epoch {ep}")
        tr_mse = tot / max(n_step, 1)
        hist.append({"epoch": ep, "horizon": h, "train_mse": tr_mse, "val_mse": v,
                     "diverged_steps": n_bad,
                     "terms": {kk: vv / max(n_step, 1) for kk, vv in terms.items()}})
        print(f"  ep {ep:3d}  horizon {h:4d}  train {tr_mse:10.5f}  val {v:10.5f}{U2}"
              f"   (|err| {np.sqrt(v):.4f}{U1} rms)" + (f"  [{n_bad} diverged]" if n_bad else ""))
        if v < best:
            best, best_state = v, learn.snapshot()

    torch.save({"fitted": best_state, "model": spec["model"], "task": task,
                "learnable": spec["learnable"], "n_cond": int(n_cond)},
               os.path.join(out, "models", "best.pt"))
    rep = {"name": spec["name"], "model": spec["model"], "corpus": corpus,
           "n_params": n_par, "n_trials": n_tr, "epochs": epochs, "batch": batch,
           "best_val_mse": best, "history": hist,
           "target_variance": float((Y[:n_tr] ** 2).mean()),
           "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(out, "results", "report.json"), "w"), indent=2)
    print(f"[done] best val {best:.5f}{U2} ({np.sqrt(best):.4f}{U1} rms) "
          f"({best / rep['target_variance']:.4f} of target variance) in {rep['seconds']:.0f} s")
    return out


def _restore(spec, device="cpu", root=None):
    """The model, the trained learnables and the run folder -- everything a held-out rollout needs."""
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    return _model(spec), learn, ck, out


# ============================================================================== test
def test(spec, device="cpu", root=None):
    """Roll the kept checkpoint out on held-out trials. One number per trial, never only a mean."""
    if spec.get("_kind") == "shape":
        return _test_shape(spec, device, root)
    if spec.get("_kind") == "recording":
        return _test_recording(spec, device, root)
    engine.quiet(True)
    sim, learn, ck, out = _restore(spec, device, root)
    task, corpus = spec["task"], _corpus(spec)
    split = "test" if os.path.isdir(os.path.join(task_dir(corpus), "test")) else "train"
    U, Y, cond = _data(spec, split, int(ck.get("n_cond", 1)), device)
    n = min(int(task["reference"].get("n_test", 24)), U.shape[0])
    ch = int(task["observe"].get("channel", 0))
    with torch.no_grad():
        _, Yp = rollout(sim, learn, U[:n], task, device, grad=False)
    per = [float(_mse(Yp[i], Y[i], ch)) for i in range(n)]
    traces = [Yp[i, :, ch].cpu().numpy() for i in range(min(n, 6))]
    var = float((Y[:n] ** 2).mean())
    nn_ = min(Yp.shape[-2], Y.shape[-2])
    err = (Yp[:n, :nn_, ch] - Y[:n, :nn_, 0]).cpu().numpy()
    # THE SETTLED SCORE BESIDE THE FULL ONE: every rollout starts from the model's seeded state,
    # so the opening fraction of a second is the circuit catching up from rest.
    settle_s = float(task.get("settle_s", 0.5))
    k = min(int(round(settle_s / float(sim.dt))), nn_ - 1)
    yl = Y[:n, k:nn_, 0].cpu().numpy()
    mse_settled = float((err[:, k:] ** 2).mean())
    var_settled = float((yl ** 2).mean())
    cells = sorted(set(cond[:n].tolist()))
    teacher = torch.load(os.path.join(task_dir(corpus), "teacher.pt"), weights_only=False)
    res = {"name": spec["name"], "model": spec["model"], "corpus": corpus, "split": split,
           "n_trials": n, "mse": float(np.mean(per)), "mse_per_trial": per,
           # PER CONDITION CELL, EACH AGAINST ITS OWN TARGET VARIANCE: a pooled normalised number
           # moves the wrong way when a louder law joins, so it cannot say whether one circuit
           # holds several laws or drops some.
           "mse_per_cell": {str(c): float(np.mean([per[i] for i in range(n) if cond[i] == c]))
                            for c in cells},
           "normalised_per_cell": {
               str(c): float(np.mean([per[i] for i in range(n) if cond[i] == c])
                             / max(float((Y[:n][cond[:n] == c, :, 0] ** 2).mean()), 1e-30))
               for c in cells},
           "cell_names": list(teacher.get("names") or ["all"]),
           "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt(np.mean(per))),
           "unit": task["observe"].get("unit"),
           "target_variance": var, "normalised_mse": float(np.mean(per)) / var,
           "settle_s": settle_s, "mse_settled": mse_settled,
           "normalised_mse_settled": mse_settled / var_settled if var_settled > 0 else None,
           "fitted": {k: {"n": int(v.numel()), "mean": float(v.mean()), "sd": float(v.std())}
                      for k, v in ck["fitted"].items()}}
    p = os.path.join(out, "results", f"{spec['name']}_{split}.json")
    json.dump(res, open(p, "w"), indent=2)
    np.save(os.path.join(out, "results", f"{spec['name']}_{split}_traces.npy"), np.array(traces))
    U1, U2 = _unit(spec)
    print(f"[test] {split}: mean |err| {res['mae']:.4f}{U1}   rmse {res['rmse']:.4f}{U1}")
    print(f"[test] {split}: {n} trials  mse {res['mse']:.5f}{U2}  "
          f"({res['normalised_mse']:.5f} of target variance)")
    if len(cells) > 1:
        print("[test] per cell, of its own variance: " + "  ".join(
            f"{res['cell_names'][int(c)] if int(c) < len(res['cell_names']) else c} "
            f"{res['normalised_per_cell'][str(c)]:.5f}" for c in cells))
    print(f"[test] wrote {p}")
    return res


# ============================================================================== analyse
def circuit_poles(H, fitted, slope=None):
    """The trained circuit's poles, linearised where it operates: A = -diag(a) + diag(g) W diag(rho'(v)).

    a is the leak (1/tau, per neuron when a `tau` block exists), g the coupling gain, W the
    recurrent matrix and rho'(v) = 1 - tanh(v)^2 averaged over a rollout. EVERYTHING IS READ FROM
    `H`, WHICH MUST BE A TRAINED ROLLOUT. The analyser this replaces read W from the checkpoint but
    tau from a fresh, untrained run of the model, so whenever tau was learnable its poles mixed a
    trained connectome with the starting time constants.

    The recurrent edge set is found structurally -- the fitted edge set whose two endpoints are
    the same set -- not by its name. Under sign-lock the effective weight is |w| times the
    sender's sign. Returns (poles_at_operating_point, poles_at_origin), or None when no recurrent
    block was fitted.
    """
    rec = next((k for k in fitted if k.endswith(".w")
                and (lambda e: e.is_edge_set and e.pre_name == e.post_name)(H.level(k.split(".")[0]))),
               None)
    if rec is None:
        return None
    es = H.level(rec.split(".")[0])
    W = es.get("w").detach().cpu().numpy().ravel()
    lvl = H.level(es.post_name)
    n = lvl.n
    sig = dict(zip(H.operator_names, H.operators)).get("neuron_signal")
    if sig is not None and getattr(sig, "dale", False):
        W = np.abs(W) * sig._dale_sign(lvl, es).squeeze(-1).detach().cpu().numpy()
    M = np.zeros((n, n))
    M[es.post.cpu().numpy(), es.pre.cpu().numpy()] = W
    P = (lvl.type_params[lvl.node_type] if getattr(lvl, "type_params", None) is not None
         else torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]).expand(n, 6))
    a = P[:, 0].detach().cpu().numpy()                  # leak, 1/s
    g = P[:, 2].detach().cpu().numpy()                  # coupling gain, dimensionless
    upd = dict(zip(H.operator_names, H.operators)).get("neuron_update")
    tb = getattr(upd, "tau_block", None)
    if tb is not None and tb in lvl.state_schema:
        a = 1.0 / lvl.get(tb)[:, 0].detach().cpu().numpy().clip(min=float(getattr(upd, "tau_min", 1e-4)))
    rho = np.ones(n) if slope is None else np.asarray(slope, float).reshape(n)
    return (np.linalg.eigvals(-np.diag(a) + np.diag(g) @ M @ np.diag(rho)),
            np.linalg.eigvals(-np.diag(a) + np.diag(g) @ M))


def analyse(spec, device="cpu", root=None):
    """Did it recover the law, or only reduce the error? The figure, and the poles."""
    if spec.get("_kind") == "shape":
        return _analyse_shape(spec, device, root)
    if spec.get("_kind") == "recording":
        print("[analyse] recording tasks: the per-beat scores are the analysis (see test)")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks.render import BG, INK, MUTED, _ax

    engine.quiet(True)
    sim, learn, ck, out = _restore(spec, device, root)
    task, corpus = spec["task"], _corpus(spec)
    split = "test" if os.path.isdir(os.path.join(task_dir(corpus), "test")) else "train"
    U, Y, _c = _data(spec, split, int(ck.get("n_cond", 1)), device)
    rep = json.load(open(os.path.join(out, "results", "report.json")))
    res_p = os.path.join(out, "results", f"{spec['name']}_{split}.json")
    res = json.load(open(res_p)) if os.path.exists(res_p) else {}
    ch = int(task["observe"].get("channel", 0))
    with torch.no_grad():
        _, Yp = rollout(sim, learn, U[:4], task, device, grad=False)
        vs = []
        H1, _ = rollout(sim, learn, U[0], task, device, grad=False,
                        watch=lambda H: vs.append(H.level("neuron").get("voltage").squeeze(-1).clone())
                        if "neuron" in H.levels else None)
    preds = [Yp[i, :, ch].cpu().numpy() for i in range(4)]
    dt = float(sim.dt)
    t = np.arange(len(preds[0])) * dt
    U1, U2 = _unit(spec)
    poles, v_typ, slope = None, float("nan"), None
    if vs:
        V = torch.stack(vs)
        slope, v_typ = (1.0 - torch.tanh(V) ** 2).mean(0).cpu().numpy(), float(V.abs().mean())
        poles = circuit_poles(H1, ck["fitted"], slope=slope)

    fig = plt.figure(figsize=(14.5, 7.6), facecolor=BG)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)
    axa = _ax(fig.add_subplot(gs[0, 0]), ylabel="input", letter="a")
    for i in range(4):
        axa.plot(t[:U.shape[1]], U[i, :, 0].cpu(), color="#4a4a4a", lw=0.7, alpha=0.8)
    axb = _ax(fig.add_subplot(gs[0, 1]), ylabel=f"{task['observe']['set']}.{task['observe']['block']}{U1}",
              letter="b")
    for i, pr in enumerate(preds):
        n = min(len(pr), Y.shape[1])
        axb.plot(t[:n], Y[i, :n, 0].cpu(), color="#2e8b4f", lw=2.2, alpha=0.9)
        axb.plot(t[:n], pr[:n], color="#000000", lw=0.9)
    axb.text(0.985, 0.04, "green: ground truth   black: model", transform=axb.transAxes,
             ha="right", va="bottom", color=MUTED, fontsize=8)
    axd = _ax(fig.add_subplot(gs[1, 0]), xlabel="Re(lambda) (1/s)", ylabel="Im/2pi (Hz)", letter="d")
    truth = torch.load(os.path.join(task_dir(corpus), "teacher.pt"), weights_only=False)
    tp = truth["per_cell"][0]
    if poles is not None:
        at_op, at_origin = poles
        axd.scatter(at_origin.real, at_origin.imag / (2 * np.pi), s=10, color="0.72", alpha=0.7)
        axd.scatter(at_op.real, at_op.imag / (2 * np.pi), s=12, color="#000000", alpha=0.9)
    axd.scatter(tp["poles_real"], np.asarray(tp["poles_imag"]) / (2 * np.pi), s=80, marker="x",
                color="#2e8b4f", lw=2.0, zorder=5)
    axd.axvline(0, color="#c0272a", lw=0.8, ls="--")
    axd.text(0.985, 0.04, "green: ground truth   black: at the operating point   grey: at v=0",
             transform=axd.transAxes, ha="right", va="bottom", color=MUTED, fontsize=7.5)
    axe = _ax(fig.add_subplot(gs[1, 1]), xlabel="time (s)", ylabel=f"residual{U1}", letter="e")
    for i, pr in enumerate(preds):
        n = min(len(pr), Y.shape[1])
        axe.plot(t[:n], pr[:n] - Y[i, :n, 0].cpu().numpy(), color="#c0522a", lw=0.8)
    axe.set_ylim(axb.get_ylim())
    axf = _ax(fig.add_subplot(gs[1, 2]), xlabel="epoch", ylabel="mse", letter="f")
    ep = [h["epoch"] for h in rep["history"]]
    axf.plot(ep, [h["train_mse"] for h in rep["history"]], color="#1f6fb8", lw=1.1, label="train")
    axf.plot(ep, [h["val_mse"] for h in rep["history"]], color="#9a7d1a", lw=1.1, label="val")
    axf.set_yscale("log")
    for x in axf.legend(frameon=False, fontsize=8, loc="upper right").get_texts():
        x.set_color(INK)
    axc = _ax(fig.add_subplot(gs[0, 2]), letter="c")
    axc.axis("off")
    blocks = [f"{k}({v.numel()})" for k, v in ck["fitted"].items()]
    pr_max = list(tp.get("poles_real") or [])
    lines = ["", spec["name"], f"model  {os.path.basename(spec['model'])}", f"corpus {corpus}",
             f"split  {split}", "", f"fitted {rep['n_params']} values over {len(blocks)} block(s):"]
    lines += ["   " + "  ".join(blocks[i:i + 2]) for i in range(0, len(blocks), 2)]
    lines += ["", f"mean |err|  {res.get('mae', float('nan')):9.4f}{U1}",
              f"best val    {rep['best_val_mse']:9.5f}{U2}",
              f"{split} mse    {res.get('mse', float('nan')):9.5f}{U2}",
              f"normalised  {res.get('normalised_mse', float('nan')):9.5f} of target variance",
              f"trained on  {rep['n_trials']} trials, {rep['epochs']} epochs, {rep['seconds']:.0f} s"]
    if poles is not None:
        at_op, at_origin = poles
        lines += ["", f"|v| at operation {v_typ:.3f}  (rho' = {float(np.mean(slope)):.3f})",
                  f"max Re(lam)  v=0 {float(max(p.real for p in at_origin)):+.4f}   "
                  f"op {float(max(p.real for p in at_op)):+.4f}   truth "
                  + (f"{max(pr_max):+.4f}" if pr_max else "n/a") + "  1/s"]
    axc.text(0.02, 0.95, "\n".join(lines), transform=axc.transAxes, va="top", ha="left",
             color=INK, fontsize=8, family="monospace")
    p = os.path.join(out, "results", f"{spec['name']}_{split}.png")
    fig.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    if poles is not None:
        res["poles"] = {"max_re_at_op": float(max(p.real for p in poles[0])),
                        "max_re_at_origin": float(max(p.real for p in poles[1])),
                        "truth_max_re": (max(pr_max) if pr_max else None),
                        "v_mean_abs": v_typ}
        json.dump(res, open(res_p, "w"), indent=2)
        print(f"[analyse] max Re(lambda) at the operating point {res['poles']['max_re_at_op']:+.4f} "
              f"1/s, at v=0 {res['poles']['max_re_at_origin']:+.4f}, truth "
              + (f"{max(pr_max):+.4f}" if pr_max else "n/a"))
    print(f"[analyse] wrote {p}")
    return p



# ============================================================================== shape tasks
# ONE WORLD, NO TRIALS: the model starts from its set-up, grows for `n_frames`, and its final
# state is scored against a shape. So there is no batch, no validation split and no horizon --
# the scheme is a list of resolution STAGES, coarse to fine, each with its own Adam and its own
# cosine from `lr` to `lr_min`. It works only because a lattice over MATERIAL coordinates means
# the same thing at 5,000 points as at 50,000: a cheap stage buys the parameters a dear one needs.
def _shape_target(spec, n, rng):
    """`n` points uniform inside the reference mesh, centred and scaled as the task says."""
    from plexus import shapes
    from plexus.morph import sample_inside
    ref = spec["task"]["reference"]
    path, _part = shapes.resolve(ref["shape"])
    if os.path.isdir(path):
        objs = sorted(glob.glob(os.path.join(path, "*.obj")))
        if not objs:
            raise FileNotFoundError(f"shape {ref['shape']!r}: {path} holds no .obj")
        path = objs[0]
    pts = sample_inside(path, n, np.asarray(ref["centre"], float), float(ref["size"]), rng)
    return pts, path


def _point_targets(spec, set_name, n, device):
    """Each particle's own target, for `point_mse`: the part `reference.points` names, put through
    EXACTLY the placement the model's `cloud_seed` gives the seeded part -- x = origin + scale q,
    every `every`-th point -- so particle i and target i are the same atom in the same frame.
    A contained (one copy per parent) or rotated cloud is refused: its placement is not built here."""
    from plexus import shapes
    raw = yaml.safe_load(open(spec["model"]))
    seeds = [o for o in (raw.get("seed") or []) + (raw.get("operators") or [])
             if o.get("op") == "cloud_seed" and o.get("at") == set_name]
    if len(seeds) != 1:
        raise ValueError(f"point_mse: {set_name!r} is not placed by exactly one `cloud_seed` "
                         f"({len(seeds)}), so its particles have no order to match targets against")
    cs = seeds[0]
    if (raw.get("sets", {}).get(set_name) or {}).get("parent") or cs.get("rotate"):
        raise ValueError("point_mse: a contained or rotated cloud is not supported yet")
    shape, part = str(spec["task"]["reference"]["points"]).split("/", 1)
    path = next((os.path.join(r, shape, "points.npz") for r in shapes.roots()
                 if os.path.exists(os.path.join(r, shape, "points.npz"))), None)
    if path is None:
        raise FileNotFoundError(f"point_mse: no shapes/{shape}/points.npz")
    with np.load(path) as z:
        q = np.asarray(z[part], np.float64)[:: max(1, int(cs.get("every", 1)))]
    if len(q) != n:
        raise ValueError(f"point_mse: {shape}/{part} gives {len(q)} targets for {n} particles")
    x = np.asarray(cs.get("origin") or [0.5, 0.5, 0.5], float) + float(cs.get("scale", 1.0)) * q
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _grid_mass(sim, X, grid):
    """Unit-weight nodal counts on the model's own `grid` field, at its current resolution."""
    from plexus.morph import mass_grid
    return mass_grid(X, torch.ones(X.shape[0], device=X.device, dtype=X.dtype),
                     float(sim.world_size[0]), int(sim.fields[grid]["n_grid"]), X.device, torch)


def _shape_rollout(sim, learn, set_name, device, grad, keep=None):
    """Run the model once; return (H, final positions, realised volume sum(p_vol det F))."""
    hook = None
    if keep is not None:
        def hook(H, frame):
            keep.append(H.level(set_name).get("pos").detach().cpu().numpy().copy())
    H, _ = engine.run(sim, device=device, progress=False, grad=grad, on_seeded=learn.inject,
                      on_frame=hook)
    q = H.level(set_name)
    vol = (q.p_vol * torch.linalg.det(q.F)).sum() if hasattr(q, "F") else torch.tensor(float("nan"))
    return H, q.get("pos"), vol


def _extent(X, um):
    e = (X.max(0).values - X.min(0).values).detach().cpu().numpy() * (um or 1.0)
    return [round(float(v), 2) for v in e]


def _train_shape(spec, device="cpu", root=None):
    tr, task = spec["training"], spec["task"]
    obs = task["observe"]
    seed = int(tr.get("seed", 0))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.backends.cuda.matmul.allow_tf32 = True          # measured free on morph: 0.015% of the loss
    torch.backends.cudnn.allow_tf32 = True
    out = out_dir(spec, root)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    shutil.copyfile(spec["_path"], os.path.join(out, "config.yaml"))
    shutil.copyfile(spec["model"], os.path.join(out, "model.yaml"))
    print(f"[run] {spec['name']} -> {out}")
    engine.quiet(True)
    learn = Learnables(spec["learnable"], device)
    lr = float(tr.get("lr", 1.0))
    lr_min = float(tr.get("lr_min", lr / 20.0))
    clip = tr.get("clip")
    guard = tr.get("guard", "restore_and_halve")
    hist, stages, t_all = [], [], time.time()
    for si, st in enumerate(tr["stages"]):
        sim = _model(spec, train=True, resolution=st.get("resolution"))
        um = getattr(sim.units, "length_um", None) if getattr(sim.units, "declared", False) else None
        n, iters = int(sim.sets[obs["set"]]["n"]), int(st["iters"])
        tgt_np, mesh = _shape_target(spec, n, rng)
        tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=device)
        T_pts = (_point_targets(spec, obs["set"], n, device)
                 if spec["task"]["reference"].get("points") else None)
        with torch.no_grad():
            m_ref = _grid_mass(sim, tgt, obs["grid"])
            if not learn.p:                            # the first rollout makes the parameters
                _shape_rollout(sim, learn, obs["set"], device, grad=False)
        opt = torch.optim.Adam(learn.groups(lr), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1), eta_min=lr_min)
        last_ok, vol0, t_st = learn.snapshot(), None, time.time()
        print(f"  stage {si}: {n:,} points, grid {sim.fields[obs['grid']]['n_grid']}, {iters} iterations, "
              f"target {os.path.basename(mesh)}")
        for it in range(iters):
            opt.zero_grad()
            _, X, vol = _shape_rollout(sim, learn, obs["set"], device, grad=True)
            vol0 = vol0 if vol0 is not None else float(vol.detach())
            parts = {}
            loss = _objective(spec, learn, {
                "log_mse": lambda red: _log_mse(_grid_mass(sim, X, obs["grid"]), m_ref),
                # THE REALISED VOLUME, sum(p_vol det F), against the stage's first rollout: the mass
                # loss alone is gameable by imploding onto a dense target (morph.py, --vol-weight).
                "volume": lambda red: (vol / vol0 - 1.0) ** 2,
                # EACH PARTICLE AGAINST ITS OWN TARGET, when the targets are known (exp04: the atoms
                # shared by two deposited structures). An outline loss cannot tell which atom went where.
                "point_mse": lambda red: ((X - T_pts) ** 2).sum(-1).mean()}, len(hist), parts)
            if not torch.isfinite(loss) and guard == "restore_and_halve":
                learn.restore(last_ok)
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                print(f"    iter {it:3d}  loss not finite -- restored, step sizes halved", flush=True)
                continue
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(learn.parameters(), float(clip))
            opt.step()
            sch.step()
            learn.clamp_()
            last_ok = learn.snapshot()
            hist.append({"stage": si, "iter": it, "loss": float(loss.detach()),
                         "volume_ratio": float(vol.detach()) / vol0, "terms": parts})
            if it % 20 == 0 or it == iters - 1:
                # VOLUME BESIDE THE EXTENT: a body can approach the target's box while the material
                # it is made of quietly disappears, and the extent alone would not say so.
                print(f"    iter {it:3d}  loss {float(loss):.6f}  vol {hist[-1]['volume_ratio']:.3f}x  "
                      f"shape {_extent(X, um)}  target {_extent(tgt, um)}"
                      + (" um" if um else ""), flush=True)
        stages.append({"stage": si, "n": n, "iters": iters, "final_loss": hist[-1]["loss"],
                       "seconds": round(time.time() - t_st, 1)})
        print(f"  stage {si}: {iters} iterations in {(time.time() - t_st) / 60:.1f} min", flush=True)
    torch.save({"fitted": learn.snapshot(), "model": spec["model"], "task": task,
                "learnable": spec["learnable"]}, os.path.join(out, "models", "best.pt"))
    rep = {"name": spec["name"], "model": spec["model"], "shape": task["reference"]["shape"],
           "final_loss": hist[-1]["loss"], "stages": stages, "history": hist,
           "n_params": sum(p.numel() for p in learn.parameters()),
           "seconds": round(time.time() - t_all, 1)}
    json.dump(rep, open(os.path.join(out, "results", "report.json"), "w"), indent=2)
    print(f"[done] final loss {rep['final_loss']:.6f} in {rep['seconds'] / 60:.1f} min")
    return out


def _last_resolution(spec):
    return spec["training"]["stages"][-1].get("resolution")


def _test_shape(spec, device="cpu", root=None):
    """The trained learnables at the last stage's resolution, scored once more without a tape."""
    engine.quiet(True)
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    obs = spec["task"]["observe"]
    sim = _model(spec, train=True, resolution=_last_resolution(spec))
    um = getattr(sim.units, "length_um", None) if getattr(sim.units, "declared", False) else None
    rng = np.random.default_rng(int(spec["training"].get("seed", 0)) + 1)   # a FRESH target draw
    tgt_np, mesh = _shape_target(spec, int(sim.sets[obs["set"]]["n"]), rng)
    tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, X0, vol0 = _shape_rollout(_model(spec, train=True, resolution=_last_resolution(spec), n_frames=0),
                                     learn, obs["set"], device, grad=False)
        _, X, vol = _shape_rollout(sim, learn, obs["set"], device, grad=False)
        loss = float(_log_mse(_grid_mass(sim, X, obs["grid"]), _grid_mass(sim, tgt, obs["grid"])))
    if spec["task"]["reference"].get("points"):
        Tp = _point_targets(spec, obs["set"], int(sim.sets[obs["set"]]["n"]), device)
        rms = float(((X - Tp) ** 2).sum(-1).mean().sqrt())
        rms0 = float(((X0 - Tp) ** 2).sum(-1).mean().sqrt())
        print(f"[test] point RMS to own target: start {rms0 * (um or 1):.4f} -> end {rms * (um or 1):.4f}"
              + (" um" if um else " (world)"))
    res = {"name": spec["name"], "shape": spec["task"]["reference"]["shape"], "mesh": mesh,
           "loss": loss, "shape_extent": _extent(X, um), "target_extent": _extent(tgt, um),
           "unit": "um" if um else None, "volume_ratio": float(vol) / float(vol0),
           **({"point_rms_start": rms0, "point_rms_end": rms} if spec["task"]["reference"].get("points") else {})}
    p = os.path.join(out, "results", f"{spec['name']}_test.json")
    json.dump(res, open(p, "w"), indent=2)
    print(f"[test] loss {loss:.6f} against a fresh draw of the target; shape {res['shape_extent']} vs "
          f"target {res['target_extent']} {res['unit'] or ''}; volume {res['volume_ratio']:.3f}x of the start")
    return res


def _analyse_shape(spec, device="cpu", root=None):
    """The trained growth, rendered as glass by `plexus.morph`'s own renderer.

    A DISPLAY ROLLOUT MAY RUN LONGER THAN THE TRAINED ONE: `render.frames` spreads the SAME total
    rest-shape change over more frames (every rate multiplied by trained_frames / render_frames),
    so the material grows more slowly for longer and the movie is smoother. It is not the trained
    trajectory sampled finer -- the mechanics get more time -- which is why it is said here.
    """
    from plexus.morph import render_movie
    engine.quiet(True)
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    obs = spec["task"]["observe"]
    rd = spec["training"].get("render") or {}
    base = _model(spec, train=False, resolution=_last_resolution(spec))
    rf = int(rd.get("frames", base.n_frames))
    sim = _model(spec, train=False, resolution=_last_resolution(spec), n_frames=rf)
    learn.scale = float(base.n_frames) / float(rf)
    frames = []
    with torch.no_grad():
        _shape_rollout(sim, learn, obs["set"], device, grad=False, keep=frames)
    rng = np.random.default_rng(int(spec["training"].get("seed", 0)) + 2)
    tgt_np, _mesh = _shape_target(spec, int(sim.sets[obs["set"]]["n"]), rng)
    d = os.path.join(out, "results")
    stats = render_movie(d, np.stack(frames), tgt_np, float(sim.world_size[0]), spec["name"],
                         px=int(rd.get("px", 1100)))
    np.savez_compressed(os.path.join(d, f"{spec['name']}_frames.npz"), frames=np.stack(frames),
                        target=tgt_np, scale=learn.scale)
    print(f"[analyse] {len(frames)} frames rendered into {d} ({stats.get('s_per_frame_total', float('nan')):.2f} s a frame)")
    return d


# ============================================================================== recording tasks
# A TISSUE AGAINST ITS OWN RECORDING, beat by beat. Each beat is its own rollout from its own rest,
# its own length, with its own edge band prescribed from what that beat did -- the tissue beyond
# the field of view enters there -- and the loss is the per-cell affine mismatch of POSITIONS
# (plexus.tasks.recording.cell_affine: A_j the cell's mean deformation, u_j its centroid
# displacement), each channel scaled by the recording's own spread. Gradients of all beats are
# summed before one step and the priors are added once, so memory is one beat's and the cost is
# one rollout per beat. This is prototype/cardio_mpm/strain/fit.py's scheme, run by the trainer.
def _affine_mse(A, u, A_ref, u_ref, w_A, w_u, cells=None):
    if cells is not None:
        A, u, A_ref, u_ref = A[:, cells], u[:, cells], A_ref[:, cells], u_ref[:, cells]
    return w_A * ((A - A_ref) ** 2).mean() + w_u * ((u - u_ref) ** 2).mean()


LOSSES["affine_mse"] = _affine_mse


def _r2(pred, ref):
    return float(1.0 - ((pred - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum())


def _recording_setup(spec, device):
    """The recording, its beat windows with their affine maps, and the model's rest layout: the
    material points' positions X0 after seeding, their cell index, the edge band and the interior
    cells (none of whose points lie in the band) -- which the loss is restricted to, because a
    cell with points in the band has part of its motion dictated rather than modelled."""
    from plexus.tasks import recording as R
    task = spec["task"]
    ref, obs, drv = task["reference"], task["observe"], task["drive"]
    rec = R.load(ref["recording"], device=device)
    wins = [R.beat_window(rec, int(b)) for b in ref["beats"]]
    for w in wins:
        if (w["onset"] - w["span"][0]) - R.PRE != 0:
            raise ValueError(f"beat {w['k']} opens {w['onset'] - w['span'][0]} frames before its "
                             f"onset, not {R.PRE}: a truncated window shifts the clock, which this "
                             f"trainer does not re-align")
    refs = [R.window_affine(rec, w) for w in wins]
    box = {}
    rest = _model(spec, train=False, n_frames=0)
    with torch.no_grad():
        H, _ = engine.run(rest, device=device, progress=False)
    q = H.level(drv["set"])
    X0 = q.get("pos").detach().clone()
    cid = H.lift_index(drv["set"], obs["of"]).long() + 1
    C = rec["n_cells"]
    if int(cid.max()) > C:
        raise ValueError(f"the model has {int(cid.max())} cells, the recording {C}")
    wd = float(drv["width"])
    band = ((X0[:, 0] < R.DOM_LO + wd) | (X0[:, 0] > R.DOM_HI - wd)
            | (X0[:, 1] < R.DOM_LO + wd) | (X0[:, 1] > R.DOM_HI - wd))
    interior = R.interior_cells(cid, band, C) if task.get("mask", "interior") == "interior" else None
    pres = [R.band_prescription(A, u, X0, cid, band) for A, u in refs]
    box.update(rec=rec, wins=wins, refs=refs, X0=X0, cid=cid, band=band, interior=interior,
               pres=pres, C=C)
    return box


def _recording_rollout(sim, learn, spec, box, bi, device, grad, watch=None):
    """One beat: prescribe the band from the recording each frame, record every cell's affine map.
    `watch(H, tick)` is called at the same moment, for a caller that needs more of the state."""
    from plexus.tasks import recording as R
    set_name = spec["task"]["drive"]["set"]
    ub, band, C = box["pres"][bi], box["band"], box["C"]
    As, us, st = [], [], {}

    def hook(H, tick):
        q = H.level(set_name)
        if tick == 0:
            st["X0"] = q.get("pos").detach().clone()
        p0, p1 = q.state_schema["pos"]
        v0, v1 = q.state_schema["vel"]
        S0 = q.state
        S = S0.clone()
        m = band[:, None]
        S[:, p0:p1] = torch.where(m, st["X0"] + ub[min(tick, ub.shape[0] - 1)], S0[:, p0:p1])
        S[:, v0:v1] = torch.where(m, torch.zeros_like(S0[:, v0:v1]), S0[:, v0:v1])
        q.state = S
        A, u = R.cell_affine(q.get("pos"), st["X0"], box["cid"], C)
        As.append(A)
        us.append(u)
        if watch is not None:
            watch(H, tick)

    engine.run(sim, device=device, progress=False, grad=grad, on_frame=hook,
               on_seeded=learn.inject, on_ready=learn.ready)
    return torch.stack(As), torch.stack(us)


def _train_recording(spec, device="cpu", root=None):
    tr, task = spec["training"], spec["task"]
    torch.manual_seed(int(tr.get("seed", 0)))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = out_dir(spec, root)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    shutil.copyfile(spec["_path"], os.path.join(out, "config.yaml"))
    shutil.copyfile(spec["model"], os.path.join(out, "model.yaml"))
    print(f"[run] {spec['name']} -> {out}")
    engine.quiet(True)
    box = _recording_setup(spec, device)
    eye = torch.eye(2, device=device)
    A0, u0 = box["refs"][0]
    w_A = 1.0 / ((A0 - eye) ** 2).mean()                      # the first beat's spread, as fit.py
    w_u = float(task.get("u_weight", 1.0)) / (u0 ** 2).mean()
    sims = [_model(spec, train=True, n_frames=len(w["frames"]) - 1) for w in box["wins"]]
    n_int = int(box["interior"].sum()) if box["interior"] is not None else box["C"]
    print(f"[data] {task['reference']['recording']}: beats {task['reference']['beats']} "
          f"({', '.join(str(len(w['frames'])) for w in box['wins'])} frames), {box['X0'].shape[0]:,} "
          f"points, {int(box['band'].sum()):,} in the band, {n_int} of {box['C']} cells in the loss")
    learn = Learnables(spec["learnable"], device)
    with torch.no_grad():                                     # the first rollout makes the parameters
        _recording_rollout(_model(spec, train=True, n_frames=1), learn, spec, box, 0, device, False)
    params = learn.parameters()
    print(f"[fit] {len(params)} tensor(s), {sum(p.numel() for p in params)} values: " + ", ".join(learn.p))
    iters = int(tr["iters"])
    lr = float(tr.get("lr", 1e-2))
    frac = float(tr.get("lr_min_frac", 0.05))
    opt = torch.optim.Adam(learn.groups(lr), lr=lr)
    # COSINE FROM 1x TO `lr_min_frac` OF EACH GROUP'S OWN STEP SIZE, so twelve families whose
    # natural steps differ fifty-fold all anneal in proportion.
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: frac + (1 - frac) * 0.5 * (1 + math.cos(math.pi * min(i, iters) / max(iters, 1))))
    guard = tr.get("guard", "restore_and_halve")
    save_every = int(tr.get("save_every", 25))
    last_ok, log, t_all = learn.snapshot(), [], time.time()
    for it in range(iters):
        t0 = time.time()
        opt.zero_grad()
        loss_v, finite, terms = 0.0, True, {}
        for bi, sim in enumerate(sims):
            A_b, u_b = box["refs"][bi]
            A, u = _recording_rollout(sim, learn, spec, box, bi, device, True)
            c, nb = box["interior"], len(sims)
            Ac, uc, Abc, ubc = ((A, u, A_b, u_b) if c is None else (A[:, c], u[:, c], A_b[:, c], u_b[:, c]))
            parts = {}
            # EVERY BEAT'S EVIDENCE, THE PRIORS ONCE, on the last beat's graph -- fit.py's order.
            loss_b = _objective(spec, learn, {
                "affine_mse": lambda red: _affine_mse(A, u, A_b, u_b, w_A, w_u, c) / nb,
                "affine_A": lambda red: w_A * ((Ac - Abc) ** 2).mean() / nb,
                "affine_u": lambda red: ((uc - ubc) ** 2).mean() / (u0 ** 2).mean() / nb},
                it, parts, prior=(bi == nb - 1))
            for kk, vv in parts.items():
                terms[kk] = terms.get(kk, 0.0) + vv
            if torch.isfinite(loss_b):
                loss_b.backward()
            else:
                finite = False
            loss_v += float(loss_b.detach())
        if not finite and guard == "restore_and_halve":
            learn.restore(last_ok)
            for g in opt.param_groups:
                g["lr"] *= 0.5
            sch.step()
            print(f"  it {it:4d} loss not finite -- restored the last finite values, step sizes halved", flush=True)
            continue
        opt.step()
        sch.step()
        learn.clamp_()
        last_ok = learn.snapshot()
        log.append({"it": it, "loss": loss_v, "seconds": time.time() - t0, "terms": terms})
        if it % 5 == 0 or it == iters - 1:
            print(f"  it {it:4d} loss {loss_v:.5f}  {log[-1]['seconds']:.1f} s", flush=True)
        if (it + 1) % save_every == 0 or it == iters - 1:
            torch.save({"fitted": learn.snapshot(), "model": spec["model"], "task": task,
                        "learnable": spec["learnable"]}, os.path.join(out, "models", "best.pt"))
    rep = {"name": spec["name"], "model": spec["model"], "recording": task["reference"]["recording"],
           "beats": task["reference"]["beats"], "final_loss": log[-1]["loss"], "history": log,
           "n_params": sum(p.numel() for p in params), "cells_in_loss": n_int,
           "seconds": round(time.time() - t_all, 1),
           "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 2 ** 30
                           if str(device).startswith("cuda") else None)}
    json.dump(rep, open(os.path.join(out, "results", "report.json"), "w"), indent=2)
    print(f"[done] final loss {rep['final_loss']:.5f} in {rep['seconds'] / 60:.1f} min")
    return out


def _test_recording(spec, device="cpu", root=None):
    """Every fitted beat, rolled out again without a tape and scored as the prototype scored it:
    R^2 of A - I over (frames, cells, 4 components) and of u over (frames, cells, 2), interior cells."""
    engine.quiet(True)
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = _recording_setup(spec, device)
    eye = torch.eye(2, device=device)
    cells = box["interior"] if box["interior"] is not None else slice(None)
    res = {"name": spec["name"], "beats": {}}
    for bi, w in enumerate(box["wins"]):
        sim = _model(spec, train=False, n_frames=len(w["frames"]) - 1)
        with torch.no_grad():
            A, u = _recording_rollout(sim, learn, spec, box, bi, device, False)
        A_ref, u_ref = box["refs"][bi]
        r = {"window": list(w["span"]), "r2_A": _r2(A[:, cells] - eye, A_ref[:, cells] - eye),
             "r2_u": _r2(u[:, cells], u_ref[:, cells])}
        res["beats"][str(w["k"])] = r
        print(f"[test] beat {w['k']} window {tuple(w['span'])}: R2(A) {r['r2_A']:.4f}  R2(u) {r['r2_u']:.4f}")
    json.dump(res, open(os.path.join(out, "results", f"{spec['name']}_test.json"), "w"), indent=2)
    return res

# ============================================================================== entry points
PHASES = {"train": train, "test": test, "analyse": analyse}


def run_phases(name_or_path, phases, device="cpu", root=None):
    """`Plexus_Main.py -o train_test_analyse <name>` lands here."""
    spec = load(name_or_path)
    for ph in phases:
        PHASES[ph](spec, device=device, root=root)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--option", nargs="+", required=True,
                    help="phases (train / test / analyse) and training spec names or paths, any order")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default=None, help="overrides the log/ root")
    a = ap.parse_args(argv)
    phases = [o for o in a.option if o in PHASES] or ["train"]
    names = [o for o in a.option if o not in PHASES]
    if not names:
        ap.error("no training spec given")
    for n in names:
        for p in (glob.glob(n) if any(c in n for c in "*?[") else [n]):
            run_phases(p, phases, device=a.device, root=a.root)


if __name__ == "__main__":
    main()
