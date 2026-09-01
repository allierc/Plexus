"""EVERY NUMBER THIS PROTOTYPE REPORTS, computed in one place.

THE RULE IS COPIED FROM `connectome_gnn/metrics.py`, and it is a rule about agreement rather than
about arithmetic: `recovery_param_metrics` is documented there as "the single public entry point
for R^2 anywhere in this codebase, computed ONCE so the scatter, console line and metrics.txt can't
disagree." That is the property worth having. A figure whose caption says 0.91 and a table that
says 0.87 have not disagreed about the world; they have disagreed about which R^2 was meant, and no
reader can tell which.

So: two public entry points, and nothing outside this module computes an R^2 or a correlation.

    recovery(gt, learned)          a PARAMETER against its known value      -> G9, G10, G13, G28-G30
    rollout(true_traj, pred_traj)  a TRAJECTORY against what was recorded   -> the tester

R^2 IS THE IDENTITY LINE, NOT A FITTED LINE, matching `_r2_slope_identity` there:

    R^2 = 1 - mean((true - pred)^2) / var(true)

which is the Nash-Sutcliffe efficiency. A fitted-line R^2 asks "is the prediction a linear function
of the truth", and answers yes to a prediction that is twice the truth everywhere. The identity-line
version asks "is it the truth", which is the question. It ranges to -inf, and a negative value is
informative: the prediction is worse than the mean of the observations. The calibration SLOPE is
returned beside it, because when R^2 is low the slope says whether the failure is scale or shape.

PEARSON IS REPORTED BESIDE R^2 AND NEVER INSTEAD OF IT. Correlation is invariant to affine rescaling,
so a rollout that drifts to twice the amplitude still scores r = 1.0. That invariance is exactly
what makes it useful for a long rollout -- phase agreement is meaningful once amplitude has drifted
-- and exactly what makes it useless alone. Reported as a pair, the two say different things:

    r high, R^2 low     the shape is right and the scale or offset is not
    r low,  R^2 low     the prediction has lost the trajectory
    r low,  R^2 high    essentially impossible; suspect a bug in the caller
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _flat(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a


def r2_identity(true, pred) -> tuple[float, float]:
    """Identity-line R^2 (Nash-Sutcliffe) and the calibration slope of pred ~ a*true + b.

    Private in spirit -- `recovery` and `rollout` are the entry points -- but exported because the
    gates measure single scalars against known values and should not have to build a dict to do it.
    """
    t, p = _flat(true), _flat(pred)
    n = min(t.size, p.size)
    t, p = t[:n], p[:n]
    vt = float(np.var(t))
    if not np.isfinite(vt) or vt <= _EPS:
        return float("nan"), float("nan")          # a constant truth has no variance to explain
    r2 = 1.0 - float(np.mean((t - p) ** 2)) / vt
    try:
        slope = float(np.polyfit(t, p, 1)[0])
    except Exception:
        slope = float("nan")
    return r2, slope


def pearson(true, pred) -> float:
    """Pearson r. NaN rather than 0.0 when either side is constant -- an undefined correlation is
    not a zero one, and reporting 0.0 there has been read as "no relationship" more than once."""
    t, p = _flat(true), _flat(pred)
    n = min(t.size, p.size)
    t, p = t[:n], p[:n]
    if t.std() <= _EPS or p.std() <= _EPS:
        return float("nan")
    return float(np.corrcoef(t, p)[0, 1])


def is_degenerate(gt, rel_eps: float = 1e-4) -> bool:
    """A ground truth that is (nearly) constant. R^2 against it is undefined, not zero."""
    g = _flat(gt)
    return bool(g.size == 0 or g.std() <= rel_eps * max(abs(float(g.mean())), _EPS))


def recovery(gt, learned) -> dict:
    """A PARAMETER against its known value. The single entry point for parameter recovery.

    `rel_err_median` is the one to quote for a scalar with a known value -- G28's threshold is
    "within 1% of the true speed" and that is a relative error, not an R^2. `r2`/`slope` are for a
    FIELD of parameters (omega_i, W_e, a_i) where the question is whether the pattern was found.
    """
    g, l = _flat(gt), _flat(learned)
    n = min(g.size, l.size)
    g, l = g[:n], l[:n]
    r2, slope = r2_identity(g, l)
    rel = np.abs(l - g) / np.maximum(np.abs(g), 1e-6)
    return dict(
        n=int(n), r2=r2, slope=slope, pearson=pearson(g, l),
        mae=float(np.mean(np.abs(l - g))) if n else float("nan"),
        rel_err_median=float(np.median(rel)) if n else float("nan"),
        rel_err_iqr=float(np.subtract(*np.percentile(rel, [75, 25]))) if n else float("nan"),
        degenerate=is_degenerate(g),
        gt_mean=float(g.mean()) if n else float("nan"),
        gt_std=float(g.std()) if n else float("nan"),
        learned_mean=float(l.mean()) if n else float("nan"),
        learned_std=float(l.std()) if n else float("nan"),
    )


def rollout(true_traj, pred_traj, mask=None) -> dict:
    """A TRAJECTORY against what was recorded. The single entry point for rollout scoring.

    `true_traj`, `pred_traj`: [K, ...] -- K rollout steps, each of any shape, aligned so that index
    k is the state k steps after the common start. `mask`: an optional boolean over the trailing
    (spatial) axes; cells outside it are excluded from every statistic, which matters when a rule
    acts on 15% of the domain and the other 85% is identically zero in both arrays and would
    otherwise inflate every correlation towards 1.

    PER STEP AND OVERALL, because the shape of the decay is the result. A single number for a
    rollout hides whether the model is accurate for two steps and then diverges, or uniformly
    mediocre; those want different fixes. `horizon_at` reports the first step at which R^2 falls
    below a threshold -- the usable horizon, in the unit of the phenomenon (recorded frames).
    """
    t = np.asarray(true_traj, dtype=np.float64)
    p = np.asarray(pred_traj, dtype=np.float64)
    if t.shape != p.shape:
        raise ValueError(f"rollout shapes differ: true {t.shape} vs pred {p.shape}")
    K = t.shape[0]
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        t = t.reshape(K, -1)[:, m.ravel()]
        p = p.reshape(K, -1)[:, m.ravel()]
    else:
        t, p = t.reshape(K, -1), p.reshape(K, -1)

    per_r2 = [r2_identity(t[k], p[k])[0] for k in range(K)]
    per_r = [pearson(t[k], p[k]) for k in range(K)]
    per_rmse = [float(np.sqrt(np.mean((t[k] - p[k]) ** 2))) for k in range(K)]
    return dict(
        horizon=K,
        r2=per_r2, pearson=per_r, rmse=per_rmse,
        r2_mean=float(np.nanmean(per_r2)), pearson_mean=float(np.nanmean(per_r)),
        r2_final=per_r2[-1], pearson_final=per_r[-1],
        r2_all=r2_identity(t, p)[0], pearson_all=pearson(t, p),
        n_cells=int(t.shape[1]),
    )


def horizon_at(res: dict, key: str = "r2", threshold: float = 0.5) -> int:
    """The last step whose `key` is still above `threshold`; 0 if the first step already fails.

    THE USABLE HORIZON, and it is quoted in RECORDED FRAMES because that is the unit the reader has
    -- "this model is good for 6 frames" is a statement about the phenomenon, where "R^2 0.62" is a
    statement about an array.
    """
    vals = res[key]
    k = 0
    for v in vals:
        if not np.isfinite(v) or v < threshold:
            break
        k += 1
    return k


def format_rollout(res: dict, name: str = "") -> str:
    """One block, so the console and the metrics file cannot render it differently."""
    lines = [f"rollout{' ' + name if name else ''}: {res['horizon']} steps, "
             f"{res['n_cells']} cells scored",
             "  step      R^2    pearson       rmse"]
    for k, (a, b, c) in enumerate(zip(res["r2"], res["pearson"], res["rmse"]), 1):
        lines.append(f"  {k:4d}  {a:8.4f}  {b:8.4f}  {c:9.3e}")
    lines.append(f"  mean  {res['r2_mean']:8.4f}  {res['pearson_mean']:8.4f}")
    lines.append(f"  usable horizon (R^2 > 0.5): {horizon_at(res)} of {res['horizon']} steps")
    return "\n".join(lines)
