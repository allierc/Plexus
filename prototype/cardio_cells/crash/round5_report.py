"""round5_report.py -- ROUND 5, stage 4.  Merge the shards, measure the floors, write the verdict.

Nothing is computed here that needs a GPU: the two shard JSONs already carry every rollout.  What
this adds is the bookkeeping the earlier rounds got wrong:

  * the per-cell SKILL denominator is the ZERO-INFORMATION BANK'S ARGMAX (round 3's rule), so a
    blind constant scores 0 and cannot be beaten by accident;
  * the FLOOR is the band [min, max] of the bank on both instruments, and every candidate is quoted
    as its distance above the TOP of that band;
  * the acceptance statistic is the HELD-OUT one-frame residual, never med|dE/E| (round 4's
    diagnosis: `null_med0_rand45` has med|dE/E| = 0.0000 and is 45 % random);
  * the GAUGE'S OWN UNCERTAINTY (the loopscore spread over grid cells that satisfy the gauge to
    10 %) is reported beside every gauged score.

usage: PYTHONPATH=/workspace/Plexus/src python round5_report.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="round5_score")
    ap.add_argument("--out", default="round5_report")
    a = ap.parse_args()

    import glob
    shards = []
    for p in sorted(glob.glob(os.path.join(HERE, f"{a.tag}_s*.json"))):
        shards.append(json.load(open(p)))
    for p in sorted(glob.glob(os.path.join(HERE, f"{a.tag}40k_*_s*.json"))
                    + glob.glob(os.path.join(HERE, f"{a.tag}bw_s*.json"))):
        shards.append(json.load(open(p)))
    if not shards:
        raise SystemExit("no shard json")
    cands = {}
    for sh in shards:
        for k, v in sh["candidates"].items():
            if k in cands and k != "theta_true":
                continue
            cands[k] = v
    R = {"n_shards": len(shards), "cite_status": shards[0]["cite_status"],
         "probes_in_anchor_band": shards[0]["probes_in_anchor_band"],
         "nulls": shards[0].get("nulls"), "campaign_nulls": shards[0].get("campaign_nulls"),
         "bank_names": shards[0]["bank_names"], "reference": shards[0]["reference"]}
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    # ---- cross-shard control: theta_true must be identical on the two GPUs --------------------
    if len(shards) == 2 and "theta_true" in shards[0]["candidates"] \
            and "theta_true" in shards[1]["candidates"]:
        t0 = shards[0]["candidates"]["theta_true"]
        t1 = shards[1]["candidates"]["theta_true"]
        d = {k: abs(t0["raw"][k] - t1["raw"][k]) for k in ("loop", "R2", "t1", "rms_dx_mean")}
        d["percell_r2"] = abs(t0["raw"]["percell"]["r2"] - t1["raw"]["percell"]["r2"])
        d["a_percell_max"] = float(np.nanmax(np.abs(
            np.array(t0["raw"]["a_percell"], float) - np.array(t1["raw"]["a_percell"], float))))
        R["control_cross_gpu_theta_true"] = d
        log("[control] theta_true, shard 0 vs shard 1 (different GPUs): "
            + "  ".join(f"{k} {v:.3e}" for k, v in d.items()))

    a_ref = np.array(shards[0]["a_ref_percell"], float)
    keep = np.array(shards[0]["keep_percell"], bool)
    ar = a_ref[keep] / a_ref[keep].mean()

    def sse_of(rec, which="gauged"):
        ah = np.array(rec[which]["a_percell"], float)[keep]
        if not np.isfinite(ah).all() or ah.mean() <= 0:
            return None
        return float((((ah / ah.mean()) - ar) ** 2).sum())

    # `null_med0_rand45` is theta_true with 45 of 100 cells replaced: it is a DECOY that defeats
    # med|dE/E|, not a zero-information vector, so it is reported separately and is NOT allowed to
    # set the floor or the skill denominator.
    bank = [n for n in R["bank_names"] if n in cands and n.startswith("bank_")]
    R["decoys"] = [n for n in R["bank_names"] if n in cands and not n.startswith("bank_")]
    sse = {n: sse_of(cands[n]) for n in cands}
    bank_ok = [n for n in bank if sse[n] is not None]
    denom_name = min(bank_ok, key=lambda n: sse[n])
    denom = sse[denom_name]
    floor_loop = max(cands[n]["gauged"]["loop"] for n in bank)
    floor_loop_name = max(bank, key=lambda n: cands[n]["gauged"]["loop"])
    floor_loop_min = min(cands[n]["gauged"]["loop"] for n in bank)
    R["floor"] = {"bank": bank, "loop_band": [floor_loop_min, floor_loop],
                  "loop_band_top_member": floor_loop_name,
                  "skill_denominator": denom_name, "skill_denominator_sse": denom,
                  "raw_loop_band": [min(cands[n]["raw"]["loop"] for n in bank),
                                    max(cands[n]["raw"]["loop"] for n in bank)]}
    log(f"[floor] zero-information bank ({len(bank)} members): gauged loopscore band "
        f"[{floor_loop_min:.4f}, {floor_loop:.4f}] (top = {floor_loop_name}); "
        f"per-cell skill denominator = {denom_name}")

    rows = {}
    for n, rec in cands.items():
        s = sse[n]
        rows[n] = {
            "med_E": rec["param"]["med_E"], "p90_E": rec["param"]["p90_E"],
            "max_E": rec["param"]["max_E"], "n_negE": rec["param"]["n_negE"],
            "n_gt5x": rec["param"]["n_cells_relE_gt5"],
            "rel_l2": rec["param"]["rel_l2"], "corr_E": rec["param"]["corr_E"],
            "rel_l2_gauge_opt": rec["param"]["rel_l2_gauge_opt"],
            "med_E_after_rescale": rec["param"]["med_E_after_rescale"],
            "med_E_after_gauge": rec["param_after_gauge"]["med_E"],
            "mean_ratio_E": rec["param"]["mean_ratio_E"],
            "holdout_cleanF": rec["holdout_1frame_cleanF"],
            "holdout_noisyF": rec["holdout_1frame_noisyF"],
            "raw_loop": rec["raw"]["loop"], "raw_t1": rec["raw"]["t1"],
            "k_E": rec["gauge"]["k_E"], "k_g": rec["gauge"]["k_g"],
            "gauge_converged": rec["gauge"]["converged"],
            "gauge_uncertainty": rec["gauge"]["gauge_uncertainty"],
            "gauged_loop": rec["gauged"]["loop"], "gauged_R2": rec["gauged"]["R2"],
            "raw_R2": rec["raw"]["R2"],
            "r2cell": rec["gauged"]["percell"]["r2"],
            "rms_dx_mean": rec["gauged"]["rms_dx_mean"],
            "rms_dx_final": rec["gauged"]["rms_dx_final"],
            "margin10_loop": rec["gauged"].get("margin10_loop"),
            "skill": (None if s is None or denom in (None, 0) else float(1.0 - s / denom)),
            "above_band_loop": rec["gauged"]["loop"] - floor_loop,
            "instruments_gauged": rec["gauged"]["margin20"] and
            {k: rec["gauged"]["margin20"][k] for k in
             ("coordination", "path_length", "peak_excursion", "orientation_error")}}
    R["rows"] = rows

    order = sorted(rows, key=lambda n: -(rows[n]["gauged_loop"] if isinstance(
        rows[n]["gauged_loop"], float) else -9))
    log(f"\n[table] {len(rows)} candidates, sorted by gauged loopscore. band top = "
        f"{floor_loop:.4f}; skill 0 = {denom_name}")
    log(f"    {'candidate':<24s} {'medE':>7s} {'neg':>4s} {'>5x':>4s} {'relL2':>7s} "
        f"{'corr':>6s} {'hold1f':>7s} {'holdN':>7s} {'raw':>8s} {'kE':>6s} {'kg':>6s} "
        f"{'GAUGED':>8s} {'+-':>5s} {'>band':>7s} {'R2':>8s} {'skill':>7s} {'rms/dx':>7s}")
    for n in order:
        r = rows[n]
        sk = r["skill"]
        log(f"    {n:<24s} {r['med_E']:>7.4f} {r['n_negE']:>4d} {r['n_gt5x']:>4d} "
            f"{r['rel_l2']:>7.3f} {r['corr_E']:>6.3f} {r['holdout_cleanF']:>7.4f} "
            f"{r['holdout_noisyF']:>7.4f} {r['raw_loop']:>8.4f} {r['k_E']:>6.3f} "
            f"{r['k_g']:>6.3f} {r['gauged_loop']:>8.4f} {r['gauge_uncertainty']:>5.3f} "
            f"{r['above_band_loop']:>+7.3f} {r['gauged_R2']:>8.4f} "
            f"{(f'{sk:.3f}' if sk is not None else 'n/a'):>7s} {r['rms_dx_mean']:>7.4f}")

    # ---- the 2x2 design, averaged over the three measurement draws ---------------------------
    R["design_2x2"] = {}
    log("\n[2x2] constraint (box) x correction (EIV), mean +- range over the 3 measurement seeds")
    log(f"    {'cell':<12s} {'medE':>16s} {'negE':>10s} {'holdout':>16s} {'gauged loop':>18s} "
        f"{'skill':>16s}")
    for k in ("naive", "eiv_snr0", "naive_box", "eiv_box"):
        got = [rows[f"s{sd}/T8/{k}"] for sd in (90210, 555, 777) if f"s{sd}/T8/{k}" in rows]
        if not got:
            continue

        def mm(key):
            v = [g[key] for g in got if g[key] is not None]
            return (float(np.mean(v)), float(np.min(v)), float(np.max(v))) if v else (None,) * 3
        cell = {q: mm(q) for q in ("med_E", "n_negE", "holdout_cleanF", "gauged_loop", "skill",
                                   "raw_loop", "gauged_R2", "med_E_after_gauge", "corr_E",
                                   "n_gt5x", "rel_l2")}
        R["design_2x2"][k] = cell
        log(f"    {k:<12s} {cell['med_E'][0]:>7.4f} [{cell['med_E'][1]:.3f},"
            f"{cell['med_E'][2]:.3f}] {cell['n_negE'][0]:>10.1f} "
            f"{cell['holdout_cleanF'][0]:>7.4f} [{cell['holdout_cleanF'][1]:.3f},"
            f"{cell['holdout_cleanF'][2]:.3f}] {cell['gauged_loop'][0]:>8.4f} "
            f"[{cell['gauged_loop'][1]:.3f},{cell['gauged_loop'][2]:.3f}] "
            f"{cell['skill'][0]:>7.3f} [{cell['skill'][1]:.2f},{cell['skill'][2]:.2f}]")

    # ---- which parameter statistic predicts the rollout? --------------------------------------
    keys = ["med_E", "p90_E", "max_E", "rel_l2", "corr_E", "rel_l2_gauge_opt",
            "med_E_after_rescale", "n_negE", "n_gt5x", "holdout_cleanF", "holdout_noisyF"]
    names = [n for n in rows if isinstance(rows[n]["gauged_loop"], float)]
    R["spearman"] = {}
    log(f"\n[predictors] Spearman over the {len(names)} scored candidates")
    log(f"    {'statistic':<22s} {'vs raw loop':>12s} {'vs gauged':>12s} {'vs R2':>12s} "
        f"{'vs skill':>12s}")
    gl = [rows[n]["gauged_loop"] for n in names]
    rl = [rows[n]["raw_loop"] for n in names]
    r2 = [rows[n]["gauged_R2"] for n in names]
    sk = [rows[n]["skill"] if rows[n]["skill"] is not None else np.nan for n in names]
    for k in keys:
        v = np.array([rows[n][k] for n in names], float)
        e = {}
        for lab, y in (("raw", rl), ("gauged", gl), ("R2", r2), ("skill", sk)):
            y = np.asarray(y, float)
            m = np.isfinite(v) & np.isfinite(y)
            e[lab] = float(spearmanr(v[m], y[m]).statistic) if m.sum() > 3 else None
        R["spearman"][k] = e
        log(f"    {k:<22s} {e['raw']:>+12.3f} {e['gauged']:>+12.3f} {e['R2']:>+12.3f} "
            f"{e['skill']:>+12.3f}")

    # ---- round 4's acceptance criteria, checked verbatim --------------------------------------
    crit = {}
    for k in ("naive_box", "eiv_box"):
        per = {}
        for sd in (90210, 555, 777):
            n = f"s{sd}/T8/{k}"
            if n not in rows:
                continue
            r = rows[n]
            per[n] = {"negE_0": r["n_negE"] == 0, "gt5x_0": r["n_gt5x"] == 0,
                      "gauged_loop_ge_0.93": r["gauged_loop"] >= 0.93,
                      "med_after_gauge_le_0.25": r["med_E_after_gauge"] <= 0.25,
                      "holdout_le_0.020": r["holdout_cleanF"] <= 0.020,
                      "values": {"n_negE": r["n_negE"], "n_gt5x": r["n_gt5x"],
                                 "gauged_loop": r["gauged_loop"],
                                 "med_E_after_gauge": r["med_E_after_gauge"],
                                 "holdout_cleanF": r["holdout_cleanF"]}}
        crit[k] = per
    R["round4_criteria"] = crit
    log("\n[criteria] round 4's prescription, checked verbatim on the three seeds")
    for k, per in crit.items():
        for n, c in per.items():
            met = sum(1 for q, v in c.items() if q != "values" and v)
            log(f"    {n:<24s} {met}/5 met  " + "  ".join(
                f"{q.split('_le')[0].split('_ge')[0]}={'PASS' if v else 'FAIL'}"
                for q, v in c.items() if q != "values"))

    json.dump(R, open(os.path.join(HERE, f"{a.out}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.out}.txt"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.out}.json / {a.out}.txt")


if __name__ == "__main__":
    main()
