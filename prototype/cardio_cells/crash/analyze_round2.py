"""analyze_round2.py -- merge the two shards, check every control, and derive the two things the
raw JSON does not contain:

  1. percell_skill_vs_blind = 1 - SSE(cand)/SSE(blind), on the mean-normalised per-cell amplitude
     field, with `blind` = the gauge-fixed per-cell-BLIND constant. r2_percell as defined has a
     floor of ~0.96 because a cell's amplitude is mostly set by WHERE IT SITS, not by its own
     theta; this renormalisation puts the null back at 0 where a null belongs. No extra rollouts:
     both quantities come from the a_percell arrays already recorded.
  2. the regressions: is loopscore still a readout of the global amplitude, and does it now track
     the per-cell error?

usage: PYTHONPATH=/workspace/Plexus/src python analyze_round2.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    S = [json.load(open(os.path.join(HERE, f"crash_round2_s{i}.json"))) for i in (0, 1)]
    out = {"shard_files": [f"crash_round2_s{i}.json" for i in (0, 1)]}

    # ---- control: the two shards ran on different GPUs; theta_true must agree -----------------
    tA = S[0]["rollouts"]["theta_true"], S[1]["rollouts"]["theta_true"]
    out["cross_device_control"] = {
        "device_shard0": S[0]["config"]["device"], "device_shard1": S[1]["config"]["device"],
        "theta_true_loopscore": [tA[0]["raw"]["margin20"]["loopscore"],
                                 tA[1]["raw"]["margin20"]["loopscore"]],
        "theta_true_a_percell_max_abs_diff": float(np.nanmax(np.abs(
            np.array(tA[0]["raw"]["a_percell"], dtype=float)
            - np.array(tA[1]["raw"]["a_percell"], dtype=float)))),
        "a_ref_max_abs_diff": float(np.nanmax(np.abs(
            np.array(S[0]["a_ref_percell"], dtype=float)
            - np.array(S[1]["a_ref_percell"], dtype=float))))}

    ro = {}
    for s in S:
        for k, v in s["rollouts"].items():
            ro.setdefault(k, v)
    out["n_candidates"] = len(ro)

    a_ref = np.array(S[0]["a_ref_percell"], dtype=float)
    keep = np.array(S[0]["keep_percell"], dtype=bool)
    ar = a_ref[keep] / a_ref[keep].mean()

    def norm_field(rec):
        a = np.array(rec["a_percell"], dtype=float)[keep]
        return a / a.mean()

    # blind reference field for the skill score: the GAUGE-FIXED tuned blind constant
    blind = norm_field(ro["blind_E130_g0.95"]["gauged"])
    sse_blind = float(((blind - ar) ** 2).sum())
    out["percell_skill_denominator"] = {
        "reference_model": "blind_E130_g0.95 (gauge-fixed)", "sse_blind": sse_blind,
        "sst_reference": float(((ar - ar.mean()) ** 2).sum()),
        "r2_of_blind": 1.0 - sse_blind / float(((ar - ar.mean()) ** 2).sum()),
        "note": "a per-cell-BLIND constant already explains this much of the per-cell amplitude "
                "field, because that field is mostly geometry. Skill is measured against it."}

    rows = []
    for name, rec in ro.items():
        r = {"name": name, "med_E": rec["theta_error"]["med_E"],
             "med_gain": rec["theta_error"]["med_gain"],
             "rel_l2": rec["theta_error"]["rel_l2"],
             "k": rec["gauge"]["k"], "gauge_status": rec["gauge"]["status"],
             "gauge_extra_rollouts": rec["gauge"]["n_extra_rollouts"]}
        for tag in ("raw", "gauged"):
            d = rec[tag]
            f = norm_field(d)
            sse = float(((f - ar) ** 2).sum())
            r[f"{tag}_loopscore"] = d["margin20"]["loopscore"]
            r[f"{tag}_coordination"] = d["margin20"]["coordination"]
            r[f"{tag}_orientation_error"] = d["margin20"]["orientation_error"]
            r[f"{tag}_path_length"] = d["margin20"]["path_length"]
            r[f"{tag}_peak_excursion"] = d["margin20"]["peak_excursion"]
            r[f"{tag}_R2_disp"] = d["coarse"]["R2_displacement_interior"]
            r[f"{tag}_E_ratio"] = d["coarse"]["motion_energy_ratio_interior"]
            r[f"{tag}_rms_dx"] = d["coarse"]["rms_pos_err_dx_mean"]
            r[f"{tag}_r2_percell"] = d["percell"]["r2"]
            r[f"{tag}_percell_skill_vs_blind"] = 1.0 - sse / sse_blind
        r["d_loopscore_gauge"] = r["gauged_loopscore"] - r["raw_loopscore"]
        rows.append(r)
    rows.sort(key=lambda x: -x["gauged_loopscore"])
    out["rows"] = rows

    # ---- is the score still a readout of amplitude? -------------------------------------------
    def arr(k):
        return np.array([r[k] for r in rows], dtype=float)

    reg = {}
    for tag in ("raw", "gauged"):
        ls, er, me = arr(f"{tag}_loopscore"), arr(f"{tag}_E_ratio"), arr("med_E")
        ok = er > 0
        reg[tag] = {
            "corr_loopscore_vs_log_E_ratio": float(np.corrcoef(ls[ok], np.log(er[ok]))[0, 1]),
            "corr_loopscore_vs_med_E": float(np.corrcoef(ls, me)[0, 1]),
            "spearman_loopscore_vs_med_E": spearman(ls, me),
            "corr_r2percell_vs_med_E": float(np.corrcoef(arr(f"{tag}_r2_percell"), me)[0, 1]),
            "spearman_skill_vs_med_E": spearman(arr(f"{tag}_percell_skill_vs_blind"), me),
            "E_ratio_range": [float(er.min()), float(er.max())]}
    out["regressions"] = reg

    # ---- the controls the diagnosis named ------------------------------------------------------
    def g(n, tag, k):
        return ro[n][tag]["margin20"][k] if k in ro[n][tag]["margin20"] else None

    out["controls"] = {
        "gauge_fix(theta_true)_k": ro["theta_true"]["gauge"]["k"],
        "gauge_fix(theta_true)_loopscore_after": ro["theta_true"]["gauged"]["margin20"]["loopscore"],
        "gauge_fix(theta_hat_frame_vel_fd)_k": ro["theta_hat_frame_ridge0"]["gauge"]["k"],
        "gauge_fix(theta_hat_frame_vel_fd)_E_ratio":
            [ro["theta_hat_frame_ridge0"]["raw"]["coarse"]["motion_energy_ratio_interior"],
             ro["theta_hat_frame_ridge0"]["gauged"]["coarse"]["motion_energy_ratio_interior"]],
        "gauge_fix(theta_hat_frame_vel_fd)_loopscore":
            [ro["theta_hat_frame_ridge0"]["raw"]["margin20"]["loopscore"],
             ro["theta_hat_frame_ridge0"]["gauged"]["margin20"]["loopscore"]],
        "predicted_k_range": [0.55, 0.60], "predicted_loopscore_after": ">= +0.60"}

    triad = ["true_gain_x1.8", "blind_E40_g1", "theta_hat_frame_ridge0"]
    pair = ["blind_E130_g0.95", "frame_DISP_oracle_rescale"]
    out["discriminating_triad"] = {
        n: {"med_E": ro[n]["theta_error"]["med_E"],
            "raw_loopscore": ro[n]["raw"]["margin20"]["loopscore"],
            "gauged_loopscore": ro[n]["gauged"]["margin20"]["loopscore"],
            "gauged_r2_percell": ro[n]["gauged"]["percell"]["r2"],
            "gauged_skill_vs_blind": [r for r in rows if r["name"] == n][0]
                                     ["gauged_percell_skill_vs_blind"]} for n in triad}
    out["discriminating_pair"] = {
        n: {"med_E": ro[n]["theta_error"]["med_E"],
            "raw_loopscore": ro[n]["raw"]["margin20"]["loopscore"],
            "gauged_loopscore": ro[n]["gauged"]["margin20"]["loopscore"],
            "gauged_r2_percell": ro[n]["gauged"]["percell"]["r2"],
            "gauged_skill_vs_blind": [r for r in rows if r["name"] == n][0]
                                     ["gauged_percell_skill_vs_blind"]} for n in pair}

    out["nulls"] = S[0]["nulls"]
    # the per-cell instrument's own nulls, renormalised against the blind constant
    for n, v in out["nulls"].items():
        pc = v.get("percell")
        if isinstance(pc, dict) and pc.get("sse") is not None:
            v["percell_skill_vs_blind"] = 1.0 - pc["sse"] / sse_blind
    out["campaign_nulls"] = S[0]["campaign_nulls"]
    out["cite_status"] = S[0]["cite_status"]
    out["solves"] = S[0]["solves"]
    out["gauge_cost_rollouts"] = {r["name"]: r["gauge_extra_rollouts"] for r in rows}

    # ---- round 1 comparison: every RAW number must reproduce -----------------------------------
    r1 = json.load(open(os.path.join(HERE, "crash_round1.json")))
    cmp_ = {}
    for name in ro:
        k1 = f"{name}|free"
        if k1 in r1["rollouts"]:
            cmp_[name] = {
                "round1_loopscore": r1["rollouts"][k1]["margin20"]["loopscore"],
                "round2_raw_loopscore": ro[name]["raw"]["margin20"]["loopscore"],
                "abs_diff": abs(r1["rollouts"][k1]["margin20"]["loopscore"]
                                - ro[name]["raw"]["margin20"]["loopscore"])}
    out["round1_reproduction"] = {"per_candidate": cmp_,
                                  "max_abs_diff_loopscore":
                                      max(v["abs_diff"] for v in cmp_.values()),
                                  "n_compared": len(cmp_)}
    json.dump(out, open(os.path.join(HERE, "round2_summary.json"), "w"), indent=1)

    # ---- print ---------------------------------------------------------------------------------
    L = []

    def p(s):
        print(s)
        L.append(s)

    p(f"[repro] {out['round1_reproduction']['n_compared']} candidates shared with round 1; "
      f"max |dloopscore| (raw, free) = {out['round1_reproduction']['max_abs_diff_loopscore']:.2e}")
    p(f"[device] theta_true on {S[0]['config']['device']} vs {S[1]['config']['device']}: "
      f"loopscore {out['cross_device_control']['theta_true_loopscore']}, per-cell amplitude "
      f"field differs by {out['cross_device_control']['theta_true_a_percell_max_abs_diff']:.2e}")
    p("")
    p(f"{'candidate':<28s} {'medE':>6s} {'medg':>6s} | {'RAW':>7s} {'Erat':>6s} | {'k':>6s} "
      f"{'GAUGED':>7s} {'dloop':>7s} | {'R2raw':>7s} {'R2gau':>7s} | {'r2cell':>7s} {'skill':>7s}")
    for r in rows:
        p(f"{r['name']:<28s} {r['med_E']:>6.3f} {r['med_gain']:>6.3f} | "
          f"{r['raw_loopscore']:>7.4f} {r['raw_E_ratio']:>6.3f} | {r['k']:>6.3f} "
          f"{r['gauged_loopscore']:>7.4f} {r['d_loopscore_gauge']:>+7.4f} | "
          f"{r['raw_R2_disp']:>7.4f} {r['gauged_R2_disp']:>7.4f} | "
          f"{r['gauged_r2_percell']:>7.4f} {r['gauged_percell_skill_vs_blind']:>+7.3f}")
    p("")
    for tag in ("raw", "gauged"):
        q = reg[tag]
        p(f"[{tag:<6s}] corr(loopscore, log E_ratio) {q['corr_loopscore_vs_log_E_ratio']:+.3f}   "
          f"corr(loopscore, med|dE/E|) {q['corr_loopscore_vs_med_E']:+.3f}   "
          f"spearman {q['spearman_loopscore_vs_med_E']:+.3f}   "
          f"spearman(percell skill, med|dE/E|) {q['spearman_skill_vs_med_E']:+.3f}   "
          f"E_ratio in [{q['E_ratio_range'][0]:.3f}, {q['E_ratio_range'][1]:.3f}]")
    p("")
    p("[triad]  " + "; ".join(
        f"{n} medE {v['med_E']:.3f}: raw {v['raw_loopscore']:+.4f} -> gauged "
        f"{v['gauged_loopscore']:+.4f} (r2cell {v['gauged_r2_percell']:.4f}, skill "
        f"{v['gauged_skill_vs_blind']:+.3f})" for n, v in out["discriminating_triad"].items()))
    p("[pair ]  " + "; ".join(
        f"{n} medE {v['med_E']:.3f}: raw {v['raw_loopscore']:+.4f} -> gauged "
        f"{v['gauged_loopscore']:+.4f} (r2cell {v['gauged_r2_percell']:.4f}, skill "
        f"{v['gauged_skill_vs_blind']:+.3f})" for n, v in out["discriminating_pair"].items()))
    p("")
    p(f"[gauge] blind per-cell field already explains r2_percell = "
      f"{out['percell_skill_denominator']['r2_of_blind']:.4f} of the reference field; skill is "
      f"measured against that. extra rollouts per candidate: "
      f"{sorted(set(out['gauge_cost_rollouts'].values()))}")
    open(os.path.join(HERE, "round2_summary.txt"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
