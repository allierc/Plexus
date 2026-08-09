"""analyze_round3.py -- merge the round-3 shards, apply the diagnosis's acceptance criteria.

Reads crash_round3_s0.json / _s1.json (and optionally finject.json), writes round3_summary.json
and round3_summary.txt. Computes nothing new about the physics: it only re-normalises the per-cell
skill against the NULL BANK's best member and checks the four criteria the round-2 diagnosis set.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    S = []
    for i in (0, 1):
        p = os.path.join(HERE, f"crash_round3_s{i}.json")
        if os.path.exists(p):
            S.append(json.load(open(p)))
    if not S:
        S = [json.load(open(os.path.join(HERE, "crash_round3.json")))]
    return S


def main():
    S = load()
    out = {}
    ro = {}
    dup = {}
    for s in S:
        for k, v in s["rollouts"].items():
            if k in ro:
                dup[k] = v
            else:
                ro[k] = v
    a_ref = np.array(S[0]["a_ref_percell"], dtype=float)
    keep = np.array(S[0]["keep_percell"], dtype=bool)
    ar = a_ref[keep] / a_ref[keep].mean()
    sst = float(((ar - ar.mean()) ** 2).sum())
    bank = [n for n in S[0]["bank_names"] if n in ro]

    # -------- control: candidates run on BOTH shards must agree bit-exactly --------------------
    out["control_cross_gpu"] = {}
    for k, v in dup.items():
        a = np.array(ro[k]["gauged"]["a_percell"], dtype=float)
        b = np.array(v["gauged"]["a_percell"], dtype=float)
        out["control_cross_gpu"][k] = {
            "d_loop_raw": abs(ro[k]["raw"]["margin20"]["loopscore"]
                              - v["raw"]["margin20"]["loopscore"]),
            "d_loop_gauged": abs(ro[k]["gauged"]["margin20"]["loopscore"]
                                 - v["gauged"]["margin20"]["loopscore"]),
            "d_a_percell_max": float(np.nanmax(np.abs(a - b)))}

    def sse(name):
        f = np.array(ro[name]["gauged"]["a_percell"], dtype=float)[keep]
        if not np.isfinite(f).all() or f.mean() <= 0:
            return np.nan
        f = f / f.mean()
        return float(((f - ar) ** 2).sum())

    sse_bank = {n: sse(n) for n in bank}
    denom_name = min(sse_bank, key=lambda n: sse_bank[n])       # the BEST zero-information member
    denom = sse_bank[denom_name]
    out["skill_denominator"] = {"name": denom_name, "sse": denom, "sst": sst,
                                "r2_of_denominator": 1 - denom / sst,
                                "note": "skill = 1 - SSE/SSE_bank_best; the bank's best member "
                                        "reads 0 by construction, everything must beat 0"}

    rows = {}
    for k, v in ro.items():
        r, g = v["raw"], v["gauged"]
        s_ = sse(k)
        rows[k] = {"med_E": v["theta_error"]["med_E"], "med_gain": v["theta_error"]["med_gain"],
                   "mean_E": v["theta_error"].get("mean_E"),
                   "raw_loop": r["margin20"]["loopscore"], "raw_t1": r["t1"], "raw_t2": r["t2"],
                   "k_E": v["gauge"]["k_E"], "k_g": v["gauge"]["k_g"],
                   "gauge_status": v["gauge"]["status"], "n_extra": v["gauge"]["n_extra"],
                   "gau_t1": g["t1"], "gau_t2": g["t2"],
                   "gau_loop": g["margin20"]["loopscore"],
                   "gau_R2": g["coarse"]["R2_displacement_interior"],
                   "gau_orient": g["margin20"]["orientation_error"],
                   "gau_coord": g["margin20"]["coordination"],
                   "r2cell": g["percell"]["r2"],
                   "skill": (np.nan if not np.isfinite(s_) else 1.0 - s_ / denom),
                   "in_bank": k in bank,
                   "medE_after_gauge": v["gauge"]["theta_error_after_k"]["med_E"]}
    out["rows"] = rows

    # -------- the band ---------------------------------------------------------------------------
    bl = [rows[n]["gau_loop"] for n in bank]
    bs = [rows[n]["skill"] for n in bank]
    sub = {"constants": [n for n in bank if "blind_E" in n],
           "prior_draws": [n for n in bank if "prior_draw" in n],
           "geometry": [n for n in bank if "geom" in n]}
    out["null_band"] = {
        "members": bank, "n": len(bank),
        "loop": {"min": float(min(bl)), "max": float(max(bl)), "span": float(max(bl) - min(bl)),
                 "argmax": bank[int(np.argmax(bl))]},
        "skill": {"min": float(np.nanmin(bs)), "max": float(np.nanmax(bs)),
                  "span": float(np.nanmax(bs) - np.nanmin(bs)), "argmax": bank[int(np.nanargmax(bs))]},
        "sub_bands": {g: {"loop": [float(min(rows[n]["gau_loop"] for n in ns)),
                                   float(max(rows[n]["gau_loop"] for n in ns))],
                          "skill": [float(min(rows[n]["skill"] for n in ns)),
                                    float(max(rows[n]["skill"] for n in ns))]}
                      for g, ns in sub.items() if ns},
        "raw_loop_span": [float(min(rows[n]["raw_loop"] for n in bank)),
                          float(max(rows[n]["raw_loop"] for n in bank))]}
    top_loop, top_skill = out["null_band"]["loop"]["max"], out["null_band"]["skill"]["max"]
    for k in rows:
        rows[k]["above_band_loop"] = rows[k]["gau_loop"] - top_loop
        rows[k]["above_band_skill"] = rows[k]["skill"] - top_skill

    # -------- the four acceptance criteria -------------------------------------------------------
    crit = {}
    tt = rows.get("theta_true")
    crit["C1_gauge_identity"] = {
        "k_E": tt["k_E"], "k_g": tt["k_g"], "gau_loop": tt["gau_loop"],
        "pass": bool(abs(tt["k_E"] - 1) <= 0.005 and abs(tt["k_g"] - 1) <= 0.005
                     and tt["gau_loop"] > 0.999)}
    if "true_gain_x1.8" in rows and "true_E_x1.8" in rows:
        g1, e1 = rows["true_gain_x1.8"], rows["true_E_x1.8"]
        crit["C2_tautology_killed"] = {
            "true_gain_x1.8": [g1["gau_loop"], g1["skill"]],
            "true_E_x1.8": [e1["gau_loop"], e1["skill"]],
            "d_loop": abs(g1["gau_loop"] - e1["gau_loop"]),
            "d_skill": abs(g1["skill"] - e1["skill"]),
            "pass": bool(abs(g1["gau_loop"] - e1["gau_loop"]) <= 0.02
                         and abs(g1["skill"] - e1["skill"]) <= 0.02
                         and min(g1["gau_loop"], e1["gau_loop"]) >= 0.99
                         and min(g1["skill"], e1["skill"]) >= 0.99)}
    crit["C3_band_tight"] = {"loop_span": out["null_band"]["loop"]["span"],
                             "skill_span": out["null_band"]["skill"]["span"],
                             "pass": bool(out["null_band"]["loop"]["span"] <= 0.10
                                          and out["null_band"]["skill"]["span"] <= 0.5)}
    if "frame_DISP" in rows:
        fd, ff = rows["frame_DISP"], rows.get("theta_hat_frame_ridge0")
        crit["C4_discrimination"] = {
            "frame_DISP_above_band_loop": fd["above_band_loop"],
            "frame_DISP_above_band_skill": fd["above_band_skill"],
            "frame_ridge0_above_band_loop": None if ff is None else ff["above_band_loop"],
            "frame_ridge0_above_band_skill": None if ff is None else ff["above_band_skill"],
            "pass": bool(fd["above_band_loop"] >= 0.15 and fd["above_band_skill"] >= 0.5
                         and (ff is None or ff["above_band_loop"] <= 0))}
    out["criteria"] = crit

    # -------- correlations ------------------------------------------------------------------------
    names = [k for k in rows if "jitter" not in k]
    def corr(xs, ys):
        x, y = np.array(xs, float), np.array(ys, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return None
        return float(np.corrcoef(x[m], y[m])[0, 1])
    from scipy.stats import spearmanr
    me = [rows[k]["med_E"] for k in names]
    out["regressions"] = {
        "n": len(names),
        "corr_gauloop_logt1_raw": corr([np.log(max(rows[k]["raw_t1"], 1e-9)) for k in names],
                                       [rows[k]["gau_loop"] for k in names]),
        "corr_rawloop_logt1_raw": corr([np.log(max(rows[k]["raw_t1"], 1e-9)) for k in names],
                                       [rows[k]["raw_loop"] for k in names]),
        "corr_gauloop_medE": corr(me, [rows[k]["gau_loop"] for k in names]),
        "corr_skill_medE": corr(me, [rows[k]["skill"] for k in names]),
        "spearman_gauloop_medE": float(spearmanr(me, [rows[k]["gau_loop"] for k in names]).statistic),
        "spearman_skill_medE": float(spearmanr(me, [rows[k]["skill"] for k in names]).statistic),
        "corr_skill_logmeanE_BANK_CONSTANTS": corr(
            [np.log(rows[n]["mean_E"]) for n in bank if "blind_E" in n],
            [rows[n]["skill"] for n in bank if "blind_E" in n])}

    # -------- nulls -------------------------------------------------------------------------------
    nl = {}
    for s in S:
        nl.update(s.get("nulls", {}))
    out["nulls"] = {k: {"loopscore": v.get("loopscore"),
                        "t1": v.get("coarse", {}).get("motion_energy_ratio_interior"),
                        "R2": v.get("coarse", {}).get("R2_displacement_interior"),
                        "r2cell": (v.get("percell") or {}).get("r2"),
                        "skill": (np.nan if "a_percell" not in v else
                                  1.0 - (lambda f: float((((f / f.mean()) - ar) ** 2).sum()))(
                                      np.array(v["a_percell"], float)[keep]) / denom)}
                    for k, v in nl.items()}
    out["campaign_nulls"] = S[0].get("campaign_nulls")
    out["cite_status"] = S[0].get("cite_status")
    out["solves"] = S[0].get("solves")
    out["gauge_cost"] = {"total_extra_rollouts": int(sum(rows[k]["n_extra"] for k in rows)),
                         "n_not_converged": int(sum(1 for k in rows
                                                    if rows[k]["gauge_status"] != "converged"
                                                    and rows[k]["n_extra"] > 0)),
                         "not_converged": [k for k in rows
                                           if rows[k]["gauge_status"] != "converged"
                                           and rows[k]["n_extra"] > 0]}

    json.dump(out, open(os.path.join(HERE, "round3_summary.json"), "w"), indent=1, default=str)

    L = []
    L.append(f"skill denominator = {denom_name} (SSE {denom:.4f}, R2 {1-denom/sst:.4f})")
    L.append(f"null band: loop [{out['null_band']['loop']['min']:.4f}, "
             f"{out['null_band']['loop']['max']:.4f}] span {out['null_band']['loop']['span']:.4f}"
             f"   skill [{out['null_band']['skill']['min']:+.3f}, "
             f"{out['null_band']['skill']['max']:+.3f}] span {out['null_band']['skill']['span']:.3f}")
    L.append("")
    hdr = (f"{'candidate':<30s} {'medE':>7s} {'rawloop':>8s} {'t1':>6s} {'t2':>6s} {'kE':>7s} "
           f"{'kg':>6s} {'gauloop':>8s} {'gauR2':>8s} {'r2cell':>7s} {'skill':>7s} "
           f"{'dLoop':>7s} {'dSkill':>7s}")
    L.append(hdr)
    for k in sorted(rows, key=lambda k: -(rows[k]["gau_loop"] if rows[k]["gau_loop"] is not None else -9)):
        r = rows[k]
        L.append(f"{('* ' if r['in_bank'] else '  ')+k:<30s} {r['med_E']:>7.4f} "
                 f"{r['raw_loop']:>8.4f} {r['raw_t1']:>6.3f} {r['raw_t2']:>6.3f} {r['k_E']:>7.3f} "
                 f"{r['k_g']:>6.3f} {r['gau_loop']:>8.4f} {r['gau_R2']:>8.4f} "
                 f"{(r['r2cell'] if r['r2cell'] is not None else float('nan')):>7.4f} "
                 f"{r['skill']:>+7.3f} {r['above_band_loop']:>+7.3f} {r['above_band_skill']:>+7.3f}")
    L.append("")
    for k, v in crit.items():
        L.append(f"{k}: pass={v['pass']}  " + json.dumps({a: b for a, b in v.items() if a != 'pass'},
                                                         default=str))
    L.append("")
    L.append("nulls: " + json.dumps(out["nulls"], indent=1, default=str))
    L.append("controls (cross-GPU): " + json.dumps(out["control_cross_gpu"], default=str))
    L.append("gauge cost: " + json.dumps(out["gauge_cost"], default=str))
    L.append("regressions: " + json.dumps(out["regressions"], indent=1, default=str))
    txt = "\n".join(L)
    open(os.path.join(HERE, "round3_summary.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
