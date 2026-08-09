"""round4_report.py -- merge round 4's artefacts into one table.  No new runs, no new metrics.

  * the estimator table from round4_eiv.json / round4_stack*.json, with the prescription's five
    acceptance criteria evaluated literally;
  * the crash test from round4_diverge.json, raw and 2-D gauged, with per-cell SKILL measured
    against round 3's null-band top (bank_prior_draw_303, gauged loopscore 0.6488) which is re-run
    inside round4_diverge as its own control;
  * a gauge-invariant parameter error: med|dE/E| after the ORACLE per-block rescale, which is what
    the 2-D gauge quotients out -- so "the parameters are attenuated" and "the parameters are
    scattered" can be told apart;
  * the anchoring verdict and the divergence taxonomy, summarised.

usage: PYTHONPATH=/workspace/Plexus/src python round4_report.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BAND_TOP_ROUND3 = 0.6488          # crash_round3: converged NULL_BANK argmax, bank_prior_draw_303
SKILL_DENOM = "bank_prior_draw_303"


def load(n):
    p = os.path.join(HERE, n)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    out = {}
    ei = load("round4_eiv.json")
    dv = load("round4_diverge.json")
    st = [load(f) for f in ("round4_stack.json", "round4_stack_s555.json",
                            "round4_stack_s777.json")]
    st = [s for s in st if s]

    # ------------------------------------------------------------------ estimator criteria ----
    if ei:
        crit = {}
        for h, row in ei["summary"].items():
            crit[h] = {
                **row,
                "PASS_mean_ratio_1pm0.10": bool(abs(row["mean_ratio_median"] - 1) <= 0.10),
                "PASS_medE_le_0.10": bool(row["med_E_median"] <= 0.10),
                "PASS_spread_le_0.05": bool(row["med_E_spread"] <= 0.05),
                "PASS_no_draw_abs_mratio_gt2": not row["any_mean_ratio_abs_gt2"],
                "PASS_lam_min_positive": (row["lam_min_min"] is None or row["lam_min_min"] > 0)}
            crit[h]["n_criteria_met"] = sum(v for k, v in crit[h].items()
                                            if k.startswith("PASS_"))
        out["T1_criteria"] = crit
        out["T1_control_clean"] = ei["control_clean"]
        out["T1_control_sigma0"] = {
            "sigma_fro_over_G_fro": ei["control_sigma0"]["extra"]["sigma_fro_over_G_fro"],
            "med_E_naive": ei["control_sigma0"]["scores"]["naive"]["med_E"],
            "med_E_eiv_snr0": ei["control_sigma0"]["scores"]["eiv_snr0"]["med_E"]}
        out["T1_information"] = {
            "n_snr_gt1_median": float(np.median([ei["draws"][d]["extra"]["n_snr_gt1"]
                                                 for d in ei["draws"]])),
            "theta_true_energy_in_SNR_gt1_median": float(np.median(
                [ei["draws"][d]["theta_true_energy_in_SNR_gt1"] for d in ei["draws"]])),
            "theta_true_energy_in_SNR_lt0.3_median": float(np.median(
                [ei["draws"][d]["theta_true_energy_in_SNR_lt0.3"] for d in ei["draws"]]))}
        out["T1_wrong_sigma"] = {k: {"eiv_snr0_med_E": v["scores"]["eiv_snr0"]["med_E"],
                                     "eiv_snr0_mean_ratio": v["scores"]["eiv_snr0"]["mean_ratio_E"]}
                                 for k, v in ei.get("wrong_sigma", {}).items()}

    # ------------------------------------------------------------------ the stack -------------
    if st:
        rows = {}
        for T in ("T1", "T2", "T4", "T8"):
            for nm in ("naive", "eiv_snr0", "eiv_snr0.3", "eiv_snr1"):
                v = [s["runs"][T]["scores"][nm] for s in st if T in s["runs"]]
                if not v:
                    continue
                rows[f"{T}/{nm}"] = {
                    "n_seeds": len(v),
                    "med_E": [round(x["med_E"], 4) for x in v],
                    "med_E_mean": float(np.mean([x["med_E"] for x in v])),
                    "mean_ratio": [round(x["mean_ratio_E"], 3) for x in v],
                    "slope": [round(x["slope_E"], 3) for x in v],
                    "n_negE": [x["n_negE"] for x in v],
                    "rank": [s["runs"][T]["extra"]["ranks"].get(nm) for s in st if T in s["runs"]],
                    "n_snr_gt1": [s["runs"][T]["extra"]["n_snr_gt1"] for s in st if T in s["runs"]]}
        out["stack"] = rows

    # ------------------------------------------------------------------ the crash test --------
    if dv:
        g = dv["C_gauged"]
        keep = None
        ar = None
        # rebuild the per-cell reference from the identity row (theta_true's gauged field IS a_ref
        # to 0 by construction; use it, and record that)
        a_true = np.array(g["theta_true"]["gauged"]["a_percell"], dtype=float)
        keep = np.isfinite(a_true) & (a_true > 0)
        ar = a_true[keep] / a_true[keep].mean()
        sst = float(((ar - ar.mean()) ** 2).sum())

        def sse(name):
            f = np.array(g[name]["gauged"]["a_percell"], dtype=float)[keep]
            if not np.isfinite(f).all() or f.mean() <= 0:
                return np.nan
            return float(((f / f.mean() - ar) ** 2).sum())

        denom = sse(SKILL_DENOM) if SKILL_DENOM in g else np.nan
        rows = {}
        for k, v in g.items():
            gg = v["gauged"]
            b = dv["B_anchor"][k]
            s_ = sse(k)
            th_err = dv["candidate_theta_error"][k]
            rows[k] = {
                "med_E": th_err["med_E"], "med_gain": th_err["med_gain"],
                "raw_loop": b["free"]["margin20"]["loopscore"],
                "raw_t1": b["free"]["t1"], "raw_t2": b["free"]["t2"],
                "k_E": v["gauge"]["k_E"], "k_g": v["gauge"]["k_g"],
                "gauge_status": v["gauge"]["status"],
                "gau_loop": gg["margin20"]["loopscore"],
                "gau_R2": gg["coarse"]["R2_displacement_interior"],
                "gau_coord": gg["margin20"]["coordination"],
                "gau_orient": gg["margin20"]["orientation_error"],
                "r2cell": gg["percell"]["r2"] if gg.get("percell") else None,
                "skill": (np.nan if not np.isfinite(s_) or not np.isfinite(denom)
                          else 1.0 - s_ / denom),
                "above_band_top": (gg["margin20"]["loopscore"] - BAND_TOP_ROUND3
                                   if isinstance(gg["margin20"]["loopscore"], float) else None),
                "tags": dv["A_diagnosis"][k]["tags"],
                "rms_dx_final": dv["A_diagnosis"][k]["growth"]["rms_dx_final"],
                "slope_rel_per_frame": dv["A_diagnosis"][k]["growth"]["slope_rel_per_frame"]}
        out["crash"] = rows
        out["skill_denominator"] = {"name": SKILL_DENOM, "sse": denom, "sst": sst}
        # does the anchor repair exactly what it pins?  delta vs the error INSIDE the band
        ba = dv["B_anchor"]
        nm = [k for k in ba if isinstance(ba[k]["free"]["margin20"]["loopscore"], float)
              and isinstance(ba[k]["free"]["margin10"]["loopscore"], float)]
        bandrms = np.array([ba[k]["free"]["coarse"]["rms_pos_err_dx_BAND_mean"] for k in nm])
        d20 = np.array([ba[k]["anchored"]["margin20"]["loopscore"]
                        - ba[k]["free"]["margin20"]["loopscore"] for k in nm])
        d10 = np.array([ba[k]["anchored"]["margin10"]["loopscore"]
                        - ba[k]["free"]["margin10"]["loopscore"] for k in nm])
        out["anchor"] = dict(dv["B_summary"])
        out["anchor"].update({
            "max_abs_delta_loopscore_m10": float(np.abs(d10).max()),
            "median_abs_delta_loopscore_m10": float(np.median(np.abs(d10))),
            "corr_delta10_vs_band_rms": float(np.corrcoef(bandrms, d10)[0, 1]),
            "corr_delta20_vs_band_rms": float(np.corrcoef(bandrms, d20)[0, 1]),
            "ratio_median_delta10_over_delta20": float(np.median(np.abs(d10))
                                                       / max(np.median(np.abs(d20)), 1e-12)),
            "anchor_delta_rms_dx_at_theta_true": float(
                ba["theta_true"]["anchored"]["coarse"]["rms_pos_err_dx_mean"]
                - ba["theta_true"]["free"]["coarse"]["rms_pos_err_dx_mean"]),
            "n": len(nm)})
        out["taxonomy"] = {k: {kk: v[kk] for kk in
                               ("coordination", "path_length", "peak_excursion",
                                "orientation_error", "loopscore", "peak_ratio", "openness",
                                "lag_global_halfperiod_frames", "lag_iqr_frames", "tags")}
                           for k, v in dv["A_modes"].items()}

        print(f"\n{'='*140}\n  ROUND 4 -- the crash test, raw and 2-D gauged "
              f"(band top from round 3 = {BAND_TOP_ROUND3:.4f})\n{'='*140}")
        print(f"  {'candidate':<28s} {'medE':>7s} {'raw':>8s} {'t1':>6s} {'kE':>6s} {'kg':>6s} "
              f"{'gauged':>8s} {'vs band':>8s} {'R2':>8s} {'r2cell':>7s} {'skill':>7s} "
              f"{'rmsend':>7s} | tags")
        for k in sorted(rows, key=lambda z: -(rows[z]["gau_loop"]
                                              if isinstance(rows[z]["gau_loop"], float) else -9)):
            r = rows[k]
            f = lambda x, w=8: (f"{x:>{w}.4f}" if isinstance(x, float) and np.isfinite(x)
                                else f"{'n/a':>{w}}")
            print(f"  {k:<28s} {r['med_E']:>7.4f} {f(r['raw_loop'])} {r['raw_t1']:>6.3f} "
                  f"{r['k_E']:>6.3f} {r['k_g']:>6.3f} {f(r['gau_loop'])} "
                  f"{f(r['above_band_top'])} {f(r['gau_R2'])} {f(r['r2cell'],7)} "
                  f"{f(r['skill'],7)} {r['rms_dx_final']:>7.4f} | {','.join(r['tags'])}")
        print(f"\n  ANCHOR: " + json.dumps(dv["B_summary"], indent=1)[:1200])

    json.dump(out, open(os.path.join(HERE, "round4_report.json"), "w"), indent=1, default=str)
    print(f"\nwrote round4_report.json")


if __name__ == "__main__":
    main()
