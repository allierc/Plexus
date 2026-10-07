"""exp17: HOW MUCH EACH TERM OF THE LAW MOVES, PER NEURON (Cedric, 2026-10-05: "the network, stimulus and leak terms per
neuron; choose representative cells in line with the table"). Local, from a run's free rollout (results/<run>_movie.npz,
800 frames over the 2 h; Omega from <run>_omega.npz) and its learned constants.

The law, per neuron i, in normalised units z = (x - mu) / sd (the recording's mean and SD):
    z_i <- z_i + f_i ( (V_i - z_i) + Omega_i(t) m_i(t) + B_i . u(t) ),   m_i = sum_j W_ji tanh(z_j)
Per neuron, the SD over the 2 h of each term: the network Omega m, the stimulus B . u (0 outside the input mask), the leak
pull V - z. The medians over all, input and non-input neurons, and the REPRESENTATIVE neurons: in each group the neuron
whose three SDs are closest (in log) to the group's medians -- two non-input neurons (the front and the back half of the
brain; they are 80 % of it) and one input neuron.

    PYTHONPATH=src:tools python tools/exp17_terms.py zap_e15_cur_siren_mesh3 [--sets 5]
Writes data/terms_<run>.json. With --sets N (Cedric, 2026-10-05: "other locations x5"): also data/terms_<run>_s<k>.json,
the same three picks made inside set k's regions only (tools/exp17_traces.py --sets: each neuron belongs to its nearest
region centre), the medians still the whole brain's, the whole brain's own three left out (they would repeat).
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
G = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "log", "training", "zapbench")


def main(run, device="cuda:0", n_sets=0):
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import neuron_graph_op, _brain_view
    spec = T.load(run)
    out = T.out_dir(spec, None)
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    op = neuron_graph_op(spec, "cpu")
    fit = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    z = np.load(os.path.join(G, run, "results", f"{run}_movie.npz"))
    fr = z["frames"]
    mu, sd = float(rec["dff"].mean()), float(rec["dff"].std())
    Zp = torch.as_tensor((z["pred"].astype(np.float32) - mu) / sd, device=device)
    A = torch.tanh(torch.nan_to_num(Zp))
    omp = os.path.join(G, run, "results", f"{run}_omega.npz")
    om = (torch.as_tensor(np.load(omp)["omega"].astype(np.float32), device=device) if op.modulation != "none"
          else torch.ones_like(A))
    M = torch.zeros_like(A)
    for s in op.EDGE_SETS:
        e = next((l for l in spec["learnable"] if l.get("param") == f"W_{s}"), None)
        if e is None:
            continue
        w = fit[T.Learnables.key(e)].float().to(device).reshape(-1)
        snd, rcv = (t.to(device) for t in op._E[s])
        M.index_add_(1, rcv, A[:, snd] * w[None])
    net = om * M
    B = fit["neuron.input"].float().to(device)
    mask = (op.input_mask.to(device).reshape(-1) if op.input_mask is not None else torch.ones(A.shape[1], device=device))
    u = torch.as_tensor(np.asarray(rec["stimulus"], np.float32)[fr], device=device)
    stim = (u @ B.T) * mask[None]
    leak = fit["neuron.rest"].float().to(device)[:, 0][None] - Zp
    ok = (torch.isfinite(Zp).all(0) & (torch.as_tensor(np.asarray(rec["dff"]).std(0), device=device) > 1e-6)).cpu().numpy()
    S = {k: v.std(0).cpu().numpy() for k, v in (("network", net), ("stimulus", stim), ("leak", leak))}
    inp = (mask > 0).cpu().numpy()
    groups = {"all": ok, "input": ok & inp, "non_input": ok & ~inp}
    med = {g: {k: float(np.median(S[k][m])) for k in S} | {"n": int(m.sum())} for g, m in groups.items()}
    P = _brain_view(np.asarray(rec["pos_um"], np.float64))
    xmid = float(np.median(P[:, 0]))

    def closest(m, keys, med_):
        d = sum((np.log(S[k][m] + 1e-6) - np.log(med_[k] + 1e-6)) ** 2 for k in keys)
        return int(np.where(m)[0][np.argmin(d)])

    def picks_in(region):
        picks = []
        for half, sel in (("front", P[:, 0] < xmid), ("back", P[:, 0] >= xmid)):
            i = closest(groups["non_input"] & sel & region, ("network", "leak"), med["non_input"])
            picks.append({"label": f"non-input, {half} half", "index": i})
        picks.append({"label": "input", "index": closest(groups["input"] & region, ("network", "stimulus", "leak"),
                                                         med["input"])})
        for p_ in picks:
            p_.update({k: float(S[k][p_["index"]]) for k in S})
        return picks
    doc = {"run": run, "medians": med, "picks": picks_in(np.ones(len(P), bool))}
    json.dump(doc, open(os.path.join(EXP, "data", f"terms_{run}.json"), "w"), indent=1)
    print(json.dumps(doc, indent=1))
    if n_sets:
        TS = [json.load(open(os.path.join(EXP, "data", f"traces_{run}_s{k}.json"))) for k in range(1, n_sets + 1)]
        C = np.concatenate([np.asarray(t["region_centres_view_um"]) for t in TS])
        owner = np.concatenate([np.full(len(t["region_centres_view_um"]), k) for k, t in enumerate(TS, 1)])
        near = np.empty(len(P), int)
        for a in range(0, len(P), 20000):                                    # each neuron's nearest region centre
            near[a:a + 20000] = np.argmin(((P[a:a + 20000, None] - C[None]) ** 2).sum(-1), 1)
        for k in range(1, n_sets + 1):
            reg_ = owner[near] == k
            reg_[[p_["index"] for p_ in doc["picks"]]] = False          # new neurons only: not the whole brain's three
            d_ = {"run": run, "set": k, "medians": med, "picks": picks_in(reg_)}
            json.dump(d_, open(os.path.join(EXP, "data", f"terms_{run}_s{k}.json"), "w"), indent=1)
            print(f"[terms] set {k}:", [(p_["label"], p_["index"]) for p_ in d_["picks"]])


if __name__ == "__main__":
    main(sys.argv[1], n_sets=int(sys.argv[sys.argv.index("--sets") + 1]) if "--sets" in sys.argv else 0)
