"""Surface proteins of an epithelium on a live basement membrane (exp 11 Phase 3, 2026-09-29).

    bm_clutch           integrin beta1 on the tissue's basal cap binding the membrane's laminin: the cell's
                        receptor pools (inactive, active, endosomal, bound) and the bonds at each membrane node,
                        nascent and reinforced, under the load the node's gap puts on them
    bm_mass             the membrane as matter: each node's laminin + collagen IV, laid down by the cells and
                        cut by their MT2-MMP; a node whose mass collapses is gone
    cell_protein_level  per-cell protein levels relaxing toward their compartment's level with a memory:
                        E-cadherin (low on the surface lineage), MT2-MMP

WHY A NEW OPERATOR AND NOT A VARIANT (the registry, 2026-09-29). No existing contract holds receptor-ligand
CHEMISTRY between a vertex tissue's basal cap and a live membrane. `bm_contact[live]` is the mechanics of that
contact (non-penetration and a tether, no molecules); `bm_sense[live]` turns the membrane's node density into a
growth signal; `integrin_adhesion` and `adhesion_*` bind a RECORDED surface; `protein_*` places clusters as
points, which at about 100 integrins per um^2 would be about 10^7 points on a gland of 1,200 cells. The clutch
of the okuda_ECM prototype (discovery_okuda/ops/adhesion_ops.py) is the ancestor; what it got wrong is listed in
experiments/exp11_bm_hole_budding/SURFACE_PROTEINS_AUDIT.md and is not repeated here:

    - receptors made at division (x25.7): here every pool is a DENSITY per unit basal area, so the row copy
      a division makes, the labels a dive and a return carry, and an MPM daughter's copy are all correct as
      they stand; nothing is split or refilled by hand
    - a global ligand: here the ligand is each node's own
    - a Bell cliff at an undeclared force scale: here the load is the node's STRETCH, the gap past the contact
      standoff, over `delta_b`, a length measured off the working point (exp 11 md, Decisions 2026-09-29)

WHY BONDS HAVE NO STATE OF THEIR OWN. Integrin-ligand bonds form and break in seconds (k_on, k_off 0.1-6.5 per
s; the reinforced state lasts ~100 s, Kong 2013), and a frame is 600 s. Within a frame they reach their steady
state, so they are SOLVED each frame from the cell's active pool and the node's ligand and load, never carried.
What persists is slow: trafficking (minutes), synthesis (hours), and -- from B3 -- the membrane's mass.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from plexus.models.base import Structural
from plexus.models.registry import register_operator
from plexus.operators.membrane_ops import _basal_lookup, _live_basal, _nearest_faces

# the bound pool, the ledger and the load, per recorded row: what the gates of B1 read
CLUTCH_TRACE = []


@register_operator("bm_clutch", family="boundary", set="particle", kind="structural",
                   title="Integrin-laminin clutch",
                   equation=r"""$$T_k=\frac{k_{on}A_iM_k(1+\rho_k)}{k_C+k_r+k_{on}A_i(1+\rho_k)/\rho_L},\quad \rho_k=\frac{k_r}{k_P},\quad k_C=k_{off}e^{x_k},\ k_r=k_{r0}e^{x_k},\ k_P=k_{P0}e^{x_k/s_P},\quad x_k=\frac{(g_k-\text{offset})_+}{\delta_b}$$""")
class BasementMembraneClutch(Structural):
    """Integrin beta1 on each cell's basal cap binding laminin on the membrane nodes it touches.

    basement_membrane_node x cell -> cell, basement_membrane_node: reads the live basal surface and the node
    positions; writes the cell's integrin pools and each node's bonds and free ligand.

    Per cell i, densities per unit basal area in units of the resting total U0 (1 = about 100 integrins per
    um^2, Paszek 2009):

        U_i = I_i + A_i + N_i + B_i          total integrin: inactive, active, endosomal, bound
        A_i = phi (I_i + A_i)                activation, in fast equilibrium (k_a 0.5/s, k_d 5/s:
                                             phi = k_a / (k_a + k_d) = 0.091; `activation` scales k_a --
                                             Mn2+ or ROCK in Wang's and Daley's arms)

    Per membrane node k in contact with cell i (gap g_k <= offset + band), the bonds at steady state:

        x_k = (g_k - offset)_+ / delta_b     the load, as stretch past the standoff in units of delta_b
        k_C = k_off e^x                      nascent bond, slip (Bell 1978)
        k_r = k_r0 e^x                       reinforcement under load (talin R3 unfolds at ~5 pN, Yao 2016)
        k_P = k_P0 e^(x / s_P)               the reinforced plaque lets go, s_P times more slowly with load
        C_k = k_on A_i (L_k / rho_L) / (k_C + k_r)    nascent; P_k = (k_r / k_P) C_k reinforced; T_k = C_k + P_k
        L_k = rho_L M_k - T_k                free ligand; M_k = 1 until the membrane has a mass (B3)

    k_on is the binding rate at the resting ligand density rho_L. With the ligand the bonds use taken out,
    C_k = k_on A_i M_k / (k_C + k_r + k_on A_i (1 + rho_k) / rho_L), rho_k = k_r / k_P. The
    cell's bound density is B_i = c_i mean_k T_k, with c_i the covered fraction of its basal area (its nodes per
    unit area over the tissue's median at the first call, capped at 1): a cell over a hole binds less. A_i and
    B_i are solved together (bisection on A_i, since B_i rises with A_i and A_i = phi (U_i - N_i - B_i)).

    Trafficking over the frame, explicitly in `substeps` steps, on the FREE pool only (engaged receptors are
    not endocytosed):

        dN/dt = k_en (I + A) - (k_rec + k_deg) N
        dU/dt = s - k_deg N,   s = k_deg N + (supply U0 - U) / tau_U

    (Arjonen 2012: 45 % of surface beta1 endocytosed per 30 min, 70 % of it recycled per 30 min, lysosomal
    half-life 18 h -> k_en 0.020, k_rec 0.040, k_deg 6.4e-4 per min.) `supply` < 1 is the beta1 block;
    tau_U the protein's turnover (12-24 h, Moreno-Layseca 2019).

    THE FIRST CALL, AND A CELL WITH NO INTEGRIN (U = 0: a set that does not declare the blocks), start at the
    resting split U = U0, N = U0 k_en / (k_en + k_rec + k_deg), nothing bound, and are counted in
    `m["itg_fresh"]`. `m["itg_carry_err"]`, the largest change of U between two calls for a cell that
    persisted, or between a mother and her daughter, is the B1 ledger: every other operator only copies it.

    READ-ONLY: no force is applied here. `bm_contact[live]` stays the membrane's mechanics; the coupling of the
    bonds to it is B2's.

    Parameters: `surface` (vertex), `cell_set` (cell), `sep_block` (sep), `offset` (0.05) and `band` (0.3) as
    the contact's, `delta_b` (0.05, world length), `rho_L` (3.0, ligand per resting integrin: 300 vs 100 per
    um^2), `k_on` (1.0 /s), `k_off` (1.0 /s), `k_r0` (0.1 /s), `k_P0` (0.01 /s), `s_P` (5.0), `phi` (0.091),
    `activation` (1.0), `k_en` (0.020 /min), `k_rec` (0.040 /min), `k_deg` (6.4e-4 /min), `U0` (1.0), `tau_U`
    (1080 min = 18 h), `supply` (1.0), `frame_s` (600 s per frame), `substeps` (10). Blocks written: on the cell
    set `itg_I`, `itg_A`, `itg_N`, `itg_B` (densities per unit basal area, in U0); on the membrane `bond_C`,
    `bond_P`, `lig_L` as AMOUNTS (density times the node's share of its cell's covered basal area, in U0 x
    world area: summed over the membrane, bond_C + bond_P is the cells' bound integrin summed as B_i x covered
    area) and `bm_gap` (g - offset, signed). A block the spec does not declare is not written.

    References: Paszek, M. J. et al. (2009). PLoS Comput Biol 5:e1000604 (densities, activation); Arjonen, A.
    et al. (2012). Traffic 13:610 (trafficking); Kong, F. et al. (2009). J Cell Biol 185:1275 and (2013) Mol Cell
    49:1060 (catch and long-lived states); Yao, M. et al. (2016). Nat Commun 7:11966 (talin); Bell, G. I. (1978).
    Science 200:618; Wang, S. et al. (2021). Cell 184:3702 (integrin beta1 in the surface cells).
    """
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    PARAM_ROLES = {"delta_b": "bond_load_length_scale", "rho_L": "ligand_per_resting_integrin",
                   "k_on": "binding_rate_per_s", "k_off": "nascent_off_rate_per_s",
                   "k_r0": "reinforcement_rate_per_s", "k_P0": "plaque_release_rate_per_s",
                   "s_P": "plaque_load_scale_ratio", "phi": "active_fraction", "activation": "activation_scale",
                   "k_en": "endocytosis_per_min", "k_rec": "recycling_per_min", "k_deg": "degradation_per_min",
                   "U0": "resting_total", "tau_U": "synthesis_relaxation_min", "supply": "synthesis_scale"}

    def __init__(self, params, device="cpu"):
        Structural.__init__(self, params, device)
        g = lambda k, d: float(params.get(k, d))                     # noqa: E731
        self.at = params.get("_at", "bm_node")
        self.surface = str(params.get("surface", "vertex"))
        self.cat = str(params.get("cell_set", "cell"))
        self.sep_block = str(params.get("sep_block", "sep"))
        self.offset, self.band, self.delta_b = g("offset", 0.05), g("band", 0.3), g("delta_b", 0.05)
        self.rho_L, self.k_on, self.k_off = g("rho_L", 3.0), g("k_on", 1.0), g("k_off", 1.0)
        self.k_r0, self.k_P0, self.s_P = g("k_r0", 0.1), g("k_P0", 0.01), g("s_P", 5.0)
        # activation scales k_a: phi / (1 - phi) = k_a / k_d is what it multiplies
        _p0, _act = g("phi", 0.5 / 5.5), g("activation", 1.0)
        _r = _p0 / max(1.0 - _p0, 1e-12) * _act
        self.phi = _r / (1.0 + _r)
        self.k_en, self.k_rec, self.k_deg = g("k_en", 0.020), g("k_rec", 0.040), g("k_deg", 6.4e-4)
        self.U0, self.tau_U, self.supply = g("U0", 1.0), g("tau_U", 1080.0), g("supply", 1.0)
        self.frame_min = g("frame_s", 600.0) / 60.0
        self.substeps = max(1, int(params.get("substeps", 10)))
        for k in ("delta_b", "rho_L", "k_on", "k_off", "tau_U"):
            if getattr(self, k) <= 0:
                raise ValueError(f"bm_clutch: {k} must be > 0, got {getattr(self, k)}")
        if not 0.0 < self.phi < 1.0:
            raise ValueError(f"bm_clutch: the active fraction phi must lie in (0, 1), got {self.phi}")
        self._geo = _basal_lookup()
        self._rho_n0 = None
        self._last = None                                             # {cell_id: U} at the end of the last call
        self.mass_block = str(params.get("mass_block", "bm_M"))       # the ligand scales with it when declared
        self._mass = None
        self._said = False

    # ------------------------------------------------------------------ helpers
    def _blocks(self, lvl, names):
        sch = getattr(lvl, "state_schema", {})
        return {k: sch[k][0] for k in names if k in sch and sch[k][1] - sch[k][0] == 1}

    def _bound(self, A, Mk, kC, kr, rho, cell, nF, cov):
        """B_i = cov_i mean_k T_k for active densities A [nF], over the contacting nodes."""
        a = A[cell]
        T = self.k_on * a * Mk * (1.0 + rho) / (kC + kr + self.k_on * a * (1.0 + rho) / self.rho_L)
        num = torch.zeros(nF, dtype=A.dtype, device=A.device).index_add_(0, cell, T)
        cnt = torch.zeros(nF, dtype=A.dtype, device=A.device).index_add_(0, cell, torch.ones_like(T))
        return cov * num / cnt.clamp_min(1.0), T

    def _solve(self, U, N, gap, cell, nF, cov):
        """The steady bonds and the active pool that feeds them: A_i = phi (U_i - N_i - B_i) with
        B_i = cov_i mean_k T_k(A_i), by bisection (B rises with A). Returns A [nF], B [nF], and per contact
        C, P (nascent, reinforced) and the load x."""
        x = ((gap - self.offset).clamp_min(0.0) / self.delta_b).clamp(max=40.0)
        kC, kr = self.k_off * torch.exp(x), self.k_r0 * torch.exp(x)
        rho = kr / (self.k_P0 * torch.exp(x / self.s_P))
        Mk = self._mass if self._mass is not None else torch.ones_like(gap)   # the node's mass (bm_mass, B3)
        free_max = (U - N).clamp_min(0.0)
        lo, hi = torch.zeros_like(U), self.phi * free_max
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            Bm, _T = self._bound(mid, Mk, kC, kr, rho, cell, nF, cov)
            over = mid > self.phi * (free_max - Bm)
            hi = torch.where(over, mid, hi); lo = torch.where(over, lo, mid)
        A = 0.5 * (lo + hi)
        B, _T = self._bound(A, Mk, kC, kr, rho, cell, nF, cov)
        B = torch.minimum(B, free_max)
        aN = A[cell]
        C = self.k_on * aN * Mk / (kC + kr + self.k_on * aN * (1.0 + rho) / self.rho_L)
        return A, B, C, rho * C, x

    # ------------------------------------------------------------------ the frame
    def forward(self, H, mask=None):
        lvl, clvl = H.level(self.at), H.level(self.cat)
        tl = H.level(self.surface)
        m = getattr(tl, "_mesh", None) if tl is not None else None
        if m is None or clvl is None or not int(m.get("nF", 0) or 0):
            return {}
        nF = int(m["nF"])
        cb = self._blocks(clvl, ("itg_I", "itg_A", "itg_N", "itg_B", "cell_id", "parent_id"))
        if not all(k in cb for k in ("itg_I", "itg_A", "itg_N", "itg_B")):
            raise ValueError(f"bm_clutch: the cell set {self.cat!r} must declare width-1 blocks itg_I, itg_A, "
                             f"itg_N and itg_B")
        nb = self._blocks(lvl, ("bond_C", "bond_P", "lig_L", "bm_gap"))
        st = clvl.state
        dev = st.device
        f64 = torch.float64
        col = lambda k: st[:nF, cb[k]].to(f64)                        # noqa: E731
        I, A, N, B = col("itg_I"), col("itg_A"), col("itg_N"), col("itg_B")
        U = I + A + N + B
        fresh = U <= 0
        n_fresh = int(fresh.sum())
        if n_fresh:
            U = torch.where(fresh, torch.full_like(U, self.U0), U)
            N = torch.where(fresh, torch.full_like(N, self.U0 * self.k_en / (self.k_en + self.k_rec + self.k_deg)), N)
            B = torch.where(fresh, torch.zeros_like(B), B)
        # THE LEDGER: U is only ever copied by the operators between two calls
        carry_err = 0.0
        if "cell_id" in cb:
            cid = st[:nF, cb["cell_id"]].detach().cpu().numpy().astype(np.int64)
            pid = (st[:nF, cb["parent_id"]].detach().cpu().numpy().astype(np.int64) if "parent_id" in cb
                   else np.full(nF, -1))
            u_now = U.detach().cpu().numpy()
            if self._last is not None:
                errs = []
                for i in np.flatnonzero(~fresh.cpu().numpy()):
                    ref = self._last.get(int(cid[i]), self._last.get(int(pid[i])))
                    if ref is not None:
                        errs.append(abs(u_now[i] - ref))
                carry_err = float(max(errs)) if errs else 0.0
        # THE CONTACT: each live node's nearest basal fan triangle, its cell and its gap
        pos = lvl.get("pos")
        alive = getattr(H, "membrane_alive", None)
        alive = (torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device) if alive is None
                 else alive.to(pos.device))
        idx = alive.nonzero(as_tuple=True)[0]
        got = _live_basal(H, self.surface, self.sep_block, self._geo, pos.device, pos.dtype)
        cov = torch.zeros(nF, dtype=f64, device=dev)
        farea = torch.zeros(nF, dtype=f64, device=dev)
        node, cell, gap = idx[:0], idx[:0], torch.zeros(0, dtype=f64, device=dev)
        share = torch.zeros(0, dtype=f64, device=dev)
        if got is not None and idx.numel():
            _m, c, M = got
            hit, tri, _w, _n, g_ = _nearest_faces(M, c, pos[idx])
            node, tri, g_ = idx[hit], tri[hit], g_[hit]
            face = M["ef"][tri]
            touch = (g_ <= self.offset + self.band) & (face < nF)
            node, cell, gap = node[touch], face[touch].to(dev), g_[touch].to(device=dev, dtype=f64)
            area = 0.5 * torch.cross(M["B"] - M["A"], M["C"] - M["A"], dim=1).norm(dim=1)
            farea = torch.zeros(nF, dtype=f64, device=dev).index_add_(0, M["ef"].to(dev), area.to(device=dev, dtype=f64))
            dens = torch.zeros(nF, dtype=f64, device=dev).index_add_(0, cell, torch.ones_like(gap)) / farea.clamp_min(1e-12)
            if self._rho_n0 is None:
                self._rho_n0 = float(dens[dens > 0].median()) if bool((dens > 0).any()) else 1.0
            cov = (dens / max(self._rho_n0, 1e-12)).clamp(0.0, 1.0)
            # each contact's share of its cell's covered basal area: what turns a density into an amount
            cnt_c = torch.zeros(nF, dtype=f64, device=dev).index_add_(0, cell, torch.ones_like(gap))
            share = (farea * cov / cnt_c.clamp_min(1.0))[cell]
        # THE LIGAND IS THE NODE'S MASS where the membrane has one (bm_mass, B3): L_k = rho_L M_k - T_k
        self._mass = (lvl.get(self.mass_block)[node, 0].to(device=dev, dtype=f64).clamp_min(0.0)
                      if self.mass_block in getattr(lvl, "state_schema", {}) else None)
        # THE BONDS AT STEADY STATE, with the active pool solved alongside
        A, B, C, P, x = self._solve(U, N, gap, cell, nF, cov)
        F = (U - N - B).clamp_min(0.0)
        # TRAFFICKING over the frame, on the free pool
        h = self.frame_min / self.substeps
        for _ in range(self.substeps):
            s = self.k_deg * N + (self.supply * self.U0 - U) / self.tau_U
            dN = self.k_en * F - (self.k_rec + self.k_deg) * N
            dU = s - self.k_deg * N
            N = (N + h * dN).clamp_min(0.0)
            U = (U + h * dU).clamp_min(0.0)
            F = (U - N - B).clamp_min(0.0)
        A = self.phi * F
        I = F - A
        dt_ = st.dtype
        for k, v in (("itg_I", I), ("itg_A", A), ("itg_N", N), ("itg_B", B)):
            st[:nF, cb[k]] = v.to(dt_)
        if nb:
            ns = lvl.state
            for k in nb:
                ns[idx, nb[k]] = 0.0
            # AMOUNTS, not densities: a node's bonds are its share of the cell's bound integrin, so the
            # membrane's total bonds and the cells' total bound integrin are one number
            if "bond_C" in nb:
                ns[node, nb["bond_C"]] = (C * share).to(ns.dtype)
            if "bond_P" in nb:
                ns[node, nb["bond_P"]] = (P * share).to(ns.dtype)
            if "lig_L" in nb:
                _Mk = self._mass if self._mass is not None else torch.ones_like(C)
                ns[node, nb["lig_L"]] = ((self.rho_L * _Mk - C - P) * share).to(ns.dtype)
            if "bm_gap" in nb:
                ns[node, nb["bm_gap"]] = (gap - self.offset).to(ns.dtype)
        if "cell_id" in cb:
            u_end = (I + A + N + B).detach().cpu().numpy()
            self._last = {int(k): float(v) for k, v in zip(cid, u_end)}
        mm = getattr(tl, "_mesh", None)
        if isinstance(mm, dict):
            mm["clutch_map"] = (getattr(H, "frame", None), node.detach(), cell.detach())
            # the matrix each cell stands on, by cell_id, for the return rate (cell_divide[reinsert] p_from
            # adhesion): its covered fraction x the mean mass of the nodes it touches
            if "cell_id" in cb:
                _cm = torch.zeros(nF, dtype=f64, device=dev).index_add_(0, cell, self._mass) if self._mass is not None \
                    else torch.zeros(nF, dtype=f64, device=dev).index_add_(0, cell, torch.ones_like(gap))
                _cn = torch.zeros(nF, dtype=f64, device=dev).index_add_(0, cell, torch.ones_like(gap)).clamp_min(1.0)
                mm["clutch_by_id"] = (cid.copy(), (cov * _cm / _cn).detach().cpu().numpy())
            # the same per face, for `bm_sense[live] source: matrix` scheduled after this operator in the frame
            mm["clutch_matrix"] = (cov * _cm / _cn).detach()
            mm["clutch_matrix_frame"] = getattr(H, "frame", None)
            mm["itg_fresh"] = int(mm.get("itg_fresh", 0)) + n_fresh
            mm["itg_carry_err"] = max(float(mm.get("itg_carry_err", 0.0)), carry_err)
            mm["itg_bound_frac"] = float((B.sum() / (I + A + N + B).sum().clamp_min(1e-12)))
            mm["itg_bound_amount"] = float((B * farea).sum()) if node.numel() else 0.0
        CLUTCH_TRACE.append((getattr(H, "frame", None), n_fresh, carry_err,
                             float(B.mean()), float(x.mean()) if x.numel() else 0.0, int(node.numel())))
        if not self._said:
            self._said = True
            print(f"[bm_clutch] integrin-laminin clutch on `{self.surface}` x `{self.at}` (read-only): "
                  f"delta_b {self.delta_b:g}, rho_L {self.rho_L:g}, phi {self.phi:.3g}, supply {self.supply:g}; "
                  f"frame {getattr(H, 'frame', None)}: {int(node.numel())} nodes in contact, bound fraction "
                  f"{float(B.sum() / (I + A + N + B).sum().clamp_min(1e-12)):.3f}, {n_fresh} cells started fresh",
                  flush=True)
        return {}


@register_operator("bm_mass", family="population", set="particle", kind="structural",
                   title="Basement-membrane mass turnover",
                   equation=r"""$$\dot M_k=k_M\,(s-Z_kM_k)-k_{coll}M_k,\qquad Z_k=z_{i(k)}(1-\iota)$$""")
class BasementMembraneMass(Structural):
    """The membrane as matter: laminin + collagen IV per node, laid down by the cells and cut by their
    protease (exp 11 Phase 3 B3, 2026-09-29).

    basement_membrane_node x cell -> basement_membrane_node, cell: reads the contact map `bm_clutch` published
    at its last call (node -> cell); writes each node's mass and each cell's MT2-MMP activity.

        dM_k/dt = k_M (s - Z_k M_k) - k_coll M_k        per node, in units of a seeded node's mass
        Z_k     = z_i (1 - inhibit)                     the protease of the cell i the node lies on;
                                                        z_free (1 - inhibit) on a node no cell touches

    s is the deposition (1: at rest M = s / Z = 1 when Z = 1), k_M the turnover rate: with the protease
    blocked (inhibit 1) the membrane thickens at k_M s per frame, and k_M = 0.028 per frame makes it 3x in
    72 frames -- Harunaga 2014's 3x collagen IV in 12 h under BB-94 -- which is also a half-life of
    ln 2 / k_M = 25 frames = 4.1 h, the FDAP turnover of Drosophila's collagen IV (Matsubayashi 2020). The
    protease is MT2-MMP, the gland's EPITHELIAL MT-MMP (Rebustini 2009): z_i is the cell's `mt2` level
    (`cell_protein_level`) where the set declares it, else `z0`. `k_coll` is exogenous collagenase (Wang 2021's arm). Integrated exactly over the frame
    for the frame's Z.

    A NODE WHOSE MASS FALLS BELOW `m_death` IS GONE: not alive, unoccupied, its bonds dead -- what
    `bm_unbond[release]` does to a released node -- so a cut membrane is a hole the live coupling reads. A
    node that is alive with no mass yet (just laid by `bm_secrete`) starts at `m0`.

    Scheduled FIRST among the membrane's operators, before `bm_bond` (which scales its stiffness by the mass)
    and `bm_clutch` (which reads it as its ligand): a node `bm_secrete` laid last frame gets its mass before any
    of them sees it. Scheduled after `bm_bond`, every new node's bonds ran one frame at half stiffness (mass 0),
    exactly where the membrane is stretching -- mc_B3_mass s1 grew its layer faster to the end (1,297 cells, the
    last quarter 0.40 of the divisions against 0.27-0.29) and missed H6 (exp 11 Finding 146). The contact map is
    therefore the last frame's (none at the first call: every node at z_free). Parameters: `k_M` (0.028 per frame), `s` (1.0),
    `z0` (1.0), `z_free` (1.0), `inhibit` (0.0; 1 is BB-94), `k_coll` (0.0 per frame; `inhibit` and `k_coll` act from the first
    recorded frame, never during the warm-up), `m0` (1.0),
    `m_death` (0.2), `mass_block` (`bm_M`), `mt2_block` (`mt2`), `surface` (vertex), `cell_set` (cell).

    References: Harunaga, J. S. et al. (2014). Dev Biol 394:197 (BB-94 thickens the membrane 3x in 12 h);
    Matsubayashi, Y. et al. (2020). Dev Cell 54:33 (collagen IV turnover); Rebustini, I. T. et al. (2009).
    Dev Cell 17:482 (MT2-MMP is epithelial); Wang, S. et al. (2021). Cell 184:3702 (MMP inhibitors and
    collagenase against budding).
    """
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    PARAM_ROLES = {"k_M": "membrane_turnover_per_frame", "s": "deposition", "z0": "cell_protease",
                   "z_free": "protease_off_the_tissue", "inhibit": "protease_inhibition", "k_coll": "collagenase_per_frame",
                   "m0": "new_node_mass", "m_death": "mass_below_which_a_node_is_gone"}

    def __init__(self, params, device="cpu"):
        Structural.__init__(self, params, device)
        g = lambda k, d: float(params.get(k, d))                     # noqa: E731
        self.at = params.get("_at", "bm_node")
        self.surface = str(params.get("surface", "vertex"))
        self.cat = str(params.get("cell_set", "cell"))
        self.k_M, self.s, self.z0, self.z_free = g("k_M", 0.028), g("s", 1.0), g("z0", 1.0), g("z_free", 1.0)
        self.inhibit, self.k_coll, self.m0, self.m_death = g("inhibit", 0.0), g("k_coll", 0.0), g("m0", 1.0), g("m_death", 0.2)
        self.mass_block = str(params.get("mass_block", "bm_M"))
        self.mt2_block = str(params.get("mt2_block", "mt2"))
        if self.k_M <= 0 or not 0.0 <= self.inhibit <= 1.0:
            raise ValueError("bm_mass: k_M > 0 and inhibit in [0, 1]")
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self.mass_block not in lvl.state_schema:
            raise ValueError(f"bm_mass: the membrane set {self.at!r} must declare the width-1 block {self.mass_block!r}")
        mi = lvl.state_schema[self.mass_block][0]
        st = lvl.state
        alive = getattr(H, "membrane_alive", None)
        alive = (torch.ones(st.shape[0], dtype=torch.bool, device=st.device) if alive is None else alive.to(st.device))
        M = st[:, mi].to(torch.float64)
        newborn = alive & (M <= 0)
        M = torch.where(newborn, torch.full_like(M, self.m0), M)
        # the protease on each node: its cell's, from the clutch's last contact map
        Z = torch.full_like(M, self.z_free)
        tl = H.level(self.surface)
        m = getattr(tl, "_mesh", None) if tl is not None else None
        clvl = H.level(self.cat)
        nF = int(m["nF"]) if m is not None else 0
        z_cell = None
        if clvl is not None and nF:
            # the cell's protease: its `mt2` level (cell_protein_level) where the set declares it, else z0
            z_cell = (clvl.get(self.mt2_block)[:nF, 0].to(torch.float64).clamp_min(0.0)
                      if self.mt2_block in clvl.state_schema else torch.full((nF,), self.z0, dtype=torch.float64,
                                                                             device=st.device))
            z_cell = torch.where(z_cell > 0, z_cell, torch.full_like(z_cell, self.z0))
        cm = m.get("clutch_map") if m is not None else None
        if cm is not None and z_cell is not None:
            node, cell = cm[1].to(st.device), cm[2].to(st.device)
            ok = cell < nF
            Z[node[ok]] = z_cell[cell[ok]]
        # A TREATMENT STARTS AT FRAME 0, as the explant's does: through the unrecorded warm-up (`general.warmup`,
        # the seed settling) the inhibitor and the collagenase are held off, or BB-94 would have tripled the
        # membrane before the first recorded frame (60 ticks x k_M 0.028 = +1.7)
        _w = int(getattr(H, "warmup", 0) or 0)
        treated = not (_w > 0 and int(getattr(H, "frame", 0) or 0) < _w)
        inhibit, k_coll = (self.inhibit, self.k_coll) if treated else (0.0, 0.0)
        Z = Z * (1.0 - inhibit)
        # exact over the frame: dM/dt = s k_M - (k_M Z + k_coll) M
        r = self.k_M * Z + k_coll
        dt = 1.0
        Mstar = torch.where(r > 0, self.k_M * self.s / r.clamp_min(1e-300), torch.zeros_like(r))
        M_new = torch.where(r > 0, Mstar + (M - Mstar) * torch.exp(-r * dt), M + self.k_M * self.s * dt)
        M_new = torch.where(alive, M_new, M)
        dead = alive & (M_new < self.m_death)
        n_dead = int(dead.sum())
        if n_dead:
            alive &= ~dead
            b = getattr(H, "membrane_bonds", None)
            if b is not None:
                bi, bj, _rest, bond_alive = b
                bond_alive &= ~(dead[bi.long()] | dead[bj.long()])
            oc = getattr(lvl, "occ", None)
            if oc is not None:
                oc[dead.to(oc.device)] = 0.0
            M_new = torch.where(dead, torch.zeros_like(M_new), M_new)
        st[:, mi] = M_new.to(st.dtype)
        if m is not None:
            a = alive
            m["bm_mass_mean"] = float(M_new[a].mean()) if bool(a.any()) else 0.0
            m["bm_dead"] = int(m.get("bm_dead", 0)) + n_dead
        if not self._said:
            self._said = True
            print(f"[bm_mass] membrane mass: k_M {self.k_M:g}/frame (half-life {math.log(2) / self.k_M:.1f} frames), "
                  f"z0 {self.z0:g}, inhibit {self.inhibit:g}, k_coll {self.k_coll:g}, m_death {self.m_death:g}; "
                  f"{int(newborn.sum())} nodes started at m0 {self.m0:g}", flush=True)
        return {}


@register_operator("cell_protein_level", family="signalling", set="vertex", kind="structural",
                   title="Per-cell protein levels with memory",
                   equation=r"""$$\dot G^{s}_i=\frac{G^{s*}_{\text{where}(i)}-G^{s}_i}{\tau_s}$$""")
class CellProteinLevel(Structural):
    """Per-cell protein LEVELS that relax toward the level of the compartment the cell is in, with a memory
    (exp 11 Phase 3 B4, 2026-09-29). One operator for every such species -- E-cadherin, MT2-MMP, and the
    tier-2 Btbd7 / Snail2 when they come -- because they are one law with different numbers.

    vertex (the layer's cells) x the interior set -> both: per species s, writes the width-1 block `s` on the
    layer's cell set and on the interior bodies.

        dG_i/dt = (G*_where - G_i) / tau        G*_where = `surface` on the layer, `interior` inside

    integrated exactly per frame. A cell with no level yet (0: seeded, or from a set that did not carry it)
    starts at its compartment's target. The level travels with the cell: a division's row copy, the dive's
    and the return's label carry (both sets declare the block).

    `species`: a list of {block, surface, interior, tau} (tau in frames). The default is E-cadherin as Wang et
    al. 2021 measure it (Fig 2D, 2F-G): {block: ecad, surface: 0.35, interior: 1.0, tau: 72} -- junctional
    E-cadherin ~0.25-0.45 of the interior's at the surface, and surface-derived daughters keep it low while
    temporarily inside, so the level is a cell state with a memory (12 h) longer than the stay, not a function
    of position: a diver inside for 24 frames rises only from 0.35 to about 0.53. MT2-MMP, the epithelial
    protease of the gland (Rebustini 2009), is {block: mt2, surface: 1, interior: 1} until something sets it
    apart.

    A TARGET MAY FOLLOW THE CELL (`input`, `gain`, `cap` on a species; default none): on the layer,

        G*_i = surface (1 + gain (x_i / median_j x_j - 1)),   clipped to [0, cap x surface]

    with x a width-1 block of the cell set (`area`: a stretched cell). It is how a protein is made where the tissue
    does something rather than everywhere: MT2-MMP following the cell's area makes the protease where the layer
    stretches -- and, since a thinner membrane confines less, possibly a feedback that places holes (Harunaga 2014:
    the tip's membrane is perforated, about 7x the equator, and the holes need protease and myosin). The interior
    target stays fixed. `input` is a width-1 block or `chem:<k>` (a reaction-diffusion field's column: a Turing spot
    makes the protease in a coherent PATCH, exp 11 C4), and `input_norm: max` reads it over its own maximum,
    G* = surface (1 + gain x / max x), for a field that is 0 on most cells (the median rule above would divide by
    ~0). Parameters: `species`, `cell_set` (cell), `interior` (icell).

    Reference: Wang, S. et al. (2021). Cell 184:3702-3716 (the E-cadherin gradient and its inheritance).
    """
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    PARAM_ROLES = {"species": "per_species_block_targets_and_memory"}
    DEFAULT_SPECIES = [{"block": "ecad", "surface": 0.35, "interior": 1.0, "tau": 72.0}]

    def __init__(self, params, device="cpu"):
        Structural.__init__(self, params, device)
        self.at = params.get("_at", "vertex")
        self.cat = str(params.get("cell_set", "cell"))
        self.interior = str(params.get("interior", "icell"))
        self.species = [dict(x) for x in (params.get("species") or self.DEFAULT_SPECIES)]
        for sp in self.species:
            sp["surface"], sp["interior"], sp["tau"] = float(sp["surface"]), float(sp["interior"]), float(sp["tau"])
            sp["gain"], sp["cap"] = float(sp.get("gain", 0.0)), float(sp.get("cap", 3.0))
            sp["input"] = sp.get("input")
            sp["input_norm"] = str(sp.get("input_norm", "median")).lower()
            # `input_floor` (default 0): with `input_norm: max`, divide by max(max x, floor) -- a DEAD field (exp 11 C5:
            # Gray-Scott decayed to 0.003) must read as nothing, not as its own noise stretched to [0, 1]
            sp["input_floor"] = float(sp.get("input_floor", 0.0))
            if sp["input_norm"] not in ("median", "max"):
                raise ValueError(f"cell_protein_level: input_norm is median or max, got {sp['input_norm']!r}")
            if sp["tau"] <= 0 or not (0 < sp["surface"] and 0 < sp["interior"]):
                raise ValueError(f"cell_protein_level: species {sp.get('block')!r} needs tau > 0 and positive targets")
        self._said = False

    @staticmethod
    def _relax(lvl, rows, block, target, tau):
        """One frame of dG/dt = (target - G) / tau, exact; `target` a number or a tensor over `rows`."""
        c = lvl.state_schema[block][0]
        G = lvl.state[rows, c].to(torch.float64)
        target = target if torch.is_tensor(target) else torch.full_like(G, float(target))
        G = torch.where(G <= 0, target, G)
        G = target + (G - target) * math.exp(-1.0 / tau)
        lvl.state[rows, c] = G.to(lvl.state.dtype)
        return G

    def forward(self, H, mask=None):
        tl = H.level(self.at)
        m = getattr(tl, "_mesh", None)
        clvl = H.level(self.cat)
        if m is None or clvl is None:
            return {}
        nF = int(m["nF"])
        ilvl = H.level(self.interior) if self.interior in getattr(H, "levels", {}) else None
        live = (ilvl.occ > 0.5).nonzero(as_tuple=True)[0] if ilvl is not None else None
        for sp in self.species:
            b = sp["block"]
            if b not in clvl.state_schema:
                raise ValueError(f"cell_protein_level: the cell set {self.cat!r} must declare the width-1 block {b!r}")
            tgt = sp["surface"]
            if sp["input"] and sp["gain"] and nF:
                # `input: <block>` (a width-1 block) or `input: chem:<k>` (column k of the cell set's chem -- a
                # reaction-diffusion field on the layer, e.g. a Turing spot that makes the protease in a PATCH)
                _blk, _, _col = str(sp["input"]).partition(":")
                if _blk not in clvl.state_schema:
                    raise ValueError(f"cell_protein_level: input {sp['input']!r} is not a block of {self.cat!r}")
                x = clvl.get(_blk)[:nF, int(_col) if _col else 0].to(torch.float64).clamp_min(0.0)
                if sp["input_norm"] == "max":
                    # x over its own maximum: 1 x the level where the field is 0, (1 + gain) x at its peak
                    tgt = (sp["surface"] * (1.0 + sp["gain"] * x / x.max().clamp_min(max(sp["input_floor"], 1e-12))))
                    tgt = tgt.clamp(0.0, sp["cap"] * sp["surface"])
                else:
                    med = x[x > 0].median() if bool((x > 0).any()) else torch.ones((), dtype=torch.float64)
                    tgt = (sp["surface"] * (1.0 + sp["gain"] * (x / med.clamp_min(1e-12) - 1.0))).clamp(0.0, sp["cap"] * sp["surface"])
                m[f"{b}_target_cv"] = float(tgt.std() / tgt.mean().clamp_min(1e-12))
            gs = self._relax(clvl, torch.arange(nF, device=clvl.state.device), b, tgt, sp["tau"]) if nF else None
            gi = None
            if ilvl is not None and b in ilvl.state_schema and live.numel():
                gi = self._relax(ilvl, live, b, sp["interior"], sp["tau"])
            m[f"{b}_surface_mean"] = float(gs.mean()) if gs is not None and gs.numel() else 0.0
            m[f"{b}_interior_mean"] = float(gi.mean()) if gi is not None and gi.numel() else 0.0
        if not self._said:
            self._said = True
            print("[cell_protein_level] " + "; ".join(f"{sp['block']}: {sp['surface']:g} on the layer, {sp['interior']:g} "
                                                    f"inside, memory {sp['tau']:g} frames" for sp in self.species), flush=True)
        return {}
