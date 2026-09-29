"""Membrane pores and channels: proteins as elastic bead networks in a bead membrane, the
electrolyte as a conductor, and the measurements that read a channel's state off them.

THE CHANNEL IS NOT AN OPERATOR THAT SAYS "OPEN". builder/instruction.md section 0b, and
experiments/exp04_membrane_channels.md: a mechanosensitive channel opens because the stretched
membrane pulls on its helices, a potassium channel conducts because ions repel each other through
its filter, a toxin's pore exists because its monomers stick together in a ring. So nothing here
knows what a channel is. There are beads (the alpha carbons of a deposited structure, lipid
patches, ions), the bonds that hold a protein's own fold, the non-bonded laws between beads, a
thermal bath, a frame that stretches the membrane the way a patch pipette does, and an
electrolyte that carries current wherever there is water. Opening, conduction and assembly are to
come out of those; the operators at the end only MEASURE them.

In the order they appear below:

    elastic_network        lateral   a protein's fold: springs between its own beads within a cutoff
                                     (Tirion 1996), at the deposited geometry
    pair_potential         exchange  non-bonded law between two sets, or within one: Lennard-Jones,
                                     its repulsive core (WCA), or screened Coulomb
    brownian               lateral   the thermal bath of an overdamped bead: sqrt(2 mu kT / dt) xi
    tether                 lateral   a harmonic hold to the seeded place, on chosen axes: the
                                     bilayer's confinement of a lipid to its plane, an anchor
    radial_drive           lateral   the patch pipette: a ring of frame beads whose radius follows
                                     an applied tension (or strain) protocol; the tension the
                                     membrane actually carries is READ from the frame's reaction
    electrolyte_conduction exchange  the bath as a conductor: div(sigma grad phi) = 0 on a grid,
                                     protein and lipid beads insulating, the membrane potential
                                     across the box; the channel current is the flux through the
                                     membrane plane -- the conductance is the pore's own geometry
    pore_probe             lateral   a measurement: the radius of the narrowest free lumen along
                                     the channel axis, from a protein set's beads

EVERY FORCE IS OVERDAMPED (`EMIT = "velocity"`): v = mobility x force. A protein bead in water, a
lipid in a bilayer, an ion -- all are at Reynolds numbers of 1e-9 and forget their momentum in
picoseconds, so inertia is never carried. The engine integrates a set ONCE per frame (substep
blocks are the MPM's), so a run here is many short frames. The time of such a run is the
relaxation clock of the bead model -- set by one bead's Stokes drag and the stiffest spring -- and
it is stated in each spec's units, not claimed as the channel's physiological clock.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Exchange, Lateral
from plexus.models.registry import register_operator


def _block(lvl, name):
    """A state block's [n, w] tensor, or None when the set does not declare it."""
    return lvl.get(name) if name and name in lvl.state_schema._slices else None


def _write(lvl, name, value):
    """Write `value` ([n] or [n, w]) into block `name` in place (a readout, not a delta)."""
    a, b = lvl.state_schema[name]
    st = lvl.state.clone()
    v = value if value.dim() == 2 else value[:, None]
    st[:, a:b] = v[:, : b - a].to(st.dtype)
    lvl.state = st


def _frame(H):
    return int(getattr(H, "frame", 0))


def _cell_pairs(Xa, Xb, cut, max_cells=3e7, max_pairs=6e7):
    """Batches of (i, j), one per neighbouring-cell offset, of every pair with |Xb_j - Xa_i| < cut.

    Both sets are binned on one grid of cubic cells at least `cut` wide, so a pair in range sits in the
    same or an adjacent cell, and each A point is compared only with the B points of its 3^D cells:
    O(N x the points per cell) where the all-pairs matrix is O(N^2). The cell grows past `cut` (never
    shrinks below it) when the box would otherwise need more than `max_cells` cells -- a short WCA range
    across a wide bath -- which costs candidates and never loses a pair."""
    import itertools
    if Xa.shape[0] == 0 or Xb.shape[0] == 0:
        return
    D = Xa.shape[1]
    lo = torch.minimum(Xa.min(0).values, Xb.min(0).values)
    ext = (torch.maximum(Xa.max(0).values, Xb.max(0).values) - lo).tolist()
    c = float(cut)
    while math.prod(int(e / c) + 1 for e in ext) > max_cells:
        c *= 1.5
    dims = torch.tensor([int(e / c) + 1 for e in ext], device=Xa.device)
    stride = torch.ones(D, dtype=torch.long, device=Xa.device)
    for a in range(D - 2, -1, -1):
        stride[a] = stride[a + 1] * dims[a + 1]
    ca = torch.minimum(((Xa - lo) / c).floor().long().clamp_min(0), dims - 1)
    cb = torch.minimum(((Xb - lo) / c).floor().long().clamp_min(0), dims - 1)
    kb = (cb * stride).sum(1)
    order = torch.argsort(kb)
    counts = torch.bincount(kb, minlength=int(dims.prod()))
    start = torch.cumsum(counts, 0) - counts
    # ALL 3^D NEIGHBOUR CELLS AT ONCE, in as few batches as `max_pairs` candidates allow: one loop pass per
    # offset was 27 launches and host syncs per call, and on an H100 node the call was host-bound (55 ms
    # against 37 on an A6000 for the same 300,000 ions).
    offs = torch.tensor(list(itertools.product((-1, 0, 1), repeat=D)), device=Xa.device)
    cn = ca[:, None, :] + offs[None]
    ok = ((cn >= 0) & (cn < dims)).all(-1)
    k = (torch.minimum(cn.clamp_min(0), dims - 1) * stride).sum(-1)
    cnt = torch.where(ok, counts[k], torch.zeros_like(k))
    row = torch.cumsum(cnt.sum(1), 0)
    tot_all = int(row[-1])
    if tot_all == 0:
        return
    nb_ = max(1, math.ceil(tot_all / max_pairs))
    cuts = [0] + torch.searchsorted(row, torch.linspace(0, tot_all, nb_ + 1, device=Xa.device)[1:-1].to(row.dtype)).tolist() + [Xa.shape[0]]
    for a0, a1 in zip(cuts[:-1], cuts[1:]):
        if a1 <= a0:
            continue
        c_ = cnt[a0:a1].reshape(-1)
        tot = int(c_.sum())
        if tot == 0:
            continue
        i = torch.repeat_interleave(torch.arange(a0, a1, device=Xa.device).repeat_interleave(offs.shape[0]), c_,
                                    output_size=tot)
        first = torch.cumsum(c_, 0) - c_
        j = order[torch.repeat_interleave(start[k[a0:a1].reshape(-1)] - first, c_, output_size=tot)
                  + torch.arange(tot, device=Xa.device)]
        m = ((Xb[j] - Xa[i]) ** 2).sum(1) < cut * cut
        yield i[m], j[m]


# the frames whose readouts go to the log: a decade ladder, then every 25,000 -- a 400,000-frame run
# was blind between 80,000 and its end (exp04, 2026-09-25)
_REPORT = frozenset((1, 10, 100, 1000, 5000, 10000, 20000, 40000, 80000) + tuple(range(100000, 4000001, 25000)))


# =============================================================================================
# THE PROTEIN'S OWN FOLD
# =============================================================================================
@register_operator("elastic_network", family="mechanics", set="particle", kind="lateral",
                   title="Elastic network of a protein",
                   equation=r"""$$\mathbf F_i=\sum_{j:\,r^0_{ij}<r_c}k\,\big(\lVert\mathbf x_j-\mathbf x_i\rVert-r^0_{ij}\big)\,\hat{\mathbf e}_{ij}\;-\;\sum_{(i,j)\in\mathcal N}\nabla_i\,\epsilon_{go}\Big[\big(\tfrac{r^0_{ij}}{r_{ij}}\big)^{12}-2\big(\tfrac{r^0_{ij}}{r_{ij}}\big)^{6}\Big],\qquad \dot{\mathbf x}_i=\mu\,\mathbf F_i$$""")
class ElasticNetwork(Lateral):
    """A protein's fold as springs between its alpha carbons, and its subunits held together by
    native contacts that can let go.

    particles of one or several sets -> the same: reads pos, emits a velocity on every set named.

        bonds   = { (i, j) in one group : |x0_j - x0_i| < cutoff }       built ONCE, at the first frame
        F_i     = sum_j k (|x_j - x_i| - r0_ij) e_ij                       e_ij the unit vector i -> j
        dx_i/dt = mu F_i

    x0 is the geometry the beads hold at the first frame -- the deposited structure, placed by
    `cloud_seed` -- so the network's rest state IS the structure and every conformational change
    away from it costs elastic energy. `cutoff` (world) is the bond range, 0.8-1.3 nm for an
    alpha-carbon network; `k` the spring constant in sim energy per world length squared; `mu`
    (`mobility`) the bead's mobility, 1 / its Stokes drag, in the spec's units.

    A GROUP IS ONE SUBUNIT. `sets: [A1, ..., A7]` makes one network over several sets, each set one
    chain -- the form the renderer needs, since it draws one skin per set and a pore can only be
    seen to part if its chains are separate skins -- and the springs stay inside a set. With a
    single set, `within: <block>` (an integer chain index per bead) draws the same boundary.
    `k_across` (default 0) adds springs across groups for an assembly meant to stay whole.

    `go_epsilon` HOLDS THE SUBUNITS TO EACH OTHER WITH CONTACTS THAT CAN LET GO. Every pair of
    beads in DIFFERENT groups closer than `go_cutoff` in the deposited structure is a native
    contact with its own deposited distance r0, and gets the 12-6 Go well

        U = eps_go [ (r0/r)^12 - 2 (r0/r)^6 ]           minimum -eps_go at r = r0, zero at infinity

    so the deposited interface IS the bound state and pulling it apart costs eps_go per contact,
    after which it is gone -- which a spring (harmonic to infinity) cannot say. The structure-based
    (Go) model of Clementi, Nymeyer & Onuchic (2000) J. Mol. Biol. 298:937 and Karanicolas & Brooks
    (2002) Protein Sci. 11:2351, for inter-chain contacts; cut at `go_range` x r0 (default 2). It
    knows the CLOSED state only: an open state is never written in, and has to be reached by
    whatever pulls hard enough to break the contacts that hold the closed one.

    `local_max_sep: n` MAKES THE SPRINGS THE BACKBONE ONLY. Harmonic bonds join beads at most n
    apart along a chain (n = 4 makes an alpha helix a rigid rod, i to i+4 being its hydrogen bond),
    and every longer-range pair within `go_cutoff` -- a TERTIARY contact, inside a chain or between
    chains -- is a Go well. This is the alpha-carbon Go model proper: secondary structure stiff,
    packing breakable. Without it every pair within `cutoff` is a spring and each chain is one
    elastic body that can bend but never repack -- and a gating transition IS a repacking (the
    TM2-TM3 contacts of MscS, the TM1 bundle of MscL).

    `go_mode: template` IS FOR IDENTICAL SUBUNITS THAT ARE NOT YET TOGETHER. The contacts are read
    once from a REFERENCE assembly -- `reference: <shape>`, whose points.npz holds the deposited
    chains as `reference_parts` in ring order -- as residue pairs (i on one subunit, j on its
    neighbour, r0), and then applied between EVERY ordered pair of subunits: subunit m's bead i
    and subunit n's bead j attract with the well at r0 whoever m and n are. That is a
    stereospecific face on identical protomers, and nothing else: which partner, how many in a
    ring, whether the ring closes, are left to the geometry. The subunits' own networks are built
    on their seeded shapes as usual.

    STATED LIMITATIONS: the springs are harmonic, so a fold that must change its contacts pays
    elastic energy for what is really a change of register; non-native contacts (a helix packing
    against a new neighbour after it moves) are not attractive here -- excluded volume between
    chains is `pair_potential`'s.

    Reference: Tirion, M.M. (1996). PRL 77:1905; Atilgan, A.R. et al. (2001). Biophys. J. 80:505;
    Valadie, H. et al. (2003). J. Mol. Biol. 332:657 (the lowest modes of MscL are its iris-like
    gating motion); Clementi, C., Nymeyer, H. & Onuchic, J.N. (2000). J. Mol. Biol. 298:937.
    """

    EMIT = "velocity"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["cutoff", "k"]
    MECHANISM_TAGS = ["elastic_network", "protein_fold", "normal_modes", "native_contacts"]
    PARAM_ROLES = {"cutoff": "bond_range_world", "k": "spring_constant_sim_energy_per_world2",
                   "mobility": "bead_mobility", "sets": "the_sets_one_network_spans_one_group_each",
                   "within": "integer_block_bonds_stay_inside_single_set_form",
                   "k_across": "spring_constant_between_groups_0_for_none",
                   "go_epsilon": "native_contact_well_between_groups_sim_energy",
                   "go_cutoff": "native_contact_range_world", "go_range": "contact_cut_in_units_of_r0",
                   "go_mode": "native_or_template", "reference": "shape_holding_the_reference_assembly",
                   "reference_parts": "the_reference_chains_in_ring_order", "reference_scale": "world_per_metre",
                   "local_max_sep": "springs_only_up_to_this_sequence_separation_tertiary_pairs_become_go_wells",
                   "f_max": "cap_on_one_bond_or_contact_force_a_guard",
                   "domain_block": "integer_block_of_domains_inside_which_the_network_is_springs",
                   "go_exclude_domains": "domains_whose_beads_get_no_go_wells",
                   "flexible_domains": "domains_that_are_chains_backbone_springs_and_go_wells_only",
                   "rest_from_reference": "rest_geometry_of_each_set_from_the_reference_not_the_seed",
                   "go_eps_between_chains": "native_contact_well_between_chains_sim_energy",
                   "go_eps_between_chains_keep": "domains_whose_inter_chain_contacts_keep_go_epsilon",
                   "open_reference": "shape_holding_the_open_state", "open_parts": "one_open_array_per_set",
                   "open_scale": "world_per_metre", "basin_offset": "open_basin_energy_above_closed_sim",
                   "basin_coupling": "multiple_basin_mixing_sim", "gate_block": "set_and_block_that_receive_the_open_weight"}
    PARAM_UNITS = {"cutoff": "length", "k": "F/L", "mobility": "mobility", "k_across": "F/L",
                   "go_epsilon": "energy", "go_cutoff": "length", "go_eps_between_chains": "energy"}
    MAY_MUTATE_INTEGRATED_STATE = True             # `gate_block`: writes the open weight, a readout, into the cell
    REFERENCE = ("Tirion, M.M. (1996). PRL 77:1905; Atilgan, A.R. et al. (2001). Biophys. J. 80:505; "
                 "Valadie, H. et al. (2003). J. Mol. Biol. 332:657; Clementi, C., Nymeyer, H. & "
                 "Onuchic, J.N. (2000). J. Mol. Biol. 298:937 (native contacts).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.sets = [str(s) for s in (params.get("sets") or [self.at])]
        self.cutoff = float(params["cutoff"])
        self.k = float(params["k"])
        self.k_across = float(params.get("k_across", 0.0))
        self.mu = float(params.get("mobility", 1.0))
        self.within = params.get("within")
        self.go_eps = float(params.get("go_epsilon", 0.0))
        # `go_eps_between_chains` -- THE INTERFACES BETWEEN SUBUNITS HAVE THEIR OWN WELL. A gating
        # transition that moves subunits apart breaks the contacts BETWEEN chains and keeps each
        # subunit's own packing; the two are different numbers (MscL's iris: each subunit's TM1-TM2
        # pair tilts as a unit, neighbours slide apart). `go_eps_between_chains_keep: [k, ...]`: an
        # inter-chain contact with both beads in these domains keeps `go_epsilon` (MscL's C-terminal
        # bundle, which the iris leaves a bundle). Absent: one well for every contact.
        # `open_reference` + `open_parts` -- A SECOND BASIN: THE OPEN STATE. A single-basin Go model of a
        # gating transition has no open state to go to; with the channel's other deposited (or modelled)
        # state as a second native geometry, the landscape has two minima and tension chooses between
        # them (Okazaki, Koga, Takada & Wolynes 2006, PNAS 103:11844: the multiple-basin model):
        #     V = (V_A + V_B + dV) / 2 - sqrt(((V_A - V_B - dV) / 2)^2 + Delta^2)
        # V_A the springs and Go wells of the closed geometry (the seed), V_B those of the open one
        # (`open_parts`: one array per set in the shape's points.npz, the SAME beads in the same order,
        # NaN where the open model lacks a residue -- a term touching such a bead is kept from A in both,
        # so it cancels), dV (`basin_offset`) the open basin's energy above the closed one at zero
        # tension, Delta (`basin_coupling`) the mixing that sets the barrier between them.
        self.open_ref = params.get("open_reference")
        self.open_parts = list(params.get("open_parts") or [])
        self.open_scale = float(params.get("open_scale", 1.0))
        self.dV = float(params.get("basin_offset", 0.0))
        self.Delta = float(params.get("basin_coupling", 1.0))
        self._B = None
        self.w_closed = float("nan")
        # `gate_block: [set, block]` -- THE GATE'S STATE FOR THE REST OF THE MODEL: the open basin's weight
        # 1 - w_closed written into that one-wide block each frame, where the ion paths read it
        # (`slab_barrier.gate`, `electrolyte_conduction.gate`).
        self.gate_block = params.get("gate_block")
        self.go_eps_inter = params.get("go_eps_between_chains")
        self.go_eps_inter = float(self.go_eps_inter) if self.go_eps_inter is not None else None
        self.go_inter_keep = [int(v) for v in (params.get("go_eps_between_chains_keep") or [])]
        self._eps_g = None
        self.go_cut = float(params.get("go_cutoff", self.cutoff))
        self.go_range = float(params.get("go_range", 2.0))
        # `local_max_sep: n` -- THE SPRINGS ARE THE BACKBONE ONLY: harmonic bonds between beads at most
        # n apart along the chain (n = 4 makes a helix a rigid rod, i to i+4 being its hydrogen bond),
        # and every longer-range pair within `go_cutoff` -- a TERTIARY contact, inside the chain or
        # to another -- is a breakable Go well instead. Without it (None) every pair within `cutoff`
        # is a spring and the whole chain is one elastic body that can bend but never change its
        # packing, which is exactly the change a gating transition is.
        self.local = params.get("local_max_sep")
        self.local = int(self.local) if self.local is not None else None
        # `domain_block: <block>` -- A DOMAIN IS RIGID, ITS INTERFACES ARE NOT. With sets (one per
        # chain) and an integer block naming each bead's domain (paddle, pore helices, cage), every
        # pair within `cutoff` INSIDE one domain of one chain is a spring -- the domain keeps its
        # fold, as a Tirion network holds a protein -- the backbone joins domains by its springs to
        # `local_max_sep`, and every other native pair within `go_cutoff` -- between two domains of
        # a chain, or between chains -- is a breakable Go well. Springs only along the backbone
        # (round 1 of exp04) let a Go model at 1.5 kT melt: native contacts 100 -> 60% in 20,000
        # frames; this keeps what a gating transition leaves intact rigid and makes breakable
        # exactly the interfaces it repacks.
        self.domain_block = params.get("domain_block")
        # `go_exclude_domains: [k, ...]` -- no Go well touches a bead of these domains (by the value of
        # `domain_block`): the domain is held to its neighbours only by the backbone. A test of whether
        # an interface LOCKS a transition (MscL's C-terminal bundle, next to the V21 gate), by removal.
        self.go_exclude = [int(v) for v in (params.get("go_exclude_domains") or [])]
        # `flexible_domains: [k, ...]` -- A DOMAIN THAT IS A CHAIN, NOT A BODY: inside it only the
        # backbone (to `local_max_sep`) is springs and every longer pair is a Go well, so it can be
        # unfolded and fold (alpha-hemolysin's stem, disordered until it zips into the barrel).
        self.flex = [int(v) for v in (params.get("flexible_domains") or [])]
        # `rest_from_reference: true` -- THE NATIVE STATE IS NOT THE SEED. Each set's springs and Go
        # wells take their rest lengths from `reference` (part `reference_parts[0]`, one protomer, the
        # same for every set), while the run starts from whatever the seed placed: a stem seeded as a
        # detached coil still knows the hairpin it belongs in. Pairs are then only inside a set; the
        # contacts BETWEEN sets are a template elastic_network's.
        self.rest_ref = bool(params.get("rest_from_reference", False))
        self.go_mode = str(params.get("go_mode", "native")).lower()
        if self.go_mode not in ("native", "template"):
            raise ValueError(f"elastic_network: go_mode is native or template, got {self.go_mode!r}")
        self.reference = params.get("reference")
        self.reference_parts = list(params.get("reference_parts") or [])
        self.reference_scale = float(params.get("reference_scale", 1.0))
        # `f_max` -- A CAP ON ONE BOND'S OR ONE CONTACT'S FORCE (sim force), a numerical guard and not
        # a law, counted and printed whenever it acts. The Go well's 12-6 wall is stiff far below its
        # minimum -- two beads of a 0.5 nm contact pushed to 0.3 nm by the bath feel a force that, at
        # a time step set by the wall's curvature AT r0, moves them over a nanometre in one step --
        # and exp04's first round blew up at frame ~8,000 (pore probe to 0, skins shredded). The cap
        # bounds one step's displacement to mobility x f_max x dt; zero (the default) is off.
        self.f_max = float(params.get("f_max", 0.0))
        self._bonds = None
        self._go = None
        self.q_native = float("nan")
        self.n_capped = 0

    def _pairs_within(self, X, cut):
        n = X.shape[0]
        I, J = [], []
        for s in range(0, n, 2048):
            ii, jj = torch.nonzero(torch.cdist(X[s:s + 2048], X) < cut, as_tuple=True)
            ii = ii + s
            keep = ii < jj
            I.append(ii[keep]); J.append(jj[keep])
        return torch.cat(I), torch.cat(J)

    def _build(self, X, g, sizes, dom=None):
        ii, jj = self._pairs_within(X, self.cutoff)
        if self.local is not None:                       # backbone springs only (sequence = index order in a set)
            local = (g[ii] == g[jj]) & ((jj - ii).abs() <= self.local)
            if dom is not None:                          # ... plus the whole network inside each domain
                rigid_d = torch.ones_like(dom, dtype=torch.bool)
                for fd in self.flex:                     # ... that is a body, not a chain
                    rigid_d &= dom != fd
                local = local | ((g[ii] == g[jj]) & (dom[ii] == dom[jj]) & rigid_d[ii])
            keep = (g[ii] != g[jj]) | local
            ii, jj = ii[keep], jj[keep]
        same = g[ii] == g[jj]
        kk = torch.where(same, torch.full_like(ii, 1, dtype=X.dtype) * self.k,
                         torch.full_like(ii, 1, dtype=X.dtype) * self.k_across)
        m = kk > 0
        ii, jj, kk = ii[m], jj[m], kk[m]
        self._bonds = (ii, jj, (X[ii] - X[jj]).norm(dim=1), kk)
        n, nb = X.shape[0], int(ii.numel())
        print(f"[elastic_network] {'+'.join(self.sets) if len(self.sets) <= 3 else self.sets[0] + '..' + self.sets[-1]}: "
              f"{nb:,} bonds among {n:,} beads within {self.cutoff:.4f} world ({2.0 * nb / max(n, 1):.1f} per bead), "
              f"inside {int(g.max()) + 1} group(s)", flush=True)
        if self.go_eps <= 0:
            return
        if self.go_mode == "native":
            gi, gj = self._pairs_within(X, self.go_cut)
            keep = g[gi] != g[gj]
            if self.local is not None:                   # tertiary contacts inside a chain are Go wells too
                intra = (g[gi] == g[gj]) & ((gj - gi).abs() > self.local)
                if dom is not None:                      # ... but not inside a domain: that is springs
                    same_d = dom[gi] == dom[gj]
                    for fd in self.flex:                 # ... unless the domain is a chain
                        same_d &= dom[gi] != fd
                    intra = intra & ~same_d
                keep = keep | intra
            if self.go_exclude and dom is not None:
                ex = torch.zeros_like(keep)
                for dval in self.go_exclude:
                    ex |= (dom[gi] == dval) | (dom[gj] == dval)
                keep = keep & ~ex
            gi, gj = gi[keep], gj[keep]
            self._go = (gi, gj, (X[gi] - X[gj]).norm(dim=1))
            self._go_inter = (g[gi] != g[gj])
            if self.go_eps_inter is not None:
                eps = torch.full((gi.numel(),), self.go_eps, device=X.device, dtype=X.dtype)
                sel = self._go_inter.clone()
                if self.go_inter_keep and dom is not None:
                    kk_ = torch.zeros_like(sel)
                    for kd in self.go_inter_keep:
                        kk_ |= (dom[gi] == kd) & (dom[gj] == kd)
                    sel &= ~kk_
                eps[sel] = self.go_eps_inter
                self._eps_g = eps
                print(f"[elastic_network] {int(sel.sum()):,} contacts between chains at {self.go_eps_inter:g}, "
                      f"{int((~sel).sum()):,} others at {self.go_eps:g} (sim energy; "
                      f"{float(eps.sum()):.4g} if all broke)", flush=True)
        else:
            # THE TEMPLATE: residue pairs across the (0, 1) interface of the reference ring, applied
            # between every ordered pair of subunits. Subunits are identical, so bead i of one is
            # bead i of any other; the offsets of the sets in the union say where each begins.
            import os
            import numpy as np
            from plexus import shapes
            path = next((os.path.join(r, str(self.reference), "points.npz") for r in shapes.roots()
                         if os.path.exists(os.path.join(r, str(self.reference), "points.npz"))), None)
            if path is None:
                raise FileNotFoundError(f"elastic_network: no shapes/{self.reference}/points.npz")
            with np.load(path) as z:
                ref = [torch.as_tensor(np.asarray(z[p], np.float64) * self.reference_scale, device=X.device, dtype=X.dtype)
                       for p in self.reference_parts]
            if len(ref) < 2 or ref[0].shape[0] != sizes[0]:
                raise ValueError(f"elastic_network template: reference parts {self.reference_parts} must be the "
                                 f"deposited chains, each {sizes[0]} beads like the subunits")
            # THE ONE INTERFACE, chain 0 against chain 1 (its neighbour, the chains being numbered
            # around the axis): chain 0's right face on chain 1's left. Applied to every ORDERED pair
            # (m, q) it puts m's right face on q's left; the reverse orientation is the pair (q, m),
            # so no contact is counted twice. Chain 0's contacts with its OTHER neighbour are this
            # same interface read from the other side and are not added again.
            d = torch.cdist(ref[0], ref[1])
            a, b = torch.nonzero(d < self.go_cut, as_tuple=True)
            T = torch.stack([a.to(X.dtype), b.to(X.dtype), d[a, b]], 1)
            offs = torch.tensor([0] + list(np.cumsum(sizes)[:-1]), device=X.device)
            GI, GJ, GR = [], [], []
            ns = len(sizes)
            for m in range(ns):
                for q in range(ns):
                    if m == q:
                        continue
                    GI.append(offs[m] + T[:, 0].long()); GJ.append(offs[q] + T[:, 1].long()); GR.append(T[:, 2])
            self._go = (torch.cat(GI), torch.cat(GJ), torch.cat(GR))
            print(f"[elastic_network] template: {int(T.shape[0])} residue pairs on a protomer's faces (from "
                  f"{self.reference}: {', '.join(self.reference_parts)}), applied between all "
                  f"{ns * (ns - 1)} ordered pairs of subunits", flush=True)
        ng = int(self._go[0].numel())
        print(f"[elastic_network] {ng:,} native contacts between groups ({self.go_mode}), Go well {self.go_eps:g} "
              f"each ({ng * self.go_eps:.4g} sim energy if all broke)", flush=True)

    def _energy(self, X, bonds, go, eps_g):
        """The springs' and Go wells' energy of one basin, sim energy (the 12-6 well of the force code,
        eps [(r0/r)^12 - 2 (r0/r)^6], zero beyond go_range r0)."""
        i, j, r0, k = bonds
        r = (X[j] - X[i]).norm(dim=1)
        V = 0.5 * (k * (r - r0) ** 2).sum()
        if go is not None:
            gi, gj, g0 = go
            rg = (X[gj] - X[gi]).norm(dim=1).clamp_min(1e-12)
            q6 = (g0 / rg) ** 6
            u = eps_g * (q6 * q6 - 2.0 * q6)
            V = V + torch.where(rg < self.go_range * g0, u, torch.zeros_like(u)).sum()
        return float(V)

    def _forces(self, X, bonds, go, eps_g):
        """Force and energy of one basin's terms (the open basin's; the closed one's force is the
        forward's own)."""
        i, j, r0, k = bonds
        d = X[j] - X[i]
        r = d.norm(dim=1).clamp_min(1e-12)
        fmag = k * (r - r0)
        nc = 0
        if self.f_max > 0:
            nc += int((fmag.abs() > self.f_max).sum()); fmag = fmag.clamp(-self.f_max, self.f_max)
        f = (fmag / r)[:, None] * d
        F = torch.zeros_like(X).index_add_(0, i, f).index_add_(0, j, -f)
        if go is not None:
            gi, gj, g0 = go
            dg = X[gj] - X[gi]
            rg = dg.norm(dim=1).clamp_min(1e-12)
            live = rg < self.go_range * g0
            q6 = (g0 / rg) ** 6
            mag = torch.where(live, 12.0 * eps_g * (q6 * q6 - q6) / rg, torch.zeros_like(rg))
            if self.f_max > 0:
                nc += int((mag.abs() > self.f_max).sum()); mag = mag.clamp(-self.f_max, self.f_max)
            fg = -(mag / rg)[:, None] * dg
            F = F.index_add_(0, gi, fg).index_add_(0, gj, -fg)
        return F, self._energy(X, bonds, go, eps_g), nc

    def _build_open(self, X, g, sizes, dom):
        """The open basin's terms, by the SAME rules as the closed one's, on the open geometry."""
        import os
        import numpy as np
        from plexus import shapes
        path = next((os.path.join(r, str(self.open_ref), "points.npz") for r in shapes.roots()
                     if os.path.exists(os.path.join(r, str(self.open_ref), "points.npz"))), None)
        if path is None:
            raise FileNotFoundError(f"elastic_network: no shapes/{self.open_ref}/points.npz")
        with np.load(path) as z:
            XO = np.concatenate([np.asarray(z[p_], np.float64) * self.open_scale for p_ in self.open_parts])
        if XO.shape[0] != X.shape[0]:
            raise ValueError(f"elastic_network open basin: {XO.shape[0]} open beads for {X.shape[0]} beads")
        miss = ~np.isfinite(XO).all(1)
        XO = np.where(miss[:, None], X.detach().cpu().numpy(), XO)          # missing: the closed geometry
        XO = torch.as_tensor(XO, device=X.device, dtype=X.dtype)
        A_bonds, A_go, A_inter, A_eps = self._bonds, self._go, getattr(self, "_go_inter", None), self._eps_g
        self._build(XO, g, sizes, dom)                                     # the open terms, same rules
        B_bonds, B_go, B_eps = self._bonds, self._go, self._eps_g
        self._bonds, self._go, self._go_inter, self._eps_g = A_bonds, A_go, A_inter, A_eps
        # a term touching a bead the open model lacks: A's, in both basins, so it cancels in V_A - V_B
        m_ = torch.as_tensor(miss, device=X.device)
        ia, ja = A_bonds[0], A_bonds[1]
        keepA = m_[ia] | m_[ja]
        ib, jb = B_bonds[0], B_bonds[1]
        keepB = ~(m_[ib] | m_[jb])
        B_bonds = tuple(torch.cat([b_[keepB], a_[keepA]]) for b_, a_ in zip(B_bonds, A_bonds))
        if B_go is not None and A_go is not None:
            ga, gb = A_go, B_go
            kA = m_[ga[0]] | m_[ga[1]]; kB = ~(m_[gb[0]] | m_[gb[1]])
            B_go = tuple(torch.cat([b_[kB], a_[kA]]) for b_, a_ in zip(gb, ga))
            ea = A_eps if A_eps is not None else torch.full((ga[0].numel(),), self.go_eps, device=X.device, dtype=X.dtype)
            eb = B_eps if B_eps is not None else torch.full((gb[0].numel(),), self.go_eps, device=X.device, dtype=X.dtype)
            B_eps = torch.cat([eb[kB], ea[kA]])
        self._B = (B_bonds, B_go, B_eps if B_eps is not None else self.go_eps)
        print(f"[elastic_network] second basin from {self.open_ref} ({', '.join(self.open_parts)}): "
              f"{int(B_bonds[0].numel()):,} springs, {int(B_go[0].numel()) if B_go is not None else 0:,} Go wells "
              f"({int(miss.sum())} beads missing from the open model), offset {self.dV:.4g}, coupling {self.Delta:.4g}",
              flush=True)

    def forward(self, H, mask=None):
        lvls = [H.level(s) for s in self.sets]
        X = torch.cat([lv.get("pos") for lv in lvls])
        occ = torch.cat([lv.occ for lv in lvls])
        sizes = [lv.get("pos").shape[0] for lv in lvls]
        if self._bonds is None:
            if len(self.sets) > 1:
                g = torch.cat([torch.full((s,), k, device=X.device, dtype=torch.long) for k, s in enumerate(sizes)])
            else:
                b = _block(lvls[0], self.within)
                g = b[:, 0].round().long() if b is not None else torch.zeros(sizes[0], device=X.device, dtype=torch.long)
            dom = None
            if self.domain_block:
                bs = [_block(lv, self.domain_block) for lv in lvls]
                if all(x is not None for x in bs):
                    dom = torch.cat([x[:, 0] for x in bs]).round().long()
            Xb = X.detach()
            if self.rest_ref:
                import os
                import numpy as np
                from plexus import shapes
                path = next((os.path.join(r, str(self.reference), "points.npz") for r in shapes.roots()
                             if os.path.exists(os.path.join(r, str(self.reference), "points.npz"))), None)
                if path is None:
                    raise FileNotFoundError(f"elastic_network: no shapes/{self.reference}/points.npz")
                with np.load(path) as z:
                    R0 = torch.as_tensor(np.asarray(z[self.reference_parts[0]], np.float64) * self.reference_scale,
                                         device=X.device, dtype=X.dtype)
                if any(n != R0.shape[0] for n in sizes):
                    raise ValueError(f"elastic_network rest_from_reference: every set must have the reference's "
                                     f"{R0.shape[0]} beads, got {sizes}")
                # each set its own copy of the reference, placed far apart so no pair spans two sets
                far = 10.0 * float((R0.max(0).values - R0.min(0).values).max() + self.go_cut + self.cutoff)
                Xb = torch.cat([R0 + torch.tensor([far * k, 0.0, 0.0], device=X.device, dtype=X.dtype)
                                for k in range(len(sizes))])
                print(f"[elastic_network] rest geometry from {self.reference}/{self.reference_parts[0]}, "
                      f"not the seed (the seed's own springs would hold it where it was placed)", flush=True)
            self._build(Xb, g, sizes, dom)
            if self.open_ref and self.open_parts:
                self._build_open(X, g, sizes, dom)
        i, j, r0, k = self._bonds
        d = X[j] - X[i]
        r = d.norm(dim=1).clamp_min(1e-12)
        fmag = k * (r - r0)                                      # signed: + stretched (pull), - compressed
        n_cap = 0
        if self.f_max > 0:
            n_cap += int((fmag.abs() > self.f_max).sum())
            fmag = fmag.clamp(-self.f_max, self.f_max)
        f = (fmag / r)[:, None] * d                              # force on i, toward j when stretched
        F = torch.zeros_like(X).index_add_(0, i, f).index_add_(0, j, -f)
        fr = _frame(H)
        if self._go is not None:
            gi, gj, g0 = self._go
            dg = X[gj] - X[gi]
            rg = dg.norm(dim=1).clamp_min(1e-12)
            live = rg < self.go_range * g0
            q6 = (g0 / rg) ** 6
            # -dU/dr = 12 eps (q12 - q6) / r : positive = repulsive; the force on gi is along -dg
            eps_g = self._eps_g if self._eps_g is not None else self.go_eps
            mag = torch.where(live, 12.0 * eps_g * (q6 * q6 - q6) / rg, torch.zeros_like(rg))
            if self.f_max > 0:
                n_cap += int((mag.abs() > self.f_max).sum())
                mag = mag.clamp(-self.f_max, self.f_max)
            fg = -(mag / rg)[:, None] * dg
            F = F.index_add_(0, gi, fg).index_add_(0, gj, -fg)
            made = (rg < 1.2 * g0).to(X.dtype)                          # a contact is MADE within 20% of r0
            self.q_native = float(made.mean())
            gin = getattr(self, "_go_inter", None)
            if gin is not None and bool(gin.any()) and bool((~gin).any()):
                self.q_inter = float(made[gin].mean()); self.q_intra = float(made[~gin].mean())
        if self._B is not None:
            # THE SECOND BASIN, and the mixing: V_A from this frame's closed terms, V_B from the open ones
            V_A = self._energy(X, self._bonds, self._go, self._eps_g if self._eps_g is not None else self.go_eps)
            F_B, V_B, nb = self._forces(X, *self._B)
            n_cap += nb
            D = 0.5 * (V_A - V_B - self.dV)
            S_ = math.sqrt(D * D + self.Delta * self.Delta)
            wA = 0.5 * (1.0 - D / S_)
            self.w_closed = wA
            F = wA * F + (1.0 - wA) * F_B
            if self.gate_block:
                lv_ = H.level(str(self.gate_block[0]))
                _write(lv_, str(self.gate_block[1]), torch.full((lv_.get("pos").shape[0], 1), 1.0 - wA,
                                                               dtype=X.dtype, device=X.device))
        self.n_capped += n_cap
        if fr in _REPORT and self._B is not None:
            print(f"[elastic_network f{fr}] two basins: weight of the closed one {self.w_closed:.3f} "
                  f"(V_closed - V_open - dV = {2 * D:.4g} sim energy, coupling {self.Delta:.4g})", flush=True)
        if fr in _REPORT:
            st = ((r - r0) / r0)
            if self.f_max > 0:
                print(f"[elastic_network f{fr}] force cap f_max acted {self.n_capped} times so far "
                      f"({n_cap} this frame)", flush=True)
            if not bool(torch.isfinite(X).all()):
                print(f"[elastic_network f{fr}] NON-FINITE POSITIONS: {int((~torch.isfinite(X)).sum())} coordinates", flush=True)
            print(f"[elastic_network f{fr}] bond strain rms {float(st.pow(2).mean().sqrt()) if st.numel() else 0.0:.4f}, "
                  f"max {float(st.abs().max()) if st.numel() else 0.0:.4f}"
                  + (f"; native contacts made {self.q_native * 100:.1f}%" if self._go is not None else "")
                  + (f" (between chains {self.q_inter * 100:.1f}%, inside chains {self.q_intra * 100:.1f}%)"
                     if hasattr(self, "q_inter") else ""), flush=True)
        v = self.mu * F * occ[:, None]
        out, s0 = {}, 0
        for name, n in zip(self.sets, sizes):
            out[name] = v[s0:s0 + n]
            s0 += n
        return out


# =============================================================================================
# NON-BONDED LAWS BETWEEN BEADS
# =============================================================================================
@register_operator("pair_potential", family="interaction", set="particle", kind="exchange",
                   title="Pair potential between two sets",
                   equation=r"""$$U_{\mathrm{LJ}}=4\epsilon\Big[\big(\tfrac{\sigma}{r}\big)^{12}-\big(\tfrac{\sigma}{r}\big)^{6}\Big],\qquad U_{\mathrm C}=\frac{k_C\,q_i q_j\,e^{-r/\lambda_D}}{r}$$""")
class PairPotential(Exchange):
    """A non-bonded pair law between the beads of two groups of sets (or within one), both ways.

    A -> A and B -> B: reads both sides' pos; emits a velocity on every set of A and, with
    `react`, the equal and opposite on every set of B (mobility-weighted: v = mu F on each side).

        law: lj        U = 4 eps [(s/r)^12 - (s/r)^6],  cut at `cutoff` (2.5 s by default)
        law: wca       the same, cut at its minimum 2^(1/6) s: repulsion only (excluded volume)
        law: coulomb   U = k_C q_i q_j exp(-r / debye) / r    (screened, Debye-Hueckel)
        law: harmonic  U = eps/2 (1 - r/s)^2 for r < s, 0 beyond: SOFT spheres of diameter s that
                       may overlap (O'Hern et al. 2003) -- the centre-based cell's excluded
                       volume, where a daughter born half a diameter from its mother must be
                       pushed apart smoothly (WCA's r^-12 there is a force the guards would cap).
                       Stiffness eps/s^2 per contact; a velocity mu F, so an overdamped set is
                       stable while mu (eps/s^2) (contacts per bead) dt stays below ~2.

    `law` is one of them or a LIST, summed pair by pair (`[coulomb, wca]`: charged ions with hard
    cores). Side A is `sets: [...]` (default the `at` set alone); side B is `with: <set>` or
    `with_sets: [...]` (default: the same as A, pairs i < j, no self). `epsilon` (sim energy) as
    written; `sigma` (world) a number, or `{set: sigma_i}` with sigma_ij = (sigma_i + sigma_j) / 2;
    `cutoff` a number or `{law: range}` (WCA is always cut at its minimum). For Coulomb `k_c` is kT
    times the Bjerrum length in sim energy x world (the generator derives it from the permittivity),
    `charge` and `charge_with` the beads' charges in e -- a number, the name of a block, or
    `{set: ...}` -- and `debye` the screening length (world; 0 = unscreened). `mobility` and
    `mobility_with` a number or `{set: mu}`. Within one side, `exclude_same_set: true` skips pairs
    in the same set (two beads of one chain, whose fold `elastic_network` already holds) and
    `exclude: <block>` skips pairs whose integer block agrees. `step_max` (world, a number or
    `{law: step}`) caps the step one pair's force makes a bead take in a frame, |F| <= step/(mu dt);
    `f_max` caps the force itself (sim force) -- guards against seeding overlaps, printed when they
    act, never a law.

    WHY BOTH WAYS. A lipid pulling on a helix is a helix pulling on a lipid; with `react` (the
    default when B is another set) the reaction is returned on B, so the membrane's tension
    reaches the protein and the protein's resistance reaches the membrane. When B is one set that
    declares a 3-wide block `force`, the total force on each B bead is written there too -- that is
    how the patch frame reads the tension the membrane carries (`radial_drive`).

    Reference: Lennard-Jones, J.E. (1924). Proc. R. Soc. A 106:463; Weeks, J.D., Chandler, D. &
    Andersen, H.C. (1971). J. Chem. Phys. 54:5237 (the repulsive core); Debye, P. & Hueckel, E.
    (1923). Phys. Z. 24:185; O'Hern, C.S., Silbert, L.E., Liu, A.J. & Nagel, S.R. (2003). Phys. Rev.
    E 68:011306 (the harmonic soft sphere).
    """

    EMIT = "velocity"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    INPUTS = ["particle", "particle"]
    OUTPUTS = ["particle", "particle"]
    READS = ["pos"]
    REQUIRES_PARAMS = ["law"]
    MECHANISM_TAGS = ["lennard_jones", "excluded_volume", "adhesion", "electrostatics", "screened_coulomb",
                      "soft_sphere"]
    PARAM_ROLES = {"sets": "side_A_default_the_at_set", "with": "side_B_one_set", "with_sets": "side_B_several_sets",
                   "law": "lj_wca_coulomb_cooke_or_harmonic_or_a_list_summed", "epsilon": "well_depth_sim_energy",
                   "sigma": "contact_distance_world_or_per_set_map_lorentz_mixed",
                   "tail": "cooke_cosine_tail_width_world",
                   "cutoff": "range_world", "k_c": "kT_times_bjerrum_length",
                   "charge": "charge_of_A_in_e_or_a_block", "charge_with": "charge_of_B_in_e_or_a_block",
                   "debye": "screening_length_world", "exclude": "integer_block_whose_equal_pairs_are_skipped",
                   "exclude_same_set": "skip_pairs_inside_one_set", "mobility": "mobility_of_A",
                   "mobility_with": "mobility_of_B", "react": "return_the_reaction_on_B",
                   "f_max": "cap_on_one_pair_force_a_guard",
                   "step_max": "cap_on_the_step_one_pair_force_makes_per_frame_a_guard", "slab": "z_window_and_eps_ratio_of_a_low_permittivity_region"}
    PARAM_UNITS = {"epsilon": "energy", "sigma": "length", "cutoff": "length", "debye": "length",
                   "mobility": "mobility", "mobility_with": "mobility", "f_max": "force", "step_max": "length"}
    MAY_MUTATE_INTEGRATED_STATE = True             # writes B's `force` readout block in place
    REFERENCE = ("Lennard-Jones, J.E. (1924). Proc. R. Soc. A 106:463; Weeks, Chandler & Andersen "
                 "(1971). J. Chem. Phys. 54:5237; Debye & Hueckel (1923). Phys. Z. 24:185.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.A = [str(s) for s in (params.get("sets") or [self.at])]
        if params.get("with_sets"):
            self.B = [str(s) for s in params["with_sets"]]
        elif params.get("with"):
            self.B = [str(params["with"])]
        else:
            self.B = list(self.A)
        self.same = self.B == self.A
        # `law`: ONE LAW OR A LIST, summed pair by pair. exp04's hole spec took EIGHT instances for its
        # ions -- Coulomb and WCA for each of K-K, Cl-Cl, K-Cl, and WCA for each ion with the lipids --
        # because a law, a charge, a size and a mobility were one scalar per instance. With a list of
        # laws and per-set values they are two instances (ions with ions, ions with the lipids) giving
        # the same forces: tests/test_channel_ops.py::test_merged_pair_potential_equals_the_split_one.
        _law = params["law"]
        self.laws = [str(l).lower() for l in (_law if isinstance(_law, (list, tuple)) else [_law])]
        for l in self.laws:
            if l not in ("lj", "wca", "coulomb", "cooke", "harmonic"):
                raise ValueError(f"pair_potential: law is lj, wca, coulomb, cooke or harmonic (or a list of them), "
                                 f"got {l!r}")
        if len(set(self.laws)) != len(self.laws):
            raise ValueError(f"pair_potential: a law listed twice in {self.laws}")
        self.law = "+".join(self.laws)
        self.eps = float(params.get("epsilon", 1.0))
        # `sigma`: a number, or `{set: sigma_i}` -- then a pair's sigma_ij = (sigma_i + sigma_j) / 2
        # (Lorentz's rule; for ions sized by 2^(1/6) sigma_i = 2 r_i it is the contact r_i + r_j).
        _sg = params.get("sigma", 0.01)
        self.sigma_of = {str(k): float(v) for k, v in _sg.items()} if isinstance(_sg, dict) else None
        self.sigma = max(self.sigma_of.values()) if self.sigma_of else float(_sg)
        # `law: cooke` -- THE COSINE TAIL (Cooke, Kremer & Deserno 2005, PRE 72:011506): the WCA core,
        # a flat well -eps out to r_c = 2^(1/6) sigma, and a smooth cos^2 tail of width `tail` (world)
        # beyond it. A BROAD attraction keeps coarse-grained lipids a cohesive FLUID at eps ~ kT --
        # the Lennard-Jones layer of exp04 round 1, a liquid near its critical point, yielded under
        # tension (10 mN/m applied, 1.5 carried) -- which is the property the model was built for.
        self.tail_given = float(params["tail"]) if "tail" in params else None
        self.tail = self.tail_given if self.tail_given is not None else 1.6 * self.sigma
        self.r_c = 2.0 ** (1.0 / 6.0) * self.sigma
        # `cutoff`: a number for every law, or `{law: range}`; a law not given takes its own default
        # (lj 2.5 sigma, coulomb 0.25, cooke r_c + tail), and WCA never reaches past its minimum
        # 2^(1/6) sigma. The neighbour search runs to the largest of them, `self.cutoff`.
        _cut = params.get("cutoff")
        self.cut_given = {l: (float(_cut[l]) if isinstance(_cut, dict) and l in _cut else
                              None if isinstance(_cut, dict) or _cut is None else float(_cut)) for l in self.laws}
        reach = {"lj": 2.5 * self.sigma, "wca": 2.0 ** (1.0 / 6.0) * self.sigma, "coulomb": 0.25,
                 "cooke": self.r_c + self.tail, "harmonic": self.sigma}
        self.cutoff = max((self.cut_given[l] if self.cut_given[l] is not None else reach[l])
                          if l not in ("wca", "harmonic")
                          else min(self.cut_given[l] if self.cut_given[l] is not None else reach[l], reach[l])
                          for l in self.laws)
        self.k_c = float(params.get("k_c", 0.0))
        # `charge` / `charge_with`: the beads' charges in e -- a number, the name of a block, or
        # `{set: number | block}`. `charge_with` defaults to `charge` (a map is looked up on B's sets).
        self.q_a = params.get("charge", 1.0)
        self.q_b = params.get("charge_with", self.q_a)
        self.debye = float(params.get("debye", 0.0))
        # `slab: {z: [z0, z1], eps_ratio: r}` -- A LOW-PERMITTIVITY REGION. When BOTH charges of a pair
        # are inside the membrane's slab (a channel's filter and cavity: protein and lipid around, little
        # water), their interaction is unscreened and multiplied by r = eps_water / eps_slab; outside it
        # is the bath's (k_c at eps_water, Debye-screened). The one physical lever on how hard ions in a
        # filter push on each other (knock-on) against how hard its carbonyls hold them -- the balance
        # Berneche & Roux (2001) computed, and a continuum dielectric boundary would set.
        _sl = params.get("slab")
        self.slab_z = [float(v) for v in _sl["z"]] if _sl else None
        self.slab_r = float(_sl.get("eps_ratio", 1.0)) if _sl else 1.0
        # `slab.edge` (world) -- THE REGION'S BOUNDARY SMOOTH, AND THE FORCE CONSERVATIVE. With edge 0 the region is a
        # switch: a pair's Coulomb jumps by eps_ratio when both charges are in, with no force for the jump, so an ion
        # carried through the region and out gains or loses the jump for free -- a pump. Measured along Kv1.2's
        # filter at ratio 8: the charged residues' term ended 14 kT below where it began once the ion was out
        # (tools/channel_axis_energy.py, 2026-09-26), against the 8 kT a 200 mV clamp supplies. With edge > 0 the
        # pair energy is U = k_c qa qb f / r, f = 1 + (eps_ratio - 1) w(za) w(zb), w a cosine window, and the force
        # is its whole gradient, the z-derivative of f included (unscreened: `debye` must be 0).
        # `slab.born: {set: kT}` -- THE ION'S OWN COST OF BEING THERE: +born w(z) on each bead of that set on side A,
        # the Born energy 56 / (2 a) (1 / eps_region - 1 / eps_water) kT of an ion of radius a (nm) moved from water
        # into the region. The carbonyls' stronger pull at the region's permittivity is what pays it back; one
        # without the other is a trap (the pull alone) or a wall (the cost alone).
        self.slab_e = float(_sl.get("edge", 0.0)) if _sl else 0.0
        self.slab_born = {str(k): float(v) for k, v in (_sl.get("born") or {}).items()} if _sl else {}
        if self.slab_e > 0 and self.debye > 0:
            raise ValueError("pair_potential: a smooth `slab` (edge > 0) is unscreened; set `debye: 0`")
        self.exclude = params.get("exclude")
        self.excl_set = bool(params.get("exclude_same_set", False))
        # `mobility` / `mobility_with`: a number, or `{set: mu}` (v = mu F on each bead of that set).
        _mu = params.get("mobility", 1.0)
        self.mu_a = {str(k): float(v) for k, v in _mu.items()} if isinstance(_mu, dict) else float(_mu)
        _mw = params.get("mobility_with", _mu)
        self.mu_b = {str(k): float(v) for k, v in _mw.items()} if isinstance(_mw, dict) else float(_mw)
        self.react = bool(params.get("react", not self.same))
        # TWO GUARDS AGAINST SEEDING OVERLAPS, printed when they act, never a law. `f_max` caps one
        # pair's force (sim force) whatever the law. `step_max` caps the STEP that force makes the
        # A bead take in one frame (world; a number, or `{law: step}`): |F| <= step / (mu_i dt). The
        # step is what the guard is about, and it holds per set with one number -- the f_max it
        # replaces had to be written once per set, each divided by that set's mobility and dt.
        self.f_max = float(params.get("f_max", 0.0))
        _st = params.get("step_max")
        self.step_max = ({l: (float(_st[l]) if isinstance(_st, dict) and l in _st else
                              None if isinstance(_st, dict) or _st is None else float(_st)) for l in self.laws})

    @staticmethod
    def _per_bead(lvls, names, v, what, X):
        """One value per bead of a side: `v` a number, a block's name, or `{set: number | block}`."""
        parts = []
        for nm, lv in zip(names, lvls):
            u = v.get(nm) if isinstance(v, dict) else v
            if u is None:
                raise ValueError(f"pair_potential: no {what} for set {nm!r} in {v}")
            if isinstance(u, str):
                b = _block(lv, u)
                if b is None:
                    raise ValueError(f"pair_potential: {what} block {u!r} not declared on {nm!r}")
                parts.append(b[:, 0].to(X.dtype))
            else:
                parts.append(torch.full((lv.get("pos").shape[0],), float(u), device=X.device, dtype=X.dtype))
        return torch.cat(parts)

    def _side(self, H, names, q, mu, moved=True):
        lvls = [H.level(n) for n in names]
        X = torch.cat([lv.get("pos") for lv in lvls])
        occ = torch.cat([lv.occ for lv in lvls]) > 0
        sizes = [lv.get("pos").shape[0] for lv in lvls]
        grp = torch.cat([torch.full((s,), k, device=X.device, dtype=torch.long) for k, s in enumerate(sizes)])
        blk = None
        if self.exclude:
            bs = [_block(lv, self.exclude) for lv in lvls]
            if all(b is not None for b in bs):
                blk = torch.cat([b[:, 0] for b in bs]).round().long()
        qq = self._per_bead(lvls, names, q, "charge", X) if "coulomb" in self.laws else None
        mm = self._per_bead(lvls, names, mu, "mobility", X) if moved and isinstance(mu, dict) else None
        sg = self._per_bead(lvls, names, self.sigma_of, "sigma", X) if self.sigma_of else None
        return lvls, X, occ, sizes, grp, blk, qq, mm, sg

    def _force(self, d, r, qa=None, qb=None, inslab=None, sig=None, mu_i=None, h=1.0):
        """Force on the A bead of each pair (vector; d = x_b - x_a points A -> B), summed over the
        laws, and the number of pair-law terms a guard capped. `sig` is the pair's sigma (a number
        or one per pair), `mu_i` the A bead's mobility (for `step_max`)."""
        s = self.sigma if sig is None else sig
        per_pair = torch.is_tensor(s)
        mag, n_capped = None, 0
        ez = None
        for l in self.laws:
            if l in ("lj", "wca"):
                sr6 = (s / r) ** 6
                m = 24.0 * self.eps * (2.0 * sr6 * sr6 - sr6) / r        # -dU/dr: positive = repulsion
            elif l == "cooke":
                r_c = 2.0 ** (1.0 / 6.0) * s if per_pair else self.r_c
                tl = self.tail if (self.tail_given is not None or not per_pair) else 1.6 * s
                sr6 = (s / r) ** 6
                core = torch.where(r < r_c, 24.0 * self.eps * (2.0 * sr6 * sr6 - sr6) / r, torch.zeros_like(r))
                x = (math.pi / (2.0 * tl)) * (r - r_c)
                m = core + torch.where((r >= r_c) & (r <= r_c + tl),
                                       -self.eps * (math.pi / (2.0 * tl)) * torch.sin(2.0 * x), torch.zeros_like(r))
            elif l == "harmonic":                                       # -dU/dr = eps (1 - r/s) / s, r < s
                m = self.eps * (1.0 - r / s) / s
            else:
                scr = torch.exp(-r / self.debye) if self.debye > 0 else torch.ones_like(r)
                m = self.k_c * qa * qb * scr * (1.0 / (r * r) + ((1.0 / (self.debye * r)) if self.debye > 0 else 0.0))
                if isinstance(inslab, tuple):                              # the smooth region: f and its z-gradient
                    wa_, wb_, dwa_, dwb_ = inslab
                    g_ = self.k_c * qa * qb * (self.slab_r - 1.0) / r
                    m = self.k_c * qa * qb * (1.0 + (self.slab_r - 1.0) * wa_ * wb_) / (r * r)
                    ez = (-g_ * dwa_ * wb_, -g_ * wa_ * dwb_)
                elif inslab is not None:                                   # inside the slab: unscreened, eps lowered
                    m = torch.where(inslab, self.k_c * self.slab_r * qa * qb / (r * r), m)
            # THIS LAW'S OWN RANGE, where it is shorter than the search's: WCA's minimum per pair,
            # or a law listed beside a longer-ranged one. One law alone is searched at its own range.
            if l in ("wca", "harmonic"):
                cut = 2.0 ** (1.0 / 6.0) * s if l == "wca" else s
                if self.cut_given[l] is not None:
                    cut = torch.clamp(cut, max=self.cut_given[l]) if per_pair else min(cut, self.cut_given[l])
            elif l == "cooke" and self.cut_given[l] is None:
                cut = (2.0 ** (1.0 / 6.0) * s + (self.tail if self.tail_given is not None else 1.6 * s)) \
                    if per_pair else self.r_c + self.tail
            elif l == "lj" and self.cut_given[l] is None:
                cut = 2.5 * s
            else:
                cut = self.cut_given[l] if self.cut_given[l] is not None else 0.25
            if per_pair or cut < self.cutoff:
                m = torch.where(r < cut, m, torch.zeros_like(m))
            if self.f_max > 0:
                m = m.clamp(-self.f_max, self.f_max)
                n_capped += int((m.abs() >= 0.999 * self.f_max).sum())
            if self.step_max[l] is not None:
                cap = self.step_max[l] / (mu_i * h)                 # a number, or one per pair
                n_capped += int((m.abs() >= 0.999 * cap).sum())
                m = torch.maximum(torch.minimum(m, cap), -cap) if torch.is_tensor(cap) else m.clamp(-cap, cap)
            mag = m if mag is None else mag + m
        f = -(mag / r)[:, None] * d
        if ez is None:
            return f, n_capped, None
        if self.step_max.get("coulomb") is not None:                  # the z-term under the Coulomb guard as well
            cap = self.step_max["coulomb"] / (mu_i * h)
            ez = tuple(torch.maximum(torch.minimum(e_, cap), -cap) if torch.is_tensor(cap) else e_.clamp(-cap, cap)
                       for e_ in ez)
        return f, n_capped, ez

    # THE NEIGHBOUR SEARCH CHANGES METHOD ABOVE THIS MANY CANDIDATE PAIRS (na x nb), not the pairs it
    # finds. All-pairs in chunks of 4096 is exact and fast up to ~1e8 -- every exp04 rig -- and at a bath
    # holding 100 channels (3e5 ions, 1e11 pairs) it is a day per frame; the cell grid finds the same pairs
    # in O(N). The threshold sits far above any single-channel rig, so those run the path they always ran.
    CELL_LIST_ABOVE = 2e9

    def _pairs(self, Xa, Xb, oa, ob, ga, gb, ba, bb):
        """Batches of (i, j) -- i into A, j into B -- of every live pair nearer than `cutoff`, each unordered
        pair once when A is B, less the pairs `exclude` / `exclude_same_set` remove."""
        na, nb = Xa.shape[0], Xb.shape[0]
        if float(na) * float(nb) <= self.CELL_LIST_ABOVE:
            step = 4096
            colb = torch.arange(nb, device=Xa.device)
            for s in range(0, na, step):
                xa = Xa[s:s + step]
                near = (torch.cdist(xa, Xb) < self.cutoff) & oa[s:s + step, None] & ob[None, :]
                if self.same:
                    near &= torch.arange(s, s + xa.shape[0], device=Xa.device)[:, None] < colb[None, :]
                    if self.excl_set:
                        near &= ga[s:s + step, None] != gb[None, :]
                if ba is not None and bb is not None:
                    near &= ba[s:s + step, None] != bb[None, :]
                ii, jj = torch.nonzero(near, as_tuple=True)
                if ii.numel():
                    yield ii + s, jj
            return
        la, lb = torch.nonzero(oa).squeeze(1), torch.nonzero(ob).squeeze(1)
        for i, j in _cell_pairs(Xa[la], Xb[lb], self.cutoff):
            i, j = la[i], lb[j]
            keep = torch.ones_like(i, dtype=torch.bool)
            if self.same:
                keep &= i < j
                if self.excl_set:
                    keep &= ga[i] != gb[j]
            if ba is not None and bb is not None:
                keep &= ba[i] != bb[j]
            yield i[keep], j[keep]

    def forward(self, H, mask=None):
        LA, Xa, oa, sa, ga, ba, qa, ma, sga = self._side(H, self.A, self.q_a, self.mu_a)
        if self.same:
            LB, Xb, ob, sb, gb, bb, qb, mb, sgb = LA, Xa, oa, sa, ga, ba, qa, ma, sga
        else:
            LB, Xb, ob, sb, gb, bb, qb, mb, sgb = self._side(H, self.B, self.q_b, self.mu_b, moved=self.react)
        h = float(getattr(H, "dt", 1.0))
        na, nb = Xa.shape[0], Xb.shape[0]
        Fa = torch.zeros_like(Xa)
        Fb = torch.zeros_like(Xb)
        n_pairs, n_capped = 0, 0
        if self.slab_z is not None and self.slab_e > 0:
            zm_, hz_ = 0.5 * (self.slab_z[0] + self.slab_z[1]), 0.5 * (self.slab_z[1] - self.slab_z[0])

            def _w(X_):
                u_ = X_[:, 2] - zm_
                w_, dw_ = SlabBarrier._step(u_.abs(), hz_, self.slab_e)
                return w_, dw_ * torch.sign(u_)
            wA, dwA = _w(Xa)
            wB, dwB = (wA, dwA) if self.same else _w(Xb)
            if self.slab_born:                                          # the ion's own cost of being in the region
                s0 = 0
                for nm_, n_ in zip(self.A, sa):
                    if self.slab_born.get(nm_, 0.0):
                        Fa[s0:s0 + n_, 2] += -self.slab_born[nm_] * dwA[s0:s0 + n_]
                    s0 += n_
        for gi, jj in self._pairs(Xa, Xb, oa, ob, ga, gb, ba, bb):
            d = Xb[jj] - Xa[gi]
            r = d.norm(dim=1).clamp_min(1e-9)
            inslab = None
            if self.slab_z is not None and "coulomb" in self.laws and self.slab_e > 0:
                inslab = (wA[gi], wB[jj], dwA[gi], dwB[jj])
            elif self.slab_z is not None and "coulomb" in self.laws:
                za, zb = Xa[gi, 2], Xb[jj, 2]
                inslab = ((za > self.slab_z[0]) & (za < self.slab_z[1]) & (zb > self.slab_z[0]) & (zb < self.slab_z[1]))
            sig = None if sga is None else 0.5 * (sga[gi] + sgb[jj])
            mu_i = self.mu_a if ma is None else ma[gi]
            f, nc, ez = self._force(d, r, None if qa is None else qa[gi], None if qb is None else qb[jj], inslab,
                                    sig, mu_i, h)
            n_capped += nc
            Fa.index_add_(0, gi, f)
            Fb.index_add_(0, jj, -f)
            if ez is not None:
                Fa[:, 2].index_add_(0, gi, ez[0])
                Fb[:, 2].index_add_(0, jj, ez[1])
            n_pairs += int(gi.numel())
        fr = _frame(H)
        if fr in _REPORT:
            name_a = self.A[0] + (f"..{self.A[-1]}" if len(self.A) > 1 else "")
            name_b = self.B[0] + (f"..{self.B[-1]}" if len(self.B) > 1 else "")
            print(f"[pair_potential f{fr}] {self.law} {name_a}<->{name_b}: {n_pairs:,} pairs in range"
                  + (f"; {n_capped} CAPPED by a guard" if n_capped else ""), flush=True)
        out = {}

        def _put(names, sizes, V):
            s0 = 0
            for nm, n in zip(names, sizes):
                out[nm] = out.get(nm, 0) + V[s0:s0 + n]
                s0 += n

        def _mob(m, scalar, F):
            return scalar * F if m is None else m[:, None] * F
        if self.same:
            _put(self.A, sa, _mob(ma, self.mu_a, Fa + Fb) * oa[:, None].to(Xa.dtype))
            return out
        if len(LB) == 1 and "force" in LB[0].state_schema._slices:
            _write(LB[0], "force", Fb.detach())
        _put(self.A, sa, _mob(ma, self.mu_a, Fa) * oa[:, None].to(Xa.dtype))
        if self.react:
            _put(self.B, sb, _mob(mb, self.mu_b, Fb) * ob[:, None].to(Xb.dtype))
        return out


# pair_potential[typed] -- the well depth per pair of cell TYPES inside one set (experiment 9: Steinberg
# 1963 / Graner & Glazier 1992 adhesion tables; a cell retyped by `contact_retype` adheres by its new type).
# Moved here from sorting_ops.py on 2026-09-27, unchanged.
def _type_index(lvl, name, where):
    names = list(getattr(lvl, "type_names", None) or [])
    if name not in names:
        raise ValueError(f"{where}: type {name!r} is not one of the set's types {names}")
    return names.index(name)


@register_operator("pair_potential", model="typed", family="interaction", set="particle", kind="exchange",
                   title="Pair potential with a well depth per pair of types",
                   equation=r"""$$U_{ij}=\epsilon_{\tau_i\tau_j}\,U_1(r_{ij})$$""")
class PairPotentialTyped(PairPotential):
    """`pair_potential` within one set, the well depth of each pair read from a table of the two
    cells' TYPES:

        U_ij = eps[tau_i, tau_j] U_1(r_ij)          F_ij = eps[tau_i, tau_j] F_1(r_ij)

    tau_i is cell i's current `node_type`, U_1 and F_1 the declared law(s) at depth 1 (`cooke`,
    `lj`, `wca`), and eps the symmetric table `epsilon_types: {A: {A: 1.0, B: 0.7}, B: {B: 0.6}}`
    keyed by the set's type names; every pair of types must be given (one order is enough). The
    table is looked up every frame, so a cell retyped by `contact_retype` adheres with its new
    type's depth from the next frame on. Coulomb is refused: its strength is a charge, not a depth.

    Reference: Steinberg, M.S. (1963). Science 141:401-408 (the works of adhesion per pair of
    types); Graner, F. & Glazier, J.A. (1992). Phys. Rev. Lett. 69:2013-2016 (the J table); Cooke,
    I.R., Kremer, K. & Deserno, M. (2005). Phys. Rev. E 72:011506 (the law).
    """
    REQUIRES_PARAMS = ["law", "epsilon_types"]
    PARAM_ROLES = dict(PairPotential.PARAM_ROLES, epsilon_types="well_depth_per_pair_of_types")
    # THE CELL GRID FROM 10 MILLION CANDIDATE PAIRS, not the parent's 2 billion. The parent's threshold
    # keeps every exp04 rig on the all-pairs path it always ran; an aggregate of 50,000 cells in one set
    # is 2.5e9 candidates a frame all-pairs (~100 ms on an L4) against ~5e6 on the grid. `_cell_pairs`
    # never loses a pair (tests/test_exp09_variants.py::test_typed_cell_list_equals_all_pairs).
    CELL_LIST_ABOVE = 1e7

    def __init__(self, params, device="cpu"):
        p = dict(params)
        p["epsilon"] = 1.0
        super().__init__(p, device)
        if "coulomb" in self.laws:
            raise ValueError("pair_potential[typed]: coulomb's strength is a charge, not a depth")
        if not self.same or len(self.A) != 1:
            raise ValueError("pair_potential[typed]: one set, with itself (its types carry the table)")
        self.table_spec = params["epsilon_types"]
        self._tab = None
        self._nt = None
        self._cur = None

    def _table(self, lvl, device, dtype):
        names = list(getattr(lvl, "type_names", None) or [])
        K = len(names)
        tab = torch.full((K, K), float("nan"), device=device, dtype=dtype)
        for a, row in self.table_spec.items():
            for b, e in (row or {}).items():
                i = _type_index(lvl, str(a), "pair_potential[typed]")
                j = _type_index(lvl, str(b), "pair_potential[typed]")
                tab[i, j] = tab[j, i] = float(e)
        miss = [(names[i], names[j]) for i in range(K) for j in range(i, K) if torch.isnan(tab[i, j])]
        if miss:
            raise ValueError(f"pair_potential[typed]: no depth for the type pairs {miss}")
        return tab

    def _pairs(self, *a, **kw):
        for gi, jj in super()._pairs(*a, **kw):
            self._cur = (gi, jj)
            yield gi, jj

    def _force(self, d, r, *a, **kw):
        f, nc, ez = super()._force(d, r, *a, **kw)
        gi, jj = self._cur
        return f * self._tab[self._nt[gi], self._nt[jj]][:, None], nc, ez

    def forward(self, H, mask=None):
        lvl = H.level(self.A[0])
        X = lvl.get("pos")
        if self._tab is None or self._tab.device != X.device:
            self._tab = self._table(lvl, X.device, X.dtype)
        self._nt = lvl.node_type.to(X.device).long()
        return super().forward(H, mask)


# =============================================================================================
# THE BATH, THE PLANE, THE PIPETTE
# =============================================================================================
@register_operator("brownian", family="motion", set="particle", kind="lateral",
                   title="Thermal bath of an overdamped bead",
                   equation=r"""$$\dot{\mathbf x}_i=\sqrt{2\mu k_BT/\Delta t}\;\boldsymbol\xi_i$$""")
class Brownian(Lateral):
    """The thermal half of an overdamped bead's Langevin equation.

    particle -> particle: emits a velocity.

        dx_i/dt = sqrt(2 mu kT / dt) xi_i      xi_i a standard normal vector, fresh each frame

    mu (`mobility`) is the same mobility the forces on the set use, kT (sim energy) the thermal
    energy, dt the frame step: the fluctuation-dissipation partner of v = mu F, so the set samples
    exp(-U/kT) whatever the forces are. `axes` restricts it (a lipid's noise in its plane).

    Reference: Einstein, A. (1905). Ann. Phys. 17:549; Ermak, D.L. & McCammon, J.A. (1978). J. Chem.
    Phys. 69:1352 (Brownian dynamics).
    """

    EMIT = "velocity"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["kT"]
    MECHANISM_TAGS = ["thermal_noise", "brownian_dynamics", "fluctuation_dissipation"]
    PARAM_ROLES = {"kT": "thermal_energy_sim", "mobility": "bead_mobility", "axes": "axes_the_noise_acts_on",
                   "seed": "rng_seed"}
    PARAM_UNITS = {"kT": "energy", "mobility": "mobility"}
    REFERENCE = "Einstein (1905). Ann. Phys. 17:549; Ermak & McCammon (1978). J. Chem. Phys. 69:1352."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.kT = float(params["kT"])
        self.mu = float(params.get("mobility", 1.0))
        self.axes = params.get("axes")
        self.seed = int(params.get("seed", 0))
        self._gen = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        if self._gen is None:
            self._gen = torch.Generator(device=X.device)
            self._gen.manual_seed(self.seed)
        h = float(getattr(H, "dt", 1.0))
        v = math.sqrt(2.0 * self.mu * self.kT / h) * torch.randn(X.shape, generator=self._gen,
                                                                 device=X.device, dtype=X.dtype)
        if self.axes is not None:
            keep = torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)
            keep[[int(a) for a in self.axes]] = 1.0
            v = v * keep[None, :]
        v = v * lvl.occ[:, None]
        if mask is not None:
            v = v * mask[:, None].to(v.dtype)
        return {self.at: v}


@register_operator("brownian", model="anneal", family="motion", set="particle", kind="lateral",
                   title="Thermal bath cooled over the run",
                   equation=r"""$$\dot{\mathbf x}_i=\sqrt{2\mu k_BT(t)/\Delta t}\;\boldsymbol\xi_i,\qquad k_BT(t)=k_BT_0+(k_BT_1-k_BT_0)\,\min(1,t/t_1)$$""")
class BrownianAnneal(Brownian):
    """`brownian` whose temperature is cooled linearly from `kT` to `kT_end` over `frames` frames, then
    held -- an annealing schedule.

    WHY (experiment 9, Phase 1 Finding 14, direction 2). In a thermal particle model of cell sorting one
    temperature sets both how freely cells move and how deeply the two types demix: at kT 0.35 of the
    strongest adhesion the aggregate crystallised (hexagonal order 0.73-0.83) and sorted at half the
    paper's pace; at 0.42 it sat near its demixing point and stalled at demix 0.19. A schedule crosses
    the two regimes in time: fluid first, while the domains form, colder later, when they must deepen.
    `kT` is the starting temperature (the default operator's parameter), `kT_end` the final one,
    `frames` the length of the ramp in frames; with kT_end = kT it is the default exactly.

    Reference: Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization by simulated
    annealing. Science 220:671-680.
    """
    PARAM_ROLES = dict(Brownian.PARAM_ROLES, kT_end="final_thermal_energy_sim", frames="ramp_length_frames")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.kT0 = self.kT
        self.kT1 = float(params.get("kT_end", self.kT))
        self.frames = max(1, int(params.get("frames", 1)))
        self._step = 0

    def forward(self, H, mask=None):
        self.kT = self.kT0 + (self.kT1 - self.kT0) * min(1.0, self._step / self.frames)
        self._step += 1
        return super().forward(H, mask)


@register_operator("brownian", model="active_cell", family="motion", set="vertex", kind="lateral",
                   title="Self-propelled cells of a vertex tissue (the active vertex model)",
                   equation=r"""$$\dot{\mathbf x}_v=v_0\,\langle\mathbf n_c\rangle_{c\ni v},\qquad \mathbf n_c=(\cos\theta_c,\sin\theta_c),\qquad d\theta_c=\sqrt{2D_r}\,dW_c$$""")
class BrownianActiveCell(Lateral):
    """The ACTIVE half of a motile cell's equation of motion on a vertex mesh: each cell crawls at a
    constant speed along its own polarity, and the polarity diffuses.

    vertex -> vertex: reads the mesh (which cells own each vertex), emits a velocity.

        dx_v/dt   = v0 <n_c>_{c owns v}         the mean polarity of the cells meeting at vertex v
        n_c       = (cos theta_c, sin theta_c)  in the tissue's plane (the two axes other than plane_axis)
        dtheta_c  = sqrt(2 Dr dt) xi            xi a standard normal, fresh each frame per cell

    v0 (`v0`, world units per unit time) is the crawling speed, Dr (`Dr`, rad^2 per unit time) the
    rotational diffusion of the polarity: the persistence time is 1/Dr, the persistence length v0/Dr.
    Added to `cell_mechanics`' velocity (mu F), it is Bi et al. 2016's self-propelled Voronoi / Barton
    et al. 2017's active vertex model, dr/dt = mu F + v0 n: the forces of the shape energy and the
    tensions, plus a self-propulsion the forces do not derive from.

    WHY (experiment 16, step 3). A vertex aggregate with the paper's type-pair tensions and a cortical
    fluctuation makes ~90 T1 flips a frame for 368 cells, yet the median cell moves under one cell width
    in 72 h: the flips flicker in place. Cerchiari et al. 2015's LEP and MEP crawl; a persistent crawl
    is what carries a cell across the aggregate so the tensions can choose where it stays. `brownian` is
    the thermal kick of a bead (white, per vertex); this is its active, persistent, per-cell form.
    v0 = 0 is no motion at all (the tissue is `cell_mechanics` alone).

    Reference: Bi, D., Yang, X., Marchetti, M.C. & Manning, M.L. (2016). Motility-driven glass and
    jamming transitions in biological tissues. Phys. Rev. X 6:021011; Barton, D.L., Henkes, S.,
    Weijer, C.J. & Sknepnek, R. (2017). Active vertex model for cell-resolution description of
    epithelial tissue mechanics. PLoS Comput. Biol. 13:e1005569.
    """

    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["v0"]
    MECHANISM_TAGS = ["motility", "self_propulsion", "active_vertex_model", "persistent_random_walk"]
    PARAM_ROLES = {"v0": "crawling_speed_world_per_time", "Dr": "polarity_rotational_diffusion",
                   "plane_axis": "the_axis_normal_to_the_tissue_plane", "cell_set": "the_mesh_faces_set",
                   "seed": "rng_seed"}
    REFERENCE = ("Bi, D. et al. (2016). Phys. Rev. X 6:021011; Barton, D.L. et al. (2017). PLoS Comput. "
                 "Biol. 13:e1005569.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.v0 = float(params["v0"])
        self.Dr = float(params.get("Dr", 0.1))
        self.plane_axis = params.get("plane_axis")
        self._cat = params.get("cell_set")
        self.seed = int(params.get("seed", 0))
        self._gen = None
        self._theta = None

    def forward(self, H, mask=None):
        from plexus.operators.vertex_ops import resolve_cell_set
        lvl = H.level(self.at)
        X = lvl.get("pos")
        v = torch.zeros_like(X)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {self.at: v}
        clvl = H.level(resolve_cell_set(H, self.at, self._cat))
        n_buf = int(clvl.node_type.shape[0])
        h = float(getattr(H, "dt", 1.0))
        if self._theta is None or self._theta.shape[0] != n_buf:
            self._gen = torch.Generator(device="cpu").manual_seed(self.seed)
            self._theta = 2.0 * math.pi * torch.rand(n_buf, generator=self._gen, dtype=torch.float64)
        else:
            self._theta = self._theta + math.sqrt(2.0 * self.Dr * h) * torch.randn(n_buf, generator=self._gen,
                                                                                   dtype=torch.float64)
        if self.v0 == 0.0:
            return {self.at: v}
        pa = self.plane_axis if self.plane_axis is not None else (m.get("mech") or {}).get("plane_axis", 2)
        ax = [a for a in range(X.shape[1]) if a != int(pa)][:2]
        th = self._theta.to(device=X.device, dtype=X.dtype)
        n = torch.zeros(n_buf, X.shape[1], device=X.device, dtype=X.dtype)
        n[:, ax[0]], n[:, ax[1]] = torch.cos(th), torch.sin(th)
        es, ef = m["E_srce"].long().to(X.device), m["E_face"].long().to(X.device)
        s = torch.zeros_like(X).index_add_(0, es, n[ef])
        c = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype).index_add_(0, es, torch.ones_like(es, dtype=X.dtype))
        v = self.v0 * s / c.clamp_min(1.0)[:, None]
        v = v * lvl.occ[:, None].to(v.dtype)
        if mask is not None:
            v = v * mask[:, None].to(v.dtype)
        return {self.at: v}


@register_operator("tether", family="boundary", set="particle", kind="lateral",
                   title="Harmonic hold to the seeded place",
                   equation=r"""$$\dot{x}_{i,a}=-\mu\,k\,(x_{i,a}-x^0_{i,a}),\quad a\in\text{axes}$$""")
class Tether(Lateral):
    """A harmonic hold of each bead to the place it was seeded, on the chosen axes.

    particle -> particle: reads pos (and a mask block), emits a velocity.

        dx_ia/dt = -mu k (x_ia - x0_ia)      for a in `axes`, for beads whose `block` > 0.5 (all if none)

    Two uses, both physical: the BILAYER'S CONFINEMENT of a lipid bead to its plane (axes [2]: the
    hydrophobic core holds a lipid at the membrane's depth while it flows freely in-plane), and an
    ANCHOR -- the part of a protein held by what the box leaves out (a cytoplasmic domain bound to
    the cell's interior, a cell wall). `k` in sim energy per world squared.

    `thin_with: {set: cell, block: strain, mid: z}` -- A BILAYER THAT THINS AS IT STRETCHES. Its
    hydrocarbon is incompressible, so its thickness goes as one over its area: each bead's held
    height becomes mid + (z0 - mid) / (1 + area strain), the strain read every frame from the named
    block (radial_drive writes it). With it a stretched membrane is thinner than the closed channel's
    hydrophobic belt, the mismatch the gating literature finds tilts mechanosensitive channels open
    (Perozo et al. 2002, Nature 418:942); without it the leaflets keep their seeded depth under any
    strain. Hydrocarbon volume conservation: Rawicz et al. 2000, Biophys. J. 79:328.

    Reference: none -- a boundary condition, stated.
    """

    EMIT = "velocity"
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["k"]
    MECHANISM_TAGS = ["anchor", "confinement", "membrane_plane"]
    PARAM_ROLES = {"k": "spring_constant_sim_energy_per_world2", "axes": "axes_held", "block": "mask_block",
                   "mobility": "bead_mobility", "thin_with": "set_block_and_midplane_of_a_volume_conserving_bilayer"}
    PARAM_UNITS = {"k": "F/L", "mobility": "mobility"}
    REFERENCE = "Plexus (this work); Rawicz, W. et al. (2000). Biophys. J. 79:328 (thinning)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.k = float(params["k"])
        self.mu = float(params.get("mobility", 1.0))
        self.axes = [int(a) for a in (params.get("axes") or [0, 1, 2])]
        self.block = params.get("block")
        self.thin = params.get("thin_with")
        self._x0 = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        if self._x0 is None:
            self._x0 = X.detach().clone()
        x0 = self._x0
        if self.thin and str(self.thin.get("set", "cell")) in H.levels:
            b = _block(H.level(str(self.thin.get("set", "cell"))), str(self.thin.get("block", "strain")))
            if b is not None:
                s_ = float(b[0, 0])
                zm = float(self.thin["mid"])
                x0 = x0.clone()
                x0[:, 2] = zm + (x0[:, 2] - zm) / max(1.0 + s_, 0.2)
        m = _block(lvl, self.block)
        w = torch.ones(X.shape[0], device=X.device, dtype=X.dtype) if m is None else (m[:, 0] > 0.5).to(X.dtype)
        keep = torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)
        keep[self.axes] = 1.0
        v = -self.mu * self.k * (X - x0) * keep[None, :] * (w * lvl.occ)[:, None]
        return {self.at: v}


def _path_from_beads(H, cfg, r_ion, z0):
    """THE ION PATH OF THE PROTEIN AS IT IS NOW, from its beads (one per residue): per height z (bins `dz` from
    `z` lo to hi, world), R = the nearest bead to the axis less `reach` (a residue's side chain beyond its alpha
    carbon, fitted per channel so the deposited states' all-atom paths are reproduced), ENCLOSED when the beads
    within R + reach + `ring` cover `sectors` of 8 sectors round the axis, and HYDROPHOBIC when fewer than
    `hydrophobic_below` of the beads lining that height (within `lining` of the nearest) are polar (`hyd` block
    0). Returns the barrier's table [z, rp, R, dry, cx, cy] (world) for an ion of radius `r_ion`, the axis the
    xy centroid of the beads within `axis_half` of the mid-plane. The same rules as channel_spec._ion_paths."""
    sets = [str(s_) for s_ in cfg["sets"]]
    A = torch.cat([H.level(s_).get("pos").detach() for s_ in sets]).to(torch.float64)
    hb = cfg.get("hyd")
    hyd = (torch.cat([_block(H.level(s_), hb)[:, 0] for s_ in sets]).to(torch.float64) if hb
           else torch.ones(A.shape[0], dtype=torch.float64, device=A.device))
    ah = float(cfg.get("axis_half", 0.1))
    core = (A[:, 2] - z0).abs() < ah
    c = A[core].mean(0) if bool(core.any()) else A.mean(0)
    zlo, zhi = [float(v) for v in cfg["z"]]
    dz = float(cfg["dz"]); win = float(cfg.get("window", dz * 3))
    reach = float(cfg["reach"]); ring = float(cfg.get("ring", reach * 2)); lin = float(cfg.get("lining", reach * 0.6))
    zb = torch.arange(zlo, zhi + 1e-12, dz, dtype=torch.float64, device=A.device)
    dx = A[:, 0] - c[0]; dy = A[:, 1] - c[1]
    r = torch.sqrt(dx * dx + dy * dy)
    sec = (((torch.atan2(dy, dx) + math.pi) / (2 * math.pi) * 8).long() % 8)
    m = (A[None, :, 2] - zb[:, None]).abs() < win                                   # [bins, beads]
    big = torch.full_like(r, 1e9)
    rmin = torch.where(m, r[None, :], big[None, :]).min(1).values                   # [bins]
    has = m.any(1)
    ringm = m & (r[None, :] < (rmin[:, None] + ring))
    onehot = torch.nn.functional.one_hot(sec, 8).to(torch.float64)                 # [beads, 8]
    cover = (ringm.to(torch.float64) @ onehot > 0).sum(1)                           # sectors covered
    enclosed = has & (cover >= int(cfg.get("sectors", 6)))
    linm = m & (r[None, :] < (rmin[:, None] + lin))
    npol = (linm.to(torch.float64) * (1.0 - hyd)[None, :]).sum(1)
    nlin = linm.to(torch.float64).sum(1).clamp_min(1.0)
    polar = (npol / nlin) >= float(cfg.get("hydrophobic_below", 0.2))
    R = torch.where(has, rmin - reach, torch.zeros_like(rmin)).clamp_min(0.0)
    rp = torch.where(enclosed, (R - r_ion).clamp_min(0.0), torch.zeros_like(R))
    sf = float(cfg.get("single_file", 0.0))
    rp = torch.where(enclosed & polar & (rp < sf), torch.full_like(rp, sf), rp)
    d0, d1 = [float(v) for v in cfg["dry"]]
    t = ((R - d0) / max(d1 - d0, 1e-12)).clamp(0.0, 1.0)
    dry = torch.where(enclosed & ~polar, 0.5 * (1.0 + torch.cos(math.pi * t)), torch.zeros_like(R))
    w = max(int(round(float(cfg.get("dry_smooth", 2 * dz)) / dz)), 1)
    pad = torch.nn.functional.pad(dry[None, None, :], (w, w))
    dmax = torch.nn.functional.max_pool1d(pad, 2 * w + 1, stride=1)[0, 0]
    ker = torch.cat([torch.arange(1, w + 1), torch.arange(w, 0, -1)]).to(torch.float64).to(A.device)
    ker = ker / ker.sum()
    dry_s = torch.nn.functional.conv1d(torch.nn.functional.pad(dmax[None, None, :], (w - 1, w)), ker[None, None, :])[0, 0]
    dry_s = dry_s[:dry.numel()].clamp(0.0, 1.0)
    cx = torch.full_like(R, float(c[0])); cy = torch.full_like(R, float(c[1]))
    return torch.stack([zb, rp, R, dry_s, cx, cy], 1)


@register_operator("morph_gate", family="mechanics", set="particle", kind="lateral",
                   title="A channel's gate moving along the path between its two deposited states",
                   equation=r"""$$\mathbf x_i=\mathbf x_i^{c}+\lambda\,\mathbf d_i,\qquad \dot\lambda=\frac{\mu}{\sum_i|\mathbf d_i|^2}\Big(\sum_i\mathbf f_i\cdot\mathbf d_i-\frac{\partial U}{\partial\lambda}\Big)+\text{noise},\quad U=\Delta V\,\lambda$$""")
class MorphGate(Lateral):
    """The protein's conformation constrained to the straight path from its closed deposited state to its open one,
    x_i = x_i^c + lambda d_i (d_i = x_i^o - x_i^c), and driven along it by whatever pushes on it.

    particle -> particle: reads pos, emits a velocity that puts every bead back on the path; writes lambda into a
    cell block (`gate_block`), which the ion paths blend by (`slab_barrier.gate`, `electrolyte_conduction.gate`).

        each frame:  delta_i = x_i - (x_i^c + lambda d_i)            what the other operators did to the bead
                     dlambda = sum_i delta_i . d_i / sum_i |d_i|^2  -  mu dt dU/dlambda / sum_i |d_i|^2
                     lambda <- clamp(lambda + dlambda, 0, 1)

    The projection is exact overdamped dynamics on the one coordinate: the beads' mobility mu gives lambda the
    mobility mu / sum|d|^2 and the generalized force sum f.d -- the lipids' push on the protein's outline, the
    ions', the bath's kicks (their projection is lambda's own thermal noise, fluctuation-dissipation intact).
    U = dV lambda (`basin_offset`, kT): the open state's free energy above the closed one at zero tension --
    tau_1/2 x dA from the channel's paper. Under tension the membrane does tau x dA(lambda) of work along the
    path, so the gate opens where tau passes dV / dA_model -- the model's own area change sets the threshold.
    WHY NOT THE TWO-BASIN ELASTIC NETWORK: with springs stiff enough to keep helices helices, the network's
    energies between the two deposited geometries are thousands of kT (batch 11: V_closed - V_open - dV
    = -2,300 kT for MscS at 8.6 mN/m), against tens of kT of tension work -- the basins could not switch.
    STATED LIMITATION: the protein moves only along the straight path between its two structures (no other
    deformation, no rotation or drift), and the path's energy is linear (no barrier unless `barrier` kT is set,
    4 barrier lambda (1 - lambda)).

    Reference: the constrained-coordinate reduction of a conformational change (e.g. Maragliano et al. 2006,
    J. Chem. Phys. 125:024106, the string/path coordinate); dV = tau_1/2 dA (Sukharev et al. 1999, Biophys. J.
    76:2394; Wiggins & Phillips 2004, PNAS 101:4071).
    """

    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    MAY_MUTATE_INTEGRATED_STATE = True             # writes lambda, a readout, into the cell
    REQUIRES_PARAMS = ["sets", "open_reference", "open_parts"]
    MECHANISM_TAGS = ["gating", "conformational_path", "mechanosensation"]
    PARAM_ROLES = {"sets": "the_protein_sets_in_order", "open_reference": "shape_holding_the_open_state",
                   "open_parts": "one_open_array_per_set", "open_scale": "world_per_metre", "origin": "where_0_0_0_lands",
                   "basin_offset": "open_state_free_energy_above_closed_kT", "barrier": "path_barrier_kT",
                   "gate_block": "set_and_block_that_receive_lambda", "mobility": "bead_mobility", "lambda0": "start",
                   "tension_block": "set_and_block_holding_the_membranes_measured_tension",
                   "area_change": "the_paths_in_plane_area_change_world2"}
    REFERENCE = ("Maragliano et al. (2006) J. Chem. Phys. 125:024106; Sukharev et al. (1999) Biophys. J. 76:2394; "
                 "Wiggins & Phillips (2004) PNAS 101:4071.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.sets = [str(s_) for s_ in params["sets"]]
        self.open_ref = str(params["open_reference"])
        self.open_parts = list(params["open_parts"])
        self.scale = float(params.get("open_scale", 1.0))
        self.origin = [float(v) for v in (params.get("origin") or [0.0, 0.0, 0.0])]
        self.dV = float(params.get("basin_offset", 0.0))
        self.barrier = float(params.get("barrier", 0.0))
        self.gate_block = params.get("gate_block")
        self.mu = float(params.get("mobility", 1.0))
        self.lam = float(params.get("lambda0", 0.0))
        # `tension_block: [set, block]` + `area_change` (world^2) -- THE MEMBRANE'S WORK ON THE GATE, STATED: the
        # tension the membrane itself carries (measured on the patch's frame, not prescribed) times the path's
        # area change, tau dA -- the work term of a mechanosensitive channel's free energy (dG = dV - tau dA;
        # Sukharev 1999, Wiggins & Phillips 2004). It stands in for the lipids' pull on the protein's outline,
        # which in batch 12 did not move MscS or MscL at 12 mN/m (lambda 0.09 with tension, 0.10 without).
        self.tension_block = params.get("tension_block")
        self.dA = float(params.get("area_change", 0.0))
        self._Xc = None

    def _load(self, X):
        import os
        import numpy as np
        from plexus import shapes
        path = next((os.path.join(r, self.open_ref, "points.npz") for r in shapes.roots()
                     if os.path.exists(os.path.join(r, self.open_ref, "points.npz"))), None)
        if path is None:
            raise FileNotFoundError(f"morph_gate: no shapes/{self.open_ref}/points.npz")
        with np.load(path) as z:
            XO = np.concatenate([np.asarray(z[p_], np.float64) * self.scale for p_ in self.open_parts])
        XO = XO + np.asarray(self.origin)[None, :]
        miss = ~np.isfinite(XO).all(1)
        Xc = X.detach().cpu().numpy()
        XO = np.where(miss[:, None], Xc, XO)                           # a bead the open model lacks stays put
        self._Xc = X.detach().clone()
        self._D = torch.as_tensor(XO, device=X.device, dtype=X.dtype) - self._Xc
        self._D2 = float((self._D * self._D).sum())
        print(f"[morph_gate] path from the seed to {self.open_ref}: {X.shape[0]} beads ({int(miss.sum())} missing from "
              f"the open model), rms displacement {math.sqrt(self._D2 / max(X.shape[0], 1)):.4g} world, dV {self.dV:.4g}",
              flush=True)

    def forward(self, H, mask=None):
        lvls = [H.level(s_) for s_ in self.sets]
        X = torch.cat([lv.get("pos") for lv in lvls])
        if self._Xc is None:
            self._load(X)
        h = float(getattr(H, "dt", 1.0))
        on = self._Xc + self.lam * self._D
        delta = (X - on).detach()
        dU = self.dV + self.barrier * 4.0 * (1.0 - 2.0 * self.lam)
        tau = 0.0
        if self.tension_block:
            tau = float(_block(H.level(str(self.tension_block[0])), str(self.tension_block[1]))[0, 0])
            tau = tau if math.isfinite(tau) else 0.0
            dU -= tau * self.dA
        dlam = float((delta * self._D).sum()) / self._D2 - self.mu * h * dU / self._D2
        if math.isfinite(dlam):
            self.lam = min(max(self.lam + dlam, 0.0), 1.0)
        target = self._Xc + self.lam * self._D
        V = (target - X) / h
        if self.gate_block:
            lv_ = H.level(str(self.gate_block[0]))
            _write(lv_, str(self.gate_block[1]), torch.full((lv_.get("pos").shape[0], 1), self.lam, dtype=X.dtype, device=X.device))
        fr = _frame(H)
        if fr in _REPORT:
            print(f"[morph_gate f{fr}] lambda {self.lam:.4f} (0 closed, 1 open); the last step's push along the path "
                  f"{float((delta * self._D).sum()) / self._D2:+.3e}, the offset's pull {-self.mu * h * dU / self._D2:+.3e}", flush=True)
        out, s0 = {}, 0
        for nm_, lv in zip(self.sets, lvls):
            n = lv.get("pos").shape[0]
            out[nm_] = V[s0:s0 + n] * lv.occ[:, None]
            s0 += n
        return out


@register_operator("slab_barrier", family="boundary", set="particle", kind="lateral",
                   title="The hydrocarbon core's Born barrier to ions",
                   equation=r"""$$U=W\,f(z)\,g(r),\quad W=\frac{q^2 e^2}{8\pi\varepsilon_0 a}\Big(\frac{1}{\varepsilon_m}-\frac{1}{\varepsilon_w}\Big)$$""")
class SlabBarrier(Lateral):
    """An ion cannot enter a membrane's hydrocarbon core, except through a pore.

    particle -> particle: reads pos, emits a velocity.

        U(x) = W f(z) g(r),   f = 1 inside |z - z0| < h, 0 outside (cosine edges `edge` wide)
                              g = 0 inside r < pore_radius of the axis, 1 outside (cosine edge)

    W is the Born energy of moving the ion from water into the core: q^2 e^2 / (8 pi eps0 a) x
    (1 / eps_m - 1 / eps_w), ~99 kT for K+ (a 0.138 nm, eps_m 2, eps_w 78.5; Parsegian 1969) -- the
    generator computes it. WHY: the bead membrane is two sheets of lipid heads 2 nm apart with
    nothing between them, so ions went round the patch's edge and through the empty core (exp04
    round 4: of 1,289 K+ that crossed the membrane plane, 3 went through the pore). The core is an
    insulator for ions for the same reason it is for current, and this is that reason as a force.
    Inside `pore_radius` of the axis the protein's own beads decide where an ion can go.

    Reference: Parsegian, A. (1969). Nature 221:844 (the energy of an ion crossing a low dielectric).
    """

    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["height", "z0", "half_thickness"]
    MECHANISM_TAGS = ["born_energy", "membrane_core", "ion_exclusion"]
    PARAM_ROLES = {"height": "born_energy_sim", "z0": "membrane_mid_plane_world", "half_thickness": "core_half_thickness_world",
                   "edge": "cosine_edge_width_world", "pore_radius": "no_barrier_within_this_of_the_axis_world",
                   "pore_edge": "cosine_edge_of_the_pore_world", "axis_xy": "the_axis_x_y_world", "mobility": "mobility",
                   "pore_profile": "list_of_z_and_pore_radius_world",
                   "lining": "sets_block_value_margin_of_the_pore_lining_beads", "lining_every": "frames_between_rereads",
                   "pores": "list_of_ion_path_profiles_z_rp_R_dry_cx_cy_world", "plug_margin": "dry_plug_extends_past_the_lumen_world",
                   "path_from": "measure_the_path_on_these_moving_beads_every_n_frames", "r_ion": "the_ion_radius_world",
                   "pores_open": "the_open_states_paths_same_heights", "gate": "set_and_block_holding_the_open_weight",
                   "step_max": "cap_on_one_frames_barrier_step_world_a_guard",
                   "hard": "return_an_ion_found_in_the_core_outside_every_path",
                   "parents": "set_whose_every_entity_carries_the_pores_table_in_its_own_frame",
                   "rotate": "parent_block_of_degrees_about_z"}
    PARAM_UNITS = {"height": "energy", "mobility": "mobility"}
    REFERENCE = ("Parsegian, A. (1969). Nature 221:844; Beckstein, O. & Sansom, M.S.P. (2003). PNAS 100:7063 "
                 "(hydrophobic gating: liquid-vapour oscillations of water in nanopores); Smart, O.S. et al. (1996). "
                 "J. Mol. Graph. 14:354 (HOLE).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.W = float(params["height"])
        self.z0 = float(params["z0"]); self.h = float(params["half_thickness"])
        self.e = float(params.get("edge", 0.0))
        self.rp = float(params.get("pore_radius", 0.0)); self.er = float(params.get("pore_edge", 0.0))
        # `pore_profile: [[z, r], ...]` (world) -- THE PORE'S OWN RADIUS AT EACH HEIGHT, interpolated,
        # instead of one cylinder: a potassium channel's filter is ~0.3 nm wide and its cavity ~1 nm, and
        # a 1 nm cylinder let ions slip between the filter's helices (round 6's first readouts: 175-380 pS
        # against 10-30). Outside the profile's z range, `pore_radius`.
        prof = params.get("pore_profile")
        self.prof = (torch.tensor([[float(a), float(b)] for a, b in prof], dtype=torch.float64)
                     if prof else None)
        # `lining: {sets, block, value, margin}` -- A PORE THAT OPENS WITH THE CHANNEL: the barrier-free
        # radius is the median distance from the axis of the lining beads (the `sets`' beads whose
        # `block` equals `value`) inside the core, less `margin` -- re-read every `lining_every` frames.
        # A mechanosensitive channel's pore widens as it opens; a fixed radius would let ions round a
        # shut gate or bar them from an open one. The axis follows the lining's own centroid.
        self.lining = params.get("lining")
        self.lin_every = int(params.get("lining_every", 100))
        self._rp_now = self.rp
        self.axy = [float(v) for v in (params.get("axis_xy") or [0.5, 0.5])]
        self.mu = float(params.get("mobility", 1.0))
        # `pores: [{profile: [[z, rp, R, dry, cx, cy], ...]}, ...]` (world) -- THE ION PATHS OF A DEPOSITED
        # STRUCTURE, measured HOLE-like on all its atoms (tools/channel_anatomy.py): per height, the radius
        # rp an ion's centre may reach from the path's centre (cx, cy), the lumen's own radius R, and `dry`,
        # the fraction of a hydrophobic constriction too narrow to hold liquid water (Beckstein & Sansom
        # 2003). The core's barrier holds outside every path; a dry segment is a PLUG of the same height
        # across its lumen at ANY height -- MscS's closed gate (L105/L109) lies below the hydrocarbon core.
        # Several paths: a porin's three barrels. `plug_margin` (world) widens the plug past R.
        self.pores = None
        if params.get("pores"):
            self.pores = [torch.tensor([[float(v) for v in row] for row in po["profile"]], dtype=torch.float64)
                          for po in params["pores"]]
        self.plug_margin = float(params.get("plug_margin", 0.0))
        # `step_max` (world): no frame's barrier force may move an ion further than this -- the path's slope along z
        # is steep where a measured radius jumps between two heights (a 0.83 nm kick in open MscS, 2026-09-26),
        # and a kick longer than the wall is thick would carry an ion through it. A guard, never a law.
        self.step_max = float(params.get("step_max", 0.0))
        # `hard: true` -- the core outside every path a constraint as well as an energy (see forward)
        self.hard = bool(params.get("hard", False))
        self._prev = None
        self.n_returned = 0
        # `path_from: {sets, reach, hyd, z, dz, dry, every, ...}` -- A PATH THAT MOVES WITH THE PROTEIN: re-measured
        # on its beads every `every` frames (`_path_from_beads`), so a gate that opens under its stimulus opens
        # for the ions too. `r_ion` is this set's radius (world). Replaces `pores` while it is set.
        self.path_from = params.get("path_from")
        self.r_ion = float(params.get("r_ion", 0.0))
        # `pores_open` + `gate: [set, block]` -- TWO MEASURED STATES AND THE PROTEIN'S OWN CHOICE BETWEEN THEM:
        # the paths of the closed (`pores`) and the open (`pores_open`) deposited states, on the same heights,
        # blended row by row with the open weight the elastic network's two basins publish (`gate_block`).
        # The path is exact at both ends; STATED LIMITATION: in between it is the linear blend, not measured.
        self.pores_c = self.pores
        self.pores_o = ([torch.tensor([[float(v) for v in row] for row in po["profile"]], dtype=torch.float64)
                         for po in params["pores_open"]] if params.get("pores_open") else None)
        self.gate = params.get("gate")
        self.w_open = float("nan")
        # `parents: <set>` (+ `rotate: <block>`) -- ONE CHANNEL'S PATHS, CARRIED BY EVERY ENTITY OF A SET: `pores`
        # are then written in a channel's own frame (cx, cy from its centre, the frame `cloud_seed` places the
        # protein's copies in), and each ion is read in the frame of the NEAREST channel in the membrane plane,
        # turned back by that channel's own angle about z (degrees, the parent block `rotate` names). One table
        # serves a hundred channels at the cost of one; the core's barrier between channels, far from every
        # path, is the same whichever channel is nearest, so the switch between frames is seamless.
        self.parents = params.get("parents")
        self.rotate = params.get("rotate")
        if self.parents and (self.pores is None or self.hard or self.path_from or self.pores_o is not None):
            raise ValueError("slab_barrier: `parents` places a fixed `pores` table on every channel; it takes "
                             "neither `hard`, `path_from` nor `pores_open`")

    def _in_channel_frame(self, H, X):
        """Each position in the frame of its nearest parent, and that frame's (cos, sin) about z."""
        par = H.level(self.parents)
        C = par.get("pos")[:, :2].to(X.dtype)
        k = torch.cdist(X[:, :2], C).argmin(1)
        d = X[:, :2] - C[k]
        if self.rotate:
            th = torch.deg2rad(_block(par, self.rotate)[:, 0].to(X.dtype))[k]
            c, s = torch.cos(th), torch.sin(th)
        else:
            c, s = torch.ones_like(d[:, 0]), torch.zeros_like(d[:, 0])
        Xl = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], X[:, 2]], 1)
        return Xl, c, s

    @staticmethod
    def _step(u, a, w):
        """1 below a - w/2, 0 above a + w/2, a cosine between; and its derivative."""
        if w <= 0:
            return (u < a).to(u.dtype), torch.zeros_like(u)
        t = ((u - (a - 0.5 * w)) / w).clamp(0.0, 1.0)
        f = 0.5 * (1.0 + torch.cos(math.pi * t))
        df = torch.where((t > 0) & (t < 1), -0.5 * math.pi * torch.sin(math.pi * t) / w, torch.zeros_like(u))
        return f, df

    def _lining_radius(self, H):
        L_ = self.lining
        A = torch.cat([H.level(s_).get("pos").detach() for s_ in L_["sets"]])
        b = torch.cat([_block(H.level(s_), str(L_.get("block", "domain")))[:, 0] for s_ in L_["sets"]])
        m = ((b - float(L_.get("value", 0))).abs() < 0.5) & ((A[:, 2] - self.z0).abs() < self.h)
        if not bool(m.any()):
            return 0.0, None
        c = A[m].mean(0)
        r = (A[m, :2] - c[:2]).norm(dim=1).median()
        return max(float(r) - float(L_.get("margin", 0.0)), 0.0), [float(c[0]), float(c[1])]

    @staticmethod
    def _interp(P, zq):
        """Columns of a [z, ...] table at heights zq, linear, and d/dz; zero outside the table."""
        zp = P[:, 0]
        zc = zq.clamp(float(zp.min()), float(zp.max()))
        k = torch.searchsorted(zp.contiguous(), zc.contiguous()).clamp(1, zp.numel() - 1)
        h = (zp[k] - zp[k - 1]).clamp_min(1e-12)
        t = ((zc - zp[k - 1]) / h)[:, None]
        V = P[k - 1, 1:] + t * (P[k, 1:] - P[k - 1, 1:])
        dV = (P[k, 1:] - P[k - 1, 1:]) / h[:, None]
        inside = ((zq >= zp.min()) & (zq <= zp.max()))[:, None]
        return torch.where(inside, V, torch.zeros_like(V)), torch.where(inside, dV, torch.zeros_like(dV))

    def _forward_pores(self, H, lvl, X):
        dz = X[:, 2] - self.z0
        f, dfdz_abs = self._step(dz.abs(), self.h, self.e)                 # 1 inside the core
        n = X.shape[0]
        outside_all = torch.ones(n, dtype=X.dtype, device=X.device)
        parts = []
        F = torch.zeros_like(X)
        for P in self.pores:
            P = P.to(X)
            V, dV = self._interp(P, X[:, 2])
            rp, R, dry, cx, cy = V[:, 0], V[:, 1], V[:, 2], V[:, 3], V[:, 4]
            dx = X[:, 0] - cx; dy = X[:, 1] - cy
            r = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-12)
            # THE PATH'S OWN SLOPE ALONG z. g = S(r - rp(z)) with r measured from the centre (cx(z), cy(z)), so
            # dg/dz = S' (dr/dz - drp/dz), dr/dz = -(dx dcx/dz + dy dcy/dz) / r. Leaving it out (batch 10) let an
            # ion walk along the axis from where the path is open into where it narrows or ends, with no force,
            # onto the core's flat plateau and out through the far face: the prepore (10g) leaked 4 ions, and
            # 30 of open MscS's 43 and all 19 of open MscL's crossings left their path inside the core.
            drdz = -(dx * dV[:, 3] + dy * dV[:, 4]) / r
            # THE STEP ENDS AT rp (1 inside rp - er, 0 from rp out), so a path of zero radius is no path and the
            # energy is continuous as rp -> 0 -- centred on rp it left half a path on the axis and a jump of W/2
            # where the table's rp reached 0, which no force can see. The free radius is rp - er/2, stated.
            if self.er > 0:
                t = ((r - (rp - self.er)) / self.er).clamp(0.0, 1.0)
                gi = 0.5 * (1.0 + torch.cos(math.pi * t))
                dgi = torch.where((t > 0) & (t < 1), -0.5 * math.pi * torch.sin(math.pi * t) / self.er, torch.zeros_like(r))
            else:
                gi = (r < rp).to(r.dtype); dgi = torch.zeros_like(r)
            dgdz = dgi * (drdz - dV[:, 0])
            parts.append((gi, dgi, dx, dy, r, dgdz))
            outside_all = outside_all * (1.0 - gi)
            # THE DRY PLUG: U = W dry(z) L(r), L = 1 within R + margin of the path's centre
            Rl = R + self.plug_margin
            w = max(self.er, 1e-12)
            tl = ((r - (Rl - 0.5 * w)) / w).clamp(0.0, 1.0)
            Lr = 0.5 * (1.0 + torch.cos(math.pi * tl))
            dL = torch.where((tl > 0) & (tl < 1), -0.5 * math.pi * torch.sin(math.pi * tl) / w, torch.zeros_like(r))
            F[:, 2] += -self.W * (dV[:, 2] * Lr + dry * dL * (drdz - dV[:, 1]))
            Fr = -self.W * dry * dL
            F[:, 0] += Fr * dx / r; F[:, 1] += Fr * dy / r
        # THE CORE: U = W f(z) prod_k (1 - g_k(r_k))
        F[:, 2] += -self.W * outside_all * dfdz_abs * torch.sign(dz)
        for k, (gi, dgi, dx, dy, r, dgdz) in enumerate(parts):
            rest = torch.ones_like(gi)
            for j, (gj, _, _, _, _, _) in enumerate(parts):
                if j != k:
                    rest = rest * (1.0 - gj)
            Fr = self.W * f * dgi * rest                                  # -dU/dr = +W f g_k' prod_j(1-g_j)
            F[:, 0] += Fr * dx / r; F[:, 1] += Fr * dy / r
            F[:, 2] += self.W * f * dgdz * rest                           # -dU/dz through the path's slope
        fr = _frame(H)
        if fr in _REPORT:
            inside = (f > 0.5) & (outside_all > 0.5)
            print(f"[slab_barrier f{fr}] {self.at}: {int(inside.sum())} in the core outside every ion path "
                  f"({len(self.pores)} path(s); barrier {self.W:.4g} sim energy)", flush=True)
        V_ = self.mu * F
        h_ = float(getattr(H, "dt", 1.0))
        if self.step_max > 0:
            sp_ = V_.norm(dim=1, keepdim=True) * h_
            V_ = torch.where(sp_ > self.step_max, V_ * (self.step_max / sp_.clamp_min(1e-30)), V_)
        # THE CORE OUTSIDE EVERY PATH IS FORBIDDEN, as a constraint and not only as an energy: an ion found there is
        # returned to where it stood a frame before. The barrier's force alone, capped per frame against kicks
        # (`step_max`), was out-pushed by a crowd of ions in a dead-end lumen: at 1 M and -200 mV seven K+ went
        # down alpha-hemolysin's prepore stems, through the core below them, and out (batch 11, 11g) -- 99 kT
        # (a Boltzmann factor of 1e-43) is no energy an ion ever pays; the constraint says so.
        #
        # RETURNED TO ITS LAST ALLOWED PLACE, not to the frame before: `_prev` holds, per ion, the last position it
        # stood at OUTSIDE the forbidden core (its seed, for an ion seeded inside it). Returning to the frame before
        # was wrong whenever that position was itself forbidden: x(n+1) = x(n-1) + the other forces' step, so the
        # ion flipped between two places every frame, their distance random-walking to 4.4 nm (0.40 of the box) --
        # and when one of the two fell outside the core the ion was left there, 6.6 nm from where it stood one
        # record earlier (exp04 14i, the crystal K+ seeded into closed KcsA's filter, 2026-09-26). The records,
        # every 100 frames (an even number), saw one side of the flip only, so the ruler's per-record jump did not.
        banned = (f > 0.5) & (outside_all > 0.5)
        if self.hard and self._prev is not None and self._prev.shape == X.shape:
            back = (self._prev.to(X) - X) / h_
            V_ = torch.where(banned.unsqueeze(1), back, V_)
            self.n_returned += int(banned.sum())
            self._prev = torch.where(banned.unsqueeze(1), self._prev.to(X), X).detach().clone()
        else:
            self._prev = X.detach().clone()
        if fr in _REPORT and self.hard:
            print(f"[slab_barrier f{fr}] {self.at}: {self.n_returned} returns from the forbidden core so far", flush=True)
        return {self.at: V_ * lvl.occ[:, None]}

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        if self.path_from:
            ev = int(self.path_from.get("every", 100))
            if self.pores is None or _frame(H) % max(ev, 1) == 0:
                self.pores = [_path_from_beads(H, self.path_from, self.r_ion, self.z0)]
        if self.pores_o is not None and self.gate:
            w = float(_block(H.level(str(self.gate[0])), str(self.gate[1]))[0, 0])
            w = min(max(w if math.isfinite(w) else 0.0, 0.0), 1.0)
            if not (abs(w - self.w_open) < 1e-3):
                self.w_open = w
                self.pores = [(1.0 - w) * c_ + w * o_ for c_, o_ in zip(self.pores_c, self.pores_o)]
        if self.parents:
            Xl, c, s = self._in_channel_frame(H, X)
            V = self._forward_pores(H, lvl, Xl)[self.at]
            return {self.at: torch.stack([c * V[:, 0] - s * V[:, 1], s * V[:, 0] + c * V[:, 1], V[:, 2]], 1)}
        if self.pores is not None:
            return self._forward_pores(H, lvl, X)
        if self.lining and (_frame(H) % max(self.lin_every, 1) == 0 or _frame(H) <= 1):
            self._rp_now, axy = self._lining_radius(H)
            if axy is not None:
                self.axy = axy
            self.rp = self._rp_now
        dz = X[:, 2] - self.z0
        az = dz.abs()
        f, dfdz_abs = self._step(az, self.h, self.e)                      # 1 inside the core
        dx = X[:, 0] - self.axy[0]; dy = X[:, 1] - self.axy[1]
        r = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-12)
        if self.prof is not None:
            zp = self.prof[:, 0].to(X); rp_ = self.prof[:, 1].to(X)
            zq = X[:, 2].clamp(float(zp.min()), float(zp.max()))
            k_ = torch.searchsorted(zp.contiguous(), zq.contiguous()).clamp(1, zp.numel() - 1)
            t_ = (zq - zp[k_ - 1]) / (zp[k_] - zp[k_ - 1]).clamp_min(1e-12)
            rpz = rp_[k_ - 1] + t_ * (rp_[k_] - rp_[k_ - 1])
            inside_z = (X[:, 2] >= zp.min()) & (X[:, 2] <= zp.max())
            rpz = torch.where(inside_z, rpz, torch.full_like(rpz, self.rp))
            # the step about a per-particle radius: same cosine edge; its z-gradient is neglected (the
            # profile varies over ~0.3 nm against the barrier's 1.4 nm half-thickness)
            t2 = ((r - (rpz - 0.5 * self.er)) / max(self.er, 1e-12)).clamp(0.0, 1.0) if self.er > 0 else (r >= rpz).to(r.dtype)
            gi = 0.5 * (1.0 + torch.cos(math.pi * t2)) if self.er > 0 else (r < rpz).to(r.dtype)
            dgi = (torch.where((t2 > 0) & (t2 < 1), -0.5 * math.pi * torch.sin(math.pi * t2) / self.er, torch.zeros_like(r))
                   if self.er > 0 else torch.zeros_like(r))
        else:
            gi, dgi = self._step(r, self.rp, self.er)                     # 1 inside the pore
        g = 1.0 - gi; dg = -dgi
        F = torch.zeros_like(X)
        F[:, 2] = -self.W * g * dfdz_abs * torch.sign(dz)
        Fr = -self.W * f * dg
        F[:, 0] = Fr * dx / r; F[:, 1] = Fr * dy / r
        fr = _frame(H)
        if fr in _REPORT:
            inside = (f > 0.5) & (g > 0.5)
            print(f"[slab_barrier f{fr}] {self.at}: {int(inside.sum())} in the core outside the pore "
                  f"(barrier {self.W:.4g} sim energy" + (f", pore radius {self.rp:.4g} world from the lining"
                                                          if self.lining else "") + ")", flush=True)
        return {self.at: self.mu * F * lvl.occ[:, None]}


@register_operator("residue_slab", family="interaction", set="particle", kind="lateral",
                   title="Residues partitioning into a membrane's hydrocarbon core",
                   equation=r"""$$U=\sum_i \Delta g_i\,f(z_i),\qquad f=1\ \text{in the core, 0 in water}$$""")
class ResidueSlab(Lateral):
    """The hydrophobic effect, per residue: a residue in the membrane's core gains or pays its transfer
    free energy from water to a hydrocarbon-like interior.

    particle -> particle: reads pos and a per-bead block (`dg`, kT), emits a velocity along the normal.

        U = sum_i dg_i f(z_i),   f = 1 inside |z - z0| < h, 0 outside, cosine edges `edge` wide

    dg is the residue's side chain on the Wimley-White octanol scale (whole residue minus glycine,
    kcal/mol -> kT; Wimley, Creamer & White 1996, Biochemistry 35:5109): leucine -4.0 kT, isoleucine
    -3.8, valine -2.7 pull a residue into the core; aspartate +4.2, lysine +2.8 push it out -- the
    force that carries a pore-former's hairpin across a bilayer and leaves its charged turn on the far
    side. The generator writes dg from the deposited sequence. STATED LIMITATION: a residue facing a
    water-filled lumen inside the membrane (a barrel's polar face) is charged as if it faced the
    lipids; the drive is the side chain's, the backbone's hydrogen bonds are assumed made (a sheet).

    Reference: Wimley, W.C., Creamer, T.P. & White, S.H. (1996). Biochemistry 35:5109.
    """

    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["z0", "half_thickness", "block"]
    MECHANISM_TAGS = ["hydrophobic_effect", "membrane_insertion", "transfer_free_energy"]
    PARAM_ROLES = {"z0": "membrane_mid_plane_world", "half_thickness": "core_half_thickness_world",
                   "edge": "cosine_edge_width_world", "block": "per_bead_transfer_free_energy_block_sim_energy",
                   "sets": "the_protein_sets", "mobility": "mobility"}
    PARAM_UNITS = {"mobility": "mobility"}
    REFERENCE = "Wimley, W.C., Creamer, T.P. & White, S.H. (1996). Biochemistry 35:5109."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.sets = [str(v) for v in (params.get("sets") or [self.at])]
        self.z0 = float(params["z0"]); self.h = float(params["half_thickness"])
        self.e = float(params.get("edge", 0.0))
        self.block = str(params["block"])
        self.mu = float(params.get("mobility", 1.0))

    def forward(self, H, mask=None):
        out = {}
        for nm_ in self.sets:
            lvl = H.level(nm_)
            X = lvl.get("pos")
            dg = _block(lvl, self.block)
            if dg is None:
                continue
            dz = X[:, 2] - self.z0
            _, df = SlabBarrier._step(dz.abs(), self.h, self.e)
            v = torch.zeros_like(X)
            v[:, 2] = -self.mu * dg[:, 0] * df * torch.sign(dz)
            out[nm_] = v * lvl.occ[:, None]
        return out


@register_operator("friction_zone", family="motion", set="particle", kind="lateral",
                   title="Brownian motion with a slower mobility in a zone",
                   equation=r"""$$\Delta x = M(x)\,\Delta x_{F} + \sqrt{2M(x)\,\mu kT\,\Delta t}\,\xi + \mu kT\,\nabla M\,\Delta t$$""")
class FrictionZone(Lateral):
    """An ion's Brownian motion, with its mobility cut to M < 1 of the bulk's inside a zone.

    particle -> particle: REPLACES the set's `brownian` (none may run on it); runs FIRST in the schedule.
    At each frame it reads the displacement the forces produced in the last step (the set's positions
    now, less where it left them), and makes it the step of an overdamped particle whose mobility is
    M(x) times its own (Ito, with the spurious-drift term that keeps the Boltzmann distribution):

        dx = M(x) dx_forces + sqrt(2 M(x) mu kT dt) xi + mu kT grad M dt

    M = `factor` inside a cylinder of `radius` about the axis between heights `z` (world), 1 outside,
    with cosine edges `edge` wide. WHY: an ion in a potassium channel's filter moves in single file
    with the water between its sites, and Brownian-dynamics models of these channels take its
    diffusion coefficient there as ~0.1 of the bulk's (Chung, Allen, Hoyles & Kuyucak 1999, Biophys. J.
    77:2517; the molecular-dynamics estimates of Allen, Kuyucak & Chung 2000, Biophys. J. 79:2295) --
    not in the experiment's papers, stated. Without it the model's filter passed 150-640 pS against
    Kv1.2's 10-30 (exp04 rounds 6 and 8).

    Reference: Chung, S.H., Allen, T.W., Hoyles, M. & Kuyucak, S. (1999). Biophys. J. 77:2517.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["factor", "radius", "z"]
    MECHANISM_TAGS = ["brownian", "position_dependent_mobility", "single_file"]
    PARAM_ROLES = {"factor": "mobility_inside_as_a_fraction_of_bulk", "radius": "zone_radius_world",
                   "z": "zone_heights_world", "edge": "cosine_edge_world", "axis_xy": "the_axis_world",
                   "kT": "thermal_energy_sim", "mobility": "bulk_mobility", "seed": "rng_seed"}
    PARAM_UNITS = {"mobility": "mobility"}
    MAY_MUTATE_INTEGRATED_STATE = True
    REFERENCE = "Chung, S.H. et al. (1999). Biophys. J. 77:2517; Allen, T.W. et al. (2000). Biophys. J. 79:2295."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.f = float(params["factor"]); self.R = float(params["radius"])
        self.z0, self.z1 = [float(v) for v in params["z"]]
        self.e = float(params.get("edge", 0.0))
        self.axy = [float(v) for v in (params.get("axis_xy") or [0.5, 0.5])]
        self.kT = float(params.get("kT", 1.0)); self.mu = float(params.get("mobility", 1.0))
        self._gen = torch.Generator(device="cpu").manual_seed(int(params.get("seed", 7)))
        self._prev = None
        self.n_in = 0

    def _M(self, X):
        """M(x) and grad M: 1 - (1 - factor) g(r) h(z), g and h cosine steps."""
        dx = X[:, 0] - self.axy[0]; dy = X[:, 1] - self.axy[1]
        r = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-12)
        g, dg = SlabBarrier._step(r, self.R, self.e)                      # 1 inside the radius
        zc = 0.5 * (self.z0 + self.z1); hz = 0.5 * (self.z1 - self.z0)
        dz = X[:, 2] - zc
        h, dh = SlabBarrier._step(dz.abs(), hz, self.e)                   # 1 inside the heights
        a = 1.0 - self.f
        M = 1.0 - a * g * h
        G = torch.zeros_like(X)
        G[:, 0] = -a * dg * h * dx / r; G[:, 1] = -a * dg * h * dy / r
        G[:, 2] = -a * g * dh * torch.sign(dz)
        return M, G

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        h = float(getattr(H, "dt", 1.0))
        if self._prev is None or self._prev.shape != X.shape:
            self._prev = X.detach().clone()
            return {}
        Xp = self._prev
        M, G = self._M(Xp)
        dF = X - Xp                                                       # the forces' step (no brownian on this set)
        wrap = dF.abs().max(dim=1).values > 0.25                          # a wrap through the box (flux_counter)
        noise = torch.randn(X.shape, generator=self._gen, dtype=X.dtype).to(X.device)
        step = M[:, None] * dF + torch.sqrt(2.0 * M * self.mu * self.kT * h)[:, None] * noise + self.mu * self.kT * G * h
        Xn = torch.where(wrap[:, None], X + torch.sqrt(torch.tensor(2.0 * self.mu * self.kT * h, dtype=X.dtype)) * noise, Xp + step)
        a, b = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[:, a:b] = Xn.to(st.dtype); lvl.state = st
        self._prev = Xn.detach().clone()
        self.n_in = int((M < 0.5 * (1.0 + self.f)).sum())
        if _frame(H) in _REPORT:
            print(f"[friction_zone f{_frame(H)}] {self.at}: {self.n_in} inside the zone (mobility x{self.f:g} there)", flush=True)
        return {}


@register_operator("compartment_count", family="hierarchy", set="particle", kind="lateral",
                   title="Ions inside and outside the cell",
                   equation=r"""$$N^{s}_{\mathrm{in}}=\#\{i\in s: z_i<z_m\},\qquad I=\frac{d}{dt}\sum_s q_s\,e\,\big(N^{s}_{\mathrm{in}}(t)-N^{s}_{\mathrm{in}}(0)\big)$$""")
class CompartmentCount(Lateral):
    """A measurement, as an operator: how many ions of each species are inside the cell, and the current.

    ion sets -> cell: reads the sets' positions, writes the cell's blocks. Inside is below the membrane's
    mid-plane `z_m` (the cytoplasm is at -z in every exp04 spec). Started with N/2 of each species on
    each side of a membrane that only the channel (or a hole) lets them cross, the change of N_in IS the
    flow: the net number of each species that went through -- no plane to count crossings at, no
    wrap-around topping the baths up. The two compartments are closed; the voltage is held by the clamp
    (the electrolyte's face potentials), as a patch-clamp amplifier holds it.

        Q(t) = sum_s q_s (N_in^s(t) - N_in^s(0))     the charge carried INTO the cell, elementary charges
        I(t) = dQ/dt, a running mean over `window` frames, charges per sim time (+ = cations in)

    `sets` and `charges` are parallel lists; `blocks_in` names each species' count block, `block_charge`
    Q and `block_current` I (each optional).

    Reference: none -- a count, stated.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["z_m", "cell", "sets"]
    MECHANISM_TAGS = ["measurement", "ion_count", "flux"]
    PARAM_ROLES = {"z_m": "the_membrane_mid_plane_world", "cell": "set_receiving_the_counts",
                   "sets": "the_ion_sets", "charges": "their_charges_e", "blocks_in": "one_count_block_per_set",
                   "block_charge": "cell_block_for_the_charge_moved_in", "block_current": "cell_block_for_the_current",
                   "window": "running_mean_frames"}
    MAY_MUTATE_INTEGRATED_STATE = True          # a derived readout written into the cell's (non-integrated) blocks

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.z_m = float(params["z_m"]); self.cell = str(params["cell"])
        self.sets = [str(v) for v in params["sets"]]
        self.q = [float(v) for v in (params.get("charges") or [1.0] * len(self.sets))]
        self.b_in = list(params.get("blocks_in") or [None] * len(self.sets))
        self.b_q = params.get("block_charge"); self.b_i = params.get("block_current")
        self.window = float(params.get("window", 5000.0))
        self._n0 = None; self._Q_prev = 0.0; self._I = 0.0

    def forward(self, H, mask=None):
        n_in = []
        for s_ in self.sets:
            lv = H.level(s_)
            z = lv.get("pos")[:, 2]
            n_in.append(int(((z < self.z_m) & (lv.occ > 0)).sum()))
        if self._n0 is None:
            self._n0 = list(n_in)
        Q = sum(q * (n - n0) for q, n, n0 in zip(self.q, n_in, self._n0))
        h = float(getattr(H, "dt", 1.0))
        w = 1.0 - math.exp(-1.0 / max(self.window, 1.0))
        self._I += w * ((Q - self._Q_prev) / h - self._I)
        self._Q_prev = Q
        if self.cell in H.levels:
            cl = H.level(self.cell)
            one = torch.ones(cl.n, device=cl.state.device, dtype=cl.state.dtype)
            sl = cl.state_schema._slices
            for b_, n in zip(self.b_in, n_in):
                if b_ and b_ in sl:
                    _write(cl, b_, one * n)
            if self.b_q and self.b_q in sl:
                _write(cl, self.b_q, one * Q)
            if self.b_i and self.b_i in sl:
                _write(cl, self.b_i, one * self._I)
        fr = _frame(H)
        if fr in _REPORT:
            print(f"[compartment_count f{fr}] inside: " + ", ".join(f"{s_} {n} ({n - n0:+d})" for s_, n, n0 in
                                                                   zip(self.sets, n_in, self._n0))
                  + f"; charge moved into the cell {Q:+.0f} e", flush=True)
        return {}


@register_operator("radial_drive", family="boundary", set="particle", kind="lateral",
                   title="Patch frame under an applied tension",
                   equation=r"""$$\tau_{\mathrm{mem}}=-\frac{\sum_{k\in\mathrm{frame}}\mathbf F_k\cdot\hat{\mathbf r}_k}{2\pi R},\qquad \dot R=\mu_R\,2\pi R\,\big(\tau_{\mathrm{app}}(t)-\tau_{\mathrm{mem}}\big)$$""")
class RadialDrive(Lateral):
    """The patch pipette: a ring of frame beads that holds the membrane's edge and stretches it.

    frame -> frame (and -> cell for the readout): reads the frame set's `force` block (the pull
    of the membrane on each frame bead, written by `pair_potential`), writes the frame's
    positions and the cell's `tension` / `strain` blocks in place.

        tau_mem = - sum_k F_k . r_k / (2 pi R)            the tension the membrane actually carries
        mode tension:  dR/dt = mu_R 2 pi R (tau_app(t) - tau_mem)
        mode strain:   R = R0 (1 + strain(t))
        x_k = c + (x0_k - c)_inplane R / R0 + (x0_k - c)_normal

    tau_app(t) (or strain(t)) is `protocol`, a list of [frame, value] pairs linearly interpolated
    -- the experiment's own pressure protocol, like the suction ramp of a patch clamp. The tension
    is sim force per world; the generator converts mN/m. THE TENSION IS MEASURED, NOT IMPOSED: in
    tension mode the frame moves until the membrane's pull on it balances the applied value, and
    what the membrane carries is read from that pull, so a channel that opens and relieves the
    membrane shows up as a dip in `tau_mem` before the frame catches up.

    `axis` is the membrane normal (default z); `centre` the patch centre (default the frame's
    centroid at the first frame). `cell` names the set whose `tension` and `strain` blocks
    receive the readout.

    `affine_sets: [set, ...]` -- THE STRETCH REACHES THE WHOLE PATCH AT ONCE, as a surface-tension
    barostat applies it in molecular dynamics (Berendsen et al. 1984; Zhang, Feller, Brooks &
    Pastor 1995): each frame, every bead of these sets (one bead per lipid, so this is the
    MOLECULAR scaling of a barostat) has its in-plane offset from the centre multiplied by the
    frame's own radius ratio R(t) / R(t - 1). The protein is not scaled: it is pulled open, or
    not, by the lipids around it. Without it the frame moves only the patch's EDGE and the strain
    reaches the channel by diffusion through the lipid layer in ~ zeta n R^2 / K_A: exp04 round 3
    measured that relaxation time at ~580,000 frames against a 120,000-frame run, and lipids
    2.5-4 nm from the axis gained 8-23% in area per lipid while those at 4-7 nm gained 36-53%.

    Reference: Sachs, F. (2010). Physiology 25:50 (stretch-activated channels and the patch);
    Sukharev, S. et al. (1999). Biophys. J. 76:2382 (pressure -> tension in a patch, Laplace).
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["protocol"]
    MECHANISM_TAGS = ["membrane_tension", "patch_clamp", "boundary_condition"]
    PARAM_ROLES = {"protocol": "list_of_frame_value_pairs", "mode": "tension_or_strain",
                   "mobility_R": "frame_radius_mobility", "cell": "set_receiving_tension_and_strain",
                   "axis": "membrane_normal_axis", "centre": "patch_centre_world",
                   "avg_frames": "running_mean_of_the_measured_tension_in_frames", "max_step": "guard_on_R_per_frame",
                   "affine_sets": "sets_scaled_in_plane_with_the_frame_each_frame_the_barostat"}
    PARAM_UNITS = {"mobility_R": "mobility"}
    MAY_MUTATE_INTEGRATED_STATE = True
    REFERENCE = ("Sachs, F. (2010). Physiology 25:50; Sukharev, S. et al. (1999). Biophys. J. 76:2382; "
                 "Berendsen, H.J.C. et al. (1984). J. Chem. Phys. 81:3684; Zhang, Y. et al. (1995). J. Chem. Phys. 103:10252.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "frame")
        self.protocol = [[float(a), float(b)] for a, b in params["protocol"]]
        self.mode = str(params.get("mode", "tension")).lower()
        if self.mode not in ("tension", "strain"):
            raise ValueError(f"radial_drive: mode is tension or strain, got {self.mode!r}")
        self.mu_R = float(params.get("mobility_R", 1e-4))
        self.cell = params.get("cell", "cell")
        self.axis = int(params.get("axis", 2))
        self.centre = params.get("centre")
        self.max_step = float(params.get("max_step", 1e-4))       # world per frame, a guard on R
        # `avg_frames`: the membrane's tension as an exponential running mean over this many frames.
        # One frame's pull on the frame beads is a sum of ~10^3 thermal pair forces and swings by tens
        # of mN/m (render smoke test, 2026-09-25); the mean is what the membrane carries, and it is
        # both what the frame answers to and what the curve shows.
        self.avg = float(params.get("avg_frames", 0.0))
        self.affine = [str(v) for v in (params.get("affine_sets") or [])]
        self._tau_bar = None
        self._x0 = None

    def _value(self, fr):
        p = self.protocol
        if fr <= p[0][0]:
            return p[0][1]
        for (f0, v0), (f1, v1) in zip(p, p[1:]):
            if fr <= f1:
                return v0 + (v1 - v0) * (fr - f0) / max(f1 - f0, 1e-9)
        return p[-1][1]

    def forward(self, H, mask=None):
        fl = H.level(self.at)
        X = fl.get("pos")
        if self._x0 is None:
            self._x0 = X.detach().clone()
            c = torch.tensor([float(v) for v in self.centre], device=X.device, dtype=X.dtype) \
                if self.centre is not None else self._x0.mean(0)
            self._c = c
            rel = self._x0 - c
            rel[:, self.axis] = 0.0
            self._r_hat = rel / rel.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self._R0 = float(rel.norm(dim=1).mean())
            self._R = self._R0
        F = _block(fl, "force")
        Fr = (F * self._r_hat).sum(1) if F is not None else torch.zeros(X.shape[0], device=X.device)
        tau_now = -float(Fr.sum()) / (2.0 * math.pi * self._R)
        if self.avg > 0:
            w = 1.0 - math.exp(-1.0 / self.avg)
            # FROM ZERO, not from the first reading: the seeded lattice's first-frame pull (tens of mN/m,
            # gone in ~100 frames) otherwise sat in a 2,000-frame mean for ~5,000 frames -- a bias of
            # ~8 mN/m still there when the ramp began (MscL crash check, 2026-09-25).
            self._tau_bar = (0.0 if self._tau_bar is None else self._tau_bar) + w * (tau_now - (0.0 if self._tau_bar is None else self._tau_bar))
            tau_mem = self._tau_bar
        else:
            tau_mem = tau_now
        fr = _frame(H)
        val = self._value(fr)
        h = float(getattr(H, "dt", 1.0))
        R_prev = self._R
        if self.mode == "tension":
            dR = h * self.mu_R * 2.0 * math.pi * self._R * (val - tau_mem)
            self._R += max(-self.max_step, min(self.max_step, dR))
        else:
            self._R = self._R0 * (1.0 + val)
        if self.affine and self._R != R_prev:
            s_ = self._R / R_prev
            for nm_ in self.affine:
                if nm_ not in H.levels:
                    continue
                lv = H.level(nm_)
                Y = lv.get("pos")
                d = Y - self._c.to(Y.dtype)
                d_n = torch.zeros_like(d)
                d_n[:, self.axis] = d[:, self.axis]
                Yn = self._c.to(Y.dtype) + (d - d_n) * s_ + d_n
                a_, b_ = lv.state_schema["pos"]
                st_ = lv.state.clone(); st_[:, a_:b_] = Yn.to(st_.dtype); lv.state = st_
        rel = self._x0 - self._c
        normal = torch.zeros_like(rel)
        normal[:, self.axis] = rel[:, self.axis]
        inplane = rel - normal
        newX = self._c + inplane * (self._R / self._R0) + normal
        a, b = fl.state_schema["pos"]
        st = fl.state.clone(); st[:, a:b] = newX.to(st.dtype); fl.state = st
        strain = (self._R / self._R0) ** 2 - 1.0                   # area strain of the patch
        if self.cell and self.cell in H.levels:
            cl = H.level(self.cell)
            one = torch.ones(cl.n, device=X.device, dtype=cl.state.dtype)
            if "tension" in cl.state_schema._slices:
                _write(cl, "tension", one * tau_mem)
            if "tension_applied" in cl.state_schema._slices:
                _write(cl, "tension_applied", one * (val if self.mode == "tension" else tau_mem))
            if "strain" in cl.state_schema._slices:
                _write(cl, "strain", one * strain)
        if fr in _REPORT:
            print(f"[radial_drive f{fr}] {self.mode} {val:.4g} applied; membrane carries {tau_mem:.4g} (sim tension); "
                  f"R {self._R:.5f} world (R0 {self._R0:.5f}), area strain {strain:.4f}", flush=True)
        return {}


# =============================================================================================
# THE ELECTROLYTE, AND THE CURRENT THROUGH WHAT THE PROTEIN LEAVES OPEN
# =============================================================================================
@register_operator("electrolyte_conduction", family="fields", set="particle", kind="exchange",
                   title="Ionic current through the electrolyte",
                   equation=r"""$$\nabla\!\cdot\!\big(\sigma(\mathbf x)\nabla\phi\big)=0,\quad \phi\big|_{\mathrm{in}}-\phi\big|_{\mathrm{out}}=\psi,\qquad I=\int_{z=z_m}\sigma\,\partial_z\phi\,dA$$""")
class ElectrolyteConduction(Exchange):
    """The bath as a conductor: the ionic current through whatever the protein leaves open.

    sets -> field, field -> cell: reads the positions of the insulating sets (protein and lipid
    beads, each with an exclusion radius), the frame's radius and the cell's potential `psi`;
    solves for the electrolyte potential on a grid field; writes the current and the conductance
    into the cell's blocks and the potential and current density into the field's two channels.

        sigma(x) = sigma_bulk  where no insulating bead is within its radius of the voxel
                   0           inside a bead, and in the membrane's slab outside the patch seal
        div(sigma grad phi) = 0,  phi = +psi/2 on the bottom face (cytoplasm), -psi/2 on the top
        I = sum over the membrane's mid-plane of sigma_face (phi_below - phi_above) / dx * dx^2

    Solved by red-black successive over-relaxation, warm-started from the last solve, every
    `every` frames. THE CONDUCTANCE IS NOT A PARAMETER: it is whatever the geometry of the open
    lumen gives in a medium of conductivity sigma_bulk -- the access resistance at each mouth
    and the resistance along the pore both come out of the field (Hille's formula,
    R = L / (sigma pi r^2) + 1 / (2 sigma r), is the baseline it is to be checked against, and
    lives in the analysis, not here).

    THIS IS OHM'S LAW IN THE BATH, the zero mode of Poisson-Nernst-Planck for a symmetric salt
    at uniform concentration. STATED LIMITATIONS: no selectivity (a K+ channel is not a
    continuum), no dewetting of a hydrophobic gate wider than a water molecule (such a pore
    conducts here in proportion to its geometric opening), and an exclusion radius per bead that
    stands in for the van der Waals envelope of the residue an alpha carbon carries.

    `probe_radius` (world) -- THE ION-ACCESSIBLE VOLUME, as a Poisson-Boltzmann solver draws it:
    an ion's centre comes no closer to a bead than the bead's radius PLUS its own (K+ 0.133 nm,
    water 0.14 nm), so a voxel conducts only if a sphere of `probe_radius` centred on it touches
    no bead. Without it a gate narrower than a water molecule still conducted through a single
    column of voxels: exp04 round 3 read a 0-to-6 pA telegraph (0.06 nS) through MscL gates whose
    free diameter never exceeded 0.175 nm, as often at zero tension as at 15 mN/m. Default 0.

    Units: `sigma` in S/m; `volt_per_sim` converts the cell's psi (sim e-psi) to volts;
    `dx_m` is one grid voxel in metres; `sim_current_per_A` converts amperes to the spec's
    current unit (elementary charges per sim time). A one-channel field holds the current density
    |j| in pA/nm^2 (what the movie draws); with two channels, channel 0 is phi in mV and channel 1
    is |j|. `lumen_radius` (world) with `axis_sets`: inside the membrane's slab current passes only
    within that distance of the channel axis -- the bilayer's core insulates by physics, and two
    leaflets of beads leave a layer at its centre that bead exclusion alone does not close.

    Reference: Hille, B. (2001). Ion Channels of Excitable Membranes, 3rd ed., ch. 11 (access and
    pore resistance); Eisenberg, R.S. (1996). J. Membr. Biol. 150:1 (PNP); Hall, J.E. (1975).
    J. Gen. Physiol. 66:531 (access resistance).
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    INPUTS = ["particle", "cell"]
    OUTPUTS = ["field", "cell"]
    REQUIRES_PARAMS = ["field", "insulators", "sigma", "volt_per_sim", "dx_m", "sim_current_per_A"]
    MECHANISM_TAGS = ["ionic_current", "ohm", "access_resistance", "electrolyte"]
    PARAM_ROLES = {"field": "grid_field_2_channels", "insulators": "list_of_set_and_radius",
                   "sigma": "bulk_conductivity_S_per_m", "slab": "membrane_slab_z_world_lo_hi",
                   "seal": "set_whose_radius_seals_the_slab", "cell": "set_with_psi_and_current_blocks",
                   "every": "frames_between_solves", "iters": "sor_sweeps_per_solve",
                   "iters_first": "sor_sweeps_on_the_first_solve", "omega": "sor_relaxation",
                   "lumen_radius": "slab_conducts_only_within_this_of_the_axis_world",
                   "lining": "block_and_value_of_the_pore_lining_beads_whose_ring_bounds_the_lumen",
                   "lumen_core": "radius_and_z_window_of_the_ion_path_kept_as_electrolyte",
                   "axis_sets": "sets_whose_centroid_is_the_channel_axis",
                   "probe_radius": "ion_radius_added_to_every_insulator_world",
                   "lumens": "ion_path_profiles_z_rp_R_dry_cx_cy_world_the_slab_conducts_only_inside",
                   "path_from": "the_lumen_measured_on_moving_beads_at_every_solve", "path_z0": "membrane_mid_plane_world",
                   "lumens_open": "the_open_states_lumens_same_heights", "gate": "set_and_block_holding_the_open_weight"}
    MAY_MUTATE_INTEGRATED_STATE = True
    REFERENCE = ("Hille, B. (2001). Ion Channels of Excitable Membranes, ch. 11; Hall, J.E. (1975). "
                 "J. Gen. Physiol. 66:531; Eisenberg, R.S. (1996). J. Membr. Biol. 150:1.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field = str(params["field"])
        self.ins = [(str(d["set"]), float(d["radius"])) for d in params["insulators"]]
        self.probe = float(params.get("probe_radius", 0.0))        # the ion's own radius, world
        self.sigma = float(params["sigma"])
        self.vps = float(params["volt_per_sim"])
        self.dx_m = float(params["dx_m"])
        self.i_conv = float(params["sim_current_per_A"])
        self.slab = [float(v) for v in (params.get("slab") or [0.45, 0.55])]
        self.seal = params.get("seal")
        self.seal_margin = float(params.get("seal_margin", 0.0))
        # THE LUMEN: inside the membrane's slab, current may pass only within `lumen_radius` (world)
        # of the channel axis -- the axis through the centroid of `axis_sets`. The bilayer is an
        # insulator by physics (its core is 3 nm of hydrocarbon), and two leaflets of lipid beads
        # leave a layer at the bilayer's centre that a bead exclusion alone does not close; what
        # sets the CONSTRICTION inside the lumen is still the protein's beads.
        self.lumen = params.get("lumen_radius")
        self.lumen = float(self.lumen) if self.lumen is not None else None
        self.axis_sets = [str(s) for s in (params.get("axis_sets") or [])]
        # `lining: {block: domain, value: k}` -- A LUMEN THAT OPENS WITH THE PORE. The radius is not a
        # number but the median distance from the axis of the pore-lining beads (those of the
        # `axis_sets` whose `block` equals `value`) inside the slab, re-read at every solve: the
        # current may pass only inside the ring the lining helices draw, so the gaps an alpha-carbon
        # model leaves BETWEEN helices do not conduct (the leak of the smoke test, 2026-09-25: a
        # closed synthetic ring conducting 1.19 nS through a fixed 2.1 nm cylinder), and when the
        # helices move out the lumen moves out with them. Inside it, the constriction is the beads'.
        self.lining = params.get("lining")
        self.lumen_now = float("nan")
        # `lumen_core: {radius, z: [z0, z1]}` -- THE ION PATH IS ELECTROLYTE. A potassium channel's
        # filter is 0.15 nm across: an alpha-carbon model with a residue's 0.5 nm reach fills it, and
        # the solve would then leave the voltage to fall linearly across the whole box instead of
        # across the filter. Voxels within `radius` of the axis in the z window are water whatever
        # beads are near -- the fact that ions pass there, not the answer to where the drop is.
        self.core = params.get("lumen_core")
        self.lumens = ([torch.tensor([[float(v) for v in row] for row in L_["profile"]], dtype=torch.float64)
                        for L_ in params["lumens"]] if params.get("lumens") else None)
        self.path_from = params.get("path_from")                 # the lumen re-measured on the moving protein
        self.path_z0 = float(params.get("path_z0", 0.5))
        self.lumens_c = self.lumens
        self.lumens_o = ([torch.tensor([[float(v) for v in row] for row in L_["profile"]], dtype=torch.float64)
                          for L_ in params["lumens_open"]] if params.get("lumens_open") else None)
        self.gate = params.get("gate")
        self.verbose = bool(params.get("verbose", False))
        self._dump = params.get("dump")                              # a debugging path; off by default
        self._cache = None
        self.cell = str(params.get("cell", "cell"))
        self.every = int(params.get("every", 10))
        self.iters = int(params.get("iters", 60))
        self.iters_first = int(params.get("iters_first", 3000))
        # GAUSS-SEIDEL, NOT OVER-RELAXED, BY DEFAULT. Red-black SOR at omega 1.9 converges on one
        # geometry, but a run changes the insulating mask a little between solves and continues each
        # solve from the last field with a few dozen sweeps -- a SEQUENCE of different, strongly
        # non-normal iteration matrices, which amplified the error without bound: exp04 round 2's
        # potential reached 3.8e6 V in 906,460 voxels at frame 5,500 and its membrane potential blew up
        # after. At omega 1 every update is a weighted average of neighbours and cannot leave
        # [-V/2, V/2]; the field is also clamped there after every solve (the maximum principle the
        # exact solution obeys), as a guard.
        self.omega = float(params.get("omega", 1.0))
        self._phi = None
        self._I = 0.0
        self._n = 0

    def _mask(self, H, g, shape, dx):
        """sigma / sigma_bulk on the voxel centres: 1 in water, 0 in an insulator."""
        nx, ny, nz = shape
        dev = g.device
        occ = torch.zeros(shape, device=dev, dtype=torch.bool)
        for name, rad in self.ins:
            rad = rad + self.probe                                    # the ion-accessible reach
            X = H.level(name).get("pos").detach()
            live = H.level(name).occ > 0
            X = X[live]
            k = int(math.ceil(rad / dx))
            ax = torch.arange(-k, k + 1, device=dev)
            off = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
            off = off[((off.to(X.dtype) * dx) ** 2).sum(1) <= (rad + 0.5 * dx) ** 2]
            base = torch.floor(X / dx).long()                         # the voxel each bead sits in
            idx = base[:, None, :] + off[None, :, :]
            # a voxel is inside when its CENTRE is within `rad` of the bead
            ctr = (idx.to(X.dtype) + 0.5) * dx
            inside = ((ctr - X[:, None, :]) ** 2).sum(-1) <= rad * rad
            idx = idx[inside]
            ok = (idx >= 0).all(1) & (idx[:, 0] < nx) & (idx[:, 1] < ny) & (idx[:, 2] < nz)
            idx = idx[ok]
            occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        # THE SEAL: in the membrane's slab, outside the patch, the membrane goes on (the rest of the
        # cell, or the pipette's glass) and carries no current -- without it the box's edge is a leak.
        z0, z1 = int(self.slab[0] / dx), int(math.ceil(self.slab[1] / dx))
        if (self.lumen is not None or self.lining) and self.axis_sets:
            A = torch.cat([H.level(s).get("pos").detach() for s in self.axis_sets])
            c = A.mean(0)
            R_l = self.lumen
            if self.lining:
                blk = str(self.lining.get("block", "domain")); val = float(self.lining.get("value", 0))
                m = torch.cat([(_block(H.level(s), blk)[:, 0] - val).abs() < 0.5 for s in self.axis_sets])
                zc = A[:, 2]
                m = m & (zc > self.slab[0]) & (zc < self.slab[1])
                if bool(m.any()):
                    R_l = float((A[m, :2] - c[:2]).norm(dim=1).median())
                else:
                    # NO LINING IN THE MEMBRANE, NO LUMEN: a pore-former whose stems have not yet entered
                    # the bilayer leaves it whole (alpha-hemolysin's detached prestems); without this the
                    # lumen fell back to unbounded and the gaps of the lipid layer carried 0.29 nS.
                    R_l = 0.0
            self.lumen_now = R_l if R_l is not None else float("nan")
            xs = (torch.arange(nx, device=dev, dtype=A.dtype) + 0.5) * dx - c[0]
            ys = (torch.arange(ny, device=dev, dtype=A.dtype) + 0.5) * dx - c[1]
            rr = (xs[:, None] ** 2 + ys[None, :] ** 2).sqrt()
            if R_l is not None:
                occ[:, :, z0:z1] |= (rr > R_l)[:, :, None]
        if self.path_from:
            self.lumens = [_path_from_beads(H, self.path_from, 0.0, self.path_z0)]
        if self.lumens_o is not None and self.gate:
            w = float(_block(H.level(str(self.gate[0])), str(self.gate[1]))[0, 0])
            w = min(max(w if math.isfinite(w) else 0.0, 0.0), 1.0)
            self.lumens = [(1.0 - w) * c_ + w * o_ for c_, o_ in zip(self.lumens_c, self.lumens_o)]
        if self.lumens:
            # THE SAME ION PATHS AS THE BORN BARRIER'S (`lumens: [{profile: [[z, rp, R, dry, cx, cy]]}]`): in the
            # slab the electrolyte is only inside some path's lumen R(z); a DRY segment insulates its lumen at
            # any height -- the continuum's hydrophobic gate, which bead exclusion alone cannot close (the
            # stated limitation above, removed where a deposited structure gives the path).
            xs = (torch.arange(nx, device=dev, dtype=torch.float64) + 0.5) * dx
            ys = (torch.arange(ny, device=dev, dtype=torch.float64) + 0.5) * dx
            zk = (torch.arange(nz, device=dev, dtype=torch.float64) + 0.5) * dx
            allowed = torch.zeros(nx, ny, nz, dtype=torch.bool, device=dev)
            for P in self.lumens:
                P = P.to(dev)
                V, _ = SlabBarrier._interp(P, zk)
                for k in range(nz):
                    R_, dry, cx, cy = float(V[k, 1]), float(V[k, 2]), float(V[k, 3]), float(V[k, 4])
                    if R_ <= 0:
                        continue
                    d = ((xs[:, None] - cx) ** 2 + (ys[None, :] - cy) ** 2).sqrt()
                    allowed[:, :, k] |= d <= R_
                    if dry > 0.5:
                        occ[:, :, k] |= d <= R_
            occ[:, :, z0:z1] |= ~allowed[:, :, z0:z1]
        if self.core and self.axis_sets:
            A = torch.cat([H.level(s).get("pos").detach() for s in self.axis_sets])
            c = A.mean(0)
            rc = float(self.core["radius"]); zc0, zc1 = [float(v) for v in self.core["z"]]
            xs = (torch.arange(nx, device=dev, dtype=A.dtype) + 0.5) * dx - c[0]
            ys = (torch.arange(ny, device=dev, dtype=A.dtype) + 0.5) * dx - c[1]
            rr = (xs[:, None] ** 2 + ys[None, :] ** 2).sqrt()
            k0, k1 = int(zc0 / dx), int(math.ceil(zc1 / dx))
            occ[:, :, k0:k1] &= ~(rr <= max(rc, 0.5 * dx))[:, :, None]
        if self.seal and self.seal in H.levels:
            F = H.level(self.seal).get("pos").detach()
            c = F.mean(0)
            R = float((F[:, :2] - c[:2]).norm(dim=1).min()) - self.seal_margin
            xs = (torch.arange(nx, device=dev, dtype=F.dtype) + 0.5) * dx - c[0]
            ys = (torch.arange(ny, device=dev, dtype=F.dtype) + 0.5) * dx - c[1]
            rr = (xs[:, None] ** 2 + ys[None, :] ** 2).sqrt()
            outside = rr >= R
            occ[:, :, z0:z1] |= outside[:, :, None]
        return (~occ).to(torch.float32)

    def _solve(self, s, V, iters):
        """Red-black SOR for div(s grad phi) = 0, phi = +V/2 at z = 0, -V/2 at z = nz-1."""
        phi = self._phi
        nx, ny, nz = phi.shape
        # face conductances (harmonic mean), with no-flux x/y walls (face 0 on the outer side)
        def hm(a, b):
            return 2.0 * a * b / (a + b).clamp_min(1e-12)
        sxp = torch.zeros_like(s); sxp[:-1] = hm(s[:-1], s[1:])
        sxm = torch.zeros_like(s); sxm[1:] = sxp[:-1]
        syp = torch.zeros_like(s); syp[:, :-1] = hm(s[:, :-1], s[:, 1:])
        sym = torch.zeros_like(s); sym[:, 1:] = syp[:, :-1]
        szp = torch.zeros_like(s); szp[:, :, :-1] = hm(s[:, :, :-1], s[:, :, 1:])
        szm = torch.zeros_like(s); szm[:, :, 1:] = szp[:, :, :-1]
        den = sxp + sxm + syp + sym + szp + szm
        solid = den <= 1e-12
        den = den.clamp_min(1e-12)
        if self._cache is None or self._cache[0] != (nx, ny, nz):
            ii, jj, kk = torch.meshgrid(torch.arange(nx, device=s.device), torch.arange(ny, device=s.device),
                                        torch.arange(nz, device=s.device), indexing="ij")
            red = ((ii + jj + kk) % 2 == 0)
            interior = torch.ones_like(red); interior[:, :, 0] = False; interior[:, :, -1] = False
            self._cache = ((nx, ny, nz), red & interior, (~red) & interior)
        _, red_in, black_in = self._cache
        phi[:, :, 0] = 0.5 * V
        phi[:, :, -1] = -0.5 * V
        for _ in range(iters):
            for colour in (red_in, black_in):
                nb = torch.zeros_like(phi)
                nb[:-1] += sxp[:-1] * phi[1:]
                nb[1:] += sxm[1:] * phi[:-1]
                nb[:, :-1] += syp[:, :-1] * phi[:, 1:]
                nb[:, 1:] += sym[:, 1:] * phi[:, :-1]
                nb[:, :, :-1] += szp[:, :, :-1] * phi[:, :, 1:]
                nb[:, :, 1:] += szm[:, :, 1:] * phi[:, :, :-1]
                new = (1.0 - self.omega) * phi + self.omega * nb / den
                upd = colour & ~solid
                phi = torch.where(upd, new, phi)
        half = 0.5 * abs(V)
        phi = phi.clamp(-half, half)
        self._phi = phi
        self._sxp, self._syp = sxp, syp
        return szp

    def forward(self, H, mask=None):
        fr = _frame(H)
        fld = H.fields[self.field]
        g = fld.grid
        shape = tuple(g.shape[1:])
        dx = 1.0 / shape[0]                                          # world per voxel (the box is 1)
        cl = H.level(self.cell)
        psi_sim = float(cl.get("psi")[:, 0].mean()) if "psi" in cl.state_schema._slices else 0.0
        V = psi_sim * self.vps                                       # volts across the box
        if self._phi is None or (fr - self._n) >= self.every:
            first = self._phi is None
            s = self._mask(H, g, shape, dx)
            if first:
                nz = shape[2]
                zz = torch.linspace(0.5, -0.5, nz, device=g.device)
                self._phi = (zz[None, None, :] * V).expand(shape).clone().to(torch.float32)
            szp = self._solve(s, V, self.iters_first if first else self.iters)
            # THE CURRENT THROUGH THE MEMBRANE'S MID-PLANE: sigma dphi/dz summed over the plane.
            km = int(0.5 * (self.slab[0] + self.slab[1]) / dx)
            dphi = self._phi[:, :, km] - self._phi[:, :, km + 1]      # volts, below minus above
            I = float((szp[:, :, km] * dphi).sum()) * self.sigma * self.dx_m   # sigma dphi/dx dx^2 = sigma dphi dx
            if getattr(self, "_dump", None) and V and abs(I / V) * 1e9 > 3.0 and not getattr(self, "_dumped", False):
                import numpy as _np
                _np.savez(self._dump, phi=self._phi.detach().cpu().numpy(), s=s.detach().cpu().numpy(),
                          szp=szp.detach().cpu().numpy(), frame=fr, I=I, V=V, km=km)
                self._dumped = True
                print(f"[electrolyte_conduction f{fr}] ABNORMAL conductance {abs(I / V) * 1e9:.3f} nS: fields dumped to {self._dump}", flush=True)
            self._I = I
            self._n = fr
            # the field's two channels, for the movie: phi in mV, |j| in pA/nm^2
            # |j| FROM THE FACE FLUXES, each face carrying sigma_face (phi_i - phi_i+1) / dx and the
            # voxel the mean of its two faces per axis. A centred difference straddles an insulator
            # -- the bilayer's slab carries the whole voltage drop inside it -- and drew two bright
            # sheets of "current" along the membrane's faces where no charge moves (render smoke
            # test, 2026-09-25); a face with zero conductance carries zero flux, which is the physics.
            phi = self._phi
            fx = torch.zeros_like(phi); fx[:-1] = self._sxp[:-1] * (phi[:-1] - phi[1:])
            fy = torch.zeros_like(phi); fy[:, :-1] = self._syp[:, :-1] * (phi[:, :-1] - phi[:, 1:])
            fz = torch.zeros_like(phi); fz[:, :, :-1] = szp[:, :, :-1] * (phi[:, :, :-1] - phi[:, :, 1:])
            jx = fx.clone(); jx[1:] = 0.5 * (fx[1:] + fx[:-1])
            jy = fy.clone(); jy[:, 1:] = 0.5 * (fy[:, 1:] + fy[:, :-1])
            jz = fz.clone(); jz[:, :, 1:] = 0.5 * (fz[:, :, 1:] + fz[:, :, :-1])
            jmag = self.sigma * (jx * jx + jy * jy + jz * jz).sqrt() / self.dx_m          # A/m^2
            out = g.clone()
            if out.shape[0] == 1:                                    # one channel: the current density, what the movie shows
                out[0] = jmag * 1e12 / 1e18                          # pA per nm^2
            else:
                out[0] = phi * 1e3
                out[1] = jmag * 1e12 / 1e18
            fld.grid = out
            if first or fr in _REPORT or self.verbose:
                water = float(s[:, :, int(self.slab[0] / dx):int(math.ceil(self.slab[1] / dx))].mean())
                print(f"[electrolyte_conduction f{fr}] psi {V * 1e3:.1f} mV; current {I * 1e12:.3f} pA "
                      f"-> conductance {abs(I / V) * 1e9 if V else 0.0:.4f} nS; water in the membrane slab "
                      f"{water * 100:.3f}%; lumen radius {self.lumen_now:.4f} world", flush=True)
        one = torch.ones(cl.n, device=g.device, dtype=cl.state.dtype)
        if "j_channel" in cl.state_schema._slices:
            _write(cl, "j_channel", one * (self._I * self.i_conv))
        if "g_channel" in cl.state_schema._slices:
            _write(cl, "g_channel", one * (abs(self._I / V) * 1e12 if V else 0.0))   # pS, a plain number
        return {}


@register_operator("pore_probe", family="hierarchy", set="particle", kind="lateral",
                   title="Radius of the channel's narrowest lumen",
                   equation=r"""$$r_{\mathrm{pore}}=\min_{z\in[z_0,z_1]}\;\min_{i:\,|z_i-z|<h}\big(\rho_i-a\big)$$""")
class PoreProbe(Lateral):
    """A measurement, as an operator: the radius of the narrowest free lumen along the axis.

    protein sets -> cell: reads the positions of the protein's sets, writes the cell's `pore_r`.

        rho_i = distance of bead i from the channel axis (through the protein's centroid, along z)
        r(z)  = min over beads within h of height z of (rho_i - a)      the free radius at z
        r_pore = min over z in [z0, z1] of r(z)                          the constriction

    `a` (`bead_radius`, world) is the van der Waals reach of the residue an alpha carbon stands
    for, the same number `electrolyte_conduction` excludes; `sets` the protein's sets (default the
    `at` set); `z` the height window [z0, z1] (world) the pore spans; `h` the slice half-thickness.
    Heights with no bead are skipped. Nothing reads this but the curves and the ruler.

    Reference: Smart, O.S. et al. (1996). J. Mol. Graph. 14:354 (HOLE, the same question asked of an
    atomic model).
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["bead_radius", "z"]
    MECHANISM_TAGS = ["measurement", "pore_radius"]
    PARAM_ROLES = {"bead_radius": "residue_reach_world", "z": "height_window_world", "h": "slice_half_thickness",
                   "cell": "set_receiving_pore_r", "slices": "number_of_heights", "sets": "the_protein_sets"}
    PARAM_UNITS = {"bead_radius": "length", "h": "length"}
    MAY_MUTATE_INTEGRATED_STATE = True
    REFERENCE = "Smart, O.S. et al. (1996). J. Mol. Graph. 14:354."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.sets = [str(s) for s in (params.get("sets") or [self.at])]
        self.a = float(params["bead_radius"])
        self.z = [float(v) for v in params["z"]]
        self.h = float(params.get("h", 0.01))
        self.n = int(params.get("slices", 24))
        self.cell = str(params.get("cell", "cell"))

    def forward(self, H, mask=None):
        X = torch.cat([H.level(s).get("pos").detach() for s in self.sets])
        c = X.mean(0)
        rho = (X[:, :2] - c[:2]).norm(dim=1) - self.a
        zs = torch.linspace(self.z[0], self.z[1], self.n, device=X.device, dtype=X.dtype)
        near = (X[None, :, 2] - zs[:, None]).abs() < self.h                    # [slices, n]
        inf = torch.full((self.n, rho.shape[0]), float("inf"), device=X.device, dtype=X.dtype)
        r_z = torch.where(near, rho[None, :].expand(self.n, -1), inf).min(1).values   # +inf where no bead
        r_pore = float(r_z.min().clamp_min(0.0)) if torch.isfinite(r_z).any() else 0.0
        cl = H.level(self.cell)
        if "pore_r" in cl.state_schema._slices:
            _write(cl, "pore_r", torch.full((cl.n,), r_pore, device=X.device, dtype=cl.state.dtype))
        fr = _frame(H)
        if fr in _REPORT:
            print(f"[pore_probe f{fr}] narrowest free radius {r_pore:.5f} world at z "
                  f"{float(zs[int(r_z.argmin())]):.4f}", flush=True)
        return {}


# =============================================================================================
# IONS: THE FIELD THAT DRIVES THEM, AND THE CURRENT THEY CARRY
# =============================================================================================
@register_operator("field_force", family="coupling", set="particle", kind="exchange",
                   title="Electric force on charged beads",
                   equation=r"""$$\dot{\mathbf x}_i=\mu\,q_i\,\big(-\nabla\phi(\mathbf x_i)\big)$$""")
class FieldForce(Exchange):
    """The electrolyte's own potential pushing the charges that sit in it.

    field -> set: reads a grid field's potential channel (mV, as `electrolyte_conduction` writes it
    in a two-channel field), emits a velocity on the set.

        v_i = mu q_i E(x_i),   E = -grad phi, trilinearly interpolated from the grid

    `phi` is the potential of the bath with the membrane and the protein as insulators and the
    cell's potential across the box, so the field an ion feels is concentrated where the
    geometry puts the drop -- across the filter of a potassium channel, not spread evenly over the
    box. `charge` in e (a number or a block), `mobility` the ion's, `volt_per_sim` converts the
    field's volts to the spec's energy per charge.

    `constant_field: {cell, slab: [z_lo, z_hi]}` IN PLACE OF `field` -- GOLDMAN'S CONSTANT FIELD: the
    cell's potential psi (its `psi` block, sim energy per charge) drops linearly across the slab and
    nowhere else,

        E = (0, 0, psi / (z_hi - z_lo)) inside the slab,  0 in both baths

    in `electrolyte_conduction`'s convention (phi = +psi/2 below, -psi/2 above). What it leaves out is
    what the solve is for -- the access resistance at each mouth and the drop concentrated on a pore's
    narrowest part -- and what it needs is no grid, which is the point when a bath holding a hundred
    channels is too wide for one at the 0.15 nm a pore needs.

    Reference: Roux, B. (1997). Biophys. J. 73:2980 (the membrane potential as the field of the
    electrolyte with the membrane's geometry); Khalili-Araghi, F. et al. (2013). Biophys. J.
    104:2045 (applied-field simulations of ion channels); Goldman, D.E. (1943). J. Gen. Physiol.
    27:37 (the constant field).
    """

    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["electric_field", "electrophoresis", "membrane_potential"]
    PARAM_ROLES = {"field": "grid_field_with_phi_in_mV", "channel": "the_phi_channel", "charge": "charge_in_e_or_a_block",
                   "mobility": "ion_mobility", "volt_per_sim": "volts_per_sim_energy_per_charge",
                   "constant_field": "cell_set_with_psi_and_the_slab_z_lo_hi_world_it_drops_across"}
    PARAM_UNITS = {"mobility": "mobility"}
    REFERENCE = ("Roux, B. (1997). Biophys. J. 73:2980; Khalili-Araghi, F. et al. (2013). Biophys. J. 104:2045; "
                 "Goldman, D.E. (1943). J. Gen. Physiol. 27:37.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.cf = params.get("constant_field")
        if (self.cf is None) == (params.get("field") is None):
            raise ValueError("field_force: give `field` (a solved potential) or `constant_field` (Goldman's), one of them")
        if self.cf is None and params.get("volt_per_sim") is None:
            raise ValueError("field_force: a `field` in mV needs `volt_per_sim`")
        self.field = None if self.cf else str(params["field"])
        self.ch = int(params.get("channel", 0))
        self.q = params.get("charge", 1.0)
        self.mu = float(params.get("mobility", 1.0))
        self.vps = float(params.get("volt_per_sim", 1.0))

    def _constant(self, H, lvl, X):
        z0, z1 = (float(v) for v in self.cf["slab"])
        psi = float(H.level(str(self.cf["cell"])).get("psi")[:, 0].mean())
        inside = ((X[:, 2] > z0) & (X[:, 2] < z1)).to(X.dtype)
        q = _block(lvl, self.q)[:, 0].to(X.dtype) if isinstance(self.q, str) else float(self.q)
        v = torch.zeros_like(X)
        v[:, 2] = self.mu * q * (psi / (z1 - z0)) * inside * lvl.occ
        return {self.at: v}

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        if self.cf:
            return self._constant(H, lvl, X)
        g = H.fields[self.field].grid[self.ch].to(X.dtype)                    # mV
        n = torch.tensor(g.shape, device=X.device, dtype=X.dtype)
        phi = g * (1e-3 / self.vps)                                           # sim e-psi
        # E at the voxel centres by central differences (world units: dx = 1/n per axis)
        E = torch.zeros((3,) + tuple(g.shape), device=X.device, dtype=X.dtype)
        E[0, 1:-1] = -(phi[2:] - phi[:-2]) * (0.5 * n[0])
        E[1, :, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) * (0.5 * n[1])
        E[2, :, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) * (0.5 * n[2])
        # trilinear interpolation at the beads (voxel centres at (i + 0.5) / n)
        u = X * n - 0.5
        i0 = torch.floor(u).long()
        f = u - i0
        out = torch.zeros_like(X)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ix = (i0[:, 0] + dx).clamp(0, g.shape[0] - 1)
                    iy = (i0[:, 1] + dy).clamp(0, g.shape[1] - 1)
                    iz = (i0[:, 2] + dz).clamp(0, g.shape[2] - 1)
                    w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                         * (f[:, 2] if dz else 1 - f[:, 2]))
                    out += w[:, None] * E[:, ix, iy, iz].T
        q = _block(lvl, self.q)[:, 0].to(X.dtype) if isinstance(self.q, str) else float(self.q)
        v = self.mu * (q[:, None] if torch.is_tensor(q) else q) * out * lvl.occ[:, None]
        return {self.at: v}


@register_operator("flux_counter", family="hierarchy", set="particle", kind="lateral",
                   title="Ionic current as counted crossings",
                   equation=r"""$$I=\frac{q\,(N_{\downarrow}-N_{\uparrow})}{\Delta t_{\mathrm{window}}}$$""")
class FluxCounter(Lateral):
    """A measurement and a closed circuit: charges counted as they cross the membrane's plane,
    and each one that reaches the far face of the box carried round to the near one.

    ions -> cell: reads the set's positions, writes the cell's current block and, with `wrap`,
    the ions' z in place.

        crossing  a bead that was above z_m last frame and is below it now (or the reverse),
                  within `radius` of the channel axis
        I         = q (downward - upward crossings) / window, an exponential running mean over
                    `window` frames, in charges per sim time (the `current` unit)
        wrap      a bead past the box's top (bottom) face re-enters at the bottom (top): the
                  applied-field circuit of Khalili-Araghi et al. 2013, so the bath on each side
                  keeps its ions while current flows

    Downward (outside -> inside, top -> bottom) is counted positive for a positive charge, the
    convention of an INWARD current being negative is applied by the sign the spec gives `sign`.
    `n_filter`, with `filter_z: [z0, z1]`, also writes how many beads sit inside that window
    within `radius` -- the filter's occupancy, the ruler's second number.

    Reference: Khalili-Araghi, F. et al. (2013). Biophys. J. 104:2045; Berneche, S. & Roux, B.
    (2001). Nature 414:73 (knock-on conduction through a K+ channel's filter).
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["z_m"]
    MECHANISM_TAGS = ["measurement", "ionic_current", "periodic_circuit"]
    PARAM_ROLES = {"z_m": "the_membrane_plane_world", "radius": "within_this_of_the_axis", "window": "running_mean_frames",
                   "charge": "charge_per_bead_e", "cell": "set_with_the_blocks", "block": "current_block",
                   "wrap": "carry_a_bead_past_the_top_to_the_bottom", "filter_z": "occupancy_window_world",
                   "occupancy_block": "cell_block_for_the_occupancy", "sign": "+1_or_-1", "axis_xy": "channel_axis_world"}
    MAY_MUTATE_INTEGRATED_STATE = True
    REFERENCE = "Khalili-Araghi, F. et al. (2013). Biophys. J. 104:2045; Berneche & Roux (2001). Nature 414:73."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.zm = float(params["z_m"])
        self.r = float(params.get("radius", 1.0))
        self.win = float(params.get("window", 1000.0))
        self.q = float(params.get("charge", 1.0))
        self.cell = str(params.get("cell", "cell"))
        self.block = str(params.get("block", "j_channel"))
        self.wrap = bool(params.get("wrap", True))
        self.fz = params.get("filter_z")
        self.occ_block = str(params.get("occupancy_block", "n_filter"))
        self.sign = float(params.get("sign", 1.0))
        self.axis = params.get("axis_xy")
        self._prev = None
        self._I = 0.0
        self.n_down = 0
        self.n_up = 0

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        dev = X.device
        c = torch.tensor([float(v) for v in self.axis], device=dev, dtype=X.dtype) if self.axis else X[:, :2].mean(0)
        near = (X[:, :2] - c).norm(dim=1) < self.r
        z = X[:, 2]
        h = float(getattr(H, "dt", 1.0))
        cnt = 0.0
        if self._prev is not None:
            was = self._prev
            down = (was > self.zm) & (z <= self.zm) & near & (lvl.occ > 0)
            up = (was <= self.zm) & (z > self.zm) & near & (lvl.occ > 0)
            nd, nu = int(down.sum()), int(up.sum())
            self.n_down += nd; self.n_up += nu
            cnt = float(nd - nu)
        w = 1.0 - math.exp(-1.0 / self.win)
        self._I = self._I + w * (self.sign * self.q * cnt / h - self._I)
        if self.wrap:
            top = z >= 1.0 - 1e-6
            bot = z <= 1e-6
            if bool(top.any()) or bool(bot.any()):
                a, b = lvl.state_schema["pos"]
                st = lvl.state.clone()
                st[top, a + 2] = 2e-3
                st[bot, a + 2] = 1.0 - 2e-3
                lvl.state = st
                z = lvl.get("pos")[:, 2]
        self._prev = z.detach().clone()
        cl = H.level(self.cell)
        one = torch.ones(cl.n, device=dev, dtype=cl.state.dtype)
        if self.block in cl.state_schema._slices:
            _write(cl, self.block, one * self._I)
        if "n_crossed" in cl.state_schema._slices:                   # net charges carried across, cumulative
            _write(cl, "n_crossed", one * float(self.sign * (self.n_down - self.n_up)))
        if self.fz is not None and self.occ_block in cl.state_schema._slices:
            inside = near & (z > float(self.fz[0])) & (z < float(self.fz[1])) & (lvl.occ > 0)
            _write(cl, self.occ_block, one * float(inside.sum()))
        fr = _frame(H)
        if fr in _REPORT:
            print(f"[flux_counter f{fr}] {self.at}: {self.n_down} down, {self.n_up} up so far; running current "
                  f"{self._I:.4g} charges per sim time", flush=True)
        return {}


# =============================================================================================
# RIGID DOMAINS
# =============================================================================================
@register_operator("shape_match", family="mechanics", set="particle", kind="lateral",
                   title="Rigid domains by shape matching",
                   equation=r"""$$\mathbf x_i\leftarrow\mathbf x_i+\alpha\big(\mathbf c_g+R_g(\mathbf x^0_i-\mathbf c^0_g)-\mathbf x_i\big),\quad R_g=\arg\min_{R\in SO(3)}\sum_{i\in g}\lVert R(\mathbf x^0_i-\mathbf c^0_g)-(\mathbf x_i-\mathbf c_g)\rVert^2$$""")
class ShapeMatch(Lateral):
    """Each domain of a protein kept at its deposited shape: moved only as a rigid body.

    sets -> sets: reads the positions (and the integer `domain_block`), writes them in place.

        for every group g = (set, domain):
            c_g, c0_g  the group's centroid now and in the seeded structure
            R_g        the rotation that best maps the seeded shape onto the current one (Kabsch:
                       the polar factor of sum_i (x_i - c_g)(x0_i - c0_g)^T, by SVD, det +1)
            x_i       <- x_i + alpha (c_g + R_g (x0_i - c0_g) - x_i)

    Run first in the schedule, it projects the last frame's motion -- every force the other operators
    applied to every bead -- onto the rigid motions of each domain: the net force of the membrane on
    a helix moves the helix, its net torque turns it, and nothing bends it. With `alpha` 1 the domains
    are rigid; below 1 they are soft (Mueller et al. 2005's stiffness). The backbone springs that
    cross a domain boundary (`elastic_network` with `local_max_sep`) are the hinges, and the Go wells
    between domains and between chains the breakable packing.

    WHY. A gating transition at this resolution is domains moving as bodies -- MscS's paddles
    tilting, TM3a turning about its kink at G113, MscL's TM1 bundle tilting open (Wang et al. 2014,
    eLife 3:e01834: the helix-tilt model). An alpha-carbon network that must hold each domain's fold
    with springs and Go wells at kT melted in exp04's first two attempts (native contacts 100 -> 55%
    in 20,000 frames); a domain held by shape matching cannot melt, and what it cannot do is exactly
    what it should not do.

    Reference: Mueller, M., Heidelberger, B., Teschner, M. & Gross, M. (2005). Meshless deformations
    based on shape matching. ACM Trans. Graph. 24:471; Kabsch, W. (1976). Acta Cryst. A32:922.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["rigid_body", "shape_matching", "domain"]
    PARAM_ROLES = {"sets": "the_protein_sets_one_per_chain", "domain_block": "integer_block_naming_each_bead_s_domain",
                   "alpha": "0_to_1_stiffness_1_rigid", "flexible": "domain_values_left_free_not_shape_matched"}
    MAY_MUTATE_INTEGRATED_STATE = True
    REFERENCE = ("Mueller, M. et al. (2005). ACM Trans. Graph. 24:471; Kabsch, W. (1976). Acta Cryst. A32:922.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.sets = [str(s) for s in (params.get("sets") or [self.at])]
        self.dblock = params.get("domain_block")
        self.alpha = float(params.get("alpha", 1.0))
        # `flexible: [k, ...]` -- DOMAINS THAT ARE CHAINS, NOT BODIES: their beads are not shape-matched,
        # so they move under their own springs and contacts (alpha-hemolysin's stem, disordered until
        # it zips into the barrel -- Chatterjee et al. 2026). Default: every domain rigid.
        self.flex = [int(v) for v in (params.get("flexible") or [])]
        self._groups = None

    def _build(self, H):
        """The groups, flattened once: for the union of the sets, each bead's group index and its
        rest offset from its group's rest centroid -- so a frame is ONE batched SVD, not one per
        domain (28 small SVDs a frame tripled the cost of an MscS frame, 8 -> 27 ms)."""
        gid, Q, sizes, n_groups = [], [], [], 0
        for s in self.sets:
            lv = H.level(s)
            X0 = lv.get("pos").detach().clone()
            b = _block(lv, self.dblock) if self.dblock else None
            dom = b[:, 0].round().long() if b is not None else torch.zeros(X0.shape[0], dtype=torch.long, device=X0.device)
            g = torch.empty(X0.shape[0], dtype=torch.long, device=X0.device)
            q = torch.empty_like(X0)
            g.fill_(-1); q.zero_()
            for d in torch.unique(dom).tolist():
                if int(d) in self.flex:
                    continue                                              # a chain: left to its own forces
                idx = torch.nonzero(dom == d).reshape(-1)
                g[idx] = n_groups
                q[idx] = X0[idx] - X0[idx].mean(0)
                n_groups += 1
            gid.append(g); Q.append(q); sizes.append(X0.shape[0])
        self._gid = torch.cat(gid); self._Q = torch.cat(Q); self._sizes = sizes; self._G = n_groups
        # the free beads ride along as one extra, never-used group, and are masked out of the update
        self._free = self._gid < 0
        self._gid = torch.where(self._free, torch.full_like(self._gid, n_groups), self._gid)
        cnt = torch.bincount(self._gid[~self._free], minlength=n_groups)
        self._cnt = cnt.clamp_min(1).to(self._Q.dtype)
        self._groups = True
        print(f"[shape_match] {n_groups} rigid domains over {len(self.sets)} sets (sizes {int(cnt.min())}-{int(cnt.max())} "
              f"beads), alpha {self.alpha}", flush=True)

    def forward(self, H, mask=None):
        if self._groups is None:
            self._build(H)
        lvls = [H.level(s) for s in self.sets]
        X = torch.cat([lv.get("pos") for lv in lvls])
        dt = X.dtype
        g, Q = self._gid, self._Q.to(dt)
        G1 = self._G + 1                                                    # + the free beads' dummy group
        cnt = torch.cat([self._cnt.to(dt), torch.ones(1, device=X.device, dtype=dt)])
        c = torch.zeros(G1, 3, device=X.device, dtype=dt).index_add_(0, g, X) / cnt[:, None]
        P = X - c[g]
        A = torch.zeros(G1, 3, 3, device=X.device, dtype=dt).index_add_(0, g, P[:, :, None] * Q[:, None, :])
        U, S_, Vt = torch.linalg.svd(A)
        d = torch.sign(torch.det(U @ Vt))
        D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], 1))
        R = U @ D @ Vt                                                      # [G, 3, 3]
        goal = c[g] + torch.einsum("nij,nj->ni", R[g], Q)
        goal = torch.where(self._free[:, None], X, goal)                    # a flexible domain is not moved
        worst = float((goal - X).norm(dim=1).max())
        Xn = X + self.alpha * (goal - X)
        s0 = 0
        for lv, n in zip(lvls, self._sizes):
            a, b = lv.state_schema["pos"]
            st = lv.state.clone(); st[:, a:b] = Xn[s0:s0 + n].to(st.dtype); lv.state = st
            s0 += n
        fr = _frame(H)
        if fr in _REPORT:
            print(f"[shape_match f{fr}] largest correction {worst:.5f} world this frame", flush=True)
        return {}



@register_operator("density_field", family="fields", set="particle", kind="exchange",
                   title="Running density of a set on a grid",
                   equation=r"""$$\rho(\mathbf y)\leftarrow\rho+w\Big(\sum_i \frac{e^{-\lvert\mathbf y-\mathbf x_i\rvert^2/2s^2}}{(2\pi s^2)^{3/2}}-\rho\Big),\quad w=1-e^{-1/\tau}$$""")
class DensityField(Exchange):
    """Where a set's beads spend their time: a Gaussian splat of the set, averaged over frames.

    set -> field: reads the set's positions, writes one channel of a grid field in place.

        rho(y) <- rho(y) + w (sum_i G_s(y - x_i) - rho(y)),   w = 1 - exp(-1 / window)

    G_s a normalised Gaussian of width `s` (world). In beads per world^3; the movie contours it. For
    ions in a potassium channel this is the picture cryo-EM gives of them -- density at the binding
    sites of the filter (Wu et al. 2023's IS1-IS4) -- and here it is an OUTCOME: where the ions sit
    on average, not where a site was placed. `every` frames between deposits, `within` an optional
    [x0, x1, y0, y1, z0, z1] box (world) outside which beads are not splatted (the bath's ions would
    wash the filter's out).

    Reference: none -- a time average, stated.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["field", "s"]
    MECHANISM_TAGS = ["measurement", "density", "time_average"]
    PARAM_ROLES = {"field": "grid_field", "channel": "which_channel", "s": "gaussian_width_world",
                   "window": "running_mean_frames", "every": "frames_between_deposits", "within": "box_world"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.field = str(params["field"])
        self.ch = int(params.get("channel", 0))
        self.s = float(params["s"])
        self.win = float(params.get("window", 500.0))
        self.every = max(1, int(params.get("every", 1)))
        self.within = params.get("within")

    def forward(self, H, mask=None):
        fr = _frame(H)
        if fr % self.every:
            return {}
        lvl = H.level(self.at)
        X = lvl.get("pos").detach()
        keep = lvl.occ > 0
        if self.within:
            b = [float(v) for v in self.within]
            keep = keep & (X[:, 0] > b[0]) & (X[:, 0] < b[1]) & (X[:, 1] > b[2]) & (X[:, 1] < b[3]) & (X[:, 2] > b[4]) & (X[:, 2] < b[5])
        X = X[keep]
        fld = H.fields[self.field]
        g = fld.grid
        n = g.shape[1]
        dx = 1.0 / n
        k = int(math.ceil(3 * self.s / dx))
        ax = torch.arange(-k, k + 1, device=X.device)
        off = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
        rho = torch.zeros(g.shape[1:], device=X.device, dtype=g.dtype)
        if X.shape[0]:
            base = torch.floor(X / dx).long()
            idx = base[:, None, :] + off[None, :, :]
            ctr = (idx.to(X.dtype) + 0.5) * dx
            w = torch.exp(-((ctr - X[:, None, :]) ** 2).sum(-1) / (2 * self.s ** 2)) / ((2 * math.pi * self.s ** 2) ** 1.5)
            ok = (idx >= 0).all(-1) & (idx < n).all(-1)
            idx, w = idx[ok], w[ok]
            rho.index_put_((idx[:, 0], idx[:, 1], idx[:, 2]), w.to(g.dtype), accumulate=True)
        a = 1.0 - math.exp(-self.every / self.win)
        out = g.clone()
        out[self.ch] = out[self.ch] + a * (rho - out[self.ch])
        fld.grid = out
        return {}


@register_operator("assembly_probe", family="hierarchy", set="particle", kind="lateral",
                   title="Which subunits have assembled",
                   equation=r"""$$b_{mn}=\big[\#\{(i,j):\lVert\mathbf x_{m,i}-\mathbf x_{n,j}\rVert<d\}\ge c\big]$$""")
class AssemblyProbe(Lateral):
    """A measurement, as an operator: which of a set of subunits are bound, and into what.

    subunit sets -> cell: reads the positions of the subunits' sets, writes the cell's `n_largest`,
    `n_bonds` and `ring` blocks.

        b_mn   subunits m and n are BOUND when at least `min_contacts` of their bead pairs are
               closer than `contact` (world)
        n_largest   the size of the largest connected cluster of bound subunits
        ring        1 when that cluster is CLOSED -- every member bound to exactly two others in it --
                    and has at least three members; 0 otherwise

    Nothing reads these but the curves and the ruler: whether seven toxin protomers close a ring, or
    eight crowd into one, or an arc is left open, is what the run is asked (Chatterjee et al. 2026
    find arcs, heptamers and octamers).

    Reference: none -- a graph of contacts, stated.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["contact"]
    MECHANISM_TAGS = ["measurement", "assembly", "oligomer"]
    PARAM_ROLES = {"sets": "the_subunit_sets", "contact": "bead_contact_distance_world",
                   "min_contacts": "bead_pairs_that_make_a_bond", "every": "frames_between_readings",
                   "cell": "set_receiving_the_blocks"}
    MAY_MUTATE_INTEGRATED_STATE = True
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.sets = [str(s) for s in (params.get("sets") or [self.at])]
        self.d = float(params["contact"])
        self.cmin = int(params.get("min_contacts", 10))
        self.every = max(1, int(params.get("every", 100)))
        self.cell = str(params.get("cell", "cell"))
        self._last = (0, 0, 0)

    def forward(self, H, mask=None):
        fr = _frame(H)
        if fr % self.every == 0:
            Xs = [H.level(s).get("pos").detach() for s in self.sets]
            n = len(Xs)
            adj = [[False] * n for _ in range(n)]
            for m in range(n):
                for q in range(m + 1, n):
                    if float((Xs[m].mean(0) - Xs[q].mean(0)).norm()) > 4.0 * self.d * 20:
                        continue
                    c = int((torch.cdist(Xs[m], Xs[q]) < self.d).sum())
                    adj[m][q] = adj[q][m] = c >= self.cmin
            seen, best = set(), []
            for s0 in range(n):
                if s0 in seen:
                    continue
                comp, stack = [], [s0]
                seen.add(s0)
                while stack:
                    u = stack.pop(); comp.append(u)
                    for v in range(n):
                        if adj[u][v] and v not in seen:
                            seen.add(v); stack.append(v)
                if len(comp) > len(best):
                    best = comp
            nb = sum(adj[m][q] for m in range(n) for q in range(m + 1, n))
            ring = int(len(best) >= 3 and all(sum(adj[u][v] for v in best) == 2 for u in best))
            self._last = (len(best), nb, ring)
        cl = H.level(self.cell)
        one = torch.ones(cl.n, device=cl.state.device, dtype=cl.state.dtype)
        for k, v in zip(("n_largest", "n_bonds", "ring"), self._last):
            if k in cl.state_schema._slices:
                _write(cl, k, one * float(v))
        if fr in _REPORT:
            print(f"[assembly_probe f{fr}] largest cluster {self._last[0]} of {len(self.sets)}, {self._last[1]} bonds, "
                  f"ring {'CLOSED' if self._last[2] else 'open'}", flush=True)
        return {}
