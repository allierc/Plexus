"""KNOWN-ODE operators: the generator's own rules with their parameters made learnable.

WHAT "KNOWN ODE" MEANS HERE, and it is copied deliberately from `connectome-gnn`
(`src/connectome_gnn/models/known_ode.py`). A known-ODE model is NOT a network that might learn
the physics. It IS the ground-truth equation, written out, with every constant replaced by an
`nn.Parameter`. No MLP, no message network, no embedding table -- the activation function and the
update rule are the true ones, and the only thing gradient descent has to do is find the numbers.

WHY IT COMES FIRST, before the GNN. It separates two failures that otherwise arrive together. If
the known-ODE fit cannot recover `c`, `K` and `omega_i` from this data, then the data does not
determine them, and no GNN will do better -- the defect is in the toy. Only once the known-ODE fit
succeeds does a GNN's failure mean anything about the GNN. It is the upper bound on the whole
family, measured before the family is built. Every toy discarded in this prototype so far was
discarded for exactly the reason this catches.

===============================================================================================
THE TWO EQUATIONS TO BE FITTED
===============================================================================================

COARSE -- pure transport, first order in space and time, ONE unknown scalar:

    du/dt  =  -c du/dx                                                                      (C1)

    u(x, t)    the coarse field, dimensionless, on the 256^2 (2-D) / 64^3 (3-D) grid
    c          PHASE SPEED in DOMAIN WIDTHS PER FRAME. This is the single learnable of the
               coarse rule. True value 0.000833333, i.e. one full traverse of the domain in
               1,200 frames.
    du/dx      centred difference on the periodic axis, (u_{m+1} - u_{m-1}) / (2 dx),
               dx = 1/n in domain units.

    Its exact solution is u(x,t) = f(x - ct), which is why the generator can advance it by a
    permutation (an integer-cell roll) and lose no amplitude. The MODEL is not given that
    solution; it is given (C1) and must find c.

FINE -- locally coupled phase oscillators, ONE scalar plus ONE FIELD of unknowns:

    dphi_i/dt  =  omega_i  +  K SUM_{j in N(i)} sin(phi_j - phi_i)                          (F1)
    v_i        =  sin(phi_i) * m_i                                                          (F2)

    phi_i      the phase at pixel i, radians. NOT OBSERVED.
    v_i        the observable, dimensionless in [-1, 1].
    m_i        the region mask, 1 inside a disc/tube and 0 outside. Known, not fitted.
    N(i)       the 2D nearest neighbours on the lattice (4 in 2-D, 6 in 3-D).
    K          COUPLING STRENGTH, rad per unit time per neighbour. One learnable scalar.
               True value 0.90.
    omega_i    NATURAL FREQUENCY at pixel i, rad per unit time. ONE LEARNABLE PER PIXEL, and
               THIS IS THE HETEROGENEITY -- the quantity `a_i` exists to carry. True values are
               a per-region mean (0.6, 0.95, 1.3, 1.65 times omega_mean = 0.035) plus a
               per-pixel uniform offset of half-width 0.012.

THE PHASE IS HIDDEN, AND THAT IS THE ONE PLACE THIS DIFFERS FROM `connectome_gnn`'s known-ODE.
There, the state (voltage) is observed and only parameters are unknown. Here (F2) is a
many-to-one observation: sin(phi) does not determine phi. Writing the rule in the quadrature pair

    v = sin(phi),   w = cos(phi)

closes it, because sin(phi_j - phi_i) = v_j w_i - w_j v_i, so (F1)-(F2) become a system in the
observables alone:

    r_i        =  omega_i  +  K SUM_{j in N(i)} ( v_j w_i - w_j v_i )                       (F3)
    dv_i/dt    =   w_i r_i * m_i                                                            (F4)
    dw_i/dt    =  -v_i r_i * m_i                                                            (F5)

AND (F3) IS ALREADY A MESSAGE-PASSING LAYER, which is the reason this toy is worth fitting at all
rather than a reason to be clever. Read it against the form this prototype is testing:

    message from j to i     W_ij * g(v_j, w_j)  with g the identity and W_ij = K
    receiver gauge          the (w_i, -v_i) rotation applied to the aggregate
    node embedding          omega_i, additive, per node

So a GNN that recovers K as an edge weight and omega_i as a per-node embedding has recovered the
Kuramoto rule exactly, and the known-ODE operator below is that GNN with the network removed.

THE COST OF WRITING IT THIS WAY: the generator must record BOTH components. A run that stores only
sin(phi) has thrown away w, and no fit of (F3)-(F5) is possible from it -- which is why
`kuramoto_field` grows an `emit: quadrature` option rather than this module reconstructing w by an
arcsin that cannot know the branch.
"""

from __future__ import annotations

import torch
from torch import nn

from plexus.models.base import Operator
from plexus.models.registry import register_operator


def _dx_centred(u, axis, n):
    """du/dx on a periodic axis in DOMAIN UNITS, second-order centred.

    In domain units because that is the unit `c` is quoted in -- a speed in cells per frame would
    change meaning when the resolution changes, and the coarse grid's resolution is a config knob.
    """
    return (torch.roll(u, -1, axis) - torch.roll(u, 1, axis)) * (n / 2.0)


@register_operator("transport_known_ode", family="fields", set="field", kind="field",
                   model="transport_fit")
class TransportKnownODE(Operator):
    """Equation (C1) with `c` learnable. One parameter for the whole coarse field.

    THE ESTIMATE IS NOT AN OPTIMISATION PROBLEM AT ALL, in the one-step case, and that is worth
    knowing before a training curve is read as evidence. (C1) is linear in c, so the least-squares
    estimate over a batch of frames is the closed form

        c* = - <du/dt, du/dx> / <du/dx, du/dx>

    and `c_closed_form` returns it. A gradient fit that does not land on that number has a bug in
    the trainer, not a hard problem -- which makes this the cheapest possible check that the
    training loop is wired correctly, independent of any model.
    """

    EMIT = None
    INPUTS: list = []
    OUTPUTS: list = []
    READS: list = []
    WRITES: list = []
    MAPS: list = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS: list = []
    MECHANISM_TAGS = ["advection", "transport", "known_ode"]
    PARAM_ROLES = {"velocity": "phase_velocity_domain_per_frame"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.channel = int(params.get("channel", 0))
        self.dim = int(params.get("dim", 2))
        self.dt = self.tunable(params.get("dt"), 1.0)
        init = params.get("velocity_init", 0.0)
        init = [float(x) for x in init] if isinstance(init, (list, tuple)) \
            else [float(init)] * self.dim
        # ONE PARAMETER, A VECTOR. The generator's wave travels obliquely so that the coarse mesh
        # is visible as a mesh, and an oblique wave is not described by a scalar down one axis.
        # (C1) stays LINEAR in the unknown, so the closed form stays a least-squares solve -- it is
        # a D x D normal-equation solve rather than a ratio of two scalars, and it is BETTER
        # conditioned than the axis-aligned case, which excited only one direction.
        self.v = nn.Parameter(torch.tensor(init, device=device))

    def rhs(self, u):
        """du/dt from (C1) = -(v . grad u), for a field of shape [n]*D."""
        out = torch.zeros_like(u)
        for d in range(u.dim()):
            out = out - self.v[d] * _dx_centred(u, d, u.shape[d])
        return out

    @staticmethod
    def v_closed_form(traj, dt=1.0, lo=None, hi=None, report=False):
        """The least-squares velocity VECTOR over a WHOLE TRAJECTORY -- the reference `v*`.

        Minimises || du/dt + SUM_d v_d du/dx_d ||^2 over v, whose normal equations are the D x D
        system  A v = -b  with  A_de = <du/dx_d, du/dx_e>  and  b_d = <du/dt, du/dx_d>.

        POOLED OVER EVERY PAIR INTO ONE SOLVE, not averaged over per-pair solves, and the
        difference is not cosmetic. A single pair whose displacement happens to be near zero gives
        an almost-singular system and an arbitrary answer; averaging those is averaging noise. On
        the 64^2 dataset 71.5% of pairs move by nothing at all, so per-pair solving is not merely
        worse there, it is undefined. Accumulating A and b first is what "batch least squares"
        means, and it is unbiased whatever any individual pair does.

        `report=True` also returns the condition number of A and the R^2 of (C1) at `v*` -- both
        needed, because a small residual with a huge condition number means the fit found ONE
        identifiable direction and invented the rest. That is exactly what a single plane wave
        does: cond(A) = 5.0e6, R^2 = 0.88, and a velocity 568% wrong as a vector.

        FORWARD TIME DIFFERENCE, deliberately, matching `forward()`'s own forward-Euler step. A
        centred difference is the sharper estimator of the DATA (R^2 0.957 against 0.867) but it is
        not what this model computes, and `v*` has to be the number this model could reach.
        """
        D = traj.dim() - 1
        lo = 0 if lo is None else lo
        hi = (traj.shape[0] - 1) if hi is None else hi
        A = torch.zeros(D, D, dtype=torch.float64, device=traj.device)
        b = torch.zeros(D, dtype=torch.float64, device=traj.device)
        for t in range(lo, hi):
            u = traj[t].double()
            g = [_dx_centred(u, d, u.shape[d]) for d in range(D)]
            dudt = (traj[t + 1].double() - u) / dt
            for d in range(D):
                b[d] += (dudt * g[d]).sum()
                for e in range(D):
                    A[d, e] += (g[d] * g[e]).sum()
        v = torch.linalg.solve(A, -b)
        if not report:
            return v
        num = den = 0.0
        for t in range(lo, hi):
            u = traj[t].double()
            g = [_dx_centred(u, d, u.shape[d]) for d in range(D)]
            dudt = (traj[t + 1].double() - u) / dt
            r = dudt + sum(v[d] * g[d] for d in range(D))
            num += float((r ** 2).sum())
            den += float((dudt ** 2).sum())
        return v, {"cond": float(torch.linalg.cond(A)), "r2": 1.0 - num / den,
                   "pairs": hi - lo}

    def forward(self, H, mask=None):
        # THE GRID IS REBUILT, NOT WRITTEN INTO. See KuramotoKnownODE.forward for the failure this
        # avoids; the same rule applies here even though a one-channel field happens to survive it.
        fld = H.fields[self.field_name]
        u = fld.grid[self.channel]
        nxt = u + self.dt * self.rhs(u)
        fld.grid = torch.cat([fld.grid[:self.channel], nxt[None], fld.grid[self.channel + 1:]], 0)
        return {}


@register_operator("kuramoto_known_ode", family="fields", set="field", kind="field",
                   model="phase_fit")
class KuramotoKnownODE(Operator):
    """Equations (F3)-(F5) with `K` and `omega_i` learnable. The fine rule, network removed.

    TWO PARAMETER GROUPS, AND THEY ARE NOT THE SAME KIND OF THING, which is why they get separate
    learning rates in the trainer exactly as `W`, `a` and `f_theta` do in the production configs:

        K         one scalar shared by every pixel -- a COUPLING, the analogue of `W`
        omega_i   one number per pixel -- the HETEROGENEITY, the analogue of the embedding `a_i`

    The gauge question that dogs `W` in the connectome models does not arise here: K multiplies a
    quantity (v_j w_i - w_j v_i) that is fixed by the observation, so there is no per-sender rescale
    that leaves the prediction invariant. K is identifiable outright, and omega_i is identifiable up
    to nothing at all -- it is an additive per-pixel rate. That makes this toy STRICTLY EASIER than
    the connectome case, which is the point of a toy.

    THE MASK IS NOT FITTED. `m_i` is where the fine rule acts, and it is given. Learning the support
    as well would be a different experiment (where does the fast mechanism live), worth doing later
    and not confounded into this one.
    """

    EMIT = None
    INPUTS: list = []
    OUTPUTS: list = []
    READS: list = []
    WRITES: list = []
    MAPS: list = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS: list = []
    MECHANISM_TAGS = ["synchronisation", "phase_coupling", "known_ode"]
    PARAM_ROLES = {"K": "coupling_strength", "omega": "natural_frequency_per_pixel"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.dt = self.tunable(params.get("dt"), 0.25)
        self.substeps = int(params.get("substeps", 12))
        self.K = nn.Parameter(torch.tensor(float(params.get("K_init", 0.0)), device=device))
        # REGISTERED AS `omega`, THE NAME `PARAM_ROLES` DECLARES, and registered as None here so
        # the name exists before the shape does. The first version assigned `self._omega =
        # nn.Parameter(...)` in `bind` and then re-registered it as "omega"; because
        # `named_parameters` de-duplicates by tensor identity, only the FIRST name survived, so a
        # trainer group asking for `omega` matched nothing while the parameter was there all along.
        self.register_parameter("omega", None)
        self._mask = None
        self.device_ = device

    def bind(self, shape, mask, omega_init=0.0):
        """Allocate `omega_i` against a field shape and fix the region mask.

        Separate from `__init__` because the parameter's SHAPE is the field's, and the field is
        not known until a spec is built -- the same reason `KuramotoField` builds its own state
        lazily. `bind` is idempotent so a trainer may call it before every epoch without resetting
        what has been learned.
        """
        if self.omega is None:
            self.omega = nn.Parameter(torch.full(tuple(shape), float(omega_init),
                                                 device=self.device_))
        self._mask = mask.to(self.device_)
        return self

    def coupling(self, v, w):
        """SUM_{j in N(i)} ( v_j w_i - w_j v_i ), the message aggregate of (F3).

        Written as two rolls per axis rather than as an explicit edge list because the lattice IS
        the graph here; a GNN over the same neighbourhood computes the identical number, and the
        comparison between the two is one of the gates.
        """
        s = torch.zeros_like(v)
        for d in range(v.dim()):
            for shift in (1, -1):
                s = s + torch.roll(v, shift, d) * w - torch.roll(w, shift, d) * v
        return s

    def rhs(self, v, w):
        """(dv/dt, dw/dt) from (F3)-(F5)."""
        r = self.omega + self.K * self.coupling(v, w)
        m = self._mask
        return w * r * m, -v * r * m

    def forward(self, H, mask=None):
        """`substeps` explicit Euler steps of (F4)-(F5), then the grid is REBUILT.

        NOT `fld.grid[0], fld.grid[1] = v, w`. `v` and `w` start as VIEWS of `fld.grid`, and the
        substep loop needs those original values for its backward pass; assigning into the same
        storage bumps its autograd version counter and the backward fails with

            one of the variables needed for gradient computation has been modified by an
            inplace operation ... is at version 2; expected version 0

        This is the second instance of one rule, and it is a rule about HIERARCHIES rather than
        about autograd: a hierarchy's buffers persist across calls, so an operator that wants a
        gradient through itself must PRODUCE a new state rather than overwrite the old one. That is
        also what plexus2.tex means by operators being pure transformations.
        """
        fld = H.fields[self.field_name]
        v, w = fld.grid[0], fld.grid[1]
        for _ in range(self.substeps):
            dv, dw = self.rhs(v, w)
            v, w = v + self.dt * dv, w + self.dt * dw
            # PROJECT BACK ONTO THE UNIT CIRCLE, AND THIS IS NOT A NUMERICAL NICETY -- WITHOUT IT
            # THE MODEL CANNOT REPRESENT THE TRUTH AT ALL.
            #
            # (v, w) = (sin phi, cos phi) lives on a circle, and (F4)-(F5) are a rotation at rate
            # r_i. Explicit Euler on a rotation is unstable outward: one step takes the radius from
            # 1 to sqrt(1 + (dt r)^2). At the generator's own settings -- dt 0.25, 12 substeps per
            # tick -- and the TRUE K of 0.90, r reaches omega + K*4 = 3.6 and dt*r = 0.9, so the
            # radius grows 1.35x per substep and 1.35^72 over one recorded interval. Measured: the
            # rollout loss evaluated AT THE GROUND TRUTH was NaN.
            #
            # So the fits that came before this line were not finding a degenerate optimum -- they
            # were being pushed away from the truth by an integrator that diverges there. K stalled
            # at 4% one-step and 45% at horizon 2 while omega inflated 3.5x to compensate, and the
            # obvious reading ("omega absorbs the coupling") was only half of it.
            #
            # The generator has no such problem because it integrates the PHASE and recomputes
            # sin/cos, so its radius is 1 by construction. Renormalising is the same statement in
            # the observables: the state is a phase, and a phase has unit modulus. It is
            # differentiable, it is exact on the constraint manifold, and it changes nothing about
            # what is learnable -- K and omega are untouched.
            n = torch.sqrt(v * v + w * w).clamp_min(1e-12)
            v, w = v / n, w / n
        # outside the mask the field is identically zero, and dividing 0/0 would have put NaN there
        fld.grid = torch.cat([torch.stack([v, w]) * self._mask, fld.grid[2:]], 0)
        return {}
