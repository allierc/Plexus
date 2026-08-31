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
    PARAM_ROLES = {"speed": "phase_speed_domain_per_frame", "axis": "propagation_axis"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.channel = int(params.get("channel", 0))
        self.axis = int(params.get("axis", 0))
        self.dt = self.tunable(params.get("dt"), 1.0)
        init = float(params.get("speed_init", 0.0))
        self.c = nn.Parameter(torch.tensor(float(init), device=device))

    def rhs(self, u):
        """du/dt from (C1), for a field of shape [n]*D."""
        n = u.shape[self.axis]
        return -self.c * _dx_centred(u, self.axis, n)

    @staticmethod
    def c_closed_form(u_t, u_next, axis, dt=1.0):
        """The least-squares `c` from a pair of frames, used as the reference `c*`."""
        n = u_t.shape[axis]
        gx = _dx_centred(u_t, axis, n)
        dudt = (u_next - u_t) / dt
        return float(-(dudt * gx).sum() / (gx * gx).sum().clamp_min(1e-30))

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        u = fld.grid[self.channel]
        fld.grid[self.channel] = u + self.dt * self.rhs(u)
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
        self._omega = None                      # allocated on the first field it sees
        self._mask = None
        self.device_ = device

    def bind(self, shape, mask, omega_init=0.0):
        """Allocate `omega_i` against a field shape and fix the region mask.

        Separate from `__init__` because the parameter's SHAPE is the field's, and the field is
        not known until a spec is built -- the same reason `KuramotoField` builds its own state
        lazily. `bind` is idempotent so a trainer may call it before every epoch without resetting
        what has been learned.
        """
        if self._omega is None:
            self._omega = nn.Parameter(torch.full(tuple(shape), float(omega_init),
                                                  device=self.device_))
            self.register_parameter("omega", self._omega)
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
        r = self._omega + self.K * self.coupling(v, w)
        m = self._mask
        return w * r * m, -v * r * m

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        v, w = fld.grid[0], fld.grid[1]
        for _ in range(self.substeps):
            dv, dw = self.rhs(v, w)
            v, w = v + self.dt * dv, w + self.dt * dw
        fld.grid[0], fld.grid[1] = v, w
        return {}
