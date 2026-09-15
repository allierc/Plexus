"""The oculomotor plant, REDUCED: six muscle drives in, three gaze angles out.

An eye is a body on springs, not a lookup table. A muscle drive does not produce a gaze angle;
it produces a gaze angle the eye then has to TRAVEL to. That one fact is why this module has two
contracts rather than one, and why a controller cannot simply invert the first: by the time the
eye arrives, the target has moved.

    muscle_gaze_map   aggregate   where the eye would come to rest if these drives were held
                                  forever -- the static map g(m), no time in it
    eye_mechanics     lateral     and how it gets there -- the damped second-order body

The two entities they act on are registered here as well:

    eye       gaze (3) | pose_rate (3) | pose_target (3)     one eye
    muscle    drive (1)                                    six per eye, its extraocular muscles

THE EYE THIS WAS FITTED FROM IS IN `prototype/eye/`, AND IT IS A RICHER OBJECT. There, an eye is a
deformable MLS-MPM body: a set `eye` holding the globe, its tissue as `mpm_particle`, six
`muscle`s whose own tissue is `muscle_particle`, all coupled through ONE shared `mpm_grid`
field. Its own `muscle_ops.py` states the rule outright -- "no operator applies a force to the
eye; the globe rotates because a muscle got shorter" -- and there `gaze` is an integration:none
READOUT, aggregated from the globe's material points by `eye_pose`, while a muscle carries an
integrated activation `act` and reports `length` and `tension`.

Here `pose` is instead an integrated second-order coordinate of a three-angle rigid body, and a
muscle carries a `drive` and nothing else. That is a deliberately poorer model and it exists for
one reason: a controller cannot be trained through the MPM eye. Fitting a circuit to a tracking
task is thousands of trials of hundreds of frames, and differentiating an MLS-MPM rollout that
many times is not affordable, whereas these two operators are a matmul and a 3x3 solve. So the
expensive mechanism is characterised ONCE, offline, and what is trained against is this fit of
it -- the same move `prototype/dot_tracking/train_eyeG.py` makes, and the reason `fit:` names a
measured file rather than exposing parameters to tune.

The consequence to keep in view: a result obtained here is a result about the FIT. Where the fit
is good the two agree by construction; where it is not, only the MPM eye is evidence. The names
`eye` and `muscle` are shared with the prototype on purpose, because they are the same biology
at two resolutions, and a spec's own inline `state:` block always outranks the schemas declared
below (`engine._resolve_schema` step 1), so declaring the richer MPM layout still gets it.

THE COEFFICIENTS ARE MEASURED, NOT ASSUMED. Both operators read one `fit:` json, the result of
characterising a soft-body eye: 27 coefficients per axis for the static map, and a 3x3 damping C
and stiffness K for the mechanics. Nothing here is fitted at run time and nothing here is a
`nn.Parameter` -- the plant is FROZEN, and a controller trained through it is trained through a
measurement. See `prototype/eye/` for the characterisation that produces the file, and
`prototype/dot_tracking/train_eyeG.py` in the connectome-gnn repository for the fitter.

WHAT IS NOT HERE: THE MIRROR. A fitted eye is one eye -- a LEFT eye, in the file this was built
against -- and the right eye is its reflection through the sagittal plane, horizontal gaze and
torsion flipping sign. That reflection is deliberately NOT applied here. `gaze` is always in the
eye's OWN frame, so both eyes run the identical plant from the identical coefficients, and
putting the two into one world frame is a convention of whatever reads them. It belongs with the
objective, beside the equally conventional choice of whether the two eyes are asked to track the
same world point (conjugate) or opposite ones (vergence). Mirroring inside the plant would also
be wrong rather than merely misplaced: M u = rollout(M u_inf, M K M^-1, M C M^-1) for the
reflection M = diag(-1, 1, -1), and conjugating K flips its theta-phi and phi-psi couplings, so
mirroring the commanded equilibrium without also conjugating the mechanics is a different plant.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from plexus.models.base import Aggregate, Lateral
from plexus.models.registry import register_entity, register_operator
from plexus.models.state import (
    NONE, SECOND_ORDER_COORDINATE, SECOND_ORDER_RATE, BOUNDARY_FREE, Block, StateSchema,
)

# The muscle order is part of the contract with the fit file: the 27 coefficients were fitted
# against drives in THIS order, so a spec whose muscles are declared in another one gets a
# different eye without any error being raised. `muscle_gaze_map` checks the count and takes the
# position within the parent's fibre as the muscle index, and this list is what that index means.
MUSCLES = ("LR", "SR", "MR", "IR", "SO", "IO")
N_MUSCLE = len(MUSCLES)

# The 15 unordered pairs i<j, in the order the fit packs them. Combined with the six linear and
# six square terms that is the 27 columns of `beta`.
PAIRS = [(i, j) for i in range(N_MUSCLE) for j in range(i + 1, N_MUSCLE)]
N_QUAD = 2 * N_MUSCLE + len(PAIRS)                       # 6 + 6 + 15 = 27

AXES = ("theta", "phi", "psi")                           # horizontal, vertical, torsion


# --------------------------------------------------------------------------- the entities
def organ_schema(dim: int) -> StateSchema:
    """`gaze` | `pose_rate` | `pose_target`, three degrees each, whatever the world's dimension.

    THE WIDTH IS 3 AND IT IS NOT `dim`. A rigid body has three rotational degrees of freedom no matter how many
    spatial dimensions the run declares, so these blocks do not narrow to 2 in a 2-D world the
    way `pos` does. The third angle is torsion, and it is not decoration: on a real eye the
    muscle synergies leak into it hard enough that a two-angle plant cannot be fitted.

    `gaze` is the coordinate and `pose_rate` its rate, paired as a second-order block so the
    engine integrates `pose_rate += dt * a; gaze += dt * pose_rate` from whatever
    `eye_mechanics` emits. `boundary` is FREE on both, because these are DEGREES and not
    positions -- wrapping them into the world box would fold the gaze back on itself.

    `pose_target` is where the eye is heading, not where it is: the equilibrium the current drives
    command, written each frame by `muscle_gaze_map` and integrated by nothing.
    """
    return StateSchema([
        Block("pose", 3, role="coordinate", integration=SECOND_ORDER_COORDINATE,
              boundary=BOUNDARY_FREE, unit=None),
        Block("pose_rate", 3, role="rate", integration=SECOND_ORDER_RATE,
              boundary=BOUNDARY_FREE, record=False),
        Block("pose_target", 3, role="readout", integration=NONE, boundary=BOUNDARY_FREE),
    ])


def muscle_schema(dim: int) -> StateSchema:
    """`drive` -- one non-negative contraction per muscle, dimensionless, nominally in [0, 1].

    Integrated by nothing. A muscle drive is an instantaneous function of whatever is driving it
    (a motor neuron's rate, a hand-written waveform), not a state with dynamics of its own; the
    dynamics are all downstream, in the body the drives pull on.

    NON-NEGATIVITY IS THE MUSCLE'S PROPERTY, NOT THE READOUT'S. A muscle pulls or does nothing;
    it cannot push. So whatever writes this block is responsible for the rectification, and a
    muscle nothing drives sits at EXACTLY zero rather than at the value some nonlinearity maps
    zero input to. That distinction is load-bearing wherever a circuit reaches only some of the
    six: the unreached ones must contribute nothing at all, not a tonic contraction nobody asked
    for.
    """
    return StateSchema([
        Block("drive", 1, role="readout", integration=NONE, boundary=BOUNDARY_FREE),
    ])


@register_entity("organ", "eye", depth=1, state_schema=organ_schema,
                 render={"color_by": "node_type", "arrows": None})
class Organ:
    """A body its effectors move: a pose, the rate that pose is changing, and the pose being
    commanded.

    `eye` IS AN ALIAS, not a separate entity. An eye is one organ of this shape -- a mass on
    springs whose muscles ask it to go somewhere -- and so are a jaw, a limb segment and a fin.
    Naming the general kind and letting a spec call its own set `eye` keeps the vocabulary
    honest without making every spec say `organ` when it means an eye.

    `depth=1` because it holds a contained set (its effectors), the same hint `cell` carries.
    Nothing dispatches on depth; the containment the engine traverses is `parent`.
    """


@register_entity("muscle", depth=0, state_schema=muscle_schema,
                 render={"color_by": "node_type", "arrows": None})
class Muscle:
    """One extraocular muscle, as an element of a set contained in its eye.

    A set rather than a six-wide block on the eye, because a muscle is a thing a mechanism can
    act on: a motor neuron innervates a MUSCLE, and writing that as an edge set whose `post` is
    this set makes the innervation an ordinary relation rather than a column index chosen inside
    an operator. It is also what lets a circuit reach some muscles and not others by simply
    having no edge to the rest.
    """


# --------------------------------------------------------------------------- the coefficients
def _coefficients(params, need):
    """The plant's numbers, from a `fit:` json OR written out in the spec. Exactly one.

    A FILE IS THE HONEST FORM FOR A MEASUREMENT and stays the default: `fit:` names the
    characterisation the coefficients came from, so two runs of one spec read the same eye and
    the provenance is a path rather than 99 anonymous floats. But a spec that carries its own
    numbers is SELF-CONTAINED -- it runs on a clone with no characterisation archive anywhere --
    and for a rig that is being handed to someone else that is worth more than the provenance.
    So both are accepted and giving both is refused, because a spec that names a file AND
    restates its contents is a spec whose author believes two things about which eye it runs.

    `need` is the keys this caller actually reads, so `muscle_gaze_map` may be given `beta`
    alone and `eye_mechanics` `C` and `K` alone.
    """
    inline = [k for k in ("beta", "C", "K") if k in params]
    if "fit" in params and inline:
        raise ValueError(
            f"eye coefficients given twice: `fit: {params['fit']!r}` and also {inline} written "
            f"out in the spec. Name one -- the file, or the numbers.")
    if "fit" in params:
        return _load_fit(params["fit"], need)
    missing = [k for k in need if k not in params]
    if missing:
        raise ValueError(
            f"this operator needs {list(need)}; the spec gives neither `fit:` nor {missing}. "
            f"Either name a characterisation json or write the coefficients out.")
    return {k: _check(params[k], k, (N_QUAD, 3) if k == "beta" else (3, 3), "the spec")
            for k in need}


def _check(value, key, shape, where):
    arr = np.asarray(value, np.float64)
    if arr.shape != shape:
        raise ValueError(
            f"{where}: {key!r} is {arr.shape}, expected {shape}. For `beta` that is "
            f"{N_MUSCLE} linear + {N_MUSCLE} square + {len(PAIRS)} cross terms per axis, in "
            f"the muscle order {MUSCLES}.")
    return arr


def _load_fit(path, need=("beta", "C", "K")):
    """The characterisation's json: `beta` (27, 3), `C` (3, 3), `K` (3, 3).

    Resolved relative to the repository root when not absolute, so a spec names the file the way
    it names a region and two runs of one spec read the same coefficients.
    """
    if not os.path.isabs(path):
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        cand = os.path.join(root, path)
        path = cand if os.path.exists(cand) else path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"eye fit {path!r} not found. Both eye operators read the SAME characterisation "
            f"json -- `beta` (27, 3), `C` (3, 3), `K` (3, 3) -- so name one file and give it to "
            f"both, or they describe two different eyes.")
    with open(path) as f:
        spec = json.load(f)
    for key in need:
        if key not in spec:
            raise ValueError(f"eye fit {path!r} has no {key!r}.")
        spec[key] = _check(spec[key], key, (N_QUAD, 3) if key == "beta" else (3, 3),
                           f"eye fit {path!r}")
    return spec


# --------------------------------------------------------------------------- g -- the static map
@register_operator("muscle_pose_map", family="mechanics", set="muscle", kind="aggregate")
class MusclePoseMap(Aggregate):
    """g, the static map: where the eye would come to rest if these six drives were held.

    muscle -> eye: reads every muscle's `drive`, writes its parent eye's `pose_target`. The relation
    traversed is the containment map from a muscle to the eye that owns it.

        g^k(m) = sum_i a^k_i m_i  +  sum_{i<=j} b^k_ij m_i m_j,     k in (theta, phi, psi)

    m_i is muscle i's drive, dimensionless; a^k_i is that muscle's linear pull on axis k, in
    degrees per unit drive; b^k_ij is the quadratic term, in degrees per unit drive squared, with
    i == j the muscle's own curvature and i < j the interaction between two muscles. The 27
    coefficients per axis are `beta`, packed as the six linear terms, the six squares, then the
    fifteen crosses in the order of `PAIRS`, so the three sums are one matrix multiply.

    IT IS A FULL QUADRATIC AND THE CROSS TERMS ARE THE POINT. The eye this was fitted against was
    screened for additivity and failed on all fifteen muscle pairs, which is why the fit is a
    joint quadratic over a space-filling sweep of the six-dimensional drive cube rather than six
    marginals with a few interactions bolted on. A model without the crosses leaves several
    degrees on the table against a per-axis noise floor an order of magnitude smaller.

    A QUADRATIC AGGREGATE, WHICH IS WORTH SAYING OUT LOUD. Aggregate is a whole reading its
    parts, and that is exactly what this is. It is not, however, a plain `sum_pi` of a per-child
    quantity: the fifteen cross terms are products of SIBLINGS, pairs of muscles within one eye.
    They stay well defined because a fibre `pi^-1(eye)` is that eye's own six muscles and nothing
    crosses between eyes, and keeping them here keeps all 27 coefficients of an axis in one
    place. The alternative -- a `muscle_pair` relation given entity status, a lateral computing
    the products, then a linear aggregate over both -- is more literally `sum_pi` and splits one
    measured object across two operators for nothing; nothing else will ever want the bare
    products.

    THE MUSCLE ORDER IS A CONTRACT. `beta` was fitted against drives ordered LR, SR, MR, IR, SO,
    IO, and this operator takes a muscle's position within its parent's fibre as its index into
    that order. A spec that declares the six in another order gets a different eye with no error
    raised anywhere, so the count is checked here and the order is stated in `MUSCLES`.
    """

    EMIT = None                        # writes the parent's pose_target; no integrable delta
    INPUTS = ["muscle"]
    OUTPUTS = ["organ"]
    READS = ["drive"]
    WRITES = ["pose_target"]
    SUPPORTED_DIMS = [2, 3]            # acts on angles, not on the world's coordinates
    DIFFERENTIABLE = True
    MAY_MUTATE_INTEGRATED_STATE = True  # writes a block on the parent -- that is what it is for
    REQUIRES_PARAMS = []               # `fit:` OR inline coefficients; see `_coefficients`
    MECHANISM_TAGS = ["oculomotor", "muscle", "static_map", "hammerstein", "quadratic_synergy"]
    PARAM_ROLES = {"fit": "eye_characterisation_json", "beta": "inline_static_map_coefficients",
                   "parent": "organ_set_name",
                   "gain": "scale_on_every_coefficient"}
    REFERENCE = ("The static half of a Hammerstein cascade -- a measured memoryless "
                 "nonlinearity ahead of a linear body. Coefficients fitted from the soft-body "
                 "eye characterisation in Plexus prototype/eye/.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle")
        self.parent = params.get("parent", "organ")
        self.gain = float(params.get("gain", 1.0))       # one scalar on every coefficient
        spec = _coefficients(params, ("beta",))
        beta = torch.as_tensor(spec["beta"], dtype=torch.float32, device=device)
        self.register_buffer_like = None                  # operators are not nn.Modules here
        self._beta = beta * self.gain                     # (27, 3), FROZEN
        self._pairs = torch.as_tensor(np.asarray(PAIRS), dtype=torch.long, device=device)

    def forward(self, H, mask=None):
        mus = H.level(self.at)
        eye = H.level(self.parent)
        if self.parent not in H.ancestors(mus.name):
            raise ValueError(
                f"muscle_gaze_map: {self.at!r} is not contained in {self.parent!r} (its chain is "
                f"{' -> '.join(H.ancestors(mus.name)) or '(none)'}). This operator aggregates a "
                f"muscle onto the eye that owns it; without the containment map there is no "
                f"fibre to read.")
        pidx = H.lift_index(mus.name, self.parent)        # (n_muscle,) -> eye index
        if mus.n != eye.n * N_MUSCLE:
            raise ValueError(
                f"muscle_gaze_map: {mus.n} muscles for {eye.n} eyes. The fit is a map from "
                f"EXACTLY {N_MUSCLE} drives {MUSCLES}, so set `per_parent: {N_MUSCLE}` -- a "
                f"different count would index `beta` with a muscle the eye does not have.")
        dev, dt_ = mus.state.device, mus.state.dtype
        beta = self._beta.to(device=dev, dtype=dt_)
        # A muscle's ordinal WITHIN its own eye is its index into MUSCLES. With per_parent = 6 the
        # children of one parent are contiguous, which is what makes the reshape below the fibre.
        m = (mus.get("drive")[:, 0] * mus.occ).reshape(eye.n, N_MUSCLE)      # (n_eye, 6)
        if mask is not None:
            m = m * mask.reshape(eye.n, N_MUSCLE).to(m.dtype)
        cross = m[:, self._pairs[:, 0]] * m[:, self._pairs[:, 1]]            # (n_eye, 15)
        design = torch.cat([m, m ** 2, cross], dim=-1)                       # (n_eye, 27)
        pose_target = design @ beta                                             # (n_eye, 3), degrees
        g0, g1 = eye.state_schema["pose_target"]
        if torch.is_grad_enabled():
            # Clone-and-reassign so the tape keeps the previous state alive; see
            # `aggregate_centroid` for why the forward path must NOT do this.
            st = eye.state.clone()
            st[:, g0:g1] = pose_target
            eye.state = st
        else:
            eye.state[:, g0:g1] = pose_target
        _ = pidx                                          # the fibre is the reshape; kept for the check above
        return {}


# --------------------------------------------------------------------------- the body
@register_operator("organ_mechanics", family="mechanics", set="organ", kind="lateral")
class OrganMechanics(Lateral):
    """The plant: a damped second-order body pulled toward the commanded equilibrium.

    eye -> eye: reads `gaze`, `pose_rate` and `pose_target`, emits the gaze acceleration.

        u_ddot + C u_dot + K u = K u_inf      i.e.      u_ddot = K (u_inf - u) - C u_dot

    u is `gaze`, the three angles (theta horizontal, phi vertical, psi torsion) in degrees;
    u_dot is `pose_rate`, in degrees per second; u_inf is `pose_target`, the equilibrium
    `muscle_gaze_map` wrote, in degrees. K is the stiffness, a 3x3 in inverse seconds squared --
    how hard the eye is pulled toward u_inf. C is the damping, a 3x3 in inverse seconds -- how
    hard it resists moving. Both are full matrices and not diagonal: the axes are coupled, which
    is the same fact that makes a horizontal movement leak into torsion.

    The fit this was built against gives natural frequencies sqrt(eig K) of 5.78, 6.18 and 7.29
    radians per second -- 0.92 to 1.16 cycles per second -- at damping ratios eig C / (2 w_n) of
    0.40, 0.41 and 0.39. So the eye is UNDERDAMPED: it overshoots and rings at about 1 Hz before
    settling, and that ~1 Hz corner is also what low-passes anything faster than the plant out of
    a gaze-based objective.

    IT IS REGISTERED `lateral` AND THAT IS THE WRONG WORD, for the same reason `neuron_update`'s
    is. Lateral means what neighbours do to each other; this law has no neighbour in it. Each eye
    reads its own three blocks and nothing else -- no gather, no scatter, no relation traversed.
    It is filed under `lateral` because `KINDS` has no name for an entity's own dynamics, which
    is where `drag`, `gravity`, `glide`, `velocity_cruise`, `metabolite_homeostasis` and the
    `cell_cycle` timers all sit too.

    SECOND ORDER, AND THE ENGINE OWNS THE INTEGRATION. `EMIT = "acceleration"` is the whole of
    this operator's contract with the clock: it hands back u_ddot and the engine advances
    `pose_rate += dt * a; gaze += dt * pose_rate`. Getting that wrong is not subtle -- a
    first-order eye has no overshoot, no ringing and almost no lag, and the control problem it
    poses stops being the one a real eye poses.

    THE DAMPING IS TAKEN IMPLICITLY, AND THAT IS THE DEFAULT. This body returns the acceleration
    that the implicit update `u_dot' = (I + dt C)^-1 (u_dot + dt K (u_inf - u))` would produce,
    i.e. `a = (u_dot' - u_dot) / dt`, so the engine's own explicit step `u_dot += dt * a` lands
    exactly where the implicit one would. The dt cancels, which is what makes this expressible
    without the engine knowing anything about it.

    TWO REASONS, AND ONLY ONE OF THEM IS ABOUT STABILITY. Explicit damping is stable only while
    dt * max(eig C) < 2, and a fitter minimising error at one timestep will walk C toward that
    edge; an earlier fit of this eye reached 1.980 of the limit of 2 at its own 0.009 s step and
    then diverged when a controller rolled it out at 1/60 s. THE FIT IN USE IS NOWHERE NEAR THAT:
    max(eig C) is 5.741 per second, so dt * max(eig C) is 0.096 at dt = 1/60 s, a factor of 21
    inside the limit, and an explicit step is perfectly stable on it. What the implicit form buys
    on THIS eye is not stability but agreement -- it is the discretisation the reference
    implementation uses, so it is the one that reproduces a trajectory the reference produced,
    and a parity check against a trained controller is only worth running if the plant underneath
    it is stepped the same way.

    `implementation: explicit` is registered below and evaluates -C u_dot at the rate the eye
    already has. Keeping it is deliberate: it makes the difference measurable rather than
    asserted. Run both over one trial and report the gaze difference in degrees; if it is under
    the fit's own residual, the choice is genuinely free and the run may say so.
    """

    EMIT = "acceleration"              # second-order: an organ has inertia
    INTEGRAND = "pose"                 # not the set's spatial coordinate -- the angles
    INPUTS = ["organ"]
    OUTPUTS = ["organ"]
    READS = ["pose", "pose_rate", "pose_target"]
    WRITES = ["pose"]
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = []               # `fit:` OR inline coefficients; see `_coefficients`
    MECHANISM_TAGS = ["oculomotor", "plant", "second_order", "damped_oscillator",
                      "hammerstein"]
    PARAM_ROLES = {"fit": "eye_characterisation_json", "C": "inline_damping_matrix",
                   "K": "inline_stiffness_matrix",
                   "damping_scale": "multiplier_on_C", "stiffness_scale": "multiplier_on_K"}
    REFERENCE = ("The linear half of a Hammerstein cascade. C and K fitted to the step "
                 "responses of the soft-body eye in Plexus prototype/eye/.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "organ")
        spec = _coefficients(params, ("C", "K"))
        c_s = float(params.get("damping_scale", 1.0))
        k_s = float(params.get("stiffness_scale", 1.0))
        self._C = torch.as_tensor(spec["C"], dtype=torch.float32, device=device) * c_s
        self._K = torch.as_tensor(spec["K"], dtype=torch.float32, device=device) * k_s

    def _accel(self, u, u_dot, u_inf, dt):
        """The implicit-damping acceleration: what `u_dot += dt * a` must be to equal
        `u_dot' = (I + dt C)^-1 (u_dot + dt K (u_inf - u))`."""
        C = self._C.to(device=u.device, dtype=u.dtype)
        K = self._K.to(device=u.device, dtype=u.dtype)
        eye3 = torch.eye(3, device=u.device, dtype=u.dtype)
        minv = torch.linalg.inv(eye3 + dt * C)
        target = u_dot + dt * ((u_inf - u) @ K.T)
        return ((target @ minv.T) - u_dot) / dt

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        u = lvl.get("pose")
        u_dot = lvl.get("pose_rate")
        u_inf = lvl.get("pose_target")
        dt = float(getattr(H.config, "dt", 1.0)) or 1.0
        a = self._accel(u, u_dot, u_inf, dt) * lvl.occ[:, None]
        if mask is not None:
            a = a * mask[:, None].to(a.dtype)
        return {self.at: a}


@register_operator("organ_mechanics", family="mechanics", set="organ", kind="lateral",
                   implementation="explicit")
class OrganMechanicsExplicit(OrganMechanics):
    """The same mechanics with the damping taken EXPLICITLY: -C u_dot at the rate the eye has.

    Same biology, same C and same K; only the discretisation differs, so this is an
    `implementation` and not a `model` -- a numerical choice, never a biological one.

        a = K (u_inf - u) - C u_dot

    It is registered so the default's claim can be MEASURED, and it has been. Over one 8 s trial
    at dt = 1/60 s on a six-muscle drive waveform, against the same reference rollout the default
    reproduces to 4e-6 degrees, this body differs by AT MOST 0.175 degrees of gaze -- inside the
    static fit's own residual of 0.394 degrees. So on THIS eye the discretisation is within the
    noise of the measurement it is discretising, and either body is defensible; the default is
    chosen for exact agreement with the reference, not because the other one is wrong. On a
    stiffer fit, where dt * max(eig C) approaches the explicit limit of 2 rather than sitting at
    0.096, they would separate and this is the body that would show it.
    """

    def _accel(self, u, u_dot, u_inf, dt):
        C = self._C.to(device=u.device, dtype=u.dtype)
        K = self._K.to(device=u.device, dtype=u.dtype)
        return (u_inf - u) @ K.T - u_dot @ C.T
