"""eye_ops -- the oculomotor plant as Plexus2 operators (PROTOTYPE-LOCAL, not promoted).

Importing this module registers six new operator contracts with the Plexus registry, so a
`spec.yaml` can name them exactly like the built-in ones. Nothing here touches
`src/plexus`; the spec is loaded and run by the stock schema + engine.

THE DECOMPOSITION INTO SETS
---------------------------
    eye            the globe as one ORGAN (n=1). State: pos (centroid readout) + gaze
                   (h, v, t in degrees -- the derived orientation readout).
    mpm_particle   the globe's MATERIAL POINTS (parent: eye). MLS-MPM carries pos/vel plus
                   F, C, mass, mu/la, p_vol. Regional identity (sclera / cornea / iris /
                   pupil / choroid / vitreous / lens) is a per-particle label.
    muscle         the SIX extraocular muscles (n=6). State: act (neural activation, the
                   only integrated block) + tension (derived readout). Geometry -- origin,
                   insertion, line of action, rotation axis -- lives in buffers.
    orbit          the bony socket (n=1). State: pos, the centre of the cup.

    insertion      the MAP muscle -> mpm_particle. Per Plexus2 a map is not a runtime
                   object: it is an index buffer (`m_id`, `m_w`) carried by the particle
                   set and NAMED by the operators that traverse it (MAPS = ["insertion"]).

THE OPERATORS
-------------
    eye_anatomy        rewire     build the ovoid + the regional identity of every point
    muscle_insertion   rewire     place the six muscles; build the `insertion` map
    eye_pose           aggregate  particles -> organ: centroid + Kabsch orientation readout
    oculomotor_drive   lateral    gaze program -> per-muscle innervation (reciprocal)
    muscle_traction    broadcast  muscle -> particles: tendon traction at the insertion
    orbit_socket       lateral    bony cup contact + orbital-fat suspension

The two `rewire` operators run once (`before_frame: 1`): they establish which entities
participate in the later operators, which is exactly what Rewire is for.

The causal chain per frame is
    pose readout -> gaze error -> innervation -> tension -> traction -> MLS-MPM -> motion
and it closes: the drive is a feedback controller on the pose the mechanics produced.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from plexus.models.base import Lateral, Aggregate, Broadcast, Rewire
from plexus.models.registry import register_operator

import eye_anatomy as EA


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _t(a, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


def _unit(v, eps=1e-9):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _rot_from_z_to(g):
    """Shortest rotation matrix carrying +z onto the unit vector `g` (Rodrigues).
    numpy, [3,3]. This is the ZERO-TORSION reference orientation for a given gaze."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, g)
    c = float(np.dot(z, g))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _euler_from_R(R):
    """(horizontal, vertical, torsion) in DEGREES from a rotation matrix, using the
    clinical convention documented in eye_anatomy: gaze = R z_hat; torsion is the residual
    roll about the gaze axis once the shortest (zero-torsion) rotation is removed."""
    g = R @ np.array([0.0, 0.0, 1.0])
    g = g / (np.linalg.norm(g) + 1e-12)
    h = math.degrees(math.atan2(g[0], g[2]))
    v = math.degrees(math.asin(float(np.clip(g[1], -1.0, 1.0))))
    R_tors = _rot_from_z_to(g).T @ R
    t = math.degrees(math.atan2(R_tors[1, 0], R_tors[0, 0]))
    return h, v, t


def _R_from_euler(h_deg, v_deg, t_deg):
    """Inverse of `_euler_from_R`: the rotation with that gaze direction and torsion."""
    h, v = math.radians(h_deg), math.radians(v_deg)
    g = np.array([math.sin(h) * math.cos(v), math.sin(v), math.cos(h) * math.cos(v)])
    R0 = _rot_from_z_to(g)
    t = math.radians(t_deg)
    Rz = np.array([[math.cos(t), -math.sin(t), 0.0],
                   [math.sin(t), math.cos(t), 0.0],
                   [0.0, 0.0, 1.0]])
    return R0 @ Rz


# --------------------------------------------------------------------------- #
#  1. eye_anatomy (rewire): the ovoid globe and the regional identity of its points
# --------------------------------------------------------------------------- #
@register_operator("eye_anatomy", family="anatomy", set="particle", kind="rewire")
class EyeAnatomy(Rewire):
    """Turn the seeded uniform BALL of material points into the zebrafish globe.

    Three things happen once, at frame 0:

    1. THE OVOID. Teleost eyes are flattened along the optic axis. The `mpm_particle`
       entity can only seed a uniform ball, so the rest configuration is made ovoid by the
       affine squash  z -> axial_ratio * z  about the globe centre. An affine map of a
       uniform ball is a uniform ellipsoid, so density stays uniform; the particle volume
       and mass are scaled by the same factor. Because this is an INITIAL CONDITION and not
       a deformation, F stays the identity -- the ovoid is the undeformed state and carries
       no residual stress.

    2. REGIONAL IDENTITY. Every point is labelled by the tissue it belongs to (vitreous,
       choroid, sclera, cornea, iris, pupil, lens) from its position in the normalized REST
       ball, and the lens and cornea get their own Lame parameters. This is what makes the
       eye deformable but not uniform: a soft vitreous gel inside a stiff scleral shell,
       with a hard lens -- so muscle traction dimples the sclera at the insertions and the
       stress panel has something to show.

    3. THE REST FRAME. The rest offsets (used by `eye_pose`'s Kabsch fit) and the rest unit
       DIRECTIONS in the pre-squash sphere (used by `muscle_insertion` to place the tendons
       by polar angle) are stored on the set.

    `kind=rewire`: it establishes which entities participate in the later operators without
    adding dynamics. It writes `pos` (the ovoid rest configuration) and so declares
    MAY_MUTATE_INTEGRATED_STATE -- an initial-condition write, not an integration.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True       # writes the ovoid REST configuration at frame 0
    INPUTS = ["mpm_particle"]
    OUTPUTS = ["mpm_particle"]
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["morphogenesis_static", "regional_identity", "material_heterogeneity"]
    PARAM_ROLES = {"axial_ratio": "ovoid_flattening", "lens_youngs": "lens_stiffness",
                   "cornea_youngs": "cornea_stiffness"}
    REFERENCE = "Soules, K. A. & Link, B. A. (2005). BMC Dev. Biol. 5:12 (zebrafish eye morphology)."

    # tissue label ids (the renderer reads TISSUE_NAMES)
    TISSUE_NAMES = ["vitreous", "choroid", "sclera", "cornea", "iris", "fleck", "pupil", "lens"]
    VITREOUS, CHOROID, SCLERA, CORNEA, IRIS, FLECK, PUPIL, LENS = range(8)

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.center = [float(x) for x in params.get("center", EA.GLOBE_CENTER)]
        self.a_eq = float(params.get("a_eq", EA.A_EQ))
        self.axial_ratio = float(params.get("axial_ratio", EA.AXIAL_RATIO))
        self.lens_youngs = float(params.get("lens_youngs", EA.LENS_YOUNGS))
        self.cornea_youngs = float(params.get("cornea_youngs", 260.0))
        self.nu = float(params.get("poisson", 0.2))
        self._done = False

    def _lame(self, E):
        mu = E / (2 * (1 + self.nu))
        la = E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        return mu, la

    def forward(self, H, mask=None):
        if self._done:
            return {}
        p = H.level(self.at)
        dev = p.state.device
        c = _t(self.center, dev)
        X = p.get("pos")
        local = X - c                                    # offsets in the seeded BALL
        r = local.norm(dim=1)
        rn = (r / self.a_eq).clamp(max=1.0)              # normalized ball radius in [0,1]
        d = _unit(local)                                 # rest unit DIRECTION (pre-squash)

        # --- regional identity, in the normalized REST ball ---------------------- #
        tissue = torch.full((p.n,), self.VITREOUS, dtype=torch.long, device=dev)
        tissue = torch.where(rn > EA.R_VITREOUS, torch.full_like(tissue, self.CHOROID), tissue)
        tissue = torch.where(rn > EA.R_INNER, torch.full_like(tissue, self.SCLERA), tissue)
        # the lens: a hard ball pushed anteriorly (zebrafish lenses nearly touch the cornea)
        lc = _t(EA.LENS_CENTER, dev)
        in_lens = ((local / self.a_eq) - lc).norm(dim=1) < EA.LENS_RADIUS
        tissue = torch.where(in_lens, torch.full_like(tissue, self.LENS), tissue)
        # cosmetic anterior surface: pupil disc, iris ring, golden iridophore flecks
        polar = torch.rad2deg(torch.acos(d[:, 2].clamp(-1.0, 1.0)))
        azim = torch.rad2deg(torch.atan2(d[:, 1], d[:, 0])) % 360.0
        shell = rn > EA.R_SHELL
        on_iris = shell & (polar < EA.IRIS_DEG) & (polar >= EA.PUPIL_DEG)
        fleck = torch.zeros_like(on_iris)
        for a0 in EA.IRIS_FLECK_DEG:
            da = (azim - float(a0) + 180.0) % 360.0 - 180.0
            fleck = fleck | (on_iris & (da.abs() < EA.IRIS_FLECK_WIDTH_DEG))
        tissue = torch.where(shell & (polar < EA.PUPIL_DEG), torch.full_like(tissue, self.PUPIL), tissue)
        tissue = torch.where(on_iris, torch.full_like(tissue, self.IRIS), tissue)
        tissue = torch.where(fleck, torch.full_like(tissue, self.FLECK), tissue)
        # the cornea is the anterior CAP of the shell -- it covers pupil+iris optically but
        # mechanically it is the stiff anterior wall, so it is a MATERIAL band under them.
        cornea = (rn > EA.R_INNER) & (polar < EA.IRIS_DEG + 6.0)

        # --- per-region Lame parameters (the deformability the movie shows) ------ #
        mu_l, la_l = self._lame(self.lens_youngs)
        mu_c, la_c = self._lame(self.cornea_youngs)
        mu, la = p.mu.clone(), p.la.clone()
        mu = torch.where(in_lens, torch.full_like(mu, mu_l), mu)
        la = torch.where(in_lens, torch.full_like(la, la_l), la)
        mu = torch.where(cornea & ~in_lens, torch.full_like(mu, mu_c), mu)
        la = torch.where(cornea & ~in_lens, torch.full_like(la, la_c), la)
        p.mu, p.la = mu, la

        # --- the ovoid: affine squash of the rest configuration ------------------ #
        k = self.axial_ratio
        squashed = torch.stack([local[:, 0], local[:, 1], local[:, 2] * k], dim=1)
        new = p.state.clone()
        pa, pb = p.state_schema["pos"]
        new[:, pa:pb] = c + squashed
        p.state = new
        p.p_vol = p.p_vol * k                       # an affine squash scales volume by k ...
        p.mass = p.mass * k                         # ... and, at fixed density, mass with it

        p.register_buffer("tissue", tissue)
        p.register_buffer("rest", squashed.clone())      # rest OFFSETS (ovoid) for the Kabsch fit
        p.register_buffer("rest_dir", d.clone())         # rest DIRECTIONS (sphere) for tendon placement
        p.register_buffer("rest_rn", rn.clone())
        self._done = True
        n_shell = int(shell.sum())
        print(f"[eye_anatomy] ovoid a={self.a_eq:.3f} c={self.a_eq * k:.3f} (ratio {k:.2f}); "
              f"{p.n} points, {n_shell} on the shell, {int(in_lens.sum())} in the lens", flush=True)
        return {}


# --------------------------------------------------------------------------- #
#  2. muscle_insertion (rewire): place the six muscles, build the `insertion` map
# --------------------------------------------------------------------------- #
@register_operator("muscle_insertion", family="anatomy", set="muscle", kind="rewire")
class MuscleInsertion(Rewire):
    """Place the six extraocular muscles and build the map muscle -> mpm_particle.

    Each muscle is given (i) an effective ORIGIN in world coordinates -- the annulus of Zinn
    for the four recti, the TROCHLEA for the superior oblique (its tendon turns through a
    pulley, so the trochlea and not the apex is where it pulls from) and the anteromedial
    orbital floor for the inferior oblique -- and (ii) an INSERTION on the sclera, given as a
    polar/azimuthal direction in the globe's rest frame. The recti insert anterior to the
    equator, the obliques posterior to it, which is what makes the obliques torters.

    The map itself is an index buffer on the particle set, per Plexus2: `m_id` (which muscle
    a point belongs to, -1 for none) and `m_w` (its normalized share of the tendon load).
    Membership is decided in the REST frame and never recomputed -- material points are
    Lagrangian, so a tendon stays attached to the tissue it was attached to.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    INPUTS = ["muscle", "mpm_particle"]
    OUTPUTS = ["muscle", "mpm_particle"]
    READS = ["pos"]
    WRITES = []
    MAPS = ["insertion"]
    MECHANISM_TAGS = ["tendon_insertion", "muscle_pulley", "map_construction"]
    PARAM_ROLES = {"patch_deg": "tendon_arc_half_width", "particles": "target_particle_set"}
    REFERENCE = "Apt, L. (1980). Trans. Am. Ophthalmol. Soc. 78:365 (spiral of Tillaux); Demer, J. L. (2002). Invest. Ophthalmol. Vis. Sci. 43:2179 (orbital pulleys)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle")
        self.particles = params.get("particles", "mpm_particle")
        self.patch_deg = float(params.get("patch_deg", EA.INSERTION_PATCH_DEG))
        self.shell_min = float(params.get("shell_min", 0.90))
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        m = H.level(self.at)
        p = H.level(self.particles)
        dev = p.state.device
        if not hasattr(p, "rest_dir"):
            raise RuntimeError("muscle_insertion requires `eye_anatomy` to run first "
                               "(it needs the rest directions); put it earlier in the schedule.")
        if m.n != EA.N_MUSCLE:
            raise ValueError(f"set {self.at!r} has n={m.n}; the plant has {EA.N_MUSCLE} muscles")

        ins_dir = _t(EA.insertion_dirs(), dev)                 # [6,3] rest insertion directions
        origins = _t(EA.origins_world(), dev)                  # [6,3] world origins/pulleys
        tmax = _t(EA.peak_tensions(), dev)                     # [6]

        d = p.rest_dir                                          # [N,3]
        shell = p.rest_rn > self.shell_min
        cos_lim = math.cos(math.radians(self.patch_deg))
        cosang = d @ ins_dir.T                                  # [N,6]
        cosang = torch.where(shell[:, None], cosang, torch.full_like(cosang, -2.0))
        best = cosang.argmax(dim=1)
        hit = cosang.gather(1, best[:, None]).squeeze(1) > cos_lim
        m_id = torch.where(hit, best, torch.full_like(best, -1))

        # smooth (raised-cosine) load share across the tendon arc, normalized per muscle
        ang = torch.acos(cosang.gather(1, best.clamp(min=0)[:, None]).squeeze(1).clamp(-1.0, 1.0))
        w = torch.cos(0.5 * math.pi * (ang / math.radians(self.patch_deg)).clamp(max=1.0)) ** 2
        w = torch.where(hit, w, torch.zeros_like(w))
        tot = torch.zeros(m.n, device=dev).index_add_(0, m_id.clamp(min=0), w * hit.float())
        m_w = w / tot.clamp(min=1e-9)[m_id.clamp(min=0)]
        m_w = torch.where(hit, m_w, torch.zeros_like(m_w))

        p.register_buffer("m_id", m_id)                        # the `insertion` map ...
        p.register_buffer("m_w", m_w)                          # ... and its weights
        m.register_buffer("origin", origins)
        m.register_buffer("tmax", tmax)
        m.register_buffer("pull", torch.zeros(m.n, 3, device=dev))     # unit line of action
        m.register_buffer("axis", torch.zeros(m.n, 3, device=dev))     # unit rotation axis
        m.register_buffer("ins_pos", torch.zeros(m.n, 3, device=dev))  # current insertion centroid
        counts = [int((m_id == i).sum()) for i in range(m.n)]
        print(f"[muscle_insertion] tendon patches (points/muscle): "
              + "  ".join(f"{k}={n}" for k, n in zip(EA.MUSCLE_KEYS, counts)), flush=True)
        self._done = True
        return {}


# --------------------------------------------------------------------------- #
#  3. eye_pose (aggregate): particles -> organ, the orientation readout
# --------------------------------------------------------------------------- #
@register_operator("eye_pose", family="hierarchy", set="eye", kind="aggregate")
class EyePose(Aggregate):
    """The globe's POSE, aggregated from its material points along the `parent` map.

    Position is the occupancy-weighted centroid (as the stock `aggregate`); ORIENTATION is
    the least-squares rotation carrying the rest offsets onto the current ones (Kabsch), so
    the eye's gaze is measured from the deforming tissue itself rather than prescribed. It
    is fitted on the SHELL only: the sclera is the stiff part that actually rotates, whereas
    the soft vitreous lags and would bias the fit.

    The fitted rotation is reported as the clinical triple (horizontal, vertical, torsion)
    in degrees, written to the organ's `gaze` block. This is a DERIVED READOUT -- it is what
    closes the loop back to `oculomotor_drive` -- so it writes state directly and declares
    MAY_MUTATE_INTEGRATED_STATE, exactly as the stock `aggregate` centroid does.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True       # a derived readout (same precedent as `aggregate`)
    INPUTS = ["mpm_particle"]
    OUTPUTS = ["eye"]
    READS = ["pos"]
    WRITES = ["pos", "gaze"]
    MAPS = ["parent"]
    MECHANISM_TAGS = ["rigid_body_readout", "orientation_estimate", "hierarchical_readout"]
    PARAM_ROLES = {"shell_min": "fit_band_inner_radius"}
    REFERENCE = "Kabsch, W. (1976). Acta Cryst. A32:922 (optimal rotation superposition)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "eye")
        self.child = params.get("child", "mpm_particle")
        self.shell_min = float(params.get("shell_min", 0.86))
        self.R = np.eye(3)                     # last fitted rotation (read by the renderer)

    def forward(self, H, mask=None):
        parent = H.level(self.at)
        p = H.level(self.child)
        if not hasattr(p, "rest"):
            return {}                                        # anatomy has not run yet
        sel = p.rest_rn > self.shell_min
        X = p.get("pos")
        cur_c = (X * p.occ[:, None]).sum(0) / p.occ.sum().clamp(min=1.0)
        A = p.rest[sel]                                      # rest offsets (already centred)
        B = X[sel] - cur_c
        C = (A.t() @ B).double()
        U, S, Vh = torch.linalg.svd(C)
        d = torch.sign(torch.linalg.det(Vh.t() @ U.t()))
        D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
        R = (Vh.t() @ D @ U.t()).float()                     # cur ~ R @ rest
        Rn = R.detach().cpu().numpy().astype(np.float64)
        self.R = Rn
        h, v, t = _euler_from_R(Rn)

        new = parent.state.clone()
        pa, pb = parent.state_schema["pos"]
        new[:, pa:pb] = cur_c[None, :]
        ga, gb = parent.state_schema["gaze"]
        new[:, ga:gb] = _t([h, v, t], parent.state.device)[None, :]
        parent.state = new
        return {}


# --------------------------------------------------------------------------- #
#  4. oculomotor_drive (lateral): gaze program -> reciprocal innervation
# --------------------------------------------------------------------------- #
@register_operator("oculomotor_drive", family="signalling", set="muscle", kind="lateral")
class OculomotorDrive(Lateral):
    """The neural command: a gaze program becomes six muscle activations.

    A `program` of `[frame, h, v, t]` waypoints (degrees) defines the commanded pose. The
    error against the pose `eye_pose` measured, damped by the measured pose RATE, gives a
    desired angular velocity in the head frame

        omega_des = kp * (h*-h) y_hat + kp * (v*-v) (-x_hat) + kp * (t*-t) z_hat  -  kd * omega

    which is then projected onto each muscle's own rotation axis (computed from its geometry
    by `muscle_traction`, never tabulated):

        a_target_m = tonic + gain * relu( omega_des . axis_m )

    The rectification is Sherrington's law of reciprocal innervation: a muscle is driven only
    when it pulls the eye TOWARD the target, and falls back to its tonic level otherwise, so
    an agonist/antagonist pair never fights itself. Which muscles are recruited for a given
    command is therefore a CONSEQUENCE of the anatomy: a pure torsion command reaches only the
    obliques, a vertical command splits between a vertical rectus and the opposing oblique
    whose torsions cancel.

    The returned delta is the first-order activation dynamics  da/dt = (a_target - a)/tau,
    the muscle's electromechanical delay; `act` is the muscle set's integrated block, so the
    engine integrates it (EMIT=velocity) and the activation can never step discontinuously.
    """

    EMIT = "velocity"                        # delta is da/dt ...
    INTEGRAND = "act"                        # ... integrated into the `act` block, not the
                                             # muscle's coordinate (its points carry pos/vel)
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["program"]
    INPUTS = ["eye", "muscle"]
    OUTPUTS = ["muscle"]
    READS = ["gaze", "act"]
    WRITES = ["act"]
    MECHANISM_TAGS = ["motor_command", "reciprocal_innervation", "feedback_control",
                      "activation_dynamics"]
    PARAM_ROLES = {"kp": "position_gain", "kd": "rate_damping", "tonic": "resting_innervation",
                   "gain": "recruitment_gain", "tau": "activation_time_constant",
                   "program": "gaze_waypoints"}
    REFERENCE = "Sherrington, C. S. (1893). Proc. R. Soc. Lond. 53:407 (reciprocal innervation); Robinson, D. A. (1975). Basic Mech. Ocular Motility, 337-374 (oculomotor plant)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle")
        self.eye = params.get("eye", "eye")
        self.kp = float(params.get("kp", 0.09))
        self.kd = float(params.get("kd", 0.014))
        self.tonic = float(params.get("tonic", 0.22))
        self.gain = float(params.get("gain", 1.0))
        self.tau = float(params.get("tau", 0.02))          # in SIM TIME (general.dt units)
        prog = params["program"]
        self.program = np.asarray([[float(x) for x in row] for row in prog], float)
        self._prev = None                                   # previous (h, v, t) for the rate estimate
        self.last = {}                                      # diagnostics for the renderer

    def target(self, frame: int) -> np.ndarray:
        """The commanded (h, v, t) at `frame`: the most recent waypoint (a step command --
        the brainstem burst generator issues a new position, not a trajectory)."""
        idx = np.searchsorted(self.program[:, 0], frame, side="right") - 1
        return self.program[max(idx, 0), 1:4]

    def forward(self, H, mask=None):
        m = H.level(self.at)
        eye = H.level(self.eye)
        dev = m.state.device
        frame = int(getattr(H, "frame", 0))
        gaze = eye.get("gaze")[0].detach().cpu().numpy().astype(np.float64)
        tgt = self.target(frame)
        err = tgt - gaze                                     # (dh, dv, dt) in degrees
        rate = np.zeros(3) if self._prev is None else (gaze - self._prev)
        self._prev = gaze.copy()

        # desired angular velocity in the head frame (see eye_anatomy for the axis mapping)
        e = self.kp * err - self.kd * rate / max(float(getattr(H.config, "dt", 1.0)), 1e-9)
        omega = np.array([-e[1], e[0], e[2]])                # -x elevation, +y abduction, +z intorsion

        axis = m.axis.detach().cpu().numpy().astype(np.float64) if hasattr(m, "axis") else np.zeros((m.n, 3))
        drive = axis @ omega                                 # [6] projection onto each muscle's action
        a_target = np.clip(self.tonic + self.gain * np.maximum(drive, 0.0), 0.0, 1.0)

        a = m.get("act")[:, 0]
        d_act = (_t(a_target, dev) - a) / self.tau
        if mask is not None:
            d_act = d_act * mask.float()
        self.last = {"target": tgt.copy(), "gaze": gaze.copy(), "omega": omega.copy(),
                     "a_target": a_target.copy()}
        return {self.at: d_act[:, None]}


# --------------------------------------------------------------------------- #
#  5. muscle_traction (broadcast): muscle -> particles, along the `insertion` map
# --------------------------------------------------------------------------- #
@register_operator("muscle_traction", family="mechanics", set="particle", kind="broadcast")
class MuscleTraction(Broadcast):
    """Tendon traction: an activated muscle pulls its scleral insertion toward its origin.

    Per muscle and per frame:

      * the current insertion CENTROID is the load-weighted mean of its tendon patch, so the
        line of action follows the globe as it rotates (no prescribed kinematics);
      * the line of action is the unit vector from that centroid to the muscle's effective
        origin -- the apex for a rectus, the trochlea for the superior oblique;
      * that line is WRAPPED onto the sclera: a rectus running from an anterior insertion to
        a deep posterior apex would otherwise pull straight through the globe. Removing the
        fraction `wrap` of the line's inward normal component is the standard arc-of-contact
        approximation. The residual normal component is real and useful -- it is the force
        that seats the globe in its socket, which is why the wrap is not total;
      * the total force is `tension = tmax * act`, distributed over the patch by the map
        weights and divided by each point's mass, so the resulting motion does not depend on
        how finely the tendon is discretized.

    The muscle's ROTATION AXIS, `n_hat x u_hat`, is computed here and published on the muscle
    set; `oculomotor_drive` reads it next frame. The textbook actions of the six muscles are
    therefore never written down anywhere -- they are a consequence of where each muscle
    inserts and where it pulls from.

    `EMIT=mpm_acceleration`: the traction is a body force consumed by the MLS-MPM substep
    (`mpm_scatter` reads it as a_ext), not integrated by the engine.
    """

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["amplitude"]
    MAY_MUTATE_INTEGRATED_STATE = True       # writes the muscle's `tension` readout block
    INPUTS = ["muscle", "mpm_particle"]
    OUTPUTS = ["mpm_particle", "muscle"]
    READS = ["act", "pos"]
    WRITES = ["tension"]
    MAPS = ["insertion"]
    MECHANISM_TAGS = ["active_tension", "tendon_traction", "arc_of_contact", "muscle_pulley"]
    PARAM_ROLES = {"amplitude": "peak_tendon_force", "wrap": "arc_of_contact_fraction",
                   "muscles": "source_muscle_set"}
    REFERENCE = "Robinson, D. A. (1975). Basic Mech. Ocular Motility, 337-374; Miller, J. M. & Robinson, D. A. (1984). Comput. Biomed. Res. 17:436."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.muscles = params.get("muscles", "muscle")
        self.amplitude = float(params["amplitude"])
        self.wrap = float(params.get("wrap", EA.WRAP))

    def forward(self, H, mask=None):
        p = H.level(self.at)
        m = H.level(self.muscles)
        dev = p.state.device
        if not hasattr(p, "m_id"):
            return {}                                          # the map is not built yet
        X = p.get("pos")
        mid = p.m_id
        w = p.m_w
        live = mid >= 0
        idx = mid.clamp(min=0)
        M = m.n

        # current insertion centroid per muscle (load-weighted; follows the rotating globe)
        ins = torch.zeros(M, 3, device=dev).index_add_(0, idx, X * (w * live.float())[:, None])
        m.ins_pos = ins

        centre = (X * p.occ[:, None]).sum(0) / p.occ.sum().clamp(min=1.0)
        n_hat = _unit(ins - centre[None, :])                    # outward scleral normal at the tendon
        u_line = _unit(m.origin - ins)                          # straight pull toward origin/pulley
        # wrap the line onto the sclera (arc of contact): drop most of the inward normal part
        u = u_line - self.wrap * (u_line * n_hat).sum(1, keepdim=True) * n_hat
        u = _unit(u)
        m.pull = u
        m.axis = _unit(torch.cross(n_hat, u, dim=1))            # the muscle's rotation axis

        act = m.get("act")[:, 0].clamp(min=0.0)
        tension = self.amplitude * m.tmax * act                 # total force per muscle
        force = tension[:, None] * u                            # [M,3]
        acc = force[idx] * (w * live.float())[:, None] / p.mass.clamp(min=1e-12)[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()

        new = m.state.clone()                                   # publish the tension readout
        ta, tb = m.state_schema["tension"]
        new[:, ta:tb] = tension[:, None]
        m.state = new
        return {self.at: acc}


# --------------------------------------------------------------------------- #
#  6. orbit_socket (lateral): the bony cup + the orbital fat
# --------------------------------------------------------------------------- #
@register_operator("orbit_socket", family="mechanics", set="particle", kind="lateral")
class OrbitSocket(Lateral):
    """The skull: a bony cup that keeps the globe in the orbit, plus the fat that suspends it.

    Two mechanisms, both body forces on the material points:

      * CUP CONTACT. The socket is a SPHERICAL cup of radius `radius` centred on the orbit,
        present everywhere except an anterior aperture of half-angle `aperture` (the
        palpebral opening). A point outside the cup wall is pushed back along the inward
        normal with stiffness `k` and its outward normal velocity is damped. The cup is
        spherical rather than ovoid on purpose: the globe is an ovoid and could not rotate
        inside a socket that matched it, and in the animal the gap is filled by soft tissue.
        Because the aperture half-angle is under 90 deg the cup wraps past the equator, so
        the globe is captured -- it cannot translate out of the orbit -- while remaining free
        to rotate about its centre.

      * ORBITAL FAT. A soft restoring force toward the socket centre, applied UNIFORMLY to
        every point of the globe: a uniform body force exerts no torque about the centroid,
        so the retrobulbar fat pad recentres the eye without ever resisting gaze. Without it
        the residual normal component of muscle tension would slowly walk the globe around
        inside the cup.

    `EMIT=mpm_acceleration`: a body force for the MLS-MPM substep, not engine-integrated.
    """

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["k"]
    INPUTS = ["orbit", "mpm_particle"]
    OUTPUTS = ["mpm_particle"]
    READS = ["pos", "vel"]
    WRITES = []
    MECHANISM_TAGS = ["contact_constraint", "penalty_contact", "ball_and_socket_joint",
                      "elastic_suspension"]
    PARAM_ROLES = {"k": "cup_stiffness", "damp": "contact_damping", "radius": "cup_radius",
                   "aperture": "anterior_opening_half_angle", "k_fat": "fat_suspension_stiffness",
                   "c_fat": "fat_suspension_damping"}
    REFERENCE = "Koornneef, L. (1977). Doc. Ophthalmol. Proc. Ser. 12 (orbital connective tissue); Schutte, S. et al. (2006). Vision Res. 46:1724 (FE orbit model)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.orbit = params.get("orbit", "orbit")
        self.k = float(params["k"])
        self.damp = float(params.get("damp", 12.0))
        self.radius = float(params.get("radius", EA.CUP_RADIUS))
        self.aperture = float(params.get("aperture", EA.CUP_APERTURE_DEG))
        self.k_fat = float(params.get("k_fat", 260.0))
        self.c_fat = float(params.get("c_fat", 18.0))
        self._c0 = None

    def forward(self, H, mask=None):
        p = H.level(self.at)
        dev = p.state.device
        o = H.level(self.orbit)
        c = o.get("pos")[0]
        X, V = p.get("pos"), p.get("vel")
        if self._c0 is None:
            self._c0 = c.clone()

        rel = X - c[None, :]
        r = rel.norm(dim=1).clamp(min=1e-9)
        n = rel / r[:, None]                                    # outward radial direction
        cos_ap = math.cos(math.radians(self.aperture))
        walled = n[:, 2] < cos_ap                               # the cup exists outside the aperture
        pen = (r - self.radius).clamp(min=0.0) * walled.float() # penetration into the bone
        vn = (V * n).sum(1).clamp(min=0.0)                      # outward normal speed
        acc = -(self.k * pen + self.damp * vn * (pen > 0).float())[:, None] * n

        # orbital fat: uniform (torque-free) recentring of the whole globe
        cen = (X * p.occ[:, None]).sum(0) / p.occ.sum().clamp(min=1.0)
        vel_c = (V * p.occ[:, None]).sum(0) / p.occ.sum().clamp(min=1.0)
        acc = acc + (self.k_fat * (self._c0 - cen) - self.c_fat * vel_c)[None, :]

        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
