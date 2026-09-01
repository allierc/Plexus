"""The prototype's spec: `plexus.schema.Spec` plus a `data:` and a `training:` section.

WHY THIS EXISTS. A Plexus spec describes a *forward* model -- sets, fields, operators, a schedule.
This prototype fits, so it needs two things the language does not carry: where the observations come
from, and how the optimisation is run. Neither belongs in `plexus.schema` (a forward run has no
training loop), and neither may be hardcoded here (three datasets share one model).

`plexus.schema.load` reads only the keys it knows and ignores the rest, so `data:` and `training:`
sit at the top level of the same yaml and are parsed here from the same raw dict. The Plexus half of
the spec is returned UNCHANGED, straight from `plexus.schema.load` -- this module adds, it does not
reinterpret.

THE FOUR ARCHITECTURAL OPTIONS are enums, not free strings, so a typo fails at load rather than
silently selecting a default:

    model.encoder_decoder   off | on          -- transfer through a background grid, or not
    model.message           simple | graphcast -- NeuralGNN's form, or the edge-stateful one
    model.n_passes          int >= 1           -- 1 is NeuralGNN, 16 is GraphCast
    model.embedding         none | free | multires

UNITS ARE REQUIRED. `plexus.units` says a model with no `units:` block is dimensionless and no
statement about it may carry a unit. Every measurement-tier gate in this prototype is a comparison
with a quantity (a GCaMP decay in seconds, a washout timescale in minutes), so a spec without units
cannot be gated at that tier and the loader says so rather than warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import field as dc_field
from typing import Any, Optional

import yaml

from plexus import schema as plexus_schema
from plexus.units import Units
# POPULATES THE OPERATOR REGISTRY. `plexus.schema.load` validates every `op:` line against the
# registry and reports "Available: []" if nothing has registered yet, so importing the operator
# package is a precondition of loading a spec, not an optimisation. Importing it here means a
# caller cannot forget -- the first version of this module did, and the failure looked like a
# missing operator rather than a missing import.
import plexus.operators  # noqa: F401,E402


# --- the option vocabularies -------------------------------------------------------------- #
ENCODER_DECODER = ("off", "on")
MESSAGE_KINDS = ("simple", "graphcast")
EMBEDDING_KINDS = ("none", "free", "multires")
OBSERVATION_KINDS = ("identity", "calcium")
GRAPH_KINDS = ("knn", "radius", "none")
DATA_SOURCES = ("simulate", "zarr", "tiff_stack")
TARGET_KINDS = ("increment", "state")
# The trainer's own vocabulary, kept apart from plexus.models.base.KINDS on purpose: KINDS say how
# an operator moves data inside a hierarchy during a step, and a loss does not do that. Mirrored in
# ops_trainer.ROLES, which is the registry that enforces it.
# Mirrored in ops_trainer.ROLES. No `predict`: the model operators of the spec's own
# `operators:` are named directly in `fit.schedule:`, and running them IS the prediction.
TRAINER_ROLES = ("loss", "regularize")
STEP_TOKEN = "step"     # the last entry of fit.schedule; its params are `fit.trainer:`
# `fit.schedule:` NAMES THE FITTING HALF ONLY -- the loss terms and `step`. The MODEL half is the
# spec's own top-level `schedule:`, and naming it again here would be two lists that must agree:
# a spec whose `schedule:` ran one operator while `fit.schedule:` claimed another would be silently
# wrong in the direction that is hardest to see. The runtime prints the composed order instead, so
# what a reader wants -- the whole iteration in one place -- is in the log rather than duplicated
# in the file.
TARGET_NORMS = ("none", "inverse_increment_variance")


def _one_of(value, allowed, what):
    """Validate against a vocabulary, after undoing YAML's boolean coercion.

    YAML 1.1 -- which pyyaml implements -- resolves bare `on`, `off`, `yes` and `no` to booleans.
    So `encoder_decoder: off` arrives as `False`, and a naive membership test rejects the most
    natural way to write the option. Requiring quotes in every config would work and would be
    forgotten exactly once; coercing here is the fix that cannot be forgotten.
    """
    if isinstance(value, bool) and {"on", "off"} <= set(allowed):
        value = "on" if value else "off"
    if value not in allowed:
        raise ValueError(f"{what}: {value!r} is not one of {', '.join(map(repr, allowed))}")
    return value


@dataclass
class GraphSpec:
    """How the interaction relation is built. SPATIAL for now -- `connectome` is a later set."""
    kind: str = "knn"
    k: int = 16
    radius: Optional[float] = None

    @classmethod
    def parse(cls, raw: dict) -> "GraphSpec":
        raw = raw or {}
        kind = _one_of(raw.get("kind", "knn"), GRAPH_KINDS, "data.graph.kind")
        if kind == "radius" and raw.get("radius") is None:
            raise ValueError("data.graph.kind is 'radius' but data.graph.radius is not set")
        return cls(kind=kind, k=int(raw.get("k", 16)),
                   radius=None if raw.get("radius") is None else float(raw["radius"]))


@dataclass
class DataSpec:
    """Where the observations live and how they are split. One dataclass, three datasets."""
    source: str
    path: Optional[str] = None
    state: str = "traces"
    positions: Optional[str] = "positions"
    stimulus: Optional[str] = None
    types: Optional[str] = None                  # ground-truth label array, toys only
    graph: GraphSpec = field(default_factory=GraphSpec)
    split: dict = field(default_factory=dict)    # {train: [a,b], val: [...], test: [...]}
    subsample_neurons: Optional[int] = None
    # WHERE THE GENERATOR RUNS, and it is a data question rather than a training one. A 256^3
    # field stepped 1,200 times with 12 substeps is 22 billion lattice updates; on CPU that is
    # tens of minutes and on GPU it is 205 seconds. It was hardcoded to "cpu" in engine.py, which
    # is exactly the kind of value R5 says belongs in the spec -- and it silently made the 3-D
    # toy look intractable when only the placement was wrong.
    device: str = "cpu"
    toy: dict = field(default_factory=dict)      # only for source == "simulate"

    @classmethod
    def parse(cls, raw: dict) -> "DataSpec":
        if not raw:
            raise ValueError("spec is missing the required `data:` section")
        source = _one_of(raw.get("source"), DATA_SOURCES, "data.source")
        if source != "simulate" and not raw.get("path"):
            raise ValueError(f"data.source is {source!r} but data.path is not set")
        toy = raw.get("toy", {}) or {}
        if source == "simulate":
            need = ("set", "edge_set", "amplitude", "length_scale", "inhibitory_fraction")
            missing = [k for k in need if k not in toy]
            if missing:
                raise ValueError(f"data.source is 'simulate' but data.toy is missing {missing}; "
                                 f"the toy's kernel is a config value, never a default")
        split = raw.get("split", {}) or {}
        for name, rng in split.items():
            if not (isinstance(rng, (list, tuple)) and len(rng) == 2 and rng[0] < rng[1]):
                raise ValueError(f"data.split.{name} must be [start, stop] with start < stop, got {rng!r}")
        return cls(
            device=str(raw.get("device", "cpu")),
            source=source,
            path=raw.get("path"),
            state=raw.get("state", "traces"),
            positions=raw.get("positions", "positions"),
            stimulus=raw.get("stimulus"),
            types=raw.get("types"),
            graph=GraphSpec.parse(raw.get("graph")),
            split=split,
            subsample_neurons=(None if raw.get("subsample_neurons") is None
                               else int(raw["subsample_neurons"])),
            toy=toy,
        )


@dataclass
class ModelSpec:
    """The four options, plus the widths. Nothing here has a dataset-specific default."""
    encoder_decoder: str = "off"
    message: str = "simple"
    n_passes: int = 1
    embedding: str = "free"
    embedding_dim: int = 2
    latent_dim: int = 64
    hidden_dim: int = 64
    n_hidden_layers: int = 1
    observation: str = "identity"
    mesh_resolution: Optional[list] = None       # required when encoder_decoder == "on"
    multires: dict = field(default_factory=dict)  # n_levels / log2_table / features, embedding=multires

    @classmethod
    def parse(cls, raw: dict) -> "ModelSpec":
        raw = raw or {}
        enc = _one_of(raw.get("encoder_decoder", "off"), ENCODER_DECODER, "model.encoder_decoder")
        msg = _one_of(raw.get("message", "simple"), MESSAGE_KINDS, "model.message")
        emb = _one_of(raw.get("embedding", "free"), EMBEDDING_KINDS, "model.embedding")
        obs = _one_of(raw.get("observation", "identity"), OBSERVATION_KINDS, "model.observation")
        n_passes = int(raw.get("n_passes", 1))
        if n_passes < 1:
            raise ValueError(f"model.n_passes must be >= 1, got {n_passes}")
        mesh_res = raw.get("mesh_resolution")
        if enc == "on" and mesh_res is None:
            raise ValueError("model.encoder_decoder is 'on' but model.mesh_resolution is not set; "
                             "the grid resolution is a config value, never a default")
        if msg == "simple" and n_passes > 1:
            raise ValueError("model.message 'simple' with n_passes > 1 is not a supported "
                             "combination: the simple form carries no edge state, so repeating it "
                             "is a different model than it claims to be. Use 'graphcast'.")
        return cls(
            encoder_decoder=enc, message=msg, n_passes=n_passes, embedding=emb,
            embedding_dim=int(raw.get("embedding_dim", 2)),
            latent_dim=int(raw.get("latent_dim", 64)),
            hidden_dim=int(raw.get("hidden_dim", 64)),
            n_hidden_layers=int(raw.get("n_hidden_layers", 1)),
            observation=obs,
            mesh_resolution=None if mesh_res is None else [int(x) for x in mesh_res],
            multires=raw.get("multires", {}) or {},
        )

    def tag(self) -> str:
        """A short, stable name for this option combination -- used for log dirs and gate rows."""
        return (f"{'ed' if self.encoder_decoder == 'on' else 'noed'}"
                f"_{self.message}_p{self.n_passes}_{self.embedding}")


@dataclass
class RolloutSpec:
    """Rollout as a TAIL fine-tune, which is what GraphCast actually does (supplement sec. 4.5):
    96% of its updates are one-step, then 11k updates at K rising 2->12 at a learning rate 3300x
    below peak. `schedule` is the horizon ramp; `tail` is when it starts and how far the LR drops."""
    schedule: list = field(default_factory=lambda: [1])
    tail_fraction: float = 0.0        # fraction of n_iter spent in the rollout tail
    tail_lr_scale: float = 1.0e-3
    # THE KNOBS THE WEEKEND GRID SEPARATED (papers/weekend_experiment_2026_08_28.md, task 1).
    # Its result: plain t+1 wins. `pushforward` (bptt_window 1) costs 0.082 in R^2_W, so the long
    # gradient chain is load-bearing; `last` (endpoint-only) costs 0.028, so dense per-step
    # supervision earns its keep. Both are the two arms that cleared the 0.015 resolution floor,
    # and both are things GraphCast also refuses to do. They are implemented so the finding can be
    # reproduced here rather than assumed.
    bptt_window: Optional[int] = None      # None = full BPTT; 1 = pushforward
    shooting_stride: Optional[int] = None  # re-anchor on real data every n steps
    step_weighting: str = "uniform"        # uniform | discount | last
    discount: float = 0.5

    @classmethod
    def parse(cls, raw: dict) -> "RolloutSpec":
        raw = raw or {}
        tail = raw.get("tail", {}) or {}
        sched = [int(k) for k in raw.get("schedule", [1])]
        if any(k < 1 for k in sched):
            raise ValueError(f"training.rollout.schedule entries must be >= 1, got {sched}")
        sw = str(raw.get("step_weighting", "uniform"))
        if sw not in ("uniform", "discount", "last"):
            raise ValueError(f"training.rollout.step_weighting: {sw!r} is not one of "
                             f"'uniform', 'discount', 'last'")
        return cls(schedule=sched,
                   tail_fraction=float(tail.get("fraction", 0.0)),
                   tail_lr_scale=float(tail.get("lr_scale", 1.0e-3)),
                   bptt_window=(None if raw.get("bptt_window") is None
                                else int(raw["bptt_window"])),
                   shooting_stride=(None if raw.get("shooting_stride") is None
                                    else int(raw["shooting_stride"])),
                   step_weighting=sw, discount=float(raw.get("discount", 0.5)))


@dataclass
class TrainingSpec:
    """The optimisation. Defaults follow GraphCast where they are load-bearing and measured:
    AdamW beta2 = 0.95, global grad-norm clip 32, cosine decay TO ZERO (the weekend benchmark's
    one-checkpoint collapses of 0.2-0.5 in R2_W are the symptom of never annealing), and the target
    normalised by the inverse variance of the INCREMENT, not of the state."""
    n_iter: int = 10_000
    batch_frames: int = 8
    lr: float = 1.0e-3
    lr_W: Optional[float] = None            # W gets its OWN rate in every production config
    lr_embedding: Optional[float] = None
    # --- the regularisers connectome-gnn's production configs actually run with -------------- #
    # (config/fly/flyvis_noise_005_calib_nominal_l4.yaml). These are not decoration: the weekend
    # benchmark measured the group lasso recovering +0.153 in R^2_W on 5/5 folds, by removing a
    # degeneracy rather than by expressing a preference for simple models.
    coeff_W_L1: float = 0.0
    coeff_W_L2: float = 0.0
    coeff_msg_weight_L1: float = 0.0        # g_phi weight L1
    coeff_msg_weight_L2: float = 0.0
    coeff_update_weight_L1: float = 0.0     # f_theta weight L1
    coeff_update_weight_L2: float = 0.0
    coeff_msg_smooth: float = 0.0           # the g_phi_diff smoothness prior
    coeff_msg_input_group: float = 0.0      # group lasso over g_phi's input blocks
    regul_annealing_rate: float = 0.0
    lr_scheduler: str = "linear_warmup_cosine"
    warmup_iters: int = 1_000
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 32.0
    target: str = "increment"
    target_norm: str = "inverse_increment_variance"
    rollout: RolloutSpec = field(default_factory=RolloutSpec)
    checkpoint_every: int = 1_000
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def parse(cls, raw: dict) -> "TrainingSpec":
        raw = raw or {}
        betas = raw.get("betas", [0.9, 0.95])
        return cls(
            n_iter=int(raw.get("n_iter", 10_000)),
            batch_frames=int(raw.get("batch_frames", 8)),
            lr=float(raw.get("lr", 1.0e-3)),
            lr_W=(None if raw.get("lr_W") is None else float(raw["lr_W"])),
            lr_embedding=(None if raw.get("lr_embedding") is None else float(raw["lr_embedding"])),
            coeff_W_L1=float(raw.get("coeff_W_L1", 0.0)),
            coeff_W_L2=float(raw.get("coeff_W_L2", 0.0)),
            coeff_msg_weight_L1=float(raw.get("coeff_msg_weight_L1", 0.0)),
            coeff_msg_weight_L2=float(raw.get("coeff_msg_weight_L2", 0.0)),
            coeff_update_weight_L1=float(raw.get("coeff_update_weight_L1", 0.0)),
            coeff_update_weight_L2=float(raw.get("coeff_update_weight_L2", 0.0)),
            coeff_msg_smooth=float(raw.get("coeff_msg_smooth", 0.0)),
            coeff_msg_input_group=float(raw.get("coeff_msg_input_group", 0.0)),
            regul_annealing_rate=float(raw.get("regul_annealing_rate", 0.0)),
            lr_scheduler=str(raw.get("lr_scheduler", "linear_warmup_cosine")),
            warmup_iters=int(raw.get("warmup_iters", 1_000)),
            weight_decay=float(raw.get("weight_decay", 0.1)),
            betas=(float(betas[0]), float(betas[1])),
            grad_clip=float(raw.get("grad_clip", 32.0)),
            target=_one_of(raw.get("target", "increment"), TARGET_KINDS, "training.target"),
            target_norm=_one_of(raw.get("target_norm", "inverse_increment_variance"),
                                TARGET_NORMS, "training.target_norm"),
            rollout=RolloutSpec.parse(raw.get("rollout")),
            checkpoint_every=int(raw.get("checkpoint_every", 1_000)),
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "cuda")),
        )


@dataclass
class FitBlock:
    """The `fit:` section: what to fit the spec's OWN model against, and how.

    A fit spec is a Plexus spec. Its `general` / `sets` / `fields` / `operators` / `schedule` are
    the model -- the learnable twins of a simulation's operators, carrying a `learn:` list -- and
    this block adds only the three things a simulation does not need: which recorded run to score
    against, the objective, and the optimiser. So a fit spec and its simulation twin can be read
    side by side and diffed, which is the point.

        data       {run, field, split}   the recorded trajectory, and how it is divided
        operators  a list, as above      loss / regularize / step, each with its params
        schedule   a list of names       the order they run in, as a simulation's schedule is
        rollout    {recurrent, horizon, weighting, gamma}   one step, or unrolled
    """
    data: dict
    schedule: list
    trainer: dict = dc_field(default_factory=dict)
    operators: list = dc_field(default_factory=list)
    rollout: dict = dc_field(default_factory=dict)
    batch_frames: int = 4
    record_stride: Optional[int] = None

    @property
    def run(self) -> str: return str(self.data["run"])

    @property
    def field(self) -> str: return str(self.data["field"])

    @property
    def split(self) -> dict: return self.data.get("split", {})

    @classmethod
    def parse(cls, raw: Optional[dict]) -> Optional["FitBlock"]:
        if not raw:
            return None
        for key in ("data", "schedule", "trainer"):
            if key not in raw:
                raise ValueError(f"fit: section is missing required key {key!r}")
        for key in ("run", "field"):
            if key not in raw["data"]:
                raise ValueError(f"fit.data: is missing {key!r}; `run` names the recorded run "
                                 f"directory and `field` the field to score against")
        if raw["schedule"][-1] != STEP_TOKEN:
            raise ValueError(f"fit.schedule must END with {STEP_TOKEN!r} -- the parameter update is "
                             f"the last thing a training iteration does, and its params live in "
                             f"`fit.trainer:`. Got {raw['schedule']}")
        return cls(data=raw["data"], operators=raw.get("operators") or [],
                   schedule=raw["schedule"], trainer=raw["trainer"],
                   rollout=raw.get("rollout") or {},
                   batch_frames=int(raw.get("batch_frames", 4)),
                   record_stride=(None if raw.get("record_stride") is None
                                  else int(raw["record_stride"])))


@dataclass
class FitSpec:
    """Everything one yaml declares: the Plexus forward spec plus data, model and training."""
    plexus: Any                       # plexus.schema.Spec, UNCHANGED
    data: DataSpec
    model: ModelSpec
    training: TrainingSpec
    units: Units
    name: str
    path: str
    # THE RAW `general:` BLOCK. A FIT spec on recorded data declares no sets/fields/operators, so
    # `plexus` is None on it -- but it still declares dim, dt, n_frames and the boundary, and the
    # trainer needs them. Reaching through `fit.plexus` for those worked only for generate specs.
    general: dict = dc_field(default_factory=dict)
    fit: Optional[FitBlock] = None

    @property
    def dim(self) -> int: return int(self.general.get("dim", 2))

    @property
    def n_frames(self) -> int: return int(self.general.get("n_frames", 1))

    @property
    def dt(self) -> float: return float(self.general.get("dt", 1.0))

    @property
    def seed(self) -> int: return int(self.general.get("seed", 42))

    @property
    def has_plexus_block(self) -> bool:
        return self.plexus is not None


def load(path: str, require_plexus_block: Optional[bool] = None) -> FitSpec:
    """Parse one yaml into a FitSpec.

    `require_plexus_block`: a GENERATE spec must carry sets/fields/operators/schedule (it runs a
    forward simulation); a FIT spec on recorded data need not. None => infer from the file.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: spec did not parse to a mapping")

    general = raw.get("general", {}) or {}
    name = general.get("name", raw.get("name"))
    if not name:
        raise ValueError(f"{path}: spec missing required key 'name' (under general:)")

    units_raw = general.get("units")
    if units_raw is None:
        raise ValueError(
            f"{path}: spec has no `general.units:` block. plexus.units is explicit that a model "
            f"without one is dimensionless and no result from it may carry a unit -- and every "
            f"measurement-tier gate in this prototype is a comparison with a quantity. Declare at "
            f"least `length_um` and `time_s`.")
    known = {"length_um", "time_s", "force_nN", "amount"}
    unknown = set(units_raw) - known
    if unknown:
        raise ValueError(f"{path}: general.units has unknown key(s) {sorted(unknown)}; the three "
                         f"base scales are {sorted(known - {'amount'})} plus an optional 'amount'. "
                         f"Everything else is DERIVED (plexus/units.py) and a second declaration "
                         f"is a second chance to disagree.")
    units = Units(
        length_um=float(units_raw.get("length_um", 1.0)),
        time_s=float(units_raw.get("time_s", 1.0)),
        force_nN=(None if units_raw.get("force_nN") is None else float(units_raw["force_nN"])),
        amount=(None if units_raw.get("amount") is None else str(units_raw["amount"])),
        declared=True,
    )

    has_forward = all(k in raw for k in ("sets", "fields", "operators", "schedule"))
    want = has_forward if require_plexus_block is None else require_plexus_block
    if want and not has_forward:
        missing = [k for k in ("sets", "fields", "operators", "schedule") if k not in raw]
        raise ValueError(f"{path}: a generate spec needs the Plexus forward block; missing {missing}")
    plexus_spec = plexus_schema.load(path) if has_forward else None

    return FitSpec(
        plexus=plexus_spec,
        # A FIT SPEC DECLARES `fit:`; A GENERATE SPEC DECLARES `data:`. Neither needs the other's,
        # so each is required only when the other is absent. Demanding both made every fit spec
        # carry three blocks of dead scalars whose only job was to satisfy a parser.
        data=DataSpec.parse(raw.get("data")) if raw.get("data") or not raw.get("fit") else None,
        model=(ModelSpec.parse(raw.get("model")) if raw.get("model") or not raw.get("fit")
               else ModelSpec.parse({})),
        training=(TrainingSpec.parse(raw.get("training")) if raw.get("training")
                  or not raw.get("fit") else TrainingSpec.parse({"device": "cuda"})),
        general=general,
        fit=FitBlock.parse(raw.get("fit")),
        units=units,
        name=str(name),
        path=path,
    )
