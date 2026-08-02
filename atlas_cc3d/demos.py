"""demos -- one reference run per CompuCell3D mechanism, in isolation.

The unit is the MECHANISM, not the paper figure. CompuCell3D is a framework rather than a single
paper, and per-mechanism is the shape the first atlas already used: one spec exercising one thing,
one evidence folder, one observable that would change if that mechanism were removed. Phases 1-4
consume exactly this -- the excavator reads the mechanism, the differ needs a run of it alone.

Each entry supplies:
    specs      python source that builds `spec_list`, a list of PyCoreSpecs objects
    steppable  optional extra steppable source (mitosis needs one; most do not)
    headline   the series whose curve IS the demonstration
    control    the same mechanism switched off, so the curve can be read as caused rather than
               coincident. `None` where the mechanism has no meaningful off state.

Every observable here is a POPULATION or TOPOLOGY statistic, never a trajectory. That is not a
stylistic choice: a Potts model is a Metropolis chain, so a matched trajectory is not a meaningful
object and a differential test in Phase 4 will have to compare distributions. Building the
observables that way now means the evidence folders are already in the right currency.
"""

COMMON_HEAD = '''
import warnings, sys; warnings.filterwarnings("ignore")
from cc3d.core.PyCoreSpecs import *
seed, steps, dim = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
potts = PottsCore(dim_x=dim, dim_y=dim, dim_z=1, steps=steps, fluctuation_amplitude=FLUCT,
                  neighbor_order=2, random_seed=seed)
com = CenterOfMassPlugin()
srf_track = SurfaceTrackerPlugin()
'''

TAIL = '''
body = "\\n".join(s.xml.getCC3DXMLElementString() for s in spec_list)
sys.stdout.write('<CompuCell3D Revision="0" Version="4.10.0">\\n' + body + '\\n</CompuCell3D>\\n')
'''

# --------------------------------------------------------------------------------------------- #
#  1. contact adhesion -- differential adhesion drives cell sorting
# --------------------------------------------------------------------------------------------- #
ADHESION = '''
ct = CellTypePlugin("Condensing", "NonCondensing")
vol = VolumePlugin()
vol.param_new("Condensing", target_volume=25, lambda_volume=2.0)
vol.param_new("NonCondensing", target_volume=25, lambda_volume=2.0)
con = ContactPlugin(neighbor_order=2)
con.param_new("Medium", "Condensing", 16); con.param_new("Medium", "NonCondensing", 16)
con.param_new("Condensing", "Condensing", E_CC)
con.param_new("NonCondensing", "NonCondensing", 11)
con.param_new("Condensing", "NonCondensing", 11)
blob = BlobInitializer()
blob.region_new(width=5, radius=dim // 3, center=(dim // 2, dim // 2, 0),
                cell_types=("Condensing", "NonCondensing"))
spec_list = [potts, ct, vol, con, com, srf_track, blob]
'''

# --------------------------------------------------------------------------------------------- #
#  2. volume constraint -- cells relax to a target volume they do not start at
# --------------------------------------------------------------------------------------------- #
VOLUME = '''
ct = CellTypePlugin("Cell")
vol = VolumePlugin()
vol.param_new("Cell", target_volume=TARGET_V, lambda_volume=LAM_V)
con = ContactPlugin(neighbor_order=2)
con.param_new("Medium", "Cell", 16); con.param_new("Cell", "Cell", 16)
blob = BlobInitializer()
# cells are initialised at 5x5=25 sites; the target is elsewhere, so any approach to it is the
# constraint doing work rather than the initial condition already being right.
blob.region_new(width=5, radius=dim // 3, center=(dim // 2, dim // 2, 0), cell_types=("Cell",))
spec_list = [potts, ct, vol, con, com, srf_track, blob]
'''

# --------------------------------------------------------------------------------------------- #
#  3. surface constraint -- a perimeter penalty rounds cells up
# --------------------------------------------------------------------------------------------- #
SURFACE = '''
ct = CellTypePlugin("Cell")
vol = VolumePlugin(); vol.param_new("Cell", target_volume=25, lambda_volume=2.0)
sur = SurfacePlugin(); sur.param_new("Cell", target_surface=TARGET_S, lambda_surface=LAM_S)
con = ContactPlugin(neighbor_order=2)
con.param_new("Medium", "Cell", 16); con.param_new("Cell", "Cell", 16)
blob = BlobInitializer()
blob.region_new(width=5, radius=dim // 3, center=(dim // 2, dim // 2, 0), cell_types=("Cell",))
spec_list = [potts, ct, vol, sur, con, com, srf_track, blob]
'''

# --------------------------------------------------------------------------------------------- #
#  4. chemotaxis -- cells climb a diffusing field they do not produce
# --------------------------------------------------------------------------------------------- #
CHEMOTAXIS = '''
ct = CellTypePlugin("Cell")
vol = VolumePlugin(); vol.param_new("Cell", target_volume=25, lambda_volume=2.0)
con = ContactPlugin(neighbor_order=2)
con.param_new("Medium", "Cell", 16); con.param_new("Cell", "Cell", 16)
# a fixed source at the right-hand wall: a gradient with an unambiguous direction, so "did the
# population move up it" is a one-number question.
diff = DiffusionSolverFE()
f = diff.field_new("ATTR")
f.diff_data.diff_global = 0.10
f.diff_data.decay_global = 0.003
# BOUNDARYTYPESPDE is [Value, Flux, Periodic] -- "ConstantValue" is the CC3DML spelling,
# not the PyCoreSpecs one.
f.bcs.x_min_type = "Value"; f.bcs.x_min_val = 0.0
f.bcs.x_max_type = "Value"; f.bcs.x_max_val = 100.0
chem = ChemotaxisPlugin()
cf = chem.param_new("ATTR", "DiffusionSolverFE")
cf.params_new("Cell", lambda_chemo=LAM_CHEMO)
blob = BlobInitializer()
blob.region_new(width=5, radius=dim // 5, center=(dim // 4, dim // 2, 0), cell_types=("Cell",))
spec_list = [potts, ct, vol, con, com, srf_track, diff, chem, blob]
'''

# --------------------------------------------------------------------------------------------- #
#  5. growth + mitosis -- the direct comparison point with jax-morph's `cell_divide`
# --------------------------------------------------------------------------------------------- #
MITOSIS = '''
ct = CellTypePlugin("Cell")
# NO type-level volume parameters. CC3D uses the XML's per-TYPE TargetVolume when it is declared
# and ignores `cell.targetVolume` entirely -- so a growth steppable writing the per-cell value
# has no effect and the cells quietly shrink instead. The plugin is declared bare and the
# steppable owns the per-cell values. (Measured: with type-level params, volume went 25 -> 20.9
# over 400 MCS and not one cell divided.)
vol = VolumePlugin()
con = ContactPlugin(neighbor_order=2)
con.param_new("Medium", "Cell", 16); con.param_new("Cell", "Cell", 16)
blob = BlobInitializer()
blob.region_new(width=5, radius=dim // 5, center=(dim // 2, dim // 2, 0), cell_types=("Cell",))
spec_list = [potts, ct, vol, con, com, srf_track, blob]
'''

MITOSIS_STEPPABLE = '''
from cc3d.core.PySteppables import MitosisSteppableBase

class GrowDivide(MitosisSteppableBase):
    """Grow every cell's target volume, and split it once it is big enough.

    This is CompuCell3D's idiom for proliferation and it is worth noting how different it is from
    jax-morph's: there, division is a Bernoulli draw on a per-cell rate and the daughter is placed
    by a volume-conserving offset. Here growth is a target-volume ramp, division is a THRESHOLD on
    the realised volume, and the split is a geometric bisection of the cell's lattice sites. Same
    biology, different mechanism -- exactly the kind of thing the ledger has to classify.
    """
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)

    def start(self):
        # the per-cell values the bare VolumePlugin reads
        for cell in self.cell_list:
            cell.targetVolume = 25.0
            cell.lambdaVolume = 2.0

    def step(self, mcs):
        to_divide = []
        for cell in self.cell_list:
            cell.targetVolume += GROWTH_RATE
            if cell.volume > DIVIDE_V:
                to_divide.append(cell)
        for cell in to_divide:
            self.divide_cell_random_orientation(cell)

    def update_attributes(self):
        self.parent_cell.targetVolume /= 2.0
        self.clone_parent_2_child()      # daughter inherits type, targetVolume, lambdaVolume
'''

# --------------------------------------------------------------------------------------------- #
#  6. external potential -- a body force, the simplest directed motility
# --------------------------------------------------------------------------------------------- #
EXTERNAL = '''
ct = CellTypePlugin("Cell")
vol = VolumePlugin(); vol.param_new("Cell", target_volume=25, lambda_volume=2.0)
con = ContactPlugin(neighbor_order=2)
con.param_new("Medium", "Cell", 16); con.param_new("Cell", "Cell", 16)
ext = ExternalPotentialPlugin()
ext.param_new("Cell", x=LAM_X, y=0.0, z=0.0)
blob = BlobInitializer()
blob.region_new(width=5, radius=dim // 5, center=(dim // 2, dim // 2, 0), cell_types=("Cell",))
spec_list = [potts, ct, vol, con, com, srf_track, ext, blob]
'''

DEMOS = {
    "contact_adhesion": dict(
        specs=ADHESION, steppable=None, fluct=10.0,
        params={"E_CC": 2.0}, control={"E_CC": 11.0},
        headline="heterotypic_boundary", ylabel="heterotypic boundary length",
        steps=3000, dim=70,
        desc="differential adhesion sorts two cell types",
        control_desc="equal contact energies: no adhesion difference, so no sorting"),

    "volume_constraint": dict(
        specs=VOLUME, steppable=None, fluct=10.0,
        params={"TARGET_V": 60, "LAM_V": 5.0}, control={"TARGET_V": 60, "LAM_V": 0.0},
        headline="volume_mean", ylabel="mean cell volume (target 60, start 25)",
        steps=1500, dim=60,
        desc="cells relax toward a target volume they do not start at",
        control_desc="lambda_volume = 0: the constraint is present but weightless"),

    "surface_constraint": dict(
        specs=SURFACE, steppable=None, fluct=10.0,
        params={"TARGET_S": 16, "LAM_S": 2.0}, control={"TARGET_S": 16, "LAM_S": 0.0},
        headline="surface_mean", ylabel="mean cell perimeter",
        steps=1500, dim=60,
        desc="a perimeter penalty rounds cells up",
        control_desc="lambda_surface = 0: perimeter is unconstrained"),

    "chemotaxis": dict(
        specs=CHEMOTAXIS, steppable=None, fluct=10.0,
        params={"LAM_CHEMO": 200.0}, control={"LAM_CHEMO": 0.0},
        headline="com_x_mean", ylabel="mean cell x position (source at right wall)",
        steps=900, dim=55,
        desc="cells climb a diffusing gradient toward a fixed source",
        control_desc="lambda_chemo = 0: the field exists and is ignored"),

    "growth_mitosis": dict(
        specs=MITOSIS, steppable=MITOSIS_STEPPABLE, fluct=10.0,
        params={"GROWTH_RATE": 0.25, "DIVIDE_V": 50}, control={"GROWTH_RATE": 0.0, "DIVIDE_V": 50},
        headline="n", ylabel="live cells",
        steps=1500, dim=90,
        desc="target-volume growth to a division threshold",
        control_desc="growth rate 0: nothing reaches the threshold, so nothing divides"),

    "external_potential": dict(
        specs=EXTERNAL, steppable=None, fluct=10.0,
        params={"LAM_X": -20.0}, control={"LAM_X": 0.0},
        headline="com_x_mean", ylabel="mean cell x position",
        steps=1200, dim=60,
        desc="a body force drives directed motion",
        control_desc="zero force: the same model, undriven"),
}


def build_source(name, control=False):
    """The full python source that emits this demo's CC3DML."""
    d = DEMOS[name]
    src = COMMON_HEAD + d["specs"] + TAIL
    src = src.replace("FLUCT", repr(d["fluct"]))
    for k, v in (d["control"] if control else d["params"]).items():
        src = src.replace(k, repr(v))
    return src


def build_steppable(name, control=False):
    d = DEMOS[name]
    if not d["steppable"]:
        return None
    src = d["steppable"]
    for k, v in (d["control"] if control else d["params"]).items():
        src = src.replace(k, repr(v))
    return src
