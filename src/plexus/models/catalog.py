"""The filled registry: entities, operators, and fields at every level.

Each entry is a stub tagged with its level and kind, with a docstring naming the
source class to port from the sibling repos. Importing this module populates the
registries; `registry.catalog_summary()` then lists the full action set, and
`registry.operators_at_level("cell")` enumerates what can run at a given scale.

This is the porting worklist made executable: fill each `forward` by lifting the
named class into the Operator / Field contract in base.py. It spans the full
seven-kind operator vocabulary: the four set-dynamics kinds that return a delta
(`lateral`, `aggregate`, `broadcast`, `exchange`), `field` (a field's own
self-dynamics), `rewire` (changes the relation E), and `structural` (changes the
entity set |S|).

NB: a stub here may share a name with a *validated* operator in `plexus.operators`
(e.g. `diffuse`, `radius_graph`); this module is the worklist and is NOT imported
alongside the validated package (it would re-register those names).
"""

from __future__ import annotations

from plexus.models.base import (
    Lateral, Aggregate, Broadcast, Exchange, FieldUpdate, Rewire, Structural, Field,
)
from plexus.models.registry import (
    register_entity, register_operator, register_field,
)


# --------------------------------------------------------------------------- #
#  Entities  (node kinds -> the sets / levels)
# --------------------------------------------------------------------------- #
@register_entity("molecule", level=0)
class Molecule:
    """Leaf. State = concentration. Lives inside a cell (metabolite / protein)."""

@register_entity("particle", level=0)
class Particle:
    """Leaf. State = position, velocity, deformation F. MPM material point."""

@register_entity("cell", level=1)
class Cell:
    """State = position, type, phase. A set of particles and/or molecules."""

@register_entity("population", level=2)
class Population:
    """A set of cells (tissue domain / cell population)."""

@register_entity("organism", level=3)
class Organism:
    """A set of populations."""


# --------------------------------------------------------------------------- #
#  Lateral operators  (within-set ODE interaction + discrete Laplacian)
# --------------------------------------------------------------------------- #
@register_operator("signal", level="cell", kind="lateral")
class SignalOperator(Lateral):
    """Connectome signal propagation. Port: NeuralGraph Signal_Propagation_*."""

@register_operator("interaction", level="cell", kind="lateral")
class InteractionOperator(Lateral):
    """Pairwise cell interaction. Port: cell-gnn CellGNN / Interaction_Particle."""

@register_operator("boids", level="cell", kind="lateral")
class BoidsOperator(Lateral):
    """Flocking / active matter. Port: ParticleGraph boids, active_matter_ode."""

@register_operator("gravity", level="cell", kind="lateral")
class GravityOperator(Lateral):
    """Long-range attraction. Port: ParticleGraph gravity_ode."""

@register_operator("adhesion", level="cell", kind="lateral")
class AdhesionOperator(Lateral):
    """Junction / contact mechanics (MISSING in repos — new for tissue)."""

@register_operator("sph", level="particle", kind="lateral")
class SmoothParticleOperator(Lateral):
    """Smoothed-particle hydrodynamics. Port: ParticleGraph Interaction_Smooth_Particle."""

@register_operator("spring", level="particle", kind="lateral")
class SpringOperator(Lateral):
    """Elastic spring network. Port: cell-gnn particle_spring_force_ode."""

@register_operator("population_coupling", level="population", kind="lateral")
class PopulationCouplingOperator(Lateral):
    """Population-population coupling (MISSING — new for multi-population tissue)."""


# --------------------------------------------------------------------------- #
#  Aggregate (up) operators  (children -> parent over the partition)
# --------------------------------------------------------------------------- #
@register_operator("centroid", level="cell", kind="aggregate")
class CentroidAggregate(Aggregate):
    """Cell position = aggregate of its particles. Drives 'MPM moves the cell'."""

@register_operator("metabolic_state", level="cell", kind="aggregate")
class MetabolicAggregate(Aggregate):
    """Cell metabolic state = aggregate of its molecules."""

@register_operator("pool_population", level="population", kind="aggregate")
class PopulationAggregate(Aggregate):
    """Population state = aggregate of its cells."""


# --------------------------------------------------------------------------- #
#  Broadcast (down) operators  (parent -> children along the partition)
# --------------------------------------------------------------------------- #
@register_operator("broadcast_force", level="particle", kind="broadcast")
class ForceBroadcast(Broadcast):
    """Cell-level decision -> force on its particles. 'signaling moves the cell'."""

@register_operator("broadcast_field", level="cell", kind="broadcast")
class FieldBroadcast(Broadcast):
    """Population-level field/context -> per-cell input."""


# --------------------------------------------------------------------------- #
#  Exchange operators  (set <-> field / set <-> set; the bipartite operator)
# --------------------------------------------------------------------------- #
@register_operator("p2g_g2p", level="particle", kind="exchange")
class MPMExchange(Exchange):
    """Particle <-> background grid. Port: MPM_pytorch Interaction_MPM (MPM_P2G)."""

@register_operator("reaction", level="molecule", kind="exchange")
class ReactionExchange(Exchange):
    """Metabolite <-> reaction (stoichiometry). Port: MetabolismGraph Metabolism_Propagation."""

@register_operator("secrete_sense", level="cell", kind="exchange")
class SecreteSenseExchange(Exchange):
    """Cell <-> morphogen field (deposit / sample gradient)."""

@register_operator("particle_field", level="particle", kind="exchange")
class ParticleFieldExchange(Exchange):
    """Particle <-> continuous field. Port: ParticleGraph Interaction_Particle_Field."""


# --------------------------------------------------------------------------- #
#  Field self-dynamics  (field -> field; mutate the grid, return {})
# --------------------------------------------------------------------------- #
@register_operator("react", level="field", kind="field")
class ReactionDiffusion(FieldUpdate):
    """Reaction-diffusion kinetics (Gray-Scott / RPS). Port: ParticleGraph RD_*."""

@register_operator("wave_acoustic", level="field", kind="field")
class WaveAcoustic(FieldUpdate):
    """Acoustic wave (leapfrog). Port: the_well acoustic_scattering."""

@register_operator("advect", level="field", kind="field")
class Advect(FieldUpdate):
    """Semi-Lagrangian advection by a carrier flow. Port: the_well navier_stokes."""

# (validated field self-dynamics already in plexus.operators: diffuse, decay, playback)


# --------------------------------------------------------------------------- #
#  Rewire operators  (rebuild the relation E = edge_index; emit no delta)
# --------------------------------------------------------------------------- #
@register_operator("neighbour_graph", level="particle", kind="rewire")
class NeighbourGraph(Rewire):
    """Proximity graph each tick (validated as `radius_graph`). Port: geometry.edges_radius_blockwise."""

@register_operator("membrane_ring", level="particle", kind="rewire")
class MembraneRing(Rewire):
    """Cyclic membrane bonds around a cell (MISSING -- new for deformable cells)."""


# --------------------------------------------------------------------------- #
#  Structural operators  (change cardinality |S| / membership via occupancy)
# --------------------------------------------------------------------------- #
@register_operator("divide", level="cell", kind="structural")
class Divide(Structural):
    """Mitosis: one cell -> two on a fixed buffer. Port: prototype ops_grow divide."""

@register_operator("die", level="cell", kind="structural")
class Die(Structural):
    """Apoptosis / efflux: retire a cell (occ -> 0). Port: prototype death."""

@register_operator("spawn", level="cell", kind="structural")
class Spawn(Structural):
    """Birth / nucleation: wake a dormant slot. Port: prototype dicty inflow."""


# --------------------------------------------------------------------------- #
#  Fields  (continuum discretizations)
# --------------------------------------------------------------------------- #
@register_field("grid", frame="eulerian")
class GridField(Field):
    """Regular Eulerian lattice + P2G/G2P kernels. Port: MPM_pytorch grid."""

@register_field("mesh", frame="mesh")
class MeshField(Field):
    """Mesh + Laplacian diffusion. Port: ParticleGraph Mesh_Laplacian."""

@register_field("implicit", frame="implicit")
class ImplicitField(Field):
    """Coordinate network f(t,x,y). Port: Siren_Network (all repos)."""
