"""The Platynereis region, and the facts about it that nothing is allowed to break.

Run: `pytest tests/test_platynereis.py -v`

These are not unit tests of code. They are assertions about a DATASET that the whole larva model
is built on, and each one exists because getting it wrong is silent: every picture still renders,
every run still finishes, and the answer is simply the wrong animal.

The one that matters most is the direction of the connectome. It was wrong -- `build_region.py`
read the adjacency matrix transposed, so all 4,664 edges pointed backwards -- and nothing in the
output looked odd. It was caught by asking which cell classes RECEIVE synapses, and the answer
came back "the muscles send 319 and receive none".
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

REGION = "platynereis_larva_4117"


def region_dir():
    from plexus.paths import graphs_data_path
    d = os.path.join(graphs_data_path(), "neural_regions", REGION)
    if not os.path.isdir(d):
        pytest.skip(f"no region at {d}")
    return d


@pytest.fixture(scope="module")
def data():
    d = region_dir()
    return {
        "dir": d,
        "z": np.load(os.path.join(d, "neurons.npz"), allow_pickle=True),
        "c": np.load(os.path.join(d, "connectome.npz"), allow_pickle=True),
        "m": json.load(open(os.path.join(d, "manifest.json"))),
    }


def _mask(z, cls):
    return np.asarray(z["cell_class_id"]) == list(z["cell_class_names"]).index(cls)


# ------------------------------------------------------------------ the direction of the wiring

def test_effectors_receive_synapses_and_do_not_send_them(data):
    """Muscle and ciliary-band cells are EFFECTORS: they are postsynaptic, never presynaptic.

    This is the assertion that would have caught the transposed adjacency matrix. Read backwards,
    the file gave 840 muscle cells sending 319 synapses into the nervous system and receiving
    none -- an effector with no postsynaptic sites is not a thing.
    """
    z, ei = data["z"], np.asarray(data["c"]["edge_index"])
    for cls in ("muscle", "ciliary band"):
        m = _mask(z, cls)
        sends, receives = int(m[ei[0]].sum()), int(m[ei[1]].sum())
        assert receives > 10 * max(sends, 1), (
            f"{cls}: {sends} edges sent and {receives} received. An effector receives. "
            f"If this fails the adjacency matrix has been read transposed again -- rows are "
            f"PREsynaptic. See tools/platynereis_fix_direction.py.")


def test_MC_innervates_the_prototroch(data):
    """The MC cell drives the prototroch, and is therefore PREsynaptic to all of it.

    There is exactly one cell of type `MC` in the compendium and it contacts all 23 prototroch
    cells across 340 synapses. MC is the larva's giant head ciliomotor neuron: Verasztó et al.
    (2017), eLife 6:e26000, "Ciliomotor circuitry underlying whole-body coordination of ciliary
    activity in the Platynereis larva". This is the fact the whole motor pathway rests on -- if
    it points the other way, R5 drives the cilia from the cilia.
    """
    z, ei = data["z"], np.asarray(data["c"]["edge_index"])
    ct = z["celltype"]
    mc = np.array([str(t) == "MC" for t in ct])
    pt = np.array([str(t) == "prototroch" for t in ct])
    assert mc.sum() == 1, f"expected one MC cell, found {int(mc.sum())}"
    fwd = int((mc[ei[0]] & pt[ei[1]]).sum())
    rev = int((pt[ei[0]] & mc[ei[1]]).sum())
    assert fwd >= 20 and rev == 0, (
        f"MC->prototroch {fwd}, prototroch->MC {rev}. MC innervates the prototroch; it is "
        f"presynaptic. A reversal here inverts every motor pathway in the model.")


def test_sensory_neurons_send_more_than_they_receive(data):
    """A sensory cell's job is to report. Backwards, they received 1,817 and sent 1,006."""
    z, ei = data["z"], np.asarray(data["c"]["edge_index"])
    m = _mask(z, "Sensory neuron")
    sends, receives = int(m[ei[0]].sum()), int(m[ei[1]].sum())
    assert sends > receives, f"sensory neurons send {sends} and receive {receives}"


# ------------------------------------------------------------------ what the model relies on

def test_cell_classes_are_contiguous_in_row_order(data):
    """Every class occupies ONE unbroken run of rows.

    `type_layout: ordered` hands rows to types in the order the types are declared, so declaring
    the 18 classes as 18 types only places them correctly while this holds. If it ever stops
    holding, every cell in every picture is silently mislabelled and none of them look it.
    """
    cid = np.asarray(data["z"]["cell_class_id"])
    runs = int((np.diff(cid) != 0).sum()) + 1
    assert runs == len(set(cid.tolist())), (
        f"{len(set(cid.tolist()))} classes over {runs} runs of rows -- they are no longer "
        f"contiguous, and `type_layout: ordered` would mislabel every cell.")


def test_counts_and_cube_match_the_manifest(data):
    """`neural_seed` refuses a count or a `length_um` that disagrees with the manifest, so the
    specs hard-code both. These are those numbers."""
    z, m = data["z"], data["m"]
    assert z["xyz"].shape[0] == 4117 == m["region"]["n_neurons"]
    assert m["region"]["side_um"] == pytest.approx(195.9)
    assert int(m["n_edges"]) == np.asarray(data["c"]["edge_index"]).shape[1] == 4664


def test_body_id_is_the_row_index(data):
    """`morphology_index.json` is keyed by `body_id`, and only 2,896 of 4,117 cells have a CATMAID
    skid. Keying by skid would silently drop 1,221 cells' morphology -- most of the epidermis and
    muscle, which is most of what the animal looks like."""
    assert np.array_equal(np.asarray(data["z"]["body_id"]), np.arange(4117))


def test_the_head_is_at_low_z(data):
    """The region's z runs HEAD to TAIL, which is why every view of this animal is rolled 180
    degrees: a z-up camera would draw it upside down against every figure in its own paper."""
    z = data["z"]
    seg, names = np.asarray(z["segment_id"]), list(z["segment_names"])
    xyz = np.asarray(z["xyz"], float)
    head = xyz[seg == names.index("episphere"), 2].mean()
    tail = xyz[seg == names.index("pygidium"), 2].mean()
    assert head < tail, f"episphere z {head:.0f} nm is not anterior to pygidium z {tail:.0f} nm"


def test_the_ciliary_band_is_wired(data):
    """R5 drives the cilia from the connectome, so the cilia must be IN the connectome.

    They are: 73 of the 74 ciliary-band cells carry edges, which is the exception among the
    non-neuronal classes -- the adjacency matrix joins by cell name and most of the epidermis has
    none. Degree zero means unjoined, not unwired.
    """
    z, ei = data["z"], np.asarray(data["c"]["edge_index"])
    m = _mask(z, "ciliary band")
    deg = np.zeros(m.shape[0], int)
    np.add.at(deg, ei[0], 1)
    np.add.at(deg, ei[1], 1)
    wired = int((deg[m] > 0).sum())
    assert wired >= 70, f"only {wired} of {int(m.sum())} ciliary-band cells carry an edge"
