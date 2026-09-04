"""One signature per variant -- R1(c) of the apico-basal promotion.

THE DEFECT IS SHIPPED, NOT HYPOTHETICAL, and that is why this file leads with a real operator rather
than a fixture. `register_operator` built the contract's typed signature from `cls.signature()`
inside the `if contract is None:` branch -- i.e. ONLY on the first registration of a name. Every
later variant went to the `else` branch, which checks KIND and nothing else, so whichever variant
imported first defined the contract's signature permanently.

`cell_chem_diffuse` is the live instance. Its contract carried `graph_laplacian`'s

    {inputs: [cell], reads: [chem], maps: [edge_index]}

while `interface_weighted` declares INPUTS = ["cell", "vertex"], READS = ["chem", "pos"],
MAPS = ["E_srce", "E_trgt", "E_face"] -- a second SET, a second state BLOCK and three different
MAPS, none of which appeared anywhere in the contract that `plexus2.tex` says registration records
and that the atlas and the validator read as truth.

WHY IT BLOCKS THE PROMOTION. Every apico-basal variant widens its signature the same way:
`cell_mechanics[model: apicobasal]` reads and writes a second block (`sep`) on the vertex set.
Registered against the old code it would inherit `reads: [pos], writes: [pos]` and the contract
would describe the mid-surface model while the operator touched two blocks.

`signature` IS NOT REMOVED. The audit tools and the atlas already call it, and for the ~100
contracts with a single variant nothing changes at all -- `signatures[default]` is the same object.
"""
from __future__ import annotations

import pytest

import plexus.operators                                        # noqa: F401  self-registers
from plexus.models.base import Lateral
from plexus.models.registry import _OP_CONTRACTS as CON, register_operator


def test_the_shipped_divergence_is_now_visible():
    """The regression this rung exists for, asserted on the operator it was found on."""
    c = CON["cell_chem_diffuse"]
    assert set(c.signatures) == {"graph_laplacian", "interface_weighted"}
    gl, iw = c.signatures["graph_laplacian"], c.signatures["interface_weighted"]

    assert gl["inputs"] == ["cell"]
    assert iw["inputs"] == ["cell", "vertex"], "the second SET is still invisible"
    assert iw["reads"] == ["chem", "pos"], "the second state BLOCK is still invisible"
    assert iw["maps"] == ["E_srce", "E_trgt", "E_face"], "the three MAPS are still invisible"
    assert gl != iw, (
        "the two variants read different things; a single shared signature cannot be right for "
        "both, and before this change the contract carried whichever registered first")


def test_the_default_signature_is_unchanged():
    """THE BACK-COMPATIBILITY CLAIM. `contract.signature` is what every existing caller reads."""
    for name, c in CON.items():
        if not c.signatures:
            continue
        assert c.signatures[c.default] == c.signature, (
            f"{name}: `signature` no longer agrees with the default variant's entry")


def test_every_variant_has_an_entry():
    """`signatures` is complete, not "the ones that came second".

    A dict filled only in the extension branch would hold every variant EXCEPT the default, which is
    the shape most likely to pass a spot check and fail in the atlas.
    """
    for name, c in CON.items():
        missing = set(c.implementations) - set(c.signatures)
        assert not missing, f"{name}: variants with no signature recorded: {sorted(missing)}"


def test_a_single_variant_contract_is_untouched():
    """Most contracts have one variant. For them this change must be a no-op in every field."""
    solo = [c for c in CON.values() if len(c.implementations) == 1]
    assert solo, "no single-variant contract found -- the sample is wrong, not the code"
    for c in solo:
        assert list(c.signatures) == [c.default]
        assert c.signatures[c.default] == c.signature


def test_a_new_variant_records_its_own_signature():
    """Registered live, because the defect was in the REGISTRATION path and nowhere else.

    A test that asserted over already-imported operators would pass on a `signatures` dict populated
    by any means at all; this one drives `register_operator` itself.
    """
    @register_operator("_sig_probe", set="cell", kind="lateral", family="fields")
    class _Base(Lateral):
        INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]; MAPS = []

    @register_operator("_sig_probe", set="cell", kind="lateral", family="fields",
                       implementation="wider")
    class _Wider(Lateral):
        INPUTS = ["cell", "vertex"]; OUTPUTS = ["cell"]
        READS = ["chem", "pos"]; WRITES = ["chem"]; MAPS = ["E_srce"]

    c = CON["_sig_probe"]
    assert c.signatures["default"]["inputs"] == ["cell"]
    assert c.signatures["wider"]["inputs"] == ["cell", "vertex"], (
        "the second variant inherited the first's signature -- the defect is back")
    assert c.signatures["wider"]["reads"] == ["chem", "pos"]
    assert c.signature == c.signatures["default"], "the contract's default signature moved"


def test_a_variant_still_may_not_change_the_kind():
    """The one guard the extension branch already had must survive this change."""
    @register_operator("_kind_probe", set="cell", kind="lateral", family="fields")
    class _K(Lateral):
        INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]; MAPS = []

    with pytest.raises(ValueError, match="may not change the kind"):
        @register_operator("_kind_probe", set="cell", kind="structural", family="fields",
                           implementation="wrong_kind")
        class _K2(Lateral):
            INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]; MAPS = []
