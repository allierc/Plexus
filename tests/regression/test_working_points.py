"""Layer A + B1 of tests/REGRESSION_PLAN.md: every registered working point, rerun as a cut on the
current code and compared with its fingerprint.

One test per JSON in `fingerprints/`. The cut runs through Plexus_Main on CUDA (the warp gradient
is CUDA-only and that is where the origin defect lived; CPU is not a substitute). Frame 0 is exact
in cell and half-edge count -- a seed that changed is a default that changed -- and every other
checkpoint is compared within the bands in `tools/regression_lib.BANDS`. Frame time is asserted
only on the same GPU model as the fingerprint, reported otherwise.

    pytest tests/regression -m regression            the whole registry, ~15 min on an A6000 (deselected by default)
    pytest tests/regression -m regression -k cvd2    one working point
    PLEXUS_REGRESSION_DEVICE=cuda:1 pytest ...       pick the card

Refreshing a fingerprint: `python tools/fingerprint.py refresh <name> --because "..."`, in a commit
with no source change.
"""
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import regression_lib as R  # noqa: E402

DEVICE = os.environ.get("PLEXUS_REGRESSION_DEVICE", "cuda:0")
QUICK = {"cvd2_adder_tension": 60, "cv_kv_double": 60, "apop2_ks0p1": 60}

pytestmark = pytest.mark.regression


def _cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:                                    # noqa: BLE001
        return False


def _spec(ref):
    if ref.get("archive") and os.path.exists(os.path.join(ref["archive"], "spec.yaml")):
        return yaml.safe_load(open(os.path.join(ref["archive"], "spec.yaml")))
    return yaml.safe_load(open(os.path.join(ROOT, "config", ref["family"], ref["name"] + ".yaml")))


@pytest.mark.parametrize("name", R.registered())
def test_fingerprint(name, request):
    if not _cuda_ok():
        pytest.fail("the regression series needs CUDA (the warp gradient path is where the defects live); "
                    "none available")
    ref = R.read_fingerprint(name)
    cut = ref["cut"]["n_frames"]
    if request.config.getoption("--quick", default=False):
        if name not in QUICK:
            pytest.skip("not in the quick set")
        cut = min(cut, QUICK[name])
    cps = [f for f in ref["cut"]["checkpoints"] if f <= cut]
    got = R.fingerprint_fresh(_spec(ref), ref["family"], cut, DEVICE, "test", checkpoints=cps,
                              stride=int(ref["cut"].get("record_stride", 1)))
    ref_cut = dict(ref, cut=dict(ref["cut"], checkpoints=cps))   # only checkpoints inside the cut
    bad = R.compare(ref_cut, got)
    tv, note = R.compare_timing(ref, got)
    print(f"[regression] {name}: {note}")
    assert not bad and tv is None, f"{name} drifted from its fingerprint ({ref['archived_on']}, commit {ref['generated_at_commit']}):\n  " + "\n  ".join(bad + ([tv] if tv else []))


def test_registry_is_not_empty():
    assert R.registered(), "no fingerprints registered: run tools/fingerprint.py register <archive>"
