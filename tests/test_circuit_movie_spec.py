"""`training.circuit_movie` (plexus.tasks.plot_trainer.circuit_movie): read by a corpus task, refused elsewhere,
its keys checked -- every key is read or refused (trainer.load)."""
import os

import pytest
import yaml

from plexus import trainer as T

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EYE = os.path.join(ROOT, "config", "training", "neural_eye", "zf_eye_rig.yaml")


def _write(tmp_path, spec, name):
    d = tmp_path / "neural_eye"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(spec))
    return str(p)


@pytest.mark.skipif(not os.path.exists(EYE), reason="the eye rig spec is not in this checkout")
def test_circuit_movie_is_read_by_a_corpus_task(tmp_path):
    s = yaml.safe_load(open(EYE))
    s["training"]["circuit_movie"] = {"trials": 2, "fps": 30, "stride": 2}
    sp = T.load(_write(tmp_path, s, "eye_cm"))
    assert sp["training"]["circuit_movie"] == {"trials": 2, "fps": 30, "stride": 2}
    s["training"]["circuit_movie"] = {}
    assert T.load(_write(tmp_path, s, "eye_cm0"))["training"]["circuit_movie"] == {}


@pytest.mark.skipif(not os.path.exists(EYE), reason="the eye rig spec is not in this checkout")
def test_circuit_movie_refuses_an_unknown_key(tmp_path):
    s = yaml.safe_load(open(EYE))
    s["training"]["circuit_movie"] = {"trials": 2, "frames": 100}
    with pytest.raises(ValueError, match="circuit_movie"):
        T.load(_write(tmp_path, s, "eye_bad"))
