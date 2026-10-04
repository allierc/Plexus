"""The main question (INSTRUCTION.md): exp.py launch refuses a batch until the question is answered after the last landing."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp  # noqa: E402

Q = "How does the bud emerge in the best run, and how does that differ from Wang 2021's mechanism?"
A = ("the inner mass's volume clock pushes the layer outward and the membrane holds it everywhere except at the "
     "hole, so the growth escapes there; Wang's surface layer instead outgrows its interior by re-insertion and folds")


def _md(tmp_path, body):
    md = tmp_path / "exp98_t.md"
    md.write_text(f"---\nnumber: 98\nquestion: \"{Q}\"\n---\n{body}")
    (tmp_path / "exp98_t" / "landings").mkdir(parents=True)
    rep = tmp_path / "exp98_t" / "landings" / "x.md"
    rep.write_text("landed")
    old = time.time() - 3600
    os.utime(rep, (old, old))
    return str(md)


def test_no_answer_after_the_landing_refuses(tmp_path):
    p = _md(tmp_path, "\n## The question\n\n- 2020-01-01 10:00 -- " + A + "\n")
    fm, body = exp.load(p)
    assert "answer the main question" in exp.question_answered(p, fm, body)


def test_a_fresh_answer_lets_the_batch_go(tmp_path):
    now = time.strftime("%Y-%m-%d %H:%M")
    p = _md(tmp_path, f"\n## The question\n\n- {now} -- {A}\n\n## Decisions\n")
    fm, body = exp.load(p)
    assert exp.question_answered(p, fm, body) is None


def test_a_short_answer_does_not_count_and_no_question_no_check(tmp_path):
    now = time.strftime("%Y-%m-%d %H:%M")
    p = _md(tmp_path, f"\n## The question\n\n- {now} -- growth escapes at the hole\n")
    fm, body = exp.load(p)
    assert exp.question_answered(p, fm, body) is not None
    assert exp.question_answered(p, {}, body) is None
