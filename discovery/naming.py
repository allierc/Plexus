#!/usr/bin/env python
"""naming -- run names that sort in the order things happened, or the order they were swept.

WHY. `ls log/okuda` is how anyone first looks at a campaign, and alphabetical order is the only
order it offers. A name that sorts wrongly does not merely look untidy: a sweep read top to bottom
tells a different story than the sweep, and the reader has no way to know. Two failures, both
real, both from names written by hand on 2026-08-01:

    cfl_c10_d0p16   sorts BEFORE   cfl_c2_d0p16      (10 < 2 as text)
    cfl_..._d10     sorts BEFORE   cfl_..._d2

The loop's own names were already right --- `r002c_00_5e3159` is zero-padded on both the round and
the slot, so round 10 follows round 2 and slot 10 follows slot 2. This module exists so a name
written by hand cannot be worse than one written by the loop.

THE CONVENTION
================================================================================================
    <campaign><NNN><mode><_SS>_<tag>          the loop:   r002c_00_5e3159
    <sweep>_<key><VVVpVVV>_<key><VVVpVVV>     a sweep:    cfl_c001p300_d000p160

Every number is fixed width, zero padded, decimal point written `p` so the name stays a legal
path component. Fixed width is the whole point: `p` in place of `.` only helps if the digits
before it are padded too, which is exactly the mistake made today.
"""
from __future__ import annotations

INT_W, DEC_W = 3, 3


def num(v, int_w=INT_W, dec_w=DEC_W):
    """A number as a fixed-width, zero-padded, sortable token. 1.3 -> 001p300, 10 -> 010p000."""
    s = f"{float(v):0{int_w + dec_w + 1}.{dec_w}f}"
    return s.replace(".", "p").replace("-", "m")


def sweep_name(prefix, **params):
    """`sweep_name('cfl', c=1.3, d=0.16)` -> 'cfl_c001p300_d000p160'.

    Keys are emitted in the order given, not sorted, because a sweep's first axis should lead
    the name -- that is what makes a directory listing read as the sweep.
    """
    return "_".join([prefix] + [f"{k}{num(v)}" for k, v in params.items()])


def run_name(round_id, slot, tag, mode="c"):
    """The loop's own convention, in one place: r002c_00_5e3159."""
    return f"r{int(round_id):03d}{mode[0]}_{int(slot):02d}_{tag}"


def sorts_correctly(names):
    """Does this set of names sort the way its numbers do? Returns (ok, first offending pair)."""
    import re
    keyed = []
    for n in names:
        nums = [float(x.replace("p", ".")) for x in re.findall(r"\d+p\d+|\d+", n)]
        keyed.append((nums, n))
    by_text = [n for _, n in sorted(keyed, key=lambda kv: kv[1])]
    by_num = [n for _, n in sorted(keyed, key=lambda kv: kv[0])]
    if by_text == by_num:
        return True, None
    for a, b in zip(by_text, by_num):
        if a != b:
            return False, (a, b)
    return False, None


if __name__ == "__main__":
    old = ["cfl_c1p3_d0p16", "cfl_c2_d0p16", "cfl_c10_d0p16", "cfl_c0p05_d10", "cfl_c0p05_d2"]
    new = [sweep_name("cfl", c=c, d=d) for c, d in
           [(1.3, 0.16), (2, 0.16), (10, 0.16), (0.05, 10), (0.05, 2)]]
    for label, names in (("hand-written today", old), ("through sweep_name", new)):
        ok, bad = sorts_correctly(names)
        print(f"\n  {label}: {'sorts correctly' if ok else 'OUT OF ORDER ' + str(bad)}")
        for n in sorted(names):
            print("   ", n)
    rounds = [run_name(r, s, "5e3159") for r in (2, 10) for s in (0, 1, 10)]
    ok, _ = sorts_correctly(rounds)
    print(f"\n  the loop's own names: {'sorts correctly' if ok else 'OUT OF ORDER'}")
    assert sorts_correctly(new)[0] and ok, "the convention must sort"
    print("\nnaming OK")
