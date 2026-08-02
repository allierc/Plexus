#!/usr/bin/env python
"""term -- a little colour and a few icons, only where they carry meaning.

RESTRAINT IS THE POINT. A terminal where everything is coloured is a terminal where nothing
stands out, and a round already prints several hundred lines. Colour is spent on the three things
a reader scans for:

    a REFUSAL        red     -- something was rejected, and the reason follows
    a VERDICT        green   -- something passed, or a round closed
    a WARNING        yellow  -- it worked, but not the way you would want

Everything else stays plain. Timings, poll lines and traces are dim, because they are context
rather than news.

IT SWITCHES ITSELF OFF WHEN NOT A TTY. Every one of these rounds is run under nohup with stdout
to a file, and escape codes in a log are worse than no colour at all -- they break grep, they
break the note, and they turn `tail -f` output into noise the moment it is pasted anywhere. The
check is `isatty`, and NO_COLOR is honoured because it is the convention.
"""
from __future__ import annotations

import os
import sys

_ON = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code):
    return (lambda s: f"\033[{code}m{s}\033[0m") if _ON else (lambda s: s)


red, green, yellow, blue, dim, bold = (_c("31"), _c("32"), _c("33"), _c("36"),
                                       _c("2"), _c("1"))

# One icon per KIND of event, not per line. They are here to be recognised at a glance in a wall
# of text; a different picture on every line is the same as no pictures.
I = {
    "act": "▶",        # a phase beginning
    "ok": "✓",         # passed, admitted, closed
    "no": "✗",         # refused, rejected
    "warn": "⚠",       # worked, but not as wanted
    "run": "⚙",        # the cluster is working
    "read": "◉",       # an agent read something
    "think": "◌",      # an agent is deciding
    "save": "■",       # written to disk
    "time": "⏱",       # a measurement of cost
}


def act(title, detail="", clock=""):
    """A phase banner. The one place a rule is drawn, because it is the one place it helps."""
    line = f"{I['act']} {title}"
    if clock:
        line += f"   {dim(clock)}"
    if detail:
        line += f"   {dim(detail)}"
    return f"\n{blue('=' * 96)}\n{bold(blue(line))}\n{blue('=' * 96)}"


def ok(msg, icon="ok"):
    return f"  {green(I[icon])} {msg}"


def no(msg):
    return f"  {red(I['no'])} {red(msg)}"


def warn(msg):
    return f"  {yellow(I['warn'])} {yellow(msg)}"


def step(msg, clock="", icon="think"):
    return f"    {dim(clock)} {I[icon]} {msg}" if clock else f"    {I[icon]} {msg}"


def quiet(msg):
    """Context, not news: poll lines, traces, timings."""
    return dim(f"  {msg}")


def verdict(word):
    """Colour a verdict by what it means, so a wall of them can be scanned."""
    w = str(word).lower()
    if w in ("valid", "confirmed", "supported", "ok", "pass", "admitted", "continue"):
        return green(word)
    if w in ("invalid", "refuted", "falsified", "fail", "refused", "stop", "aborted"):
        return red(word)
    if w in ("ambiguous", "inconclusive", "partial", "unchecked", "warn", "roll_back"):
        return yellow(word)
    return str(word)


if __name__ == "__main__":
    print(f"colour {'ON' if _ON else 'OFF (not a tty, or NO_COLOR)'}")
    print(act("ACT 2 - MEASURE", "the only expensive step", "[+03:12]"))
    print(ok("6 run(s) admitted"))
    print(no("r001c_03 is not evidence: P2_BUFFER_SATURATED"))
    print(warn("specimen ambiguous -- the array filled before the biology did"))
    print(step("Collector: building the round record", "[+15:00]", "save"))
    print(quiet("[22:31] run/pend=2 done=4 of 6"))
    print("  verdicts: " + "  ".join(verdict(v) for v in
                                     ("valid", "invalid", "ambiguous", "confirmed", "refuted")))
