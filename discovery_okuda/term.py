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
import re
import sys

_ON = (bool(os.environ.get("FORCE_COLOR")) or sys.stdout.isatty()) \
      and not os.environ.get("NO_COLOR")


def _c(code):
    return (lambda s: f"\033[{code}m{s}\033[0m") if _ON else (lambda s: s)


red, green, yellow, blue, dim, bold = (_c("31"), _c("32"), _c("33"), _c("36"),
                                       _c("2"), _c("1"))
# No magenta or violet: hard to read on Cedric's terminal, and a colour nobody can read is a
# colour that carries nothing. The palette is bright, high-contrast hues only.
cyan, orange, teal, gold, sky = (_c("38;5;51"), _c("38;5;208"), _c("38;5;79"),
                                _c("38;5;220"), _c("38;5;117"))

# A COLOUR PER VOICE, so a wall of agent text can be read by who is speaking without reading it.
# Grouped by what the role is FOR rather than picked at random: the two that look at the SPECIMEN
# are green-ish, the two that look at the RECORD are violet, the two that look ACROSS rounds are
# orange, and the eye-check -- the only one that looks at a picture -- is the one that stands out.
VOICE = {
    "biologist":     teal,       # is it a tissue
    "metrologist":   teal,       # is the instrument sound
    "eye-check":     gold,       # the only role that looks at SHAPE
    "reader":        cyan,       # what happened in this run
    "interpreter":   sky,        # what happened this round, and why
    "meta-review":   sky,        # what should change next round
    "supervisor":    orange,     # what runs next, and how much
    "archivist":     orange,     # across the whole history
    "diagnostician": red,        # why the apparatus failed
    "proposer":      blue,       # what to test next
    "peer-review":   blue,       # is it worth testing
    # ADDED with Phase 7. Both speak every round and had no colour, so their lines read as
    # debug output rather than as a role talking. Teal is the "is this sound?" family, which is
    # what they both are: one checks the instrument, the other checks the inference.
    "logic":         teal,       # is the conclusion earned
    "metrologist":   teal,       # is the instrument sound
    "escalate":      orange,     # the Supervisor changing the envelope
}


def voice(who):
    """The colour for a speaker, matched on the role name however it is decorated."""
    w = str(who).lower()
    for k, f in VOICE.items():
        if w.startswith(k):
            return f
    return bold

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


HEADLINE_ASK = """
END YOUR REPLY WITH ONE LINE, exactly this shape and nothing after it:

    HEADLINE: <at most 90 characters, the ONE thing a person watching the terminal should know>

It is read by a human while the round is still running, so make it the finding and not a
description of your task. "Chemistry extinct in 4 of 6; the activator went non-finite by frame
115" is a headline. "Analysed the runs and recorded the results" is not."""


def headline(text, fallback="", who=""):
    """The agent's own one-line summary, if it gave one.

    ASKED FOR, not extracted. Parsing an agent's prose for a summary means guessing which
    sentence mattered, and the guess is wrong exactly when the agent had something unexpected to
    say. A field it fills itself cannot be misread that way -- and an agent that writes its
    product to a FILE returns only a receipt, so without this there is nothing to print at all.
    """
    import re
    m = re.search(r"^\s*HEADLINE:\s*(.+?)\s*$", str(text or ""), re.M)
    if m:
        return m.group(1)[:120]
    t = " ".join(str(fallback or text or "").split())
    return t[:120]


def say(who, text, sentences=1, width=100):
    """What an agent actually said, trimmed to N sentences. Quoted, never paraphrased.

    A round prints what every role DID and almost nothing of what any of them THOUGHT. The
    reasoning is in the ledger and in analysis.md, which nobody reads while a round is running --
    so the one line that would tell you whether an agent is being sensible never reaches the
    terminal at all.
    """
    import re
    t = " ".join(str(text or "").split())
    if not t:
        return f"{voice(who)(I['think'])} {dim(who + ': (said nothing)')}"
    parts = [x for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()][:sentences]
    t = " ".join(parts)
    col = voice(who)
    # WRAPPED ON WORDS, NOT SLICED. This used to cut at t[:width] and then t[width:] again, so a
    # role's line broke mid-token -- and the continuations were indented six spaces, which put
    # the same sentence at three different left margins depending on how long it was.
    import textwrap
    head = f'{col(I["read"])} {col(bold(who))}: '
    lines = textwrap.fill(t, width=width, break_long_words=False,
                          break_on_hyphens=False).splitlines()
    out = [head + (lines[0] if lines else "")]
    out += [dim(l) for l in lines[1:]]
    return "\n".join(out)


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


# ------------------------------------------------------------------ colour every [role] line
class _Colourise:
    """Wrap stdout so any line beginning `[role]` gets that role's colour.

    THIRTEEN PRINT SITES SAID `[supervisor]`, `[proposer]`, `[archivist]` in no colour at all,
    which is why those lines read as debug output while the `say()` lines read as a role talking.
    Colouring them at the call sites means rewriting thirteen f-strings and missing the
    fourteenth; doing it here catches every one, including the ones written next week.

    Off when stdout is not a TTY, so the campaign log stays clean of escape codes.
    """

    _TAG = re.compile(r"^(\s*)\[([a-z][a-z-]{2,14})\]")

    def __init__(self, stream):
        self._s = stream

    def write(self, text):
        if not text.strip():
            return self._s.write(text)
        out = []
        for line in text.splitlines(True):
            line = line.lstrip(" \t") if line.strip() else line
            m = self._TAG.match(line)
            if m and m.group(2) in VOICE:
                # VOICE values are the colour FUNCTIONS built by _c(), not escape strings.
                paint = VOICE[m.group(2)]
                nl = "\n" if line.endswith("\n") else ""
                body = line[m.end():].rstrip("\n")
                line = f"{m.group(1)}{paint('[' + m.group(2) + ']')}{body}{nl}"
            out.append(line)
        return self._s.write("".join(out))

    def __getattr__(self, name):
        return getattr(self._s, name)


def install_line_colour():
    """Call once, at process start.

    Installed even when colour is OFF, because it also strips the per-call-site indentation --
    every print carried its own two-to-six leading spaces and the output drifted right depending
    on which role was speaking.
    """
    if not isinstance(sys.stdout, _Colourise):
        sys.stdout = _Colourise(sys.stdout)


def wrap_names(names, width=92):
    """A list of run names across as many lines as it needs, never one long one.

    Twelve names on a single line is ~250 characters: it wraps in the terminal wherever the
    window happens to end, so the same output reads differently in two windows and no name is
    findable by eye.
    """
    import textwrap
    return textwrap.fill(", ".join(names), width=width,
                         break_long_words=False, break_on_hyphens=False)
