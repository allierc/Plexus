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

# Escape codes take no columns. Anything that measures a line's width must not count them.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# ONE WIDTH FOR THE WHOLE LOOP. say() used to fold at 100 and the stdout wrapper again at 96,
# so every quoted line was folded twice at two widths. Anything that wraps reads this.
#
# IT FOLLOWS THE WINDOW, UP TO A POINT. A fixed 96 wastes a third of a 150-column terminal, but
# prose set to the full 150 is genuinely harder to read -- the eye loses the line on the way back.
# So: use the window, capped at 100, which is where typographers put the limit and near enough
# what a round's lines already want. A log file is NOT a window and keeps the fixed 96, so the
# campaign log reads the same wherever it is opened and diffs between rounds stay meaningful.
def _width():
    if not sys.stdout.isatty():
        return 96
    try:
        import shutil
        return max(72, min(100, shutil.get_terminal_size().columns - 2))
    except Exception:
        return 96


WIDTH = _width()

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
    "escalate":      orange,     # the Supervisor changing the envelope
    # THE REST OF THE CAST. Every one of these prints `[name] ...` somewhere and none of them had
    # a colour, so an actual role speaking was indistinguishable from `[ckpt]` or `[preflight]`.
    # The Grounder is the clearest case: it opens Act 1 by reporting what the PAPER says, and its
    # line arrived in the same plain grey as a checkpoint path.
    #
    # Grouped by what the role is FOR, like the block above: the specimen family is teal, the
    # record family is sky, the across-rounds family is orange, and what LOOKS at a picture is
    # gold. Roles are coloured; subsystems (cluster, caption, ckpt, engine) deliberately are not,
    # because the distinction the colour carries is "someone is speaking" against "something is
    # happening".
    "grounder":      green,      # what the paper actually says
    "critic":        blue,       # is the batch legal (the Proposer's family)
    "collector":     sky,        # builds the round record from the files (the record family)
    "reflection":    sky,        # what this round means for the next
    "eye":           gold,       # the caption wave -- the same eye as eye-check
    "watcher":       gold,       # and the role that reads it
    # PHASE 12. The Analyst absorbed reader + interpreter + meta-review + collector +
    # diagnostician, so it inherits their colour: sky is the record family, and this is now the one
    # line per round that carries the conclusion. `eye` and `proposer` already had theirs; `round`,
    # `crew`, `metrics`, `campaign` and `cluster` deliberately do NOT -- the distinction the colour
    # carries is "someone is speaking" against "something is happening", and the round runner is a
    # something.
    "analyst":       sky,        # what happened this round, and what it means
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


def say(who, text, sentences=1, width=WIDTH):
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
    # THE HEAD OCCUPIES COLUMNS TOO. Filling to the full width and then prepending "◉ reader: "
    # made the first line that much longer than the rest, which the stdout wrapper then folded
    # again -- so the first line of every quoted agent line broke, and only the first.
    n_head = len(_ANSI.sub("", head))
    lines = textwrap.wrap(t, width=width, break_long_words=False, break_on_hyphens=False,
                          initial_indent=" " * n_head) or [""]
    out = [head + lines[0][n_head:]]
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

    # THE ICON MAY COME FIRST. This anchored on `^\s*\[`, so `[archivist] ...` was coloured and
    # `◌ [archivist] continue` -- the same role, one glyph further right -- was not. The Archivist
    # IS in VOICE and always has been; what failed was the match, which is why the colour looked
    # like a missing entry in the table. Any leading icon and any ANSI already applied are
    # skipped, so a line that was coloured by its printer still finds its tag.
    _TAG = re.compile(r"^((?:\x1b\[[0-9;]*m)*[\s▶✓✗⚠⚙◉◌■⏱]*)\[([a-z][a-z-]{2,14})([^\]]*)\]")

    # LOWER CASE UNLESS THE WORD IS SHOUTING. A round's terminal is one voice reporting on itself,
    # and sentence-initial capitals in it are decoration: "Phenotype degenerate", "Continue: give
    # numeric predictions", "Round 1 bought nothing". Capitals in this loop MEAN something -- a
    # fully upper-case word is an emphasis the code chose (NOT ROUTED, BUDGET EXCEEDED, DEGRADED)
    # -- and when every sentence also starts with one, the emphasis stops being visible.
    #
    # So only a plainly Capitalised word is lowered, and the pattern is deliberately narrow:
    #   [A-Z][a-z]+  and nothing else, so NaN, GPU, P4, r001n_09 and protr_peak are all untouched
    #   not preceded or followed by  \w / - .x   so /workspace/Plexus/log stays a real path --
    #                                            paths are case-sensitive and lowering one makes
    #                                            it a path that does not exist
    #   a trailing "." is fine when it ends a sentence ("Continue.") and blocking when it starts
    #   an extension ("Plexus.log")
    _CAPWORD = re.compile(r"(?<![\w/\-.])([A-Z][a-z]+)(?![\w/\-]|\.\w)")
    _ESCSPLIT = re.compile(r"(\x1b\[[0-9;]*m)")

    def __init__(self, stream):
        self._s = stream

    @classmethod
    def _lower_prose(cls, body):
        """Lower the Capitalised words in `body`, leaving colour escapes alone.

        The escapes must be split out rather than matched around: `\\x1b[36mReader` puts a word
        character ("m") immediately before the word, which any lookbehind would read as "part of
        an identifier" and skip -- so coloured lines would keep their capitals and uncoloured
        ones would not, and the rule would appear to work in a pipe and fail in a terminal.
        """
        parts = cls._ESCSPLIT.split(body)
        for i in range(0, len(parts), 2):                 # even = text, odd = escape
            parts[i] = cls._CAPWORD.sub(lambda m: m.group(1).lower(), parts[i])
        return "".join(parts)

    WIDTH = WIDTH       # module-level, so say() and this fold at the same column

    def write(self, text):
        if not text.strip():
            return self._s.write(text)
        out = []
        for line in text.splitlines(True):
            line = line.lstrip(" \t") if line.strip() else line
            line = self._lower_prose(line)
            line = self._wrap(line)
            m = self._TAG.match(line)
            if m and m.group(2) in VOICE:
                # VOICE values are the colour FUNCTIONS built by _c(), not escape strings.
                paint = VOICE[m.group(2)]
                nl = "\n" if line.endswith("\n") else ""
                body = line[m.end():].rstrip("\n")
                # group(3) is whatever rides inside the bracket after the role -- "[eye 3/11]"
                # is the caption wave counting runs, and it is the eye speaking, not a subsystem.
                line = f"{m.group(1)}{paint('[' + m.group(2) + m.group(3) + ']')}{body}{nl}"
            out.append(line)
        return self._s.write("".join(out))

    def _wrap(self, line):
        """Fold a long line onto the next, once, for every print in the loop.

        Doing it per call site means editing forty f-strings and missing the forty-first. The
        grounder's note was 250 characters on one line; the terminal then broke it wherever the
        window happened to end, so the same output read differently in two windows.

        Left alone: banner rules, anything already short, and lines with no space to break at.
        A tag like [grounder] is kept on the first line and continuations are indented under it,
        so the speaker stays findable when scanning the left margin.
        """
        import textwrap
        body = line.rstrip("\n")
        nl = "\n" if line.endswith("\n") else ""
        # MEASURE WHAT IS VISIBLE. `len()` counts the escape codes, which occupy no columns, so a
        # coloured line looked ~9 characters longer than it printed and got folded that much too
        # early -- and by a different amount per colour, which is why one paragraph came out with
        # breaks at 82, 88 and 94 and read as ragged. say() also pre-folded at 100 against this
        # WIDTH of 96, guaranteeing a second fold; both are why prose arrived broken mid-phrase.
        plain = _ANSI.sub("", body)
        if len(plain) <= self.WIDTH or len(set(plain.strip())) <= 2:
            return line                                  # short, or a ==== / ---- rule
        if plain != body:
            # Coloured AND genuinely too long: fold the visible text, then re-open the colour on
            # every piece, because an escape sequence does not survive being cut in half.
            pre = (re.match(r"^(?:\x1b\[[0-9;]*m)+", body) or [""])[0]
            wrapped = textwrap.wrap(plain, width=self.WIDTH,
                                    break_long_words=False, break_on_hyphens=False)
            return ("\n".join(pre + w + ("\x1b[0m" if pre else "") for w in wrapped) + nl
                    if wrapped else line)
        m = re.match(r"^(\[[a-z][a-z-]*\]\s*)", body)
        head, rest = (m.group(1), body[m.end():]) if m else ("", body)
        wrapped = textwrap.wrap(rest, width=self.WIDTH - len(head),
                                break_long_words=False, break_on_hyphens=False)
        if not wrapped:
            return line
        pad = " " * min(len(head), 12)
        return head + ("\n" + pad).join(wrapped) + nl

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
