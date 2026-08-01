"""grounder -- the agent that holds the campaign's three bodies of text.

    paper/plexus2.tex        the LANGUAGE CONTRACT  -- what the language permits
    papers/okuda.pdf         the REFERENCE MODEL    -- what Okuda et al. actually do
    papers/okuda_corpus.md   the LITERATURE INDEX   -- what the field has established
    papers/*.pdf             the vendored corpus

It is called at exactly three points:

  1. PROPOSE   the Proposer needs a mechanism grounded before proposing it
               -> ground(question)          returns passages + citations
  2. GATE      the Supervisor gates a hypothesis before Discovery may propose it
               -> gate(hypothesis)          returns (ok, why, citations)
  3. REQUEST   an operator request is filed -- does the literature already name this
               mechanism, and under what conditions does it hold?
               -> name_mechanism(desc)      returns prior art or "apparently novel"

This is the agent whose absence was most expensive in the hand-run campaign. The decisive
question -- *which reaction-diffusion regime does Okuda actually use?* -- was answered by
READING THE PAPER, not by sweeping parameters, and it redirected the whole effort:

    Brusselator decays or reorganises a seeded spot;
    Gray-Scott holds it;
    Gierer-Meinhardt -- what Okuda actually cites -- AMPLIFIES it into a stable gradient peak.

Making that a scheduled call rather than a lucky afternoon is the entire point.

--------------------------------------------------------------------------------------------
DESIGN: retrieval is local, deterministic and citable. No web access, no hidden state.
Every returned passage carries (source, page/line) so a claim in knowledge.md can be audited
back to the sentence that grounded it.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PAPERS = os.path.join(ROOT, "papers")
PLEXUS2 = os.path.join(ROOT, "paper", "plexus2.tex")
CORPUS_INDEX = os.path.join(PAPERS, "okuda_corpus.md")

# The reference model, and the corpus entries most likely to matter for tubulation. Order is
# the retrieval priority: the reference model first, then the language, then the field.
PRIORITY = ["okuda.pdf", "Turing_Vertex.pdf", "LedesmaDuran_2023_turing_growing_domain.pdf",
            "Noguchi_Elgeti_2024_tissue_sheets_buckles.pdf",
            "Zhang_Schwarz_2022_tvm_3d_vertex.pdf", "Sarkar_Krajnc_2023_graph_vertex_model.pdf",
            "Pasqui_2026_VertAX_differentiable.pdf", "Deshpande_2025_jax_morph.pdf",
            "Kim_Zhang_Schwarz_2024_3d_vertex_moduli.pdf", "Moore_2023_bilayer_polarity.pdf",
            "Sorichetti_2026_vertex.pdf", "SimuCell3D.pdf"]


@dataclass
class Passage:
    source: str
    locus: str            # "p.7" for a pdf, "L1234" for tex/md
    text: str

    def cite(self):
        return f"{self.source} ({self.locus})"

    def __repr__(self):
        return f"<{self.cite()}: {self.text[:70]}...>"


# --------------------------------------------------------------------------- corpus loading
_CACHE = {}


def _load_pdf(path):
    if path in _CACHE:
        return _CACHE[path]
    try:
        import fitz
    except ImportError:
        return []
    try:
        doc = fitz.open(path)
        pages = [(i + 1, doc[i].get_text()) for i in range(len(doc))]
    except Exception:
        pages = []
    _CACHE[path] = pages
    return pages


def _load_text(path):
    if path in _CACHE:
        return _CACHE[path]
    try:
        lines = open(path, errors="ignore").read().split("\n")
    except Exception:
        lines = []
    _CACHE[path] = lines
    return lines


def available():
    """What the Grounder can actually see, in priority order.

    De-duplicated by content hash: okuda.pdf and Turing_Vertex.pdf are the SAME paper, and
    without this every citation would appear twice, silently doubling the apparent weight of
    evidence for whatever that paper says.
    """
    import hashlib
    seen_hashes = {}

    def _dup(path):
        try:
            h = hashlib.sha1(open(path, "rb").read(1 << 20)).hexdigest()
        except Exception:
            return False
        if h in seen_hashes:
            return True
        seen_hashes[h] = path
        return False

    out = []
    if os.path.exists(PLEXUS2):
        out.append(("plexus2.tex", PLEXUS2, "tex"))
    if os.path.exists(CORPUS_INDEX):
        out.append(("okuda_corpus.md", CORPUS_INDEX, "md"))
    for name in PRIORITY:
        p = os.path.join(PAPERS, name)
        if os.path.exists(p) and not _dup(p):
            out.append((name, p, "pdf"))
    for f in (sorted(os.listdir(PAPERS)) if os.path.isdir(PAPERS) else []):
        p = os.path.join(PAPERS, f)
        if f.endswith(".pdf") and f not in PRIORITY and not _dup(p):
            out.append((f, p, "pdf"))
    return out


# --------------------------------------------------------------------------- 1. PROPOSE
def ground(question, k=6, sources=None, window=420):
    """Retrieve passages relevant to a mechanism question, with citations.

    Deterministic term-overlap retrieval: no embedding model, no network, reproducible across
    runs. The campaign must be able to re-derive any citation it recorded.
    """
    terms = [t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", question.lower())
             if t not in _STOP]
    if not terms:
        return []
    hits = []
    for name, path, kind in (sources or available()):
        if kind == "pdf":
            for pno, text in _load_pdf(path):
                s = _score(text.lower(), terms, name)
                if s > 0:
                    hits.append((s, Passage(name, f"p.{pno}", _snippet(text, terms, window))))
        else:
            lines = _load_text(path)
            for i in range(0, len(lines), 40):
                chunk = "\n".join(lines[i:i + 40])
                s = _score(chunk.lower(), terms, name)
                if s > 0:
                    hits.append((s, Passage(name, f"L{i + 1}", _snippet(chunk, terms, window))))
    hits.sort(key=lambda h: -h[0])
    # keep at most 2 passages per source so one long paper cannot crowd out the rest
    out, per = [], {}
    for _, p in hits:
        if per.get(p.source, 0) >= 2:
            continue
        per[p.source] = per.get(p.source, 0) + 1
        out.append(p)
        if len(out) >= k:
            break
    return out


_STOP = {"the", "and", "for", "with", "that", "this", "are", "was", "were", "does", "which",
         "what", "how", "why", "from", "into", "than", "then", "have", "has", "not", "but",
         "can", "cannot", "will", "would", "should", "may", "might", "its", "their", "our"}


def _score(text, terms, source=None):
    """Length-NORMALISED term overlap, with a prior on the reference model.

    Raw counts let long documents win on common words: a query about tube diameter returned a
    brain-folding review and a differentiable-rendering paper ahead of okuda.pdf. Normalising by
    sqrt(length) removes that bias, and matching DISTINCT terms (not total occurrences) rewards
    a page that covers the whole question rather than repeating one word.
    """
    import math
    hits = [t for t in terms if len(t) > 3 and t in text]
    if not hits:
        return 0.0
    total = sum(text.count(t) for t in hits)
    coverage = len(hits) / max(1, len([t for t in terms if len(t) > 3]))
    base = (total / math.sqrt(max(len(text), 1))) * (0.4 + 0.6 * coverage)
    if source in ("okuda.pdf", "plexus2.tex", "okuda_corpus.md"):
        base *= 2.5                       # the reference model and the language contract lead
    return base


def _snippet(text, terms, window):
    low = text.lower()
    best, bi = -1, 0
    for t in terms:
        i = low.find(t)
        if i >= 0 and (best < 0 or i < best):
            best, bi = i, i
    a = max(0, bi - window // 3)
    return " ".join(text[a:a + window].split())


# --------------------------------------------------------------------------- 2. GATE
# Claims the reference model settles. A hypothesis contradicting one of these is not
# automatically wrong -- but the Supervisor must see the contradiction before Discovery spends
# cluster time on it.
REFERENCE_CLAIMS = [
    dict(key="quasi_static",
         claim="Okuda's process is QUASI-STATIC: the tissue relaxes to force balance "
               "(residual < E_th) between growth/division events, with tau_cycle >> eta/kappa_s.",
         probe="quasi-static force balance residual cell cycle"),
    dict(key="rd_regime",
         claim="Okuda's reaction-diffusion is GIERER-MEINHARDT (self-enhancing activator + fast "
               "lateral inhibitor), not Brusselator and not Gray-Scott.",
         probe="activator inhibitor reaction diffusion Gierer Meinhardt"),
    dict(key="no_bending",
         claim="Okuda has NO explicit bending energy; tube straightness is EMERGENT from "
               "diameter (thin undulate, thick straight).",
         probe="bending rigidity emergent tube diameter undulation"),
    dict(key="growth_driven",
         claim="Okuda's tube is produced by LOCALISED PROLIFERATION as an equilibrium shape; "
               "there is no explicit outward extrusion force.",
         probe="growth equilibrium volume proliferation tube formation"),
    dict(key="chi_diameter",
         claim="The diffusion coefficient chi sets the tube DIAMETER (spot size ~ chi^(1/4)).",
         probe="diffusion coefficient spot size tube diameter"),
]


def gate(hypothesis_claim, k=3):
    """Before Discovery proposes: does the reference model already settle this, or contradict it?

    Returns (verdict, why, citations) with verdict in
        'grounded'      the literature supports it -> proceed
        'contradicted'  the reference model says otherwise -> the Supervisor must see this
        'unsettled'     no strong prior -> proceed, flagged as genuinely open
    """
    low = hypothesis_claim.lower()
    relevant = []
    for rc in REFERENCE_CLAIMS:
        terms = [t for t in rc["probe"].split() if len(t) > 3]
        overlap = sum(1 for t in terms if t.lower() in low)
        if overlap >= 2:
            relevant.append((overlap, rc))
    if not relevant:
        return "unsettled", "no reference claim matches this hypothesis", []
    relevant.sort(key=lambda r: -r[0])
    rc = relevant[0][1]
    cits = ground(rc["probe"], k=k)
    negated = any(w in low for w in (" not ", " no ", "cannot", "without", "absent", "remove"))
    verdict = "contradicted" if negated else "grounded"
    why = (f"reference claim [{rc['key']}]: {rc['claim']}"
           + (" -- the hypothesis appears to NEGATE it; that is allowed but must be deliberate."
              if negated else ""))
    return verdict, why, [p.cite() for p in cits]


# --------------------------------------------------------------------------- 4. UNDERSTAND
def understand(question, k=8, width=900):
    """OPEN-ENDED reading. No fixed claim list, no keyword gate -- just: what does the corpus say?

    `gate()` is deliberately narrow (does this hypothesis contradict something the paper
    settles?). That narrowness was mistaken for the whole agent. Most of what a modeller needs
    from the literature is not adjudication but UNDERSTANDING: what regime is this, what sets
    that length scale, why does this figure look like that. This is that call.
    """
    passages = ground(question, k=k, window=width)
    return {"question": question,
            "passages": [{"cite": p.cite(), "text": p.text} for p in passages],
            "sources": sorted({p.source for p in passages})}


# --------------------------------------------------------------------------- 5. REPRODUCE A FIGURE
# Okuda et al. 2018 give EXPLICIT physical parameters per figure, which turns "reproduce the
# paper" from a vague aspiration into a labelled PHASE DIAGRAM in (chi, gamma):
#
#        gamma      0.01            1              100
#   chi
#   0.01          branching                    thin tube
#   0.1                         thick tube     undulation
#
# chi  = the diffusion coefficient; the paper states it sets the SPOT SIZE = tube DIAMETER.
# gamma = the ratio of patterning to deformation timescales -- the regime knob.
#
# Reproducing this 2x2 QUALITATIVELY is the campaign's primary scientific target: four distinct
# morphologies from one composition, separated only by two numbers. It is also the talk figure.
FIGURES = {
    "fig5a": dict(figure="Figure 5a", phenotype="thin tube", chi=0.01, gamma=100.0,
                  shows="time series of THIN tube formation; cells coloured by activator",
                  criterion="a single high-aspect protrusion of SMALL diameter; activator at the tip"),
    "fig5b": dict(figure="Figure 5b", phenotype="thick tube", chi=0.1, gamma=1.0,
                  shows="time series of THICK tube formation; cells coloured by activator",
                  criterion="a single protrusion of LARGER diameter than fig5a at the same length"),
    "fig6":  dict(figure="Figure 6", phenotype="branching", chi=0.01, gamma=0.01,
                  shows="whole-tissue deformation, cells coloured by local mean curvature; "
                        "plus a time series of the branch structure",
                  criterion="a protrusion that BIFURCATES -- n_tubes increases over time"),
    "fig7":  dict(figure="Figure 7", phenotype="undulation", chi=0.1, gamma=100.0,
                  shows="whole-tissue deformation; cells coloured by activator",
                  criterion="MANY shallow bumps rather than one deep protrusion -- "
                            "low protrusion, high spot count"),
}


# =============================================================================================
# THE STARTING CONDITIONS. This is the half of the paper the campaign never read.
# ---------------------------------------------------------------------------------------------
# FIGURES above says where each experiment ENDS -- the morphology, and the two numbers that select
# it. Nothing said where one BEGINS, so every batch inherited a starting cell count from whatever
# default sat in a config file. On 31 July a 27-run battery was launched at 150 cells, hit the
# mesh-buffer ceiling of 1778 in every single run, and produced no evidence at all. The paper says
# 200. Nothing in the loop had ever asked it.
#
# So these are Okuda's own starting conditions, quoted, with the sentence they come from. They are
# ADVICE, not a gate: the Proposer takes them for the faithful share of a batch and is free to
# leave them for the exploratory share (the 70/30 rule). What is NOT optional is `buffer_for()`
# below -- a run that cannot reach the cell count it is aiming at measures the array, not the
# tissue, whichever share it belongs to.
SETUP = {
    # WHICH EXPERIMENT A NUMBER BELONGS TO. Corrected 2026-08-01, and the error is worth keeping
    # on the record: the first version attached the paper's "about 200 cells" to the tubulation
    # figures. That sentence is real, but it belongs to two EARLIER control experiments -- one on
    # arrested tissue with no volume growth at all, one with growth made independent of the
    # molecular concentration. Neither is a coupling run. Figures 5, 6 and 7 are all the SAME
    # coupling experiment and all start at 2,000 cells. Cedric caught it.
    #
    # A quotation is not a citation. The quote was accurate and the mapping was wrong, which is
    # worse than having no quote at all: it made a tenfold error look verified.
    "coupled": dict(
        section="Coupling patterning and deformation drives undulation, tubulation, and branching",
        n_cells=2000, grows_to=4000, figures=["fig5a", "fig5b", "fig6", "fig7"],
        quote="Coupling patterning and deformation drives undulation, tubulation, and "
              "branching. [...] The initial tissue morphology was simply set to be a spherical "
              "vesicle of a monolayer cell sheet composed of about 2,000 cells, whose patterns "
              "reached steady states.",
        note="THE campaign case. Tubulation (Fig 5), branching (Fig 6) and undulation (Fig 7) "
             "are one experiment at one starting size; only chi and gamma differ between them."),
    "arrested": dict(
        section="Turing patterns in 3D cell aggregates",
        n_cells=200, grows_to=200, figures=[],
        quote="we simulated patterning processes while without cell volume growth [...] These "
              "tissues are composed of about 200 cells",
        note="A CONTROL, not a morphology run: patterning on tissue that cannot grow. Useful as "
             "our own no-growth control, never as the starting size for a coupling figure."),
    "uncoupled_growth": dict(
        section="Patterning hysteresis on deforming tissues",
        n_cells=200, grows_to=800, figures=[],
        quote="cell growth rate was set to be constant [...] the initial tissue morphology was "
              "simply set to be a spherical vesicle of a monolayer cell sheet composed of about "
              "200 cells",
        note="The hysteresis experiment: the tissue grows, but growth is INDEPENDENT of the "
             "morphogen. The other control -- deformation without the coupling."),
    "diagram": dict(
        section="Difference in time scales between patterning and deformation varies",
        n_cells=2000, grows_to=4000, figures=[],
        quote="The individual tissues in (b) were composed of about 4,000 cells, which were "
              "picked up on the growth process.",
        note="The (chi,gamma) morphology diagram. 4,000 is where the tissues were SAMPLED during "
             "growth, not where they were seeded -- a destination, and what the buffer must hold."),
}

# What he reports SEEING, which is what we calibrate against -- his parameter values live in a
# differently-scaled model and are not transferable (finding F009).
OBSERVED = {
    "n_spots_at_2000": 5,
    "note": "about five activator domains on the 2000-cell ball. Calibrate the chemistry until we "
            "reproduce this COUNT; never copy chi across the two models.",
}

# Table 1, normalised. Quoted so a fixture can assert them rather than a person retyping them.
# The one sentence that fixes all four normalisation constants, verbatim.
NORMALISATION_SENTENCE = ("normalized values of several physical parameters are given as vref = 1, ρu = 1, κs = 0.2, and ηc = 0.25")

TABLE1 = {
    # EVERY VALUE CARRIES THE ROW IT WAS READ FROM, and `verify_table1()` checks that the paper's
    # table really does put that number next to that description. SETUP was corrected on
    # 2026-08-01 after a real quote was attached to the wrong experiment; this register was
    # transcribed in the same reading session and had NO quotes at all -- nine numbers with
    # nothing to check them against. A transcription error here is invisible and would propagate
    # into every run claiming to be "at Okuda's settings".
    #
    # `row` is the paper's own wording, in the order the table prints it: description, then value.
    "alpha_hill":  dict(value=10.0,  row="Hill coefficient in growth rate 10",
                        note="the sharpness of a cell's growth response. OUTSIDE our old 1-8 "
                             "search box; widened in Phase 2."),
    "rho_sw":      dict(value=0.5,   row="Switching concentration for growth rate 0.5",
                        note="the activator level at which growth switches on."),
    "phi_inhib":   dict(value=10.0,  row="Diffusivity of inhibitor 10",
                        note="OUTSIDE our old 0.1-2 box; widened in Phase 2. The paper ALSO "
                             "derives it as phi ~ (3 v_ref^1/3)^2/(4 eta_c/5 kappa_s), which "
                             "evaluates to 9.0 -- it writes 'phi ~', so the table's 10 is the "
                             "set value and 9 is the design intent. Checked, not reconciled: "
                             "see verify_table1's derived-value note."),
    "tau_cycle":   dict(value=50.0,  row="Cell cycle 50",
                        note="in units of eta_c/kappa_s = 1.25, so 62.5 time units."),
    "kappa_s":       dict(value=0.2, quote=NORMALISATION_SENTENCE,
                        note="normalisation constant, stated in the text as well as the table."),
    "eta_c":         dict(value=0.25, quote=NORMALISATION_SENTENCE,
                        note="cell friction; sets the unit of time with kappa_s."),
    "v_ref":         dict(value=1.0, quote=NORMALISATION_SENTENCE,
                        note="reference cell volume; defines the unit of length."),
    "rho_u":         dict(value=1.0, quote=NORMALISATION_SENTENCE,
                        note="reference morphogen concentration; defines the unit of amount."),
    "gamma_range": dict(value=(0.01, 100.0), row="Time characteristics of patterning 0.01-100",
                        note="VARIED across the figures -- one of the two axes of the diagram."),
    "chi_range":   dict(value=(0.001, 0.1),  row="Spatial characteristics of patterning 0.001-0.1",
                        note="VARIED. NOT the same quantity as our `chi` (finding F009)."),
}

# Two regime INEQUALITIES, which are what actually define his setting. Worth asserting rather than
# tuning: they are the reason the model is quasi-static and effectively incompressible.
REGIME = [
    ("k_v >> kappa_s * v_ref**(2/3)", "incompressibility"),
    ("tau_cycle >> eta_c / kappa_s",  "quasi-static: mechanics relaxes between biological events"),
]



# ============================================================================ quote verification
# THE CHECK THAT WOULD HAVE CAUGHT IT. On 2026-08-01 this agent asserted that Okuda's tubulation
# figures start at 200 cells, and quoted the paper for it. The quote was verbatim and the mapping
# was wrong: "about 200 cells" belongs to a control experiment on arrested tissue, eight pages
# before the coupling experiment that produces Figures 5-7 at 2,000. Every faithful slot would
# have run a tenth of the paper's initial condition, and the citation would have vouched for it.
#
# A QUOTATION IS NOT A CITATION. Checking that a quote is real catches fabrication, which was
# never the failure mode -- the failure mode is a real sentence lifted out of the experiment it
# describes. So the quote must also land in the SECTION the entry claims it comes from. That is
# deterministic: the paper's result headings are distinctive, and a quote either falls between
# its declared heading and the next one or it does not.
_PAPER_CACHE = {}


def _paper_text():
    """The paper as one normalised string, for locating quotes. Cached; read-only."""
    if "t" not in _PAPER_CACHE:
        import fitz
        d = fitz.open(os.path.join(PAPERS, "Turing_Vertex.pdf"))
        _PAPER_CACHE["t"] = re.sub(r"\s+", " ",
                                   "\n".join(d[i].get_text() for i in range(d.page_count)))
    return _PAPER_CACHE["t"]


SECTION_ORDER = [
    "Turing patterns in 3D cell aggregates",
    "Patterning hysteresis on deforming tissues",
    "Coupling patterning and deformation drives undulation, tubulation, and branching",
    "Difference in time scales between patterning and deformation varies",
]


def _section_span(t, heading):
    """(start, end) of a result section: its heading to the next heading, or the end."""
    i = t.find(heading)
    if i < 0:
        return None
    later = [t.find(h) for h in SECTION_ORDER if t.find(h) > i]
    return (i, min(later) if later else len(t))


def section_of(t, at):
    """Which result section a character offset falls in. Derived, never declared."""
    spans = sorted((t.find(h), h) for h in SECTION_ORDER if t.find(h) >= 0)
    for k, (i, h) in enumerate(spans):
        j = spans[k + 1][0] if k + 1 < len(spans) else len(t)
        if i <= at < j:
            return h, (i, j)
    return None, None


def figures_named_in(t, span):
    """The figure NUMBERS a section actually refers to. The paper's own cross-references."""
    return set(re.findall(r"Fig(?:ure|\.)\s*(\d)", t[span[0]:span[1]]))


def verify_setup(entries=None):
    """Every quote must be real, and every figure an entry claims must be named where the quote is.

    THE HUMAN IS OUT OF THIS LOOP. An earlier version had each entry DECLARE the section its
    quote came from, which only catches a quote/section mismatch -- an entry that is wrong but
    self-consistent still passes, and someone still has to notice. So the section is now DERIVED
    from where the quote actually falls, and the entry is held to a claim it cannot fake: if it
    says it describes Figure 5, the section containing its quote must itself mention Figure 5.

    That is what would have caught 2026-08-01 with nobody reading. The bad entry claimed Figures
    5a/5b/6 while quoting a sentence from a section whose only cross-references are Figures 3 and
    4. The paper contradicts the entry, in the paper's own words, arithmetically.

    Entries claiming no figures cannot be checked this way -- the check bites exactly where a
    claim is made -- so their derived section is reported instead, for a reader to see.
    """
    t = _paper_text()
    bad, seen = [], {}
    for name, e in (entries or SETUP).items():
        frags = [re.sub(r"\s+", " ", f.strip()) for f in e["quote"].split("[...]") if f.strip()]
        locs = []
        for frag in frags:
            at = t.find(frag)
            if at < 0:
                bad.append(f"{name}: quote fragment is NOT in the paper: {frag[:60]!r}")
            else:
                locs.append(at)
        if not locs:
            continue
        sec, span = section_of(t, locs[0])
        if sec is None:
            bad.append(f"{name}: quote falls outside every known result section")
            continue
        # every fragment must come from the SAME section -- otherwise the quote is a splice
        for at in locs[1:]:
            s2, _ = section_of(t, at)
            if s2 != sec:
                bad.append(f"{name}: quote is spliced across sections "
                           f"({sec[:34]!r} and {str(s2)[:34]!r})")
        named = figures_named_in(t, span)
        claimed = {re.sub(r"\D", "", f)[:1] for f in e.get("figures", []) if re.sub(r"\D", "", f)}
        missing = sorted(claimed - named)
        if missing:
            bad.append(f"{name}: claims Figure(s) {', '.join(missing)}, but its quote sits in "
                       f"{sec[:44]!r}, which names only Figure(s) {', '.join(sorted(named))}")
        if e.get("section") and e["section"] != sec:
            bad.append(f"{name}: declares section {e['section'][:38]!r} but its quote is in "
                       f"{sec[:38]!r}")
        seen[name] = sec
    return bad


def setup_provenance():
    """Where each entry's quote actually lives, derived. For a reader, and for the note."""
    t = _paper_text()
    out = {}
    for name, e in SETUP.items():
        frag = re.sub(r"\s+", " ", e["quote"].split("[...]")[0].strip())
        at = t.find(frag)
        sec, span = section_of(t, at) if at >= 0 else (None, None)
        out[name] = {"section": sec, "n_cells": e["n_cells"],
                     "figures_claimed": e.get("figures", []),
                     "figures_named_there": sorted(figures_named_in(t, span)) if span else []}
    return out



def verify_table1():
    """Every Table 1 value must sit beside its description in the paper's own table.

    SETUP's guard asks WHERE a quote comes from, because its failure mode was a real sentence in
    the wrong place. This register's failure mode is different and simpler: a number typed wrong.
    So the check is different -- find the row's description in the paper and require the value we
    recorded to be the number printed next to it.

    The paper's table extracts as `<description> <value> <units> (<equations>)`, so the test is
    that our value appears in the short window immediately after the description. Ranges are
    written with an en-dash in the PDF and a hyphen in source, so both are accepted; that is a
    typography difference, not a disagreement about a number.

    Returns (failures, notes). A note is a real discrepancy the paper itself carries -- phi is
    tabulated as 10 and derived as 9.0 from the paper's own formula -- and is reported rather
    than silently reconciled, because choosing one for the campaign is a decision, not a lookup.
    """
    t = re.sub(r"[\u2013\u2014]", "-", _paper_text())
    bad, notes = [], []
    for name, e in TABLE1.items():
        if "row" not in e:                       # verified by its sentence instead
            q = re.sub(r"\s+", " ", e["quote"])
            if q not in re.sub(r"\s+", " ", _paper_text()):
                bad.append(f"{name}: its quote is not in the paper verbatim")
            elif str(e["value"]).rstrip("0").rstrip(".") not in q:
                bad.append(f"{name}: {e['value']} does not appear in the sentence it cites")
            continue
        row = re.sub(r"[\u2013\u2014]", "-", e["row"])
        desc = row.rsplit(" ", 1)[0] if not row.count("=") else row.split("=")[0].strip()
        at = t.find(desc)
        if at < 0:
            bad.append(f"{name}: its table row is not in the paper: {desc[:52]!r}")
            continue
        window = t[at + len(desc):at + len(desc) + 40]
        v = e["value"]
        fmt = lambda x: str(int(x)) if float(x).is_integer() else str(x)   # noqa: E731
        wanted = f"{fmt(v[0])}-{fmt(v[1])}" if isinstance(v, tuple) else fmt(v)
        # THE FIRST NUMBER AFTER THE DESCRIPTION, not any number in the neighbourhood. A
        # substring test passed `alpha_hill = 8` -- the paper prints "Hill coefficient in growth
        # rate 10 1 (8)", where the (8) is a cross-reference to Equation 8. So a check that only
        # asked "does 8 appear nearby" accepted the exact value our old search box was wrong
        # about. The value has to be the number in the VALUE COLUMN.
        m = re.match(r"\s*(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)", window)
        printed = m.group(1) if m else None
        if printed != wanted:
            bad.append(f"{name}: recorded {wanted}, but the paper's value column reads "
                       f"{printed!r} after {desc[:34]!r}")
        if e.get("quote"):
            q = re.sub(r"\s+", " ", e["quote"]).replace("rho_u", "\u03c1u") \
                  .replace("kappa_s", "\u03bas").replace("eta_c", "\u03b7c")
            if q not in re.sub(r"\s+", " ", _paper_text()):
                bad.append(f"{name}: its quote is not in the paper verbatim")
    # the paper's own derived value for phi, checked against its own table
    k, e_c, vr = TABLE1["kappa_s"]["value"], TABLE1["eta_c"]["value"], TABLE1["v_ref"]["value"]
    phi_derived = (3 * vr ** (1 / 3)) ** 2 / (4 * e_c / (5 * k))
    if abs(phi_derived - TABLE1["phi_inhib"]["value"]) > 1e-6:
        notes.append(f"phi: tabulated {TABLE1['phi_inhib']['value']}, but the paper's own formula "
                     f"(3 v_ref^1/3)^2/(4 eta_c/5 kappa_s) gives {phi_derived:.3f}. The paper "
                     f"writes 'phi ~', so both are its own; which one the campaign uses is a "
                     f"decision and must be made explicitly, not inherited.")
    return bad, notes


def buffer_for(target_cells, margin=1.30):
    """Reservoir sizes that can actually HOLD `target_cells`. Not advice -- arithmetic.

    A closed epithelial sheet is trivalent, so Euler gives V = 2F - 4 exactly: a vertex reservoir
    of size V caps the cell count at (V+4)/2 whatever the biology wants. That is why 3552 vertices
    produced exactly 1778 cells in all 32 runs of the overnight study AND in all 27 of the weekend
    battery -- twice, in the same week, from the same arithmetic.

    Sizing the buffer from the DESTINATION rather than from the seed is the whole fix. The margin
    is headroom for the transient over-allocation division needs, not slack for wishful thinking.
    """
    n = int(target_cells * margin)
    vertex = 2 * n + 4
    # The cell reservoir must hold what the vertex reservoir ALLOWS, not what we asked for --
    # otherwise the two disagree by four cells and whichever binds first does so silently. The
    # Critic checks exactly this, and caught it here.
    return {"cell": max_cells_for(vertex), "vertex": vertex}


def max_cells_for(vertex_buffer):
    """The hard ceiling a given vertex reservoir imposes. The inverse of buffer_for."""
    return (int(vertex_buffer) + 4) // 2


def setup(case=None):
    """Okuda's starting conditions, certified against the paper before they are handed over."""
    _require_certified()
    if case is None:
        return {"cases": SETUP, "observed": OBSERVED, "table1": TABLE1, "regime": REGIME}
    s = dict(SETUP[case])
    s["buffers"] = buffer_for(s["grows_to"])        # from the DESTINATION, never from the seed
    s["observed"] = OBSERVED
    return s


def figure_target(key, k=3):
    """What must be reproduced, with the paper's own parameters and a checkable criterion.

    This is what makes 'reproduce the figure' a campaign objective rather than an aspiration:
    the target is a (chi, gamma) point, a named phenotype, and a criterion the metric bank can
    evaluate. The DIAGRAM as a whole is the real target -- any single cell can be hit by luck,
    but four distinct morphologies separated only by two numbers cannot.
    """
    f = dict(FIGURES[key])
    f["citations"] = [p.cite() for p in
                      ground(f"{f['phenotype']} chi gamma {f['shows']}", k=k)]
    return f


def phase_diagram():
    """The whole 2x2 as the campaign's qualitative reproduction target."""
    return {k: {kk: vv for kk, vv in v.items() if kk != "citations"}
            for k, v in FIGURES.items()}


# --------------------------------------------------------------------------- 3. REQUEST
def name_mechanism(description, k=5):
    """An operator request was filed. Does the literature already name this mechanism?"""
    passages = ground(description, k=k)
    if not passages:
        return {"prior_art": [], "verdict": "apparently novel -- no corpus match",
                "note": "an operator request with no prior art is the campaign's most valuable "
                        "output; record it and say what it would need to express."}
    return {"prior_art": [p.cite() for p in passages],
            "verdict": "prior art found -- read before implementing",
            "excerpts": [p.text[:200] for p in passages[:3]]}


# --------------------------------------------------------------------------- cli / smoke
if __name__ == "__main__":
    print("=" * 78)
    print("GROUNDER -- corpus")
    print("=" * 78)
    av = available()
    for name, path, kind in av[:8]:
        sz = os.path.getsize(path) // 1024
        print(f"  {kind:3}  {name:46} {sz:>7} KB")
    print(f"  ... {len(av)} sources total\n")

    q = sys.argv[1] if len(sys.argv) > 1 else \
        "which reaction diffusion regime does Okuda use for tubulation, activator inhibitor?"
    print(f"[1] PROPOSE -- ground({q!r})\n")
    for p in ground(q, k=4):
        print(f"  {p.cite()}")
        print(f"      {p.text[:200]}...\n")

    print("[2] GATE -- a hypothesis that negates a reference claim\n")
    h = "the tube survives removal of the outward extrusion force (growth-driven equilibrium)"
    v, why, cits = gate(h)
    print(f"  hypothesis: {h}")
    print(f"  verdict   : {v}")
    print(f"  why       : {why}")
    print(f"  citations : {cits}\n")

    print("[4] UNDERSTAND -- open-ended, no fixed claim list\n")
    u = understand("what sets the tube diameter, and what does gamma control?", k=3)
    for pp in u["passages"][:2]:
        print(f"  {pp['cite']}\n      {pp['text'][:230]}...\n")

    print("[5] REPRODUCE A FIGURE -- the paper's own parameters as a phase diagram\n")
    for key in ("fig5a", "fig5b", "fig7", "fig6"):
        f = FIGURES[key]
        print(f"  {f['figure']:12} chi={f['chi']:<6} gamma={f['gamma']:<7} -> {f['phenotype']:12}"
              f" | {f['criterion'][:56]}")
    print()

    print("[3] REQUEST -- does the literature name this mechanism?\n")
    r = name_mechanism("per-cell apical basal thickness giving emergent bending rigidity")
    print(f"  verdict  : {r['verdict']}")
    for c in r.get("prior_art", [])[:4]:
        print(f"  prior art: {c}")
    print("\ngrounder OK")


# ============================================================================ the standing gate
def certify(verbose=True):
    """Both registers, checked against the paper. Called before the Grounder advises anything.

    A guard nobody runs is the condition this agent was in yesterday morning: written, correct,
    and never invoked. So this is called from `setup()` -- the one entry point the round uses --
    and a failure REFUSES rather than warns. Being unable to prove where a number came from is
    not a reason to use it anyway.
    """
    bad_s = verify_setup()
    bad_t, notes = verify_table1()
    if verbose:
        for b in bad_s + bad_t:
            print(f"  [grounder] REFUSED: {b}")
        for n in notes:
            print(f"  [grounder] note: {n}")
    return bad_s + bad_t, notes


_CERTIFIED = {}


def _require_certified():
    """Certify once per process, and refuse to advise if the registers do not check out."""
    if "ok" not in _CERTIFIED:
        bad, _ = certify(verbose=True)
        _CERTIFIED["ok"] = not bad
        _CERTIFIED["why"] = bad
    if not _CERTIFIED["ok"]:
        raise ValueError("the Grounder's registers do not match the paper, so it will not "
                         "advise: " + "; ".join(_CERTIFIED["why"][:3]))


if __name__ == "__main__":
    import sys as _s
    bad, notes = certify()
    print(f"\n  SETUP  : {len(SETUP)} entries")
    print(f"  TABLE1 : {len(TABLE1)} values")
    print(f"\n  {'REFUSED -- ' + str(len(bad)) + ' problem(s)' if bad else 'CERTIFIED'}")
    _s.exit(1 if bad else 0)
