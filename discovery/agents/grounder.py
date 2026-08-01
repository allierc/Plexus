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
    # `n_cells` is where a run STARTS. `grows_to` is where it ENDS, and it is the one the buffer
    # must be sized from -- sizing from the seed is exactly the mistake that produced 1778. Okuda
    # is explicit that his largest tissues were "picked up on the growth process", i.e. they are a
    # destination reached from a smaller seed, not a seeded count.
    "tubulation": dict(
        n_cells=200, grows_to=4000, figures=["fig5a", "fig5b", "fig6"],
        quote="the initial tissue morphology was simply set to be a spherical vesicle of a "
              "monolayer cell sheet composed of about 200 cells",
        note="the tubulation and branching cases. Our campaign ran 150."),
    "undulation": dict(
        n_cells=2000, grows_to=4000, figures=["fig7"],
        quote="a spherical vesicle of a monolayer cell sheet composed of about 2,000 cells, "
              "whose patterns reached steady states",
        note="the whole-tissue patterning case; this is the ball the ~5-spot count refers to."),
    "grown": dict(
        n_cells=4000, grows_to=4000, figures=[],
        quote="the individual tissues in (b) were composed of about 4,000 cells, which were "
              "picked up on the growth process",
        note="a destination, not a start -- the number the buffer has to accommodate."),
}

# What he reports SEEING, which is what we calibrate against -- his parameter values live in a
# differently-scaled model and are not transferable (finding F009).
OBSERVED = {
    "n_spots_at_2000": 5,
    "note": "about five activator domains on the 2000-cell ball. Calibrate the chemistry until we "
            "reproduce this COUNT; never copy chi across the two models.",
}

# Table 1, normalised. Quoted so a fixture can assert them rather than a person retyping them.
TABLE1 = {
    "alpha_hill":  (10.0,  "Hill coefficient in growth rate"),
    "rho_sw":      (0.5,   "switching concentration for growth rate"),
    "phi_inhib":   (10.0,  "diffusivity of inhibitor"),
    "tau_cycle":   (50.0,  "cell cycle, in units of eta_c/kappa_s"),
    "kappa_s":     (0.2,   "normalised surface energy"),
    "eta_c":       (0.25,  "normalised cell friction"),
    "v_ref":       (1.0,   "reference cell volume -- defines the unit of length"),
    "gamma_range": ((0.01, 100.0), "time characteristics of patterning -- VARIED"),
    "chi_range":   ((0.001, 0.1),  "spatial characteristics of patterning -- VARIED"),
}

# Two regime INEQUALITIES, which are what actually define his setting. Worth asserting rather than
# tuning: they are the reason the model is quasi-static and effectively incompressible.
REGIME = [
    ("k_v >> kappa_s * v_ref**(2/3)", "incompressibility"),
    ("tau_cycle >> eta_c / kappa_s",  "quasi-static: mechanics relaxes between biological events"),
]


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
    return {"cell": n, "vertex": 2 * n + 4}


def max_cells_for(vertex_buffer):
    """The hard ceiling a given vertex reservoir imposes. The inverse of buffer_for."""
    return (int(vertex_buffer) + 4) // 2


def setup(case=None):
    """Okuda's starting conditions. `case` in {tubulation, undulation, grown}, or all of them."""
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
