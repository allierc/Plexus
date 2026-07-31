# Standing instructions — Atlas: jax-morph

**Target.** `papers/jax-morph` — Deshpande, Mottes et al., *Engineering morphogenesis of cell
clusters with differentiable programming*, Nat Comput Sci 2025. Apache-2.0, Python/JAX, runnable.

**Two deliverables, and neither is optional.**

1. **The figure.** One headline result of the paper reproduced inside Plexus, from operators that
   were normalized from the reference rather than fitted to its output.
2. **The saturation ledger.** For every mechanism in the repository: alias, refinement, genuinely
   new, or our own unpromoted backlog. This is the measurement `plexus2.tex` App. E.1 says the
   atlas exists to make, and it is destroyed by an inflated count of "new".

**Why this target and not another.** The Okuda track reproduces a paper with *no code*. Every
disagreement there is unfalsifiable — wrong operator, wrong parameter, or wrong reading of a
figure, and no way to tell which. Here the authors' implementation runs in `_oracle/`, so
"faithful" becomes a number instead of a judgement.

## The rules

- **The record is the product.** `atlas_record.yaml`. Analysis that is not in the record does not
  exist. `campaign/analysis.md` is the append-only log, not the deliverable.
- **One mechanism per call.** Never edit a neighbouring entry; the driver reverts the whole call
  if you do.
- **A verdict is an obligation.** The baseline is the **promoted** language — the 52 registered
  contracts in `plexus.operators`, and nothing else. `new` must survive that set;
  `alias`/`refinement` must name a registered contract and say what differs. Unreviewed code in
  `prototype/` or `candidates/` is not the language, though it is often worth reading before
  reimplementing something.
- **The source wins over the paper**, and the contradiction gets recorded. That contradiction is
  among the most valuable things a reproduction can produce.
- **A threshold decided after seeing the result is not a test.** The differ writes the metric and
  the threshold into the record *before* running anything.
- **Never swallow an exception around an artefact.** A blank panel that looks deliberate is worse
  than a crash. (Three silent no-ops shipped in the discovery loop before anyone noticed.)

## The split

| local (this devcontainer) | the oracle venv |
|---|---|
| everything Plexus: torch, the engine, the registry, the record | jax, equinox, diffrax, `jax_morph` |
| the agents, the validator, the ledger | reference runs only, via `oracle.py` |

`import jax` must never succeed in a Plexus process. That is not tidiness: a process that can
reach the reference implementation can borrow its answer, and a differential test that can be
contaminated is not a test.
