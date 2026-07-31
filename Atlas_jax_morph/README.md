# Atlas — jax-morph

The twin of `discovery/`, for the other half of the Plexus 2 programme. Discovery searches for a
mechanism nobody wrote down; the Atlas takes a published system **whose code exists**, decomposes
it into the Plexus operator algebra, and measures whether that algebra is converging.

**Read `atlas_note.pdf` first** — plain English, phase by phase. This file is how to run things.

```bash
export PY=/workspace/.conda_envs/neural-graph-linux/bin/python
cd /workspace/Plexus/Atlas_jax_morph
```

## The gates — run these before trusting anything

```bash
$PY registry_view.py --json        # freeze the baseline: 52 registered contracts
$PY record.py --selftest           # 12 rules, all fire on a record built to break them
$PY saturation.py --selftest       # the ledger overrules the record, and says so
python3 oracle.py verify           # provenance + the reference imports (its own venv)
$PY atlas.py status                # the record, rung by rung
```

## The pieces

| file | what it is |
|---|---|
| `atlas_record.yaml` | **the product.** One record per repository, one entry per mechanism (Lst. E.1 of `paper/plexus2.tex`). |
| `record.py` | the schema, the status ladder, and the twelve rules that decide whether the record may be believed. |
| `registry_view.py` | the frozen baseline: what the **promoted** language can already say. Nothing in `prototype/` or `candidates/` counts. |
| `saturation.py` | the measurement — alias / refinement / new, and the cumulative-new curve. |
| `inventory.py` | seeds the record from the clone's syntax tree, plus the 4 architectural contracts a scan cannot see. |
| `oracle.py`, `smoke.py` | the reference implementation, in its own interpreter, with provenance on every artefact. |
| `paper.py` | the PDF as plain text with page markers (`--grep <term>` finds a term and its page). |
| `atlas.py` | the driver. Every agent call is a guarded transaction. |
| `agents/atlas_agents.py` | the six roles and their prompts. |

## Running the loop

```bash
$PY atlas.py prompt --role excavator --mech division     # see the prompt, call nothing
$PY atlas.py step   --role excavator --mech division     # one guarded call
$PY atlas.py phase  --role excavator --all --limit 4     # everything due at that role
$PY atlas.py step   --role normalizer --mech division --skeptic
```

Each call: snapshot → run the role → three checks → commit or **revert**.

1. **Blast radius** — only the target mechanism may have changed.
2. **Rung** — a role may only reach its own status (`excavator` → `inspected`, `differ` →
   `validated`, …). Rungs are earned by artefacts, never by assertions.
3. **The twelve rules** — `record.py` must pass. Fail once → the violations go back to the agent.
   Fail twice → the mechanism is **blocked** (`_state/blocked.json`), never silently skipped.

A call that changes nothing counts as a failure. Reverts are logged with the agent's own words in
`_state/reverts.jsonl`.

## The oracle

```bash
python3 oracle.py setup            # build _oracle/venv, install the local clone, record versions
python3 oracle.py smoke            # the authors' proliferation model -> _oracle/runs/smoke/
python3 oracle.py run my_script.py --name diff_division
```

`import jax` must never succeed in a Plexus process — a process that can reach the reference can
borrow its answer, and a differential test that can be contaminated is not a test. The oracle runs
in `_oracle/venv` (jax 0.11.0, jax-morph 0.4.0 from clone `ace08b8`), the Plexus side in
`neural-graph-linux` (torch).

## Layout

```
atlas_record.yaml   the record          _oracle/runs/    reference artefacts + provenance
atlas_note.{tex,pdf}  the note          _state/          baseline, paper text, ledger, reverts
campaign/           instruction · analysis (append-only)
agents/             the six roles
```
