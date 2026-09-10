# What to launch

All commands from the repo root, in the `neural-graph-linux` env (or `conda run -n connectome-gnn`
on the cluster). Nothing needs `PYTHONPATH`.

## Before a push (5 min, any host)

    python -m pytest tests -q

The identity suite: gradients, scaling laws, invariants, plumbing. Expected: all pass, 2 xfail.
Add `--quick` working points if the change touches a tissue operator (3 short reruns, GPU, ~4 min):

    PLEXUS_REGRESSION_DEVICE=cuda:1 python -m pytest tests/regression -m regression --quick

## Overnight (20 min on an A6000)

    nohup python tools/regression_run.py --device cuda:1 > log/regression/nightly.log 2>&1 &

or on the cluster:

    bsub -q gpu_l4 -gpu 'num=1' -n 8 -W 4:00 -J plexus_regression \
         -o jobs/logs/regression_%J.out -e jobs/logs/regression_%J.err bash jobs/regression_nightly.sh

Runs the seed table, the invariants, the scaling tests and every registered working point, and
appends one record to `log/regression/archive.jsonl`.

## Read the results

    python tools/regression_dashboard.py

Opens the browser on the archive: the run matrix (click a cell for its violations), a two-run
comparison metric by metric, and frame time per working point across runs. `--no-browser` and
`--port` if needed.

## When a run is accepted as a new working point

    python tools/fingerprint.py register graphs_data/<family>/<name>      # then commit the JSON
    python tools/fingerprint.py list

Registered that day, or it is a movie, not a working point. `tests/regression/WORKING_POINTS.md`
lists them and says why each one is there.

## When a working point is meant to change

    python tools/fingerprint.py refresh <name> --because "why"           # in its own commit, no source change

A source commit that needs a refresh to go green is a working-point change and is reviewed as one.

## When a spec's seed changes on purpose

    python tools/seed_conventions.py            # rebuilds tests/regression/seed_conventions.json
    python tools/seed_conventions.py --check    # what differs from the table

## One working point by hand

    python tools/fingerprint.py diff cvd2_adder_tension --device cuda:1
    PLEXUS_REGRESSION_DEVICE=cuda:1 python -m pytest tests/regression -m regression -k cvd2_adder_tension

Design and the bands: `tests/REGRESSION_PLAN.md`. Status of the old suite: `TEST_KNOWN_FAILURES.md`.
