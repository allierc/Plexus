# Plexus

**A language for saying what a biological mechanism *is*, precisely enough to run it.**

A biologist explains a tissue with parts, arrows, and boxes inside boxes: these cells, pulling
on each other, in this signal, dividing at this rate. That sketch is a hypothesis. It is also,
usually, where the explanation stops — because turning it into something that runs means
writing a simulator, and the simulator is a different artefact from the claim it was meant to
test. Plexus removes that gap. The sketch *is* the program.

## The idea

A model states a hypothesis in two parts, and both are declared rather than coded.

**What exists** — the entities, the state each carries, the relations between them, the
continuous fields they live in.

**What happens** — the activities, one per mechanism: adhesion, diffusion, division, death,
contraction, a beating cilium. Each names the entities, state and relations it needs, and
nothing else.

The *organization* is declared too, and that is the part a list of equations loses: where the
parts sit, what order the activities run in and at what rates, and how a level is made of the
level beneath it — with information carried up to the whole and influence pushed back down.

Everything then falls into four kinds of object. **Entities**, grouped into sets by what kind
of thing they are. **Fields**, the continua they move through. **Operators**, the activities,
each belonging to one of a handful of structural kinds according to *what it moves and where
to*. And a **schedule**, which is the organization in time.

One interpreter runs all of it. The same machine runs an epithelium as a vertex model, a
continuum as a material-point method, a tissue growing into a gel, a reaction–diffusion
pattern, a swimming larva, a beating cilium, and a pair of colliding galaxies — because none
of those is a program here. Each is a spec.

## Why it is built this way

**A vocabulary is a cost.** Every operator is a word, and two words that mean almost the same
thing make a reader compare their definitions to find out which one a model is getting. So the
library grows by *merging*, not by adding variants — and shrinks when it can. Perfection is
when there is nothing left to remove.

**It is differentiable.** A hypothesis you can run is useful; a hypothesis you can *fit* is a
different thing. Because the forward model is built from differentiable operators, observations
can be turned back into the parameters, the geometry, or the schedule that would produce them —
and a residual that will not go away points at a missing activity rather than a wrong number.

**Nothing is claimed that is not measured.** A model that renders beautifully and conserves
nothing is not a weaker result, it is not a result. Invariants — momentum, mass, the sampling a
numerical method actually requires — are checked and reported, because every one of them has
already been found broken behind a picture that looked fine.

## Where to look

- **The site, with a gallery** — <https://allierc.github.io/Plexus/>. Each clip's title opens
  the exact spec that produced it, which is the whole argument in one click.
- **The paper**, for the language itself, its schematics and its glossary —
  <https://allierc.github.io/Plexus/paper/plexus2.pdf>.
- **The operator library**, generated from the operators' own documentation —
  <https://allierc.github.io/Plexus/library.html>.

## Running one

```bash
python Plexus_Main.py -o generate tissue/sheet_morphogen_die
```

A run writes its trajectory and its movie to a data root outside the repository, and captions
its own movie. The specs live in `config/`, the language in `src/plexus/`, the operator library
in `src/plexus/operators/`, and the paper in `paper/`.
