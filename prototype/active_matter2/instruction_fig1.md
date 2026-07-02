# instruction -- Fig. 1 reproduction loop (agent-based collective states)

## Method (you are a scientist, not an optimizer)
Reproduce the SIX collective dynamic states of Ziepke et al. (2022) Fig. 1 with the
agent-based model, and make the reproduced montage AGREE with `paper_fig1.png`. Agreement
is QUALITATIVE morphology of BOTH rows: (top) particles coloured by orientation, (bottom)
the chemical field `c`. There is NO scalar loss -- the deliverable is the knowledge ledger
mapping parameter regime -> state -> morphology, plus a montage that matches the paper.

Each batch is ONE set of experiments answering ONE question. Start from OBSERVATIONS
(the previous montage), name the biggest SURPRISE, form ONE hypothesis, design the
smallest experiments that distinguish live explanations. Change ONE variable per slot
from its parent (causal inference). Keep controls/ablations.

## The model (am2_ops.py, on a periodic chemical field)
`glide` (v0 n) · `polar_align` (Gamma alignment + angular noise) · `chemotax` (turn up grad c,
gain omega) · `relay` (excitable Schmitt emission, rate beta, refractory via s) · `adapt`
(ds=eps(c-s)) · `repel` (hard core) · field `diffuse`(Dc) + `decay`(alpha).

## Slot schema (write to am2_slots_fig1.md, <=8 lines)
`name : --kind agent --state <target> --<param> <val> ...`
Params (defaults in am2_job.py AGENT_DEFAULTS): `--n --move_speed --radius --res --frames
--seed --omega --gamma --align_noise --beta --sigma --eps --diffuse --decay --repel --r0
--marker`. `--state` is a free label (which paper panel this targets). Comment lines start `#`.
Each slot runs on one GPU; up to 8 per batch.

## Ranking / success
A batch SUCCEEDS if it improves the morphology match of >=1 state OR sharpens the
parameter->state map (a clean "this knob does X" is a win). Report the best-matching slot
per state. A slot with no panel.png FAILED -- design around it.

## Files
- knowledge_fig1.md : cumulative causal ledger (UPDATE, never erase)
- analysis_fig1.md  : dated per-batch narrative (append)
- fig1_b<NN>_montage.png : our panels + the paper reference (READ it each batch)
