# Slots designed by the agent each batch.
# Format: name : SPEC specs/<file>.yaml [key val ...]  (dotted keys = op.param overrides, e.g. mpm_spin.omega)
#
# BATCH 32 (2026-07-05). STAGE 1E CLOSED -> ADVANCE to INT (integration). INT batch 1: PARTITION x PROLIFERATION.
#
# 1E CLOSED (Batch 32 read of b31): the b30 `core_a12` core-shell signal (mi_type_x 0.2229) was a SINGLE-SEED
#   FLUKE -- both seed replicates fell <=0.06 (seed1 0.0067 DECLINING, seed2 0.0539), NO slot reproduced the
#   climb, NO monotone mi_type_x trend across the self-pull ladder (a10 0.019 -> a12 0.007/0.054 -> a14 0.017
#   -> a16 0.065 = +-0.05 scatter around ctrl 0.013). 5th single-seed clean point in the campaign to fail
#   replication. The LATERAL demix HELD orthogonally (ctrl_g10 seg 0.485 == b29 [established] 0.494+-0.080).
#   1E DELIVERABLE [established]: heterotypic two-channel chemotactic cross-repulsion -> gain-scaled LATERAL
#   demix, 3 seeds (g10 0.494+-0.080 = 6.5.SD; g20 0.781+-0.085 = 9.5.SD; escape/nn_min-safe). Core-shell
#   (radial) geometry NOT robustly achievable [rejected]. 1E OPERATING POINT = embryo_1E_ctrl_g10.yaml.
#
# INT batch 1 QUESTION: does the [established] demix SURVIVE proliferation? Turn cell_divide back ON in the
#   g10/g20 demix at the escape-safe anch20/k4 substrate; grow population 2x/3x/4x (cap = max_occ 0.88 * buffer;
#   buffer 300/450/600 -> 264/396/528). Daughters inherit node_type (cell_divide.py:62) so a demix should
#   survive in principle, but proliferation crowds the core (at 4x disc-spacing ~0.025 ~ r0 0.02 = frontier),
#   adds field sources, and re-opens the 1C escape/packing frontier.
#
# HYPOTHESIS (Batch 32): the gain-scaled demix SURVIVES division through ~3x -- seg stays >= mixed baseline and
#   gain-ordered (g20 > g10) as N grows, with division DEFORMING the shell (deform_rms up vs the nodiv ctrl)
#   while TIER-1 holds (collapsed 0, escape <~0.06, nn_min >= r0 0.018). At 4x (esp. g20_4x) the packing
#   frontier bites: nn_min drops toward/below r0 and/or seg degrades as crowding overwhelms the sort.
#   Falsifier: if seg collapses to the mixed baseline at 2x already (division scrambles domains faster than it
#   reinforces them) OR every division slot breaks TIER-1 -> partition and proliferation are INCOMPATIBLE at
#   this substrate -> report [open], retreat to nodiv operating point, and probe division-rate / anchor-stiffness.
#
# Judge TIER-1 FIRST (collapsed 0, nn_min >= ~0.018, escape not runaway), THEN seg/mi survival vs the nodiv
#   ctrl (does seg hold and stay gain-ordered as N grows) AND deform_rms (does division deform the shell).
#   Single lever per slot = {gain g10/g20} x {division cap 2x/3x/4x} + a slow-fill rate probe. Roles: 4 exploit
#   (g10 ladder + g20_2x = the survival core), 3 explore (g20_3x, g20_4x frontier, slowfill), 1 control (nodiv).
#
g10_2x         : SPEC specs/embryo_INT_g10_2x.yaml frames 12000 stride 16
g10_3x         : SPEC specs/embryo_INT_g10_3x.yaml frames 12000 stride 16
g10_4x         : SPEC specs/embryo_INT_g10_4x.yaml frames 12000 stride 16
g20_2x         : SPEC specs/embryo_INT_g20_2x.yaml frames 12000 stride 16
g20_3x         : SPEC specs/embryo_INT_g20_3x.yaml frames 12000 stride 16
g20_4x         : SPEC specs/embryo_INT_g20_4x.yaml frames 12000 stride 16
slowfill_g10_4x: SPEC specs/embryo_INT_slowfill_g10_4x.yaml frames 12000 stride 16
ctrl_g10_nodiv : SPEC specs/embryo_1E_ctrl_g10.yaml frames 12000 stride 16
