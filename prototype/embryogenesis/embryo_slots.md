# Slots designed by the agent each batch.
# Format: name : SPEC specs/<file>.yaml [key val ...]
#
# BATCH 5 (2026-07-02) — Stage 1B: cross DEFORM 0.02 at ESCAPE 0 via the sparse-n + high-mass + spin route.
# Batch 4 revised the mechanism: (1) agent_mass IS a deform lever at FIXED n=95 (deform 0.0027->0.0346 over
# mass 2e-6->1e-3; Batch-3's "mass inert at fixed n" was a narrow-window artefact). (2) escape = per-cell
# push × density (BOTH needed): at n=95 mass 5e-4 -> escape 0.042, but SAME mass 5e-4 at n=44 -> escape 0
# (s7, TRUE tiling nn_min 0.0229>r0). So SPARSE n is an escape SHIELD. (3) spin is a clean amplifier: m2e4+
# ω0.6 -> deform +40% (0.0105->0.0148), migration 0.291->0.687, escape UNCHANGED. Best clean (escape≈0)
# deform so far: 0.0148 (s5), 0.0115 (s7 escape 0), 0.0106 (s6 escape 0) — none crosses 0.02 CLEANLY yet.
#
# HYPOTHESIS: at SPARSE escape-free n≈44, cells sit off the shell so agent_mass can go to ~1e-3 and
# mpm_spin.omega to ~0.6-1.0, building deform ≳0.02 via internal flow WITHOUT triggering escape — the clean
# Stage-1B pass comes from sparse-n + high-mass + spin, NOT dense high-mass. Escape fires only when n is high
# enough to press cells against the wall. Corollary: if escape RISES with mass even at n=44 (sp_m1e3), the
# sparse shield has a ceiling and spin (internal flow, no wall push) is the only route past it.
# Prediction ranking: sp_m1e3_spin & n60_m1e3_spin cross deform 0.02 at escape≤0.02; sp_m1e3 shows whether
# raw mass or spin carries it; dense_m1e4 tests the dense+low-mass route to 0.02; spinonly_hi isolates
# pure-circulation deform (no mass push). Guardrails: escape (HARD FAIL) + collapsed + r_cell_max.
# All R1-compliant (existing scalar knobs on embryo_1A.yaml: agent_mass, mpm_spin.omega, agent.div_rate).
# RE-RUN at frames=6000 stride=8 (driver restarted at batch 5 to honor the 6000-frame user directive; the
# first submission ran at frames=3000). embryo_1A.yaml already carries move_speed 0.12 + n_frames 6000. At
# 6000 frames the slow deform/migration dynamics get 2x longer to saturate (strengthens the 0.02 test), and
# division counts run a bit higher: div 0.0 -> n=44 (no division) ; div 0.05 -> n≈70-90 ; div 0.30 -> plateaus
# near the max_occ 0.9 cap (~n=250-300). ~13-min L4 per job (within the 30-min wall). Watch s4 (n60_m1e3_spin):
# mass 1e-3 as n creeps up under div 0.05 may drift toward the escape corner -- an informative shield-ceiling test.
#
# --- CONTROL (push+spin ablated to base: sparse n=44, mass 2e-6, default ω0.3 -> deform floor) ---
ctrl_sparse_base : SPEC specs/embryo_1A.yaml agent_to_mpm.agent_mass 2.0e-6 agent.div_rate 0.0
#
# --- EXPLOIT: sparse n=44, crank mass + spin toward the clean 0.02 crossing ---
sp_m1e3       : SPEC specs/embryo_1A.yaml agent_to_mpm.agent_mass 1.0e-3 agent.div_rate 0.0
sp_m5e4_spin  : SPEC specs/embryo_1A.yaml agent_to_mpm.agent_mass 5.0e-4 mpm_spin.omega 0.6 agent.div_rate 0.0
sp_m1e3_spin  : SPEC specs/embryo_1A.yaml agent_to_mpm.agent_mass 1.0e-3 mpm_spin.omega 0.6 agent.div_rate 0.0
n60_m1e3_spin : SPEC specs/embryo_1A.yaml agent_to_mpm.agent_mass 1.0e-3 mpm_spin.omega 0.6 agent.div_rate 0.05
#
# --- EXPLORE: dense + low-mass route (s6 pushed) — does moderate mass at n≈224 reach 0.02 at escape 0? ---
dense_m1e4    : SPEC specs/embryo_1A.yaml agent_to_mpm.agent_mass 1.0e-4 agent.div_rate 0.30
#
# --- EXPLORE: pure-circulation deform — high spin, base mass (no push) — how much deform from spin ALONE? ---
spinonly_hi   : SPEC specs/embryo_1A.yaml agent_to_mpm.agent_mass 2.0e-6 mpm_spin.omega 1.2 agent.div_rate 0.0
#
# --- EXPLORE: moderate mass + high spin at n=44 — migration/deform ceiling of the clean amplifier ---
sp_m2e4_spin1 : SPEC specs/embryo_1A.yaml agent_to_mpm.agent_mass 2.0e-4 mpm_spin.omega 1.0 agent.div_rate 0.0
