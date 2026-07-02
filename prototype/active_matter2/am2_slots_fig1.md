# Fig.1 batch 10 (FINAL) -- CONTINUUM-INHIBITOR VORTEX + complete 6-state montage. Batch-9 read (spiral_seed):
#   vort_seed (spiral_seed1)   P0.29  Nc6 : NOT a spiral -- small FILLED rainbow pinwheels + a worm-stream; chemical = compact blobs + worm.
#   vort_seed_dif (diffuse0.18): P0.105 Nc6 : same attractor, a few hollow C-loops + small blobs. not a spiral.
#   vort_seed_th05 (c_th0.05)  : P0.09  Nc10: same -- MORE small pinwheels (excitability = pinwheel COUNT, not one big disk).
#   vort_noseed (spiral_seed0) : P0.271 Nc6 : MORPHOLOGICALLY IDENTICAL to the seeded s0. DECISIVE.
# CONCLUSION: the one-shot broken-front seed WASHES OUT -- seed==no-seed at the final frame. ROOT CAUSE: recovery `s` (Eq 5)
#   lives on the AGENTS, not the field; agents are pulled into their own front and scramble the refractory tail in a few ticks,
#   so a phase singularity has nothing space-fixed to pin to. The vortex is a MEDIUM problem (no continuum inhibitor), not a
#   nucleation problem. NEW MECHANISM (am2_ops Refract + Relay rf_th; am2_job rf_tau/rf_gain/rf_th): a per-voxel refractory
#   FIELD `fld._rf` (d_t rf = gain*Theta(c-c_th) - rf/tau); relay with rf_th<1 is blocked where rf>rf_th, so a passed front
#   leaves a SPACE-FIXED wake the next front cannot re-invade -> a broken front pins a singularity and winds into a SUSTAINED
#   spiral. rf_tau = refractory period = spiral core size. DEFAULT-OFF (rf_th=2.0, refract unscheduled unless rf_tau>0), so all
#   other states + the no-rf control are byte-for-byte unchanged.
# KEY EXPERIMENT: run the seed on a CONTINUUM-excitable base (rf_tau>0); bracket rf_tau (core size) by ONE knob; s2 = same base
#   with NO rf and NO seed (control: does the field inhibitor change the attractor from pinwheels to a spiral?). 5 anchors -> full montage.

# --- VORTEX = continuum-inhibitor excitable spiral: broken-front seed on a FitzHugh-Nagumo (refract) base (NEW mechanism) ---
# parent vort_fhn: excitable slow-fill base + spiral_seed1 + refract rf_tau40 rf_gain0.10 rf_th0.5
vort_fhn      : --kind agent --state vortices --seed 4 --frames 1800 --n 12000 --move_speed 0.003 --radius 0.03 --res 200 --gamma 0.25 --align_noise 0.05 --omega 0.60 --beta 0.28 --sigma 1.5 --eps 0.02 --diffuse 0.30 --decay 0.02 --c_th 0.08 --c_base 0.04 --repel 0.02 --r0 0.012 --spiral_seed 1.0 --rf_tau 40 --rf_gain 0.10 --rf_th 0.5
vort_fhn_tau  : --kind agent --state vortices --seed 4 --frames 1800 --n 12000 --move_speed 0.003 --radius 0.03 --res 200 --gamma 0.25 --align_noise 0.05 --omega 0.60 --beta 0.28 --sigma 1.5 --eps 0.02 --diffuse 0.30 --decay 0.02 --c_th 0.08 --c_base 0.04 --repel 0.02 --r0 0.012 --spiral_seed 1.0 --rf_tau 80 --rf_gain 0.10 --rf_th 0.5
vort_pin      : --kind agent --state vortices --seed 4 --frames 1800 --n 12000 --move_speed 0.003 --radius 0.03 --res 200 --gamma 0.25 --align_noise 0.05 --omega 0.60 --beta 0.28 --sigma 1.5 --eps 0.02 --diffuse 0.30 --decay 0.02 --c_th 0.08 --c_base 0.04 --repel 0.02 --r0 0.012 --spiral_seed 0.0 --rf_tau 0.0

# --- solved-state ANCHORS (montage controls, re-run winners) -- complete the 6-state final montage ---
str_gamma      : --kind agent --state streams --seed 0 --frames 1000 --gamma 0.35
ring_more      : --kind agent --state ring-streams --seed 3 --frames 1600 --n 9000 --radius 0.03 --res 220 --beta 0.18 --sigma 1.3 --eps 0.035 --diffuse 0.22 --decay 0.014 --gamma 0.50 --align_noise 0.03 --omega 0.45 --repel 0.018 --r0 0.011
drop_slow      : --kind agent --state active-droplets --seed 4 --frames 1800 --n 4500 --move_speed 0.002 --radius 0.04 --beta 0.24 --sigma 1.4 --eps 0.012 --diffuse 0.13 --decay 0.025 --gamma 0.55 --align_noise 0.045 --omega 0.30 --repel 0.04 --r0 0.016
bands          : --kind agent --state polar-bands --seed 2 --frames 1000 --n 9000 --move_speed 0.007 --radius 0.035 --beta 0.08 --sigma 1.1 --eps 0.05 --diffuse 0.14 --decay 0.03 --gamma 0.42 --align_noise 0.06 --omega 0.06 --repel 0.012 --r0 0.009
aggreg         : --kind agent --state aggregation --seed 2 --frames 1200 --n 8000 --move_speed 0.005 --radius 0.03 --res 200 --gamma 0.0 --align_noise 0.06 --omega 1.3 --beta 0.16 --sigma 2.6 --eps 0.05 --diffuse 0.12 --decay 0.08 --repel 0.02 --r0 0.012 --marker dot
