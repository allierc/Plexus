# Fig.3 batch 10 -- CAPSTONE: map the PURE-ADVECTIVE arrest law with chemotaxis FULLY OFF (omega=0).
# Batch 9 answered the headline question: the residual Nc_final v0-rise is PURE ADVECTIVE (survives omega=0:
# Nc_final 2@v0.30 vs 6@v0.70 with chemotaxis OFF), Nc_max stays v0-flat ~22, and the FULL droplet->stream->
# domain cascade survives omega=0 -- flocking+pressure+advection alone reproduce Fig.3. But omega=0 was only
# sampled at v0=0.30,0.70. THIS BATCH = the clean omega=0 v0 FAMILY (the definitive non-chemotactic arrest
# curve Nc_final(v0)), the direct control for the omega=0.6/0.3 humps, plus one omega=0 low-v0 endgame long run.
# Fixed L=220, N=360, 48k, s0. One variable per slot (v0) from the batch-9 omega=0 parents w00_v030/w00_v070.

# --- the FULL v0 family at omega=0 (ZERO chemotaxis): map Nc_final(v0) as pure advective arrest. Expect Nc_max
#     v0-flat ~22, Nc_final rising with v0 (hump), no c-info re-rise, cascade at every v0. ---
w00_v010 : --kind hydro --mode coarsen --v0 0.10 --omega 0.0 --L 220 --N 360 --nsteps 48000 --seed 0
w00_v020 : --kind hydro --mode coarsen --v0 0.20 --omega 0.0 --L 220 --N 360 --nsteps 48000 --seed 0
w00_v040 : --kind hydro --mode coarsen --v0 0.40 --omega 0.0 --L 220 --N 360 --nsteps 48000 --seed 0
w00_v050 : --kind hydro --mode coarsen --v0 0.50 --omega 0.0 --L 220 --N 360 --nsteps 48000 --seed 0
w00_v060 : --kind hydro --mode coarsen --v0 0.60 --omega 0.0 --L 220 --N 360 --nsteps 48000 --seed 0
# (v0=0.30, 0.70 @omega=0 already mapped batch 9: Nc_final 2, 6 -- this family fills the curve between/around them.)
# --- omega=0 endgame: does low-v0 reach Nc=0 blob WITHOUT chemotaxis given 2x time (as vlo_v010@omega0.6 did)? ---
w00_v010_long : --kind hydro --mode coarsen --v0 0.10 --omega 0.0 --L 220 --N 360 --nsteps 96000 --seed 0
# --- seed robustness of the pure-advective endgame at mid v0 (does omega=0 wash out the metastable scatter?) ---
w00_v050_s1   : --kind hydro --mode coarsen --v0 0.50 --omega 0.0 --L 220 --N 360 --nsteps 48000 --seed 1
