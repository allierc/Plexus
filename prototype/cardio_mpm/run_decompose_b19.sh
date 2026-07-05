#!/bin/bash
# Residual decomposition for all B19 slots (PARTIAL@450-500it)
set -e

echo "=== s0 gnarr_fdev005 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.05 \
  --learn fibre,gain,dur,stiff --n_iter 2400 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b19_s0_b19_gnarr_fdev005/checkpoints/model_00450.pt \
  --eval_decompose archive/p3_b19_s0_b19_gnarr_fdev005/residual.png

echo "=== s1 gnarr_fdev005_wl35 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.05 \
  --learn fibre,gain,dur,stiff --n_iter 2400 \
  --fibre_wl 35 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b19_s1_b19_gnarr_fdev005_wl35/checkpoints/model_00450.pt \
  --eval_decompose archive/p3_b19_s1_b19_gnarr_fdev005_wl35/residual.png

echo "=== s2 gnarr_fdev01 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.1 \
  --learn fibre,gain,dur,stiff --n_iter 2400 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b19_s2_b19_gnarr_fdev01/checkpoints/model_00450.pt \
  --eval_decompose archive/p3_b19_s2_b19_gnarr_fdev01/residual.png

echo "=== s3 gnarr_fdev003 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.03 \
  --learn fibre,gain,dur,stiff --n_iter 2400 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b19_s3_b19_gnarr_fdev003/checkpoints/model_00500.pt \
  --eval_decompose archive/p3_b19_s3_b19_gnarr_fdev003/residual.png

echo "=== s4 gnarr_nofibre ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 0 \
  --learn gain,dur,stiff --n_iter 2400 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b19_s4_b19_gnarr_nofibre/checkpoints/model_00500.pt \
  --eval_decompose archive/p3_b19_s4_b19_gnarr_nofibre/residual.png

echo "=== s5 ctrl_fdev005 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 --siren_fibre 1 --fibre_dev 0.05 \
  --learn fibre,gain,dur,stiff --n_iter 2400 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b19_s5_b19_ctrl_fdev005/checkpoints/model_00450.pt \
  --eval_decompose archive/p3_b19_s5_b19_ctrl_fdev005/residual.png

echo "=== DONE ==="
