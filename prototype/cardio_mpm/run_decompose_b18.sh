#!/bin/bash
# Residual decomposition for all B18 slots (PARTIAL: ~1100it)
set -e

echo "=== s0 gnarr_wl35 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 \
  --siren_fibre 1 --fibre_dev 0.1 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 35 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b18_s0_b18_gnarr_wl35/checkpoints/model_01100.pt \
  --eval_decompose archive/p3_b18_s0_b18_gnarr_wl35/residual.png

echo "=== s1 gnarr_deep ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 \
  --siren_fibre 1 --fibre_dev 0.1 \
  --learn fibre,gain,dur,stiff --n_iter 4800 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b18_s1_b18_gnarr_deep/checkpoints/model_01050.pt \
  --eval_decompose archive/p3_b18_s1_b18_gnarr_deep/residual.png

echo "=== s2 gnarr_fdev005 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 \
  --siren_fibre 1 --fibre_dev 0.05 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b18_s2_b18_gnarr_fdev005/checkpoints/model_01100.pt \
  --eval_decompose archive/p3_b18_s2_b18_gnarr_fdev005/residual.png

echo "=== s3 fdev01_angle025 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren \
  --siren_fibre 1 --fibre_dev 0.1 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.25 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b18_s3_b18_fdev01_angle025/checkpoints/model_01100.pt \
  --eval_decompose archive/p3_b18_s3_b18_fdev01_angle025/residual.png

echo "=== s4 fdev01_stifflo60 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren \
  --siren_fibre 1 --fibre_dev 0.1 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 60 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b18_s4_b18_fdev01_stifflo60/checkpoints/model_01100.pt \
  --eval_decompose archive/p3_b18_s4_b18_fdev01_stifflo60/residual.png

echo "=== s5 ctrl_gnarr ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 \
  --siren_fibre 1 --fibre_dev 0.1 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b18_s5_b18_ctrl_gnarr/checkpoints/model_01100.pt \
  --eval_decompose archive/p3_b18_s5_b18_ctrl_gnarr/residual.png

echo "=== DONE ==="
