#!/bin/bash
# Residual decomposition for all B16 slots
set -e

echo "=== s0 fdev01 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --siren_fibre 1 --fibre_dev 0.1 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b16_s0_b16_fdev01/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b16_s0_b16_fdev01/residual.png

echo "=== s1 fdev02 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --siren_fibre 1 --fibre_dev 0.2 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b16_s1_b16_fdev02/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b16_s1_b16_fdev02/residual.png

echo "=== s2 fdev03 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --siren_fibre 1 --fibre_dev 0.3 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b16_s2_b16_fdev03/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b16_s2_b16_fdev03/residual.png

echo "=== s3 gnarrow ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b16_s3_b16_gnarrow/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b16_s3_b16_gnarrow/residual.png

echo "=== s4 gwide ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.05 --gain_hi 4.0 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b16_s4_b16_gwide/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b16_s4_b16_gwide/residual.png

echo "=== s5 ctrl ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b16_s5_b16_ctrl/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b16_s5_b16_ctrl/residual.png

echo "=== DONE ==="
