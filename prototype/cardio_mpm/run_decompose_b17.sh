#!/bin/bash
# Residual decomposition for all B17 slots
set -e

echo "=== s0 fdev005 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --siren_fibre 1 --fibre_dev 0.05 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b17_s0_b17_fdev005/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b17_s0_b17_fdev005/residual.png

echo "=== s1 fdev015 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --siren_fibre 1 --fibre_dev 0.15 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b17_s1_b17_fdev015/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b17_s1_b17_fdev015/residual.png

echo "=== s2 deep4800 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --siren_fibre 1 --fibre_dev 0.1 \
  --learn fibre,gain,dur,stiff --n_iter 4800 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b17_s2_b17_deep4800/checkpoints/model_04100.pt \
  --eval_decompose archive/p3_b17_s2_b17_deep4800/residual.png

echo "=== s3 fdev01_gnarr ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_lo 0.2 --gain_hi 1.5 \
  --siren_fibre 1 --fibre_dev 0.1 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b17_s3_b17_fdev01_gnarr/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b17_s3_b17_fdev01_gnarr/residual.png

echo "=== s4 fdev01_wl35 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --siren_fibre 1 --fibre_dev 0.1 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 35 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b17_s4_b17_fdev01_wl35/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b17_s4_b17_fdev01_wl35/residual.png

echo "=== s5 ctrl_fdev01 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --siren_fibre 1 --fibre_dev 0.1 \
  --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b17_s5_b17_ctrl_fdev01/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b17_s5_b17_ctrl_fdev01/residual.png

echo "=== DONE ==="
