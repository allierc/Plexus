#!/bin/bash
# Residual decomposition for all B15 slots
set -e

echo "=== s0 deep3600 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 3600 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b15_s0_b15_deep3600/checkpoints/model_03599.pt \
  --eval_decompose archive/p3_b15_s0_b15_deep3600/residual.png

echo "=== s1 deep4800 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 4800 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b15_s1_b15_deep4800/checkpoints/model_03950.pt \
  --eval_decompose archive/p3_b15_s1_b15_deep4800/residual.png

echo "=== s2 amp11 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 2400 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 11 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b15_s2_b15_amp11/checkpoints/model_02399.pt \
  --eval_decompose archive/p3_b15_s2_b15_amp11/residual.png

echo "=== s3 gomega3 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --gain_omega 3 --learn fibre,gain,dur,stiff --n_iter 2400 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b15_s3_b15_gomega3/checkpoints/model_02399.pt \
  --eval_decompose archive/p3_b15_s3_b15_gomega3/residual.png

echo "=== s4 durhi13 ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --gain_src siren --learn fibre,gain,dur,stiff --n_iter 2400 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 13 \
  --resume archive/p3_b15_s4_b15_durhi13/checkpoints/model_02399.pt \
  --eval_decompose archive/p3_b15_s4_b15_durhi13/residual.png

echo "=== s5 ctrl_nosgain ==="
python cardio_mpm_train.py material/material_aniso_cardio \
  --gain0 0.5 --learn fibre,gain,dur,stiff --n_iter 2400 \
  --fibre_wl 28.8 --fibre_angle 0.17 --fibre_amp 0.39 --fibre_phase 0.41 \
  --stiff_src siren --siren_omega 5 --stiff_lo 80 --stiff_hi 300 \
  --amplitude 10 --drag_k 30 --dur0 10 --dur_hi 11 \
  --resume archive/p3_b15_s5_b15_ctrl_nosgain/checkpoints/model_02399.pt \
  --eval_decompose archive/p3_b15_s5_b15_ctrl_nosgain/residual.png

echo "=== DONE ==="
