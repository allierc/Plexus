#!/bin/bash
# render_progress -- every movie of the campaign, from ONE fit, at three stages of its training.
#
#   three beats          1 and 2 held out, 3 the fit beat
#   three stages         20% and 50% of the run (fit.py --checkpoints) and the finished fit
#   three kinds          overlay on the raw frames (green recording / magenta model, x4 motion),
#                        overlay of the points (green tracking nodes / blue MPM particles),
#                        the four-panel cell view (per-cell strain maps, arrows, mean curve)
#
# 27 files named progress_<kind>_p<stage>_beat<n>.mp4, so they sort by kind then stage then beat.
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
export PYTHONPATH=/workspace/Plexus/src
cd "$(dirname "$0")"
FIT=${1:-out/fits/s4_live_r16_ckpt}
A="--per-parent 120 --drag 150 --anchor-percell"
render() {   # $1 device, $2 params file, $3 stage label, $4 beat
  $PY movie_image.py     --overlay --beat "$4" --device "$1" $A --params "$FIT/$2" --tag "p$3_beat$4" --amplify 4
  $PY movie_particles.py --overlay --beat "$4" --device "$1" $A --params "$FIT/$2" --tag "p$3_beat$4"
  $PY movie.py                     --beat "$4" --device "$1" $A --params "$FIT/$2" --tag "p$3_beat$4"
}
( for b in 1 2 3; do render cuda:0 params_p20.npz 020 $b; render cuda:0 params.npz 100 $b; done ) > out/render_gpu0.log 2>&1 &
( for b in 1 2 3; do render cuda:1 params_p50.npz 050 $b; done ) > out/render_gpu1.log 2>&1 &
wait
ls out/movies/progress_*.mp4 | wc -l
