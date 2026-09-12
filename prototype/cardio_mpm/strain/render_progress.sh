#!/bin/bash
# render_progress -- the two movies of a finished fit, each ONE rollout over three beats.
#
#   cont_cells.mp4              the four-panel cell view: per-cell strain map recorded and modelled,
#                               the deformation arrows, and the mean curve
#   cont_overlay_particles.mp4  two overlays side by side -- the raw frames (green recording, magenta
#                               model, motion x4) and the points (green tracking nodes, blue particles)
#
# `--continuous` makes the window span beats 1-3 and the fitted clock fire once per beat, so the
# model runs through all three WITHOUT being reset: what is drawn at beat 3 is the state the model
# reached by integrating from rest through beats 1 and 2. Both clips are 8 s.
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
export PYTHONPATH=/workspace/Plexus/src
cd "$(dirname "$0")"
FIT=${1:-out/fits/healthy_allbeats}/params.npz
A="--per-parent 120 --drag 150 --anchor-percell"
$PY movie.py           --continuous --device cuda:0 $A --params "$FIT" --tag cont --seconds 8
$PY movie_image.py     --continuous --overlay --device cuda:0 $A --params "$FIT" --tag cont --amplify 4 --seconds 8
$PY movie_particles.py --continuous --overlay --device cuda:1 $A --params "$FIT" --tag cont --seconds 8
FF=$($PY -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
cd out/movies
$FF -loglevel error -y -i progress_overlay_cont.mp4 -i progress_particles_cont.mp4 -filter_complex \
  "[0:v]scale=-2:1000[a];[1:v]scale=-2:1000[b];[a][b]hstack=inputs=2[v]" -map "[v]" -r 20 \
  -c:v libx264 -crf 20 -pix_fmt yuv420p cont_overlay_particles.mp4
mv -f progress_cells_cont.mp4 cont_cells.mp4
rm -f progress_overlay_cont.mp4 progress_particles_cont.mp4 progress_*.json
ls -la *.mp4
