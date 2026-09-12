#!/bin/bash
# render_progress -- every movie of the campaign, from ONE fit, at three stages of its training.
#
#   three beats          1 and 2 held out, 3 the fit beat
#   three stages         20% and 50% of the run (fit.py --checkpoints) and the finished fit
#   three kinds          overlay on the raw frames (green recording / magenta model, x4 motion),
#                        overlay of the points (green tracking nodes / blue MPM particles),
#                        the four-panel cell view (per-cell strain maps, arrows, mean curve)
#
# THREE files, all of the finished fit (STAGES below adds part-trained ones):
#   progress_overlay_p100.mp4          the raw frames in green, the model-warped rest frame in magenta
#   progress_particles_p100.mp4        tracking nodes in green, MPM particles in blue
#   progress_cells_particles_p100.mp4  the four-panel cell view beside the particles, 8 s
# Each clip runs the three beats one after another (1 and 2 held out, then the fitted 3). The
# per-beat clips are rendered first and concatenated, because each beat is its own rollout from its
# own rest configuration -- the model is never run across a beat boundary.
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
export PYTHONPATH=/workspace/Plexus/src
cd "$(dirname "$0")"
FIT=${1:-out/fits/s4_live_r16_ckpt}
A="--per-parent 120 --drag 150 --anchor-percell"
render() {   # $1 device, $2 params file, $3 stage label, $4 beat
  $PY movie_image.py     --overlay --beat "$4" --device "$1" $A --params "$FIT/$2" --tag "p$3_beat$4" --amplify 4 --seconds 4
  $PY movie_particles.py --overlay --beat "$4" --device "$1" $A --params "$FIT/$2" --tag "p$3_beat$4" --seconds 4
  $PY movie.py                     --beat "$4" --device "$1" $A --params "$FIT/$2" --tag "p$3_beat$4" --seconds 4
}
# STAGES: which checkpoints to render. "100" is the finished fit; "020 050 100" shows the training.
STAGES=${STAGES:-100}
declare -A CKPT=( [020]=params_p20.npz [050]=params_p50.npz [100]=params.npz )
i=0
for st in $STAGES; do
  ( for b in 1 2 3; do render "cuda:$((i % 2))" "${CKPT[$st]}" "$st" "$b"; done ) > "out/render_$st.log" 2>&1 &
  i=$((i + 1))
done
wait

# ---- the three beats, end to end, 4 s per clip ------------------------------------------------
FF=$($PY -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
cd out/movies
for view in overlay particles cells; do for st in $STAGES; do
  list=$(mktemp); for b in 1 2 3; do echo "file '$PWD/progress_${view}_p${st}_beat${b}.mp4'" >> "$list"; done
  n=$(for b in 1 2 3; do $FF -i progress_${view}_p${st}_beat${b}.mp4 2>&1 | grep -oP 'Duration: \K[0-9:.]+' \
        | awk -F: '{print ($3)*14}'; done | awk '{s+=$1} END {printf "%d", s}')
  r=$(python3 -c "print(max(1,round($n/4)))")
  $FF -loglevel error -y -f concat -safe 0 -i "$list" -vf "setpts=PTS*14/$r" -r $r -c:v libx264 \
      -crf 20 -pix_fmt yuv420p "progress_${view}_p${st}.mp4"
  rm -f "$list"
done; done

# ---- the cell view beside the particles, one 8 s clip ------------------------------------------
for st in $STAGES; do
  $FF -loglevel error -y -i "progress_cells_p${st}.mp4" -i "progress_particles_p${st}.mp4" -filter_complex \
    "[0:v]scale=-2:1180,setpts=PTS*2[a];[1:v]scale=-2:1180,setpts=PTS*2[b];[a][b]hstack=inputs=2[v]" \
    -map "[v]" -r 21 -c:v libx264 -crf 20 -pix_fmt yuv420p "progress_cells_particles_p${st}.mp4"
done
rm -f progress_*_beat[123].mp4 progress_*_beat[123].json      # the per-beat clips were scaffolding
ls progress_*.mp4
