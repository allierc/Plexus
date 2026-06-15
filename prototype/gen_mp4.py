"""Convert every gif in prototype/ to a small mp4 (30 fps): ~10 s, or ~20 s for the
maze runs (initial, winner_*). Speedup via setpts; libx264, well under 8 MB."""
import os, subprocess, imageio_ffmpeg
from PIL import Image
P = os.path.dirname(os.path.abspath(__file__)) + "/"
FF = imageio_ffmpeg.get_ffmpeg_exe()
for g in sorted(f for f in os.listdir(P) if f.endswith(".gif")):
    name = g[:-4]
    im = Image.open(P + g)
    orig = im.n_frames * (im.info.get("duration", 50) / 1000.0)   # seconds
    target = 20.0 if (name == "initial" or name.startswith("winner_")) else 10.0
    factor = target / max(orig, 0.1)
    crf = "26" if name in ("sim_clusters", "sim_5") else "23"      # dense ones compress less
    out = P + name + ".mp4"
    subprocess.run([FF, "-y", "-i", P + g, "-vf",
                    f"setpts={factor}*PTS,scale=trunc(iw/2)*2:-2", "-r", "30",
                    "-c:v", "libx264", "-preset", "fast", "-crf", crf,
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"{g:24s} {orig:5.1f}s -> {target:4.0f}s  {os.path.getsize(out)/1e6:4.1f} MB", flush=True)
print("ALL DONE", flush=True)
