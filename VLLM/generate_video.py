"""generate_video.py -- text -> video with LTX-Video (local, diffusers). Inverse of describe_video.py.

Reads prompts (either explicit --prompt, or the aggregate `video_descriptions.txt` produced by
describe_video.py) and renders one .mp4 per prompt. The LTX pipeline is loaded ONCE and reused.

  # from the description ledger (description + objects -> prompt), all records:
  python generate_video.py --descriptions /groups/.../graphs_data/video_descriptions.txt --out-dir ./ltx_out

  # one explicit prompt:
  python generate_video.py --prompt "orange particles migrate inward and aggregate into dense masses"

  # validate parsing/plumbing without loading the 20GB model:
  python generate_video.py --descriptions .../video_descriptions.txt --dry-run

NOTE: this renders PIXELS (a look-alike clip), not a mechanism. It is a reconstruction sanity-check,
not the Plexus world-model decoder. See describe_video.py's header and the project appendix.
"""
import os
import re
import sys
import glob
import argparse

import cv2          # NOTE: must precede torch/diffusers -- importing cv2 after them breaks libtiff/libjpeg
from PIL import Image

LTX = os.environ.get("LTX_DIR", "/workspace/Plexus/VLLM/LTX-Video")
DEV = "cuda:0"
DEFAULT_ROOT = os.environ.get(
    "PLEXUS_OUTPUT_ROOT", os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
) + "/graphs_data"
NEG = "blurry, distorted, low quality, jpeg artifacts, watermark, text"


def parse_ledger(path):
    """Parse the describe_video.py aggregate format into [{name, description, objects}].

    Records look like:
        - video name: <type>/<name>/<stem>
        - description: <one or more sentences, may span lines>
        - objects: <one or more sentences>
        <blank line>
    """
    txt = open(path).read()
    records = []
    # split on the "- video name:" marker, keep the field after it
    chunks = re.split(r"(?m)^- video name:\s*", txt)
    for ch in chunks[1:]:
        name = ch.splitlines()[0].strip()
        rest = ch[len(ch.splitlines()[0]):]
        d = re.search(r"- description:\s*(.*?)(?=^- objects:|\Z)", rest, re.S | re.M)
        o = re.search(r"- objects:\s*(.*?)(?=^- video name:|\Z)", rest, re.S | re.M)
        desc = (d.group(1).strip() if d else "")
        objs = (o.group(1).strip() if o else "")
        records.append({"name": name, "description": desc, "objects": objs})
    return records


def _prompt_of(rec):
    """Combine description + objects into one LTX prompt (LTX wants detail)."""
    parts = [rec.get("description", "").strip(), rec.get("objects", "").strip()]
    return " ".join(p for p in parts if p).replace("\n", " ").strip()


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def grab_frame(video, w, h, idx=0):
    """Frame `idx` of `video` (idx=-1 -> last) as a PIL image resized to w×h, or None."""
    if not video or not os.path.isfile(video):
        return None
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = (total - 1) if idx < 0 else idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, want))
    ok, fr = cap.read(); cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((w, h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None, help="single explicit prompt (overrides --descriptions)")
    ap.add_argument("--descriptions", default=None,
                    help="ledger from describe_video.py (default <root>/video_descriptions.txt)")
    ap.add_argument("--out-dir", default=None, help="where to write mp4s (default <ledger dir>/ltx_out)")
    ap.add_argument("--model", default=LTX)
    ap.add_argument("--width", type=int, default=704)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=97)        # LTX wants 8k+1
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap number of prompts (0 = all)")
    ap.add_argument("--condition", choices=["none", "first", "firstlast"], default="none",
                    help="'first': seed with the real first frame; 'firstlast': pin real first AND last frame")
    ap.add_argument("--init-image", default=None, help="explicit conditioning image for --prompt mode")
    ap.add_argument("--dry-run", action="store_true", help="parse + plan only; do not load the model")
    args = ap.parse_args()

    # ---- assemble the work list (prompt, out_name) ----
    if args.prompt:
        jobs = [{"name": "prompt", "prompt": args.prompt, "src": args.init_image}]
        out_dir = args.out_dir or os.path.join(os.getcwd(), "ltx_out")
    else:
        ledger = args.descriptions or os.path.join(DEFAULT_ROOT, "video_descriptions.txt")
        assert os.path.isfile(ledger), f"ledger not found: {ledger} (run describe_video.py first, or pass --prompt)"
        root = os.path.dirname(os.path.abspath(ledger))           # ledger sits at the graphs_data root
        recs = parse_ledger(ledger)
        # the real source mp4 = <root>/<name>.mp4 (name is its path relative to root, sans extension)
        jobs = [{"name": r["name"], "prompt": _prompt_of(r), "src": os.path.join(root, r["name"] + ".mp4")}
                for r in recs if _prompt_of(r)]
        out_dir = args.out_dir or os.path.join(root, "ltx_out")
    if args.limit:
        jobs = jobs[:args.limit]
    assert jobs, "no prompts to render"
    os.makedirs(out_dir, exist_ok=True)
    for j in jobs:
        j["out"] = os.path.join(out_dir, _safe(j["name"]) + "_ltx.mp4")

    print(f"[generate] {len(jobs)} prompt(s) -> {out_dir}", flush=True)
    for k, j in enumerate(jobs, 1):
        print(f"  [{k}] {_safe(j['name'])}: {j['prompt'][:140]}", flush=True)

    if args.dry_run:
        print("[dry-run] parsed + planned; model not loaded.", flush=True)
        return

    # ---- load LTX once; pipeline class depends on the conditioning mode ----
    import torch
    from diffusers import LTXPipeline, LTXImageToVideoPipeline, LTXConditionPipeline
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    from diffusers.utils import export_to_video
    assert os.path.isdir(args.model), (
        f"LTX weights not found: {args.model}\n"
        f"  download externally (HF is firewalled in-container):\n"
        f"  hf download Lightricks/LTX-Video --local-dir {args.model}")
    Pipe = {"none": LTXPipeline, "first": LTXImageToVideoPipeline,
            "firstlast": LTXConditionPipeline}[args.condition]
    print(f"[load] {Pipe.__name__} from {args.model} on {DEV} (condition={args.condition}) ...", flush=True)
    pipe = Pipe.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(DEV)
    pipe.vae.enable_tiling()                                  # keep VAE decode within memory
    if Pipe is LTXConditionPipeline:                          # this path uses the linear-quadratic schedule,
        pipe.scheduler.register_to_config(use_dynamic_shifting=False)   # which needs dynamic-shifting off (no mu)

    for k, j in enumerate(jobs, 1):
        kw, tag = {}, ""
        if args.condition == "first":
            img = grab_frame(j.get("src"), args.width, args.height, 0)
            if img is None:
                print(f"  (no source frame for {j['name']}; text-only)", flush=True)
            else:
                kw["image"], tag = img, " [first]"
        elif args.condition == "firstlast":
            f0 = grab_frame(j.get("src"), args.width, args.height, 0)
            fL = grab_frame(j.get("src"), args.width, args.height, -1)
            if f0 is None or fL is None:
                print(f"  (no source frames for {j['name']}; text-only)", flush=True)
            else:                                            # pin real endpoints; LTX fills the middle
                kw["conditions"] = [LTXVideoCondition(image=f0, frame_index=0),
                                    LTXVideoCondition(image=fL, frame_index=args.frames - 1)]
                tag = " [first+last]"
        print(f"[{k}/{len(jobs)}] rendering {os.path.basename(j['out'])}{tag} ...", flush=True)
        gen = torch.Generator(device=DEV).manual_seed(args.seed)
        frames = pipe(prompt=j["prompt"], negative_prompt=NEG, **kw,
                      width=args.width, height=args.height, num_frames=args.frames,
                      num_inference_steps=args.steps, max_sequence_length=256,  # else 128-tok prompts truncate
                      generator=gen).frames[0]
        export_to_video(frames, j["out"], fps=args.fps)
        print(f"  [saved] {j['out']}", flush=True)

    print(f"\n[done] {len(jobs)} video(s) in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
