"""describe_video.py -- caption Plexus simulation .mp4s with Gemma-4-12B (local, multimodal).

For each video it extracts evenly-spaced keyframes and asks Gemma-4 for two labeled
sections: a four-sentence DESCRIPTION of the video and a four-sentence OBJECTS
description of the sets in the scene (topology, colour, size). Results are appended
to ONE aggregate text file in the format:

    - video name: <type>/<name>/<stem>
    - description: ...
    - objects: ...

Usage:
    python describe_video.py --root DIR [--out FILE] [--frames N]   # all *.mp4 under DIR
    python describe_video.py VIDEO.mp4 [VIDEO2.mp4 ...] [--out FILE] # explicit videos

Default root is the shared graphs_data tree; default out is <root>/video_descriptions.txt.
The model is loaded ONCE and reused across all videos.
"""
import os
import sys
import glob
import json
import argparse

import cv2
import yaml
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM

_SPEC_MARKER = "# --- auto: video descriptions (gemma-4-12B) ---"

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEMMA = os.environ.get("GEMMA_DIR", "/workspace/Plexus/VLLM/gemma-4-12B-it")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_ROOT = os.environ.get(
    "PLEXUS_OUTPUT_ROOT", os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
) + "/graphs_data"

PROMPT = (
    "These are time-ordered keyframes from a scientific simulation video (particles, "
    "agents, or continuous fields evolving over time). Write your answer in exactly two "
    "labeled sections and nothing else.\n"
    "DESCRIPTION: exactly four sentences describing what happens in the video over time.\n"
    "OBJECTS: exactly four sentences describing the sets of objects in the scene -- their "
    "topology (how they are arranged or connected), their colour, and their size."
)


def keyframes(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return []
    idxs = [int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def _split_sections(text):
    """Split the model output into (description, objects) on the OBJECTS: label."""
    t = text.strip()
    low = t.lower()
    i = low.find("objects:")
    if i == -1:
        return t, ""
    desc = t[:i]
    objs = t[i + len("objects:"):]
    j = desc.lower().find("description:")
    if j != -1:
        desc = desc[j + len("description:"):]
    return desc.strip(), objs.strip()


def describe_one(proc, model, video, n_frames):
    frames = keyframes(video, n_frames)
    if not frames:
        return None
    content = [{"type": "image", "image": f} for f in frames] + [{"type": "text", "text": PROMPT}]
    msgs = [{"role": "system", "content": "You are a precise scientific assistant."},
            {"role": "user", "content": content}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt",
                                      enable_thinking=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    resp = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    parsed = (proc.parse_response(resp) if hasattr(proc, "parse_response") else resp)
    if isinstance(parsed, dict):
        parsed = parsed.get("content", parsed)
    return str(parsed).strip()


def _write_into_spec(spec_path, desc_map):
    """Append a `descriptions:` block (one entry per movie stem) to a data-folder
    spec.yaml. Idempotent: replaces any prior auto-block (below the marker), leaving
    the original spec text above it untouched."""
    try:
        text = open(spec_path).read()
    except FileNotFoundError:
        return False
    i = text.find(_SPEC_MARKER)
    if i != -1:
        text = text[:i].rstrip() + "\n"
    block = yaml.safe_dump({"descriptions": desc_map}, sort_keys=False,
                           default_flow_style=False, width=100, allow_unicode=True)
    open(spec_path, "w").write(text.rstrip() + "\n\n" + _SPEC_MARKER + "\n" + block)
    return True


def _video_name(path, root):
    """A meaningful id: <type>/<name>/<stem> relative to the graphs_data root."""
    ap = os.path.abspath(path)
    if root and ap.startswith(os.path.abspath(root)):
        rel = os.path.relpath(ap, os.path.abspath(root))
        return os.path.splitext(rel)[0]
    return os.path.splitext(os.path.basename(ap))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="explicit video paths (else use --root)")
    ap.add_argument("--root", default=None, help="recurse for *.mp4 under this dir")
    ap.add_argument("--out", default=None, help="aggregate text file (default <root>/video_descriptions.txt)")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--append", action="store_true", help="append to --out instead of truncating it")
    ap.add_argument("--config-root", default=os.path.join(_REPO, "config"),
                    help="also write each description into the repo spec config/<type>/<name>.yaml")
    args = ap.parse_args()

    if args.videos:
        videos = sorted(args.videos)
        root = args.root or os.path.commonpath([os.path.dirname(os.path.abspath(v)) for v in videos])
    else:
        root = args.root or DEFAULT_ROOT
        videos = sorted(glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True))
    out_file = args.out or os.path.join(root, "video_descriptions.txt")
    assert videos, f"no videos found (root={root})"
    assert os.path.isdir(GEMMA), f"gemma weights not found: {GEMMA}"
    print(f"[describe] {len(videos)} videos -> {out_file}", flush=True)

    print(f"[load] {GEMMA} on {DEV} ...", flush=True)
    proc = AutoProcessor.from_pretrained(GEMMA)
    model = AutoModelForMultimodalLM.from_pretrained(GEMMA, dtype="bfloat16", device_map=DEV)

    records = []
    spec_updates: dict[str, dict] = {}                             # spec.yaml -> {movie_stem: {...}}
    if not args.append:
        open(out_file, "w").close()                                # fresh aggregate
    for k, video in enumerate(videos, 1):
        name = _video_name(video, root)
        print(f"[{k}/{len(videos)}] {name}", flush=True)
        text = describe_one(proc, model, video, args.frames)
        if text is None:
            print(f"  (skip: unreadable) {video}", flush=True); continue
        desc, objs = _split_sections(text)
        entry = f"- video name: {name}\n- description: {desc}\n- objects: {objs}\n\n"
        with open(out_file, "a") as f:                             # append incrementally (crash-safe)
            f.write(entry)
        records.append({"video": video, "name": name, "description": desc, "objects": objs})
        # store the description IN the spec.yaml next to the movie AND in the repo
        # config (config/<type>/<name>.yaml), derived from the graphs_data layout.
        ddir = os.path.dirname(os.path.abspath(video))
        stem = os.path.splitext(os.path.basename(video))[0]
        typ, nm = os.path.basename(os.path.dirname(ddir)), os.path.basename(ddir)
        for tgt in (os.path.join(ddir, "spec.yaml"),
                    os.path.join(args.config_root, typ, nm + ".yaml")):
            if os.path.exists(tgt):
                spec_updates.setdefault(tgt, {})[stem] = {"description": desc, "objects": objs}
        print("  " + desc.replace("\n", " ")[:160], flush=True)

    for sp, dm in spec_updates.items():                            # write one block per spec
        if _write_into_spec(sp, dm):
            print(f"[spec] {len(dm)} description(s) -> {sp}", flush=True)

    jpath = os.path.splitext(out_file)[0] + ".json"
    prior = []
    if args.append and os.path.exists(jpath):
        try:
            prior = json.load(open(jpath)).get("records", [])
        except Exception:
            prior = []
    json.dump({"model": "gemma-4-12B-it", "root": root, "records": prior + records},
              open(jpath, "w"), indent=2)
    print(f"\n[saved] {out_file}  ({len(records)} videos)", flush=True)


if __name__ == "__main__":
    main()
