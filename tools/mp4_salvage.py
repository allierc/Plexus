#!/usr/bin/env python
"""Make a STILL-BEING-WRITTEN mp4 readable, without touching the run that is writing it.

THE PROBLEM. A plain mp4 puts its index -- the `moov` atom, which says where every frame is and
what codec decodes them -- at the END of the file, because the muxer does not know the frame offsets
until it has written them. A run that is still going has therefore written `ftyp` + `mdat` and
nothing else, and every player refuses it. Copying the file mid-run does not help: a prefix of an
unreadable file is unreadable.

    @           0  ftyp  size 32
    @          32  free  size 8
    @          40  mdat  size 135528456     <- 135 MB of perfectly good H.264, no index

THE FIX FOR RUNS THAT HAVE NOT STARTED YET is `-movflags frag_keyframe+empty_moov` plus
`-flush_packets 1`, which is what `live_movie.py` now passes. This tool is for the run that is
ALREADY four hours in and was started before that.

WHAT IT DOES. The payload of `mdat` is the H.264 access units in AVCC framing: a 4-byte big-endian
length, then that many bytes of NAL. That is fully walkable without the index -- each length says
where the next one starts, and a walk that lands exactly on EOF proves the framing was read right.
Converting each NAL to Annex-B (replace the length with the start code 00 00 00 01) gives a raw
elementary stream.

ONE THING IS GENUINELY MISSING and has to be reconstructed: the SPS and PPS. For mp4, x264 puts them
in the `avcC` box inside `moov` (`global_header`) and NOT in the stream, so the salvaged NALs are
slices with no parameter sets and nothing can decode them. They are recovered by encoding a throwaway
clip with the SAME BINARY AT THE SAME SETTINGS -- resolution, pixel format, rate, crf -- and taking
its inline parameter sets, which are a function of those settings and not of the picture content.
The tool reads those settings off the live ffmpeg's own command line (/proc/<pid>/cmdline) rather
than being told them, so they cannot drift from what is actually being written.

WHAT YOU GET, and what you do not. Every complete frame up to the last flushed one, in order, at the
original quality -- it is a remux, not a re-encode. Timestamps are synthesised from the declared
frame rate, so if the writer ever stalled the salvaged clip will not show the stall. The last partial
NAL is dropped.

    python tools/mp4_salvage.py graphs_data/si_material/si_bench_100m_fast/movie.mp4
    python tools/mp4_salvage.py <in.mp4> -o /tmp/watch.mp4      # explicit destination
"""
from __future__ import annotations

import argparse
import glob
import os
import struct
import subprocess
import sys
import tempfile


def _boxes(path):
    """Top-level atoms as (offset, type, size, header_len). `mdat` may run to EOF."""
    sz = os.path.getsize(path)
    out, off = [], 0
    with open(path, "rb") as f:
        while off < sz:
            f.seek(off)
            h = f.read(16)
            if len(h) < 8:
                break
            n = struct.unpack(">I", h[:4])[0]
            typ = h[4:8].decode("latin1", "replace")
            hdr = 8
            if n == 1:
                n, hdr = struct.unpack(">Q", h[8:16])[0], 16
            elif n == 0:
                n = sz - off                      # "to end of file", what a live muxer writes
            out.append((off, typ, n, hdr))
            off += n
    return out


def _ffmpeg():
    """The imageio-ffmpeg binary the engine itself uses, so the encoder version matches."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    for c in glob.glob("/workspace/.conda_envs/*/lib/python3*/site-packages/imageio_ffmpeg/"
                       "binaries/ffmpeg-linux-x86_64*"):
        return c
    return "ffmpeg"


def _writer_args(target):
    """The live writer's own settings, read off its command line. None if it is not running."""
    real = os.path.realpath(target)
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            argv = open(p, "rb").read().split(b"\0")
        except OSError:
            continue
        argv = [a.decode("utf8", "replace") for a in argv if a]
        if not argv or "ffmpeg" not in os.path.basename(argv[0]):
            continue
        if not any(os.path.realpath(a) == real for a in argv if a.endswith(".mp4")):
            continue
        g = lambda k, d: argv[argv.index(k) + 1] if k in argv else d          # noqa: E731
        return dict(pid=int(p.split("/")[2]), size=g("-s", "1280x1280"), rate=g("-r", "30"),
                    pix=g("-pix_fmt", "yuv420p"), crf=g("-crf", "23"),
                    vcodec=[a for a in argv if a in ("libx264", "libx265")] or ["libx264"])
    return None


def _avcc_parameter_sets(path):
    """SPS/PPS lifted from a COMPLETE mp4's `avcC` box, as Annex-B. None if there is none.

    THIS IS THE RELIABLE SOURCE, and re-encoding a throwaway clip is not. A synthesised parameter
    set has to match the real one in every field the decoder uses to size its buffers, and it does
    not: `-frames:v 2` makes x264 shrink `max_num_ref_frames` to fit a 2-frame GOP, so the salvaged
    stream's P-frames reference pictures the decoder never allocated. Measured -- the container
    parsed, ffmpeg reported 271 frames, and every one of them decoded to grey and green with
    `Reference 10 >= 2` on almost every macroblock row. A frame count is not a picture.

    A finished movie from the same writer at the same resolution has the exact bytes instead.
    """
    sz = os.path.getsize(path)
    found = []

    def walk(off, end, depth=0):
        while off < end - 8 and depth < 8:
            with open(path, "rb") as f:
                f.seek(off)
                h = f.read(8)
                if len(h) < 8:
                    return
                n = struct.unpack(">I", h[:4])[0]
                typ = h[4:8]
                hdr = 8
                if n == 1:
                    f.seek(off + 8)
                    n, hdr = struct.unpack(">Q", f.read(8))[0], 16
                elif n == 0:
                    n = end - off
                if n < 8:
                    return
                if typ == b"avcC":
                    f.seek(off + hdr)
                    found.append(f.read(n - hdr))
                    return
            if typ in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
                walk(off + hdr, off + n, depth + 1)
            elif typ == b"stsd":
                walk(off + hdr + 8, off + n, depth + 1)      # version/flags + entry_count
            elif typ in (b"avc1", b"encv"):
                walk(off + hdr + 78, off + n, depth + 1)     # VisualSampleEntry fixed part
            off += n

    walk(0, sz)
    if not found:
        return None
    a = found[0]
    # avcC: configVersion, profile, compat, level, 0b111111 + lengthSizeMinusOne,
    #       0b111 + numSPS, [len16 + SPS]*, numPPS, [len16 + PPS]*
    out, i = bytearray(), 5
    n_sps = a[i] & 0x1F
    i += 1
    for _ in range(n_sps):
        L = struct.unpack(">H", a[i:i + 2])[0]
        out += b"\x00\x00\x00\x01" + a[i + 2:i + 2 + L]
        i += 2 + L
    n_pps = a[i]
    i += 1
    for _ in range(n_pps):
        L = struct.unpack(">H", a[i:i + 2])[0]
        out += b"\x00\x00\x00\x01" + a[i + 2:i + 2 + L]
        i += 2 + L
    return bytes(out)


def _find_reference(src):
    """A finished movie.mp4 from the same writer, to borrow parameter sets from."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(src)))
    cands = sorted(glob.glob(os.path.join(root, "*", "movie.mp4")), key=os.path.getsize)
    for c in cands:
        if os.path.abspath(c) == os.path.abspath(src):
            continue
        if any(t == "moov" for _, t, _, _ in _boxes(c)):        # complete: it has its index
            ps = _avcc_parameter_sets(c)
            if ps:
                return c, ps
    return None, None


def _parameter_sets(ff, size, rate, pix, crf, vcodec):
    """SPS/PPS for these encoder settings, from a throwaway 2-frame encode with inline headers."""
    w, h = (int(x) for x in size.split("x"))
    out = tempfile.NamedTemporaryFile(suffix=".h264", delete=False).name
    cmd = [ff, "-y", "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={rate}",
           "-frames:v", "2", "-vcodec", vcodec, "-pix_fmt", pix, "-crf", str(crf),
           "-bsf:v", "dump_extra", "-f", "h264", "-v", "error", out]
    subprocess.run(cmd, check=True)
    raw = open(out, "rb").read()
    os.unlink(out)
    # keep only the leading parameter sets: everything before the first slice NAL (type 1 or 5)
    keep, i = bytearray(), 0
    starts = []
    while True:
        j = raw.find(b"\x00\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j)
        i = j + 4
    for k, j in enumerate(starts):
        t = raw[j + 4] & 0x1F
        if t in (1, 5):
            break
        end = starts[k + 1] if k + 1 < len(starts) else len(raw)
        keep += raw[j:end]
    return bytes(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    # THE DEFAULT DESTINATION IS /tmp, NOT THE RUN FOLDER. This tool reads a file another process
    # is appending to; it must not put anything beside it that a later pass could mistake for the
    # run's own output, and the run folder is a bind mount shared with the cluster. Nothing here
    # ever opens the source for writing.
    ap.add_argument("-o", "--out", default=None,
                    help="default: /tmp/<run name>_partial.mp4")
    ap.add_argument("--rate", default=None, help="override the frame rate")
    ap.add_argument("--no-reference", action="store_true",
                    help="do not borrow SPS/PPS from a finished movie; synthesise them")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    out = a.out or os.path.join("/tmp", os.path.basename(os.path.dirname(src)) + "_partial.mp4")
    ff = _ffmpeg()

    bx = _boxes(src)
    print(f"\n  {src}")
    for off, typ, n, hdr in bx:
        print(f"    @{off:>12}  {typ}  {n / 1e6:>9.2f} MB")
    if any(t == "moov" for _, t, _, _ in bx):
        print("\n  this file already has its index -- it is complete, just play it\n")
        return
    mdat = next((b for b in bx if b[1] == "mdat"), None)
    if mdat is None:
        sys.exit("  no mdat: nothing has been written yet")

    live = _writer_args(src)
    if live:
        print(f"\n  live writer pid {live['pid']}: {live['size']} @ {live['rate']} fps, "
              f"{live['vcodec'][0]} crf {live['crf']}")
    cfg = live or dict(size="1280x1280", rate="30", pix="yuv420p", crf="23", vcodec=["libx264"])
    rate = a.rate or cfg["rate"]

    # 1) walk the AVCC NALs and re-frame them as Annex-B
    off, end = mdat[0] + mdat[3], mdat[0] + mdat[2]
    es = tempfile.NamedTemporaryFile(suffix=".h264", delete=False)
    ref, ps = (None, None) if a.no_reference else _find_reference(src)
    if ps:
        print(f"  parameter sets from {ref} ({len(ps)} B of SPS/PPS)")
    else:
        print("  no finished movie to borrow SPS/PPS from -- SYNTHESISING them, which is only "
              "right if the encoder settings happen to match; check the picture, not the count")
        ps = _parameter_sets(ff, cfg["size"], rate, cfg["pix"], cfg["crf"], cfg["vcodec"][0])
    es.write(ps)
    n_nal, n_frames, truncated = 0, 0, False
    with open(src, "rb") as f:
        f.seek(off)
        while off < end:
            b = f.read(4)
            if len(b) < 4:
                truncated = True
                break
            L = struct.unpack(">I", b)[0]
            if L == 0 or off + 4 + L > end:
                truncated = True                 # the tail the muxer had not finished writing
                break
            nal = f.read(L)
            if len(nal) < L:
                truncated = True
                break
            es.write(b"\x00\x00\x00\x01" + nal)
            n_nal += 1
            if (nal[0] & 0x1F) in (1, 5):
                n_frames += 1
            off += 4 + L
    es.close()
    print(f"  walked {n_nal} NALs, {n_frames} coded frames"
          f"{', last partial NAL dropped' if truncated else ', clean to the byte'}")
    if n_frames == 0:
        os.unlink(es.name)
        sys.exit("  no complete frame yet")

    # 2) remux (no re-encode) into a normal mp4 with an index
    r = subprocess.run([ff, "-y", "-r", str(rate), "-f", "h264", "-i", es.name,
                        "-c", "copy", "-movflags", "+faststart", "-v", "error", out],
                       capture_output=True, text=True)
    os.unlink(es.name)
    if r.returncode != 0:
        sys.exit(f"  remux failed:\n{r.stderr}")

    # 3) PROVE IT, by decoding. A remux that returns 0 has only proved the container parses; the
    # claim is that the PICTURES come back, so the check is a full decode to /dev/null and the
    # frame count ffmpeg reports for it. imageio-ffmpeg ships no ffprobe, so this uses the same
    # ffmpeg binary rather than assuming a sibling that is not there.
    q = subprocess.run([ff, "-v", "error", "-stats", "-i", out, "-f", "null", "-"],
                       capture_output=True, text=True)
    tail = [ln for ln in (q.stderr or "").strip().splitlines() if "frame=" in ln]
    print(f"  -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"     decodes: {tail[-1].strip() if tail else '(no frame report)'}"
          f"{'' if q.returncode == 0 else '   DECODE ERRORS: ' + q.stderr.strip()[:200]}")
    print()


if __name__ == "__main__":
    main()
