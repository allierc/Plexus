"""Register, refresh, or diff a working point's fingerprint.

    python tools/fingerprint.py register <archive_folder> [--cut 150]
        read spec.yaml + trajectory.npz of an ACCEPTED archive, write
        tests/regression/fingerprints/<name>.json. The archive's own frame_ms becomes the timing
        reference (device "archive": reported, not asserted, until a same-GPU refresh).

    python tools/fingerprint.py refresh <name> --because "..." [--device cuda:0]
        rerun the cut on the CURRENT code and overwrite the JSON. The commit carrying the refreshed
        JSON must carry no source change: a source commit that needs a refresh to go green is a
        working-point change and is reviewed as one.

    python tools/fingerprint.py diff <name> [--device cuda:0]
        rerun the cut and print the comparison without writing anything.

    python tools/fingerprint.py list
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression_lib as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("register"); a.add_argument("folder"); a.add_argument("--cut", type=int, default=150)
    a.add_argument("--because", default="registered from the accepted archive")
    b = sub.add_parser("refresh"); b.add_argument("name"); b.add_argument("--because", required=True)
    b.add_argument("--device", default="cuda:0")
    c = sub.add_parser("diff"); c.add_argument("name"); c.add_argument("--device", default="cuda:0")
    sub.add_parser("list")
    args = ap.parse_args()

    if args.cmd == "register":
        fp = R.fingerprint_archive(args.folder, args.cut)
        fp["because"] = args.because
        if not R.moves(fp):
            sys.exit(f"{fp['name']}: nothing changes over the first {fp['cut']['n_frames']} frames "
                     f"(cells, radius, deaths, spread all flat); lengthen --cut or this is not a test")
        p = R.write_fingerprint(fp)
        print(f"registered {fp['name']} -> {os.path.relpath(p, R.ROOT)}  "
              f"cut {fp['cut']['n_frames']} frames, checkpoints {fp['cut']['checkpoints']}")
        for fr in fp["cut"]["checkpoints"]:
            cp = fp["checkpoints"][str(fr)]
            print(f"   f{fr}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in cp.items() if not isinstance(v, list)))
        if "timing" in fp:
            print(f"   timing {fp['timing']['ms_per_frame']:.1f} ms/frame ({fp['timing']['device']})")
        return

    if args.cmd == "list":
        for n in R.registered():
            fp = R.read_fingerprint(n)
            print(f"{n:28s} {fp['family']:10s} cut {fp['cut']['n_frames']:4d}  archived {fp['archived_on']}  "
                  f"commit {fp['generated_at_commit']}  {fp.get('because','')[:60]}")
        return

    ref = R.read_fingerprint(args.name)
    spec = R.spec_of(ref) if hasattr(R, "spec_of") else None
    if spec is None:
        import yaml
        if ref.get("archive") and os.path.exists(os.path.join(ref["archive"], "spec.yaml")):
            spec = yaml.safe_load(open(os.path.join(ref["archive"], "spec.yaml")))
        else:
            spec = yaml.safe_load(open(os.path.join(R.ROOT, "config", ref["family"], ref["name"] + ".yaml")))
    if args.cmd == "refresh":
        fp = R.fingerprint_fresh(spec, ref["family"], ref["cut"]["n_frames"], args.device, args.because,
                                 stride=int(ref["cut"].get("record_stride", 1)))
        fp["archive"] = ref.get("archive")
        p = R.write_fingerprint(fp)
        print(f"refreshed {fp['name']} at {fp['generated_at_commit']} on {fp['device']} -> {os.path.relpath(p, R.ROOT)}")
        return
    got = R.fingerprint_fresh(spec, ref["family"], ref["cut"]["n_frames"], args.device, "diff",
                              checkpoints=ref["cut"]["checkpoints"], stride=int(ref["cut"].get("record_stride", 1)))
    bad = R.compare(ref, got)
    tv, note = R.compare_timing(ref, got)
    print(f"{args.name}: {'PASS' if not bad and tv is None else 'DIFFER'}   {note}")
    for line in bad + ([tv] if tv else []):
        print("   " + line)
    sys.exit(1 if bad or tv else 0)


if __name__ == "__main__":
    main()
