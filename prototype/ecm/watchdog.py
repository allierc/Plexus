"""Poll the cluster every 30 minutes, launch the next wave when the current one drains, log what lands.

WHY A WATCHDOG AND NOT A CHAIN. `run_batch` waits on the queue and returns as soon as any job exits,
which reports a wave as done while most of it is still running. And a job can finish, write its
artefacts, and still leave no result -- that happened to all seven v2 runs, whose stdout LSF never
flushed, taking the SERIES line and the pass1.json update with it. So this checks the FOLDERS, which
are the thing that actually exists, and recovers the numbers from the trajectory when the log is empty.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/workspace/Plexus/discovery_okuda")
sys.path.insert(0, HERE)
import cluster  # noqa: E402

LOG = "/workspace/Plexus/log/okuda_ECM"
WAVES = [
    ["84_holes_45k_r0", "85_holes_45k_r8", "86_holes_135k_r0",
     "87_holes_135k_r8", "88_holes_270k_r0", "89_holes_270k_r8"],
    ["90_holes_45k_r0_noadh", "91_holes_45k_r8_noadh", "92_holes_135k_r0_noadh",
     "93_holes_135k_r8_noadh", "94_holes_270k_r0_noadh", "95_holes_270k_r8_noadh"],
]
PERIOD = int(os.environ.get("WD_PERIOD", 1800))


def running():
    # PLAIN COMMAND, PARSED HERE. The first version piped through awk with a $7 that had to survive
    # python quoting, a shell, ssh and a remote shell; `_ssh` swallowed the resulting error and returned
    # None, which the caller could only report as "ssh unavailable" -- indistinguishable from the network
    # being down. Fewer layers, and the parsing is somewhere it can be read.
    # RETRY, and a longer timeout. login1 hung for minutes at a stretch today -- a direct `bjobs -w`
    # did not return in 300s -- while the jobs themselves were fine, because they run on compute nodes
    # and do not need the login node. A watchdog that gives up on the first slow poll is useless exactly
    # when the cluster is busy enough to be worth watching.
    out = cluster._ssh_retry("bjobs -w", timeout=90, tries=3)
    if out is None or out.returncode != 0:
        return None                              # ssh trouble: say so rather than guess zero
    names = []
    for line in out.stdout.splitlines()[1:]:
        f = line.split()
        if len(f) >= 7 and not f[6].startswith("pg_"):
            names.append(f[6])
    return names


def state(name):
    d = os.path.join(LOG, name)
    if not os.path.isdir(d):
        return "not started"
    if os.path.exists(os.path.join(d, "pass1.json")):
        try:
            if json.load(open(os.path.join(d, "pass1.json"))).get("result"):
                return "DONE"
        except Exception:
            pass
    if os.path.exists(os.path.join(d, "traj.npz")):
        return "simulated, no result"            # the v2 failure mode: artefacts without a logged result
    err = os.path.join(LOG, "_series_jobs", name + ".err")
    if os.path.exists(err) and os.path.getsize(err) > 0:
        return "FAILED"
    return "running"


def main():
    wave = 0
    while True:
        r = running()
        stamp = time.strftime("%H:%M")
        if r is None:
            # NOT an empty queue. Firing the next wave here would double-submit the current one.
            print(f"[{stamp}] ssh to {cluster.SSH} unavailable -- holding; jobs already submitted run "
                  f"on compute nodes and are unaffected", flush=True)
        else:
            names = [n for w in WAVES for n in w]
            done = [n for n in names if state(n) == "DONE"]
            bad = [n for n in names if state(n) in ("FAILED", "simulated, no result")]
            print(f"[{stamp}] queue {len(r)} | done {len(done)}/{len(names)}"
                  + (f" | ATTENTION {bad}" if bad else ""), flush=True)
            # AN EMPTY QUEUE IS NOT THE SAME AS A FINISHED WAVE. When login1 hung, the launcher
            # submitted nothing while reporting success, so 84-89 never ran; the queue was then
            # legitimately empty and this fired the NEXT wave on top of a wave that had never started.
            # Require the previous wave to have actually PRODUCED something -- a folder each -- before
            # believing it is done.
            prev_ok = True
            if wave > 0:
                prev = WAVES[wave - 1]
                started = [n for n in prev if os.path.isdir(os.path.join(LOG, n))]
                prev_ok = len(started) == len(prev)
                if not prev_ok:
                    print(f"[{stamp}] queue empty but wave {wave-1} left no folder for "
                          f"{[n for n in prev if n not in started]} -- it never ran. Holding rather "
                          f"than stacking the next wave on top.", flush=True)
            if not r and prev_ok and wave < len(WAVES):
                print(f"[{stamp}] queue empty -> firing wave {wave}: {', '.join(WAVES[wave])}",
                      flush=True)
                subprocess.run([sys.executable, os.path.join(HERE, "series_launch.py")] + WAVES[wave],
                               cwd=HERE)
                wave += 1
            elif not r and wave >= len(WAVES):
                print(f"[{stamp}] all waves fired and the queue is empty -- stopping", flush=True)
                return
        time.sleep(PERIOD)


if __name__ == "__main__":
    main()
