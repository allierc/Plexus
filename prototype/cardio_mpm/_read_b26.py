import glob, os, json
for d in sorted(glob.glob('archive/p3_b26_s*/')):
    prog = os.path.join(d, 'progress.txt')
    cfg = os.path.join(d, 'config.json')
    print("====", d)
    if os.path.exists(prog):
        lines = open(prog).read().strip().splitlines()
        print("\n".join(lines[-5:]))
    else:
        print("NO progress.txt")
    if os.path.exists(cfg):
        c = json.load(open(cfg))
        keys = ['bwidth', 'warmup', 'stiff_hi', 'stiff_lo', 'drag_k', 'amplitude', 'gain_hi', 'n_iter']
        print("  cfg:", dict((k, c.get(k)) for k in keys))
    ckpts = sorted(glob.glob(os.path.join(d, 'checkpoints', 'dashboard_*.png')))
    print("  last dashboard:", ckpts[-1] if ckpts else "NONE")
