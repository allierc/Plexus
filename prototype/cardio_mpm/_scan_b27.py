import glob, os
for pat in ['archive/p3_b26_s*/','archive/p3_b27_s*/']:
    print('====', pat)
    for d in sorted(glob.glob(pat)):
        pngs = glob.glob(os.path.join(d,'checkpoints','dashboard_*.png'))
        prog = os.path.join(d,'progress.txt')
        last=''
        if os.path.exists(prog):
            lines=[l.strip() for l in open(prog) if l.strip()]
            last=lines[-1] if lines else ''
        print('%-42s pngs=%3d | %s' % (os.path.basename(d.rstrip('/')), len(pngs), last[:100]))
