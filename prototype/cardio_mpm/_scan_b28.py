import glob,os
dirs=sorted(glob.glob('archive/p3_b28_*'))+sorted(glob.glob('archive/p3_b27_*'))
for d in dirs:
    n=os.path.basename(d)
    p=os.path.join(d,'progress.txt')
    if os.path.isfile(p):
        lines=open(p).read().splitlines()
        hits=[l for l in lines if ('done ->' in l or 'LS=' in l)]
        last=hits[-1] if hits else (lines[-1] if lines else '')
        print(f"{n:34s} | L={len(lines):<4d} | {last[:130]}")
    else:
        files=','.join(sorted(os.listdir(d))) if os.path.isdir(d) else 'MISSING'
        print(f"{n:34s} | NOprog | {files[:100]}")
