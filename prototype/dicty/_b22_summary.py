import json
d=json.load(open('sweep_results.json'))
print('real_inner_mass', d['real_inner_mass'])
for s in d['sweeps']:
    print(f"sw {s['idx']:2d} | {s['param']:30s} | best_val={s['best_value']:>10.4g} | best_loss={s['best_loss']:.4f} | best_inner={s['best_inner']:.3f}")
    vs=s['values']; ls=s['loss']; ims=s['inner_mass']
    pairs=' '.join([f'{v:.3g}:{l:.3f}/{i:.2f}' for v,l,i in zip(vs,ls,ims)])
    print('   ', pairs)
