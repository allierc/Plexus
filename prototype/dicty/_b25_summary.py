import json

with open('sweep_results.json') as f:
    d = json.load(f)

lines = []
def P(*a): lines.append(' '.join(str(x) for x in a))

P(f"real_inner_mass={d['real_inner_mass']} real_n_mounds={d['real_n_mounds']}")
P(f"n_sweeps={len(d['sweeps'])}")
P('')

for s in d['sweeps']:
    losses = s['loss']
    inners = s['inner_mass']
    vals = s['values']
    nm = s.get('n_mounds', [])
    ms = s.get('morph_score', [])
    bi = min(range(len(losses)), key=lambda i: losses[i])
    bm = min(range(len(ms)), key=lambda i: ms[i]) if ms else bi
    P(f"sw{s['idx']:2d} {s['param']:32s}  bestL@{vals[bi]:.4g}={losses[bi]:.4f} inner={inners[bi]:.3f} nm={(nm[bi] if nm else '?')}  | bestM@{vals[bm]:.4g} morph={(ms[bm] if ms else '?')} nm={(nm[bm] if nm else '?')}  | lossR[{min(losses):.3f},{max(losses):.3f}] innerR[{min(inners):.3f},{max(inners):.3f}] nmR[{min(nm) if nm else '?'},{max(nm) if nm else '?'}]")

P('')
P('=== sweep details ===')
for s in d['sweeps']:
    P(f"--- sw{s['idx']:2d} {s['param']} ---")
    nm = s.get('n_mounds', [])
    ms = s.get('morph_score', [])
    for i, (v, l, im) in enumerate(zip(s['values'], s['loss'], s['inner_mass'])):
        extras = ''
        if nm: extras += f" nm={nm[i]}"
        if ms: extras += f" morph={ms[i]:.3f}"
        P(f"   v={v:>10.4f}  loss={l:.4f}  inner={im:.3f}{extras}")

with open('_b25_summary.txt','w') as f:
    f.write('\n'.join(lines))
