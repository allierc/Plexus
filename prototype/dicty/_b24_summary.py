import json

with open('sweep_results.json') as f:
    d = json.load(f)

print(f"real_inner_mass={d['real_inner_mass']} real_n_mounds={d['real_n_mounds']}")
print(f"n_sweeps={len(d['sweeps'])}")
print()

for s in d['sweeps']:
    losses = s['loss']
    inners = s['inner_mass']
    vals = s['values']
    lo, hi = min(losses), max(losses)
    print(f"sw{s['idx']:2d} {s['param']:28s} best={s['best_value']:>10.4f} loss={s['best_loss']:.4f} inner={s['best_inner']:.3f}  range[{lo:.3f}..{hi:.3f}] vals_range[{min(vals)}..{max(vals)}]")

print()
print("=== sweep details ===")
for s in d['sweeps']:
    print(f"\n--- sw{s['idx']:2d} {s['param']} ---")
    for v, l, im in zip(s['values'], s['loss'], s['inner_mass']):
        print(f"   v={v:>10.4f}  loss={l:.4f}  inner={im:.3f}")
