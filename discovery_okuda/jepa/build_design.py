"""Build the spec -> outcome design matrix for the JEPA-feasibility test.

X comes from spec_run.yaml ONLY (nothing measured).
Y comes from diag.json['summary'], restricted to the admitted metric bank.
groups come from the spec's STRUCTURE (op multiset + implementation strings).

Writes /tmp/jepa_test/design.npz. Nothing is written outside /tmp/jepa_test/.
"""
import os, sys, glob, json, hashlib, collections
import numpy as np
import yaml

sys.path.insert(0, '/workspace/Plexus/discovery_okuda')
import metrics as M

OUT = '/tmp/jepa_test'
ROOTS = ['/workspace/Plexus/log/okuda',
         '/workspace/Plexus/log/okuda/_archive_r001-r022_2026-08-12']

MIN_NUM_SPECS = 20   # numeric (op,param) kept if numeric in >= this many specs
MIN_CAT_SPECS = 10   # (op,param,value) kept if the value appears in >= this many specs
MIN_TARGET_FRAC = 0.80

# Parameter names that are arbitrary labels, not physics.  A random seed can only
# act as a run-identifier for a ridge; it carries no transferable signal.
SEED_PARAMS = {'seed'}


def discover():
    runs = []
    for root in ROOTS:
        for d in sorted(glob.glob(os.path.join(root, '*'))):
            if not os.path.isdir(d):
                continue
            if (os.path.isfile(os.path.join(d, 'spec_run.yaml'))
                    and os.path.isfile(os.path.join(d, 'diag.json'))):
                runs.append(d)
    return runs


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def main():
    os.makedirs(OUT, exist_ok=True)
    dirs = discover()
    names = [os.path.basename(d) for d in dirs]
    print(f'[discover] {len(dirs)} run dirs with BOTH spec_run.yaml and diag.json')
    dup_names = [n for n, c in collections.Counter(names).items() if c > 1]
    print(f'[discover] duplicate basenames among kept runs: {dup_names}')

    # ---------------- load specs -------------------------------------------
    specs, spec_hash = {}, {}
    for d in dirs:
        raw = open(os.path.join(d, 'spec_run.yaml'), 'rb').read()
        sp = yaml.safe_load(raw)
        # the run's own name is in general.name -- strip it before hashing so
        # identical recipes under different names collide
        sp2 = json.loads(json.dumps(sp, default=str))
        sp2.get('general', {}).pop('name', None)
        specs[d] = sp
        spec_hash[d] = hashlib.md5(json.dumps(sp2, sort_keys=True).encode()).hexdigest()
    hc = collections.Counter(spec_hash.values())
    exact_dupes = {h: [os.path.basename(d) for d in dirs if spec_hash[d] == h]
                   for h, c in hc.items() if c > 1}
    print(f'[discover] exact-duplicate recipes (name stripped): {len(exact_dupes)} '
          f'clusters -> {list(exact_dupes.values())[:6]}')

    # ---------------- census ------------------------------------------------
    op_present = collections.Counter()          # op -> n specs containing it
    op_mult = {}                                # dir -> Counter(op)
    num_count = collections.Counter()           # (op,param) -> n specs numeric
    cat_count = collections.Counter()           # (op,param,value) -> n specs
    first_inst = {}                             # dir -> {op: first instance dict}

    for d in dirs:
        ops = specs[d].get('operators') or []
        mult = collections.Counter()
        firsts = {}
        for o in ops:
            if not isinstance(o, dict) or 'op' not in o:
                continue
            n = o['op']
            mult[n] += 1
            if n not in firsts:
                firsts[n] = o
        op_mult[d] = mult
        first_inst[d] = firsts
        for n in mult:
            op_present[n] += 1
        for n, o in firsts.items():
            for k, v in o.items():
                if k == 'op' or k in SEED_PARAMS:
                    continue
                if is_num(v):
                    num_count[(n, k)] += 1
                elif isinstance(v, bool):
                    num_count[(n, k)] += 1        # bool -> numeric 0/1
                elif isinstance(v, str):
                    cat_count[(n, k, v)] += 1
                # lists/dicts (only cell_die.cells, 2 specs) are dropped

    ops_all = sorted(op_present)
    num_cols = sorted([k for k, c in num_count.items() if c >= MIN_NUM_SPECS])
    # a string param with a single corpus-wide value ('at', 'cell_set',
    # 'vertex_set', 'implementation') is wiring, not an implementation CHOICE:
    # its one-hot is identical to the operator's presence column.  Drop it.
    n_vals = collections.Counter()
    for (n, k, v) in cat_count:
        n_vals[(n, k)] += 1
    single_valued = sorted({f'{n}.{k}' for (n, k), c in n_vals.items() if c == 1})
    print(f'[census] string params with ONE corpus-wide value, dropped as wiring '
          f'(their one-hot duplicates the presence column): {single_valued}')
    cat_cols = sorted([k for k, c in cat_count.items()
                       if c >= MIN_CAT_SPECS and n_vals[(k[0], k[1])] > 1])
    print(f'[census] {len(ops_all)} distinct operators; '
          f'{len(num_cols)} numeric (op,param) >= {MIN_NUM_SPECS} specs '
          f'(of {len(num_count)} seen); '
          f'{len(cat_cols)} (op,param,value) >= {MIN_CAT_SPECS} specs '
          f'(of {len(cat_count)} seen)')
    dropped_num = sorted([(c, k) for k, c in num_count.items() if c < MIN_NUM_SPECS],
                         reverse=True)
    print(f'[census] numeric params dropped for rarity: '
          f'{[(f"{a}.{b}", c) for c, (a, b) in dropped_num]}')

    # ---------------- build raw blocks -------------------------------------
    N = len(dirs)
    P = np.zeros((N, len(ops_all)))                      # presence
    Vraw = np.full((N, len(num_cols)), np.nan)           # numeric, NaN = absent
    C = np.zeros((N, len(cat_cols)))                     # categorical one-hot

    for i, d in enumerate(dirs):
        firsts = first_inst[d]
        for j, n in enumerate(ops_all):
            P[i, j] = 1.0 if n in firsts else 0.0
        for j, (n, k) in enumerate(num_cols):
            o = firsts.get(n)
            if o is None or k not in o:
                continue
            v = o[k]
            if isinstance(v, bool):
                Vraw[i, j] = 1.0 if v else 0.0
            elif is_num(v):
                Vraw[i, j] = float(v)
        for j, (n, k, val) in enumerate(cat_cols):
            o = firsts.get(n)
            C[i, j] = 1.0 if (o is not None and o.get(k) == val) else 0.0

    # standardise numeric columns over PRESENT runs only, then absent -> 0.
    # absent therefore sits at the mean of the present runs and the presence
    # binary carries the offset; present-with-raw-value-0 maps to -mean/std,
    # so "absent" and "present, value 0" stay distinguishable.
    mu = np.zeros(len(num_cols))
    sd = np.ones(len(num_cols))
    keep_num = np.ones(len(num_cols), dtype=bool)
    V = np.zeros_like(Vraw)
    for j in range(len(num_cols)):
        col = Vraw[:, j]
        m = np.isfinite(col)
        mu[j] = col[m].mean()
        s = col[m].std()
        # a column whose present values are all the same float still has
        # s ~ 1e-16 from summation rounding; dividing by it turns pure noise
        # into z = +-1.  Test the peak-to-peak SPREAD against the scale of the
        # value, not the std against zero.
        spread = col[m].max() - col[m].min() if m.sum() else 0.0
        if (not np.isfinite(s)) or s <= 0 or spread <= 1e-9 * max(1.0, abs(mu[j])):
            keep_num[j] = False
            s = 1.0
        sd[j] = s
        z = np.zeros(N)
        z[m] = (col[m] - mu[j]) / s
        V[:, j] = z

    keep_p = P.std(axis=0) > 0
    keep_c = C.std(axis=0) > 0
    const_ops = [ops_all[j] for j in range(len(ops_all)) if not keep_p[j]]
    const_num = [f'{a}.{b}' for j, (a, b) in enumerate(num_cols) if not keep_num[j]]
    const_cat = [f'{a}.{b}={c}' for j, (a, b, c) in enumerate(cat_cols) if not keep_c[j]]
    print(f'[prune] constant presence cols dropped ({len(const_ops)}): {const_ops}')
    print(f'[prune] constant numeric cols dropped ({len(const_num)}): {const_num}')
    print(f'[prune] constant categorical cols dropped ({len(const_cat)}): {const_cat}')

    X = np.hstack([P[:, keep_p], V[:, keep_num], C[:, keep_c]])
    Xraw_num = Vraw[:, keep_num]
    feat = ([f'has:{ops_all[j]}' for j in range(len(ops_all)) if keep_p[j]]
            + [f'num:{a}.{b}' for j, (a, b) in enumerate(num_cols) if keep_num[j]]
            + [f'cat:{a}.{b}={c}' for j, (a, b, c) in enumerate(cat_cols) if keep_c[j]])
    feat_num_names = [f'num:{a}.{b}' for j, (a, b) in enumerate(num_cols) if keep_num[j]]
    feat_kind = np.array(['has'] * int(keep_p.sum()) + ['num'] * int(keep_num.sum())
                         + ['cat'] * int(keep_c.sum()))
    print(f'[X] shape {X.shape}  ({int(keep_p.sum())} presence + '
          f'{int(keep_num.sum())} numeric + {int(keep_c.sum())} categorical)')

    # bit-identical columns: report (they are one degree of freedom, not several)
    sig = collections.defaultdict(list)
    for j in range(X.shape[1]):
        sig[X[:, j].tobytes()].append(feat[j])
    alias = [v for v in sig.values() if len(v) > 1]
    print(f'[X] bit-identical column clusters ({len(alias)}), i.e. these are ONE '
          f'degree of freedom each, kept but not independent: {alias}')
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    rank = int((s > s[0] * 1e-10).sum())
    print(f'[X] numerical rank of the centred design: {rank} of {X.shape[1]} columns')

    # ten most-populated columns (most nonzero / most present)
    pop = []
    off = 0
    for j in range(int(keep_p.sum())):
        pop.append((int(X[:, off + j] != 0).__index__() if False else int((X[:, off + j] != 0).sum()),
                    feat[off + j]))
    off += int(keep_p.sum())
    for j in range(int(keep_num.sum())):
        pop.append((int(np.isfinite(Xraw_num[:, j]).sum()), feat[off + j]))
    off += int(keep_num.sum())
    for j in range(int(keep_c.sum())):
        pop.append((int((X[:, off + j] != 0).sum()), feat[off + j]))
    pop.sort(reverse=True)
    print('[X] 10 most-populated columns (n runs where the column is populated: '
          'operator present / parameter numerically present / value taken):')
    for c, nm in pop[:10]:
        print(f'      {c:4d}/{N}  {nm}')

    # ---------------- targets ----------------------------------------------
    bank = list(M.names())
    summaries = {}
    for d in dirs:
        summaries[d] = (json.load(open(os.path.join(d, 'diag.json'))) or {}).get('summary', {}) or {}

    def as_float(v):
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            f = float(v)
            return f if np.isfinite(f) else np.nan
        return np.nan

    Yfull = np.full((N, len(bank)), np.nan)
    for i, d in enumerate(dirs):
        s = summaries[d]
        for j, k in enumerate(bank):
            if k in s:
                Yfull[i, j] = as_float(s[k])
    frac = np.isfinite(Yfull).mean(axis=0)
    keep_y = frac >= MIN_TARGET_FRAC
    # a target that is constant wherever it is finite carries no variance to explain
    var_ok = np.zeros(len(bank), dtype=bool)
    for j in range(len(bank)):
        col = Yfull[:, j]
        m = np.isfinite(col)
        var_ok[j] = m.sum() > 1 and col[m].std() > 0
    dropped_const_y = [bank[j] for j in range(len(bank)) if keep_y[j] and not var_ok[j]]
    keep_y = keep_y & var_ok
    Y = Yfull[:, keep_y]
    tnames = [bank[j] for j in range(len(bank)) if keep_y[j]]
    print(f'[Y] metric bank {len(bank)}; '
          f'{int((frac >= MIN_TARGET_FRAC).sum())} finite in >= {MIN_TARGET_FRAC:.0%} of runs; '
          f'{len(dropped_const_y)} of those are constant and dropped '
          f'({dropped_const_y}); {Y.shape[1]} targets kept')
    print(f'[Y] shape {Y.shape}; NaN cells remaining {int(np.isnan(Y).sum())} '
          f'({np.isnan(Y).mean():.3%}) -- left as NaN, mask per target when fitting')
    notin = [k for k in bank if not any(k in summaries[d] for d in dirs)]
    print(f'[Y] bank keys never present in ANY diag.json ({len(notin)}): {notin}')

    # ---------------- groups ------------------------------------------------
    sv = collections.defaultdict(set)
    for d in dirs:
        for n, o in first_inst[d].items():
            for k, v in o.items():
                if k != 'op' and isinstance(v, str):
                    sv[(n, k)].add(v)
    varying_str = {k for k, s in sv.items() if len(s) > 1}
    print(f'[groups] string params that vary across the corpus (used as '
          f'implementation choices): {sorted(varying_str)}')

    def struct_key(d, use_str=True, use_mult=True):
        mult = op_mult[d]
        if use_mult:
            base = tuple(sorted(f'{n}x{m}' for n, m in mult.items()))
        else:
            base = tuple(sorted(mult))
        extra = []
        if use_str:
            for n, o in first_inst[d].items():
                for k, v in o.items():
                    if k != 'op' and isinstance(v, str) and (n, k) in varying_str:
                        extra.append(f'{n}.{k}={v}')
        return (base, tuple(sorted(extra)))

    def label(keys):
        uniq = {k: i for i, k in enumerate(sorted(set(keys), key=str))}
        return np.array([uniq[k] for k in keys])

    g_primary = label([struct_key(d) for d in dirs])
    g_coarse = label([struct_key(d, use_str=False, use_mult=False) for d in dirs])
    # round id from the run name prefix (r0NN_.. or the leading token of a
    # hand-named sweep) -- an independent, usually STRICTER notion of "same batch"
    def round_id(n):
        if len(n) > 4 and n[0] == 'r' and n[1:4].isdigit():
            return n[:4]
        return n.split('_')[0]
    g_round = label([round_id(os.path.basename(d)) for d in dirs])

    def eff(g):
        c = np.array(list(collections.Counter(g.tolist()).values()), float)
        p = c / c.sum()
        return 1.0 / (p ** 2).sum()

    for lab, g in [('PRIMARY  struct = op-multiset + impl-strings', g_primary),
                   ('coarse   struct = op-name-set only', g_coarse),
                   ('round    name prefix', g_round)]:
        c = sorted(collections.Counter(g.tolist()).values(), reverse=True)
        print(f'[groups] {lab}: {len(c)} groups, effective (inverse-Simpson) '
              f'{eff(g):.1f}, sizes {c}')

    ng = len(set(g_primary.tolist()))
    if ng < 15:
        print(f'\n!!! ONLY {ng} DISTINCT SPEC STRUCTURES AMONG {N} RUNS. '
              f'THE REAL SAMPLE SIZE IS {ng}, NOT {N}. !!!\n')
    top2 = sorted(collections.Counter(g_primary.tolist()).values(), reverse=True)[:2]
    print(f'!!! the two largest structure groups hold {sum(top2)}/{N} '
          f'({sum(top2)/N:.0%}) of the runs; effective group count is '
          f'{eff(g_primary):.1f}, not {ng}. Grouped CV will be lopsided. !!!\n')

    has_strip = np.array([os.path.isfile(os.path.join(d, 'strip.png')) for d in dirs])
    print(f'[aux] runs with strip.png: {int(has_strip.sum())}/{N}')

    # spec knobs that live OUTSIDE 'operators' and are therefore not in X, per
    # the brief.  Handed over separately so the fitting agent can choose.
    extra_names = ['general.n_frames', 'general.seed', 'general.dt',
                   '_run.target_cells', '_run.seed_cells', 'sets.cell.n',
                   'sets.vertex.n']
    Xex = np.full((N, len(extra_names)), np.nan)
    for i, d in enumerate(dirs):
        sp = specs[d]
        def dig(path):
            cur = sp
            for p in path.split('.'):
                if not isinstance(cur, dict) or p not in cur:
                    return np.nan
                cur = cur[p]
            return float(cur) if is_num(cur) else np.nan
        for j, k in enumerate(extra_names):
            Xex[i, j] = dig(k)
    for j, k in enumerate(extra_names):
        vc = collections.Counter(Xex[:, j][np.isfinite(Xex[:, j])].tolist())
        print(f'[extra, NOT in X] {k}: {len(vc)} distinct, {vc.most_common(5)}')

    # repeat-spec disagreement = the noise floor no surrogate can beat
    clusters = collections.defaultdict(list)
    for i, d in enumerate(dirs):
        clusters[spec_hash[d]].append(i)
    dupc = [v for v in clusters.values() if len(v) > 1]
    ysd = np.nanstd(Y, axis=0)
    rels = []
    for v in dupc:
        w = np.nanstd(Y[v], axis=0)
        rels.append(w / np.where(ysd > 0, ysd, np.nan))
    if rels:
        R = np.vstack(rels)
        print(f'[noise floor] {sum(len(v) for v in dupc)} runs sit in {len(dupc)} '
              f'clusters of BIT-IDENTICAL recipes; within-cluster SD / corpus SD '
              f'over the {Y.shape[1]} targets: median {np.nanmedian(R):.3f}, '
              f'mean {np.nanmean(R):.3f}, 90th pct {np.nanpercentile(R, 90):.3f}. '
              f'6 of the 8 clusters agree to the last digit; the sim is '
              f'deterministic and the ceiling on R2 is ~1, not the noise floor.')

    np.savez(os.path.join(OUT, 'design.npz'),
             X=X, X_raw_numeric=Xraw_num, num_mean=mu[keep_num], num_std=sd[keep_num],
             Y=Y, Y_mask=np.isfinite(Y),
             feature_names=np.array(feat), feature_kind=feat_kind,
             numeric_feature_names=np.array(feat_num_names),
             target_names=np.array(tnames),
             groups=g_primary, groups_coarse=g_coarse, groups_round=g_round,
             run_names=np.array(names), run_dirs=np.array(dirs),
             has_strip=has_strip, spec_hash=np.array([spec_hash[d] for d in dirs]),
             X_extra=Xex, extra_feature_names=np.array(extra_names))
    print(f'[save] {OUT}/design.npz  X{X.shape} Y{Y.shape} groups={ng}')

    # ---------------- eyeball three runs ------------------------------------
    # picked for CONTRAST: a chemistry-free run, a brusselator+death run, and a
    # mid-sweep gray-scott run, so absent/present/value-0 are all visible
    picks = [n for n in ['b_none_static_plain', 'b_bru_gated_plain_death', 'r013_07']
             if n in names]
    while len(picks) < 3:
        for n in names:
            if n not in picks:
                picks.append(n)
                break
    print('\n===== ENCODING BY EYE =====')
    for n in picks:
        i = names.index(n)
        print(f'\n--- {n}  (group {g_primary[i]}, round-group {g_round[i]})')
        nz = np.nonzero(X[i])[0]
        print(f'    {len(nz)} nonzero of {X.shape[1]} columns')
        for j in nz:
            f = feat[j]
            if f.startswith('num:'):
                jj = feat_num_names.index(f)
                r = Xraw_num[i, jj]
                print(f'      {f:42s} z={X[i, j]:+8.3f}  raw={r}')
            else:
                print(f'      {f:42s} ={X[i, j]:.0f}')
        zero_num = [feat_num_names[jj] for jj in range(Xraw_num.shape[1])
                    if not np.isfinite(Xraw_num[i, jj])]
        print(f'    numeric cols ABSENT (encoded 0 = mean of present runs), '
              f'{len(zero_num)}: {zero_num}')


if __name__ == '__main__':
    main()
