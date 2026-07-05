import yaml
for f in ['specs/embryo_1E_xrep.yaml', 'specs/embryo_1E_selfagg.yaml', 'specs/embryo_1E_selfagg_hi.yaml']:
    d = yaml.safe_load(open(f))
    ops = [o['op'] for o in d['operators']]
    sels = [o.get('at') for o in d['operators'] if 'type=' in str(o.get('at'))]
    print(f, 'OK', len(ops), 'ops; type-selectors:', sels)
