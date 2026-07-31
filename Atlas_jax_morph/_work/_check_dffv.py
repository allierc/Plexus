import yaml
d = yaml.safe_load(open('_work/declared_field_dataflow_validation.yaml'))
KINDS = ['lateral','aggregate','broadcast','exchange','field','structural','rewire']
FAMILIES = ['motion','interaction','polarity','fields','mechanics','mpm','coupling','hierarchy','growth','topology']
print('parsed OK; keys:', list(d.keys()))
print('verdict:', d['verdict'])
print('status :', d['status'])
c = d['contract']
print('contract.name  :', c['name'])
print('kind in KINDS  :', c['kind'] in KINDS, c['kind'])
print('family in FAMS :', c['family'] in FAMILIES, c['family'])
print('writes nonempty:', bool(c['writes']), c['writes'])
print('reads          :', c['reads'])
print('why present    :', bool(d['why']), 'len', len(d['why']))
print('of             :', d['of'])
print('implementation_of present:', 'implementation_of' in d)
missing = [p for p,r in (d.get('params') or {}).items() if not r]
print('params missing role:', missing)
