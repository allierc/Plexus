import yaml
d = yaml.safe_load(open('_work/lengthconstraint.yaml'))
assert d['verdict'] == 'new'
assert d['status'] == 'normalized'
assert d['contract']['name'] == 'elongate'
need = {'name','kind','family','set','inputs','outputs','reads','writes','maps'}
assert need <= set(d['contract'])
assert d['of'] is None and d['implementation_of'] is None
print('OK: parses, keys present, contract complete')
