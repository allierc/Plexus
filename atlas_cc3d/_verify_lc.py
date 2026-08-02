import yaml
d = yaml.safe_load(open('_work/lengthconstraint.yaml'))
print('PARSES OK')
print('status:', d['status'])
print('verdict:', d['verdict'])
print('implementation_of:', d['implementation_of'])
print('contract.name:', d['contract']['name'], d['contract']['kind'], d['contract']['family'], d['contract']['set'])
w = d['why']
print('stale claim present:', 'maps onto the existing cell_grow' in w)
print('corrected present:', 'did NOT map onto cell_grow' in w and 'volume_elasticity' in w)
