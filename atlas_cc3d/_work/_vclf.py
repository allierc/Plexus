import yaml
d = yaml.safe_load(open('_work/contactlocalflex.yaml'))
print('OK', d['id'], d['verdict'], d['implementation_of'], d['status'], d['contract']['name'])
for k in ('name','kind','family','set','inputs','outputs','reads','writes','maps'):
    assert k in d['contract'], k
print('contract fields complete')
