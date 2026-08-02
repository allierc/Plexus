import yaml, record
doc = yaml.safe_load(open('_work/neighbortracker.yaml'))
base = record.load()
full = {k: base.get(k) for k in base if k != 'mechanisms'}
full['mechanisms'] = [doc]
issues = [r for r in record.validate(full, base) if r[1] in ('-', 'neighbortracker')]
for r in issues:
    print(r)
print('DONE', 'clean' if not issues else f'{len(issues)} issues')
