import yaml
d = yaml.safe_load(open('_work/pixel_neighbourhood.yaml'))
print('OK', d['verdict'], d['of'], d['status'], sorted(d['contract'].keys()))
