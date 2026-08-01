import yaml
d = yaml.safe_load(open('/workspace/Plexus/atlas_jax_morph/_work/hertzian.yaml'))
print('OK top keys:', list(d.keys()))
print('verdict:', d['verdict'], '| of:', d['of'], '| implementation_of:', d['implementation_of'], '| status:', d['status'])
c = d['contract']
print('contract name/kind/family/set:', c['name'], c['kind'], c['family'], c['set'])
print('reads:', c['reads'])
print('writes:', c['writes'])
print('inputs/outputs:', c['inputs'], c['outputs'])
print('maps count:', len(c['maps']))
# echo the record.py rule checks that apply to a normalized entry
KINDS = ("lateral","aggregate","broadcast","exchange","field","structural","rewire")
FAMILIES = {"motion","interaction","polarity","fields","mechanics","mpm","coupling","hierarchy","growth","topology"}
assert c['kind'] in KINDS, 'R6 kind'
assert c['family'] in FAMILIES, 'R6 family'
assert c['writes'], 'R7 no-op'
assert c['name'], 'R6 name'
assert d['verdict'] in ("alias","refinement","new","out_of_scope"), 'R4 verdict'
assert d['of'], 'R4 of'
assert d['why'], 'R4 why'
for p, role in (d.get('params') or {}).items():
    assert role, f'R8 empty role {p}'
print('LOCAL RULE CHECKS PASSED (R4/R6/R7/R8)')
