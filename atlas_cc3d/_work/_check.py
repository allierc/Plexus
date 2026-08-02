import yaml
d = yaml.safe_load(open('_work/cell_as_lattice_domain.yaml'))
print('YAML OK; keys:', list(d.keys()))
c = d['contract']
print('verdict:', d['verdict'], '| name:', c['name'], '| kind:', c['kind'],
      '| family:', c['family'], '| writes:', c['writes'])
assert c['name'] not in {
 'activation_pulse','active_force','active_stress','agent_gather','agent_remodel','agent_scatter',
 'aggregate','apply_material_map','attraction_repulsion','attractor_flow','bounce','broadcast',
 'cell_divide','cell_grow','chemotax','cohesion','decay','deposit','diffuse','drag','glide','gravity',
 'mls_mpm_mechanics','mpm_anchor','mpm_gather','mpm_grid_update','mpm_scatter','mpm_spin','mpm_strain',
 'pacemaker','playback','polarity_align','polarity_flow_align','radius_graph','sediment','sense',
 'separation','signal','squared_law','stillinger_weber','velocity_align','velocity_cruise'}, 'name collides'
assert c['writes'], 'no writes'
assert c['kind'] in ('lateral','aggregate','broadcast','exchange','field','structural','rewire')
assert c['family'] in ('motion','interaction','polarity','fields','mechanics','mpm','coupling','hierarchy','growth','topology')
print('R5/R6/R7 local checks pass')
