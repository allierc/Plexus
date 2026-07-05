import yaml
for f in ['embryo_1D_base','embryo_1D_mot18','embryo_1D_mot24','embryo_1D_dense']:
    d=yaml.safe_load(open('specs/%s.yaml'%f))
    ops={o['op']:o for o in d['operators']}
    a=d['sets']['agent']
    sched=set(s if isinstance(s,str) else 'substep' for s in d['schedule'])
    print('%-18s n=%s spawn_r=%s move=%s omega=%s a2m.k=%s fa.gain=%s div.rate=%s anch.k=%s frames=%s'%(
        f,a['n'],a['spawn_radius'],a['types']['a']['move_speed'],ops['mpm_spin']['omega'],
        ops['agent_to_mpm']['k'],ops['flow_align']['gain'],ops['cell_divide']['rate'],
        ops['mpm_anchor']['k'],d['general']['n_frames']))
