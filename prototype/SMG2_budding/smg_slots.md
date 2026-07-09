# Batch slots -- 8 non-comment lines, one mechanism lever each. Format:
#   name : SPEC specs/<file>.yaml [key val ...]
# Q1: Can collective migration ALONE generate budding? (pairwise + polarity, NO growth)
# ~4 exploit / 3 explore / 1 ablation control. Author new specs for mechanism changes.
base_ref        : SPEC specs/smg_base.yaml
strong_align    : SPEC specs/smg_base.yaml polar_align.gamma 80
weak_align      : SPEC specs/smg_base.yaml polar_align.gamma 10
fast_motile     : SPEC specs/smg_base.yaml agent.move_speed 0.5
cohesive        : SPEC specs/smg_base.yaml repel.strength 10 repel.r0 0.026
loose           : SPEC specs/smg_base.yaml repel.strength 3
flowlock        : SPEC specs/smg_base.yaml flow_align.gain 200
ablation_nopolar: SPEC specs/smg_base.yaml polar_align.gamma 0 polar_align.noise 20
