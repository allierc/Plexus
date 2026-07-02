# Manual smoke batch (2 slots): baseline vs weaker coupling.
baseline : SPEC specs/embryo_base.yaml
low_k : SPEC specs/embryo_base.yaml mpm_to_agent.k 0.1 agent_to_mpm.agent_mass 5e-7
