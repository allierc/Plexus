"""Ground-truth simulators (forward models) per domain.

Each generator produces datasets the operators are trained to invert. They mirror
the sibling repos' `generators/` and will be ported behind a common interface:
ode_step + field_step writing trajectories into graphs_data/.
"""
