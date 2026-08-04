# Causal lever-map

_29 runs · coverage **10%** (1/10 cells)_

The campaign's product. Specific questions are queries against this table.

## Coverage

| block | covered | total | |
|---|---|---|---|
| solo | 1 | 10 | 10% |
| pair | 0 | 0 | 0% |
| routing | 0 | 0 | 0% |

## Solo effects — what each operator does ALONE

| operator | n(with) | n(without) | Δscore | verdict | phenotypes seen |
|---|---|---|---|---|---|
| `vesicle_growth` | 3 | 26 | 2.34 | insufficient | exploded×1, spike×1, branching×1 |
| `cell_geometry_3d` | 21 | 8 | 0.335 | raises | sphere×18, exploded×1, spike×1, branching×1 |
| `cell_react` | 27 | 2 | 0.264 | insufficient | sphere×24, exploded×1, spike×1, branching×1 |
| `reconnect_t1_3d` | 27 | 2 | 0.264 | insufficient | sphere×24, exploded×1, spike×1, branching×1 |
| `divide_3d` | 4 | 25 | -0.257 | insufficient | sphere×4 |
| `cell_adjacency` | 29 | 0 | — | insufficient | — |
| `cell_diffuse` | 28 | 1 | — | insufficient | — |
| `cell_rd_seed` | 28 | 1 | — | insufficient | — |
| `seed_mesh_3d` | 29 | 0 | — | insufficient | — |
| `shape_energy_3d` | 29 | 0 | — | insufficient | — |

## Interactions — where the joint effect is NOT the sum

_The expensive half of the map: what cannot be read off the code._

| pair | n | observed | additive prediction | interaction | verdict |
|---|---|---|---|---|---|
| _(none established yet)_ | | | | | |

## Phenotypes observed

sphere×26, exploded×1, spike×1, branching×1

