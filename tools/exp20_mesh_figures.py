"""exp20: THE MULTI-LEVEL MESH ON GLUCOSE FISH 1, exp17's mesh slides (tools/exp17_mesh_figures.py, called unchanged).

    PYTHONPATH=src:tools python tools/exp20_mesh_figures.py 3 4 5

exp17's tool is wired to its own experiment folder and to the destriped ZAPBench neurons; here, in memory only, its
folders point at exp20's, its recording at glucose fish 1's cells (gutbrain_glucose_f1_nosw: 190,346 cells, median
nearest neighbour 4.0 um against ZAPBench's 7.5 um, so the 16-um finest binned level is kept: denser cells, not
sparser), and its view maps fish 1 head UP as exp17's mesh slides draw theirs (the raw frame has the head at +x:
(-y, x, z), a rotation). Writes presentation/Movies/gcmesh_<L>.mp4 (+ .png), presentation/figs/gcmesh_<L>_levels.png
and data/gcmesh_<L>.json (per level: cube, nodes, edges, median edge; the law's three edge sets; connectivity).
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
REC = "gutbrain_glucose_f1_nosw"


def patch():
    import exp17_mesh_figures as MF
    from plexus.tasks import trace_recording as TR
    MF.EXP, MF.PRES = EXP, os.path.join(EXP, "presentation")
    MF.view = lambda P: np.stack([-P[:, 1], P[:, 0], P[:, 2]], 1)        # head (+x) up, a rotation of the raw frame
    if not getattr(TR.load, "_exp20", False):
        _load = TR.load

        def load(name, *a, **k):                                            # exp17's recording -> fish 1's cells
            return _load(REC if name == "zapbench_destripe" else name, *a, **k)
        load._exp20 = True
        TR.load = load
    return MF


if __name__ == "__main__":
    MF = patch()
    for L_ in (int(a) for a in sys.argv[1:]):
        MF.figures(L_)
