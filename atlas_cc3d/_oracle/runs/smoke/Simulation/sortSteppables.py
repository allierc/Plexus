
import os, json
from cc3d.core.PySteppables import *

class DumpSteppable(SteppableBasePy):
    """Write every live cell's state at the end of the run."""
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def finish(self):
        rows = sorted((int(c.id), int(c.type), float(c.volume), float(c.surface),
                       round(float(c.xCOM), 6), round(float(c.yCOM), 6))
                      for c in self.cell_list)
        with open(os.environ["CC3D_DUMP"], "w") as f:
            json.dump({"n": len(rows),
                       "columns": ["id", "type", "volume", "surface", "xCOM", "yCOM"],
                       "cells": rows}, f, indent=1)
