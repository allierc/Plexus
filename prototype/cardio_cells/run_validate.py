import json, numpy as np
import beat as B, validate as V
uv = B.load()
out = V.report(uv)
json.dump(out, open("validate.json", "w"), indent=1)
for k, rows in out.items():
    best = max(rows, key=lambda r: r["excess"])
    print(f"\n  {k}: most reproducible structure at sigma = {best['sigma']} grid points "
          f"({best['sigma']*15:.0f} px), excess over null {best['excess']:.4f}")
