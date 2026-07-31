# disposable scratch probe (rm sandbox-blocked). Loads _work/division.yaml and runs
# record.validate against the frozen baseline. Safe to delete.
import yaml, record, registry_view

with open("_work/division.yaml") as f:
    m = yaml.safe_load(f)
doc = {"repository": "x", "paper": "y", "model_family": "z", "commit": "c", "mechanisms": [m]}
for r, mid, msg in record.validate(doc, registry_view.load()):
    print(r, mid, msg)
