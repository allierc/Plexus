import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
import yaml, record, registry_view

entry = yaml.safe_load(open("_work/no_force.yaml"))
print("WORK entry parses OK; keys:", list(entry.keys()))
print("verdict:", entry["verdict"], "| status:", entry["status"], "| of:", entry["of"])
print("contract:", entry["contract"])

doc = record.load()
for i, m in enumerate(doc["mechanisms"]):
    if m["id"] == "no_force":
        doc["mechanisms"][i] = entry
        break
vs = record.validate(doc, registry_view.load())
nf = [x for x in vs if x[1] == "no_force"]
print("\nviolations touching no_force:", nf)
print("total violations in merged record:", len(vs))
