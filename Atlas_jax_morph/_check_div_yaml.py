import yaml
d = yaml.safe_load(open("/workspace/Plexus/Atlas_jax_morph/_work/division.yaml"))
print("OK parsed. top keys:", list(d.keys()))
print("verdict:", d["verdict"], "| of:", d["of"], "| impl_of:", d.get("implementation_of"))
print("status:", d["status"])
c = d["contract"]
print("contract:", c["name"], c["kind"], c["family"], c["set"])
print("reads:", c["reads"])
print("writes:", c["writes"])
print("maps entries:", len(c["maps"]))
# spot-check the kind/family are in the allowed sets
assert c["kind"] in ("lateral","aggregate","broadcast","exchange","field","structural","rewire")
assert c["family"] in {"motion","interaction","polarity","fields","mechanics","mpm","coupling","hierarchy","growth","topology"}
assert c["writes"], "structural exempt but writes present anyway"
print("ALL CHECKS PASS")
