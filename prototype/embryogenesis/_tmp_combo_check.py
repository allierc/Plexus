import yaml
for f in ["embryo_1E_combo", "embryo_1E_combo_lo", "embryo_1E_combo_hi", "embryo_1E_combo_hin"]:
    d = yaml.safe_load(open(f"specs/{f}.yaml"))
    ops = [o for o in d["operators"] if o["op"] == "chemotaxis"]
    n = d["sets"]["agent"]["n"]
    print(f, "OK  n=", n, " chemotaxis:", [(o["at"], o["channel"], o["gain"]) for o in ops])
