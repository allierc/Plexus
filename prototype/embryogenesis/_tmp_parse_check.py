import embryo_loop as L
slots = L.parse_slots(5)
print("n slots:", len(slots))
for s in slots:
    argv = ["showcase.py", s["spec"], "tag=" + s["name"], "frames=3000", "stride=3"] + s["ov"]
    ov = dict(kv.split("=", 1) for kv in argv[2:] if "=" in kv)
    print(f"{s['name']:30s} frames={ov.get('frames')} stride={ov.get('stride')} "
          f"div={ov.get('agent.div_rate')} mass={ov.get('agent_to_mpm.agent_mass')} omega={ov.get('mpm_spin.omega')}")
