
import json, os
import jax, jax.numpy as jnp
import jax_morph as jxm

out = {}
out["jax"] = jax.__version__
out["jax_morph"] = jxm.__version__
out["public_api"] = sorted(n for n in dir(jxm) if not n.startswith("_"))
print(json.dumps(out, indent=2))
with open(os.path.join(os.environ["OUT"], "verify.json"), "w") as f:
    json.dump(out, f, indent=2)
