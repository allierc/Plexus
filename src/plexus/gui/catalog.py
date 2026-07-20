"""Registry -> JSON catalog: the node palette is *driven by the operator registry*.

The GUI never hard-codes the operator list. It introspects the CODEBASE registry
(`plexus.operators`, the 41 canonical operators / 42 implementation classes) and
emits, per operator, exactly what a node needs to render and edit:

  * kind / family / acting set / emit / supported dims / differentiability,
  * the typed signature (reads / writes / maps / inputs / outputs) -- the ports,
  * per-IMPLEMENTATION parameter schemas (name, mechanistic role, required?, default,
    inferred type), so selecting `implementation:` reshapes the param panel,
  * mechanism tags + aliases.

Parameter defaults are not stored as class attributes, so we recover them by a light
static read of each class's source (`params.get("k", <default>)` / `params["k"]`);
`PARAM_ROLES` names the meaningful knobs and `REQUIRES_PARAMS` marks the mandatory
ones. Nothing here executes an operator.
"""

from __future__ import annotations

import ast
import inspect
import re

# importing the operators package registers all codebase operators (and only those --
# prototype op modules are intentionally NOT imported: the palette is the codebase atlas).
import plexus.operators  # noqa: F401
from plexus.models import registry as R
from plexus.models.base import KINDS, EMITS
from plexus.models.state import INTEGRATIONS


# muted, flat accents — one per operator family (the paper's taxonomy). Desaturated
# on purpose: a neat flat scheme, no neon.
FAMILY_COLORS = {
    "motion":      "#4c9a8e",
    "interaction": "#b56b86",
    "polarity":    "#8f7fb0",
    "fields":      "#5a90b0",
    "mechanics":   "#b39152",
    "mpm":         "#b3785a",
    "coupling":    "#7d9e5c",
    "hierarchy":   "#6d84b0",
    "growth":      "#b06a72",
    "topology":    "#5e9ba0",
}
SET_COLOR = "#6f9ac2"     # sets  (paper: blue, muted)
FIELD_COLOR = "#79ab7e"   # fields (paper: green, muted)

_GET_RE = re.compile(r"""params\.get\(\s*['"](\w+)['"]\s*(?:,\s*([^()]+?))?\s*\)""")
_ITEM_RE = re.compile(r"""params\[\s*['"](\w+)['"]\s*\]""")

# spec-line keys that are structural wiring, not tunable params (they get their own ports).
_RESERVED = {"op", "at", "to", "from", "implementation", "emit"}


def _hidden(key):
    return key.startswith("_") or key in _RESERVED


def _parse_default(text):
    if text is None:
        return None
    try:
        return ast.literal_eval(text.strip())
    except Exception:
        return None


def _ptype(default, role):
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, (int, float)):
        return "number"
    if isinstance(default, str):
        return "enum" if (role and "|" in role) else "str"
    return "number"


def _param_schema(cls):
    """Best-effort tunable schema for one implementation class.

    Union of PARAM_ROLES (the documented knobs), REQUIRES_PARAMS (mandatory), and any
    `params.get(...)` / `params[...]` keys found in the class source. Underscored keys
    (engine-injected, e.g. `_at`) are hidden.
    """
    roles = dict(getattr(cls, "PARAM_ROLES", {}) or {})
    required = list(getattr(cls, "REQUIRES_PARAMS", []) or [])
    defaults = {}
    order = []
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        src = ""
    for m in _GET_RE.finditer(src):
        key, dflt = m.group(1), m.group(2)
        if _hidden(key):
            continue
        if key not in order:
            order.append(key)
        if dflt is not None and key not in defaults:
            defaults[key] = _parse_default(dflt)
    for m in _ITEM_RE.finditer(src):
        key = m.group(1)
        if _hidden(key):
            continue
        if key not in order:
            order.append(key)

    # required + documented first (stable, meaningful order), then discovered extras.
    names = []
    for k in required + list(roles.keys()) + order:
        if k not in names and not _hidden(k):
            names.append(k)

    out = []
    for k in names:
        role = roles.get(k)
        default = defaults.get(k)
        typ = _ptype(default, role)
        entry = {
            "name": k,
            "role": role,
            "required": k in required,
            "default": default,
            "type": typ,
        }
        if typ == "enum" and role:
            entry["options"] = [t.strip() for t in role.split("|")]
        out.append(entry)
    return out


def _impl_entry(cls):
    sig = cls.signature() if hasattr(cls, "signature") else {}
    return {
        "params": _param_schema(cls),
        "reads": list(sig.get("reads", [])),
        "writes": list(sig.get("writes", [])),
        "maps": list(sig.get("maps", [])),
        "inputs": list(sig.get("inputs", [])),
        "outputs": list(sig.get("outputs", [])),
        "emit": getattr(cls, "EMIT", None),
        "dims": list(getattr(cls, "SUPPORTED_DIMS", [])),
        "differentiable": bool(getattr(cls, "DIFFERENTIABLE", True)),
        "requires_params": list(getattr(cls, "REQUIRES_PARAMS", []) or []),
        "mechanism_tags": list(getattr(cls, "MECHANISM_TAGS", []) or []),
        "transitional": bool(getattr(cls, "TRANSITIONAL", False)),
        "class": cls.__name__,
    }


def _op_entry(name, contract):
    default_cls = contract.implementations.get(contract.default)
    aliases = [n for n in getattr(default_cls, "REGISTERED_NAMES", [name]) if n != name]
    return {
        "name": name,
        "canonical": True,
        "kind": contract.kind,
        "family": contract.family,
        "set": contract.set,
        "color": FAMILY_COLORS.get(contract.family, "#8b93a1"),
        "mechanism_tags": list(getattr(default_cls, "MECHANISM_TAGS", []) or []),
        "default_impl": contract.default,
        "implementations": {impl: _impl_entry(cls) for impl, cls in contract.implementations.items()},
        "aliases": aliases,
    }


def build_catalog() -> dict:
    """The full node-palette catalog: every canonical codebase operator + vocab."""
    contracts = R._OP_CONTRACTS
    default_reg = R._OPERATOR_REGISTRY

    # canonical name = the first registered name of its default class (aliases excluded).
    canonical = [
        n for n, c in default_reg.items()
        if getattr(c, "REGISTERED_NAMES", [n])[0] == n
    ]

    operators = []
    for name in sorted(canonical):
        operators.append(_op_entry(name, contracts[name]))

    # alias -> canonical, so a spec that uses an alias still resolves to a node schema.
    by_name = {}
    for name, contract in contracts.items():
        default_cls = contract.implementations.get(contract.default)
        canon = getattr(default_cls, "REGISTERED_NAMES", [name])[0]
        by_name[name] = canon

    families = [
        {"name": fam, "color": FAMILY_COLORS.get(fam, "#8b93a1")}
        for fam in sorted(R.OPERATOR_FAMILIES)
    ]

    return {
        "operators": operators,
        "by_name": by_name,
        "families": families,
        "kinds": list(KINDS),
        "emits": list(EMITS),
        "integrations": list(INTEGRATIONS),
        "field_kinds": list(R._FIELD_REGISTRY.keys()),
        "boundaries": ["wall", "periodic", "free"],
        "state_boundaries": ["free", "world"],
        "set_color": SET_COLOR,
        "field_color": FIELD_COLOR,
        "counts": {
            "canonical": len(canonical),
            "contracts": len(contracts),
            "impl_classes": len({cls for c in contracts.values() for cls in c.implementations.values()}),
        },
    }
