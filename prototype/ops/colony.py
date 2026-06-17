"""colony: Lateral operator @ cell. The ant carrying-state machine (foraging brain).

This is the *one new operator* the ant colony of Sebastian Lague's "Ant-Simulation"
needs on top of the existing Plexus registry. Everything else in the model --
forward motion (`motility`), 3-sensor pheromone steering (`trail`), recency-faded
pheromone deposit (`secrete` with `runout`), bodies + wall collision (`mpm`), and
the two diffusing pheromone fields -- is reused unchanged. See:
  * paper/plexus.tex, Part III, "Emergent ant trails" (the framework thesis:
    new behaviour is one new registered operator + a spec, engine untouched);
  * papers/Ant-Simulation/Assets/Scripts/Ant.cs (the reference state machine:
    SearchingForFood <-> ReturningHome, pick up / drop off, perception steering).

What it does (per Ant.cs, ported to the grid framework):
  - Each ant carries `cell.loaded` (False = searching for food, True = carrying
    food home) -- the existing `cell[loaded=0/1]` selector the engine already
    resolves, so the dual-pheromone rule is pure spec:
        searching (loaded=0): deposit HOME pheromone, follow FOOD pheromone
        returning (loaded=1): deposit FOOD pheromone, follow HOME pheromone
  - PICK UP: a searching ant inside a food disc -> loaded:=True, turn around,
    deplete that source's stock (optional), reset the recency clock.
  - DROP OFF: a returning ant inside the home disc -> loaded:=False, turn around,
    increment H.food_delivered (the colony objective), reset the clock.
  - PERCEPTION: within `perception` of the relevant target (food when searching,
    home when returning) the ant steers its heading straight at it -- Lague's
    OverlapCircle perception, here a short-range heading lock so the last leg is
    reliable while long-range navigation stays emergent (trail-following).
  - RECENCY CLOCK `cell.t_since_goal`: ticks since the ant was last AT a goal,
    reset to 0 on pick up / drop off. `secrete(runout=...)` reads it so a marker
    is laid strongest just after leaving its goal -> each pheromone forms a
    gradient that DECREASES away from its source (home pheromone densest at the
    nest, food pheromone densest at the food). Trail-following then climbs the
    right pheromone toward the right place. This is the recruitment mechanism.

It mutates cell state + heading and returns {} (no acceleration) -- uniform with
every other structure-touching operator. `cell.heading` is shared with `motility`
and `trail`.
"""

from __future__ import annotations

import math
import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


def _as_circles(spec):
    """Accept a single circle [cx,cy,r] or a list of them -> [K,3] tensor rows."""
    if spec is None:
        return []
    if len(spec) and isinstance(spec[0], (list, tuple)):
        return [[float(c[0]), float(c[1]), float(c[2])] for c in spec]
    return [[float(spec[0]), float(spec[1]), float(spec[2])]]


@register_operator("colony", level="cell", kind="lateral")
class ColonyOperator(Lateral):
    REQUIRES_PARAMS = ["home", "food"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.home = _as_circles(params.get("home"))          # [[cx,cy,r]]
        self.food = _as_circles(params.get("food"))           # [[cx,cy,r], ...]
        self.perception = float(params.get("perception", 0.08))
        self.turn_noise = float(params.get("turn_noise", 0.4))
        self.stock = float(params.get("food_stock", 0.0))     # 0 -> infinite food
        self._home_t = torch.tensor(self.home, device=device).reshape(-1, 3)
        self._food_t = torch.tensor(self.food, device=device).reshape(-1, 3)
        self._remaining = None                                # per-source stock, lazily built

    @staticmethod
    def _inside_any(pos, circ):
        """[N] bool: is each pos inside ANY circle, and [N] index of the nearest circle."""
        d = torch.cdist(pos, circ[:, :2])                     # [N,K] center distance
        inside = d <= circ[:, 2][None, :]                     # within that circle's radius
        nearest = torch.argmin(d, dim=1)                      # [N] nearest circle index
        return inside.any(dim=1), nearest, d

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        dev = pos.device
        N = cell.n
        home = self._home_t.to(dev)
        food = self._food_t.to(dev)
        if mask is None:
            mask = torch.ones(N, dtype=torch.bool, device=dev)

        if not hasattr(cell, "loaded"):
            cell.loaded = torch.zeros(N, dtype=torch.bool, device=dev)
        if not hasattr(cell, "heading"):
            cell.heading = torch.rand(N, generator=H.rng, device=dev) * 2 * math.pi
        if not hasattr(cell, "t_since_goal"):
            cell.t_since_goal = torch.zeros(N, device=dev)
        if self._remaining is None:                           # per-food-source stock (0 -> never depletes)
            self._remaining = torch.full((food.shape[0],), self.stock, device=dev)
        H.food_delivered = getattr(H, "food_delivered", 0)

        loaded = cell.loaded
        searching = (~loaded) & mask
        returning = loaded & mask

        in_food, near_food_idx, dfood = self._inside_any(pos, food)
        in_home, _, dhome = self._inside_any(pos, home)
        # an emptied source no longer counts as food
        if self.stock > 0:
            src_live = self._remaining > 0
            in_food = in_food & src_live[near_food_idx]

        # --- PICK UP: searching ant reaches food -> carry, turn around, deplete ---
        pickup = searching & in_food
        if pickup.any():
            if self.stock > 0:                                # debit one unit from the chosen source
                hit = near_food_idx[pickup]
                self._remaining.index_add_(0, hit, -torch.ones_like(hit, dtype=self._remaining.dtype))
                self._remaining.clamp_(min=0.0)
            cell.loaded = torch.where(pickup, torch.ones_like(loaded), cell.loaded)
            cell.t_since_goal = torch.where(pickup, torch.zeros_like(cell.t_since_goal), cell.t_since_goal)
            cell.heading = torch.where(pickup, self._turn_around(cell.heading, H, dev), cell.heading)

        # --- DROP OFF: returning ant reaches home -> deliver, turn around, search ---
        drop = returning & in_home
        if drop.any():
            H.food_delivered += int(drop.sum())
            cell.loaded = torch.where(drop, torch.zeros_like(loaded), cell.loaded)
            cell.t_since_goal = torch.where(drop, torch.zeros_like(cell.t_since_goal), cell.t_since_goal)
            cell.heading = torch.where(drop, self._turn_around(cell.heading, H, dev), cell.heading)

        # --- PERCEPTION: short-range heading lock toward the relevant target ----
        # searching -> nearest food center; returning -> home center (Lague's perceptionRadius).
        if self.perception > 0:
            tgt_food = food[near_food_idx, :2]
            see_food = (~cell.loaded) & mask & (dfood.gather(1, near_food_idx[:, None]).squeeze(1) < self.perception)
            if self.stock > 0:
                see_food = see_food & (self._remaining > 0)[near_food_idx]
            see_home = cell.loaded & mask & (dhome.min(dim=1).values < self.perception)
            to_food = torch.atan2(tgt_food[:, 1] - pos[:, 1], tgt_food[:, 0] - pos[:, 0])
            to_home = torch.atan2(home[0, 1] - pos[:, 1], home[0, 0] - pos[:, 0])
            cell.heading = torch.where(see_food, to_food, cell.heading)
            cell.heading = torch.where(see_home, to_home, cell.heading)

        # --- recency clock: +1 everywhere it was masked, reset handled above ----
        bumped = mask & ~(pickup | drop)
        cell.t_since_goal = cell.t_since_goal + bumped.float()
        return {}

    def _turn_around(self, th, H, dev):
        """Reverse heading + small random perturbation (Ant.cs StartTurnAround)."""
        noise = (torch.rand(th.shape[0], generator=H.rng, device=dev) - 0.5) * 2 * self.turn_noise
        return th + math.pi + noise
