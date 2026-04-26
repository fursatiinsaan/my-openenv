"""
Neural Network Policy for AnomalyCraft Survival agents.

Each agent carries a small 2-layer feedforward network (numpy only).
Architecture: 22 inputs → 16 hidden → 9 action logits

Training via neuroevolution:
  - Fitness = survival ticks
  - On death: best weights saved to collective memory
  - New agents inherit best weights + gaussian mutation
  - Over generations, policies that survive longer dominate

Actions (9):
  0  move_away_from_danger
  1  move_toward_resource
  2  move_toward_loved_one
  3  move_random
  4  gather
  5  craft_or_build
  6  fight
  7  eat_or_rest
  8  community  (form / join / share)
"""

import random
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# ── Network dimensions ──────────────────────────────────────────────────────
INPUT_DIM  = 22
HIDDEN_DIM = 16
OUTPUT_DIM = 9

ACTION_NAMES = [
    "move_away_danger",   # 0
    "move_to_resource",   # 1
    "move_to_loved_one",  # 2
    "move_random",        # 3
    "gather",             # 4
    "craft_or_build",     # 5
    "fight",              # 6
    "eat_or_rest",        # 7
    "community",          # 8  form or join a community
]

MUTATION_RATE  = 0.05   # initial std dev of gaussian noise
MUTATION_DECAY = 0.98   # decay per generation
MUTATION_MIN   = 0.02   # floor — never go below this
MUTATION_RESET = 0.10   # reset to this on stagnation
STAGNATION_GENS = 8     # gens without improvement before reset


# ── Tiny neural net ─────────────────────────────────────────────────────────

class NeuralPolicy:
    """
    2-layer feedforward net.
    Weights stored as flat numpy arrays for easy serialisation / mutation.
    """

    def __init__(self, w1: np.ndarray = None, b1: np.ndarray = None,
                 w2: np.ndarray = None, b2: np.ndarray = None):
        if w1 is None:
            # Xavier initialisation
            scale1 = math.sqrt(2.0 / INPUT_DIM)
            scale2 = math.sqrt(2.0 / HIDDEN_DIM)
            self.w1 = np.random.randn(INPUT_DIM,  HIDDEN_DIM) * scale1
            self.b1 = np.zeros(HIDDEN_DIM)
            self.w2 = np.random.randn(HIDDEN_DIM, OUTPUT_DIM) * scale2
            self.b2 = np.zeros(OUTPUT_DIM)
        else:
            self.w1 = w1.copy()
            self.b1 = b1.copy()
            self.w2 = w2.copy()
            self.b2 = b2.copy()

    # ── forward pass ──
    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1 + self.b1)           # (HIDDEN,)
        logits = h @ self.w2 + self.b2                # (OUTPUT,)
        return logits

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def choose_action(self, x: np.ndarray) -> int:
        """Sample action from softmax distribution."""
        logits = self.forward(x)
        probs  = self.softmax(logits)
        return int(np.random.choice(OUTPUT_DIM, p=probs))

    def best_action(self, x: np.ndarray) -> int:
        """Greedy action (used for exploitation)."""
        return int(np.argmax(self.forward(x)))

    # ── evolution ──
    def mutate(self, rate: float = MUTATION_RATE) -> "NeuralPolicy":
        """Return a mutated child policy."""
        return NeuralPolicy(
            w1=self.w1 + np.random.randn(*self.w1.shape) * rate,
            b1=self.b1 + np.random.randn(*self.b1.shape) * rate,
            w2=self.w2 + np.random.randn(*self.w2.shape) * rate,
            b2=self.b2 + np.random.randn(*self.b2.shape) * rate,
        )

    @staticmethod
    def crossover(a: "NeuralPolicy", b: "NeuralPolicy") -> "NeuralPolicy":
        """Uniform crossover between two parent policies."""
        def blend(x, y):
            mask = np.random.rand(*x.shape) > 0.5
            return np.where(mask, x, y)
        return NeuralPolicy(
            w1=blend(a.w1, b.w1),
            b1=blend(a.b1, b.b1),
            w2=blend(a.w2, b.w2),
            b2=blend(a.b2, b.b2),
        )

    # ── serialisation ──
    def to_dict(self) -> Dict:
        return {
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }

    @staticmethod
    def from_dict(d: Dict) -> "NeuralPolicy":
        return NeuralPolicy(
            w1=np.array(d["w1"]),
            b1=np.array(d["b1"]),
            w2=np.array(d["w2"]),
            b2=np.array(d["b2"]),
        )


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features(agent: Dict[str, Any], world_state: Dict[str, Any]) -> np.ndarray:
    """
    Build the 22-dim input vector from agent + world state.

    Index  Feature
    ─────  ───────────────────────────────────────────────
    0      health          (0-1)
    1      energy          (0-1)
    2      hunger          (0-1)
    3      anomaly_count   (0-1, capped at 5)
    4      nearest_anomaly_dist  (0-1, 0=touching, 1=far)
    5      resource_count_nearby (0-1, capped at 10)
    6      has_wood        (0-1)
    7      has_stone       (0-1)
    8      has_iron        (0-1)
    9      has_food        (0-1)
    10     has_sword       (0-1)
    11     has_shelter_kit (0-1)
    12     near_shelter    (0-1)
    13     in_shelter      (0-1)
    14     loved_one_dist  (0-1, 0=touching, 1=far/none)
    15     bond_strength   (0-1)
    16     is_day          (0-1)
    17     weather_bad     (0-1)
    18     wood_needed     (0-1)  — wood < 8
    19     stone_needed    (0-1)  — stone < 4
    20     health_low      (0-1)  — health < 40
    21     combat_confidence (0-1)
    """
    inv     = agent.get("inventory", {})
    memory  = agent.get("memory", {})
    weather = world_state.get("weather", "clear")

    # Anomalies
    ax, ay = agent["x"], agent["y"]
    anomalies = world_state.get("anomalies", [])
    ano_count = min(len(anomalies), 5) / 5.0
    if anomalies:
        dists = [abs(ax - a["x"]) + abs(ay - a["y"]) for a in anomalies]
        nearest_ano = min(dists) / 48.0
    else:
        nearest_ano = 1.0

    # Resources nearby (within 4 tiles)
    resources = world_state.get("global_resources", {})
    nearby_res = sum(
        1 for coord in resources
        if abs(ax - int(coord.split(",")[0])) <= 4
        and abs(ay - int(coord.split(",")[1])) <= 4
    )
    res_count = min(nearby_res, 10) / 10.0

    # Shelter
    buildings = world_state.get("buildings", [])
    near_shelter = 0.0
    in_shelter   = 0.0
    for b in buildings:
        if b["type"] == "shelter":
            d = abs(ax - b["x"]) + abs(ay - b["y"])
            if d <= 1:
                in_shelter   = 1.0
                near_shelter = 1.0
                break
            elif d <= 10:
                near_shelter = 1.0

    # Loved one
    loved_id = agent.get("loved_one")
    bond     = agent.get("bond_strength", 0.0)
    if loved_id:
        agents = world_state.get("agents", {})
        if loved_id in agents and agents[loved_id].get("alive", False):
            ld = agents[loved_id]
            loved_dist = min(abs(ax - ld["x"]) + abs(ay - ld["y"]), 48) / 48.0
        else:
            loved_dist = 1.0
    else:
        loved_dist = 1.0

    # Combat confidence
    wins   = memory.get("combat_wins", 0)
    losses = memory.get("combat_losses", 0)
    combat_conf = wins / max(1, wins + losses)

    feat = np.array([
        agent["health"]  / 100.0,
        agent["energy"]  / 100.0,
        agent["hunger"]  / 100.0,
        ano_count,
        nearest_ano,
        res_count,
        min(inv.get("wood",  0), 10) / 10.0,
        min(inv.get("stone", 0), 10) / 10.0,
        min(inv.get("iron",  0), 10) / 10.0,
        1.0 if (inv.get("berry", 0) + inv.get("mushroom", 0)) > 0 else 0.0,
        1.0 if inv.get("sword", 0) > 0 else 0.0,
        1.0 if inv.get("shelter_kit", 0) > 0 else 0.0,
        near_shelter,
        in_shelter,
        loved_dist,
        bond,
        1.0 if world_state.get("is_day", True) else 0.0,
        1.0 if weather in ("rain", "snow", "blizzard", "heatwave") else 0.0,
        1.0 if inv.get("wood",  0) < 8 else 0.0,
        1.0 if inv.get("stone", 0) < 4 else 0.0,
        1.0 if agent["health"] < 40 else 0.0,
        combat_conf,
    ], dtype=np.float32)

    return feat


# ── Action translation ───────────────────────────────────────────────────────

def action_to_command(
    action_idx: int,
    agent: Dict[str, Any],
    world_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Translate a network output index into a concrete game action dict.
    Falls back gracefully when the chosen action isn't possible.
    """
    agent_id = agent["agent_id"]
    ax, ay   = agent["x"], agent["y"]
    inv      = agent.get("inventory", {})
    resources = world_state.get("global_resources", {})
    anomalies = world_state.get("anomalies", [])
    buildings = world_state.get("buildings", [])

    def _move(dx, dy):
        dirs = {(0,-1):"up",(0,1):"down",(-1,0):"left",(1,0):"right"}
        return {"agent_id": agent_id, "action_type": "move",
                "target": dirs.get((dx, dy), "up")}

    def _nearest_resource():
        # If no shelter nearby, prioritize wood for building
        buildings = world_state.get("buildings", [])
        has_shelter = any(
            b["type"] == "shelter" and abs(ax - b["x"]) + abs(ay - b["y"]) <= 12
            for b in buildings
        )
        need_wood = not has_shelter and inv.get("wood", 0) < 8

        best, bd = None, 999
        for coord, rtype in resources.items():
            rx, ry = map(int, coord.split(","))
            d = abs(ax - rx) + abs(ay - ry)
            # Heavily prefer wood when building shelter
            if need_wood and rtype == "wood":
                d = d * 0.3  # make wood appear 3x closer
            if d < bd:
                bd, best = d, (rx, ry, rtype)
        return best

    def _nearest_anomaly():
        best, bd = None, 999
        for a in anomalies:
            d = abs(ax - a["x"]) + abs(ay - a["y"])
            if d < bd:
                bd, best = d, a
        return best

    def _dir_toward(tx, ty):
        dx, dy = tx - ax, ty - ay
        if abs(dx) >= abs(dy):
            return (1 if dx > 0 else -1, 0)
        return (0, 1 if dy > 0 else -1)

    def _dir_away(tx, ty):
        dx, dy = _dir_toward(tx, ty)
        return (-dx, -dy)

    # ── 0: move away from danger ──
    if action_idx == 0:
        ano = _nearest_anomaly()
        if ano:
            dx, dy = _dir_away(ano["x"], ano["y"])
            return _move(dx, dy)
        return _move(random.choice([-1,1]), 0)

    # ── 1: move toward nearest resource ──
    if action_idx == 1:
        res = _nearest_resource()
        if res:
            rx, ry, _ = res
            if rx == ax and ry == ay:
                return {"agent_id": agent_id, "action_type": "gather"}
            dx, dy = _dir_toward(rx, ry)
            return _move(dx, dy)
        return _move(random.choice([-1,1]), 0)

    # ── 2: move toward loved one ──
    if action_idx == 2:
        loved_id = agent.get("loved_one")
        if loved_id:
            agents = world_state.get("agents", {})
            if loved_id in agents and agents[loved_id].get("alive", False):
                ld = agents[loved_id]
                if abs(ax - ld["x"]) + abs(ay - ld["y"]) <= 1:
                    # Already adjacent — rest together
                    return {"agent_id": agent_id, "action_type": "rest"}
                dx, dy = _dir_toward(ld["x"], ld["y"])
                return _move(dx, dy)
        # No loved one — explore
        return _move(random.choice([-1,0,1]), random.choice([-1,0,1]) or 1)

    # ── 3: move random ──
    if action_idx == 3:
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        dx, dy = random.choice(dirs)
        return _move(dx, dy)

    # ── 4: gather ──
    if action_idx == 4:
        if (f"{ax},{ay}") in resources:
            return {"agent_id": agent_id, "action_type": "gather"}
        # Move toward nearest resource instead
        res = _nearest_resource()
        if res:
            rx, ry, _ = res
            dx, dy = _dir_toward(rx, ry)
            return _move(dx, dy)
        return {"agent_id": agent_id, "action_type": "noop"}

    # ── 5: craft or build ──
    if action_idx == 5:
        # Build if we have a kit
        for kit_type in ("shelter", "farm", "wall"):
            if inv.get(f"{kit_type}_kit", 0) > 0:
                return {"agent_id": agent_id, "action_type": "build", "target": kit_type}
        
        # Check if there's a shelter nearby
        buildings = world_state.get("buildings", [])
        has_shelter = any(
            b["type"] == "shelter" and abs(ax - b["x"]) + abs(ay - b["y"]) <= 10
            for b in buildings
        )
        
        # No shelter — prioritize getting wood first
        if not has_shelter:
            if inv.get("wood", 0) >= 8 and inv.get("stone", 0) >= 4:
                return {"agent_id": agent_id, "action_type": "craft", "target": "shelter_kit"}
            # Move toward nearest wood resource
            for coord, rtype in resources.items():
                if rtype == "wood":
                    rx, ry = map(int, coord.split(","))
                    if ax == rx and ay == ry:
                        return {"agent_id": agent_id, "action_type": "gather"}
                    dx, dy = _dir_toward(rx, ry)
                    return _move(dx, dy)
        
        # Craft shelter kit if materials ready
        if inv.get("wood", 0) >= 8 and inv.get("stone", 0) >= 4:
            return {"agent_id": agent_id, "action_type": "craft", "target": "shelter_kit"}
        # Craft axe
        if inv.get("wood", 0) >= 2 and inv.get("stone", 0) >= 3 and not inv.get("axe", 0):
            return {"agent_id": agent_id, "action_type": "craft", "target": "axe"}
        # Craft pickaxe
        if inv.get("wood", 0) >= 3 and inv.get("stone", 0) >= 2 and not inv.get("pickaxe", 0):
            return {"agent_id": agent_id, "action_type": "craft", "target": "pickaxe"}
        # Craft sword
        if inv.get("wood", 0) >= 1 and inv.get("iron", 0) >= 4 and not inv.get("sword", 0):
            return {"agent_id": agent_id, "action_type": "craft", "target": "sword"}
        # Craft healing potion
        if inv.get("mushroom", 0) >= 3 and inv.get("berry", 0) >= 2:
            return {"agent_id": agent_id, "action_type": "craft", "target": "healing_potion"}
        # Nothing to craft — gather instead
        return {"agent_id": agent_id, "action_type": "gather"} \
            if f"{ax},{ay}" in resources else \
            {"agent_id": agent_id, "action_type": "noop"}

    # ── 6: fight ──
    if action_idx == 6:
        for ano in anomalies:
            if abs(ax - ano["x"]) + abs(ay - ano["y"]) <= 1:
                return {"agent_id": agent_id, "action_type": "attack"}
        # Move toward nearest anomaly to fight
        ano = _nearest_anomaly()
        if ano:
            dx, dy = _dir_toward(ano["x"], ano["y"])
            return _move(dx, dy)
        return {"agent_id": agent_id, "action_type": "noop"}

    # ── 7: eat or rest ──
    if action_idx == 7:
        if inv.get("berry", 0) > 0 or inv.get("mushroom", 0) > 0 or inv.get("healing_potion", 0) > 0:
            return {"agent_id": agent_id, "action_type": "eat"}
        return {"agent_id": agent_id, "action_type": "rest"}

    # ── 8: community — form or join ──
    if action_idx == 8:
        communities = world_state.get("communities", {})
        # Join nearest community if not already in one
        if not agent.get("community_id"):
            best_cid = None
            best_dist = 999
            for cid, comm in communities.items():
                dist = abs(ax - comm["territory_x"]) + abs(ay - comm["territory_y"])
                if dist < best_dist and len(comm["members"]) < 12:
                    best_dist = dist
                    best_cid = cid
            if best_cid and best_dist <= 2:
                return {"agent_id": agent_id, "action_type": "join_community", "target": best_cid}
            elif best_cid and best_dist <= 8:
                # Move towards community
                comm = communities[best_cid]
                dx, dy = _dir_toward(comm["territory_x"], comm["territory_y"])
                return _move(dx, dy)
            else:
                # No community nearby — found one
                return {"agent_id": agent_id, "action_type": "form_community"}
        # Already in community — share resources
        for res in ("wood", "stone", "berry", "mushroom"):
            if inv.get(res, 0) > 4:
                return {"agent_id": agent_id, "action_type": "share", "target": res}
        return {"agent_id": agent_id, "action_type": "noop"}


# ── Collective brain (persists across generations) ───────────────────────────

class CollectiveBrain:
    """
    Stores the best-performing neural policies across all generations.
    New agents inherit from the gene pool.

    Improvements over naive top-K:
    - Separate elite pool (top-20) from diversity pool (20 random recent)
    - Stagnation detection: if best fitness doesn't improve for N gens, reset mutation
    - Generation avg tracks actual per-gen performance, not pool avg
    - spawn_policy samples from full pool with fitness-proportionate selection
    """

    def __init__(self):
        self.elite_pool: List[Tuple[float, Dict]]   = []  # top-10 all time
        self.recent_pool: List[Tuple[float, Dict]]  = []  # last 10 gens, any fitness
        self.generation = 0
        self.mutation_rate = MUTATION_RATE
        self.best_fitness_ever = 0.0
        self.best_fitness_last_reset = 0.0
        self.gens_since_improvement = 0
        self.avg_fitness_history: List[float] = []   # per-generation avg
        self._gen_fitnesses: List[float] = []         # collected this generation

    @property
    def gene_pool(self):
        """Combined pool for backwards compat (used by stats/save/load)."""
        seen = set()
        combined = []
        for f, w in self.elite_pool + self.recent_pool:
            key = id(w)  # use object identity
            combined.append((f, w))
        # deduplicate by fitness value (approximate)
        combined.sort(key=lambda x: x[0], reverse=True)
        return combined[:20]

    @gene_pool.setter
    def gene_pool(self, val):
        """Allow external code to set the pool (used by weight loading)."""
        self.elite_pool = val[:20]
        self.recent_pool = val[20:40] if len(val) > 20 else val[20:]

    def record_death(self, policy: NeuralPolicy, fitness: float):
        """Called when an agent dies."""
        fitness = float(fitness)
        entry = (fitness, policy.to_dict())

        # Always add to recent pool (keeps diversity)
        self.recent_pool.append(entry)
        if len(self.recent_pool) > 10:
            # Drop lowest from recent
            self.recent_pool.sort(key=lambda x: x[0], reverse=True)
            self.recent_pool = self.recent_pool[:10]

        # Add to elite pool if it beats the worst elite
        self.elite_pool.append(entry)
        self.elite_pool.sort(key=lambda x: x[0], reverse=True)
        self.elite_pool = self.elite_pool[:20]

        if fitness > self.best_fitness_ever:
            self.best_fitness_ever = fitness

        self._gen_fitnesses.append(fitness)

    def new_generation(self) -> None:
        """Called at end of each generation."""
        if self._gen_fitnesses:
            gen_avg = sum(self._gen_fitnesses) / len(self._gen_fitnesses)
            self.avg_fitness_history.append(round(gen_avg))
            self._gen_fitnesses = []

        self.generation += 1

        # Stagnation detection
        if self.best_fitness_ever > self.best_fitness_last_reset + 50:
            self.gens_since_improvement = 0
            self.best_fitness_last_reset = self.best_fitness_ever
        else:
            self.gens_since_improvement += 1

        if self.gens_since_improvement >= STAGNATION_GENS:
            # Reset mutation rate to re-explore
            self.mutation_rate = MUTATION_RESET
            self.gens_since_improvement = 0
        else:
            self.mutation_rate = max(MUTATION_MIN, self.mutation_rate * MUTATION_DECAY)

    def spawn_policy(self) -> NeuralPolicy:
        """
        Create a policy for a new agent.
        70% chance: crossover from elite pool
        20% chance: crossover from recent pool (diversity)
        10% chance: fresh random (exploration)
        """
        r = random.random()

        if r < 0.10 or (not self.elite_pool and not self.recent_pool):
            # Fresh random
            return NeuralPolicy()

        pool = self.elite_pool if (r < 0.70 or not self.recent_pool) else self.recent_pool
        if not pool:
            return NeuralPolicy()

        if len(pool) == 1:
            parent = NeuralPolicy.from_dict(pool[0][1])
            return parent.mutate(self.mutation_rate)

        # Fitness-proportionate selection
        fitnesses = np.array([f for f, _ in pool], dtype=np.float64)
        fitnesses = fitnesses - fitnesses.min() + 1e-6
        probs = fitnesses / fitnesses.sum()
        idx_a, idx_b = np.random.choice(len(pool), size=2, replace=False, p=probs)
        pa = NeuralPolicy.from_dict(pool[idx_a][1])
        pb = NeuralPolicy.from_dict(pool[idx_b][1])
        child = NeuralPolicy.crossover(pa, pb)
        return child.mutate(self.mutation_rate)

    def stats(self) -> Dict:
        hist = self.avg_fitness_history[-10:]
        trend = "→"
        if len(hist) >= 2:
            trend = "↑" if hist[-1] > hist[-2] else ("↓" if hist[-1] < hist[-2] else "→")
        return {
            "generation":          self.generation,
            "gene_pool_size":      len(self.elite_pool) + len(self.recent_pool),
            "best_fitness_ever":   round(self.best_fitness_ever),
            "mutation_rate":       round(self.mutation_rate, 4),
            "avg_fitness_history": hist,
            "top_fitness":         round(self.elite_pool[0][0]) if self.elite_pool else 0,
            "trend":               trend,
            "stagnation":          self.gens_since_improvement,
        }


# ── Global brain singleton ───────────────────────────────────────────────────
_brain = CollectiveBrain()

def get_brain() -> CollectiveBrain:
    return _brain


# ════════════════════════════════════════════════════════════════════════════
# ANOMALY NEURAL POLICY
# Anomalies evolve hunting strategies just like agents evolve survival ones.
# Fitness = total damage dealt before being destroyed (or lifetime if never killed).
# ════════════════════════════════════════════════════════════════════════════

ANO_INPUT_DIM  = 12
ANO_HIDDEN_DIM = 10
ANO_OUTPUT_DIM = 5   # chase, flank_left, flank_right, retreat_and_grow, spread

ANO_ACTION_NAMES = [
    "chase",          # 0 — move directly toward nearest agent
    "flank_left",     # 1 — circle left around nearest agent
    "flank_right",    # 2 — circle right around nearest agent
    "retreat_grow",   # 3 — move away, grow severity faster
    "spread",         # 4 — move toward weakest agent instead of nearest
]


class AnomalyPolicy:
    """
    Small net for anomaly hunting behaviour.
    Same neuroevolution approach as NeuralPolicy.
    """

    def __init__(self, w1=None, b1=None, w2=None, b2=None):
        if w1 is None:
            s1 = math.sqrt(2.0 / ANO_INPUT_DIM)
            s2 = math.sqrt(2.0 / ANO_HIDDEN_DIM)
            self.w1 = np.random.randn(ANO_INPUT_DIM,  ANO_HIDDEN_DIM) * s1
            self.b1 = np.zeros(ANO_HIDDEN_DIM)
            self.w2 = np.random.randn(ANO_HIDDEN_DIM, ANO_OUTPUT_DIM) * s2
            self.b2 = np.zeros(ANO_OUTPUT_DIM)
        else:
            self.w1 = w1.copy(); self.b1 = b1.copy()
            self.w2 = w2.copy(); self.b2 = b2.copy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def choose_action(self, x: np.ndarray) -> int:
        logits = self.forward(x)
        probs  = self.softmax(logits)
        return int(np.random.choice(ANO_OUTPUT_DIM, p=probs))

    def mutate(self, rate: float = 0.15) -> "AnomalyPolicy":
        return AnomalyPolicy(
            w1=self.w1 + np.random.randn(*self.w1.shape) * rate,
            b1=self.b1 + np.random.randn(*self.b1.shape) * rate,
            w2=self.w2 + np.random.randn(*self.w2.shape) * rate,
            b2=self.b2 + np.random.randn(*self.b2.shape) * rate,
        )

    @staticmethod
    def crossover(a: "AnomalyPolicy", b: "AnomalyPolicy") -> "AnomalyPolicy":
        def blend(x, y):
            mask = np.random.rand(*x.shape) > 0.5
            return np.where(mask, x, y)
        return AnomalyPolicy(
            w1=blend(a.w1, b.w1), b1=blend(a.b1, b.b1),
            w2=blend(a.w2, b.w2), b2=blend(a.b2, b.b2),
        )

    def to_dict(self) -> Dict:
        return {"w1": self.w1.tolist(), "b1": self.b1.tolist(),
                "w2": self.w2.tolist(), "b2": self.b2.tolist()}

    @staticmethod
    def from_dict(d: Dict) -> "AnomalyPolicy":
        return AnomalyPolicy(
            w1=np.array(d["w1"]), b1=np.array(d["b1"]),
            w2=np.array(d["w2"]), b2=np.array(d["b2"]),
        )


def extract_anomaly_features(ano: Dict[str, Any], agents: List[Dict[str, Any]]) -> np.ndarray:
    """
    12-dim feature vector for an anomaly.

    Index  Feature
    ─────  ──────────────────────────────────────────
    0      severity            (0-1, capped at 10)
    1      nearest_agent_dist  (0-1)
    2      nearest_agent_dx    (-1 to 1, normalised)
    3      nearest_agent_dy    (-1 to 1, normalised)
    4      nearest_agent_health (0-1)
    5      nearest_agent_has_sword (0/1)
    6      nearest_agent_has_shield (0/1)
    7      weakest_agent_dist  (0-1)
    8      weakest_agent_dx    (-1 to 1)
    9      weakest_agent_dy    (-1 to 1)
    10     agent_count_nearby  (0-1, within 6 tiles, capped at 5)
    11     is_day              (0/1)  — anomalies are stronger at night
    """
    ax, ay = ano["x"], ano["y"]
    alive = [a for a in agents if a.get("alive", True)]

    if not alive:
        return np.zeros(ANO_INPUT_DIM, dtype=np.float32)

    # Nearest agent
    dists = [(abs(ax - a["x"]) + abs(ay - a["y"]), a) for a in alive]
    dists.sort(key=lambda t: t[0])
    near_d, near_a = dists[0]

    # Weakest agent
    weakest = min(alive, key=lambda a: a.get("health", 100))
    wd = abs(ax - weakest["x"]) + abs(ay - weakest["y"])

    # Nearby count (within 6)
    nearby_count = sum(1 for d, _ in dists if d <= 6)

    near_inv = near_a.get("inventory", {})

    feat = np.array([
        min(ano["severity"], 10) / 10.0,
        min(near_d, 48) / 48.0,
        (near_a["x"] - ax) / 48.0,
        (near_a["y"] - ay) / 48.0,
        near_a.get("health", 100) / 100.0,
        1.0 if near_inv.get("sword", 0) > 0 else 0.0,
        1.0 if near_inv.get("shield", 0) > 0 else 0.0,
        min(wd, 48) / 48.0,
        (weakest["x"] - ax) / 48.0,
        (weakest["y"] - ay) / 48.0,
        min(nearby_count, 5) / 5.0,
        ano.get("is_day", 1.0),
    ], dtype=np.float32)

    return feat


def anomaly_action_to_move(
    action_idx: int,
    ano: Dict[str, Any],
    agents: List[Dict[str, Any]],
    world_width: int = 48,
    world_height: int = 48,
) -> Tuple[int, int]:
    """
    Translate anomaly action index into (dx, dy) movement.
    Returns (0, 0) for retreat_grow (stays put, grows faster instead).
    """
    ax, ay = ano["x"], ano["y"]
    alive = [a for a in agents if a.get("alive", True)]
    if not alive:
        return (0, 0)

    def _toward(tx, ty):
        dx, dy = tx - ax, ty - ay
        if abs(dx) >= abs(dy):
            return (1 if dx > 0 else -1, 0)
        return (0, 1 if dy > 0 else -1)

    def _clamp(x, y):
        return (max(0, min(world_width - 1, x)),
                max(0, min(world_height - 1, y)))

    # Nearest agent
    nearest = min(alive, key=lambda a: abs(ax - a["x"]) + abs(ay - a["y"]))
    # Weakest agent
    weakest = min(alive, key=lambda a: a.get("health", 100))

    if action_idx == 0:   # chase nearest
        return _toward(nearest["x"], nearest["y"])

    elif action_idx == 1:  # flank left (perpendicular-left of chase vector)
        dx, dy = _toward(nearest["x"], nearest["y"])
        return (-dy, dx)   # rotate 90° left

    elif action_idx == 2:  # flank right
        dx, dy = _toward(nearest["x"], nearest["y"])
        return (dy, -dx)   # rotate 90° right

    elif action_idx == 3:  # retreat and grow — signal via (99, 99)
        dx, dy = _toward(nearest["x"], nearest["y"])
        return (-dx, -dy)  # move away

    elif action_idx == 4:  # spread — target weakest
        return _toward(weakest["x"], weakest["y"])

    return (0, 0)


class AnomalyBrain:
    """
    Gene pool for anomaly hunting policies.
    Fitness = total damage dealt.
    Same elite+recent+stagnation mechanics as CollectiveBrain.
    """

    def __init__(self):
        self.elite_pool:  List[Tuple[float, Dict]] = []
        self.recent_pool: List[Tuple[float, Dict]] = []
        self.generation = 0
        self.mutation_rate = 0.15
        self.best_fitness_ever = 0.0
        self.best_fitness_last_reset = 0.0
        self.gens_since_improvement = 0
        self.avg_fitness_history: List[float] = []
        self._gen_fitnesses: List[float] = []

    @property
    def gene_pool(self):
        combined = self.elite_pool + self.recent_pool
        combined.sort(key=lambda x: x[0], reverse=True)
        return combined[:20]

    @gene_pool.setter
    def gene_pool(self, val):
        self.elite_pool  = val[:10]
        self.recent_pool = val[10:]

    def record_death(self, policy: AnomalyPolicy, damage_dealt: float):
        fitness = float(damage_dealt)
        entry = (fitness, policy.to_dict())
        self.recent_pool.append(entry)
        if len(self.recent_pool) > 20:
            self.recent_pool.sort(key=lambda x: x[0], reverse=True)
            self.recent_pool = self.recent_pool[:20]
        self.elite_pool.append(entry)
        self.elite_pool.sort(key=lambda x: x[0], reverse=True)
        self.elite_pool = self.elite_pool[:10]
        if fitness > self.best_fitness_ever:
            self.best_fitness_ever = fitness
        self._gen_fitnesses.append(fitness)

    def new_generation(self):
        if self._gen_fitnesses:
            self.avg_fitness_history.append(round(sum(self._gen_fitnesses) / len(self._gen_fitnesses), 1))
            self._gen_fitnesses = []
        self.generation += 1
        if self.best_fitness_ever > self.best_fitness_last_reset + 10:
            self.gens_since_improvement = 0
            self.best_fitness_last_reset = self.best_fitness_ever
        else:
            self.gens_since_improvement += 1
        if self.gens_since_improvement >= STAGNATION_GENS:
            self.mutation_rate = MUTATION_RESET
            self.gens_since_improvement = 0
        else:
            self.mutation_rate = max(MUTATION_MIN, self.mutation_rate * MUTATION_DECAY)

    def spawn_policy(self) -> AnomalyPolicy:
        r = random.random()
        if r < 0.10 or (not self.elite_pool and not self.recent_pool):
            return AnomalyPolicy()
        pool = self.elite_pool if (r < 0.70 or not self.recent_pool) else self.recent_pool
        if not pool:
            return AnomalyPolicy()
        if len(pool) == 1:
            return AnomalyPolicy.from_dict(pool[0][1]).mutate(self.mutation_rate)
        fitnesses = np.array([f for f, _ in pool], dtype=np.float64)
        fitnesses = fitnesses - fitnesses.min() + 1e-6
        probs = fitnesses / fitnesses.sum()
        idx_a, idx_b = np.random.choice(len(pool), size=2, replace=False, p=probs)
        pa = AnomalyPolicy.from_dict(pool[idx_a][1])
        pb = AnomalyPolicy.from_dict(pool[idx_b][1])
        return AnomalyPolicy.crossover(pa, pb).mutate(self.mutation_rate)

    def stats(self) -> Dict:
        hist = self.avg_fitness_history[-10:]
        trend = "→"
        if len(hist) >= 2:
            trend = "↑" if hist[-1] > hist[-2] else ("↓" if hist[-1] < hist[-2] else "→")
        return {
            "generation":          self.generation,
            "gene_pool_size":      len(self.elite_pool) + len(self.recent_pool),
            "best_fitness_ever":   round(self.best_fitness_ever, 1),
            "mutation_rate":       round(self.mutation_rate, 4),
            "avg_fitness_history": hist,
            "top_fitness":         round(self.elite_pool[0][0], 1) if self.elite_pool else 0,
            "trend":               trend,
            "stagnation":          self.gens_since_improvement,
        }


# Global anomaly brain singleton
_anomaly_brain = AnomalyBrain()

def get_anomaly_brain() -> AnomalyBrain:
    return _anomaly_brain
