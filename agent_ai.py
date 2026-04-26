"""
AnomalyCraft Survival — Agent AI.
Uses neural policy when available, falls back to rule-based AI.
"""

import random
from typing import Dict, Any

# ── Try neural policy, fall back to rule-based ──────────────────────────────
try:
    from neural_policy import (
        NeuralPolicy,
        extract_features,
        action_to_command,
        get_brain,
    )
    _USE_NEURAL = True
except ImportError:
    _USE_NEURAL = False

# live policies for current agents (neural mode only)
_policies: Dict[str, Any] = {}


def get_or_create_policy(agent_id: str):
    if agent_id not in _policies:
        _policies[agent_id] = get_brain().spawn_policy()
    return _policies[agent_id]


def on_agent_death(agent_id: str, survival_ticks: int) -> None:
    if not _USE_NEURAL:
        return
    if agent_id in _policies:
        get_brain().record_death(_policies[agent_id], survival_ticks)
        del _policies[agent_id]


def on_generation_end() -> None:
    if _USE_NEURAL:
        get_brain().new_generation()
    _policies.clear()


def get_nn_stats() -> Dict:
    if _USE_NEURAL:
        return get_brain().stats()
    return {"generation": 0, "best_fitness_ever": 0, "gene_pool_size": 0,
            "mutation_rate": 0.15, "avg_fitness_history": []}


# ── Main entry point ─────────────────────────────────────────────────────────

def decide_action(agent: Dict[str, Any], world_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Neural policy when available, otherwise rule-based survival AI.
    """
    if _USE_NEURAL:
        agent_id = agent["agent_id"]
        policy   = get_or_create_policy(agent_id)
        features = extract_features(agent, world_state)
        idx      = policy.choose_action(features)
        return action_to_command(idx, agent, world_state)

    return _rule_based(agent, world_state)


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based(agent: Dict[str, Any], world_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Smart rule-based AI:
    1. Seek loved one always
    2. Fight anomalies if strong enough
    3. Flee to shelter if hurt
    4. Build shelter if none exists
    5. Craft tools
    6. Gather resources
    7. Explore
    """
    agent_id     = agent["agent_id"]
    health       = agent["health"]
    energy       = agent["energy"]
    hunger       = agent["hunger"]
    inventory    = agent.get("inventory", {})
    memory       = agent.get("memory", {})
    personality  = agent.get("personality", "peaceful")
    loved_one    = agent.get("loved_one")
    bond_strength = agent.get("bond_strength", 0.0)

    weather  = world_state.get("weather", "clear")
    is_day   = world_state.get("is_day", True)

    obs              = _observe(agent, world_state)
    available        = obs["available_actions"]
    nearby_anomalies = obs["nearby_anomalies"]
    local_resources  = obs["local_resources"]

    # ── 1. ROMANTIC: drift towards loved one ─────────────────────────────
    if loved_one and bond_strength > 0.3 and energy > 20:
        agents = world_state.get("agents", {})
        if loved_one in agents:
            loved = agents[loved_one]
            if loved["alive"]:
                dist = abs(agent["x"] - loved["x"]) + abs(agent["y"] - loved["y"])
                if 1 < dist <= 8:
                    return _move_towards(agent_id, agent["x"], agent["y"],
                                         loved["x"], loved["y"])

    # ── 2. CRITICAL NEEDS ────────────────────────────────────────────────
    if hunger < 15 and "eat" in available:
        return _act(agent_id, "eat")
    if energy < 8 and "rest" in available:
        return _act(agent_id, "rest")

    # ── 3. FIGHT anomalies if capable ────────────────────────────────────
    if nearby_anomalies and "attack" in available:
        wins   = memory.get("combat_wins", 0)
        losses = memory.get("combat_losses", 0)
        conf   = wins / max(1, wins + losses)
        has_sword = inventory.get("sword", 0) > 0
        if (has_sword and health > 40 and energy > 30) or \
           (personality == "aggressive" and health > 50) or \
           (conf > 0.7 and health > 60):
            return _act(agent_id, "attack")

    # ── 4. FLEE to shelter if hurt ───────────────────────────────────────
    buildings = world_state.get("buildings", [])
    near_shelter, in_shelter = _find_shelter(agent, buildings)

    bad_weather = weather in ("rain", "snow", "blizzard", "heatwave")
    if (nearby_anomalies and health < 50) or (bad_weather and not is_day):
        if in_shelter:
            if "rest" in available and (energy < 90 or health < 90):
                return _act(agent_id, "rest")
            if "eat" in available and hunger < 60:
                return _act(agent_id, "eat")
        elif near_shelter and energy > 5:
            return _move_towards(agent_id, agent["x"], agent["y"],
                                  near_shelter["x"], near_shelter["y"])

    # ── 5. BUILD SHELTER if none exists ──────────────────────────────────
    if not near_shelter:
        if inventory.get("shelter_kit", 0) > 0 and "build" in available:
            return _act(agent_id, "build", "shelter")

        if inventory.get("wood", 0) >= 8 and inventory.get("stone", 0) >= 4 \
                and "craft" in available:
            return _act(agent_id, "craft", "shelter_kit")

        # Gather wood
        if inventory.get("wood", 0) < 8:
            r = _find_resource(agent, local_resources, "wood")
            if r:
                return r if isinstance(r, dict) else _act(agent_id, "gather")

        # Gather stone
        if inventory.get("stone", 0) < 4:
            r = _find_resource(agent, local_resources, "stone")
            if r:
                return r if isinstance(r, dict) else _act(agent_id, "gather")

    # ── 6. CRAFT TOOLS ───────────────────────────────────────────────────
    if "craft" in available:
        if not inventory.get("axe") and inventory.get("wood", 0) >= 2 and inventory.get("stone", 0) >= 3:
            return _act(agent_id, "craft", "axe")
        if not inventory.get("pickaxe") and inventory.get("wood", 0) >= 3 and inventory.get("stone", 0) >= 2:
            return _act(agent_id, "craft", "pickaxe")
        if not inventory.get("sword") and inventory.get("wood", 0) >= 1 and inventory.get("iron", 0) >= 4:
            return _act(agent_id, "craft", "sword")
        if health < 50 and inventory.get("mushroom", 0) >= 3 and inventory.get("berry", 0) >= 2:
            return _act(agent_id, "craft", "healing_potion")

    # ── 7. GATHER ────────────────────────────────────────────────────────
    if "gather" in available and local_resources:
        prefs = memory.get("resource_preference", {})
        if prefs:
            best = max(local_resources.values(),
                       key=lambda r: prefs.get(r, 0.5), default=None)
            if best and prefs.get(best, 0.5) > 0.3:
                return _act(agent_id, "gather")
        else:
            return _act(agent_id, "gather")

    # ── 8. EXPLORE ───────────────────────────────────────────────────────
    if energy > 40:
        danger_mem = memory.get("danger_memory", {})
        dirs = ["up", "down", "left", "right"]
        safe = []
        for d in dirs:
            nx, ny = agent["x"], agent["y"]
            if d == "up":    ny -= 1
            elif d == "down": ny += 1
            elif d == "left": nx -= 1
            else:             nx += 1
            if danger_mem.get(f"{nx},{ny}", 0.0) < 0.5:
                safe.append(d)
        return _act(agent_id, "move", random.choice(safe) if safe else random.choice(dirs))

    # ── 9. REST ──────────────────────────────────────────────────────────
    if "rest" in available:
        return _act(agent_id, "rest")

    return _act(agent_id, "noop")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _act(agent_id, action_type, target=None):
    cmd = {"agent_id": agent_id, "action_type": action_type}
    if target is not None:
        cmd["target"] = target
    return cmd


def _move_towards(agent_id, ax, ay, tx, ty):
    dx, dy = tx - ax, ty - ay
    if abs(dx) >= abs(dy):
        direction = "right" if dx > 0 else "left"
    else:
        direction = "down" if dy > 0 else "up"
    return _act(agent_id, "move", direction)


def _find_shelter(agent, buildings):
    near_shelter = None
    in_shelter   = False
    for b in buildings:
        if b["type"] == "shelter":
            dist = abs(agent["x"] - b["x"]) + abs(agent["y"] - b["y"])
            if dist <= 1:
                in_shelter   = True
                near_shelter = b
                break
            elif dist <= 10:
                near_shelter = b
    return near_shelter, in_shelter


def _find_resource(agent, local_resources, rtype):
    """Return a move-towards or gather action for the nearest rtype tile."""
    for coord, res in local_resources.items():
        if res != rtype:
            continue
        rx, ry = map(int, coord.split(","))
        if agent["x"] == rx and agent["y"] == ry:
            return {"agent_id": agent["agent_id"], "action_type": "gather"}
        return _move_towards(agent["agent_id"], agent["x"], agent["y"], rx, ry)
    return None


def _observe(agent: Dict[str, Any], world_state: Dict[str, Any]) -> Dict:
    x, y = agent["x"], agent["y"]

    local_resources = {
        coord: rtype
        for coord, rtype in world_state.get("global_resources", {}).items()
        if abs(x - int(coord.split(",")[0])) <= 3
        and abs(y - int(coord.split(",")[1])) <= 3
    }

    nearby_anomalies = [
        a for a in world_state.get("anomalies", [])
        if abs(x - a["x"]) <= 4 and abs(y - a["y"]) <= 4
    ]

    inv = agent.get("inventory", {})
    actions = ["move", "rest", "noop", "eat", "attack",
               "form_community", "join_community", "share"]

    if f"{x},{y}" in world_state.get("global_resources", {}):
        actions.append("gather")
    if any(inv.get(r, 0) >= 2 for r in ("wood", "stone", "iron")):
        actions.append("craft")
    if any(inv.get(f"{t}_kit", 0) > 0 for t in ("shelter", "farm", "wall")):
        actions.append("build")

    return {
        "local_resources":  local_resources,
        "nearby_anomalies": nearby_anomalies,
        "available_actions": actions,
    }


# keep old name for compatibility
get_agent_observation = _observe
