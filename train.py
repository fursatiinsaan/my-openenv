"""
Headless trainer for AnomalyCraft Survival.

Runs the simulation at full speed (no HTTP, no UI) to evolve both
agent and anomaly neural policies via neuroevolution.

Best weights are saved to weights.json and auto-loaded by app.py
so agents start pre-trained from tick 1.

Usage:
    python3 train.py                    # 50 generations, 6 agents
    python3 train.py --gens 200         # more generations
    python3 train.py --gens 100 --ticks 2000  # longer episodes
    python3 train.py --resume           # continue from saved weights
"""

import argparse
import json
import os
import sys
import time
import random

# ── Parse args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train AnomalyCraft agents offline")
parser.add_argument("--gens",   type=int, default=50,   help="Number of generations to train")
parser.add_argument("--ticks",  type=int, default=1500, help="Max ticks per generation")
parser.add_argument("--agents", type=int, default=6,    help="Agents per generation")
parser.add_argument("--resume", action="store_true",    help="Resume from saved weights.json")
parser.add_argument("--weights", type=str, default="weights.json", help="Weights file path")
args = parser.parse_args()

WEIGHTS_FILE = args.weights

# ── Imports ──────────────────────────────────────────────────────────────────
from survival_world import SurvivalWorld, Traits
from neural_policy import (
    NeuralPolicy, CollectiveBrain,
    AnomalyPolicy, AnomalyBrain,
    extract_features, action_to_command,
    extract_anomaly_features, anomaly_action_to_move,
    get_brain, get_anomaly_brain,
)

# ── Load existing weights if resuming ────────────────────────────────────────
def load_weights(path: str):
    if not os.path.exists(path):
        print(f"  No weights file found at {path}, starting fresh.")
        return
    with open(path) as f:
        data = json.load(f)

    agent_brain = get_brain()
    ano_brain   = get_anomaly_brain()

    if "agents" in data:
        for entry in data["agents"]:
            policy = NeuralPolicy.from_dict(entry["weights"])
            agent_brain.gene_pool.append((entry["fitness"], policy.to_dict()))
        agent_brain.gene_pool.sort(key=lambda x: x[0], reverse=True)
        agent_brain.generation    = data.get("agent_generation", 0)
        agent_brain.mutation_rate = data.get("agent_mutation_rate", 0.15)
        print(f"  Loaded {len(agent_brain.elite_pool) + len(agent_brain.recent_pool)} agent policies "
              f"(gen {agent_brain.generation}, best={agent_brain.best_fitness_ever:.0f})")

    if "anomalies" in data:
        for entry in data["anomalies"]:
            policy = AnomalyPolicy.from_dict(entry["weights"])
            ano_brain.gene_pool.append((entry["fitness"], policy.to_dict()))
        ano_brain.gene_pool.sort(key=lambda x: x[0], reverse=True)
        ano_brain.generation    = data.get("ano_generation", 0)
        ano_brain.mutation_rate = data.get("ano_mutation_rate", 0.15)
        print(f"  Loaded {len(ano_brain.elite_pool) + len(ano_brain.recent_pool)} anomaly policies "
              f"(gen {ano_brain.generation}, best={ano_brain.best_fitness_ever:.1f} dmg)")


def save_weights(path: str):
    agent_brain = get_brain()
    ano_brain   = get_anomaly_brain()

    data = {
        "agent_generation":    agent_brain.generation,
        "agent_mutation_rate": agent_brain.mutation_rate,
        "ano_generation":      ano_brain.generation,
        "ano_mutation_rate":   ano_brain.mutation_rate,
        "agents": [
            {"fitness": f, "weights": w}
            for f, w in agent_brain.gene_pool
        ],
        "anomalies": [
            {"fitness": f, "weights": w}
            for f, w in ano_brain.gene_pool
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f)


# ── Composite fitness function ────────────────────────────────────────────────
def _compute_fitness(agent, survival_ticks: int, world) -> float:
    """
    Real fitness = weighted combination of survival quality metrics.
    Pure survival time is NOT enough — we reward accomplishments.
    """
    score = 0.0

    # Base survival (increased weight and cap for better survival focus)
    score += min(survival_ticks, 1000) * 1.0

    # Shelters built nearby (reduced weight to balance with survival)
    shelters_near = sum(
        1 for b in world.buildings
        if b.building_type == "shelter"
        and abs(b.x - agent.x) + abs(b.y - agent.y) <= 15
    )
    score += shelters_near * 40

    # Resources gathered (shows the agent is actively playing)
    score += agent.resources_gathered * 2

    # Items crafted (shows progression)
    score += agent.items_crafted * 15

    # Anomalies killed
    score += agent.kills * 50

    # Inventory quality at death (had tools = was doing well)
    inv = agent.inventory
    if inv.get("axe", 0):    score += 30
    if inv.get("pickaxe", 0): score += 30
    if inv.get("sword", 0):  score += 60
    if inv.get("shield", 0): score += 40

    # Health at death (died healthy = old age, not combat = good)
    score += agent.health * 0.5

    # Penalty for dying young from starvation/damage (not old age)
    if survival_ticks < 100:
        score *= 0.3   # heavy penalty for dying fast

    return round(score, 1)


# ── Single generation runner ──────────────────────────────────────────────────
def run_generation(gen_idx: int, max_ticks: int, num_agents: int) -> dict:
    """
    Run one full generation headlessly.
    Returns stats dict.
    """
    agent_brain = get_brain()
    ano_brain   = get_anomaly_brain()

    # Fresh world — harder settings for training pressure
    world = SurvivalWorld()
    # Spawn some anomalies immediately so agents face real danger
    for _ in range(2):
        world.spawn_anomaly()
    # Lower population cap during training to create selection pressure
    # (monkey-patch _try_reproduce to cap at 20 instead of 50)
    _orig_try_reproduce = world._try_reproduce
    def _capped_reproduce():
        alive = sum(1 for a in world.agents.values() if a.alive)
        if alive >= 20:
            return
        _orig_try_reproduce()
    world._try_reproduce = _capped_reproduce

    # Spawn agents with policies from gene pool
    agent_policies = {}
    for i in range(num_agents):
        aid = f"g{gen_idx}_a{i}"
        world.add_agent(aid, generation=gen_idx)
        agent_policies[aid] = agent_brain.spawn_policy()

    # Give anomalies policies when they spawn (handled in Anomaly.__init__)
    # Track per-agent survival ticks
    agent_birth_tick = {aid: 0 for aid in agent_policies}

    tick = 0
    while tick < max_ticks:
        tick += 1
        world.tick += 1
        world.is_day = (world.tick % 100) < 55
        world.season = ["spring","summer","autumn","winter"][(world.tick // 60) % 4]

        # Weather
        if world.tick % 20 == 0:
            w = ["clear","clear","rain","wind"]
            if world.season == "winter": w += ["snow","blizzard"]
            if world.season == "summer": w += ["heatwave"]
            world.weather = random.choice(w)

        alive_agents = [a for a in world.agents.values() if a.alive]
        if not alive_agents:
            break

        # ── Agent actions ──
        # Build world state dict once per tick (shared across agents)
        global_res = {f"{x},{y}": r for (x,y), r in world.resources.items()}
        agent_dicts = {aid: a.to_dict() for aid, a in world.agents.items() if a.alive}
        world_state = {
            "is_day":          world.is_day,
            "weather":         world.weather,
            "agents":          agent_dicts,
            "anomalies":       [{"x": a.x, "y": a.y, "severity": a.severity,
                                  "anomaly_type": a.anomaly_type} for a in world.anomalies],
            "global_resources": global_res,
            "buildings":       [b.to_dict() for b in world.buildings],
        }

        for agent in alive_agents:
            policy = agent_policies.get(agent.agent_id)
            if policy is None:
                continue
            try:
                feat       = extract_features(agent.to_dict(), world_state)
                action_idx = policy.choose_action(feat)
                cmd        = action_to_command(action_idx, agent.to_dict(), world_state)
                world.process_action(
                    cmd["agent_id"],
                    cmd["action_type"],
                    cmd.get("target"),
                    cmd.get("params", {}),
                )
            except Exception:
                pass

        # ── Anomaly actions (every 3 ticks) ──
        if world.tick % 3 == 0 and alive_agents:
            alive_dicts = [a.to_dict() for a in alive_agents]
            for ano in world.anomalies:
                ano.severity += 0.01
                if ano.policy is None:
                    continue
                try:
                    ano_dict = {"x": ano.x, "y": ano.y, "severity": ano.severity,
                                "is_day": 1.0 if world.is_day else 0.0}
                    feat = extract_anomaly_features(ano_dict, alive_dicts)
                    act  = ano.policy.choose_action(feat)
                    dx, dy = anomaly_action_to_move(act, ano_dict, alive_dicts,
                                                    world.width, world.height)
                    if act == 3:  # retreat_grow
                        ano.severity += 0.05
                    ano.x = max(0, min(world.width  - 1, ano.x + dx))
                    ano.y = max(0, min(world.height - 1, ano.y + dy))
                except Exception:
                    pass

        # ── Anomaly damage ──
        for ano in world.anomalies:
            dmg_mult = 0.8 if ano.anomaly_type == "Void Creep" else 0.4
            for a in alive_agents:
                if abs(a.x - ano.x) <= 1 and abs(a.y - ano.y) <= 1:
                    dmg = max(2, int(ano.severity * dmg_mult))
                    if a.inventory.get("shield", 0) > 0:
                        dmg = max(1, dmg - 4)
                    a.health -= dmg
                    ano.damage_dealt += dmg
                    a.learn_from_action("damaged", False, {"damage_taken": dmg})

        # ── World step (hunger, energy, aging, reproduction, etc.) ──
        # Run the world's own step logic but skip the anomaly section
        # (we handled it above) — use a lightweight manual tick
        for agent in list(world.agents.values()):
            if not agent.alive:
                continue
            agent.age += 1
            agent.mate_cooldown = max(0, agent.mate_cooldown - 1)

            # Hunger / energy drain
            hd = 0.15 if world.season != "winter" else 0.25
            if world.weather == "blizzard": hd = 0.4
            agent.hunger = max(0, agent.hunger - hd / agent.traits.endurance)

            ed = 0.25
            if world.weather in ("blizzard","heatwave"): ed = 0.5
            agent.energy = max(0, agent.energy - ed / agent.traits.endurance)

            if world.is_day:
                agent.energy = min(100, agent.energy + 0.15)
                agent.health = min(agent.max_health, agent.health + 0.1)

            # Auto-eat
            if agent.hunger < 50:
                for food, restore in [("berry", 15), ("mushroom", 12)]:
                    if agent.inventory.get(food, 0) > 0:
                        agent.inventory[food] -= 1
                        agent.hunger = min(100, agent.hunger + restore)
                        break

            if agent.hunger <= 0: agent.health -= 0.5
            if agent.energy <= 0: agent.health -= 0.3

            # Shelter bonus
            in_shelter = any(
                b.building_type == "shelter" and
                abs(b.x - agent.x) <= 1 and abs(b.y - agent.y) <= 1
                for b in world.buildings
            )
            if in_shelter and not world.is_day:
                agent.health = min(agent.max_health, agent.health + 0.5)
                agent.energy = min(100, agent.energy + 0.5)

            # Death checks
            if agent.age >= agent.max_age or agent.health <= 0:
                survival = world.tick - agent_birth_tick.get(agent.agent_id, 0)
                fitness = _compute_fitness(agent, survival, world)
                agent_brain.record_death(agent_policies[agent.agent_id], fitness)
                agent.alive = False
                agent.health = 0

        # Record anomaly deaths
        for ano in world.anomalies:
            if ano.health <= 0 and ano.policy is not None:
                ano_brain.record_death(ano.policy, ano.damage_dealt)

        # Clean dead anomalies
        world.anomalies = [a for a in world.anomalies if a.health > 0]

        # Spawn anomalies
        chance = 0.01 if world.is_day else 0.03
        if world.season == "winter": chance += 0.01
        if random.random() < chance and len(world.anomalies) < 5:
            world.spawn_anomaly()

        # Resource respawn
        if random.random() < 0.3:
            x, y = random.randint(0, world.width-1), random.randint(0, world.height-1)
            if (x, y) not in world.resources:
                from survival_world import BIOME_RESOURCES
                biome = world.biome_map.get((x, y), "plains")
                world.resources[(x, y)] = random.choice(BIOME_RESOURCES.get(biome, ["wood"]))

        # Reproduction (simplified)
        world._try_reproduce()
        # Give new agents policies
        for aid in list(world.agents.keys()):
            if aid not in agent_policies:
                agent_policies[aid] = agent_brain.spawn_policy()
                agent_birth_tick[aid] = world.tick

    # ── End of generation: record any still-alive agents ──
    for agent in world.agents.values():
        if agent.alive and agent.agent_id in agent_policies:
            survival = world.tick - agent_birth_tick.get(agent.agent_id, 0)
            # Composite fitness: survival + accomplishments
            fitness = _compute_fitness(agent, survival, world)
            agent_brain.record_death(agent_policies[agent.agent_id], fitness)

    # Also record any anomalies still alive (partial fitness)
    for ano in world.anomalies:
        if ano.policy is not None:
            ano_brain.record_death(ano.policy, ano.damage_dealt)

    alive_count = sum(1 for a in world.agents.values() if a.alive)
    avg_survival = (
        sum(world.tick - agent_birth_tick.get(aid, 0)
            for aid, a in world.agents.items() if a.alive)
        / max(1, alive_count)
    ) if alive_count else 0

    return {
        "ticks":        world.tick,
        "alive":        alive_count,
        "total_born":   world.total_born,
        "total_died":   world.total_died,
        "shelters":     len([b for b in world.buildings if b.building_type == "shelter"]),
        "avg_survival": avg_survival,
    }


# ── Main training loop ────────────────────────────────────────────────────────
def main():
    agent_brain = get_brain()
    ano_brain   = get_anomaly_brain()

    print("=" * 60)
    print("  AnomalyCraft Survival — Offline Trainer")
    print("=" * 60)
    print(f"  Generations : {args.gens}")
    print(f"  Max ticks   : {args.ticks}")
    print(f"  Agents/gen  : {args.agents}")
    print(f"  Weights file: {WEIGHTS_FILE}")
    print()

    if args.resume:
        print("Resuming from saved weights...")
        load_weights(WEIGHTS_FILE)
        print()

    start_gen = agent_brain.generation
    total_start = time.time()

    for gen in range(args.gens):
        gen_idx = start_gen + gen
        t0 = time.time()

        stats = run_generation(gen_idx, args.ticks, args.agents)

        # Advance generation counters
        agent_brain.new_generation()
        ano_brain.new_generation()

        elapsed = time.time() - t0
        a_best  = agent_brain.gene_pool[0][0] if agent_brain.gene_pool else 0
        a_pool  = len(agent_brain.gene_pool)
        ano_best = ano_brain.gene_pool[0][0] if ano_brain.gene_pool else 0

        # Trend arrow
        hist = agent_brain.avg_fitness_history
        if len(hist) >= 2:
            arrow = "↑" if hist[-1] > hist[-2] else ("↓" if hist[-1] < hist[-2] else "→")
        else:
            arrow = " "

        print(
            f"  Gen {gen_idx+1:>4} | "
            f"ticks={stats['ticks']:>5} | "
            f"alive={stats['alive']:>2} | "
            f"born={stats['total_born']:>4} | "
            f"shelters={stats['shelters']:>3} | "
            f"fitness={a_best:>7.0f} {arrow} | "
            f"pool={a_pool:>2} | "
            f"ano_best={ano_best:>6.1f} dmg | "
            f"mut={agent_brain.mutation_rate:.3f} | "
            f"{elapsed:.1f}s"
        )
        sys.stdout.flush()

        # Save every 10 generations
        if (gen + 1) % 10 == 0:
            save_weights(WEIGHTS_FILE)
            print(f"  💾 Saved weights to {WEIGHTS_FILE}")

    # Final save
    save_weights(WEIGHTS_FILE)
    total_elapsed = time.time() - total_start

    print()
    print("=" * 60)
    print(f"  Training complete in {total_elapsed:.1f}s")
    print(f"  Agent best fitness : {agent_brain.gene_pool[0][0]:.0f} ticks" if agent_brain.gene_pool else "  No agent data")
    print(f"  Anomaly best fitness: {ano_brain.gene_pool[0][0]:.1f} dmg" if ano_brain.gene_pool else "  No anomaly data")
    print(f"  Weights saved to   : {WEIGHTS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
