"""
AnomalyCraft Survival — Flask application.
OpenEnv-compliant HTTP server with survival environment endpoints.

On startup:
  1. Loads pre-trained weights from weights.json (if exists)
  2. Starts a background training thread that runs forever
  3. The live world shown in the browser IS the training world
  4. Terminal prints detailed per-generation training logs
"""

import os
import sys
import time
import random
import threading
from threading import Lock

from flask import Flask, jsonify, render_template, request


def load_local_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()

from survival_env import SurvivalEnv
from models import (
    MetadataResponse,
    SurvivalResetResponse,
    SurvivalStepResponse,
    SurvivalAction,
    SurvivalWorldState,
    GraderResponse,
)
from tasks import SURVIVAL_TASKS
from agent_ai import decide_action

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "openenv-local-secret")
PORT = int(os.getenv("PORT", "8000"))

# ── Shared state ──────────────────────────────────────────────────────────────
_ENV      = None
_ENV_LOCK = Lock()

# Training stats visible to the UI and terminal
_TRAIN_STATS = {
    "running":        False,
    "generation":     0,
    "tick":           0,
    "alive":          0,
    "born":           0,
    "died":           0,
    "shelters":       0,
    "agent_fitness":  0.0,
    "ano_fitness":    0.0,
    "mutation_rate":  0.15,
    "fitness_trend":  [],   # last 20 best-fitness values
    "gen_log":        [],   # last 30 generation summary lines
    "tick_log":       [],   # last 50 per-tick event lines
}
_TRAIN_LOCK = Lock()


def get_env() -> SurvivalEnv:
    global _ENV
    with _ENV_LOCK:
        if _ENV is None:
            _ENV = SurvivalEnv()
    return _ENV


# ── Weight persistence ────────────────────────────────────────────────────────
WEIGHTS_FILE = os.getenv("WEIGHTS_FILE", "weights.json")

def load_weights():
    import json
    from neural_policy import NeuralPolicy, AnomalyPolicy, get_brain, get_anomaly_brain
    if not os.path.exists(WEIGHTS_FILE):
        print("[trainer] No weights.json found — starting from scratch", flush=True)
        return
    try:
        with open(WEIGHTS_FILE) as f:
            data = json.load(f)
        ab = get_brain()
        nb = get_anomaly_brain()

        if data.get("agents"):
            entries = [(e["fitness"], e["weights"]) for e in data["agents"]]
            entries.sort(key=lambda x: x[0], reverse=True)
            ab.elite_pool  = entries[:10]
            ab.recent_pool = entries[10:20]
            ab.generation    = data.get("agent_generation", 0)
            ab.mutation_rate = data.get("agent_mutation_rate", 0.15)
            ab.best_fitness_ever = entries[0][0] if entries else 0.0

        if data.get("anomalies"):
            entries = [(e["fitness"], e["weights"]) for e in data["anomalies"]]
            entries.sort(key=lambda x: x[0], reverse=True)
            nb.elite_pool  = entries[:10]
            nb.recent_pool = entries[10:20]
            nb.generation    = data.get("ano_generation", 0)
            nb.mutation_rate = data.get("ano_mutation_rate", 0.15)
            nb.best_fitness_ever = entries[0][0] if entries else 0.0

        best_a = ab.elite_pool[0][0] if ab.elite_pool else 0
        best_n = nb.elite_pool[0][0] if nb.elite_pool else 0
        print(f"[trainer] Loaded weights — agent gen={ab.generation} best={best_a:.0f}  "
              f"anomaly gen={nb.generation} best={best_n:.1f} dmg", flush=True)
    except Exception as e:
        print(f"[trainer] Failed to load weights: {e}", flush=True)


def save_weights():
    import json
    from neural_policy import get_brain, get_anomaly_brain
    ab = get_brain()
    nb = get_anomaly_brain()
    data = {
        "agent_generation":    ab.generation,
        "agent_mutation_rate": ab.mutation_rate,
        "ano_generation":      nb.generation,
        "ano_mutation_rate":   nb.mutation_rate,
        "agents":    [{"fitness": f, "weights": w} for f, w in ab.gene_pool],
        "anomalies": [{"fitness": f, "weights": w} for f, w in nb.gene_pool],
    }
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(data, f)


# ── Fitness function (same as train.py) ──────────────────────────────────────
def _compute_fitness(agent, survival_ticks: int, world) -> float:
    score = min(survival_ticks, 1000) * 1.0
    shelters_near = sum(
        1 for b in world.buildings
        if b.building_type == "shelter"
        and abs(b.x - agent.x) + abs(b.y - agent.y) <= 15
    )
    score += shelters_near * 40
    score += agent.resources_gathered * 2
    score += agent.items_crafted * 15
    score += agent.kills * 50
    inv = agent.inventory
    if inv.get("axe"):     score += 30
    if inv.get("pickaxe"): score += 30
    if inv.get("sword"):   score += 60
    if inv.get("shield"):  score += 40
    score += agent.health * 0.5
    if survival_ticks < 100:
        score *= 0.3
    return round(score, 1)


# ── Background training thread ────────────────────────────────────────────────
def _training_loop():
    """
    Runs forever in a background thread.
    Each 'generation' = one full episode on the live world.
    The live world IS the training world — browser shows it in real time.
    """
    from neural_policy import (
        get_brain, get_anomaly_brain,
        extract_features, action_to_command,
        extract_anomaly_features, anomaly_action_to_move,
        STAGNATION_GENS,
    )
    from survival_world import SurvivalWorld, BIOME_RESOURCES as BR
    from agent_ai import on_agent_death, on_generation_end

    ab = get_brain()
    nb = get_anomaly_brain()

    gen = ab.generation
    TICKS_PER_GEN = 1500
    NUM_AGENTS    = 8
    POP_CAP       = 20   # training pressure

    def _tlog(msg: str):
        """Print to terminal AND store in tick_log for UI."""
        print(msg, flush=True)
        with _TRAIN_LOCK:
            _TRAIN_STATS["tick_log"].append(msg)
            if len(_TRAIN_STATS["tick_log"]) > 50:
                _TRAIN_STATS["tick_log"].pop(0)

    def _glog(msg: str):
        """Print generation summary to terminal AND gen_log for UI."""
        print(msg, flush=True)
        with _TRAIN_LOCK:
            _TRAIN_STATS["gen_log"].append(msg)
            if len(_TRAIN_STATS["gen_log"]) > 30:
                _TRAIN_STATS["gen_log"].pop(0)

    print("", flush=True)
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║        AnomalyCraft — Live Training Started              ║", flush=True)
    print("║  Open http://localhost:8000 to watch in the browser      ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)
    print("", flush=True)

    with _TRAIN_LOCK:
        _TRAIN_STATS["running"] = True

    while True:
        gen += 1
        t_gen_start = time.time()

        # ── Fresh world for this generation ──
        world = SurvivalWorld()
        for _ in range(2):
            world.spawn_anomaly()

        # Update the shared env so the browser sees this world
        global _ENV
        with _ENV_LOCK:
            if _ENV is None:
                _ENV = SurvivalEnv()
            _ENV.world = world

        # Population cap patch
        _orig_repr = world._try_reproduce
        def _capped():
            if sum(1 for a in world.agents.values() if a.alive) >= POP_CAP:
                return
            _orig_repr()
        world._try_reproduce = _capped

        # Spawn agents
        agent_policies  = {}
        agent_birth_tick = {}
        for i in range(NUM_AGENTS):
            aid = f"g{gen}_a{i}"
            world.add_agent(aid, generation=gen)
            agent_policies[aid]   = ab.spawn_policy()
            agent_birth_tick[aid] = 0

        deaths_this_gen = 0
        shelters_built  = 0

        print(f"\n{'─'*62}", flush=True)
        print(f"  GEN {gen:>4}  |  agents={NUM_AGENTS}  |  "
              f"pool={len(ab.gene_pool)}/20  |  mut={ab.mutation_rate:.4f}", flush=True)
        print(f"{'─'*62}", flush=True)

        for tick in range(1, TICKS_PER_GEN + 1):
            world.tick  = tick
            world.is_day = (tick % 100) < 55
            world.season = ["spring","summer","autumn","winter"][(tick // 60) % 4]

            if tick % 20 == 0:
                w = ["clear","clear","rain","wind"]
                if world.season == "winter": w += ["snow","blizzard"]
                if world.season == "summer": w += ["heatwave"]
                world.weather = random.choice(w)

            alive_agents = [a for a in world.agents.values() if a.alive]
            if not alive_agents:
                _tlog(f"  [tick {tick:>5}] ALL DEAD — ending generation early")
                break

            # Build shared world state dict
            global_res  = {f"{x},{y}": r for (x,y), r in world.resources.items()}
            agent_dicts = {aid: a.to_dict() for aid, a in world.agents.items() if a.alive}
            world_state = {
                "is_day": world.is_day, "weather": world.weather,
                "agents": agent_dicts,
                "anomalies": [{"x": a.x, "y": a.y, "severity": a.severity,
                               "anomaly_type": a.anomaly_type} for a in world.anomalies],
                "global_resources": global_res,
                "buildings": [b.to_dict() for b in world.buildings],
            }

            # Agent actions
            for agent in alive_agents:
                pol = agent_policies.get(agent.agent_id)
                if pol is None:
                    continue
                try:
                    feat = extract_features(agent.to_dict(), world_state)
                    idx  = pol.choose_action(feat)
                    cmd  = action_to_command(idx, agent.to_dict(), world_state)
                    world.process_action(cmd["agent_id"], cmd["action_type"],
                                         cmd.get("target"), cmd.get("params", {}))
                except Exception:
                    pass

            # Anomaly actions (every 3 ticks)
            if tick % 3 == 0 and alive_agents:
                alive_dicts = [a.to_dict() for a in alive_agents]
                for ano in world.anomalies:
                    ano.severity += 0.01
                    if ano.policy is None:
                        continue
                    try:
                        ad   = {"x": ano.x, "y": ano.y, "severity": ano.severity,
                                "is_day": 1.0 if world.is_day else 0.0}
                        feat = extract_anomaly_features(ad, alive_dicts)
                        act  = ano.policy.choose_action(feat)
                        dx, dy = anomaly_action_to_move(act, ad, alive_dicts,
                                                        world.width, world.height)
                        if act == 3:
                            ano.severity += 0.05
                        ano.x = max(0, min(world.width  - 1, ano.x + dx))
                        ano.y = max(0, min(world.height - 1, ano.y + dy))
                    except Exception:
                        pass

            # Anomaly damage
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

            # Agent lifecycle
            for agent in list(world.agents.values()):
                if not agent.alive:
                    continue
                agent.age += 1
                agent.mate_cooldown = max(0, agent.mate_cooldown - 1)
                hd = 0.25 if world.season == "winter" else 0.15
                if world.weather == "blizzard": hd = 0.4
                agent.hunger = max(0, agent.hunger - hd / agent.traits.endurance)
                ed = 0.5 if world.weather in ("blizzard","heatwave") else 0.25
                agent.energy = max(0, agent.energy - ed / agent.traits.endurance)
                if world.is_day:
                    agent.energy = min(100, agent.energy + 0.15)
                    agent.health = min(agent.max_health, agent.health + 0.1)
                if agent.hunger < 50:
                    for food, restore in [("berry", 15), ("mushroom", 12)]:
                        if agent.inventory.get(food, 0) > 0:
                            agent.inventory[food] -= 1
                            agent.hunger = min(100, agent.hunger + restore)
                            break
                if agent.hunger <= 0: agent.health -= 0.5
                if agent.energy <= 0: agent.health -= 0.3
                in_shelter = any(
                    b.building_type == "shelter" and
                    abs(b.x - agent.x) <= 1 and abs(b.y - agent.y) <= 1
                    for b in world.buildings
                )
                if in_shelter and not world.is_day:
                    agent.health = min(agent.max_health, agent.health + 0.5)
                    agent.energy = min(100, agent.energy + 0.5)

                if agent.age >= agent.max_age or agent.health <= 0:
                    survival = tick - agent_birth_tick.get(agent.agent_id, 0)
                    fitness  = _compute_fitness(agent, survival, world)
                    ab.record_death(agent_policies[agent.agent_id], fitness)
                    on_agent_death(agent.agent_id, survival)
                    agent.alive  = False
                    agent.health = 0
                    deaths_this_gen += 1
                    cause = "old age" if agent.age >= agent.max_age else "combat/starvation"
                    _tlog(f"  [tick {tick:>5}] 💀 {agent.agent_id} died ({cause}) "
                          f"survival={survival} fitness={fitness:.0f} "
                          f"shelters={agent.items_crafted} res={agent.resources_gathered}")

            # Anomaly deaths
            for ano in world.anomalies:
                if ano.health <= 0 and ano.policy is not None:
                    nb.record_death(ano.policy, ano.damage_dealt)
                    _tlog(f"  [tick {tick:>5}] 💥 {ano.anomaly_type} destroyed "
                          f"(dealt {ano.damage_dealt:.0f} dmg)")

            world.anomalies = [a for a in world.anomalies if a.health > 0]

            # Spawn anomalies
            chance = 0.01 if world.is_day else 0.03
            if world.season == "winter": chance += 0.01
            if random.random() < chance and len(world.anomalies) < 5:
                world.spawn_anomaly()
                _tlog(f"  [tick {tick:>5}] ⚡ New anomaly spawned "
                      f"(total={len(world.anomalies)})")

            # Resource respawn
            if random.random() < 0.3:
                rx, ry = random.randint(0, world.width-1), random.randint(0, world.height-1)
                if (rx, ry) not in world.resources:
                    biome = world.biome_map.get((rx, ry), "plains")
                    world.resources[(rx, ry)] = random.choice(BR.get(biome, ["wood"]))

            # Reproduction
            world._try_reproduce()
            for aid in list(world.agents.keys()):
                if aid not in agent_policies:
                    agent_policies[aid]   = ab.spawn_policy()
                    agent_birth_tick[aid] = tick
                    _tlog(f"  [tick {tick:>5}] 👶 {aid} born (gen {gen})")

            # Track shelters
            shelters_built = len([b for b in world.buildings if b.building_type == "shelter"])

            # Update shared stats every 50 ticks
            if tick % 50 == 0:
                alive_now = sum(1 for a in world.agents.values() if a.alive)
                a_best = ab.gene_pool[0][0] if ab.gene_pool else 0
                n_best = nb.gene_pool[0][0] if nb.gene_pool else 0
                line = (f"  [tick {tick:>5}] alive={alive_now:>2}  "
                        f"shelters={shelters_built:>3}  "
                        f"anomalies={len(world.anomalies)}  "
                        f"weather={world.weather:<10}  "
                        f"season={world.season:<7}  "
                        f"{'🌙 NIGHT' if not world.is_day else '☀️  DAY  '}")
                _tlog(line)
                with _TRAIN_LOCK:
                    _TRAIN_STATS.update({
                        "tick":          tick,
                        "alive":         alive_now,
                        "born":          world.total_born,
                        "died":          world.total_died,
                        "shelters":      shelters_built,
                        "agent_fitness": a_best,
                        "ano_fitness":   n_best,
                        "mutation_rate": ab.mutation_rate,
                        "generation":    gen,
                    })

        # ── End of generation ──
        for agent in world.agents.values():
            if agent.alive and agent.agent_id in agent_policies:
                survival = world.tick - agent_birth_tick.get(agent.agent_id, 0)
                fitness  = _compute_fitness(agent, survival, world)
                ab.record_death(agent_policies[agent.agent_id], fitness)
        for ano in world.anomalies:
            if ano.policy is not None:
                nb.record_death(ano.policy, ano.damage_dealt)

        ab.new_generation()
        nb.new_generation()
        on_generation_end()

        elapsed  = time.time() - t_gen_start
        a_best   = ab.gene_pool[0][0] if ab.gene_pool else 0
        n_best   = nb.gene_pool[0][0] if nb.gene_pool else 0
        hist     = ab.avg_fitness_history
        arrow    = ("↑" if len(hist) >= 2 and hist[-1] > hist[-2] else
                    "↓" if len(hist) >= 2 and hist[-1] < hist[-2] else "→")
        alive_end = sum(1 for a in world.agents.values() if a.alive)

        summary = (
            f"\n  ✅ GEN {gen:>4} DONE | "
            f"ticks={world.tick}  alive={alive_end}  "
            f"born={world.total_born}  died={world.total_died}  "
            f"shelters={shelters_built}\n"
            f"         agent_fitness={a_best:>8.0f} {arrow}  "
            f"pool={len(ab.elite_pool)}e+{len(ab.recent_pool)}r  "
            f"mut={ab.mutation_rate:.4f}  stag={ab.gens_since_improvement}/{STAGNATION_GENS}\n"
            f"         ano_fitness  ={n_best:>8.1f} dmg  "
            f"pool={len(nb.elite_pool)}e+{len(nb.recent_pool)}r  "
            f"mut={nb.mutation_rate:.4f}\n"
            f"         elapsed={elapsed:.1f}s"
        )
        _glog(summary)

        with _TRAIN_LOCK:
            _TRAIN_STATS["fitness_trend"].append(a_best)
            if len(_TRAIN_STATS["fitness_trend"]) > 20:
                _TRAIN_STATS["fitness_trend"].pop(0)

        # Save every 10 gens
        if gen % 10 == 0:
            save_weights()
            msg = f"  💾 Weights saved to {WEIGHTS_FILE} (gen {gen})"
            _glog(msg)


# ── Load weights and start training thread ────────────────────────────────────
load_weights()

_train_thread = threading.Thread(target=_training_loop, daemon=True, name="trainer")
_train_thread.start()


def get_env() -> SurvivalEnv:
    global _ENV
    with _ENV_LOCK:
        if _ENV is None:
            _ENV = SurvivalEnv()
    return _ENV


# ─── Shared routes ────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "env": "anomalycraft-survival",
            "version": "2.0.0",
            "supported_modes": ["survival"],
        }
    )


@app.route("/metadata")
def metadata():
    payload = MetadataResponse(
        name="anomalycraft-survival",
        version="2.0.0",
        description="A multi-agent survival and crafting RL training environment.",
        framework="flask+pydantic",
        endpoints={
            "survival_tasks": "/survival/tasks",
            "survival_reset": "/survival/reset",
            "survival_step": "/survival/step",
            "survival_state": "/survival/state",
            "survival_grader": "/survival/grader",
        },
        task_count=len(SURVIVAL_TASKS),
        supports_web_ui=True,
    )
    return jsonify(payload.model_dump())


@app.route("/schema")
def schema():
    return jsonify(
        {
            "survival": {
                "action_schema": SurvivalAction.model_json_schema(),
                "reset_schema": SurvivalResetResponse.model_json_schema(),
                "step_schema": SurvivalStepResponse.model_json_schema(),
                "state_schema": SurvivalWorldState.model_json_schema(),
                "grader_schema": GraderResponse.model_json_schema(),
            }
        }
    )


# ─── Survival routes ──────────────────────────────────────────────────────────

@app.route("/survival/tasks")
def survival_tasks():
    return jsonify({"tasks": SURVIVAL_TASKS})


@app.route("/survival/reset", methods=["GET", "POST"])
def survival_reset():
    global _ENV
    task_id = int(request.args.get("task_id", 101))

    with _ENV_LOCK:
        _ENV = SurvivalEnv()
        env = _ENV

    observation = env.reset(task_id=task_id)
    return jsonify(
        SurvivalResetResponse(
            observation=observation,
            world_state=env.state,
        ).model_dump()
    )


@app.route("/survival/state")
def survival_state():
    env = get_env()
    agent_id = request.args.get("agent_id")
    if agent_id and agent_id in env.world.agents:
        return jsonify(
            SurvivalResetResponse(
                observation=env.state_for_agent(agent_id),
                world_state=env.state,
            ).model_dump()
        )
    return jsonify(env.state.model_dump())


@app.route("/survival/step", methods=["POST"])
def survival_step():
    payload = request.get_json(silent=True) or {}
    env = get_env()
    observation, reward, done, info = env.step(payload)
    return jsonify(
        SurvivalStepResponse(
            observation=observation,
            world_state=env.state,
            reward=reward,
            done=done,
            info=info,
        ).model_dump()
    )


@app.route("/survival/grader")
def survival_grader():
    env = get_env()
    task_id = request.args.get("task_id")
    if task_id is not None:
        result = env.grade()
        # Override task_id if explicitly requested
        from grader import grade as _grade
        result = _grade(env, int(task_id))
    else:
        result = env.grade()
    return jsonify(result)


@app.route("/survival/ai_step", methods=["POST"])
def survival_ai_step():
    """Run AI for all alive agents and advance world."""
    env = get_env()
    state = env.state.model_dump()
    
    # Run AI for each alive agent
    for agent_id, agent_data in state["agents"].items():
        if not agent_data["alive"]:
            continue
        
        action = decide_action(agent_data, state)
        try:
            env.step(action)
        except Exception as e:
            print(f"AI step error for {agent_id}: {e}")
    
    # Return updated state
    return jsonify({
        "tick": env.world.tick,
        "alive": sum(1 for a in env.world.agents.values() if a.alive),
        "score": env.score,
    })


@app.route("/survival/place_resource", methods=["POST"])
def place_resource():
    """Place a resource at specified location (click to place)."""
    payload = request.get_json(silent=True) or {}
    x = payload.get("x", 0)
    y = payload.get("y", 0)
    resource_type = payload.get("resource_type", "wood")
    
    env = get_env()
    valid_resources = ["wood", "stone", "iron", "crystal", "berry", "mushroom"]
    if resource_type in valid_resources:
        env.world.resources[(x, y)] = resource_type
        return jsonify({"success": True, "message": f"Placed {resource_type} at ({x},{y})"})
    return jsonify({"success": False, "message": "Invalid resource type"})


@app.route("/survival/player_action", methods=["POST"])
def player_action():
    """Execute player action for a specific agent."""
    payload = request.get_json(silent=True) or {}
    agent_id = payload.get("agent_id")
    action_type = payload.get("action_type")
    target = payload.get("target")
    params = payload.get("params", {})
    
    env = get_env()
    if agent_id not in env.world.agents:
        return jsonify({"success": False, "message": "Agent not found"})
    
    agent = env.world.agents[agent_id]
    if not agent.alive:
        return jsonify({"success": False, "message": "Agent is dead"})
    
    try:
        env.world.process_action(agent_id, action_type, target, params)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/survival/save", methods=["POST"])
def save_game():
    """Save current game state to file."""
    env = get_env()
    import json
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"save_{timestamp}.json"
    
    save_data = {
        "tick": env.world.tick,
        "agents": {aid: a.to_dict() for aid, a in env.world.agents.items()},
        "communities": {cid: c.to_dict() for cid, c in env.world.communities.items()},
        "buildings": [b.to_dict() for b in env.world.buildings],
        "resources": {f"{x},{y}": r for (x,y), r in env.world.resources.items()},
        "anomalies": [{"id": a.anomaly_id, "type": a.anomaly_type, "x": a.x, "y": a.y, "severity": a.severity, "health": a.health} for a in env.world.anomalies],
        "score": env.score,
        "total_born": env.world.total_born,
        "total_died": env.world.total_died,
    }
    
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    return jsonify({"success": True, "filename": filename})


@app.route("/survival/stats")
def survival_stats():
    """Get detailed statistics dashboard."""
    env = get_env()
    
    agents = list(env.world.agents.values())
    alive_agents = [a for a in agents if a.alive]
    
    # Personality distribution
    personality_counts = {}
    for a in alive_agents:
        p = a.personality
        personality_counts[p] = personality_counts.get(p, 0) + 1
    
    # Average traits
    avg_traits = {"speed": 0, "strength": 0, "intelligence": 0, "endurance": 0}
    if alive_agents:
        for a in alive_agents:
            for trait, val in a.traits.to_dict().items():
                avg_traits[trait] += val
        for trait in avg_traits:
            avg_traits[trait] /= len(alive_agents)
    
    # Resource totals
    resource_counts = {}
    for a in alive_agents:
        for res, count in a.inventory.items():
            resource_counts[res] = resource_counts.get(res, 0) + count
    
    return jsonify({
        "tick": env.world.tick,
        "population": {
            "alive": len(alive_agents),
            "total_born": env.world.total_born,
            "total_died": env.world.total_died,
        },
        "personalities": personality_counts,
        "avg_traits": avg_traits,
        "resources": resource_counts,
        "communities": len(env.world.communities),
        "buildings": len(env.world.buildings),
        "anomalies": len(env.world.anomalies),
        "score": env.score,
    })


@app.route("/survival/nn_stats")
def nn_stats():
    """Return neural network learning statistics for both agents and anomalies."""
    from agent_ai import get_nn_stats
    from neural_policy import get_anomaly_brain
    with _TRAIN_LOCK:
        train = _TRAIN_STATS.copy()
    return jsonify({
        "agents":    get_nn_stats(),
        "anomalies": get_anomaly_brain().stats(),
        "training":  train,
    })


@app.route("/survival/train_log")
def train_log():
    """Return live training logs for the terminal panel in the UI."""
    with _TRAIN_LOCK:
        return jsonify({
            "gen_log":  _TRAIN_STATS["gen_log"][-20:],
            "tick_log": _TRAIN_STATS["tick_log"][-30:],
            "stats":    {k: v for k, v in _TRAIN_STATS.items()
                         if k not in ("gen_log", "tick_log")},
        })


# ── OpenEnv WebSocket layer (FastAPI) ─────────────────────────────────────────
# Wraps the Flask app via WSGI middleware so both HTTP routes AND /ws work
# on the same port. Judges can connect via WebSocket at ws://host:8000/ws.

from fastapi import FastAPI as _FastAPI, WebSocket as _WebSocket, WebSocketDisconnect as _WSD
from a2wsgi import WSGIMiddleware as _WSGIMiddleware
# from openenv_core.env_server.types import (
#     WSResetMessage, WSStepMessage, WSStateMessage, WSCloseMessage,
#     WSObservationResponse, WSStateResponse, WSErrorResponse, WSErrorCode,
# )
import json as _json

# _fastapi_app = _FastAPI(title="AnomalyCraft Survival", version="2.0.0")


# WebSocket code commented out due to missing types


# _fastapi_app.mount("/", _WSGIMiddleware(app))

# # This is what uvicorn/gunicorn serves
# asgi_app = _fastapi_app


if __name__ == "__main__":
    # Run with Flask
    app.run(host="0.0.0.0", port=PORT, debug=True)
