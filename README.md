---
title: AnomalyCraft Survival
emoji: 🌍
colorFrom: purple
colorTo: red
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - survival
  - multi-agent
  - neuroevolution
  - hackathon-2026
pinned: false
---

# 🌍 AnomalyCraft Survival

> **OpenEnv Hackathon 2026 — Meta × PyTorch × Hugging Face**

A multi-agent survival simulation where AI agents and anomalies both evolve neural network policies via neuroevolution. The environment is fully OpenEnv-compliant with 5 tasks (easy → nightmare), WebSocket transport, and a live pixel-art web UI.

## 🔗 Deliverables

| Deliverable | Link |
|---|---|
| 🤗 **Hugging Face Space** | [anomalycraft-survival on HF Spaces](https://huggingface.co/spaces/nimeshyadav/OpenEnv) |
| 📓 **GitHub Repo** | [fursatiinsaan/OrgSim](https://github.com/fursatiinsaan/OrgSim) |
| 📊 **Training Curves** | See below |

---

## 📈 Training Curves

### Reward Curve — Agent Fitness Over Generations
![Reward Curve](training_curves/reward_curve.png)

### Loss Proxy — Fitness Gap (Best − Avg) Over Generations
![Loss Curve](training_curves/loss_curve.png)

### Anomaly Neural Policy — Damage Dealt Over Generations
![Anomaly Curve](training_curves/anomaly_curve.png)

### Shelters Built Per Generation
![Shelters Curve](training_curves/shelters_curve.png)

---

## 🎮 What It Is

OrgSim is a 48×48 grid world where:
- **Agents** gather resources, craft tools, build shelters, form communities, and fight anomalies
- **Anomalies** hunt agents using their own evolving neural policies
- Both sides improve via **neuroevolution** — an arms race across generations
- The world auto-restarts when all agents die, with smarter agents each generation

### 5 Tasks (Easy → Nightmare)

| ID | Name | Difficulty | Objective |
|---|---|---|---|
| 101 | First Steps | Easy | Gather 5+ resources, survive 30 ticks |
| 102 | Craft and Survive | Medium | Craft 2+ items, survive 80 ticks |
| 103 | Anomaly Outbreak | Hard | Destroy 1+ anomaly, 2+ agents alive |
| 104 | Build a Civilization | Expert | Community + 2 buildings + pop 8+ |
| 105 | Winter Siege | Nightmare | Survive winter, destroy 3 anomalies |

---

## 🧠 Neural Network Policy

Each agent carries a **22-input → 16-hidden → 9-output** feedforward network (numpy only, no GPU needed).

**9 actions the network learns to choose:**
- `move_away_danger`, `move_to_resource`, `move_to_loved_one`, `move_random`
- `gather`, `craft_or_build`, `fight`, `eat_or_rest`, `noop`

**Fitness = composite score:**
- Survival ticks (capped at 500) × 0.5
- Shelters built nearby × 80
- Resources gathered × 2
- Items crafted × 15
- Anomalies killed × 50
- Tool bonuses (axe +30, sword +60, etc.)

**Anomaly policy** (12-input → 10-hidden → 5-output):
- Actions: `chase`, `flank_left`, `flank_right`, `retreat_grow`, `spread`
- Fitness = total damage dealt

**Evolution mechanics:**
- Elite pool (top-10 all-time) + Recent pool (last 10 gens, any fitness)
- Stagnation detection: mutation resets to 0.12 after 8 gens without improvement
- 70% elite crossover, 20% recent diversity, 10% fresh random

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (generates weights.json + training_curves/*.png)
python3 plot_training.py --gens 60

# Run server (loads weights, trains live, shows web UI)
python3 app.py
# Open http://localhost:8000
```

### Docker

```bash
docker build -t anomalycraft .
docker run -p 8000:8000 anomalycraft
```

---

## 🔌 OpenEnv API

### HTTP Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server status |
| `/survival/reset?task_id=101` | POST | Reset environment |
| `/survival/step` | POST | Execute action |
| `/survival/state` | GET | World state |
| `/survival/grader?task_id=101` | GET | Grade current episode |

### WebSocket

Connect to `ws://localhost:8000/ws` for persistent sessions:

```python
from client import SurvivalClient

async with SurvivalClient(base_url="ws://localhost:8000") as env:
    result = await env.reset(task_id=101)
    while not result.done:
        action = {"agent_id": "agent_1", "action_type": "gather"}
        result = await env.step(action)
```

### Action Schema

```json
{
  "agent_id": "agent_1",
  "action_type": "move",
  "target": "right"
}
```

### Observation Schema

```json
{
  "agent_stats": {"health": 95, "energy": 80, "hunger": 70, "inventory": {"wood": 5}},
  "local_resources": {"10,12": "wood", "11,12": "stone"},
  "nearby_anomalies": [{"anomaly_type": "Void Creep", "severity": 2.1}],
  "available_actions": ["move", "gather", "rest"],
  "tick": 42,
  "done": false,
  "reward": 1.05
}
```

---

## 📊 Baseline Scores

Scores from heuristic agent (no LLM, rule-based fallback):

| Task | Difficulty | Score | Success |
|---|---|---|---|
| 101 | Easy | 0.82 | ✅ |
| 102 | Medium | 0.61 | ✅ |
| 103 | Hard | 0.44 | ❌ |
| 104 | Expert | 0.31 | ❌ |
| 105 | Nightmare | 0.18 | ❌ |
| **Avg** | | **0.47** | |

---

## 🏗️ Architecture

```
app.py              — FastAPI (WebSocket /ws) + Flask (HTTP routes) + training thread
survival_env.py     — OpenEnv Environment subclass
survival_world.py   — World simulation (agents, anomalies, resources, buildings)
neural_policy.py    — NeuralPolicy + AnomalyPolicy + CollectiveBrain + AnomalyBrain
agent_ai.py         — Per-agent policy dispatch
train.py            — Headless offline trainer
plot_training.py    — Training + curve generation
client.py           — OpenEnv EnvClient subclass (WebSocket)
models.py           — Pydantic models (inherit from OpenEnv Action/Observation/State)
inference.py        — LLM agent runner (OpenAI-compatible)
openenv.yaml        — Environment manifest
```

---

## 📝 License

MIT — see [LICENSE](./LICENSE)