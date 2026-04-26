"""
AnomalyCraft Survival — Inference script.
Runs an LLM agent against all 5 survival tasks and produces reproducible scores.

Required environment variables:
    API_BASE_URL   — LLM API endpoint
    MODEL_NAME     — model identifier
    API_KEY        — API key (also accepts HF_TOKEN)

Log format (strictly followed for automated evaluation):
    [START] task=<id> env=anomalycraft-survival model=<model>
    [STEP]  step=<n> action=<str> reward=<float> done=<bool> error=<str|null>
    [END]   success=<bool> steps=<n> score=<float> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import signal
import textwrap
import time
from typing import List, Optional

import requests
from openai import OpenAI

# ─── Config ──────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY: str = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

TEMPERATURE: float = 0.2
MAX_TOKENS: int = 256
MAX_STEPS: int = 40          # per task — well within 20-min budget
TASK_IDS: List[int] = [101, 102, 103, 104, 105]
BENCHMARK: str = "anomalycraft-survival"

# 20-minute hard timeout (1200s) with 60s buffer as required by OpenEnv spec
TIMEOUT_SECONDS: int = 20 * 60
_START_TIME: float = time.time()


def _check_timeout() -> None:
    """Raise TimeoutError if we are within 60s of the 20-min deadline."""
    elapsed = time.time() - _START_TIME
    if elapsed >= TIMEOUT_SECONDS - 60:
        raise TimeoutError(
            f"Approaching 20-minute limit ({elapsed:.0f}s elapsed). Stopping early."
        )

# ─── Logging helpers (strict format required by evaluator) ───────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── Environment client (HTTP) ───────────────────────────────────────────────

def env_reset(task_id: int) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/survival/reset",
        params={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/survival/step",
        json=action,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_grader(task_id: int) -> dict:
    resp = requests.get(
        f"{ENV_BASE_URL}/survival/grader",
        params={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ─── Prompt helpers ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent playing AnomalyCraft Survival — a multi-agent survival game.
    You control agents on a 24x24 grid world. Each turn you must choose ONE action for ONE agent.

    Available action_types:
      move        — target: "up"|"down"|"left"|"right"
      gather      — collect resource at current tile (no target needed)
      eat         — consume food from inventory (no target needed)
      rest        — recover energy and health (no target needed)
      craft       — target: recipe name (e.g. "axe", "shelter_kit", "void_stabilizer")
      build       — target: building type (e.g. "shelter", "farm", "wall")
      attack      — attack a nearby anomaly (no target needed)
      form_community — found a new tribe (no target needed)
      join_community — target: community_id
      share       — target: resource name to share with community
      noop        — do nothing

    Respond with ONLY a valid JSON object, no explanation:
    {"agent_id": "agent_1", "action_type": "gather"}
    {"agent_id": "agent_2", "action_type": "move", "target": "right"}
    {"agent_id": "agent_1", "action_type": "craft", "target": "axe"}
""").strip()


def build_user_prompt(
    step: int,
    obs: dict,
    last_reward: float,
    history: List[str],
    task_id: int,
) -> str:
    agent = obs.get("agent_stats", {})
    resources = obs.get("local_resources", {})
    anomalies = obs.get("nearby_anomalies", [])
    actions = obs.get("available_actions", [])
    inventory = agent.get("inventory", {})

    history_block = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(f"""
        Task ID: {task_id}
        Step: {step}/{obs.get('max_steps', '?')}
        Tick: {obs.get('tick', 0)}
        Last reward: {last_reward:.2f}

        Agent: {agent.get('agent_id')} at ({agent.get('x')},{agent.get('y')})
        Health: {agent.get('health')} | Energy: {agent.get('energy')} | Hunger: {agent.get('hunger')}
        Inventory: {json.dumps(inventory)}
        Gathering level: {agent.get('gathering_level')} | Crafting level: {agent.get('crafting_level')}

        Local resources: {json.dumps(resources)}
        Nearby anomalies: {len(anomalies)} (types: {[a.get('anomaly_type') for a in anomalies]})
        Available actions: {actions}

        Recent history:
        {history_block}

        Choose your next action as JSON.
    """).strip()


# ─── LLM call ────────────────────────────────────────────────────────────────

def get_model_action(
    client: OpenAI,
    step: int,
    obs: dict,
    last_reward: float,
    history: List[str],
    task_id: int,
) -> dict:
    user_prompt = build_user_prompt(step, obs, last_reward, history, task_id)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback heuristic action
        return _heuristic_action(obs)


def _heuristic_action(obs: dict) -> dict:
    """Simple rule-based fallback when LLM is unavailable."""
    agent = obs.get("agent_stats", {})
    agent_id = agent.get("agent_id", "agent_1")
    available = obs.get("available_actions", [])
    inventory = agent.get("inventory", {})

    if agent.get("hunger", 100) < 40 and "eat" in available:
        return {"agent_id": agent_id, "action_type": "eat"}
    if agent.get("energy", 100) < 20 and "rest" in available:
        return {"agent_id": agent_id, "action_type": "rest"}
    if "gather" in available:
        return {"agent_id": agent_id, "action_type": "gather"}
    if "attack" in available and obs.get("nearby_anomalies"):
        return {"agent_id": agent_id, "action_type": "attack"}
    if "craft" in available:
        # Try to craft something useful
        if inventory.get("wood", 0) >= 2 and inventory.get("stone", 0) >= 3:
            return {"agent_id": agent_id, "action_type": "craft", "target": "axe"}
        if inventory.get("crystal", 0) >= 5 and inventory.get("iron", 0) >= 5:
            return {"agent_id": agent_id, "action_type": "craft", "target": "void_stabilizer"}
    import random
    direction = random.choice(["up", "down", "left", "right"])
    return {"agent_id": agent_id, "action_type": "move", "target": direction}


# ─── Run one task episode ─────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: int) -> float:
    log_start(task=str(task_id), env=BENCHMARK, model=MODEL_NAME)

    reset_data = env_reset(task_id)
    obs = reset_data.get("observation", {})
    last_reward = 0.0
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    done = obs.get("done", False)

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        _check_timeout()  # enforce 20-min guard

        action = get_model_action(client, step, obs, last_reward, history, task_id)
        action_str = json.dumps(action, separators=(",", ":"))

        error: Optional[str] = None
        try:
            result = env_step(action)
            obs = result.get("observation", obs)
            last_reward = result.get("reward", 0.0)
            done = result.get("done", False)
            info = result.get("info", {})
            if not info.get("success", True):
                error = info.get("msg", "action failed")
        except Exception as exc:
            error = str(exc)
            last_reward = 0.0

        rewards.append(last_reward)
        steps_taken = step
        history.append(f"step={step} action={action_str} reward={last_reward:.2f}")

        log_step(step=step, action=action_str, reward=last_reward, done=done, error=error)

    # Get final grader score
    try:
        grader = env_grader(task_id)
        final_score = grader.get("score", 0.0)
        success = grader.get("success", False)
    except Exception as exc:
        print(f"[DEBUG] Grader failed: {exc}", flush=True)
        final_score = 0.0
        success = False

    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
    return final_score


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-key")

    all_scores: List[float] = []
    for task_id in TASK_IDS:
        try:
            score = run_task(client, task_id)
        except TimeoutError as e:
            print(f"[TIMEOUT] task={task_id} {e}", flush=True)
            score = 0.0
        all_scores.append(score)
        print(f"[SUMMARY] task={task_id} score={score:.3f}", flush=True)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[SUMMARY] overall_avg={avg:.3f} tasks={len(all_scores)}", flush=True)


if __name__ == "__main__":
    main()
