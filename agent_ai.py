"""
Neural-network-driven AI for AnomalyCraft Survival agents.

Each agent carries a NeuralPolicy (small 2-layer net, numpy only).
On death the policy is scored by survival ticks and stored in the
CollectiveBrain gene pool.  New agents inherit crossover + mutated
weights from the best survivors, so the population genuinely improves.
"""

from typing import Dict, Any
from neural_policy import (
    NeuralPolicy,
    extract_features,
    action_to_command,
    get_brain,
)

# agent_id -> NeuralPolicy  (live policies for current agents)
_policies: Dict[str, NeuralPolicy] = {}


def get_or_create_policy(agent_id: str) -> NeuralPolicy:
    """Return existing policy or spawn a new one from the gene pool."""
    if agent_id not in _policies:
        _policies[agent_id] = get_brain().spawn_policy()
    return _policies[agent_id]


def decide_action(agent: Dict[str, Any], world_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point called every AI tick.
    1. Extract feature vector from agent + world state
    2. Forward-pass through the agent's neural policy
    3. Translate the chosen action index into a game command
    """
    agent_id = agent["agent_id"]
    policy   = get_or_create_policy(agent_id)

    features    = extract_features(agent, world_state)
    action_idx  = policy.choose_action(features)
    return action_to_command(action_idx, agent, world_state)


def on_agent_death(agent_id: str, survival_ticks: int) -> None:
    """
    Called when an agent dies.
    Records the policy fitness in the collective brain and removes it
    from the live pool.
    """
    if agent_id in _policies:
        brain = get_brain()
        brain.record_death(_policies[agent_id], survival_ticks)
        del _policies[agent_id]


def on_generation_end() -> None:
    """Called when all agents die (world restart)."""
    brain = get_brain()
    brain.new_generation()
    _policies.clear()


def get_nn_stats() -> Dict:
    """Return learning stats for the UI."""
    return get_brain().stats()
