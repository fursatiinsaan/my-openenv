"""
AnomalyCraft Survival — Grader module.
Deterministic, per-task scoring that returns a score in [0.0, 1.0].
Each grader evaluates partial progress so reward signal is dense, not sparse.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from survival_env import SurvivalEnv


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def grade_task_101(env: "SurvivalEnv") -> dict:
    """
    Task 101 — First Steps (easy)
    Score = 0.5 * (resources_gathered / 5) + 0.5 * (ticks_survived / 30)
    """
    world = env.world
    cond = env.task_conditions

    total_gathered = sum(
        a.resources_gathered for a in world.agents.values()
    )
    ticks_survived = world.tick

    gather_score = _clamp(total_gathered / 5.0)
    survive_score = _clamp(ticks_survived / 30.0)

    score = 0.5 * gather_score + 0.5 * survive_score

    alive = sum(1 for a in world.agents.values() if a.alive)
    success = total_gathered >= 5 and ticks_survived >= 30 and alive >= 1

    return {
        "task_id": 101,
        "score": round(_clamp(score), 4),
        "success": success,
        "breakdown": {
            "resources_gathered": total_gathered,
            "gather_score": round(gather_score, 4),
            "ticks_survived": ticks_survived,
            "survive_score": round(survive_score, 4),
            "alive_agents": alive,
        },
    }


def grade_task_102(env: "SurvivalEnv") -> dict:
    """
    Task 102 — Craft and Survive (medium)
    Score = 0.4 * (items_crafted / 2) + 0.4 * (ticks_survived / 80) + 0.2 * alive_bonus
    """
    world = env.world

    total_crafted = sum(a.items_crafted for a in world.agents.values())
    ticks_survived = world.tick
    alive = sum(1 for a in world.agents.values() if a.alive)

    craft_score = _clamp(total_crafted / 2.0)
    survive_score = _clamp(ticks_survived / 80.0)
    alive_bonus = _clamp(alive / max(1, len(world.agents)))

    score = 0.4 * craft_score + 0.4 * survive_score + 0.2 * alive_bonus

    success = total_crafted >= 2 and ticks_survived >= 80 and alive >= 1

    return {
        "task_id": 102,
        "score": round(_clamp(score), 4),
        "success": success,
        "breakdown": {
            "items_crafted": total_crafted,
            "craft_score": round(craft_score, 4),
            "ticks_survived": ticks_survived,
            "survive_score": round(survive_score, 4),
            "alive_agents": alive,
            "alive_bonus": round(alive_bonus, 4),
        },
    }


def grade_task_103(env: "SurvivalEnv") -> dict:
    """
    Task 103 — Anomaly Outbreak (hard)
    Score = 0.35 * (anomalies_destroyed / 1) + 0.35 * (ticks_survived / 120)
            + 0.3 * (alive_agents / 2)
    """
    world = env.world

    anomalies_destroyed = env.anomalies_destroyed
    ticks_survived = world.tick
    alive = sum(1 for a in world.agents.values() if a.alive)

    destroy_score = _clamp(anomalies_destroyed / 1.0)
    survive_score = _clamp(ticks_survived / 120.0)
    alive_score = _clamp(alive / 2.0)

    score = 0.35 * destroy_score + 0.35 * survive_score + 0.3 * alive_score

    success = anomalies_destroyed >= 1 and ticks_survived >= 120 and alive >= 2

    return {
        "task_id": 103,
        "score": round(_clamp(score), 4),
        "success": success,
        "breakdown": {
            "anomalies_destroyed": anomalies_destroyed,
            "destroy_score": round(destroy_score, 4),
            "ticks_survived": ticks_survived,
            "survive_score": round(survive_score, 4),
            "alive_agents": alive,
            "alive_score": round(alive_score, 4),
        },
    }


def grade_task_104(env: "SurvivalEnv") -> dict:
    """
    Task 104 — Build a Civilization (expert)
    Score = 0.25 * community_score + 0.25 * building_score
            + 0.25 * population_score + 0.25 * survive_score
    """
    world = env.world

    has_community = 1.0 if len(world.communities) >= 1 else 0.0
    buildings_built = len(world.buildings)
    population = sum(1 for a in world.agents.values() if a.alive)
    ticks_survived = world.tick

    community_score = has_community
    building_score = _clamp(buildings_built / 2.0)
    population_score = _clamp(population / 8.0)
    survive_score = _clamp(ticks_survived / 200.0)

    score = (
        0.25 * community_score
        + 0.25 * building_score
        + 0.25 * population_score
        + 0.25 * survive_score
    )

    success = (
        len(world.communities) >= 1
        and buildings_built >= 2
        and population >= 8
        and ticks_survived >= 200
    )

    return {
        "task_id": 104,
        "score": round(_clamp(score), 4),
        "success": success,
        "breakdown": {
            "communities": len(world.communities),
            "community_score": round(community_score, 4),
            "buildings_built": buildings_built,
            "building_score": round(building_score, 4),
            "population": population,
            "population_score": round(population_score, 4),
            "ticks_survived": ticks_survived,
            "survive_score": round(survive_score, 4),
        },
    }


def grade_task_105(env: "SurvivalEnv") -> dict:
    """
    Task 105 — Winter Siege (nightmare)
    Score = 0.25 * survive_score + 0.25 * pop_score
            + 0.25 * destroy_score + 0.25 * shelter_score
    """
    world = env.world

    ticks_survived = world.tick
    alive = sum(1 for a in world.agents.values() if a.alive)
    anomalies_destroyed = env.anomalies_destroyed
    has_shelter = any(b.building_type == "shelter" for b in world.buildings)

    survive_score = _clamp(ticks_survived / 60.0)
    pop_score = _clamp(alive / 4.0)
    destroy_score = _clamp(anomalies_destroyed / 3.0)
    shelter_score = 1.0 if has_shelter else 0.0

    score = (
        0.25 * survive_score
        + 0.25 * pop_score
        + 0.25 * destroy_score
        + 0.25 * shelter_score
    )

    success = (
        ticks_survived >= 60
        and alive >= 4
        and anomalies_destroyed >= 3
        and has_shelter
    )

    return {
        "task_id": 105,
        "score": round(_clamp(score), 4),
        "success": success,
        "breakdown": {
            "ticks_survived": ticks_survived,
            "survive_score": round(survive_score, 4),
            "alive_agents": alive,
            "pop_score": round(pop_score, 4),
            "anomalies_destroyed": anomalies_destroyed,
            "destroy_score": round(destroy_score, 4),
            "has_shelter": has_shelter,
            "shelter_score": shelter_score,
        },
    }


GRADERS = {
    101: grade_task_101,
    102: grade_task_102,
    103: grade_task_103,
    104: grade_task_104,
    105: grade_task_105,
}


def grade(env: "SurvivalEnv", task_id: int) -> dict:
    """Run the grader for the given task_id. Returns score in [0.0, 1.0]."""
    fn = GRADERS.get(task_id)
    if fn is None:
        return {"task_id": task_id, "score": 0.0, "success": False, "error": "Unknown task_id"}
    return fn(env)
