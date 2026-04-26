"""
AnomalyCraft Survival — Environment wrapper.
Implements the OpenEnv interface: reset(), step(), state(), grade().
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

# ── Graceful fallback if openenv_core is not installed ──────────────────────
try:
    from openenv_core.env_server.interfaces import Environment as OpenEnvEnvironment
except ImportError:
    class OpenEnvEnvironment:
        def __init__(self): pass

from survival_world import SurvivalWorld
from models import (
    SurvivalObservation,
    SurvivalWorldState,
    AgentStats,
    SurvivalAnomaly,
    SurvivalActionType,
    SurvivalAction,
    CommunityInfo,
    BuildingInfo,
)
from tasks import SURVIVAL_TASKS


# Map task_id → task dict for quick lookup
_TASK_MAP: Dict[int, dict] = {t["id"]: t for t in SURVIVAL_TASKS}


class SurvivalEnv(OpenEnvEnvironment):
    """
    AnomalyCraft Survival environment — OpenEnv compliant.
    Inherits from openenv.core.env_server.interfaces.Environment.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__()
        self.world = SurvivalWorld()
        self.score = 0.0
        self.current_task_id: int = 101
        self.task_conditions: dict = {}
        self.anomalies_destroyed: int = 0
        self._step_count: int = 0
        self._initialize_game()

    # ─── Internal helpers ───────────────────────────────────────────────────

    def _initialize_game(self):
        for i in range(6):
            self.world.add_agent(f"agent_{i + 1}")

    def _get_agent_stats(self, agent_id: str) -> AgentStats:
        a = self.world.agents[agent_id]
        return AgentStats(
            agent_id=a.agent_id,
            x=a.x,
            y=a.y,
            health=round(a.health),
            energy=round(a.energy),
            hunger=round(a.hunger),
            age=a.age,
            max_age=a.max_age,
            generation=a.generation,
            parent_ids=a.parent_ids,
            traits=a.traits.to_dict(),
            memory=a.memory.to_dict(),
            personality=a.personality,
            messages=a.messages[-5:] if hasattr(a, 'messages') else [],
            loved_one=a.loved_one if hasattr(a, 'loved_one') else None,
            bond_strength=a.bond_strength if hasattr(a, 'bond_strength') else 0.0,
            inventory=a.inventory.copy(),
            gathering_level=a.gathering_level,
            crafting_level=a.crafting_level,
            combat_level=round(a.combat_level, 1),
            xp=a.xp,
            community_id=a.community_id,
            alive=a.alive,
            kills=a.kills,
            items_crafted=a.items_crafted,
            resources_gathered=a.resources_gathered,
        )

    def _get_survival_world_state(self) -> SurvivalWorldState:
        # Create a snapshot of agents to avoid dictionary changed during iteration
        agent_ids = list(self.world.agents.keys())
        agents = {aid: self._get_agent_stats(aid) for aid in agent_ids if aid in self.world.agents}
        
        anomalies = [
            SurvivalAnomaly(
                anomaly_id=a.anomaly_id,
                anomaly_type=a.anomaly_type,
                x=a.x,
                y=a.y,
                severity=a.severity,
                health=a.health,
            )
            for a in self.world.anomalies
        ]
        global_res = {
            f"{rx},{ry}": rtype for (rx, ry), rtype in self.world.resources.items()
        }
        biome_map = {
            f"{bx},{by}": btype for (bx, by), btype in self.world.biome_map.items()
        }
        communities = {
            cid: CommunityInfo(**c.to_dict())
            for cid, c in self.world.communities.items()
        }
        buildings = [BuildingInfo(**b.to_dict()) for b in self.world.buildings]

        alive_count = sum(1 for a in self.world.agents.values() if a.alive)
        task = _TASK_MAP.get(self.current_task_id, {})
        max_steps = task.get("max_steps", 999999)
        done = alive_count == 0 or self._step_count >= max_steps

        return SurvivalWorldState(
            tick=self.world.tick,
            is_day=self.world.is_day,
            season=self.world.season,
            weather=self.world.weather,
            agents=agents,
            anomalies=anomalies,
            communities=communities,
            buildings=buildings,
            biome_map=biome_map,
            global_resources=global_res,
            map_size={"width": self.world.width, "height": self.world.height},
            event_log=self.world.event_log[-15:],
            total_population=alive_count,
            total_born=self.world.total_born,
            total_died=self.world.total_died,
            score=self.score,
            done=done,
            current_task_id=self.current_task_id,
            step_count=self._step_count,
            max_steps=max_steps,
            anomalies_destroyed=self.anomalies_destroyed,
            generation_number=self.world.generation_number,
        )
    def state_for_agent(self, agent_id: str) -> SurvivalObservation:
        agent = self.world.agents[agent_id]
        local_res = {}
        for (rx, ry), rtype in self.world.resources.items():
            if abs(agent.x - rx) <= 3 and abs(agent.y - ry) <= 3:
                local_res[f"{rx},{ry}"] = rtype

        nearby_ano = [
            SurvivalAnomaly(
                anomaly_id=a.anomaly_id,
                anomaly_type=a.anomaly_type,
                x=a.x,
                y=a.y,
                severity=a.severity,
                health=a.health,
            )
            for a in self.world.anomalies
            if abs(agent.x - a.x) <= 4 and abs(agent.y - a.y) <= 4
        ]

        actions: list[SurvivalActionType] = [
            "move", "rest", "noop", "eat", "attack",
            "form_community", "join_community", "share",
        ]
        if (agent.x, agent.y) in self.world.resources:
            actions.append("gather")
        if any(agent.inventory.get(r, 0) >= 2 for r in ("wood", "stone", "iron")):
            actions.append("craft")
        if any(
            agent.inventory.get(f"{t}_kit", 0) > 0
            for t in ("shelter", "farm", "wall")
        ):
            actions.append("build")

        task = _TASK_MAP.get(self.current_task_id, {})
        max_steps = task.get("max_steps", 999999)
        done = not agent.alive or self._step_count >= max_steps

        return SurvivalObservation(
            agent_stats=self._get_agent_stats(agent_id),
            local_resources=local_res,
            nearby_anomalies=nearby_ano,
            available_actions=actions,
            tick=self.world.tick,
            done=done,
            current_task_id=self.current_task_id,
            step_count=self._step_count,
            max_steps=max_steps,
        )

    # ─── OpenEnv interface ──────────────────────────────────────────────────

    @property
    def state(self) -> SurvivalWorldState:
        return self._get_survival_world_state()

    def reset(
        self,
        task_id: int = 101,
        seed=None,
        episode_id=None,
        **kwargs,
    ) -> "SurvivalObservation":
        """Reset the environment for a specific task. Returns initial observation."""
        task = _TASK_MAP.get(task_id)
        if task is None:
            task_id = 101
            task = _TASK_MAP[task_id]

        self.current_task_id = task_id
        self.task_conditions = task.get("success_conditions", {})
        self.world = SurvivalWorld()
        self.score = 0.0
        self.anomalies_destroyed = 0
        self._step_count = 0

        # Task-specific world setup
        if task_id == 105:
            self.world.season = "winter"
            self.world.weather = "blizzard"
            self.world.tick = 180
            for _ in range(3):
                self.world.spawn_anomaly()

        self._initialize_game()
        return self.state_for_agent("agent_1")

    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> "SurvivalObservation":
        """
        Process one action and advance the world by one tick.
        Accepts SurvivalAction, dict, or any action-like object.
        """
        # Normalise action to dict
        if hasattr(action, "model_dump"):
            action_dict = action.model_dump()
        elif isinstance(action, dict):
            action_dict = action
        else:
            action_dict = {}

        obs, reward, done, info = self._step_internal(action_dict)
        obs.reward_update = reward
        obs.done = done
        return obs

    def _step_internal(
        self, action_dict: Dict[str, Any]
    ) -> Tuple[SurvivalObservation, float, bool, Dict[str, Any]]:
        """
        Internal step — returns (observation, reward, done, info) tuple.
        Used by Flask routes and the training loop.
        """
        agent_id = action_dict.get("agent_id", "agent_1")
        action_type = action_dict.get("action_type", "noop")
        target = action_dict.get("target")
        params = action_dict.get("params", {})

        # Track anomaly count before action to detect kills
        ano_before = len(self.world.anomalies)

        success, msg = self.world.process_action(agent_id, action_type, target, params)
        self.world.step()
        self._step_count += 1

        # Count anomalies destroyed this tick
        ano_after = len(self.world.anomalies)
        newly_destroyed = max(0, ano_before - ano_after)
        self.anomalies_destroyed += newly_destroyed

        # ── Dense reward shaping ──────────────────────────────────────────
        reward = 0.0

        if success:
            base_rewards = {
                "gather": 1.0,
                "craft": 5.0,
                "attack": 3.0,
                "build": 8.0,
                "form_community": 10.0,
                "join_community": 2.0,
                "share": 1.5,
                "eat": 0.5,
                "rest": 0.1,
            }
            reward += base_rewards.get(action_type, 0.0)

        # Bonus for destroying anomalies
        reward += newly_destroyed * 15.0

        # Survival bonus: small reward each tick an agent is alive
        agent = self.world.agents.get(agent_id)
        if agent and agent.alive:
            reward += 0.05
        elif agent and not agent.alive:
            reward -= 50.0  # death penalty

        # Task-specific shaping
        reward += self._task_shaped_reward(action_type, success)

        self.score += reward

        # ── Build observation ─────────────────────────────────────────────
        if agent_id in self.world.agents and self.world.agents[agent_id].alive:
            obs = self.state_for_agent(agent_id)
        else:
            alive_agents = [
                aid for aid, a in self.world.agents.items() if a.alive
            ]
            if alive_agents:
                obs = self.state_for_agent(alive_agents[0])
            else:
                obs = self.state_for_agent(list(self.world.agents.keys())[0])

        obs.reward_update = reward

        alive_count = sum(1 for a in self.world.agents.values() if a.alive)
        task = _TASK_MAP.get(self.current_task_id, {})
        max_steps = task.get("max_steps", 999999)
        done = alive_count == 0 or self._step_count >= max_steps

        return obs, reward, done, {"msg": msg, "success": success}

    def _task_shaped_reward(self, action_type: str, success: bool) -> float:
        """Extra reward shaping based on current task objectives."""
        tid = self.current_task_id
        if not success:
            return 0.0

        if tid == 101:
            # Encourage gathering
            if action_type == "gather":
                return 0.5
        elif tid == 102:
            # Encourage crafting
            if action_type == "craft":
                return 2.0
        elif tid == 103:
            # Encourage combat and crafting void_stabilizer
            if action_type == "attack":
                return 2.0
            if action_type == "craft":
                return 1.0
        elif tid == 104:
            # Encourage community building
            if action_type in ("form_community", "build"):
                return 3.0
            if action_type == "join_community":
                return 1.0
        elif tid == 105:
            # Encourage everything under pressure
            if action_type == "build":
                return 5.0
            if action_type == "attack":
                return 3.0
        return 0.0

    def grade(self) -> dict:
        """Return the grader report for the current task."""
        from grader import grade as _grade
        return _grade(self, self.current_task_id)
