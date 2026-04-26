from typing import Any, Dict, Optional, Tuple
from survival_world import SurvivalWorld
from models import (
    SurvivalObservation, SurvivalWorldState, AgentStats,
    SurvivalAnomaly, SurvivalActionType, CommunityInfo, BuildingInfo,
)

class SurvivalEnv:
    def __init__(self):
        self.world = SurvivalWorld()
        self.max_steps = 999999  # Infinite — the game never forcibly ends
        self.score = 0.0
        self._initialize_game()

    def _initialize_game(self):
        for i in range(6):
            self.world.add_agent(f"agent_{i+1}")

    def _get_agent_stats(self, agent_id: str) -> AgentStats:
        a = self.world.agents[agent_id]
        return AgentStats(
            agent_id=a.agent_id, x=a.x, y=a.y,
            health=round(a.health), energy=round(a.energy),
            hunger=round(a.hunger), age=a.age, max_age=a.max_age,
            generation=a.generation, parent_ids=a.parent_ids,
            traits=a.traits.to_dict(), inventory=a.inventory.copy(),
            gathering_level=a.gathering_level, crafting_level=a.crafting_level,
            combat_level=round(a.combat_level, 1), xp=a.xp,
            community_id=a.community_id, alive=a.alive,
            kills=a.kills, items_crafted=a.items_crafted,
            resources_gathered=a.resources_gathered,
        )

    def _get_survival_world_state(self) -> SurvivalWorldState:
        agents = {aid: self._get_agent_stats(aid) for aid in self.world.agents}
        anomalies = [
            SurvivalAnomaly(anomaly_id=a.anomaly_id, anomaly_type=a.anomaly_type,
                            x=a.x, y=a.y, severity=a.severity, health=a.health)
            for a in self.world.anomalies
        ]
        global_res = {f"{rx},{ry}": rtype for (rx, ry), rtype in self.world.resources.items()}
        biome_map = {f"{bx},{by}": btype for (bx, by), btype in self.world.biome_map.items()}
        communities = {
            cid: CommunityInfo(**c.to_dict()) for cid, c in self.world.communities.items()
        }
        buildings = [BuildingInfo(**b.to_dict()) for b in self.world.buildings]

        alive_count = sum(1 for a in self.world.agents.values() if a.alive)
        done = alive_count == 0

        return SurvivalWorldState(
            tick=self.world.tick, is_day=self.world.is_day,
            season=self.world.season, weather=self.world.weather,
            agents=agents, anomalies=anomalies,
            communities=communities, buildings=buildings,
            biome_map=biome_map, global_resources=global_res,
            map_size={"width": self.world.width, "height": self.world.height},
            event_log=self.world.event_log[-15:],
            total_population=alive_count,
            total_born=self.world.total_born,
            total_died=self.world.total_died,
            score=self.score, done=done,
        )

    def state_for_agent(self, agent_id: str) -> SurvivalObservation:
        agent = self.world.agents[agent_id]
        local_res = {}
        for (rx, ry), rtype in self.world.resources.items():
            if abs(agent.x - rx) <= 3 and abs(agent.y - ry) <= 3:
                local_res[f"{rx},{ry}"] = rtype
        nearby_ano = [
            SurvivalAnomaly(anomaly_id=a.anomaly_id, anomaly_type=a.anomaly_type,
                            x=a.x, y=a.y, severity=a.severity, health=a.health)
            for a in self.world.anomalies
            if abs(agent.x - a.x) <= 4 and abs(agent.y - a.y) <= 4
        ]
        actions: list[SurvivalActionType] = ["move", "rest", "noop", "eat", "attack",
                                              "form_community", "join_community", "share"]
        if (agent.x, agent.y) in self.world.resources:
            actions.append("gather")
        if any(agent.inventory.get(r, 0) >= 2 for r in ("wood", "stone", "iron")):
            actions.append("craft")
        if any(agent.inventory.get(f"{t}_kit", 0) > 0 for t in ("shelter", "farm", "wall")):
            actions.append("build")

        return SurvivalObservation(
            agent_stats=self._get_agent_stats(agent_id),
            local_resources=local_res, nearby_anomalies=nearby_ano,
            available_actions=actions, tick=self.world.tick,
            done=not agent.alive,
        )

    @property
    def state(self) -> SurvivalWorldState:
        return self._get_survival_world_state()

    def reset(self) -> SurvivalObservation:
        self.world = SurvivalWorld()
        self._initialize_game()
        self.score = 0.0
        return self.state_for_agent("agent_1")

    def step(self, action_dict: Dict[str, Any]) -> Tuple[SurvivalObservation, float, bool, Dict[str, Any]]:
        agent_id = action_dict.get("agent_id", "agent_1")
        action_type = action_dict.get("action_type", "noop")
        target = action_dict.get("target")
        params = action_dict.get("params", {})

        success, msg = self.world.process_action(agent_id, action_type, target, params)
        self.world.step()

        reward = 0.0
        if success:
            rewards = {"gather": 1.0, "craft": 5.0, "attack": 3.0, "build": 8.0,
                       "form_community": 10.0, "eat": 0.5}
            reward = rewards.get(action_type, 0.0)

        agent = self.world.agents.get(agent_id)
        if agent and not agent.alive:
            reward -= 50.0
        self.score += reward

        if agent_id in self.world.agents:
            obs = self.state_for_agent(agent_id)
        else:
            obs = self.state_for_agent(list(self.world.agents.keys())[0])
        obs.reward_update = reward

        alive_count = sum(1 for a in self.world.agents.values() if a.alive)
        done = alive_count == 0
        return obs, reward, done, {"msg": msg, "success": success}
