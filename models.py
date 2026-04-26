"""
AnomalyCraft Survival — Pydantic models.
Typed schemas for all OpenEnv actions, observations, and state.
Models inherit from openenv.core base types for full compliance.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import Field

from openenv_core.env_server.types import (
    Action as OpenEnvAction,
    Observation as OpenEnvObservation,
    State as OpenEnvState,
)

SurvivalActionType = Literal[
    "move", "gather", "craft", "rest", "attack", "eat",
    "build", "form_community", "join_community", "share",
    "attack_agent", "noop",
]


class SurvivalAction(OpenEnvAction):
    model_config = {"extra": "allow", "validate_assignment": True, "arbitrary_types_allowed": True}
    agent_id: str
    action_type: SurvivalActionType
    target: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class AgentStats(OpenEnvObservation):
    model_config = {"extra": "allow", "validate_assignment": True, "arbitrary_types_allowed": True}
    agent_id: str
    x: int
    y: int
    health: int
    energy: int
    hunger: int
    age: int
    max_age: int
    generation: int
    parent_ids: List[str] = Field(default_factory=list)
    traits: Dict[str, float] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    personality: str = "peaceful"
    messages: List[str] = Field(default_factory=list)
    loved_one: Optional[str] = None
    bond_strength: float = 0.0
    inventory: Dict[str, int] = Field(default_factory=dict)
    gathering_level: int = 1
    crafting_level: int = 1
    combat_level: float = 1.0
    xp: int = 0
    community_id: Optional[str] = None
    alive: bool = True
    kills: int = 0
    items_crafted: int = 0
    resources_gathered: int = 0


class SurvivalAnomaly(OpenEnvObservation):
    model_config = {"extra": "allow", "validate_assignment": True, "arbitrary_types_allowed": True}
    anomaly_id: str
    anomaly_type: str
    x: int
    y: int
    severity: float
    health: float


class CommunityInfo(OpenEnvState):
    model_config = {"extra": "allow", "validate_assignment": True, "arbitrary_types_allowed": True}
    community_id: str
    name: str
    members: List[str]
    shared_resources: Dict[str, int] = Field(default_factory=dict)
    buildings: List[Dict] = Field(default_factory=list)
    territory_x: int
    territory_y: int


class BuildingInfo(OpenEnvState):
    model_config = {"extra": "allow", "validate_assignment": True, "arbitrary_types_allowed": True}
    type: str
    x: int
    y: int
    community_id: str
    health: int


class SurvivalObservation(OpenEnvObservation):
    model_config = {"extra": "allow", "validate_assignment": True, "arbitrary_types_allowed": True}
    agent_stats: AgentStats
    local_resources: Dict[str, str] = Field(default_factory=dict)
    nearby_anomalies: List[SurvivalAnomaly] = Field(default_factory=list)
    available_actions: List[SurvivalActionType] = Field(default_factory=list)
    tick: int = 0
    reward_update: float = 0.0
    current_task_id: int = 101
    step_count: int = 0
    max_steps: int = 999999


class SurvivalWorldState(OpenEnvState):
    tick: int = 0
    is_day: bool = True
    season: str = "spring"
    weather: str = "clear"
    agents: Dict[str, AgentStats] = Field(default_factory=dict)
    anomalies: List[SurvivalAnomaly] = Field(default_factory=list)
    communities: Dict[str, CommunityInfo] = Field(default_factory=dict)
    buildings: List[BuildingInfo] = Field(default_factory=list)
    biome_map: Dict[str, str] = Field(default_factory=dict)
    global_resources: Dict[str, str] = Field(default_factory=dict)
    map_size: Dict[str, int] = Field(default_factory=lambda: {"width": 48, "height": 48})
    event_log: List[str] = Field(default_factory=list)
    total_population: int = 0
    total_born: int = 0
    total_died: int = 0
    score: float = 0.0
    current_task_id: int = 101
    max_steps: int = 999999
    anomalies_destroyed: int = 0
    generation_number: int = 0


from pydantic import BaseModel


class SurvivalResetResponse(BaseModel):
    observation: SurvivalObservation
    world_state: SurvivalWorldState


class SurvivalStepResponse(BaseModel):
    observation: SurvivalObservation
    world_state: SurvivalWorldState
    reward: float
    done: bool
    info: Dict[str, object] = Field(default_factory=dict)


class GraderResponse(BaseModel):
    task_id: int
    score: float
    success: bool
    breakdown: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class MetadataResponse(BaseModel):
    name: str
    version: str
    description: str
    framework: str
    endpoints: Dict[str, str]
    task_count: int
    supports_web_ui: bool
