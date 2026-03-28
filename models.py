from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class StepAction(BaseModel):
    action_type: Literal["report", "noop"] = Field(
        default="report",
        description="Type of action the agent wants to take.",
    )
    content: str = Field(
        default="",
        description='Issue text for report actions. Example: report("sql injection").',
    )


class Observation(BaseModel):
    task_title: str
    task_domain: str
    task_objective: str
    code: str
    feedback: str
    found: List[str]
    remaining: List[str]
    step_count: int
    score: float
    done: bool
    reward: Optional[float] = None
    history: List[str] = Field(default_factory=list)


class EnvironmentState(BaseModel):
    episode_id: str
    task_id: int
    task_title: str
    task_domain: str
    task_objective: str
    task_difficulty: str
    step_count: int
    max_steps: int
    score: float
    found: List[str]
    remaining: List[str]
    history: List[str]
    done: bool


class ResetResponse(BaseModel):
    observation: Observation
    code: str


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class MetadataResponse(BaseModel):
    name: str
    version: str
    description: str
    framework: str
    endpoints: Dict[str, str]
    task_count: int
    supports_web_ui: bool
