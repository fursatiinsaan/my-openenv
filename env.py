import re
from difflib import SequenceMatcher
from uuid import uuid4

from inference import act, clear_last_error, get_last_error
from memory import update_memory
from models import EnvironmentState, Observation
from tasks import TASKS


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def matches_issue(text, issue):
    lowered = text.lower().strip()
    if not lowered:
        return False

    candidates = [issue["label"].replace("_", " ").lower()]
    candidates.extend(keyword.lower() for keyword in issue["keywords"])
    candidates.extend(alias.lower() for alias in issue.get("aliases", []))

    return any(
        lowered in candidate
        or candidate in lowered
        or similarity(lowered, candidate) >= 0.82
        for candidate in candidates
    )


class CodeReviewEnv:
    def __init__(self, max_steps=8):
        self.max_steps = max_steps
        self.auto_run_limit = 4
        self.task = None
        self.task_id = 0
        self._state = None
        self.reset(0)

    @property
    def state(self):
        return self._state

    def reset(self, task_id=0):
        self.task_id = int(task_id)
        self.task = TASKS[self.task_id]
        self._state = EnvironmentState(
            episode_id=str(uuid4()),
            task_id=self.task_id,
            task_title=self.task["title"],
            task_domain=self.task["domain"],
            task_objective=self.task["objective"],
            task_difficulty=self.task["difficulty"],
            step_count=0,
            max_steps=self.max_steps,
            score=0.0,
            found=[],
            remaining=[issue["label"] for issue in self.task["issues"]],
            history=[],
            done=False,
        )
        return self._obs("Environment reset. Review the snippet and report the highest-impact issue first.")

    def extract_text(self, action):
        match = re.search(r'report\("(.*?)"\)', action)
        return match.group(1).lower() if match else ""

    def step(self, action):
        if self.task is None or self._state is None:
            self.reset(self.task_id)

        self._state.step_count += 1
        text = self.extract_text(action)
        reward = 0.0
        feedback = []
        matched = False

        for issue in self.task["issues"]:
            label = issue["label"]
            if label not in self._state.remaining:
                continue
            if matches_issue(text, issue):
                self._state.found.append(label)
                self._state.remaining.remove(label)
                reward += issue["weight"]
                feedback.append(f"Found issue: {label}")
                matched = True

        if not matched and action != "noop()":
            reward -= 0.3
            feedback.append("Reported issue was not relevant to the current snippet.")

        if action in self._state.history:
            reward -= 0.5
            feedback.append("Repeated action.")

        self._state.history.append(action)
        self._state.score = round(self._state.score + reward, 2)
        self._state.done = (
            len(self._state.remaining) == 0
            or self._state.step_count >= self._state.max_steps
        )

        update_memory(action, reward)

        if not feedback:
            feedback.append("No new issues found.")

        observation = self._obs("\n".join(feedback), reward=round(reward, 2))
        return observation, round(reward, 2), self._state.done, {
            "episode_id": self._state.episode_id,
            "remaining_issues": len(self._state.remaining),
            "task_title": self._state.task_title,
        }

    def run_agent(self):
        observation = self.reset(self.task_id)
        steps = []
        clear_last_error()

        while not self._state.done and len(steps) < self.auto_run_limit:
            action = act(observation)
            if action in steps:
                break
            steps.append(action)
            observation, _, _, _ = self.step(action)

        return {
            "score": round(self._state.score, 2),
            "steps": steps,
            "found": self._state.found,
            "remaining": self._state.remaining,
            "episode_id": self._state.episode_id,
            "error": get_last_error(),
            "auto_run_limit": self.auto_run_limit,
        }

    def _obs(self, feedback, reward=None):
        return Observation(
            task_title=self.task["title"],
            task_domain=self.task["domain"],
            task_objective=self.task["objective"],
            code=self.task["code"],
            feedback=feedback,
            found=list(self._state.found),
            remaining=list(self._state.remaining),
            step_count=self._state.step_count,
            score=round(self._state.score, 2),
            done=self._state.done,
            reward=reward,
            history=list(self._state.history),
        )
