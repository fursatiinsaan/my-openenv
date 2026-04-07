import os
from threading import Lock
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, session


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

from env import CodeReviewEnv
from grader import grade
from models import EnvironmentState, MetadataResponse, ResetResponse, StepAction, StepResponse
from tasks import TASKS

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "openenv-local-secret")
PORT = int(os.getenv("PORT", "8000"))
ENVIRONMENTS = {}
ENVIRONMENTS_LOCK = Lock()


def get_env():
    env_id = session.get("env_id")
    if env_id is None:
        env_id = str(uuid4())
        session["env_id"] = env_id

    with ENVIRONMENTS_LOCK:
        env = ENVIRONMENTS.get(env_id)
        if env is None:
            env = CodeReviewEnv()
            ENVIRONMENTS[env_id] = env
    return env


def compare_lists(user_actions, ai_actions):
    if not ai_actions:
        return 0.0
    overlap = len(set(user_actions) & set(ai_actions))
    return round(overlap / len(set(ai_actions)), 2)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "env": "openenv-code-review"})


@app.route("/metadata")
def metadata():
    payload = MetadataResponse(
        name="openenv-code-review",
        version="1.1.0",
        description="An OpenEnv training arena with realistic engineering scenarios, deterministic issue labels, and a polished review dashboard.",
        framework="flask+pydantic",
        endpoints={
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "health": "/health",
            "schema": "/schema",
            "tasks": "/tasks",
        },
        task_count=len(TASKS),
        supports_web_ui=True,
    )
    return jsonify(payload.model_dump())


@app.route("/schema")
def schema():
    return jsonify(
        {
            "action_schema": StepAction.model_json_schema(),
            "reset_schema": ResetResponse.model_json_schema(),
            "step_schema": StepResponse.model_json_schema(),
            "state_schema": EnvironmentState.model_json_schema(),
        }
    )


@app.route("/tasks")
def tasks():
    return jsonify({"tasks": TASKS})


@app.route("/reset", methods=["GET", "POST"])
def reset():
    payload = request.get_json(silent=True) or {} if request.method == "POST" else {}
    raw_task_id = payload.get("task_id", request.args.get("task_id", 0))
    env = get_env()
    try:
        observation = env.reset(int(raw_task_id))
    except (TypeError, ValueError, IndexError):
        return jsonify({"error": f"Unknown task_id: {raw_task_id}"}), 400
    return jsonify(ResetResponse(observation=observation, code=observation.code).model_dump())


@app.route("/state")
def state():
    env = get_env()
    return jsonify(env.state.model_dump())


@app.route("/step", methods=["POST"])
def step():
    payload = request.get_json(silent=True) or {}
    env = get_env()
    action = StepAction(
        action_type=payload.get("action_type", "report"),
        content=payload.get("content", ""),
    )

    if action.action_type == "noop":
        action_text = "noop()"
    elif action.content.startswith("report("):
        action_text = action.content
    else:
        action_text = f'report("{action.content}")'

    observation, reward, done, info = env.step(action_text)
    return jsonify(
        StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        ).model_dump()
    )


@app.route("/grader")
def grader_route():
    env = get_env()
    return jsonify({"score": grade(env.task, env.state)})


@app.route("/auto_ai")
def auto_ai():
    env = get_env()
    return jsonify(env.run_agent())


@app.route("/compare", methods=["POST"])
def compare():
    payload = request.get_json(silent=True) or {}
    return jsonify(
        {
            "accuracy": compare_lists(
                payload.get("user", []),
                payload.get("ai", []),
            )
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
