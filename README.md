---
title: OpenEnv
emoji: "🤖"
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# OpenEnv Scenario Lab

OpenEnv Scenario Lab is a lightweight benchmark for evaluating engineering agents on realistic product-review scenarios. Each task presents a production-style code snippet, a concrete objective, and deterministic issue labels so agents can be graded on security, reliability, and data-quality findings.

## Why This Feels More Like A Real Environment

- `reset`, `step`, and `state` APIs for interactive evaluation
- Typed Pydantic models for actions, observations, and environment state
- Eleven tasks spanning easy, hard, very hard, and extreme difficulty
- Realistic domains across backend, web, data, messaging, orchestration, and ML systems
- Root-level `inference.py` baseline using an OpenAI-compatible client
- Reward shaping plus lightweight in-memory agent context during baseline runs
- Polished web UI with mission brief, task tags, live issue tracker, AI compare mode, and score visualization
- Dockerfile and `openenv.yaml` for packaging and submission

## What The Agent Does

Each scenario is designed to feel closer to an internal engineering review arena than a toy benchmark. The agent is expected to:

- inspect a realistic code snippet
- identify the highest-impact issue first
- submit short issue labels such as `sql injection` or `path traversal`
- accumulate score as valid findings are discovered

That makes the project useful both as a benchmark environment and as a compact demo of agent-evaluation design.

## Scenario Mix

- Easy onboarding tasks for quick sanity checks and demos
- Hard production scenarios for backend and systems review
- Very hard and extreme scenarios for ML, orchestration, platform, and data security

This mix makes the environment easier to demo while still feeling serious enough for agent training.

## Project Structure

```text
OpenEv/
├── app.py
├── env.py
├── grader.py
├── inference.py
├── memory.py
├── models.py
├── openenv.yaml
├── smoke_test.py
├── tasks.py
├── Dockerfile
├── requirements.txt
└── templates/
    └── index.html
```

## API Endpoints

- `GET /health` returns service status
- `GET /metadata` returns environment metadata
- `GET /schema` returns JSON schemas for action, reset, step, and state
- `GET /tasks` returns the task catalog used by the UI and evaluators
- `GET|POST /reset` resets the environment to a specific task
- `POST /step` applies a `report` or `noop` action
- `GET /state` returns the current environment state
- `GET /grader` returns a normalized score for the current episode
- `GET /auto_ai` runs the baseline agent through the active task

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`, then run the app:

```bash
python3 app.py
```

Open:

- `http://localhost:8000`
- `http://localhost:8000/health`
- `http://localhost:8000/tasks`
- `http://localhost:8000/state`

## Runtime Notes

The container now uses `gunicorn` instead of Flask's development server:

```bash
gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 4 app:app
```

This is the same command used by the Docker image and is a better fit for Hugging Face Spaces or any simple container deployment.

## Inference Setup

`app.py` auto-loads `.env` on startup. The baseline agent expects:

- `API_BASE_URL`
- `MODEL_NAME`
- `API_KEY`

If those values are missing or a provider request fails, the baseline safely falls back to `noop()` and surfaces the error in the UI. During baseline execution, the agent uses lightweight in-memory context for the current run only; it does not persist cross-run memory to disk.

## Example API Usage

Reset the first task:

```bash
curl "http://localhost:8000/reset?task_id=0"
```

Submit a step:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"report","content":"sql injection"}'
```

Read environment state:

```bash
curl http://localhost:8000/state
```

## Smoke Test

Run a quick local verification before deploying:

```bash
python3 smoke_test.py
```

This checks `/health`, `/metadata`, `/tasks`, `/reset`, `/state`, and a sample `/step`.

## Demo Flow

1. Start the server with `python3 app.py`.
2. Open `http://localhost:8000`.
3. Pick an easy scenario first to verify the interaction loop.
4. Submit one or two manual findings to see score and progress update.
5. Run the baseline agent with `Run AI` to compare its path with your own.
6. Switch to a harder scenario to show multi-step review behavior.

## UI Highlights

- Scenario-driven landing section instead of a plain form
- Task metadata, tags, and objective surfaced beside the code
- Live review progress with pending and found issue states
- Compare panel for user-vs-agent overlap instead of only a browser alert
- Session summary cards that make demos easier to follow

## Remaining External Steps

- Deploy to Hugging Face Spaces
- Confirm the public Space returns `200` on `/health`, `/reset`, `/step`, and `/state`
- Run the official validator before final submission
