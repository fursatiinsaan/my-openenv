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

An OpenEnv-style benchmark for training and evaluating engineering agents on realistic product scenarios. The environment pairs production-like software snippets with deterministic issue labels, then rewards agents for surfacing the most important security, reliability, and data-quality findings.

## Why This Feels More Like A Real Environment

- `reset`, `step`, and `state` APIs for interactive evaluation
- Typed Pydantic models for actions, observations, and environment state
- Eleven tasks spanning easy, hard, very hard, and extreme difficulty
- Realistic domains across backend, web, data, messaging, orchestration, and ML systems
- Root-level `inference.py` baseline using an OpenAI-compatible client
- Reward shaping plus lightweight agent memory for repeated practice
- Polished web UI with mission brief, task tags, live issue tracker, AI compare mode, and score visualization
- Dockerfile and `openenv.yaml` for packaging and submission

## Product Framing

This project is designed to feel closer to an internal engineering training arena than a toy benchmark. Each scenario has:

- a believable production story
- a concrete review objective
- deterministic grading targets
- a fast manual workflow and an AI baseline workflow

That makes it useful both as a hackathon submission and as a portfolio project that demonstrates agent evaluation design.

## Benchmark Mix

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

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`, then run:

```bash
python3 app.py
```

Open:

- `http://localhost:8000`
- `http://localhost:8000/health`
- `http://localhost:8000/tasks`
- `http://localhost:8000/state`

## Production Run

The container now uses `gunicorn` instead of Flask's development server:

```bash
gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 4 app:app
```

This is the same command used by the Docker image and is a better fit for Hugging Face Spaces.

## Inference Setup

`app.py` auto-loads `.env` on startup. The baseline agent expects:

- `API_BASE_URL`
- `MODEL_NAME`
- `API_KEY`

For local manual testing, `HF_TOKEN` is also accepted as a fallback, but the validator-provided `API_KEY` is preferred so inference runs through the required proxy.

If those values are missing or a provider request fails, the baseline safely falls back to `noop()` and surfaces the error in the UI.

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

1. Start the server with `python3 app.py`
2. Open `http://localhost:8000`
3. Choose an easy task to verify the loop quickly
4. Use the task tags, mission brief, and issue tracker to review manually
5. Run `Run AI` to compare the baseline agent against your own path
6. Switch to an extreme task to demonstrate harder multi-step review

## Hackathon Readiness

This repo now covers the local pieces of the Round 1 checklist:

- `openenv.yaml` present
- `inference.py` present at project root
- Typed models present in `models.py`
- `reset`, `step`, and `state` implemented
- 3+ realistic tasks with graders
- Dockerfile included
- Polished UI for local demos
- Pinned dependencies for reproducible deploys
- Production-ready `gunicorn` container entrypoint

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
