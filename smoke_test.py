"""
AnomalyCraft Survival — Smoke test.
Verifies all OpenEnv-required endpoints respond correctly.
"""

from app import app
from tasks import SURVIVAL_TASKS


def check(name: str, response, expected_status: int = 200):
    print(f"  {name}: {response.status_code}", end="")
    if response.status_code != expected_status:
        print(f"  ✗ FAILED (expected {expected_status})")
        raise SystemExit(f"{name} failed with status {response.status_code}")
    print("  ✓")


def run_shared_routes(client):
    print("\n── Shared routes ──")
    check("GET /", client.get("/"))
    check("GET /health", client.get("/health"))
    check("GET /metadata", client.get("/metadata"))
    check("GET /schema", client.get("/schema"))


def run_survival_routes(client):
    print("\n── Survival routes ──")

    # Tasks
    r = client.get("/survival/tasks")
    check("GET /survival/tasks", r)
    tasks = r.get_json()["tasks"]
    assert len(tasks) == 5, f"Expected 5 tasks, got {len(tasks)}"
    print(f"    tasks: {[t['id'] for t in tasks]}")

    # Test each task reset + step + grader
    for task in SURVIVAL_TASKS:
        tid = task["id"]
        print(f"\n  Task {tid} ({task['difficulty']}) — {task['title']}")

        # Reset
        r = client.post(f"/survival/reset?task_id={tid}")
        check(f"  POST /survival/reset?task_id={tid}", r)
        reset_data = r.get_json()
        obs = reset_data["observation"]
        assert obs["current_task_id"] == tid, "task_id mismatch in observation"
        assert obs["agent_stats"]["alive"] is True
        print(f"    agent_1 at ({obs['agent_stats']['x']},{obs['agent_stats']['y']})")

        # State
        r = client.get("/survival/state")
        check(f"  GET /survival/state", r)
        state = r.get_json()
        assert state["current_task_id"] == tid

        # Agent-specific state
        r = client.get("/survival/state?agent_id=agent_1")
        check(f"  GET /survival/state?agent_id=agent_1", r)

        # Step — gather or noop
        available = obs.get("available_actions", [])
        action_type = "gather" if "gather" in available else "noop"
        r = client.post(
            "/survival/step",
            json={"agent_id": "agent_1", "action_type": action_type},
        )
        check(f"  POST /survival/step ({action_type})", r)
        step_data = r.get_json()
        reward = step_data["reward"]
        done = step_data["done"]
        print(f"    reward={reward:.2f}  done={done}")

        # Move step
        r = client.post(
            "/survival/step",
            json={"agent_id": "agent_2", "action_type": "move", "target": "right"},
        )
        check(f"  POST /survival/step (move)", r)

        # Grader
        r = client.get(f"/survival/grader?task_id={tid}")
        check(f"  GET /survival/grader?task_id={tid}", r)
        grader = r.get_json()
        score = grader["score"]
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range"
        print(f"    grader score={score:.4f}  success={grader['success']}")
        print(f"    breakdown={grader['breakdown']}")


def main():
    client = app.test_client()

    print("=" * 50)
    print("AnomalyCraft Survival — Smoke Test")
    print("=" * 50)

    run_shared_routes(client)
    run_survival_routes(client)

    print("\n" + "=" * 50)
    print("✓ All smoke tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
