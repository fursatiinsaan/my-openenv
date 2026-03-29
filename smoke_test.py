from app import app
from tasks import TASKS


def check(name, response):
    print(f"{name}: {response.status_code}")
    if response.status_code != 200:
        raise SystemExit(f"{name} failed with status {response.status_code}")


def main():
    client = app.test_client()
    first_issue = TASKS[0]["issues"][0]["label"].replace("_", " ")

    check("/health", client.get("/health"))
    check("/metadata", client.get("/metadata"))
    check("/tasks", client.get("/tasks"))

    reset_response = client.get("/reset?task_id=0")
    check("/reset", reset_response)

    state_response = client.get("/state")
    check("/state", state_response)

    step_response = client.post(
        "/step",
        json={"action_type": "report", "content": first_issue},
    )
    check("/step", step_response)

    payload = step_response.get_json()
    print("found:", payload["observation"]["found"])
    print("reward:", payload["reward"])
    print("smoke test passed")


if __name__ == "__main__":
    main()
