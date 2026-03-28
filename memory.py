import json
import os

MEMORY_FILE = "agent_memory.json"


def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return {
        "good": [],
        "bad": []
    }


def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def update_memory(action, reward):
    memory = load_memory()

    entry = {
        "action": action,
        "reward": reward
    }

    if reward > 0:
        memory["good"].append(entry)
    else:
        memory["bad"].append(entry)

    memory["good"] = memory["good"][-20:]
    memory["bad"] = memory["bad"][-20:]

    save_memory(memory)