MEMORY_LIMIT = 20
_MEMORY = {
    "good": [],
    "bad": [],
}


def load_memory():
    return {
        "good": list(_MEMORY["good"]),
        "bad": list(_MEMORY["bad"]),
    }


def save_memory(memory):
    _MEMORY["good"] = list(memory.get("good", []))[-MEMORY_LIMIT:]
    _MEMORY["bad"] = list(memory.get("bad", []))[-MEMORY_LIMIT:]


def clear_memory():
    save_memory({"good": [], "bad": []})


def update_memory(action, reward):
    memory = load_memory()

    entry = {
        "action": action,
        "reward": reward,
    }

    if reward > 0:
        memory["good"].append(entry)
    else:
        memory["bad"].append(entry)

    save_memory(memory)
