from difflib import SequenceMatcher


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _found_issues(state):
    if hasattr(state, "found"):
        return state.found
    return state.get("found", [])


def grade(task, state):
    total_weight = sum(issue["weight"] for issue in task["issues"])
    score = 0.0
    found_issues = _found_issues(state)

    for issue in task["issues"]:
        label = issue["label"]
        for found in found_issues:
            if similarity(label, found) > 0.5:
                score += issue["weight"]
                break

    if total_weight == 0:
        return 0.0
    return round(score / total_weight, 2)
