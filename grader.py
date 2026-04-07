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
        return 0.5

    normalized_score = score / total_weight
    bounded_score = max(0.01, min(0.99, normalized_score))
    return round(bounded_score, 2)
