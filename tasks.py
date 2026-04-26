"""
AnomalyCraft Survival — Task definitions.
5 tasks ranging easy → medium → hard → expert → nightmare.
Each task defines: objective, success conditions, max_steps, reward_range, difficulty.
"""

SURVIVAL_TASKS = [
    {
        "id": 101,
        "title": "First Steps",
        "difficulty": "easy",
        "domain": "survival",
        "description": (
            "A single agent must gather at least 5 resources and survive for 30 ticks. "
            "No crafting or combat required — just explore and collect."
        ),
        "objective": "Gather 5+ resources and survive 30 ticks.",
        "success_conditions": {
            "resources_gathered_min": 5,
            "survive_ticks": 30,
        },
        "max_steps": 50,
        "reward_range": [0.0, 1.0],
        "tags": ["survival", "gathering", "single-agent"],
    },
    {
        "id": 102,
        "title": "Craft and Survive",
        "difficulty": "medium",
        "domain": "survival",
        "description": (
            "Agents must gather resources and craft at least 2 items (tools or consumables) "
            "within 80 ticks. Crafting requires collecting the right materials first."
        ),
        "objective": "Craft 2+ items and keep at least 1 agent alive for 80 ticks.",
        "success_conditions": {
            "items_crafted_min": 2,
            "survive_ticks": 80,
            "min_alive_agents": 1,
        },
        "max_steps": 100,
        "reward_range": [0.0, 1.0],
        "tags": ["survival", "crafting", "multi-agent"],
    },
    {
        "id": 103,
        "title": "Anomaly Outbreak",
        "difficulty": "hard",
        "domain": "survival",
        "description": (
            "A Void Storm anomaly has appeared. Agents must craft a void_stabilizer "
            "and destroy the anomaly before it wipes out the population. "
            "Requires coordinated gathering, crafting, and combat."
        ),
        "objective": "Destroy at least 1 anomaly and keep 2+ agents alive for 120 ticks.",
        "success_conditions": {
            "anomalies_destroyed_min": 2,
            "survive_ticks": 150,
            "min_alive_agents": 3,
        },
        "max_steps": 150,
        "reward_range": [0.0, 1.0],
        "tags": ["survival", "combat", "crafting", "multi-agent"],
    },
    {
        "id": 104,
        "title": "Build a Civilization",
        "difficulty": "expert",
        "domain": "survival",
        "description": (
            "Agents must form a community, build at least 2 structures (shelter + farm or wall), "
            "and grow the population to 8+ agents through reproduction. "
            "Requires long-horizon planning across gathering, crafting, building, and social actions."
        ),
        "objective": "Form a community, build 2+ structures, reach population 8+.",
        "success_conditions": {
            "communities_min": 1,
            "buildings_min": 3,
            "population_min": 10,
            "survive_ticks": 250,
        },
        "max_steps": 250,
        "reward_range": [0.0, 1.0],
        "tags": ["survival", "building", "community", "multi-agent", "long-horizon"],
    },
    {
        "id": 105,
        "title": "Winter Siege",
        "difficulty": "nightmare",
        "domain": "survival",
        "description": (
            "A blizzard hits during winter. Anomalies spawn at triple rate at night. "
            "Agents must survive a full winter season (60 ticks), maintain population above 4, "
            "destroy 3+ anomalies, and have a community with a shelter built before winter ends. "
            "The hardest test of multi-agent coordination under extreme pressure."
        ),
        "objective": "Survive winter (60 ticks), pop >= 4, destroy 3 anomalies, build shelter.",
        "success_conditions": {
            "survive_ticks": 60,
            "min_alive_agents": 5,
            "anomalies_destroyed_min": 5,
            "buildings_min": 2,
            "building_type_required": "shelter",
            "season_required": "winter",
        },
        "max_steps": 80,
        "reward_range": [0.0, 1.0],
        "tags": ["survival", "combat", "building", "community", "multi-agent", "extreme"],
    },
]
