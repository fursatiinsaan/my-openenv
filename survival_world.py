import random
import math
from typing import Dict, List, Optional, Tuple

# ─── Biomes ───
BIOMES = ["plains", "forest", "desert", "mountain", "swamp"]
BIOME_RESOURCES = {
    "plains":   ["wood", "stone", "berry"],
    "forest":   ["wood", "wood", "berry", "mushroom"],
    "desert":   ["stone", "iron", "crystal"],
    "mountain": ["stone", "iron", "iron", "crystal"],
    "swamp":    ["mushroom", "berry", "crystal"],
}
BIOME_COLORS = {
    "plains": "#4a7c3f", "forest": "#2d5a1e",
    "desert": "#c2a64e", "mountain": "#7a7a7a", "swamp": "#3a5c3a",
}
SEASONS = ["spring", "summer", "autumn", "winter"]
SEASON_TICKS = 60  # ticks per season

# ─── Traits (genetic, heritable) ───
class Traits:
    def __init__(self, speed=1.0, strength=1.0, intelligence=1.0, endurance=1.0):
        self.speed = round(speed, 2)
        self.strength = round(strength, 2)
        self.intelligence = round(intelligence, 2)
        self.endurance = round(endurance, 2)

    def mutate(self):
        return Traits(
            speed=max(0.5, self.speed + random.uniform(-0.15, 0.15)),
            strength=max(0.5, self.strength + random.uniform(-0.15, 0.15)),
            intelligence=max(0.5, self.intelligence + random.uniform(-0.15, 0.15)),
            endurance=max(0.5, self.endurance + random.uniform(-0.15, 0.15)),
        )

    @staticmethod
    def crossover(a, b):
        child = Traits(
            speed=(a.speed + b.speed) / 2,
            strength=(a.strength + b.strength) / 2,
            intelligence=(a.intelligence + b.intelligence) / 2,
            endurance=(a.endurance + b.endurance) / 2,
        )
        return child.mutate()

    def to_dict(self):
        return {"speed": self.speed, "strength": self.strength,
                "intelligence": self.intelligence, "endurance": self.endurance}

# ─── Agent ───
class Agent:
    _counter = 0

    def __init__(self, agent_id: str, x: int, y: int, traits: Traits = None,
                 generation: int = 0, parent_ids: List[str] = None):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.health = 100.0
        self.max_health = 100.0
        self.energy = 100.0
        self.hunger = 100.0  # 100 = full, 0 = starving
        self.age = 0
        self.max_age = random.randint(400, 600)
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.traits = traits or Traits(
            speed=random.uniform(0.8, 1.2), strength=random.uniform(0.8, 1.2),
            intelligence=random.uniform(0.8, 1.2), endurance=random.uniform(0.8, 1.2),
        )
        self.inventory: Dict[str, int] = {}
        self.gathering_level = 1
        self.crafting_level = 1
        self.combat_level = 1
        self.xp = 0
        self.community_id: Optional[str] = None
        self.mate_cooldown = 0
        self.alive = True
        self.kills = 0
        self.items_crafted = 0
        self.resources_gathered = 0

    def add_xp(self, amount: int):
        self.xp += int(amount * self.traits.intelligence)
        threshold = self.gathering_level * 15
        if self.xp >= threshold:
            self.gathering_level += 1
            self.xp -= threshold

    def die(self):
        self.alive = False
        self.health = 0

    def to_dict(self):
        return {
            "agent_id": self.agent_id, "x": self.x, "y": self.y,
            "health": round(self.health), "energy": round(self.energy),
            "hunger": round(self.hunger), "age": self.age, "max_age": self.max_age,
            "generation": self.generation, "parent_ids": self.parent_ids,
            "traits": self.traits.to_dict(), "inventory": self.inventory.copy(),
            "gathering_level": self.gathering_level, "crafting_level": self.crafting_level,
            "combat_level": self.combat_level, "xp": self.xp,
            "community_id": self.community_id, "alive": self.alive,
            "kills": self.kills, "items_crafted": self.items_crafted,
            "resources_gathered": self.resources_gathered,
        }

# ─── Community ───
class Community:
    def __init__(self, community_id: str, founder_id: str, x: int, y: int):
        self.community_id = community_id
        self.name = f"Tribe-{community_id[:4].upper()}"
        self.members: List[str] = [founder_id]
        self.shared_resources: Dict[str, int] = {}
        self.buildings: List[Dict] = []
        self.territory_x = x
        self.territory_y = y
        self.founded_tick = 0

    def to_dict(self):
        return {
            "community_id": self.community_id, "name": self.name,
            "members": self.members[:], "shared_resources": self.shared_resources.copy(),
            "buildings": self.buildings[:], "territory_x": self.territory_x,
            "territory_y": self.territory_y,
        }

# ─── Building ───
class Building:
    def __init__(self, building_type: str, x: int, y: int, community_id: str):
        self.building_type = building_type
        self.x = x
        self.y = y
        self.community_id = community_id
        self.health = {"shelter": 50, "farm": 30, "wall": 80, "workshop": 40}.get(building_type, 30)
        self.max_health = self.health

    def to_dict(self):
        return {"type": self.building_type, "x": self.x, "y": self.y,
                "community_id": self.community_id, "health": self.health}

# ─── Anomaly / Hostile ───
class Anomaly:
    def __init__(self, anomaly_id: str, anomaly_type: str, x: int, y: int, severity: int):
        self.anomaly_id = anomaly_id
        self.anomaly_type = anomaly_type
        self.x = x
        self.y = y
        self.severity = severity
        self.health = severity * 15

# ─── World ───
class SurvivalWorld:
    def __init__(self, width: int = 24, height: int = 24):
        self.width = width
        self.height = height
        self.tick = 0
        self.is_day = True
        self.season = "spring"
        self.weather = "clear"
        self.agents: Dict[str, Agent] = {}
        self.dead_agents: List[str] = []
        self.anomalies: List[Anomaly] = []
        self.resources: Dict[Tuple[int, int], str] = {}
        self.biome_map: Dict[Tuple[int, int], str] = {}
        self.communities: Dict[str, Community] = {}
        self.buildings: List[Building] = []
        self.event_log: List[str] = []
        self.total_born = 0
        self.total_died = 0
        self._next_id = 100
        self.recipes = {
            "pickaxe": {"wood": 3, "stone": 2},
            "axe": {"wood": 2, "stone": 3},
            "sword": {"wood": 1, "iron": 4},
            "shield": {"iron": 3, "wood": 2},
            "shelter_kit": {"wood": 8, "stone": 4},
            "farm_kit": {"wood": 4, "berry": 2},
            "wall_kit": {"stone": 10, "iron": 2},
            "void_stabilizer": {"crystal": 5, "iron": 5},
            "healing_potion": {"mushroom": 3, "berry": 2},
            "energy_elixir": {"mushroom": 2, "crystal": 1},
        }
        self._generate_biomes()
        self._generate_map()

    def _generate_biomes(self):
        # Voronoi-ish biome generation
        seeds = [(random.randint(0, self.width-1), random.randint(0, self.height-1), random.choice(BIOMES)) for _ in range(8)]
        for x in range(self.width):
            for y in range(self.height):
                closest = min(seeds, key=lambda s: abs(s[0]-x) + abs(s[1]-y))
                self.biome_map[(x, y)] = closest[2]

    def _generate_map(self):
        for x in range(self.width):
            for y in range(self.height):
                if random.random() < 0.25:
                    biome = self.biome_map.get((x, y), "plains")
                    choices = BIOME_RESOURCES.get(biome, ["wood", "stone"])
                    self.resources[(x, y)] = random.choice(choices)

    def _new_id(self):
        self._next_id += 1
        return f"agent_{self._next_id}"

    def add_agent(self, agent_id: str, traits: Traits = None, generation: int = 0, parent_ids=None):
        x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
        a = Agent(agent_id, x, y, traits=traits, generation=generation, parent_ids=parent_ids or [])
        self.agents[agent_id] = a
        return a

    def spawn_anomaly(self):
        x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
        types = ["Void Storm", "Temporal Rift"]
        if not self.is_day:
            types += ["Void Creep"] * 3
        t = random.choice(types)
        self.anomalies.append(Anomaly(f"ano_{self.tick}_{x}_{y}", t, x, y, random.randint(1, 3)))

    # ─── Main step ───
    def step(self):
        self.tick += 1
        self.is_day = (self.tick % 100) < 55
        self.season = SEASONS[(self.tick // SEASON_TICKS) % 4]

        # Weather
        if self.tick % 20 == 0:
            w = ["clear", "clear", "rain", "wind"]
            if self.season == "winter": w += ["snow", "blizzard"]
            if self.season == "summer": w += ["heatwave"]
            self.weather = random.choice(w)

        alive_agents = [a for a in self.agents.values() if a.alive]

        for agent in alive_agents:
            agent.age += 1
            agent.mate_cooldown = max(0, agent.mate_cooldown - 1)

            # Hunger drain (very slow — agents can survive ~500+ ticks without eating)
            hunger_drain = 0.15
            if self.season == "winter": hunger_drain = 0.25
            if self.weather == "blizzard": hunger_drain = 0.4
            agent.hunger = max(0, agent.hunger - hunger_drain / agent.traits.endurance)

            # Energy drain (very slow — resting recovers fast)
            energy_drain = 0.25
            if self.weather in ("blizzard", "heatwave"): energy_drain = 0.5
            agent.energy = max(0, agent.energy - energy_drain / agent.traits.endurance)

            # Auto energy regen during day
            if self.is_day:
                agent.energy = min(100, agent.energy + 0.15)
                agent.health = min(agent.max_health, agent.health + 0.1)  # natural heal

            # Auto-eat if hungry and has food
            if agent.hunger < 50:
                for food, restore in [("berry", 15), ("mushroom", 12)]:
                    if agent.inventory.get(food, 0) > 0:
                        agent.inventory[food] -= 1
                        agent.hunger = min(100, agent.hunger + restore)
                        break

            # Starvation (gentle)
            if agent.hunger <= 0:
                agent.health -= 0.5
            # Exhaustion
            if agent.energy <= 0:
                agent.health -= 0.3

            # Shelter bonus
            in_shelter = any(b.building_type == "shelter" and abs(b.x - agent.x) <= 1 and abs(b.y - agent.y) <= 1 for b in self.buildings)
            if in_shelter and not self.is_day:
                agent.health = min(agent.max_health, agent.health + 0.5)
                agent.energy = min(100, agent.energy + 0.5)

            # Farm bonus (auto food near farms)
            near_farm = any(b.building_type == "farm" and abs(b.x - agent.x) <= 1 and abs(b.y - agent.y) <= 1 for b in self.buildings)
            if near_farm and self.season in ("spring", "summer"):
                agent.hunger = min(100, agent.hunger + 0.3)

            # Aging death
            if agent.age >= agent.max_age:
                agent.die()
                self.total_died += 1
                self.event_log.append(f"[{self.tick}] {agent.agent_id} died of old age (gen {agent.generation})")
                continue

            # Health death
            if agent.health <= 0:
                agent.die()
                self.total_died += 1
                self.event_log.append(f"[{self.tick}] {agent.agent_id} perished")

        # Anomaly AI
        for ano in self.anomalies[:]:
            ano.severity += 0.05
            if ano.anomaly_type == "Void Creep":
                nearest = min(alive_agents, key=lambda a: abs(a.x-ano.x)+abs(a.y-ano.y), default=None)
                if nearest:
                    if ano.x < nearest.x: ano.x += 1
                    elif ano.x > nearest.x: ano.x -= 1
                    elif ano.y < nearest.y: ano.y += 1
                    elif ano.y > nearest.y: ano.y -= 1
                    if abs(ano.x - nearest.x) <= 1 and abs(ano.y - nearest.y) <= 1:
                        dmg = max(1, int(ano.severity))
                        if nearest.inventory.get("shield", 0) > 0: dmg = max(1, dmg - 4)
                        nearest.health -= dmg
            else:
                for a in alive_agents:
                    if abs(a.x - ano.x) <= 1 and abs(a.y - ano.y) <= 1:
                        a.health -= max(1, int(ano.severity * 0.3))

        # Spawn anomalies (rare)
        chance = 0.02 if self.is_day else 0.06
        if self.season == "winter": chance += 0.02
        if random.random() < chance and len(self.anomalies) < 8:
            self.spawn_anomaly()

        # Resource respawn (generous)
        respawn_rate = 0.4 if self.season in ("spring", "summer") else 0.2
        for _ in range(3):  # Spawn up to 3 resources per tick
            if random.random() < respawn_rate:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (x, y) not in self.resources:
                    biome = self.biome_map.get((x, y), "plains")
                    self.resources[(x, y)] = random.choice(BIOME_RESOURCES.get(biome, ["wood"]))

        # Auto-reproduction check
        self._try_reproduce()

        # Clean dead anomalies
        self.anomalies = [a for a in self.anomalies if a.health > 0]
        # Clean dead agents from communities
        for c in self.communities.values():
            c.members = [m for m in c.members if m in self.agents and self.agents[m].alive]

        # Trim event log
        if len(self.event_log) > 50:
            self.event_log = self.event_log[-50:]

    def _try_reproduce(self):
        alive = [a for a in self.agents.values() if a.alive and a.age > 30 and a.mate_cooldown == 0 and a.hunger > 40 and a.health > 40]
        if len(alive) < 2:
            return
        random.shuffle(alive)
        for i in range(0, len(alive) - 1, 2):
            p1, p2 = alive[i], alive[i+1]
            if abs(p1.x - p2.x) <= 4 and abs(p1.y - p2.y) <= 4:
                child_id = self._new_id()
                child_traits = Traits.crossover(p1.traits, p2.traits)
                child = self.add_agent(child_id, traits=child_traits,
                                       generation=max(p1.generation, p2.generation) + 1,
                                       parent_ids=[p1.agent_id, p2.agent_id])
                child.x, child.y = p1.x, p1.y
                p1.mate_cooldown = 30
                p2.mate_cooldown = 30
                p1.energy -= 15
                p2.energy -= 15
                self.total_born += 1
                self.event_log.append(f"[{self.tick}] {child_id} born (gen {child.generation}) from {p1.agent_id} + {p2.agent_id}")
                if p1.community_id:
                    child.community_id = p1.community_id
                    self.communities[p1.community_id].members.append(child_id)

    # ─── Actions ───
    def process_action(self, agent_id: str, action_type: str, target: str = None, params: dict = None) -> Tuple[bool, str]:
        if agent_id not in self.agents:
            return False, "Agent not found"
        agent = self.agents[agent_id]
        if not agent.alive:
            return False, "Agent is dead"

        if action_type == "move":
            dx, dy = 0, 0
            if target == "up": dy = -1
            elif target == "down": dy = 1
            elif target == "left": dx = -1
            elif target == "right": dx = 1
            nx, ny = agent.x + dx, agent.y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                agent.x, agent.y = nx, ny
                agent.energy = max(0, agent.energy - 1.5 / agent.traits.speed)
                return True, f"Moved to {nx},{ny}"
            return False, "Out of bounds"

        elif action_type == "gather":
            if (agent.x, agent.y) in self.resources:
                res = self.resources[(agent.x, agent.y)]
                amt = max(1, int(agent.gathering_level * agent.traits.strength))
                if agent.inventory.get("pickaxe", 0) > 0 and res in ("stone", "iron", "crystal"): amt += 2
                if agent.inventory.get("axe", 0) > 0 and res == "wood": amt += 2
                agent.inventory[res] = agent.inventory.get(res, 0) + amt
                agent.energy = max(0, agent.energy - 4)
                agent.add_xp(5)
                agent.resources_gathered += amt
                del self.resources[(agent.x, agent.y)]
                return True, f"Gathered {amt} {res}"
            return False, "Nothing here"

        elif action_type == "eat":
            foods = {"berry": 15, "mushroom": 12}
            for food, restore in foods.items():
                if agent.inventory.get(food, 0) > 0:
                    agent.inventory[food] -= 1
                    agent.hunger = min(100, agent.hunger + restore)
                    return True, f"Ate {food} (+{restore} hunger)"
            if agent.inventory.get("healing_potion", 0) > 0:
                agent.inventory["healing_potion"] -= 1
                agent.health = min(agent.max_health, agent.health + 30)
                agent.hunger = min(100, agent.hunger + 10)
                return True, "Drank healing potion"
            return False, "No food"

        elif action_type == "craft":
            recipe = target
            if recipe not in self.recipes:
                return False, "Unknown recipe"
            reqs = self.recipes[recipe]
            for r, amt in reqs.items():
                if agent.inventory.get(r, 0) < amt:
                    return False, f"Need {amt} {r}"
            for r, amt in reqs.items():
                agent.inventory[r] -= amt
            agent.inventory[recipe] = agent.inventory.get(recipe, 0) + 1
            agent.crafting_level += 1
            agent.items_crafted += 1
            agent.add_xp(10)
            agent.energy = max(0, agent.energy - 8)
            return True, f"Crafted {recipe}"

        elif action_type == "build":
            btype = target
            kit = f"{btype}_kit"
            if agent.inventory.get(kit, 0) <= 0:
                return False, f"Need {kit}"
            agent.inventory[kit] -= 1
            b = Building(btype, agent.x, agent.y, agent.community_id or "none")
            self.buildings.append(b)
            agent.add_xp(15)
            self.event_log.append(f"[{self.tick}] {agent.agent_id} built a {btype}")
            return True, f"Built {btype}"

        elif action_type == "rest":
            agent.energy = min(100, agent.energy + 15 * agent.traits.endurance)
            agent.health = min(agent.max_health, agent.health + 3)
            return True, "Rested"

        elif action_type == "form_community":
            cid = f"c_{self.tick}_{agent.agent_id}"
            c = Community(cid, agent.agent_id, agent.x, agent.y)
            c.founded_tick = self.tick
            self.communities[cid] = c
            agent.community_id = cid
            self.event_log.append(f"[{self.tick}] {agent.agent_id} founded {c.name}")
            return True, f"Founded {c.name}"

        elif action_type == "join_community":
            cid = target
            if cid in self.communities:
                c = self.communities[cid]
                if agent.agent_id not in c.members:
                    c.members.append(agent.agent_id)
                agent.community_id = cid
                return True, f"Joined {c.name}"
            return False, "Community not found"

        elif action_type == "share":
            # Share resources with community
            if not agent.community_id or agent.community_id not in self.communities:
                return False, "Not in community"
            c = self.communities[agent.community_id]
            res_type = target
            if agent.inventory.get(res_type, 0) > 0:
                amt = agent.inventory[res_type]
                agent.inventory[res_type] = 0
                c.shared_resources[res_type] = c.shared_resources.get(res_type, 0) + amt
                return True, f"Shared {amt} {res_type}"
            return False, "Nothing to share"

        elif action_type == "attack":
            # Attack a nearby anomaly or agent
            for ano in self.anomalies:
                if abs(agent.x - ano.x) <= 1 and abs(agent.y - ano.y) <= 1:
                    dmg = int(agent.combat_level * agent.traits.strength * 5)
                    if agent.inventory.get("sword", 0) > 0: dmg += 10
                    if agent.inventory.get("void_stabilizer", 0) > 0 and ano.anomaly_type == "Void Storm": dmg += 30
                    ano.health -= dmg
                    agent.energy = max(0, agent.energy - 8)
                    agent.combat_level += 0.1
                    if ano.health <= 0:
                        agent.kills += 1
                        agent.add_xp(30)
                        self.event_log.append(f"[{self.tick}] {agent.agent_id} destroyed {ano.anomaly_type}")
                        return True, f"Destroyed {ano.anomaly_type}!"
                    return True, f"Attacked {ano.anomaly_type} (hp:{int(ano.health)})"
            return False, "Nothing to attack"

        elif action_type == "attack_agent":
            target_id = target
            if target_id in self.agents:
                target_agent = self.agents[target_id]
                if target_agent.alive and abs(agent.x - target_agent.x) <= 1 and abs(agent.y - target_agent.y) <= 1:
                    dmg = int(agent.combat_level * agent.traits.strength * 8)
                    if agent.inventory.get("sword", 0) > 0: dmg += 15
                    if target_agent.inventory.get("shield", 0) > 0: dmg = max(1, dmg - 10)
                    target_agent.health -= dmg
                    agent.energy = max(0, agent.energy - 10)
                    agent.combat_level += 0.2
                    if target_agent.health <= 0:
                        agent.kills += 1
                        agent.add_xp(50)
                        self.event_log.append(f"[{self.tick}] {agent.agent_id} murdered {target_agent.agent_id}!")
                        return True, f"Murdered {target_id}"
                    self.event_log.append(f"[{self.tick}] {agent.agent_id} attacked {target_agent.agent_id} for {dmg} dmg")
                    return True, f"Attacked {target_id}"
            return False, "Target not in range or invalid"

        elif action_type == "noop":
            return True, "Idle"

        return False, "Unknown action"
