import random
import math
from typing import Dict, List, Optional, Tuple

# Neural policy callbacks (imported lazily to avoid circular imports)
def _nn_on_death(agent_id: str, ticks: int):
    try:
        from agent_ai import on_agent_death
        on_agent_death(agent_id, ticks)
    except Exception:
        pass

def _nn_on_generation_end():
    try:
        from agent_ai import on_generation_end
        on_generation_end()
    except Exception:
        pass

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
        """Random mutation - genetic variation."""
        return Traits(
            speed=max(0.5, self.speed + random.uniform(-0.15, 0.15)),
            strength=max(0.5, self.strength + random.uniform(-0.15, 0.15)),
            intelligence=max(0.5, self.intelligence + random.uniform(-0.15, 0.15)),
            endurance=max(0.5, self.endurance + random.uniform(-0.15, 0.15)),
        )

    @staticmethod
    def crossover(a, b):
        """Sexual reproduction - combine parent traits."""
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


# ─── Memory (learned, not inherited) ───
class Memory:
    """Agent memory - learned behaviors that improve with experience."""
    def __init__(self):
        # Resource preferences (learned from success)
        self.resource_preference = {}  # resource_type -> success_rate
        # Danger awareness (learned from damage)
        self.danger_memory = {}  # location -> danger_level
        # Crafting knowledge (learned from crafting)
        self.known_recipes = set()
        # Combat experience (learned from fights)
        self.combat_wins = 0
        self.combat_losses = 0
        # Social bonds (learned from interactions)
        self.trusted_agents = set()
        self.rival_agents = set()
    
    def learn_resource(self, resource_type: str, success: bool):
        """Learn which resources are valuable."""
        if resource_type not in self.resource_preference:
            self.resource_preference[resource_type] = 0.5
        # Update preference based on success
        if success:
            self.resource_preference[resource_type] = min(1.0, self.resource_preference[resource_type] + 0.1)
        else:
            self.resource_preference[resource_type] = max(0.0, self.resource_preference[resource_type] - 0.05)
    
    def learn_danger(self, x: int, y: int, damage: float):
        """Remember dangerous locations."""
        key = f"{x},{y}"
        if key not in self.danger_memory:
            self.danger_memory[key] = 0.0
        self.danger_memory[key] = min(1.0, self.danger_memory[key] + damage / 100.0)
    
    def learn_recipe(self, recipe: str):
        """Remember successful crafts."""
        self.known_recipes.add(recipe)
    
    def learn_combat(self, won: bool):
        """Learn from combat experience."""
        if won:
            self.combat_wins += 1
        else:
            self.combat_losses += 1
    
    def get_danger_level(self, x: int, y: int) -> float:
        """Check if location is dangerous."""
        key = f"{x},{y}"
        return self.danger_memory.get(key, 0.0)
    
    def get_combat_confidence(self) -> float:
        """How confident in combat (0.0 - 1.0)."""
        total = self.combat_wins + self.combat_losses
        if total == 0:
            return 0.5
        return self.combat_wins / total
    
    def to_dict(self):
        return {
            "resource_preference": self.resource_preference.copy(),
            "danger_memory": self.danger_memory.copy(),
            "known_recipes": list(self.known_recipes),
            "combat_wins": self.combat_wins,
            "combat_losses": self.combat_losses,
            "trusted_agents": list(self.trusted_agents),
            "rival_agents": list(self.rival_agents),
        }

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
        self.memory = Memory()  # NEW: Learning system
        # PERSONALITY TYPES (affects behavior)
        self.personality = random.choice(["aggressive", "peaceful", "explorer", "builder"])
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
        self.messages: List[str] = []  # Communication messages
        self.loved_one: Optional[str] = None  # The one they love and seek
        self.bond_strength: float = 0.0  # How strong is the bond (0-1)

    def add_xp(self, amount: int):
        self.xp += int(amount * self.traits.intelligence)
        threshold = self.gathering_level * 15
        if self.xp >= threshold:
            self.gathering_level += 1
            self.xp -= threshold

    def die(self):
        """Agent dies and contributes knowledge to collective memory."""
        self.alive = False
        self.health = 0
        # Record death location as dangerous
        death_location = f"{self.x},{self.y}"
        # Record what killed them (for learning)
        return {
            "location": death_location,
            "age": self.age,
            "generation": self.generation,
            "traits": self.traits.to_dict(),
            "personality": self.personality,
            "danger_memory": self.memory.danger_memory.copy(),
            "resources_gathered": self.resources_gathered,
            "items_crafted": self.items_crafted,
            "kills": self.kills,
        }

    def learn_from_action(self, action_type: str, success: bool, context: dict = None):
        """Learn from experience - FASTER learning for real survival."""
        learning_rate = self.traits.intelligence * 2.0  # DOUBLED learning rate
        context = context or {}
        
        if action_type == "gather" and success:
            resource = context.get("resource")
            if resource:
                # Learn MUCH faster which resources are good
                if resource not in self.memory.resource_preference:
                    self.memory.resource_preference[resource] = 0.5
                self.memory.resource_preference[resource] = min(1.0, 
                    self.memory.resource_preference[resource] + 0.2)  # Was 0.1
        
        elif action_type == "craft" and success:
            recipe = context.get("recipe")
            if recipe:
                self.memory.learn_recipe(recipe)
        
        elif action_type == "attack":
            won = context.get("won", False)
            self.memory.learn_combat(won)
        
        # Learn danger IMMEDIATELY from any damage
        if context.get("damage_taken", 0) > 0:
            damage = context["damage_taken"]
            # Remember danger strongly
            self.memory.learn_danger(self.x, self.y, damage * 2.0)  # DOUBLED danger memory

    def to_dict(self):
        return {
            "agent_id": self.agent_id, "x": self.x, "y": self.y,
            "health": round(self.health), "energy": round(self.energy),
            "hunger": round(self.hunger), "age": self.age, "max_age": self.max_age,
            "generation": self.generation, "parent_ids": self.parent_ids,
            "traits": self.traits.to_dict(), 
            "memory": self.memory.to_dict(),
            "personality": self.personality,  # NEW
            "messages": self.messages[-5:],  # Last 5 messages
            "loved_one": self.loved_one,  # Who they love
            "bond_strength": round(self.bond_strength, 2),  # How much they love
            "inventory": self.inventory.copy(),
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
        self.health = {"shelter": 50, "farm": 30, "wall": 80, "workshop": 40, "tower": 100, "mine": 60, "temple": 120, "market": 50, "hospital": 40, "campfire": 20}.get(building_type, 30)
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
        self.damage_dealt = 0.0   # fitness tracker
        # Neural policy — spawned from the anomaly gene pool
        try:
            from neural_policy import get_anomaly_brain
            self.policy = get_anomaly_brain().spawn_policy()
        except Exception:
            self.policy = None

# ─── World ───
class SurvivalWorld:
    def __init__(self, width: int = 48, height: int = 48):  # DOUBLED: 48x48 instead of 24x24
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
        # EVOLUTION MEMORY: Carry knowledge across generations
        self.generation_number = 0
        self.collective_memory = {
            "dangerous_locations": {},  # Locations that killed agents
            "safe_locations": {},  # Locations where agents survived long
            "successful_strategies": [],  # What worked
            "failed_strategies": [],  # What didn't work
            "best_traits": {"speed": 1.0, "strength": 1.0, "intelligence": 1.0, "endurance": 1.0},
        }
        self.recipes = {
            # Basic tools (Tier 1)
            "pickaxe": {"wood": 3, "stone": 2},
            "axe": {"wood": 2, "stone": 3},
            # Combat (Tier 1)
            "sword": {"wood": 1, "iron": 4},
            "shield": {"iron": 3, "wood": 2},
            "bow": {"wood": 5, "iron": 2},
            "arrow": {"wood": 1, "stone": 1},
            # Armor (Tier 1)
            "helmet": {"iron": 3, "wood": 1},
            "chestplate": {"iron": 5, "wood": 2},
            "boots": {"iron": 2, "wood": 1},
            # Buildings (Tier 1)
            "shelter_kit": {"wood": 8, "stone": 4},
            "farm_kit": {"wood": 4, "berry": 2},
            "wall_kit": {"stone": 10, "iron": 2},
            # Advanced buildings (Tier 2 - requires tech)
            "tower_kit": {"stone": 15, "iron": 8, "wood": 5},
            "mine_kit": {"stone": 12, "iron": 10, "pickaxe": 1},
            "temple_kit": {"stone": 20, "crystal": 5, "iron": 5},
            "workshop_kit": {"wood": 10, "stone": 8, "iron": 4},
            "market_kit": {"wood": 6, "stone": 6, "iron": 3},
            "hospital_kit": {"wood": 8, "stone": 4, "mushroom": 5},
            # Advanced tools (Tier 2)
            "iron_pickaxe": {"iron": 5, "wood": 2, "pickaxe": 1},
            "iron_axe": {"iron": 5, "wood": 2, "axe": 1},
            "crystal_sword": {"crystal": 3, "iron": 6, "sword": 1},
            "crossbow": {"wood": 8, "iron": 4, "bow": 1},
            # Advanced armor (Tier 2)
            "iron_helmet": {"iron": 5, "crystal": 1, "helmet": 1},
            "iron_chestplate": {"iron": 8, "crystal": 1, "chestplate": 1},
            "iron_boots": {"iron": 4, "crystal": 1, "boots": 1},
            # Special items
            "void_stabilizer": {"crystal": 5, "iron": 5},
            "healing_potion": {"mushroom": 3, "berry": 2},
            "energy_elixir": {"mushroom": 2, "crystal": 1},
            "trade_token": {"crystal": 1, "iron": 2},
            "teleport_scroll": {"crystal": 3, "mushroom": 2},
            "fire_starter": {"wood": 2, "stone": 1, "iron": 1},
            "fishing_rod": {"wood": 4, "iron": 1},
            "campfire_kit": {"wood": 5, "stone": 3},
            # Food items
            "cooked_meat": {"berry": 1, "wood": 1},  # Placeholder, need meat
            "bread": {"berry": 3, "wood": 1},
            "stew": {"mushroom": 2, "berry": 2, "wood": 1},
        }
        # Technology tree (unlocked recipes)
        self.unlocked_tech = set([
            "pickaxe", "axe", "sword", "shield", "bow", "arrow", "helmet", "chestplate", "boots",
            "shelter_kit", "farm_kit", "wall_kit", "tower_kit", "mine_kit", "temple_kit", "workshop_kit",
            "market_kit", "hospital_kit", "iron_pickaxe", "iron_axe", "crystal_sword", "crossbow",
            "iron_helmet", "iron_chestplate", "iron_boots", "void_stabilizer", "healing_potion",
            "energy_elixir", "trade_token", "teleport_scroll", "fire_starter", "fishing_rod",
            "campfire_kit", "cooked_meat", "bread", "stew"
        ])
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
        
        # SMARTER AGENTS: Use collective memory to spawn in safe locations
        if self.collective_memory["safe_locations"]:
            # Prefer safe locations
            safe_locs = list(self.collective_memory["safe_locations"].keys())
            if safe_locs and random.random() < 0.7:  # 70% chance to spawn in safe area
                safe_coord = random.choice(safe_locs)
                x, y = map(int, safe_coord.split(","))
        
        # SMARTER AGENTS: Improve traits based on what worked before
        if traits is None and self.collective_memory["best_traits"]:
            best = self.collective_memory["best_traits"]
            # Start with better base traits
            traits = Traits(
                speed=best["speed"] + random.uniform(-0.1, 0.1),
                strength=best["strength"] + random.uniform(-0.1, 0.1),
                intelligence=best["intelligence"] + random.uniform(-0.1, 0.1),
                endurance=best["endurance"] + random.uniform(-0.1, 0.1),
            )
        
        a = Agent(agent_id, x, y, traits=traits, generation=generation, parent_ids=parent_ids or [])
        
        # SMARTER AGENTS: Pre-load collective danger memory
        if self.collective_memory["dangerous_locations"]:
            for loc, danger_level in self.collective_memory["dangerous_locations"].items():
                a.memory.danger_memory[loc] = danger_level
        
        self.agents[agent_id] = a
        return a
    
    def learn_from_death(self, death_data: dict):
        """Learn from agent death to make next generation smarter."""
        # Record dangerous location
        loc = death_data["location"]
        self.collective_memory["dangerous_locations"][loc] = \
            self.collective_memory["dangerous_locations"].get(loc, 0.0) + 0.5
        
        # Update best traits if agent survived long
        if death_data["age"] > 200:  # Survived a long time
            traits = death_data["traits"]
            # Blend with current best traits
            for trait, value in traits.items():
                current = self.collective_memory["best_traits"][trait]
                self.collective_memory["best_traits"][trait] = (current * 0.7 + value * 0.3)
        
        # Learn from their danger memory
        for loc, danger in death_data["danger_memory"].items():
            current = self.collective_memory["dangerous_locations"].get(loc, 0.0)
            self.collective_memory["dangerous_locations"][loc] = max(current, danger)
    
    def restart_with_smarter_agents(self):
        """Restart world with smarter agents that learned from previous deaths."""
        self.generation_number += 1

        # Notify neural network system that a generation ended
        _nn_on_generation_end()

        # Advance anomaly brain generation too
        try:
            from neural_policy import get_anomaly_brain
            get_anomaly_brain().new_generation()
        except Exception:
            pass
        
        # Keep collective memory but reset world
        old_memory = self.collective_memory.copy()
        
        # Reset world state
        self.tick = 0
        self.is_day = True
        self.season = "spring"
        self.weather = "clear"
        self.agents = {}
        self.dead_agents = []
        self.anomalies = []
        self.communities = {}
        self.buildings = []
        
        # Restore collective memory
        self.collective_memory = old_memory
        
        # Regenerate resources
        self._generate_map()
        
        # Spawn smarter agents (they inherit collective knowledge)
        num_agents = 6
        for i in range(num_agents):
            agent_id = f"agent_gen{self.generation_number}_{i+1}"
            self.add_agent(agent_id, generation=self.generation_number)
        
        brain_gen = self.generation_number
        try:
            from agent_ai import get_nn_stats
            stats = get_nn_stats()
            brain_gen = stats.get("generation", self.generation_number)
            best = stats.get("best_fitness_ever", 0)
            self.event_log.append(
                f"[RESTART] Gen {self.generation_number} | NN gen {brain_gen} | best survival: {best} ticks"
            )
        except Exception:
            self.event_log.append(f"[RESTART] Generation {self.generation_number} begins with collective wisdom!")
        
        return self.generation_number

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
                death_data = agent.die()
                self.learn_from_death(death_data)
                self.total_died += 1
                _nn_on_death(agent.agent_id, agent.age)
                self.event_log.append(f"[{self.tick}] {agent.agent_id} died of old age (gen {agent.generation})")
                continue

            # Health death
            if agent.health <= 0:
                death_data = agent.die()
                self.learn_from_death(death_data)
                self.total_died += 1
                _nn_on_death(agent.agent_id, agent.age)
                self.event_log.append(f"[{self.tick}] {agent.agent_id} perished")

        # Anomaly AI - neural policy driven, moves every 3 ticks
        if self.tick % 3 == 0 and alive_agents:
            agent_dicts = [a.to_dict() for a in alive_agents]
            for ano in self.anomalies[:]:
                ano.severity += 0.01
                if ano.policy is None:
                    continue
                try:
                    from neural_policy import extract_anomaly_features, anomaly_action_to_move
                    ano_dict = {"x": ano.x, "y": ano.y, "severity": ano.severity,
                                "is_day": 1.0 if self.is_day else 0.0}
                    feat = extract_anomaly_features(ano_dict, agent_dicts)
                    action_idx = ano.policy.choose_action(feat)
                    dx, dy = anomaly_action_to_move(action_idx, ano_dict, agent_dicts,
                                                    self.width, self.height)
                    # retreat_grow (action 3) also boosts severity
                    if action_idx == 3:
                        ano.severity += 0.05
                    ano.x = max(0, min(self.width  - 1, ano.x + dx))
                    ano.y = max(0, min(self.height - 1, ano.y + dy))
                except Exception:
                    pass

        # Anomaly damage (every tick if touching)
        for ano in self.anomalies[:]:
            dmg_mult = 0.8 if ano.anomaly_type == "Void Creep" else 0.4
            for a in alive_agents:
                if abs(a.x - ano.x) <= 1 and abs(a.y - ano.y) <= 1:
                    dmg = max(2, int(ano.severity * dmg_mult))
                    if a.inventory.get("shield", 0) > 0:
                        dmg = max(1, dmg - 4)
                    a.health -= dmg
                    ano.damage_dealt += dmg   # track fitness
                    a.learn_from_action("damaged", False, {"damage_taken": dmg})

        # Spawn anomalies (rare)
        chance = 0.01 if self.is_day else 0.03  # MUCH RARER (was 0.02/0.06)
        if self.season == "winter": chance += 0.01  # Less in winter too
        if random.random() < chance and len(self.anomalies) < 5:  # Max 5 (was 8)
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

        # AUTO-RESTART: If all agents dead, restart with smarter generation
        alive_count = sum(1 for a in self.agents.values() if a.alive)
        if alive_count == 0 and len(self.agents) > 0:
            self.event_log.append(f"[{self.tick}] ALL AGENTS DIED - Restarting with Generation {self.generation_number + 1}")
            self.restart_with_smarter_agents()
            return  # Exit early after restart

        # Record anomaly deaths into gene pool before cleaning
        for ano in self.anomalies:
            if ano.health <= 0 and ano.policy is not None:
                try:
                    from neural_policy import get_anomaly_brain
                    get_anomaly_brain().record_death(ano.policy, ano.damage_dealt)
                except Exception:
                    pass

        # Clean dead anomalies
        self.anomalies = [a for a in self.anomalies if a.health > 0]
        # Clean dead agents from communities
        for c in self.communities.values():
            c.members = [m for m in c.members if m in self.agents and self.agents[m].alive]

        # Trim event log
        if len(self.event_log) > 50:
            self.event_log = self.event_log[-50:]

    def _try_reproduce(self):
        """
        ROMANTIC SURVIVAL: Agents form bonds and seek their loved ones!
        - Agents spend time together to form bonds
        - Only reproduce with their loved one
        - Seek loved one when in danger (not just anyone)
        - Population cap at 50 to prevent crashes
        """
        alive = [a for a in self.agents.values() if a.alive and a.age > 20 and a.hunger > 30 and a.health > 20]
        if len(alive) < 2:
            return
        
        # POPULATION CAP: Don't reproduce if we have 50+ agents
        total_alive = len([a for a in self.agents.values() if a.alive])
        if total_alive >= 50:
            return
        
        # FORM BONDS: Agents near each other form emotional bonds
        for agent in alive:
            for other in alive:
                if agent.agent_id == other.agent_id:
                    continue
                
                dist = abs(agent.x - other.x) + abs(agent.y - other.y)
                
                # If close together, bond strengthens
                if dist <= 2:
                    # If no loved one yet, this could be the one!
                    if agent.loved_one is None:
                        agent.loved_one = other.agent_id
                        agent.bond_strength = 0.1
                        agent.messages.append(f"💕 Met {other.agent_id}")
                    
                    # If this is their loved one, strengthen bond
                    elif agent.loved_one == other.agent_id:
                        agent.bond_strength = min(1.0, agent.bond_strength + 0.05)
                        if agent.bond_strength > 0.5 and len(agent.messages) == 0:
                            agent.messages.append(f"❤️ Love grows with {other.agent_id}")
        
        # Check if there's danger nearby (anomalies)
        danger_nearby = len(self.anomalies) > 0
        
        # SEEK LOVED ONE: When in danger OR always wanting to be together
        for agent in alive:
            if agent.loved_one and agent.loved_one in self.agents:
                loved = self.agents[agent.loved_one]
                if not loved.alive:
                    # Loved one died - heartbreak
                    agent.messages.append(f"💔 Lost {agent.loved_one}")
                    agent.loved_one = None
                    agent.bond_strength = 0.0
                    continue
                
                dist = abs(agent.x - loved.x) + abs(agent.y - loved.y)
                
                # ALWAYS want to be near loved one (stronger bond = stronger pull)
                if dist > 1 and dist <= 8 and agent.bond_strength > 0.3:
                    # Move towards loved one
                    if agent.energy > 15:  # Need energy to move
                        dx = loved.x - agent.x
                        dy = loved.y - agent.y
                        if abs(dx) > abs(dy):
                            agent.x += 1 if dx > 0 else -1
                        else:
                            agent.y += 1 if dy > 0 else -1
                        # Clamp to world bounds
                        agent.x = max(0, min(self.width - 1, agent.x))
                        agent.y = max(0, min(self.height - 1, agent.y))
                        agent.energy -= 1
        
        # REPRODUCE: Only with loved one when bond is strong
        reproduced = set()
        for agent in alive:
            if agent.agent_id in reproduced or agent.mate_cooldown > 0:
                continue
            
            # Must have a loved one with strong bond
            if not agent.loved_one or agent.bond_strength < 0.5:
                continue
            
            if agent.loved_one not in self.agents:
                continue
            
            loved = self.agents[agent.loved_one]
            if not loved.alive or loved.mate_cooldown > 0:
                continue
            
            # Must be touching
            dist = abs(agent.x - loved.x) + abs(agent.y - loved.y)
            if dist <= 1:
                # REPRODUCE WITH LOVED ONE!
                child_id = self._new_id()
                child_traits = Traits.crossover(agent.traits, loved.traits)
                child = self.add_agent(child_id, traits=child_traits,
                                       generation=max(agent.generation, loved.generation) + 1,
                                       parent_ids=[agent.agent_id, loved.agent_id])
                child.x, child.y = agent.x, agent.y
                
                # Shorter cooldown when in danger (survival instinct)
                cooldown = 15 if danger_nearby else 25
                agent.mate_cooldown = cooldown
                loved.mate_cooldown = cooldown
                
                agent.energy -= 10
                loved.energy -= 10
                self.total_born += 1
                
                # Strengthen bond after having child together
                agent.bond_strength = min(1.0, agent.bond_strength + 0.2)
                loved.bond_strength = min(1.0, loved.bond_strength + 0.2)
                
                danger_msg = " [DANGER]" if danger_nearby else ""
                love_msg = f" ❤️ (bond: {agent.bond_strength:.1f})"
                self.event_log.append(f"[{self.tick}] {child_id} born from {agent.agent_id} + {loved.agent_id}{love_msg}{danger_msg}")
                
                agent.messages.append(f"👶 Had child with {loved.agent_id}")
                loved.messages.append(f"👶 Had child with {agent.agent_id}")
                
                if agent.community_id:
                    child.community_id = agent.community_id
                    self.communities[agent.community_id].members.append(child_id)
                
                reproduced.add(agent.agent_id)
                reproduced.add(loved.agent_id)

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
                agent.learn_from_action("gather", True, {"resource": res})  # LEARN
                del self.resources[(agent.x, agent.y)]
                return True, f"Gathered {amt} {res}"
            agent.learn_from_action("gather", False)  # LEARN from failure
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
                    agent.learn_from_action("craft", False, {"recipe": recipe})  # LEARN
                    return False, f"Need {amt} {r}"
            for r, amt in reqs.items():
                agent.inventory[r] -= amt
            agent.inventory[recipe] = agent.inventory.get(recipe, 0) + 1
            agent.crafting_level += 1
            agent.items_crafted += 1
            agent.add_xp(10)
            agent.energy = max(0, agent.energy - 8)
            agent.learn_from_action("craft", True, {"recipe": recipe})  # LEARN
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
                        agent.learn_from_action("attack", True, {"won": True})  # LEARN victory
                        self.event_log.append(f"[{self.tick}] {agent.agent_id} destroyed {ano.anomaly_type}")
                        return True, f"Destroyed {ano.anomaly_type}!"
                    agent.learn_from_action("attack", True, {"won": False})  # LEARN combat
                    return True, f"Attacked {ano.anomaly_type} (hp:{int(ano.health)})"
            agent.learn_from_action("attack", False)  # LEARN failure
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
