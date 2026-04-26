# I Built a World Where AI Agents Fight to Survive — And They Got Scary Good

*A dev log on building AnomalyCraft Survival for the OpenEnv Hackathon 2026*

---

I didn't plan to spend three weeks obsessing over tiny pixel people running from purple blobs. But here we are.

This is the story of how I built **AnomalyCraft Survival** — a multi-agent survival game where AI agents evolve neural network brains, form communities, fall in love, build shelters, and fight monsters. And how after 1,500+ generations of training, they got genuinely good at it.

---

## The Idea

The brief was simple: build an OpenEnv-compliant training environment. An environment where AI agents can learn to do something.

Most people built grid worlds with a single agent picking up coins. I wanted something messier. Something where the agents had to *want* to survive — not just follow a reward signal.

So I asked: what if the agents had personalities? What if they formed bonds? What if the monsters also had neural networks and were *also* evolving?

That's the arms race I wanted to build.

---

## What the World Looks Like

The world is a 48×48 tile grid. Each tile has a biome — plains, forest, desert, mountain, swamp — and each biome produces different resources. Wood grows in forests. Iron hides in mountains. Crystal shows up in deserts if you're lucky.

Six agents spawn at the start of each generation. They can:

- **Gather** resources from the tile they're standing on
- **Craft** tools and items (axe, pickaxe, sword, shelter kit, healing potion...)
- **Build** structures (shelters, farms, walls, towers, temples)
- **Fight** anomalies — the hostile entities that hunt them
- **Form communities** and share resources with their tribe
- **Reproduce** — but only when they're physically touching another agent they've bonded with

That last one took a while to get right.

---

## The Agents Have Feelings (Sort Of)

Early on, agents would reproduce with whoever was closest. It looked chaotic and felt wrong. So I added a bond system.

When two agents spend time near each other, their bond strength increases. Once it crosses a threshold, one of them becomes the other's `loved_one`. From that point on, the agent will always drift towards that specific agent — not just anyone nearby.

They only reproduce with their loved one. If their loved one dies, they get a 💔 message and the bond resets.

It sounds sentimental. But it actually changed the population dynamics in a meaningful way. Agents started clustering into pairs, which naturally led to community formation. Communities built shelters. Shelters kept agents alive longer. Longer survival meant more generations of learning.

The romance was load-bearing architecture.

---

## The Neural Network

Each agent runs a small feedforward network. 22 inputs, 16 hidden neurons, 9 outputs. No GPU needed — it's pure numpy.

The 22 inputs cover everything the agent can perceive: health, energy, hunger, nearby resources, nearby anomalies, distance to loved one, distance to shelter, current weather, time of day, season.

The 9 outputs map to actions:
- Move away from danger
- Move towards nearest resource
- Move towards loved one
- Move randomly (exploration)
- Gather
- Craft or build
- Fight
- Eat or rest
- Do nothing

The network doesn't know what these actions mean. It just learns which output to fire given the inputs. The meaning comes from the environment.

---

## The Anomalies Also Have Brains

This is the part I'm most proud of.

The anomalies — Void Storms, Temporal Rifts, Void Creeps — aren't scripted. They run their own neural policies. 12 inputs, 10 hidden, 5 outputs. Their actions are: chase, flank left, flank right, retreat and grow stronger, spread damage.

Their fitness is measured by total damage dealt. The agents' fitness is measured by survival time, shelters built, resources gathered, items crafted, and kills.

Both populations evolve in parallel. When agents get better at avoiding anomalies, the anomaly gene pool adapts. When anomalies get better at flanking, agents learn to build walls.

It's a genuine arms race. By generation 1,000, both sides were doing things I didn't explicitly program.

---

## How Evolution Works

I used a simple neuroevolution setup — no backprop, no gradients.

Each agent's neural network weights are stored as a flat array. When an agent dies, its weights and fitness score go into a gene pool. The gene pool has two tiers:

- **Elite pool** — top 10 performers of all time
- **Recent pool** — last 10 generations, regardless of score (for diversity)

When a new agent spawns, its weights are created by:
- 70% chance: crossover between two elite agents
- 20% chance: crossover with a recent agent (prevents convergence)
- 10% chance: completely fresh random weights (exploration)

Then mutation is applied — small random noise added to each weight.

If the population stagnates (no improvement for 8 generations), the mutation rate resets to a higher value to shake things out. This prevented the local optima problem that killed my first few attempts.

---

## Generation 0 vs Generation 1,500

**Generation 0:** Agents wander randomly. They die within 50 ticks. No shelters. No communities. The anomalies kill them almost immediately.

**Generation 1,500:** Agents build 80+ shelters per generation. Population hits the cap of 20 within 200 ticks. Agents live to old age (400-600 ticks). Communities form within the first 100 ticks. Anomalies are actively fought off by agents with swords.

The fitness score went from ~50 to ~3,800.

That's not a tweak. That's the network actually learning what survival means.

---

## The Faces

One thing I added late that ended up mattering more than I expected: the agents have pixel art faces.

When an agent is healthy and safe, it shows a small smile. When it's near an anomaly, the eyes go wide and the mouth opens in an O. When it's hurt, the eyes become X's. When it's hungry, it frowns.

It's 8 pixels. But watching a crowd of tiny faces all go scared at the same time when an anomaly spawns — that hits different. It makes the simulation feel alive in a way that bar charts don't.

---

## The Collective Memory

When all agents die and the world restarts, the new generation doesn't start from zero.

The world maintains a **collective memory** — a map of dangerous locations (where agents died), a record of the best-performing trait values, and a list of safe spawn zones (where agents survived longest).

New agents spawn in safe zones 70% of the time. They start with traits biased towards the best performers. And their danger memory is pre-loaded with every location that killed a previous agent.

It's not perfect. But it means each generation starts slightly smarter than the last, even before the neural network kicks in.

---

## What I'd Do Differently

**The world is too big.** 48×48 tiles means agents spend a lot of time wandering before they find resources. I'd probably drop to 32×32 and increase resource density.

**The anomaly AI is too passive.** After generation 500, the anomaly gene pool converged on "chase the nearest agent." It works, but it's not interesting. I'd add more anomaly types with different objectives.

**The bond system needs more depth.** Right now bonds are just a number. I'd love to add actual social dynamics — agents defending their loved ones, grieving when they die, forming alliances between pairs.

**Training curves.** I should have logged more. I have fitness over generations but I don't have a good record of *when* specific behaviors emerged. Did shelter-building come before or after community formation? I don't know.

---

## The Stack

- **Python 3.11** — the whole thing
- **Flask** — HTTP server for the OpenEnv API
- **NumPy** — neural network math (no PyTorch, no TensorFlow)
- **Pydantic v2** — request/response validation
- **HTML Canvas** — the 2D visualization, all drawn in JavaScript
- **No external ML libraries** — the neuroevolution is hand-rolled

The whole thing runs on CPU. A generation of 1,500 ticks takes about 1.2 seconds on a MacBook Air.

---

## Try It

The environment is live on Hugging Face Spaces. Open it, click **START AUTOPLAY**, and watch.

The agents will start gathering wood within seconds. By minute two, shelters appear. By minute five, you'll see communities with shared resource pools and agents actively hunting anomalies.

If you want to mess with it: **Ctrl+Click** to drop resources anywhere on the map. **Shift+Click** to spawn an anomaly. Watch how the agents react.

The scared faces are worth it.

---

*Built for the OpenEnv Hackathon 2026. All code is open source.*

*— Nimesh*
