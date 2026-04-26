# I Built a World Where AI Agents Fight to Survive — And They Got Scary Good

*A dev log on building AnomalyCraft Survival for the OpenEnv Hackathon 2026*

---

I didn't plan to spend three weeks obsessing over tiny pixel people running from purple blobs. But here we are.

This is the story of how I built **AnomalyCraft Survival** — a multi-agent survival game where AI agents evolve neural network brains, form communities, fall in love, build shelters, and fight monsters. And how after 1,500+ generations of training, they got genuinely good at it.

---

## The Idea

The brief was simple: build an OpenEnv-compliant training environment.

Most people built grid worlds with a single agent picking up coins. I wanted something messier. Something where the agents had to *want* to survive — not just follow a reward signal.

What if the agents had personalities? What if they formed bonds? What if the monsters also had neural networks and were *also* evolving?

That's the arms race I wanted to build.

---

## What the World Looks Like

A 48×48 tile grid. Each tile has a biome — plains, forest, desert, mountain, swamp — and each biome produces different resources. Wood grows in forests. Iron hides in mountains. Crystal shows up in deserts if you're lucky.

Six agents spawn at the start of each generation. They gather resources, craft tools, build shelters, form communities, and fight anomalies. They also reproduce — but only when they're physically touching another agent they've bonded with. That last part took a while to get right.

---

## The Agents Have Feelings (Sort Of)

Early on, agents would reproduce with whoever was closest. It looked chaotic and felt wrong. So I added a bond system.

When two agents spend time near each other, their bond strength increases. Once it crosses a threshold, one becomes the other's `loved_one`. From that point on, the agent drifts towards that specific agent — not just anyone nearby. They only reproduce with their loved one. If their loved one dies, they get a 💔 message and the bond resets.

It sounds sentimental. But it changed the population dynamics in a real way. Agents started clustering into pairs, which led to community formation. Communities built shelters. Shelters kept agents alive longer. Longer survival meant more generations of learning.

The romance was load-bearing architecture.

---

## The Neural Network

Each agent runs a small feedforward network — 22 inputs, 16 hidden neurons, 9 outputs. Pure numpy, no GPU.

The inputs cover everything the agent can perceive: health, energy, hunger, nearby resources, nearby anomalies, distance to loved one, distance to shelter, weather, time of day, season.

The 9 outputs map to actions: move away from danger, move to resource, move to loved one, explore, gather, craft or build, fight, eat or rest, do nothing.

The network doesn't know what these actions mean. It just learns which output to fire given the inputs. The meaning comes from the environment.

---

## The Anomalies Also Have Brains

The anomalies — Void Storms, Temporal Rifts, Void Creeps — aren't scripted. They run their own neural policies. 12 inputs, 10 hidden, 5 outputs. Their actions: chase, flank left, flank right, retreat and grow stronger, spread damage.

Their fitness is measured by total damage dealt. The agents' fitness is measured by survival time, shelters built, resources gathered, items crafted, and kills.

Both populations evolve in parallel. When agents get better at avoiding anomalies, the anomaly gene pool adapts. When anomalies get better at flanking, agents learn to build walls.

By generation 1,000, both sides were doing things I didn't explicitly program.

---

## How Evolution Works

No backprop, no gradients. Each agent's neural network weights are stored as a flat array. When an agent dies, its weights and fitness score go into a gene pool with two tiers:

- **Elite pool** — top 10 performers of all time
- **Recent pool** — last 10 generations, regardless of score (for diversity)

New agent weights are created by: 70% crossover between two elite agents, 20% crossover with a recent agent, 10% completely fresh random weights.

Then mutation is applied — small random noise on each weight. If the population stagnates for 8 generations without improvement, the mutation rate resets higher to shake things out. This fixed the local optima problem that killed my first few attempts.

---

## Generation 0 vs Generation 1,500

**Gen 0:** Agents wander randomly. Die within 50 ticks. No shelters. No communities. Anomalies kill them almost immediately.

**Gen 1,500:** Agents build 80+ shelters per generation. Population hits the cap of 20 within 200 ticks. Agents live to old age (400–600 ticks). Communities form within the first 100 ticks. Anomalies are actively fought off by agents carrying swords.

Fitness went from ~50 to ~3,800. That's not a tweak. That's the network actually learning what survival means.

---

## The Faces

One thing I added late that ended up mattering more than expected: the agents have pixel art faces.

Healthy and safe — small smile. Near an anomaly — eyes go wide, mouth opens in an O. Hurt — eyes become X's. Hungry — frown.

It's 8 pixels. But watching a crowd of tiny faces all go scared at the same time when an anomaly spawns — that hits different. It makes the simulation feel alive in a way that bar charts don't.

---

## The Collective Memory

When all agents die and the world restarts, the new generation doesn't start from zero.

The world keeps a collective memory — dangerous locations where agents died, best-performing trait values, safe spawn zones where agents survived longest. New agents spawn in safe zones 70% of the time, start with traits biased towards the best performers, and have their danger memory pre-loaded.

Each generation starts slightly smarter than the last, even before the neural network kicks in.

---

## What I'd Do Differently

**The world is too big.** 48×48 means agents spend a lot of time wandering before finding resources. I'd drop to 32×32 and increase resource density.

**The anomaly AI converged too early.** After generation 500, the anomaly pool settled on "chase the nearest agent." It works but it's not interesting. More anomaly types with different objectives would keep the arms race going longer.

**The bond system needs more depth.** Bonds are just a number right now. Agents defending their loved ones, grieving when they die, forming alliances between pairs — that's the next version.

**Log more.** I have fitness over generations but no record of *when* specific behaviors emerged. Did shelter-building come before or after community formation? I genuinely don't know.

---

## The Stack

Python 3.11, Flask, NumPy, Pydantic v2, HTML Canvas. No external ML libraries — the neuroevolution is hand-rolled. The whole thing runs on CPU. A generation of 1,500 ticks takes about 1.2 seconds on a MacBook Air.

---

## Try It

Live on Hugging Face Spaces — the simulation runs automatically when you open it.

Agents start gathering wood within seconds. By minute two, shelters appear. By minute five, communities have shared resource pools and agents are hunting anomalies.

Ctrl+Click to drop resources anywhere on the map. Shift+Click to spawn an anomaly. Watch the faces.

---

*Built for the OpenEnv Hackathon 2026.*

*— Nimesh*
