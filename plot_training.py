"""
Run training for N generations and save reward/fitness curve plots as PNG.
Produces:
  - training_curves/reward_curve.png   — agent best fitness per generation
  - training_curves/loss_curve.png     — avg generation fitness (proxy for loss)
  - training_curves/anomaly_curve.png  — anomaly best damage per generation
  - training_curves/shelters_curve.png — shelters built per generation
"""

import os
import sys
import json

# ── Run training ──────────────────────────────────────────────────────────────
from train import run_generation, load_weights, save_weights, WEIGHTS_FILE
from neural_policy import get_brain, get_anomaly_brain

GENS   = 100
TICKS  = 2000
AGENTS = 10

os.makedirs("training_curves", exist_ok=True)

ab = get_brain()
nb = get_anomaly_brain()

# Resume from saved weights if available
if os.path.exists(WEIGHTS_FILE):
    load_weights(WEIGHTS_FILE)
    print(f"Resumed from gen {ab.generation}, best={ab.best_fitness_ever:.0f}")

start_gen = ab.generation

# Tracking arrays
gens_x          = []
best_fitness     = []
avg_fitness      = []
ano_best         = []
shelters_per_gen = []

print(f"Training {GENS} generations ({TICKS} ticks each, {AGENTS} agents)...")
print(f"{'Gen':>5} | {'BestFit':>8} | {'AvgFit':>8} | {'AnoDmg':>8} | {'Shelters':>8}")
print("─" * 50)

for i in range(GENS):
    gen_idx = start_gen + i
    stats = run_generation(gen_idx, TICKS, AGENTS)

    ab.new_generation()
    nb.new_generation()

    g   = gen_idx + 1
    bf  = ab.elite_pool[0][0] if ab.elite_pool else 0
    af  = ab.avg_fitness_history[-1] if ab.avg_fitness_history else 0
    adf = nb.elite_pool[0][0] if nb.elite_pool else 0
    sh  = stats["shelters"]

    gens_x.append(g)
    best_fitness.append(bf)
    avg_fitness.append(af)
    ano_best.append(adf)
    shelters_per_gen.append(sh)

    print(f"{g:>5} | {bf:>8.0f} | {af:>8.0f} | {adf:>8.1f} | {sh:>8}")
    sys.stdout.flush()

    if (i + 1) % 10 == 0:
        save_weights(WEIGHTS_FILE)

save_weights(WEIGHTS_FILE)

# Save raw data for reproducibility
with open("training_curves/training_data.json", "w") as f:
    json.dump({
        "generations":       gens_x,
        "best_fitness":      best_fitness,
        "avg_fitness":       avg_fitness,
        "anomaly_best_dmg":  ano_best,
        "shelters_per_gen":  shelters_per_gen,
    }, f, indent=2)

# ── Plot ──────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
}
plt.rcParams.update(STYLE)


def _save(fig, name):
    path = f"training_curves/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# 1. Reward curve — best fitness per generation
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(gens_x, best_fitness, color="#58a6ff", linewidth=2, label="Best fitness (all-time)")
ax.plot(gens_x, avg_fitness,  color="#3fb950", linewidth=1.5, linestyle="--", label="Avg fitness (this gen)")
ax.fill_between(gens_x, avg_fitness, best_fitness, alpha=0.15, color="#58a6ff")
ax.set_title("Agent Reward Curve — AnomalyCraft Survival", fontsize=14, pad=12)
ax.set_xlabel("Generation")
ax.set_ylabel("Fitness Score")
ax.legend(framealpha=0.3)
ax.grid(True)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
_save(fig, "reward_curve.png")

# 2. Loss curve — use (best_fitness_ever - avg_fitness) as a proxy for "loss"
# (gap between best known and current avg — shrinks as policy converges)
loss_proxy = [max(0, best_fitness[i] - avg_fitness[i]) for i in range(len(gens_x))]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(gens_x, loss_proxy, color="#f85149", linewidth=2)
ax.fill_between(gens_x, 0, loss_proxy, alpha=0.2, color="#f85149")
ax.set_title("Training Loss Proxy — AnomalyCraft Survival\n(Best − Avg fitness; lower = more consistent)", fontsize=13, pad=12)
ax.set_xlabel("Generation")
ax.set_ylabel("Fitness Gap (loss proxy)")
ax.grid(True)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
_save(fig, "loss_curve.png")

# 3. Anomaly damage curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(gens_x, ano_best, color="#d29922", linewidth=2)
ax.fill_between(gens_x, 0, ano_best, alpha=0.2, color="#d29922")
ax.set_title("Anomaly Best Damage — AnomalyCraft Survival\n(Anomaly neural policy fitness)", fontsize=13, pad=12)
ax.set_xlabel("Generation")
ax.set_ylabel("Total Damage Dealt")
ax.grid(True)
_save(fig, "anomaly_curve.png")

# 4. Shelters built per generation
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(gens_x, shelters_per_gen, color="#388bfd", alpha=0.8, width=0.8)
ax.set_title("Shelters Built Per Generation — AnomalyCraft Survival\n(Agents learning to build)", fontsize=13, pad=12)
ax.set_xlabel("Generation")
ax.set_ylabel("Shelters Built")
ax.grid(True, axis="y")
_save(fig, "shelters_curve.png")

print("\nAll plots saved to training_curves/")
print(f"Final best fitness: {best_fitness[-1]:.0f}")
print(f"Final anomaly best: {ano_best[-1]:.1f} dmg")
