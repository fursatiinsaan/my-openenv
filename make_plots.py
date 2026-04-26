"""Read saved training data and generate PNG plots."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs("training_curves", exist_ok=True)

with open("training_curves/training_data.json") as f:
    d = json.load(f)

gens      = d["generations"]
best_fit  = d["best_fitness"]
avg_fit   = d["avg_fitness"]
ano_best  = d["anomaly_best_dmg"]
shelters  = d["shelters_per_gen"]

STYLE = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "text.color": "#c9d1d9", "grid.color": "#21262d",
    "grid.linestyle": "--", "grid.alpha": 0.6,
}
plt.rcParams.update(STYLE)

def save(fig, name):
    p = f"training_curves/{name}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")

# 1. Reward curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(gens, best_fit, color="#58a6ff", lw=2, label="Best fitness (all-time)")
ax.plot(gens, avg_fit,  color="#3fb950", lw=1.5, ls="--", label="Avg fitness (this gen)")
ax.fill_between(gens, avg_fit, best_fit, alpha=0.15, color="#58a6ff")
ax.set_title("Agent Reward Curve — AnomalyCraft Survival", fontsize=14, pad=12)
ax.set_xlabel("Generation"); ax.set_ylabel("Fitness Score")
ax.legend(framealpha=0.3); ax.grid(True)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
save(fig, "reward_curve.png")

# 2. Loss proxy
loss = [max(0, best_fit[i] - avg_fit[i]) for i in range(len(gens))]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(gens, loss, color="#f85149", lw=2)
ax.fill_between(gens, 0, loss, alpha=0.2, color="#f85149")
ax.set_title("Training Loss Proxy — AnomalyCraft Survival\n(Best − Avg fitness; lower = more consistent)", fontsize=13, pad=12)
ax.set_xlabel("Generation"); ax.set_ylabel("Fitness Gap (loss proxy)")
ax.grid(True)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
save(fig, "loss_curve.png")

# 3. Anomaly damage
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(gens, ano_best, color="#d29922", lw=2)
ax.fill_between(gens, 0, ano_best, alpha=0.2, color="#d29922")
ax.set_title("Anomaly Best Damage — AnomalyCraft Survival\n(Anomaly neural policy fitness)", fontsize=13, pad=12)
ax.set_xlabel("Generation"); ax.set_ylabel("Total Damage Dealt")
ax.grid(True)
save(fig, "anomaly_curve.png")

# 4. Shelters
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(gens, shelters, color="#388bfd", alpha=0.8, width=0.8)
ax.set_title("Shelters Built Per Generation — AnomalyCraft Survival\n(Agents learning to build)", fontsize=13, pad=12)
ax.set_xlabel("Generation"); ax.set_ylabel("Shelters Built")
ax.grid(True, axis="y")
save(fig, "shelters_curve.png")

print("All plots saved to training_curves/")
