import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from matplotlib.lines import Line2D

# -----------------------------
# 1) DATA
# -----------------------------
no_fire_initial = [
    (23.90, 40.50, 0, "no fire", "initial"),
    (23.90, 42.90, 0, "no fire", ""),
    (23.80, 47.00, 0, "no fire", ""),
    (23.80, 54.60, 0, "no fire", ""),
    (23.90, 52.50, 0, "no fire", ""),
    (23.80, 48.00, 0, "no fire", ""),
    (23.70, 46.70, 0, "no fire", ""),
    (23.70, 46.40, 0, "no fire", ""),
    (23.70, 46.00, 0, "no fire", ""),
    (23.70, 45.60, 0, "no fire", ""),
    (25.50, 52.90, 0, "no fire", ""),
    (25.70, 49.40, 0, "no fire", ""),
    (24.90, 45.60, 0, "no fire", ""),
]

fire_block = [
    (30.10, 46.20, 1, "fire", ""),
    (33.70, 47.80, 1, "fire", ""),
    (36.80, 48.90, 1, "fire", ""),
    (36.30, 48.10, 1, "fire", ""),
    (35.50, 46.90, 1, "fire", ""),
    (34.60, 46.00, 1, "fire", ""),
    (56.00, 57.70, 1, "fire", ""),
    (50.20, 53.40, 1, "fire", ""),
    (45.70, 50.20, 1, "fire", ""),
    (43.20, 47.90, 1, "fire", ""),
    (41.90, 46.70, 1, "fire", ""),
]

no_fire_transition = [
    (47.20, 50.00, 0, "no fire", "fire stopped / transition"),
]

no_fire_cooling = [
    (81.90, 75.10, 0, "no fire", "cooldown (sensor cooling)"),
    (70.60, 64.90, 0, "no fire", ""),
    (62.30, 58.90, 0, "no fire", ""),
    (56.00, 54.60, 0, "no fire", ""),
    (50.90, 50.90, 0, "no fire", ""),
    (46.60, 47.50, 0, "no fire", ""),
    (43.40, 44.80, 0, "no fire", ""),
]

rows = no_fire_initial + fire_block + no_fire_transition + no_fire_cooling

df = pd.DataFrame(rows, columns=["Temperature_C", "Humidity_pct", "IR", "Conclusion", "Note"])
df.insert(0, "Record", np.arange(1, len(df) + 1))
df["Fire"] = (df["Conclusion"] == "fire").astype(int)

# Phase label (ca să evidențiem “cooldown” etc.)
df["Phase"] = "no_fire_initial"
start_fire = len(no_fire_initial) + 1
end_fire = len(no_fire_initial) + len(fire_block)
df.loc[df["Record"].between(start_fire, end_fire), "Phase"] = "fire"
df.loc[df["Record"] == end_fire + 1, "Phase"] = "no_fire_transition"
df.loc[df["Record"] >= end_fire + 2, "Phase"] = "no_fire_cooling"

out_dir = Path(".")  # schimbă dacă vrei alt folder

# -----------------------------
# 2) FIGURA 1: Time series
#    Temp + Humidity + IR (3 axe) + marcaj fire
# -----------------------------
x = df["Record"].to_numpy()

fig = plt.figure(figsize=(8.5, 5.2))
ax1 = fig.add_subplot(111)

# Temperature (axa stângă)
ax1.plot(x, df["Temperature_C"].to_numpy(), marker="o", linewidth=1)
ax1.set_xlabel("Record (ordine măsurări)")
ax1.set_ylabel("Temperatura (°C)")
ax1.grid(True, alpha=0.3)

# Humidity (axa dreaptă)
ax2 = ax1.twinx()
ax2.plot(x, df["Humidity_pct"].to_numpy(), marker="s", linewidth=1, linestyle="--")
ax2.set_ylabel("Umiditate (%)")

# IR (a treia axă, deplasată în exterior)
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 55))
ax3.step(x, df["IR"].to_numpy(), where="post", linewidth=1)
ax3.set_ylim(-0.1, 1.1)
ax3.set_ylabel("IR (0/1)")

# Marcaj fire pe curba temperaturii (puncte triunghi)
fire_x = df[df["Fire"] == 1]["Record"].to_numpy()
fire_temp = df[df["Fire"] == 1]["Temperature_C"].to_numpy()
ax1.scatter(fire_x, fire_temp, marker="^")

# Legendă cu proxy artists (fără să alegem culori manual)
proxies = [
    Line2D([0], [0], marker="o", linestyle="-", label="Temperatura (°C)"),
    Line2D([0], [0], marker="s", linestyle="--", label="Umiditate (%)"),
    Line2D([0], [0], linestyle="-", label="IR (0/1)"),
    Line2D([0], [0], marker="^", linestyle="None", label="Marcaj: fire (pe temperatură)"),
]
ax1.legend(handles=proxies, frameon=True, loc="upper left")

ax1.set_title("Serii temporale: Temperatură, Umiditate și IR (cu marcaj fire)")

fig1_path = out_dir / "fire_timeseries_temp_hum_ir.png"
fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# 3) FIGURA 2: Scatter plot
#    Temperature vs Humidity (fire/no fire)
# -----------------------------
fig = plt.figure(figsize=(7.0, 5.2))
ax = fig.add_subplot(111)

sub0 = df[df["Fire"] == 0]
sub1 = df[df["Fire"] == 1]

ax.scatter(sub0["Temperature_C"], sub0["Humidity_pct"], marker="o", label="no fire")
ax.scatter(sub1["Temperature_C"], sub1["Humidity_pct"], marker="s", label="fire")

# Evidențiem tranziția (punctul imediat după stingere)
trans = df[df["Phase"] == "no_fire_transition"]
if len(trans):
    ax.scatter(trans["Temperature_C"], trans["Humidity_pct"], marker="X", label="no fire (transition)")

# Adăugăm o etichetă pentru clusterul de cooldown (fără să încărcăm graficul)
cool = df[df["Phase"] == "no_fire_cooling"]
if len(cool):
    cx, cy = cool["Temperature_C"].mean(), cool["Humidity_pct"].mean()
    ax.annotate("cooldown cluster", (cx, cy), textcoords="offset points", xytext=(6, 6))

ax.set_xlabel("Temperatura (°C)")
ax.set_ylabel("Umiditate (%)")
ax.set_title("Scatter: Temperatură vs Umiditate (clasat fire/no fire)")
ax.grid(True, alpha=0.3)
ax.legend(frameon=True)

fig2_path = out_dir / "fire_scatter_temp_vs_humidity.png"
fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close(fig)


print("Saved:", fig1_path, fig2_path, pdf_path)