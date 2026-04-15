# -*- coding: utf-8 -*-
"""
sensitivity_analysis.py
-----------------------
Weight sensitivity analysis for the GOA fitness function.

Fitness (minimise):
    w_peak * peak_norm + w_cost * cost_norm + w_var * var_norm + w_par * par_norm
    where w_par = 1 - w_peak - w_cost - w_var  (weights always sum to 1)

Grid search: w_peak, w_cost, w_var each in [0.1, 0.3, 0.5, 0.7]
             only combinations where all four weights > 0 are kept.

Outputs (all in results/):
    sensitivity_results.csv          -- full grid results
    sensitivity_heatmap_peak_cost.png -- peak vs cost trade-off heatmap
    sensitivity_heatmap_peak_var.png  -- peak vs variance trade-off heatmap
    sensitivity_heatmap_cost_var.png  -- cost vs variance trade-off heatmap
    sensitivity_pareto.png            -- 3-D Pareto front scatter
    sensitivity_analysis.tex          -- LaTeX table (top-10 + Pareto-optimal)

Usage:
    python sensitivity_analysis.py
"""

import os
import sys
import io
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ---------------------------------------------------------------------------
# Shared problem (identical seed/setup to goa_optimization.py quick-test)
# ---------------------------------------------------------------------------
np.random.seed(0)
N_HOURS   = 24
PRED_LOAD = np.random.uniform(150, 350, N_HOURS)
PRICE     = np.random.uniform(0.08, 0.22, N_HOURS)

load_norm_01 = ((PRED_LOAD - PRED_LOAD.min()) /
                (PRED_LOAD.max() - PRED_LOAD.min() + 1e-10))
LB = PRED_LOAD * (0.90 - 0.15 * load_norm_01)
UB = PRED_LOAD * 1.00

REF_PEAK = float(PRED_LOAD.max())
REF_COST = float(np.sum(PRED_LOAD * PRICE))
REF_VAR  = float(np.var(PRED_LOAD)) or 1.0
REF_PAR  = float(PRED_LOAD.max() / PRED_LOAD.mean())

N_GRASSHOPPERS = 20   # reduced for grid-search speed
MAX_ITER       = 50
RANDOM_STATE   = 42

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

WEIGHT_GRID = [0.1, 0.3, 0.5, 0.7]


# ---------------------------------------------------------------------------
# GOA with parametric weights
# ---------------------------------------------------------------------------

def _s(dist: float) -> float:
    return 0.5 * np.exp(-dist / 1.5) - np.exp(-dist)


def _fitness(schedule: np.ndarray,
             w_peak: float, w_cost: float, w_var: float, w_par: float) -> float:
    peak_norm = schedule.max() / REF_PEAK
    cost_norm = np.sum(schedule * PRICE) / REF_COST
    var_norm  = np.var(schedule) / REF_VAR
    mean      = schedule.mean()
    par_norm  = (schedule.max() / mean if mean != 0 else 1.0) / REF_PAR
    return w_peak * peak_norm + w_cost * cost_norm + w_var * var_norm + w_par * par_norm


def run_goa(w_peak: float, w_cost: float, w_var: float, w_par: float) -> np.ndarray:
    """Run GOA with given weights; return best schedule found."""
    np.random.seed(RANDOM_STATE)
    pos  = LB + np.random.rand(N_GRASSHOPPERS, N_HOURS) * (UB - LB)
    fits = np.array([_fitness(pos[i], w_peak, w_cost, w_var, w_par)
                     for i in range(N_GRASSHOPPERS)])
    best_pos = pos[np.argmin(fits)].copy()
    best_fit = fits.min()

    for it in range(MAX_ITER):
        c = 1.0 - it * (1.0 - 0.00004) / MAX_ITER
        new_pos = np.zeros_like(pos)
        for i in range(N_GRASSHOPPERS):
            social = np.zeros(N_HOURS)
            for j in range(N_GRASSHOPPERS):
                if i == j:
                    continue
                diff = pos[j] - pos[i]
                dist = np.linalg.norm(diff) + 1e-10
                social += c * ((UB - LB) / 2) * _s(dist) * (diff / dist)
            new_pos[i] = np.clip(
                c * social + 0.5 * pos[i] + 0.5 * best_pos
                + np.random.normal(0, 0.01, N_HOURS),
                LB, UB,
            )
        pos  = new_pos
        fits = np.array([_fitness(pos[i], w_peak, w_cost, w_var, w_par)
                         for i in range(N_GRASSHOPPERS)])
        if fits.min() < best_fit:
            best_fit = fits.min()
            best_pos = pos[np.argmin(fits)].copy()

    return best_pos


def compute_reductions(schedule: np.ndarray) -> tuple:
    """Return (peak_red_pct, cost_red_pct, var_red_pct) vs PRED_LOAD baseline."""
    peak_red = (REF_PEAK - schedule.max()) / REF_PEAK * 100
    cost_red = (REF_COST - np.sum(schedule * PRICE)) / REF_COST * 100
    var_red  = (REF_VAR  - np.var(schedule)) / REF_VAR  * 100
    return peak_red, cost_red, var_red


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

records = []
combos  = list(itertools.product(WEIGHT_GRID, repeat=3))
valid   = [(w1, w2, w3) for w1, w2, w3 in combos if w1 + w2 + w3 <= 0.9]
total   = len(valid)

print(f"Grid search: {total} valid weight combinations\n")

for idx, (w1, w2, w3) in enumerate(valid, 1):
    w4 = round(1.0 - w1 - w2 - w3, 4)   # w_par (always > 0 by filter above)
    schedule = run_goa(w1, w2, w3, w4)
    pr, cr, vr = compute_reductions(schedule)
    records.append({
        "w_peak": w1, "w_cost": w2, "w_var": w3, "w_par": w4,
        "peak_red_%": round(pr, 4),
        "cost_red_%": round(cr, 4),
        "var_red_%":  round(vr, 4),
    })
    print(f"  [{idx:>3}/{total}] w=({w1},{w2},{w3},{w4:.2f}) | "
          f"peak={pr:+.2f}%  cost={cr:+.2f}%  var={vr:+.2f}%")

df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Pareto-optimal identification
# (maximise all three reductions simultaneously — no solution dominates)
# ---------------------------------------------------------------------------

def is_pareto(costs: np.ndarray) -> np.ndarray:
    """
    costs : (n, 3) array where HIGHER is better (reductions).
    Returns boolean mask of Pareto-optimal rows.
    """
    n    = len(costs)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # dominated if another point is >= in all dims and > in at least one
        dominated = np.all(costs >= costs[i], axis=1) & np.any(costs > costs[i], axis=1)
        dominated[i] = False
        if dominated.any():
            mask[i] = False
    return mask

obj_matrix  = df[["peak_red_%", "cost_red_%", "var_red_%"]].values
pareto_mask = is_pareto(obj_matrix)
df["pareto"] = pareto_mask

n_pareto = pareto_mask.sum()
print(f"\nPareto-optimal solutions: {n_pareto}")
print(df[pareto_mask][["w_peak","w_cost","w_var","w_par",
                        "peak_red_%","cost_red_%","var_red_%"]].to_string(index=False))

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
csv_path = os.path.join(RESULTS_DIR, "sensitivity_results.csv")
df.to_csv(csv_path, index=False)
print(f"\nCSV saved -> {csv_path}")

# ---------------------------------------------------------------------------
# Heatmaps  (pivot: rows=w_peak, cols=w_cost, value=reduction, averaged over w_var)
# ---------------------------------------------------------------------------

def save_heatmap(pivot_df: pd.DataFrame, title: str, cbar_label: str,
                 fname: str, cmap: str = "RdYlGn") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot_df.values, cmap=cmap, aspect="auto",
                   vmin=pivot_df.values.min(), vmax=pivot_df.values.max())
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot_df.columns], fontsize=9)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot_df.index], fontsize=9)
    for r in range(pivot_df.shape[0]):
        for c in range(pivot_df.shape[1]):
            val = pivot_df.values[r, c]
            ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    ax.set_xlabel("w_cost", fontsize=10)
    ax.set_ylabel("w_peak", fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Heatmap saved -> {path}")


# Average over w_var for each (w_peak, w_cost) cell
hm_peak = df.pivot_table(index="w_peak", columns="w_cost",
                          values="peak_red_%", aggfunc="mean")
hm_cost = df.pivot_table(index="w_peak", columns="w_cost",
                          values="cost_red_%", aggfunc="mean")
hm_var  = df.pivot_table(index="w_peak", columns="w_cost",
                          values="var_red_%",  aggfunc="mean")

save_heatmap(hm_peak, "Peak Reduction % (avg over w_var)\nw_peak vs w_cost",
             "Peak Reduction (%)", "sensitivity_heatmap_peak_cost.png")
save_heatmap(hm_cost, "Cost Reduction % (avg over w_var)\nw_peak vs w_cost",
             "Cost Reduction (%)", "sensitivity_heatmap_cost_var.png", cmap="Blues")
save_heatmap(hm_var,  "Variance Reduction % (avg over w_var)\nw_peak vs w_cost",
             "Variance Reduction (%)", "sensitivity_heatmap_peak_var.png", cmap="Purples")

# ---------------------------------------------------------------------------
# Pareto scatter (3-D)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection="3d")

non_pareto = df[~pareto_mask]
pareto_pts = df[pareto_mask]

ax.scatter(non_pareto["peak_red_%"], non_pareto["cost_red_%"], non_pareto["var_red_%"],
           c="steelblue", alpha=0.35, s=20, label="Dominated")
ax.scatter(pareto_pts["peak_red_%"], pareto_pts["cost_red_%"], pareto_pts["var_red_%"],
           c="crimson", s=80, marker="*", label=f"Pareto-optimal (n={n_pareto})", zorder=5)

# Annotate Pareto points with weight tuple
for _, row in pareto_pts.iterrows():
    ax.text(row["peak_red_%"], row["cost_red_%"], row["var_red_%"],
            f"({row['w_peak']},{row['w_cost']},{row['w_var']})",
            fontsize=6, color="darkred")

ax.set_xlabel("Peak Red. (%)", fontsize=9)
ax.set_ylabel("Cost Red. (%)", fontsize=9)
ax.set_zlabel("Var Red. (%)",  fontsize=9)
ax.set_title("GOA Weight Sensitivity: Pareto Front\n(w_peak, w_cost, w_var)", fontsize=11)
ax.legend(fontsize=8)
plt.tight_layout()
pareto_path = os.path.join(RESULTS_DIR, "sensitivity_pareto.png")
plt.savefig(pareto_path, dpi=150)
plt.close()
print(f"Pareto plot saved -> {pareto_path}")

# ---------------------------------------------------------------------------
# LaTeX table  (Pareto-optimal rows + top-5 by each objective)
# ---------------------------------------------------------------------------
top_peak = df.nlargest(5, "peak_red_%")
top_cost = df.nlargest(5, "cost_red_%")
top_var  = df.nlargest(5, "var_red_%")
table_df = (pd.concat([df[pareto_mask], top_peak, top_cost, top_var])
              .drop_duplicates()
              .sort_values("peak_red_%", ascending=False)
              .reset_index(drop=True))

def tex_row(row: pd.Series) -> str:
    tag = r"\textbf{P}" if row["pareto"] else ""
    return (
        f"{row['w_peak']:.1f} & {row['w_cost']:.1f} & "
        f"{row['w_var']:.1f} & {row['w_par']:.2f} & "
        f"{row['peak_red_%']:.2f} & {row['cost_red_%']:.2f} & "
        f"{row['var_red_%']:.2f} & {tag} \\\\"
    )


latex = "\n".join([
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{GOA Weight Sensitivity Analysis: Peak, Cost, and Variance Reduction "
    r"across Weight Combinations (Pareto-optimal marked \textbf{P})}",
    r"\label{tab:sensitivity}",
    r"\begin{tabular}{ccccrrrl}",
    r"\hline",
    r"$w_\text{peak}$ & $w_\text{cost}$ & $w_\text{var}$ & $w_\text{par}$ & "
    r"\textbf{Peak Red.\%} & \textbf{Cost Red.\%} & \textbf{Var Red.\%} & \textbf{Pareto} \\",
    r"\hline",
    *[tex_row(row) for _, row in table_df.iterrows()],
    r"\hline",
    r"\multicolumn{8}{l}{\footnotesize "
    r"w\_par = 1 - w\_peak - w\_cost - w\_var; "
    r"grid values $\in \{0.1, 0.3, 0.5, 0.7\}$; "
    r"GOA: 20 agents, 50 iterations, seed=42.} \\",
    r"\end{tabular}",
    r"\end{table}",
])

tex_path = os.path.join(RESULTS_DIR, "sensitivity_analysis.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex)
print(f"LaTeX saved -> {tex_path}")
print("\nDone.")
