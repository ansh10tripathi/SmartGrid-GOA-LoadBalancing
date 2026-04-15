# -*- coding: utf-8 -*-
"""
pareto_analysis.py
------------------
Multi-objective Pareto front analysis for GOA load scheduling.

Method  : Weighted-sum scalarisation with 200 Dirichlet-sampled weight
          vectors (w_peak, w_cost, w_var), each summing to 1.  Running GOA
          once per weight vector traces the Pareto-optimal trade-off surface
          between Peak Reduction, Cost Reduction, and Variance Reduction.

Outputs
-------
  results/pareto_front.png   -- 4-panel figure (3 x 2-D projections + 3-D)
  results/pareto_front.csv   -- all 200 runs with Pareto flag
  results/pareto_front.tex   -- LaTeX table of Pareto-optimal solutions

Usage
-----
  python pareto_analysis.py
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ---------------------------------------------------------------------------
# Problem setup  (identical seed to sensitivity_analysis.py and GOA quick-test)
# ---------------------------------------------------------------------------
np.random.seed(0)
N_HOURS   = 24
PRED_LOAD = np.random.uniform(150, 350, N_HOURS)
PRICE     = np.random.uniform(0.08, 0.22, N_HOURS)

_load_norm = (PRED_LOAD - PRED_LOAD.min()) / (PRED_LOAD.max() - PRED_LOAD.min() + 1e-10)
LB = PRED_LOAD * (0.90 - 0.15 * _load_norm)
UB = PRED_LOAD * 1.00

REF_PEAK = float(PRED_LOAD.max())
REF_COST = float(np.sum(PRED_LOAD * PRICE))
REF_VAR  = float(np.var(PRED_LOAD)) or 1.0

# Physical constraint thresholds (mirrors goa_optimization.py defaults)
MAX_RAMP  = 0.15 * float(PRED_LOAD.mean())
GRID_MAX  = 1.10 * REF_PEAK
LOAD_MIN  = 0.60 * float(PRED_LOAD.min())
PEN_W     = 10.0          # penalty weight for constraint violations

# Run parameters — kept small so 200 runs finish in ~60 s
N_AGENTS  = 15
MAX_ITER  = 50
N_RUNS    = 200

# Paper's current weight vector (w_peak, w_cost, w_var)
CURRENT_W = (0.4, 0.3, 0.3)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# GOA internals
# ---------------------------------------------------------------------------

def _s(dist: float) -> float:
    return 0.5 * np.exp(-dist / 1.5) - np.exp(-dist)


def _penalty(schedule: np.ndarray) -> float:
    ramp_exc  = np.maximum(0.0, np.abs(np.diff(schedule)) - MAX_RAMP)
    cap_exc   = np.maximum(0.0, schedule - GRID_MAX)
    floor_def = np.maximum(0.0, LOAD_MIN - schedule)
    n = len(schedule)
    return (ramp_exc.sum()  / max((n - 1) * MAX_RAMP, 1e-10)
            + cap_exc.sum() / max(n * GRID_MAX,        1e-10)
            + floor_def.sum()/ max(n * LOAD_MIN,       1e-10))


def _fitness(schedule: np.ndarray, w_peak: float, w_cost: float, w_var: float) -> float:
    obj = (w_peak * schedule.max() / REF_PEAK
           + w_cost * np.sum(schedule * PRICE) / REF_COST
           + w_var  * np.var(schedule) / REF_VAR)
    return obj + PEN_W * _penalty(schedule) ** 2


def _run_goa(w_peak: float, w_cost: float, w_var: float, seed: int) -> np.ndarray:
    """Return best schedule for one (w_peak, w_cost, w_var) weight vector."""
    np.random.seed(seed)
    pos  = LB + np.random.rand(N_AGENTS, N_HOURS) * (UB - LB)
    fits = np.array([_fitness(pos[i], w_peak, w_cost, w_var) for i in range(N_AGENTS)])
    best_pos = pos[np.argmin(fits)].copy()
    best_fit = fits.min()

    for it in range(MAX_ITER):
        c = 1.0 - it * (1.0 - 0.00004) / MAX_ITER
        new_pos = np.zeros_like(pos)
        for i in range(N_AGENTS):
            social = np.zeros(N_HOURS)
            for j in range(N_AGENTS):
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
        fits = np.array([_fitness(pos[i], w_peak, w_cost, w_var) for i in range(N_AGENTS)])
        if fits.min() < best_fit:
            best_fit = fits.min()
            best_pos = pos[np.argmin(fits)].copy()

    return best_pos


def _reductions(schedule: np.ndarray) -> tuple:
    """Return (peak_red%, cost_red%, var_red%) relative to PRED_LOAD baseline."""
    return (
        (REF_PEAK - schedule.max())           / REF_PEAK * 100,
        (REF_COST - np.sum(schedule * PRICE)) / REF_COST * 100,
        (REF_VAR  - np.var(schedule))         / REF_VAR  * 100,
    )


# ---------------------------------------------------------------------------
# Pareto dominance filter  (maximise all objectives)
# ---------------------------------------------------------------------------

def _pareto_mask(obj: np.ndarray) -> np.ndarray:
    """
    obj  : (n, k) array — higher is better in every column.
    Returns boolean mask; True = non-dominated (Pareto-optimal).
    """
    n, k = obj.shape
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # i is dominated if any other point is >= in all objectives and > in one
        dom = (np.all(obj >= obj[i], axis=1)
               & np.any(obj >  obj[i], axis=1))
        dom[i] = False
        if dom.any():
            mask[i] = False
    return mask


# ---------------------------------------------------------------------------
# 200-run sweep with Dirichlet-sampled weights
# ---------------------------------------------------------------------------

# Dirichlet(1,1,1) gives uniform coverage of the 3-simplex (w1+w2+w3=1, all>0)
rng     = np.random.default_rng(seed=7)
weights = rng.dirichlet(alpha=[1, 1, 1], size=N_RUNS)   # (200, 3)

print(f"Running {N_RUNS} GOA instances with Dirichlet-sampled weights ...\n")

records = []
for idx, (w1, w2, w3) in enumerate(weights):
    sched = _run_goa(w1, w2, w3, seed=idx)
    pr, cr, vr = _reductions(sched)
    records.append({
        "run":    idx,
        "w_peak": round(w1, 6),
        "w_cost": round(w2, 6),
        "w_var":  round(w3, 6),
        "peak_red_%": round(pr, 4),
        "cost_red_%": round(cr, 4),
        "var_red_%":  round(vr, 4),
    })
    if (idx + 1) % 40 == 0:
        print(f"  {idx+1:>3}/{N_RUNS} done")

df = pd.DataFrame(records)

# Pareto filter
obj_mat = df[["peak_red_%", "cost_red_%", "var_red_%"]].values
df["pareto"] = _pareto_mask(obj_mat)

n_pareto = int(df["pareto"].sum())
print(f"\nPareto-optimal solutions: {n_pareto} / {N_RUNS}")

# Current-solution run (w_peak=0.4, w_cost=0.3, w_var=0.3, seed=999)
cur_sched = _run_goa(*CURRENT_W, seed=999)
cur_pr, cur_cr, cur_vr = _reductions(cur_sched)
print(f"Current solution (w={CURRENT_W}): "
      f"peak={cur_pr:.2f}%  cost={cur_cr:.2f}%  var={cur_vr:.2f}%")

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
csv_path = os.path.join(RESULTS_DIR, "pareto_front.csv")
df.to_csv(csv_path, index=False)
print(f"CSV saved -> {csv_path}")

# ---------------------------------------------------------------------------
# Figure  — 4 panels
# ---------------------------------------------------------------------------
dom  = df[~df["pareto"]]
par  = df[ df["pareto"]].sort_values("peak_red_%")

# Colour each Pareto point by its w_peak value for extra information density
par_colors = par["w_peak"].values

fig = plt.figure(figsize=(15, 12))
gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
ax_pc  = fig.add_subplot(gs[0, 0])   # Peak vs Cost
ax_cv  = fig.add_subplot(gs[0, 1])   # Cost vs Variance
ax_pv  = fig.add_subplot(gs[1, 0])   # Peak vs Variance
ax_3d  = fig.add_subplot(gs[1, 1], projection="3d")

_DOM_KW    = dict(c="lightsteelblue", alpha=0.35, s=18, zorder=1)
_PAR_KW    = dict(s=70, zorder=4, edgecolors="black", linewidths=0.5)
_CUR_KW    = dict(marker="D", s=160, color="gold", edgecolors="black",
                  linewidths=1.2, zorder=6, label=f"Current w={CURRENT_W}")
_FRONT_KW  = dict(color="crimson", lw=1.2, ls="--", zorder=3, alpha=0.7)


def _draw_front(ax, x_par, y_par, xlabel, ylabel, title):
    """Scatter + step-front line for one 2-D projection."""
    ax.scatter(dom["peak_red_%"] if xlabel.startswith("Peak") else
               (dom["cost_red_%"] if xlabel.startswith("Cost") else dom["peak_red_%"]),
               dom["cost_red_%"] if ylabel.startswith("Cost") else
               (dom["var_red_%"]  if ylabel.startswith("Var")  else dom["cost_red_%"]),
               **_DOM_KW, label="Dominated")

    sc = ax.scatter(x_par, y_par, c=par_colors, cmap="plasma",
                    vmin=0, vmax=1, **_PAR_KW, label="Pareto-optimal")

    # Step-front line (sort by x, draw staircase)
    order   = np.argsort(x_par)
    x_front = np.array(x_par)[order]
    y_front = np.array(y_par)[order]
    ax.step(x_front, y_front, where="post", **_FRONT_KW)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, pad=6)
    ax.grid(alpha=0.25)
    return sc


# Panel 1 — Peak vs Cost
sc1 = _draw_front(ax_pc,
                  par["peak_red_%"].values, par["cost_red_%"].values,
                  "Peak Reduction (%)", "Cost Reduction (%)",
                  "Pareto Front: Peak vs Cost")
ax_pc.scatter(cur_pr, cur_cr, **_CUR_KW)
ax_pc.annotate(f"  Current\n  ({cur_pr:.1f}%, {cur_cr:.1f}%)",
               xy=(cur_pr, cur_cr), fontsize=7.5, color="saddlebrown")

# Panel 2 — Cost vs Variance
sc2 = _draw_front(ax_cv,
                  par["cost_red_%"].values, par["var_red_%"].values,
                  "Cost Reduction (%)", "Variance Reduction (%)",
                  "Pareto Front: Cost vs Variance")
ax_cv.scatter(cur_cr, cur_vr, **_CUR_KW)
ax_cv.annotate(f"  Current\n  ({cur_cr:.1f}%, {cur_vr:.1f}%)",
               xy=(cur_cr, cur_vr), fontsize=7.5, color="saddlebrown")

# Panel 3 — Peak vs Variance
sc3 = _draw_front(ax_pv,
                  par["peak_red_%"].values, par["var_red_%"].values,
                  "Peak Reduction (%)", "Variance Reduction (%)",
                  "Pareto Front: Peak vs Variance")
ax_pv.scatter(cur_pr, cur_vr, **_CUR_KW)
ax_pv.annotate(f"  Current\n  ({cur_pr:.1f}%, {cur_vr:.1f}%)",
               xy=(cur_pr, cur_vr), fontsize=7.5, color="saddlebrown")

# Shared colourbar for panels 1-3
cbar = fig.colorbar(sc1, ax=[ax_pc, ax_cv, ax_pv],
                    orientation="vertical", fraction=0.015, pad=0.02)
cbar.set_label("w_peak (Pareto solutions)", fontsize=9)

# Panel 4 — 3-D overview
ax_3d.scatter(dom["peak_red_%"], dom["cost_red_%"], dom["var_red_%"],
              c="lightsteelblue", alpha=0.25, s=12, label="Dominated")
ax_3d.scatter(par["peak_red_%"], par["cost_red_%"], par["var_red_%"],
              c=par_colors, cmap="plasma", s=60, edgecolors="black",
              linewidths=0.4, label=f"Pareto-optimal (n={n_pareto})", zorder=4)
ax_3d.scatter(cur_pr, cur_cr, cur_vr,
              marker="D", s=120, color="gold", edgecolors="black",
              linewidths=1.0, zorder=6, label=f"Current w={CURRENT_W}")
ax_3d.set_xlabel("Peak Red. (%)", fontsize=8, labelpad=4)
ax_3d.set_ylabel("Cost Red. (%)", fontsize=8, labelpad=4)
ax_3d.set_zlabel("Var Red. (%)",  fontsize=8, labelpad=4)
ax_3d.set_title("3-D Pareto Front Overview", fontsize=10)
ax_3d.legend(fontsize=7, loc="upper left")

# Shared legend for 2-D panels
handles = [
    plt.scatter([], [], **{**_DOM_KW, "c": "lightsteelblue"}, label="Dominated"),
    plt.scatter([], [], c="crimson", s=70, edgecolors="black",
                linewidths=0.5, label="Pareto-optimal"),
    plt.scatter([], [], **_CUR_KW),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=True)

fig.suptitle(
    f"GOA Multi-Objective Pareto Front  "
    f"({N_RUNS} Dirichlet-sampled weight vectors, "
    f"{n_pareto} Pareto-optimal solutions)\n"
    f"Objectives: Peak Reduction, Cost Reduction, Variance Reduction  "
    f"[higher = better]",
    fontsize=12, y=1.01,
)

png_path = os.path.join(RESULTS_DIR, "pareto_front.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved -> {png_path}")

# ---------------------------------------------------------------------------
# LaTeX table  (Pareto-optimal solutions, sorted by peak reduction desc)
# ---------------------------------------------------------------------------
par_df = (df[df["pareto"]]
          .sort_values("peak_red_%", ascending=False)
          .reset_index(drop=True))

def _tex_row(row: pd.Series) -> str:
    return (f"{row['w_peak']:.3f} & {row['w_cost']:.3f} & {row['w_var']:.3f} & "
            f"{row['peak_red_%']:.2f} & {row['cost_red_%']:.2f} & "
            f"{row['var_red_%']:.2f} \\\\")

latex = "\n".join([
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Pareto-Optimal GOA Solutions from 200 Dirichlet-Sampled Weight Vectors "
    r"(sorted by Peak Reduction, descending)}",
    r"\label{tab:pareto_front}",
    r"\begin{tabular}{cccrrr}",
    r"\hline",
    r"$w_\text{peak}$ & $w_\text{cost}$ & $w_\text{var}$ & "
    r"\textbf{Peak Red.\%} & \textbf{Cost Red.\%} & \textbf{Var Red.\%} \\",
    r"\hline",
    *[_tex_row(row) for _, row in par_df.iterrows()],
    r"\hline",
    r"\multicolumn{6}{l}{\footnotesize "
    r"Weights sampled from Dirichlet(1,1,1); $w_\text{peak}+w_\text{cost}+w_\text{var}=1$. "
    r"Current solution: $w=(0.4,\,0.3,\,0.3)$. "
    rf"GOA: {N_AGENTS} agents, {MAX_ITER} iterations.}} \\",
    r"\end{tabular}",
    r"\end{table}",
])

tex_path = os.path.join(RESULTS_DIR, "pareto_front.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex)
print(f"LaTeX saved -> {tex_path}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n-- Pareto-optimal solutions --")
print(par_df[["w_peak","w_cost","w_var",
              "peak_red_%","cost_red_%","var_red_%"]].to_string(index=False))
print(f"\nCurrent solution (w={CURRENT_W}):")
print(f"  peak={cur_pr:.2f}%  cost={cur_cr:.2f}%  var={cur_vr:.2f}%")

# Is the current solution on the Pareto front?
cur_obj    = np.array([[cur_pr, cur_cr, cur_vr]])
all_obj    = np.vstack([obj_mat, cur_obj])
cur_pareto = _pareto_mask(all_obj)[-1]
print(f"  On Pareto front: {cur_pareto}")
print("\nDone.")
