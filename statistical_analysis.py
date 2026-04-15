# -*- coding: utf-8 -*-
"""
statistical_analysis.py
-----------------------
Peer-review statistical comparison of GOA vs PSO, GA, DE.
Runs each algorithm 30 times (seeds 1-30), records best fitness,
performs Wilcoxon signed-rank tests, and exports a LaTeX results table.

Usage:
    python statistical_analysis.py
"""

import os
import sys
import io
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Force stdout to UTF-8 so print() never hits cp1252 on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# -- Shared problem setup (mirrors goa_optimization.py) ----------------------
np.random.seed(0)
N_HOURS   = 24
PRED_LOAD = np.random.uniform(150, 350, N_HOURS)
PRICE     = np.random.uniform(0.08, 0.22, N_HOURS)

LB = PRED_LOAD * 0.85
UB = PRED_LOAD * 1.00

REF_PEAK = float(np.max(PRED_LOAD))
REF_COST = float(np.sum(PRED_LOAD * PRICE))
REF_VAR  = float(np.var(PRED_LOAD)) or 1.0
REF_PAR  = float(np.max(PRED_LOAD) / np.mean(PRED_LOAD))

N_RUNS      = 30
MAX_ITER    = 100
POP_SIZE    = 30
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def fitness(schedule: np.ndarray) -> float:
    peak_norm = np.max(schedule) / REF_PEAK
    cost_norm = np.sum(schedule * PRICE) / REF_COST
    var_norm  = np.var(schedule) / REF_VAR
    mean      = np.mean(schedule)
    par_norm  = (np.max(schedule) / mean if mean != 0 else 1.0) / REF_PAR
    return 0.35 * peak_norm + 0.25 * cost_norm + 0.15 * var_norm + 0.25 * par_norm


def clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, LB, UB)


# -- Algorithm implementations ------------------------------------------------

def run_goa(seed: int) -> float:
    np.random.seed(seed)
    pos  = LB + np.random.rand(POP_SIZE, N_HOURS) * (UB - LB)
    fits = np.array([fitness(pos[i]) for i in range(POP_SIZE)])
    best_pos = pos[np.argmin(fits)].copy()
    best_fit = fits.min()

    for it in range(MAX_ITER):
        c = 1.0 - it * (1.0 - 0.00004) / MAX_ITER
        new_pos = np.zeros_like(pos)
        for i in range(POP_SIZE):
            social = np.zeros(N_HOURS)
            for j in range(POP_SIZE):
                if i == j:
                    continue
                diff  = pos[j] - pos[i]
                dist  = np.linalg.norm(diff) + 1e-10
                s_val = 0.5 * np.exp(-dist / 1.5) - np.exp(-dist)
                social += c * ((UB - LB) / 2) * s_val * (diff / dist)
            new_pos[i] = clip(
                c * social + 0.5 * pos[i] + 0.5 * best_pos
                + np.random.normal(0, 0.01, N_HOURS)
            )
        pos  = new_pos
        fits = np.array([fitness(pos[i]) for i in range(POP_SIZE)])
        if fits.min() < best_fit:
            best_fit = fits.min()
            best_pos = pos[np.argmin(fits)].copy()
    return best_fit


def run_pso(seed: int) -> float:
    np.random.seed(seed)
    pos       = LB + np.random.rand(POP_SIZE, N_HOURS) * (UB - LB)
    vel       = np.zeros_like(pos)
    pbest     = pos.copy()
    pbest_fit = np.array([fitness(pos[i]) for i in range(POP_SIZE)])
    gbest     = pbest[np.argmin(pbest_fit)].copy()
    gbest_fit = pbest_fit.min()

    w, c1, c2 = 0.7, 1.5, 1.5
    for _ in range(MAX_ITER):
        r1 = np.random.rand(POP_SIZE, N_HOURS)
        r2 = np.random.rand(POP_SIZE, N_HOURS)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = clip(pos + vel)
        fits = np.array([fitness(pos[i]) for i in range(POP_SIZE)])
        improved = fits < pbest_fit
        pbest[improved]     = pos[improved].copy()
        pbest_fit[improved] = fits[improved]
        if fits.min() < gbest_fit:
            gbest_fit = fits.min()
            gbest     = pos[np.argmin(fits)].copy()
    return gbest_fit


def run_ga(seed: int) -> float:
    np.random.seed(seed)
    pop  = LB + np.random.rand(POP_SIZE, N_HOURS) * (UB - LB)
    fits = np.array([fitness(pop[i]) for i in range(POP_SIZE)])
    best_fit = fits.min()

    for _ in range(MAX_ITER):
        new_pop = np.zeros_like(pop)
        for i in range(POP_SIZE):
            p1, p2 = pop[np.random.choice(POP_SIZE, 2, replace=False)]
            pt    = np.random.randint(1, N_HOURS)
            child = np.concatenate([p1[:pt], p2[pt:]])
            mask  = np.random.rand(N_HOURS) < 0.1
            child[mask] += np.random.normal(0, 0.05 * (UB - LB)[mask])
            new_pop[i] = clip(child)
        pop  = new_pop
        fits = np.array([fitness(pop[i]) for i in range(POP_SIZE)])
        if fits.min() < best_fit:
            best_fit = fits.min()
    return best_fit


def run_de(seed: int) -> float:
    np.random.seed(seed)
    pop  = LB + np.random.rand(POP_SIZE, N_HOURS) * (UB - LB)
    fits = np.array([fitness(pop[i]) for i in range(POP_SIZE)])
    best_fit = fits.min()

    F, CR = 0.8, 0.9
    for _ in range(MAX_ITER):
        for i in range(POP_SIZE):
            idxs   = [x for x in range(POP_SIZE) if x != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant  = clip(a + F * (b - c))
            cross   = np.random.rand(N_HOURS) < CR
            trial   = np.where(cross, mutant, pop[i])
            t_fit   = fitness(trial)
            if t_fit < fits[i]:
                pop[i]  = trial
                fits[i] = t_fit
        if fits.min() < best_fit:
            best_fit = fits.min()
    return best_fit


# -- Run 30 trials per algorithm ----------------------------------------------

ALGORITHMS = {"GOA": run_goa, "PSO": run_pso, "GA": run_ga, "DE": run_de}

print("Running 30 trials per algorithm (seeds 1-30)...\n")
results: dict = {}
for name, fn in ALGORITHMS.items():
    runs = np.array([fn(seed) for seed in range(1, N_RUNS + 1)])
    results[name] = runs
    print(f"  {name}: mean={runs.mean():.6f}  std={runs.std():.6f}  "
          f"best={runs.min():.6f}  worst={runs.max():.6f}")

# -- Wilcoxon signed-rank tests (GOA vs each competitor) ---------------------

def sig_marker(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


rows = []
for name in ["PSO", "GA", "DE"]:
    stat, p = wilcoxon(results["GOA"], results[name])
    rows.append({
        "Algorithm": name,
        "Mean":    results[name].mean(),
        "Std":     results[name].std(),
        "Best":    results[name].min(),
        "Worst":   results[name].max(),
        "W-stat":  stat,
        "p-value": p,
        "Sig":     sig_marker(p),
    })

goa_row = {
    "Algorithm": "GOA (proposed)",
    "Mean":    results["GOA"].mean(),
    "Std":     results["GOA"].std(),
    "Best":    results["GOA"].min(),
    "Worst":   results["GOA"].max(),
    "W-stat":  "-",
    "p-value": "-",
    "Sig":     "-",
}

df = pd.DataFrame([goa_row] + rows)
print("\n-- Statistical Results Table --")
print(df.to_string(index=False))

# -- CSV export ---------------------------------------------------------------
csv_path = os.path.join(RESULTS_DIR, "statistical_comparison.csv")
df.to_csv(csv_path, index=False)
print(f"\nCSV saved -> {csv_path}")

# -- LaTeX export -------------------------------------------------------------
def fmt(v) -> str:
    return f"{v:.6f}" if isinstance(v, float) else str(v)


latex_lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Statistical Comparison of GOA vs Benchmark Algorithms over 30 Independent Runs}",
    r"\label{tab:statistical_comparison}",
    r"\begin{tabular}{lccccccl}",
    r"\hline",
    (r"\textbf{Algorithm} & \textbf{Mean} & \textbf{Std} & \textbf{Best} & "
     r"\textbf{Worst} & \textbf{W-stat} & \textbf{$p$-value} & \textbf{Sig.} \\"),
    r"\hline",
]

for _, row in df.iterrows():
    line = (
        f"{row['Algorithm']} & {fmt(row['Mean'])} & {fmt(row['Std'])} & "
        f"{fmt(row['Best'])} & {fmt(row['Worst'])} & {fmt(row['W-stat'])} & "
        f"{fmt(row['p-value'])} & {row['Sig']} \\\\"
    )
    latex_lines.append(line)

latex_lines += [
    r"\hline",
    (r"\multicolumn{8}{l}{\footnotesize * $p < 0.05$; ** $p < 0.01$; "
     r"ns = not significant (Wilcoxon signed-rank test, $n=30$)} \\"),
    r"\end{tabular}",
    r"\end{table}",
]

tex_path = os.path.join(RESULTS_DIR, "statistical_comparison.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines))
print(f"LaTeX saved -> {tex_path}")
print("\nDone. (* p<0.05  ** p<0.01  ns = not significant)")
