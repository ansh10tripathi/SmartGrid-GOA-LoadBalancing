"""
src/benchmark_algorithms.py
----------------------------
Benchmarks four metaheuristic algorithms on the same load-scheduling
fitness function to validate the GOA paper claim.

Algorithms
----------
  GOA  — native implementation from goa_optimization.py (no mealpy)
  PSO  — mealpy PSO.OriginalPSO
  GA   — mealpy GA.OriginalGA
  DE   — mealpy DE.OriginalDE

Fitness (minimise, raw — NOT normalised)
-----------------------------------------
  F(schedule) = w1 * peak  +  w2 * cost  +  w3 * variance
  w1 = 0.4,  w2 = 0.3,  w3 = 0.3

  peak     = max(schedule)
  cost     = sum(schedule * price)
  variance = var(schedule)

  All three terms are normalised against the reference (predicted) load
  before weighting so they are dimensionally comparable regardless of
  absolute load magnitude.

Shared configuration
--------------------
  Population size : 50
  Iterations      : 100
  Random seed     : 42
  Bounds          : same non-uniform [lb, ub] per dimension as GOA

Outputs
-------
  results/algorithm_comparison.csv   — final metrics for all algorithms
  results/algorithm_comparison.png   — grouped bar chart
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_SRC_DIR)
RESULTS_DIR = os.path.join(_ROOT, "results")

# ── Benchmark hyper-parameters ────────────────────────────────────────────────
POP_SIZE    = 50
MAX_ITER    = 100
SEED        = 42

# Fitness weights (paper claim configuration)
W_PEAK = 0.4
W_COST = 0.3
W_VAR  = 0.3

# Algorithm display config
_ALGO_COLORS = {
    "GOA": "#2196F3",   # blue
    "PSO": "#4CAF50",   # green
    "GA":  "#FF9800",   # orange
    "DE":  "#F44336",   # red
}

_METRIC_LABELS = {
    "Final Fitness":   "Final Fitness\n(lower = better)",
    "Peak Load (MW)":  "Peak Load (MW)\n(lower = better)",
    "Total Cost ($)":  "Total Cost ($)\n(lower = better)",
    "Variance (MW²)":  "Variance (MW²)\n(lower = better)",
    "Runtime (s)":     "Runtime (s)\n(lower = better)",
}


# ═════════════════════════════════════════════════════════════════════════════
# Shared fitness function
# ═════════════════════════════════════════════════════════════════════════════

class _FitnessEvaluator:
    """
    Encapsulates the shared fitness function and reference values so every
    algorithm evaluates exactly the same objective.

    Normalisation: each raw term is divided by its reference value
    (computed from the predicted load) so the three terms are on the
    same scale before weighting.

    F = w1 * (peak / ref_peak)  +  w2 * (cost / ref_cost)  +  w3 * (var / ref_var)
    """

    def __init__(self, predicted_load: np.ndarray, price: np.ndarray):
        self.predicted_load = predicted_load
        self.price          = price
        self.ref_peak       = float(np.max(predicted_load))
        self.ref_cost       = float(np.sum(predicted_load * price))
        self.ref_var        = float(np.var(predicted_load)) or 1.0

        # Non-uniform bounds — same as GOA
        load_norm  = ((predicted_load - predicted_load.min()) /
                      (predicted_load.max() - predicted_load.min() + 1e-10))
        self.lb    = predicted_load * (0.90 - 0.15 * load_norm)
        self.ub    = predicted_load * 1.00
        self.dim   = len(predicted_load)

    def __call__(self, schedule: np.ndarray) -> float:
        """Evaluate fitness for a candidate schedule (1-D array)."""
        schedule = np.clip(schedule, self.lb, self.ub)
        peak_n   = np.max(schedule)       / self.ref_peak
        cost_n   = np.sum(schedule * self.price) / self.ref_cost
        var_n    = np.var(schedule)       / self.ref_var
        return W_PEAK * peak_n + W_COST * cost_n + W_VAR * var_n

    def kpi(self, schedule: np.ndarray) -> dict:
        """Return raw (un-normalised) KPIs for reporting."""
        schedule = np.clip(schedule, self.lb, self.ub)
        return {
            "Peak Load (MW)": round(float(np.max(schedule)),              4),
            "Total Cost ($)": round(float(np.sum(schedule * self.price)), 4),
            "Variance (MW²)": round(float(np.var(schedule)),              4),
        }


# ═════════════════════════════════════════════════════════════════════════════
# GOA runner  (native — no mealpy)
# ═════════════════════════════════════════════════════════════════════════════

def _run_goa(evaluator: _FitnessEvaluator) -> tuple:
    """
    Run the native GOA from goa_optimization.py but with the benchmark
    fitness function (w1/w2/w3 weights) instead of the 4-term normalised
    fitness used in the main pipeline.

    Returns (best_schedule, best_fitness, fitness_history, runtime_s).
    """
    import sys
    sys.path.insert(0, _ROOT)

    np.random.seed(SEED)
    lb, ub = evaluator.lb, evaluator.ub
    dim    = evaluator.dim

    # ── Initialise ────────────────────────────────────────────────────────────
    positions    = lb + np.random.rand(POP_SIZE, dim) * (ub - lb)
    fitness_vals = np.array([evaluator(positions[i]) for i in range(POP_SIZE)])
    best_idx     = int(np.argmin(fitness_vals))
    best_pos     = positions[best_idx].copy()
    best_fitness = float(fitness_vals[best_idx])
    history      = [best_fitness]

    c_min, c_max = 0.00004, 1.0

    def _s(r, f=0.5, l=1.5):
        return f * np.exp(-r / l) - np.exp(-r)

    t0 = time.perf_counter()

    for iteration in range(MAX_ITER):
        c             = c_max - iteration * (c_max - c_min) / MAX_ITER
        new_positions = np.zeros_like(positions)

        for i in range(POP_SIZE):
            social = np.zeros(dim)
            for j in range(POP_SIZE):
                if i == j:
                    continue
                diff      = positions[j] - positions[i]
                dist      = np.linalg.norm(diff) + 1e-10
                direction = diff / dist
                s_val     = _s(np.array([dist]))[0]
                social   += c * ((ub - lb) / 2) * s_val * direction

            new_positions[i] = (c * social +
                                 0.5 * positions[i] +
                                 0.5 * best_pos +
                                 np.random.normal(0, 0.01, dim))

        positions    = np.clip(new_positions, lb, ub)
        fitness_vals = np.array([evaluator(positions[i]) for i in range(POP_SIZE)])
        cur_best     = int(np.argmin(fitness_vals))
        if fitness_vals[cur_best] < best_fitness:
            best_fitness = float(fitness_vals[cur_best])
            best_pos     = positions[cur_best].copy()
        history.append(best_fitness)

    runtime = time.perf_counter() - t0
    return best_pos, best_fitness, history, runtime


# ═════════════════════════════════════════════════════════════════════════════
# mealpy runners  (PSO / GA / DE)
# ═════════════════════════════════════════════════════════════════════════════

def _run_mealpy(algo_name: str, evaluator: _FitnessEvaluator) -> tuple:
    """
    Run a mealpy algorithm using the shared fitness function.

    Returns (best_schedule, best_fitness, fitness_history, runtime_s).

    mealpy 3.x problem dict:
        bounds   : FloatVar(lb=..., ub=..., name=...)
        obj_func : callable(solution) -> float
        minmax   : "min"
    """
    from mealpy import PSO, GA, DE, FloatVar

    # mealpy expects lb/ub as lists/tuples
    lb_list = evaluator.lb.tolist()
    ub_list = evaluator.ub.tolist()

    problem_dict = {
        "bounds":   FloatVar(lb=lb_list, ub=ub_list, name="schedule"),
        "obj_func": evaluator,
        "minmax":   "min",
    }

    if algo_name == "PSO":
        model = PSO.OriginalPSO(epoch=MAX_ITER, pop_size=POP_SIZE,
                                c1=2.05, c2=2.05, w=0.4)
    elif algo_name == "GA":
        model = GA.OriginalGA(epoch=MAX_ITER, pop_size=POP_SIZE,
                              pc=0.9, pm=0.05)
    elif algo_name == "DE":
        model = DE.OriginalDE(epoch=MAX_ITER, pop_size=POP_SIZE,
                              wf=0.8, cr=0.9, strategy=0)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    t0      = time.perf_counter()
    g_best  = model.solve(problem_dict, seed=SEED)
    runtime = time.perf_counter() - t0

    best_schedule = np.array(g_best.solution)
    best_fitness  = float(g_best.target.fitness)

    # Extract per-epoch best fitness from history
    history = [float(v) for v in model.history.list_global_best_fit]

    return best_schedule, best_fitness, history, runtime


# ═════════════════════════════════════════════════════════════════════════════
# Convergence plot
# ═════════════════════════════════════════════════════════════════════════════

def _plot_convergence(histories: dict, save_path: str) -> None:
    """Line plot of best fitness per iteration for all algorithms."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for name, hist in histories.items():
        ax.plot(hist, label=name, color=_ALGO_COLORS[name],
                linewidth=1.8, alpha=0.9)

    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Best Fitness", fontsize=10)
    ax.set_title("Convergence Curves — GOA vs PSO vs GA vs DE\n"
                 f"(pop={POP_SIZE}, iter={MAX_ITER}, "
                 f"w1={W_PEAK}, w2={W_COST}, w3={W_VAR})",
                 fontsize=10)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [benchmark] Convergence plot saved -> {save_path}")
    if os.name == "nt":
        os.startfile(save_path)


# ═════════════════════════════════════════════════════════════════════════════
# Grouped bar chart
# ═════════════════════════════════════════════════════════════════════════════

def _plot_bar_chart(df: pd.DataFrame, save_path: str) -> None:
    """
    Grouped bar chart: one subplot per metric, one bar per algorithm.
    Best bar per panel gets a gold border + ★ annotation.
    """
    metrics = ["Final Fitness", "Peak Load (MW)", "Total Cost ($)",
               "Variance (MW²)", "Runtime (s)"]
    algos   = df["Algorithm"].tolist()
    colors  = [_ALGO_COLORS[a] for a in algos]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 5.2))

    for ax, metric in zip(axes, metrics):
        values   = df[metric].values.astype(float)
        val_span = (values.max() - values.min()) if values.max() != values.min() else 1.0
        best_idx = int(np.argmin(values))          # all metrics: lower is better

        bars = ax.bar(np.arange(len(algos)), values, width=0.55,
                      color=colors, alpha=0.88,
                      edgecolor="white", linewidth=0.6)

        # Gold border on best bar
        bars[best_idx].set_edgecolor("goldenrod")
        bars[best_idx].set_linewidth(2.4)

        for i, (bar, val) in enumerate(zip(bars, values)):
            is_best = (i == best_idx)
            dp      = 4 if metric == "Final Fitness" else 2
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + val_span * 0.025,
                    f"{val:.{dp}f}",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold" if is_best else "normal")
            if is_best:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + val_span * 0.11,
                        "★", ha="center", va="bottom",
                        fontsize=11, color="goldenrod")

        ax.set_xticks(np.arange(len(algos)))
        ax.set_xticklabels(algos, rotation=20, ha="right", fontsize=9)
        ax.set_title(_METRIC_LABELS[metric], fontsize=8.5, pad=5)
        ax.set_ylabel(metric, fontsize=8)
        ax.grid(axis="y", alpha=0.28, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, values.max() + val_span * 0.30)

    # Shared legend
    handles = [Patch(facecolor=_ALGO_COLORS[a], label=a, alpha=0.88) for a in algos]
    fig.legend(handles=handles, loc="lower center", ncol=len(algos),
               fontsize=9, bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.suptitle(
        f"Algorithm Comparison — Load Scheduling Benchmark\n"
        f"(pop={POP_SIZE}, iter={MAX_ITER}, "
        f"w1={W_PEAK}, w2={W_COST}, w3={W_VAR}  |  ★ = best per metric)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [benchmark] Bar chart saved -> {save_path}")
    if os.name == "nt":
        os.startfile(save_path)


# ═════════════════════════════════════════════════════════════════════════════
# Console summary
# ═════════════════════════════════════════════════════════════════════════════

def _print_summary(df: pd.DataFrame) -> None:
    metrics = ["Final Fitness", "Peak Load (MW)", "Total Cost ($)",
               "Variance (MW²)", "Runtime (s)"]
    best    = {m: df[m].min() for m in metrics}

    sep = "=" * 82
    print(f"\n{sep}")
    print(f"  ALGORITHM BENCHMARK RESULTS  "
          f"(pop={POP_SIZE}, iter={MAX_ITER}, "
          f"w1={W_PEAK}, w2={W_COST}, w3={W_VAR})")
    print(f"  {'Algorithm':<10} {'Fitness':>10} {'Peak (MW)':>11} "
          f"{'Cost ($)':>13} {'Var (MW²)':>12} {'Time (s)':>10}")
    print("-" * 82)
    for _, row in df.iterrows():
        def _mark(col, dp=2):
            v    = row[col]
            star = " *" if abs(v - best[col]) < 1e-9 else "  "
            return f"{v:{10}.{dp}f}{star}"
        print(f"  {row['Algorithm']:<10}"
              f"{_mark('Final Fitness', 4)}"
              f"{_mark('Peak Load (MW)', 2)}"
              f"{_mark('Total Cost ($)', 2)}"
              f"{_mark('Variance (MW²)', 2)}"
              f"{_mark('Runtime (s)', 3)}")
    print(sep)
    print("  * = best per metric\n")


# ═════════════════════════════════════════════════════════════════════════════
# Master pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    predicted_load: np.ndarray,
    price:          np.ndarray,
) -> pd.DataFrame:
    """
    Run GOA, PSO, GA, DE on the shared fitness function and produce:
      results/algorithm_comparison.csv
      results/algorithm_comparison.png   (grouped bar chart)
      results/algorithm_convergence.png  (convergence curves)

    Parameters
    ----------
    predicted_load : 1-D array of ML-predicted load values (MW)
    price          : 1-D array of TOU electricity prices ($/kWh)

    Returns
    -------
    pd.DataFrame  rows = algorithms, cols = metrics
    """
    evaluator = _FitnessEvaluator(predicted_load, price)

    print("\n" + "=" * 60)
    print("  METAHEURISTIC ALGORITHM BENCHMARK")
    print(f"  Fitness: {W_PEAK}*peak + {W_COST}*cost + {W_VAR}*variance  (normalised)")
    print(f"  pop={POP_SIZE}  iter={MAX_ITER}  seed={SEED}  dim={evaluator.dim}")
    print("=" * 60)

    runners = {
        "GOA": lambda: _run_goa(evaluator),
        "PSO": lambda: _run_mealpy("PSO", evaluator),
        "GA":  lambda: _run_mealpy("GA",  evaluator),
        "DE":  lambda: _run_mealpy("DE",  evaluator),
    }

    rows      = []
    histories = {}

    for name, runner in runners.items():
        print(f"\n  Running {name}...")
        best_schedule, best_fitness, history, runtime = runner()

        kpis = evaluator.kpi(best_schedule)
        row  = {
            "Algorithm":      name,
            "Final Fitness":  round(best_fitness, 6),
            "Peak Load (MW)": kpis["Peak Load (MW)"],
            "Total Cost ($)": kpis["Total Cost ($)"],
            "Variance (MW²)": kpis["Variance (MW²)"],
            "Runtime (s)":    round(runtime, 3),
        }
        rows.append(row)
        histories[name] = history
        print(f"  {name} done — fitness={best_fitness:.6f}  "
              f"peak={kpis['Peak Load (MW)']:.2f}  "
              f"cost={kpis['Total Cost ($)']:.2f}  "
              f"var={kpis['Variance (MW²)']:.2f}  "
              f"time={runtime:.2f}s")

    df = pd.DataFrame(rows)
    _print_summary(df)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "algorithm_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [benchmark] CSV saved -> {csv_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    bar_path = os.path.join(RESULTS_DIR, "algorithm_comparison.png")
    _plot_bar_chart(df, bar_path)

    # ── Convergence curves ────────────────────────────────────────────────────
    conv_path = os.path.join(RESULTS_DIR, "algorithm_convergence.png")
    _plot_convergence(histories, conv_path)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Entry point — loads real DUQ predictions from saved model
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, _ROOT)

    from src.preprocessing import preprocess
    from src.evaluation    import _load_sklearn_models

    print("Loading data and best model...")
    X_train, X_test, y_train, y_test, scaler, train_df, test_df = \
        preprocess(os.path.join(_ROOT, "dataset", "DUQ_hourly.csv"))

    # Use the saved best model for predictions; fall back to y_test if missing
    models = _load_sklearn_models()
    if models:
        best_model     = next(iter(models.values()))
        predicted_load = best_model.predict(X_test)
    else:
        print("  No saved model found — using y_test as predicted load.")
        predicted_load = np.asarray(y_test)

    price = test_df["tou_price"].values[:len(predicted_load)]

    df_results = run_benchmark(predicted_load, price)
    print("\nFinal DataFrame:\n", df_results.to_string(index=False))
