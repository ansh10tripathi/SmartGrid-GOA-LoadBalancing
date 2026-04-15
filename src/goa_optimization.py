"""
src/goa_optimization.py
-----------------------
Grasshopper Optimization Algorithm (GOA) for energy load scheduling.

Reference:
  Saremi, S., Mirjalili, S., & Lewis, A. (2017).
  Grasshopper Optimisation Algorithm: Theory and Application.
  Advances in Engineering Software, 105, 30-47.

Fitness (normalised, minimise):
  0.35 * peak_norm  +  0.25 * par_norm  +  0.25 * cost_norm  +  0.15 * var_norm
  + penalty_weight * constraint_penalty

Physical constraints (penalty method):
  1. Ramp-rate  : |load[t] - load[t-1]| <= max_ramp_rate  (default 15% of mean load)
  2. Cap ceiling: load[t] <= grid_max                      (default 1.10 * max load)
  3. Load floor : load[t] >= load_min                      (default 0.60 * min load)

Constraints are enforced via a quadratic penalty added to the fitness value.
The penalty coefficient (PENALTY_WEIGHT = 10.0) is large enough to make
infeasible solutions uncompetitive while keeping the fitness surface smooth.
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constraint parameters  (module-level defaults, overridable per call)
# ---------------------------------------------------------------------------
RAMP_RATE_FRACTION = 0.15   # max_ramp_rate = fraction * mean(predicted_load)
CAPACITY_FRACTION  = 1.10   # grid_max      = fraction * max(predicted_load)
FLOOR_FRACTION     = 0.60   # load_min      = fraction * min(predicted_load)
PENALTY_WEIGHT     = 10.0   # multiplier on the normalised constraint violation


# ---------------------------------------------------------------------------
# Social interaction function  S(r)
# ---------------------------------------------------------------------------

def _s_function(r: np.ndarray, f: float = 0.5, l: float = 1.5) -> np.ndarray:
    """
    Attraction when r is small, repulsion when r is large.
    f  - intensity of attraction
    l  - attractive length scale
    """
    return f * np.exp(-r / l) - np.exp(-r)


# ---------------------------------------------------------------------------
# Constraint violation  (returns a single non-negative scalar)
# ---------------------------------------------------------------------------

def _constraint_violation(
    schedule:      np.ndarray,
    max_ramp_rate: float,
    grid_max:      float,
    load_min:      float,
) -> float:
    """
    Compute the total normalised constraint violation for one schedule.

    Three constraints, each normalised by its own reference so that
    violations from different constraints are on the same scale:

      1. Ramp-rate : sum of excess |delta| beyond max_ramp_rate
         normalised by (n_steps * max_ramp_rate)

      2. Capacity  : sum of excess above grid_max
         normalised by (n_steps * grid_max)

      3. Floor     : sum of deficit below load_min
         normalised by (n_steps * load_min)

    Returns
    -------
    float >= 0  (0.0 means fully feasible)
    """
    n = len(schedule)

    # 1. Ramp-rate violation
    deltas       = np.abs(np.diff(schedule))                      # (n-1,)
    ramp_excess  = np.maximum(0.0, deltas - max_ramp_rate)
    ramp_viol    = ramp_excess.sum() / max((n - 1) * max_ramp_rate, 1e-10)

    # 2. Capacity ceiling violation
    cap_excess   = np.maximum(0.0, schedule - grid_max)
    cap_viol     = cap_excess.sum() / max(n * grid_max, 1e-10)

    # 3. Minimum floor violation
    floor_deficit = np.maximum(0.0, load_min - schedule)
    floor_viol    = floor_deficit.sum() / max(n * load_min, 1e-10)

    return float(ramp_viol + cap_viol + floor_viol)


# ---------------------------------------------------------------------------
# Fitness function  (objective + penalty)
# ---------------------------------------------------------------------------

def _fitness(
    schedule:      np.ndarray,
    price:         np.ndarray,
    ref_peak:      float,
    ref_cost:      float,
    ref_var:       float,
    ref_par:       float,
    max_ramp_rate: float,
    grid_max:      float,
    load_min:      float,
) -> float:
    """
    Combined objective + quadratic constraint penalty.

    Objective terms (all normalised, minimise):
      0.35 * peak_norm + 0.25 * cost_norm + 0.15 * var_norm + 0.25 * par_norm

    Penalty term:
      PENALTY_WEIGHT * constraint_violation^2
    """
    peak_norm = np.max(schedule) / ref_peak
    cost_norm = np.sum(schedule * price) / ref_cost
    var_norm  = np.var(schedule) / ref_var
    mean      = np.mean(schedule)
    par_norm  = (np.max(schedule) / mean if mean != 0 else 1.0) / ref_par

    objective = (0.35 * peak_norm
                 + 0.25 * cost_norm
                 + 0.15 * var_norm
                 + 0.25 * par_norm)

    violation = _constraint_violation(schedule, max_ramp_rate, grid_max, load_min)
    penalty   = PENALTY_WEIGHT * violation ** 2

    return objective + penalty


# ---------------------------------------------------------------------------
# Constraint diagnostics helper
# ---------------------------------------------------------------------------

def check_constraints(
    schedule:      np.ndarray,
    max_ramp_rate: float,
    grid_max:      float,
    load_min:      float,
    label:         str = "schedule",
) -> dict:
    """
    Return a dict summarising constraint compliance for a schedule.

    Keys
    ----
    ramp_violations   : number of time steps where ramp limit is exceeded
    ramp_max_excess   : worst single ramp excess (kWh)
    cap_violations    : number of steps above grid_max
    cap_max_excess    : worst single ceiling excess (kWh)
    floor_violations  : number of steps below load_min
    floor_max_deficit : worst single floor deficit (kWh)
    feasible          : True if all three constraints are fully satisfied
    """
    deltas      = np.abs(np.diff(schedule))
    ramp_mask   = deltas > max_ramp_rate
    cap_mask    = schedule > grid_max
    floor_mask  = schedule < load_min

    info = {
        "label":            label,
        "ramp_violations":  int(ramp_mask.sum()),
        "ramp_max_excess":  float(np.max(deltas - max_ramp_rate, initial=0.0)),
        "cap_violations":   int(cap_mask.sum()),
        "cap_max_excess":   float(np.max(schedule - grid_max,    initial=0.0)),
        "floor_violations": int(floor_mask.sum()),
        "floor_max_deficit":float(np.max(load_min - schedule,    initial=0.0)),
    }
    info["feasible"] = (
        info["ramp_violations"] == 0
        and info["cap_violations"] == 0
        and info["floor_violations"] == 0
    )
    return info


# ---------------------------------------------------------------------------
# Constraint visualisation
# ---------------------------------------------------------------------------

def plot_constraint_comparison(
    predicted_load:    np.ndarray,
    optimized_load:    np.ndarray,
    max_ramp_rate:     float,
    grid_max:          float,
    load_min:          float,
    save_path:         str = "results/constraint_comparison.png",
) -> None:
    """
    Four-panel figure showing before/after GOA with constraint boundaries:

      Panel 1 (top)   : Load schedule — before vs after with ceiling/floor bands
      Panel 2 (middle): Ramp rates — before vs after with ramp limit line
      Panel 3 (bottom-left) : Constraint violation count bar chart
      Panel 4 (bottom-right): Fitness improvement (objective only, no penalty)
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    t = np.arange(len(predicted_load))

    ramp_before = np.abs(np.diff(predicted_load))
    ramp_after  = np.abs(np.diff(optimized_load))

    info_before = check_constraints(predicted_load, max_ramp_rate, grid_max, load_min, "Before GOA")
    info_after  = check_constraints(optimized_load, max_ramp_rate, grid_max, load_min, "After GOA")

    fig = plt.figure(figsize=(14, 11))
    gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, :])    # full-width top
    ax2 = fig.add_subplot(gs[1, :])    # full-width middle
    ax3 = fig.add_subplot(gs[2, 0])    # bottom-left
    ax4 = fig.add_subplot(gs[2, 1])    # bottom-right

    # ── Panel 1: Load schedule ───────────────────────────────────────────────
    ax1.fill_between(t, load_min, grid_max,
                     alpha=0.08, color="seagreen", label="Feasible band")
    ax1.axhline(grid_max, color="crimson",   lw=1.4, ls="--", label=f"Ceiling = {grid_max:.1f}")
    ax1.axhline(load_min, color="darkorange",lw=1.4, ls="--", label=f"Floor   = {load_min:.1f}")
    ax1.plot(t, predicted_load, color="steelblue", lw=1.5, label="Before GOA")
    ax1.plot(t, optimized_load, color="seagreen",  lw=1.5, label="After GOA")

    # Mark ceiling violations in the before schedule
    cap_viol_idx = np.where(predicted_load > grid_max)[0]
    if len(cap_viol_idx):
        ax1.scatter(cap_viol_idx, predicted_load[cap_viol_idx],
                    color="crimson", zorder=5, s=30, label="Ceiling violation (before)")

    # Mark floor violations in the before schedule
    floor_viol_idx = np.where(predicted_load < load_min)[0]
    if len(floor_viol_idx):
        ax1.scatter(floor_viol_idx, predicted_load[floor_viol_idx],
                    color="darkorange", zorder=5, s=30, marker="v",
                    label="Floor violation (before)")

    ax1.set_ylabel("Load (kWh)")
    ax1.set_title("Load Schedule: Before vs After GOA  |  Capacity & Floor Constraints",
                  fontsize=11)
    ax1.legend(fontsize=7, ncol=4, loc="upper right")
    ax1.grid(alpha=0.25)

    # ── Panel 2: Ramp rates ──────────────────────────────────────────────────
    t_ramp = np.arange(len(ramp_before))
    ax2.axhline(max_ramp_rate, color="crimson", lw=1.4, ls="--",
                label=f"Ramp limit = {max_ramp_rate:.1f}")
    ax2.fill_between(t_ramp, 0, max_ramp_rate,
                     alpha=0.07, color="seagreen", label="Feasible ramp zone")
    ax2.plot(t_ramp, ramp_before, color="steelblue", lw=1.2, alpha=0.8, label="Before GOA")
    ax2.plot(t_ramp, ramp_after,  color="seagreen",  lw=1.2, alpha=0.8, label="After GOA")

    # Highlight ramp violations
    ramp_viol_before = np.where(ramp_before > max_ramp_rate)[0]
    ramp_viol_after  = np.where(ramp_after  > max_ramp_rate)[0]
    if len(ramp_viol_before):
        ax2.scatter(ramp_viol_before, ramp_before[ramp_viol_before],
                    color="steelblue", edgecolors="crimson", lw=1.2,
                    zorder=5, s=35, label=f"Ramp violation before ({len(ramp_viol_before)})")
    if len(ramp_viol_after):
        ax2.scatter(ramp_viol_after, ramp_after[ramp_viol_after],
                    color="seagreen", edgecolors="crimson", lw=1.2,
                    zorder=5, s=35, label=f"Ramp violation after ({len(ramp_viol_after)})")

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("|Load[t] - Load[t-1]| (kWh)")
    ax2.set_title("Ramp Rate: Before vs After GOA  |  Ramp-Rate Constraint", fontsize=11)
    ax2.legend(fontsize=7, ncol=3, loc="upper right")
    ax2.grid(alpha=0.25)

    # ── Panel 3: Violation count comparison ─────────────────────────────────
    constraint_labels = ["Ramp\nViolations", "Ceiling\nViolations", "Floor\nViolations"]
    before_counts = [
        info_before["ramp_violations"],
        info_before["cap_violations"],
        info_before["floor_violations"],
    ]
    after_counts = [
        info_after["ramp_violations"],
        info_after["cap_violations"],
        info_after["floor_violations"],
    ]
    x = np.arange(len(constraint_labels))
    w = 0.35
    bars_b = ax3.bar(x - w/2, before_counts, w, color="steelblue", alpha=0.85, label="Before GOA")
    bars_a = ax3.bar(x + w/2, after_counts,  w, color="seagreen",  alpha=0.85, label="After GOA")
    for bar in list(bars_b) + list(bars_a):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                 str(int(h)), ha="center", va="bottom", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(constraint_labels, fontsize=9)
    ax3.set_ylabel("Number of Violations")
    ax3.set_title("Constraint Violations: Before vs After GOA", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.25)
    feasible_label = "FEASIBLE" if info_after["feasible"] else "INFEASIBLE"
    feasible_color = "seagreen" if info_after["feasible"] else "crimson"
    ax3.text(0.98, 0.97, f"After GOA: {feasible_label}",
             transform=ax3.transAxes, ha="right", va="top",
             fontsize=9, color=feasible_color, fontweight="bold")

    # ── Panel 4: KPI summary ─────────────────────────────────────────────────
    kpi_names  = ["Peak\nLoad", "Max\nRamp", "Ceiling\nExcess", "Floor\nDeficit"]
    kpi_before = [
        predicted_load.max(),
        float(ramp_before.max()),
        info_before["cap_max_excess"],
        info_before["floor_max_deficit"],
    ]
    kpi_after = [
        optimized_load.max(),
        float(ramp_after.max()),
        info_after["cap_max_excess"],
        info_after["floor_max_deficit"],
    ]
    x4 = np.arange(len(kpi_names))
    bars_kb = ax4.bar(x4 - w/2, kpi_before, w, color="steelblue", alpha=0.85, label="Before GOA")
    bars_ka = ax4.bar(x4 + w/2, kpi_after,  w, color="seagreen",  alpha=0.85, label="After GOA")
    for bar in list(bars_kb) + list(bars_ka):
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + kpi_before[0] * 0.005,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=7)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(kpi_names, fontsize=9)
    ax4.set_ylabel("Value (kWh)")
    ax4.set_title("Key Constraint Metrics: Before vs After GOA", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"GOA Physical Constraint Analysis\n"
        f"Ramp limit: {max_ramp_rate:.1f} kWh/step  |  "
        f"Ceiling: {grid_max:.1f} kWh  |  Floor: {load_min:.1f} kWh",
        fontsize=12, y=1.01,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[GOA] Constraint plot saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def grasshopper_optimization(
    predicted_load: np.ndarray,
    price:          np.ndarray,
    n_grasshoppers: int   = 30,
    max_iter:       int   = 100,
    c_min:          float = 0.00004,
    c_max:          float = 1.0,
    random_state:   int   = 42,
    # Physical constraint overrides (None = use module defaults)
    max_ramp_rate:  float = None,
    grid_max:       float = None,
    load_min:       float = None,
) -> dict:
    """
    Run GOA to find an optimised load schedule with physical constraints.

    Parameters
    ----------
    predicted_load : 1-D array of ML-predicted load values (kWh)
    price          : 1-D array of electricity prices ($/kWh), same length
    n_grasshoppers : swarm size
    max_iter       : number of iterations
    c_min / c_max  : comfort-factor bounds (Eq. 2.8 in paper)
    max_ramp_rate  : max |load[t]-load[t-1]| per step; default = 15% of mean load
    grid_max       : hard capacity ceiling; default = 1.10 * max(predicted_load)
    load_min       : hard floor; default = 0.60 * min(predicted_load)

    Returns
    -------
    dict with keys:
        optimized_load      - best feasible (or least-infeasible) schedule
        best_fitness        - fitness value of best schedule (includes penalty)
        fitness_history     - list of best fitness per iteration
        constraints         - dict from check_constraints() for the best schedule
        max_ramp_rate       - ramp limit used
        grid_max            - capacity ceiling used
        load_min            - floor used
    """
    np.random.seed(random_state)

    predicted_load = np.asarray(predicted_load, dtype=float)
    price          = np.asarray(price,          dtype=float)
    dim            = len(predicted_load)

    # ── Derive constraint thresholds ─────────────────────────────────────────
    mean_load = float(predicted_load.mean())
    if max_ramp_rate is None:
        max_ramp_rate = RAMP_RATE_FRACTION * mean_load
    if grid_max is None:
        grid_max = CAPACITY_FRACTION * float(predicted_load.max())
    if load_min is None:
        load_min = FLOOR_FRACTION * float(predicted_load.min())

    print(f"[GOA] Constraints: ramp<={max_ramp_rate:.2f}  "
          f"ceil<={grid_max:.2f}  floor>={load_min:.2f}")

    # ── Non-uniform position bounds (same as original) ────────────────────────
    load_norm = ((predicted_load - predicted_load.min()) /
                 (predicted_load.max() - predicted_load.min() + 1e-10))
    lb = predicted_load * (0.90 - 0.15 * load_norm)
    ub = predicted_load * 1.00

    # ── Reference values for normalisation ───────────────────────────────────
    ref_peak = float(predicted_load.max())
    ref_cost = float(np.sum(predicted_load * price))
    ref_var  = float(np.var(predicted_load)) or 1.0
    ref_par  = float(predicted_load.max() / mean_load)

    def fitness(s: np.ndarray) -> float:
        return _fitness(s, price, ref_peak, ref_cost, ref_var, ref_par,
                        max_ramp_rate, grid_max, load_min)

    # ── Initialise swarm ──────────────────────────────────────────────────────
    positions    = lb + np.random.rand(n_grasshoppers, dim) * (ub - lb)
    fitness_vals = np.array([fitness(positions[i]) for i in range(n_grasshoppers)])

    best_idx     = int(np.argmin(fitness_vals))
    best_pos     = positions[best_idx].copy()
    best_fitness = float(fitness_vals[best_idx])
    fitness_history = [best_fitness]

    # ── Main loop ─────────────────────────────────────────────────────────────
    for iteration in range(max_iter):

        c = c_max - iteration * (c_max - c_min) / max_iter

        new_positions = np.zeros_like(positions)

        for i in range(n_grasshoppers):
            social_sum = np.zeros(dim)

            for j in range(n_grasshoppers):
                if i == j:
                    continue
                diff      = positions[j] - positions[i]
                dist      = np.linalg.norm(diff) + 1e-10
                direction = diff / dist
                s_val     = _s_function(np.array([dist]))[0]
                social_sum += c * ((ub - lb) / 2) * s_val * direction

            new_positions[i] = (
                c * social_sum
                + 0.5 * positions[i]
                + 0.5 * best_pos
                + np.random.normal(0, 0.01, dim)
            )

        positions    = np.clip(new_positions, lb, ub)
        fitness_vals = np.array([fitness(positions[i]) for i in range(n_grasshoppers)])

        current_best = int(np.argmin(fitness_vals))
        if fitness_vals[current_best] < best_fitness:
            best_fitness = float(fitness_vals[current_best])
            best_pos     = positions[current_best].copy()

        fitness_history.append(best_fitness)

        if (iteration + 1) % 20 == 0:
            print(f"  [GOA] iter {iteration+1:>3}/{max_iter}  "
                  f"best_fitness={best_fitness:.6f}")

    constraints = check_constraints(best_pos, max_ramp_rate, grid_max, load_min,
                                    label="After GOA")
    feasible_str = "FEASIBLE" if constraints["feasible"] else "INFEASIBLE"
    print(f"[GOA] Done. Best fitness={best_fitness:.6f}  |  Solution: {feasible_str}")
    print(f"      Ramp violations : {constraints['ramp_violations']}")
    print(f"      Ceiling violations: {constraints['cap_violations']}")
    print(f"      Floor violations  : {constraints['floor_violations']}")

    return {
        "optimized_load":  best_pos,
        "best_fitness":    best_fitness,
        "fitness_history": fitness_history,
        "constraints":     constraints,
        "max_ramp_rate":   max_ramp_rate,
        "grid_max":        grid_max,
        "load_min":        load_min,
    }


# ---------------------------------------------------------------------------
# Quick test + constraint visualisation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

    np.random.seed(0)
    n         = 24
    pred_load = np.random.uniform(150, 350, n)
    price     = np.random.uniform(0.08, 0.15, n)

    print("=" * 60)
    print("  GOA WITH PHYSICAL CONSTRAINTS  --  quick test")
    print("=" * 60)

    result = grasshopper_optimization(
        pred_load, price,
        n_grasshoppers=20, max_iter=100,
    )

    opt = result["optimized_load"]
    c   = result["constraints"]

    print("\nOriginal load (first 5):", pred_load[:5].round(2))
    print("Optimised load (first 5):", opt[:5].round(2))
    print(f"\nPeak  before: {pred_load.max():.2f}  |  after: {opt.max():.2f}")
    print(f"Cost  before: {np.sum(pred_load*price):.2f}  |  "
          f"after: {np.sum(opt*price):.2f}")
    print(f"\nConstraint summary (after GOA):")
    for k, v in c.items():
        if k != "label":
            print(f"  {k:<22}: {v}")

    # Before-GOA constraint diagnostics
    c_before = check_constraints(
        pred_load,
        result["max_ramp_rate"],
        result["grid_max"],
        result["load_min"],
        label="Before GOA",
    )
    print(f"\nConstraint summary (before GOA):")
    for k, v in c_before.items():
        if k != "label":
            print(f"  {k:<22}: {v}")

    # Generate constraint visualisation
    os.makedirs("results", exist_ok=True)
    plot_constraint_comparison(
        pred_load, opt,
        result["max_ramp_rate"],
        result["grid_max"],
        result["load_min"],
        save_path="results/constraint_comparison.png",
    )
