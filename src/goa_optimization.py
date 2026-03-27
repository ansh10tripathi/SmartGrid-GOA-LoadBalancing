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

All terms are normalised against the reference (predicted) load so they are
comparable regardless of absolute load magnitude.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# GOA Core
# ─────────────────────────────────────────────────────────────────────────────

def _s_function(r: np.ndarray, f: float = 0.5, l: float = 1.5) -> np.ndarray:
    """
    Social interaction function S(r).
    Attraction when r is small, repulsion when r is large.
    f  – intensity of attraction
    l  – attractive length scale
    """
    return f * np.exp(-r / l) - np.exp(-r)


def _fitness(schedule: np.ndarray, price: np.ndarray,
             ref_peak: float, ref_cost: float, ref_var: float, ref_par: float) -> float:
    peak_norm = np.max(schedule) / ref_peak
    cost_norm = np.sum(schedule * price) / ref_cost
    var_norm  = np.var(schedule) / ref_var
    mean = np.mean(schedule)
    par_norm  = (np.max(schedule) / mean if mean != 0 else 1.0) / ref_par
    return 0.35 * peak_norm + 0.25 * cost_norm + 0.15 * var_norm + 0.25 * par_norm


def grasshopper_optimization(
    predicted_load: np.ndarray,
    price:          np.ndarray,
    n_grasshoppers: int   = 30,
    max_iter:       int   = 100,
    c_min: float = 0.00004,
    c_max: float = 1.0,
    random_state: int = 42,
) -> dict:
    """
    Run GOA to find an optimised load schedule.

    Parameters
    ----------
    predicted_load : 1-D array of ML-predicted load values (kWh)
    price          : 1-D array of electricity prices ($/kWh), same length
    n_grasshoppers : swarm size
    max_iter       : number of iterations
    c_min / c_max  : comfort-factor bounds (from original paper, Eq. 2.8)

    Returns
    -------
    dict with keys:
        optimized_load  – best schedule found
        best_fitness    – fitness value of best schedule
        fitness_history – list of best fitness per iteration
    """
    np.random.seed(random_state)

    dim = len(predicted_load)
    # Non-uniform bounds: high-load steps can drop more (lb=0.75x),
    # low-load steps are held up (lb=0.90x) — flattens curve, reduces PAR
    load_norm = ((predicted_load - predicted_load.min()) /
                 (predicted_load.max() - predicted_load.min() + 1e-10))
    lb = predicted_load * (0.90 - 0.15 * load_norm)
    ub = predicted_load * 1.00   # never exceed predicted load

    # ── Initialise grasshopper positions uniformly in [lb, ub] ──────────────
    # positions shape: (n_grasshoppers, dim)
    positions = lb + np.random.rand(n_grasshoppers, dim) * (ub - lb)

    # Reference values from predicted load for normalization
    ref_peak = float(np.max(predicted_load))
    ref_cost = float(np.sum(predicted_load * price))
    ref_var  = float(np.var(predicted_load))
    ref_par  = float(np.max(predicted_load) / np.mean(predicted_load))
    if ref_var == 0:
        ref_var = 1.0

    def fitness(s):
        return _fitness(s, price, ref_peak, ref_cost, ref_var, ref_par)

    # Evaluate initial fitness
    fitness_vals = np.array([fitness(positions[i]) for i in range(n_grasshoppers)])

    best_idx      = np.argmin(fitness_vals)
    best_pos      = positions[best_idx].copy()
    best_fitness  = fitness_vals[best_idx]
    fitness_history = [best_fitness]

    # ── Main loop ────────────────────────────────────────────────────────────
    for iteration in range(max_iter):

        # Linearly decrease comfort factor c  (Eq. 2.8 in paper)
        c = c_max - iteration * (c_max - c_min) / max_iter

        new_positions = np.zeros_like(positions)

        for i in range(n_grasshoppers):
            social_sum = np.zeros(dim)

            for j in range(n_grasshoppers):
                if i == j:
                    continue

                diff = positions[j] - positions[i]          # (dim,)
                dist = np.linalg.norm(diff) + 1e-10          # scalar

                # Normalised direction vector
                direction = diff / dist

                # S-function value (scalar)
                s_val = _s_function(np.array([dist]))[0]

                social_sum += c * ((ub - lb) / 2) * s_val * direction

            # Position update (Eq. 2.7): social interaction + attraction to target
            # Add exploration + current position influence
            new_positions[i] = (
                c * social_sum +
                0.5 * positions[i] +
                0.5 * best_pos
            )
            # 🔥 Add randomness (VERY IMPORTANT)
            noise = np.random.normal(0, 0.01, dim)
            new_positions[i] += noise

        # Clip to bounds
        positions = np.clip(new_positions, lb, ub)

        # Re-evaluate fitness
        fitness_vals = np.array([fitness(positions[i]) for i in range(n_grasshoppers)])

        current_best_idx = np.argmin(fitness_vals)
        if fitness_vals[current_best_idx] < best_fitness:
            best_fitness = fitness_vals[current_best_idx]
            best_pos     = positions[current_best_idx].copy()

        fitness_history.append(best_fitness)

        if (iteration + 1) % 20 == 0:
            print(f"  [GOA] iter {iteration+1:>3}/{max_iter}  "
                  f"best_fitness={best_fitness:.4f}")

    print(f"[GOA] Optimisation complete. Best fitness = {best_fitness:.4f}")
    return {
        "optimized_load":  best_pos,
        "best_fitness":    best_fitness,
        "fitness_history": fitness_history,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    n = 24
    pred_load = np.random.uniform(150, 350, n)
    price     = np.random.uniform(0.08, 0.15, n)

    result = grasshopper_optimization(pred_load, price, n_grasshoppers=20, max_iter=50)

    print("\nOriginal load (first 5):", pred_load[:5].round(2))
    print("Optimised load (first 5):", result["optimized_load"][:5].round(2))
    print(f"Peak before: {pred_load.max():.2f}  |  Peak after: {result['optimized_load'].max():.2f}")
