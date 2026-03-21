"""
src/goa_optimization.py
-----------------------
Grasshopper Optimization Algorithm (GOA) for energy load scheduling.

Reference:
  Saremi, S., Mirjalili, S., & Lewis, A. (2017).
  "Grasshopper Optimisation Algorithm: Theory and Application."
  Advances in Engineering Software, 105, 30-47.

Fitness = w1 * PeakLoad  +  w2 * Cost  +  w3 * Variance
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
             w1: float, w2: float, w3: float) -> float:
    """
    Composite fitness (lower is better).
      w1 * peak_load  +  w2 * total_cost  +  w3 * variance
    """
    peak     = np.max(schedule)
    cost     = np.sum(schedule * price)
    variance = np.var(schedule)
    return w1 * peak + w2 * cost + w3 * variance


def grasshopper_optimization(
    predicted_load: np.ndarray,
    price:          np.ndarray,
    n_grasshoppers: int   = 30,
    max_iter:       int   = 100,
    w1: float = 0.4,
    w2: float = 0.3,
    w3: float = 0.3,
    c_min: float = 0.00004,
    c_max: float = 1.0,
    lb_factor: float = 0.80,   # lower bound = 80 % of predicted load
    ub_factor: float = 1.20,   # upper bound = 120 % of predicted load
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
    w1, w2, w3     : fitness weights (peak, cost, variance)
    c_min / c_max  : comfort-factor bounds (from original paper)
    lb_factor      : lower bound as fraction of predicted load
    ub_factor      : upper bound as fraction of predicted load

    Returns
    -------
    dict with keys:
        optimized_load  – best schedule found
        best_fitness    – fitness value of best schedule
        fitness_history – list of best fitness per iteration
    """
    np.random.seed(random_state)

    dim = len(predicted_load)
    lb  = predicted_load * lb_factor   # shape (dim,)
    ub  = predicted_load * ub_factor   # shape (dim,)

    # ── Initialise grasshopper positions uniformly in [lb, ub] ──────────────
    # positions shape: (n_grasshoppers, dim)
    positions = lb + np.random.rand(n_grasshoppers, dim) * (ub - lb)

    # Evaluate initial fitness
    fitness_vals = np.array([
        _fitness(positions[i], price, w1, w2, w3)
        for i in range(n_grasshoppers)
    ])

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
            new_positions[i] = social_sum + best_pos

        # Clip to bounds
        positions = np.clip(new_positions, lb, ub)

        # Re-evaluate fitness
        fitness_vals = np.array([
            _fitness(positions[i], price, w1, w2, w3)
            for i in range(n_grasshoppers)
        ])

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
