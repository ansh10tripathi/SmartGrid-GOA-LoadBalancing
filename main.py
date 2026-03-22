"""
main.py
-------
End-to-end pipeline runner.
Usage:  python main.py
"""

from src.preprocessing      import preprocess
from src.forecasting_model  import run_forecasting_pipeline
from src.goa_optimization   import grasshopper_optimization
from src.evaluation         import compare_before_after


def main():
    # ── Step 1: Preprocess ───────────────────────────────────────────────────
    print("\nStep 1: Preprocessing...")
    X, y, scaler, raw_df = preprocess("dataset/smartgrid.csv")

    # ── Step 2: Train & evaluate Random Forest ───────────────────────────────
    print("\nStep 2: Training Model...")
    model, metrics, y_test, y_pred, X_test = run_forecasting_pipeline(
        X, y, use_search=True
    )
    print("  Metrics:", metrics)

    # ── Step 3: GOA optimisation ─────────────────────────────────────────────
    print("\nStep 3: Running GOA Optimization...")
    test_start = len(y) - len(y_test)
    price_test = raw_df["price"].values[test_start : test_start + len(y_test)]

    goa_result = grasshopper_optimization(
        predicted_load = y_pred,
        price          = price_test,
        n_grasshoppers = 30,
        max_iter       = 100,
        w1=0.4, w2=0.3, w3=0.3,
    )
    optimized_load = goa_result["optimized_load"]

    # ── Step 4: Evaluate before vs after ────────────────────────────────────
    print("\nStep 4: Evaluating Results...")
    compare_before_after(y_pred, optimized_load, price_test)

    print("\n✅ Project Execution Completed!")


if __name__ == "__main__":
    main()
