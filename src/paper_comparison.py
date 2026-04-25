"""
src/paper_comparison.py
-----------------------
Publication-ready model comparison table and chart for the DUQ dataset.

This module is a thin entry-point that delegates all metric computation,
LaTeX rendering, and chart generation to evaluation.build_model_comparison_table.
All logic lives in one place — evaluation.py Layer 3.

Outputs (written by evaluation.py)
-------
  results/paper_comparison.png   — grouped bar chart (4 metrics x N models)
  results/paper_table.tex        — LaTeX booktabs table, best per metric bold
  results/paper_table.csv        — raw numbers for reference
"""

import os
import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _ROOT)


def run_paper_comparison(
    X_test=None,
    y_test=None,
    target_scaler=None,
    caption: str = (
        "Performance comparison of forecasting models on the DUQ hourly "
        "load dataset (test set, 2005--2018). "
        "Best result per metric shown in bold."
    ),
    label: str = "tab:model_comparison",
) -> pd.DataFrame:
    """
    Full pipeline: load saved models -> metrics -> table -> chart.
    Delegates entirely to evaluation.build_model_comparison_table.

    CRITICAL: X_test, y_test, and target_scaler MUST be passed from main.py
    to ensure the same test set used during training is used for evaluation.
    DO NOT call preprocess() here — that creates a new split and causes leakage.

    Parameters
    ----------
    X_test        : test features (same split used during training)
    y_test        : test targets (same split used during training)
    target_scaler : scaler fitted on y_train only

    Returns the results DataFrame (rows = models, cols = metrics).
    """
    from src.evaluation import build_model_comparison_table

    if X_test is None or y_test is None:
        raise ValueError(
            "X_test and y_test must be provided from main.py. "
            "DO NOT call preprocess() here — use the same test set from training."
        )

    print("\n[paper_comparison] Using provided test set (no re-split).")
    return build_model_comparison_table(
        X_test, y_test, target_scaler=target_scaler, caption=caption, label=label
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_paper_comparison()
