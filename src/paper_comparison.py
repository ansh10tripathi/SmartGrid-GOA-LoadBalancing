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
    caption: str = (
        "Performance comparison of forecasting models on the DUQ hourly "
        "load dataset (test set, 2005--2018). "
        "Best result per metric shown in bold."
    ),
    label: str = "tab:model_comparison",
) -> pd.DataFrame:
    """
    Full pipeline: load data -> load saved models -> metrics -> table -> chart.
    Delegates entirely to evaluation.build_model_comparison_table.

    Returns the results DataFrame (rows = models, cols = metrics).
    """
    from src.preprocessing import preprocess
    from src.evaluation    import build_model_comparison_table

    print("\n[paper_comparison] Loading and preprocessing data...")
    _, X_test, _, y_test, _, _, _ = \
        preprocess(os.path.join(_ROOT, "dataset", "DUQ_hourly.csv"))

    return build_model_comparison_table(
        X_test, y_test, caption=caption, label=label
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_paper_comparison()
