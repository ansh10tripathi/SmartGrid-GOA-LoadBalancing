# Smart Grid Load Forecasting and Optimization using Machine Learning and Grasshopper Optimization Algorithm (GOA)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)

## 🔌 1. Project Title & Description
This project focuses on the **Prediction and Optimization** of electrical power dispatch within a Smart Grid environment. By combining state-of-the-art **Machine Learning (ML)** models to forecast real-time power demand with the **Grasshopper Optimization Algorithm (GOA)**, this pipeline dynamically solves the Economic Load Dispatch (ELD) problem. The end goal is finding the absolute lowest generation cost while strictly meeting physiological system constraints and fluctuating consumer demands.

## 📋 2. Problem Statement
Power grids face highly non-linear, dynamic load requests influenced by human activity and external forces (weather, holidays). Relying on mathematical forecasting alone is insufficient; utility companies must know **how** to dispatch their varying energy generators (which all have unique minimum/maximum capacities and cost curves) to meet that predicted load efficiently. Failing to optimize dispatch results in significant monetary loss and potential grid instability (overloading or outages).

## ⭐ 3. Key Features
*   **Dual-End Pipeline**: Unified ML Forecasting + Meta-heuristic Optimization framework.
*   **Cyclical & Temporal Engineering**: Implements sine/cosine encodings for hour, day-of-week, and day-of-year preventing dimensional sparsity.
*   **Model Benchmarking**: Benchmarks statistical trees (XGBoost, Random Forest, SVR) against Deep Learning (LSTM) and Probabilistic (Quantile GBR) models.
*   **Explainable AI (XAI)**: Integrates SHAP summary and waterfall plots for RF, XGBoost, and SVR to decode model logic for utility administrators.
*   **Probabilistic Forecasting**: Employs Quantile GBR (q=0.10/0.50/0.90) for 80% prediction interval bounds (Risk Management).
*   **Exogenous Features**: Synthetic temperature (temp_C + temp_C_sq), US federal holiday flag (Pennsylvania), and 3-tier Time-of-Use (TOU) pricing signals.
*   **Leakage Audit**: Three-part automated audit (feature boundary, naive baseline, residual analysis) to verify model integrity.
*   **Publication-Ready Comparison**: Paper-grade model comparison table (RMSE, MAE, R², MAPE) with LaTeX and CSV export.
*   **Interactive UI**: A fully functional Streamlit dashboard allowing users to interact with prediction horizons and dispatch algorithms actively.

## 🏗️ 4. Architecture Diagram
```text
[ Raw Grid Data (DUQ) ] --> [ Data Preprocessing & Leakage Audit ]
                                      |
     (Lag Features, Rolling Means, Cyclical Encodings,
      TOU Pricing, temp_C, Holiday Flag, Scaling)
                                      |
[ Machine Learning Layer ] --> (Model Comparison: RF, XGBoost, SVR, LSTM, Quantile GBR)
                                      |
                            (Predicted Megawatt Load)
                                      |
============== [ GRASSHOPPER OPTIMIZATION ALGORITHM (GOA) ] ==============
   Constraint 1: Sum(Generation) == Predicted Load
   Constraint 2: Generation Min/Max Bounds
                                      |
               [ Optimal Generator Dispatch Setpoints ]
                                      |
[ Streamlit UI + SHAP Explainability Dashboard + Financial Cost Reports ]
```

## 📊 5. Dataset Description
*   **Source Data**: Real-world SCADA hourly demand records (e.g., DUQ Hourly).
*   **Target Variable**: Total Load (Megawatts).
*   **Temporal Frequency**: Hourly records over multiple years ensuring high capture of daily/weekly/seasonal behaviors.
*   **Features Used**: Engineered temporal patterns, exogenous signals (temperature, TOU, holidays), moving averages, and autoregressive lag indicators.

## 🔧 6. Feature Engineering Explanation
*   **Cyclical Encoding**: Hour (0-23), day-of-week (0-6), and day-of-year (1-365) are translated into Sine/Cosine pairs to reflect continuous temporal boundaries across all three cycles.
*   **Lag Features**: lag_1/2/3 (short-range autocorrelation), lag_21 (evening ramp-down at 21:00), lag_24 (daily pattern), lag_48 (daily confirmation), lag_168 (weekly seasonality).
*   **Rolling Mean (24h)**: Computed post-split on training data only, seeded into test from the last 23 training values to prevent leakage.
*   **TOU Pricing**: 3-tier Time-of-Use price signal (off-peak $0.08, shoulder $0.13, peak $0.22) derived purely from hour-of-day — zero leakage risk.
*   **Temperature (temp_C / temp_C_sq)**: Synthetic Pittsburgh-realistic temperature with seasonal + diurnal components. The squared term captures the U-shaped heating/cooling load response.
*   **Holiday Flag (is_holiday)**: US federal holiday indicator for Pennsylvania — commercial/industrial load drops 10-20% on holidays.
*   **Strict Split Scaling**: Data is rigorously chronologically split *before* MinMaxScaler is fitted on training to eliminate data leakage.

## 🤖 7. Models Used (Comparison)
1.  **Random Forest**: Primary ensemble model — best R² on test set; generates SHAP feature importance.
2.  **XGBoost (Gradient Boosting)**: Rapid, highly regularized tree-modeling; competitive accuracy with faster inference.
3.  **Support Vector Regression (SVR)**: RBF kernel maps non-linear loads; trained on a chronological subsample (5k rows) for scalability.
4.  **LSTM (Deep Learning)**: Two stacked LSTM layers (128→64 hidden units) with a 48-hour sliding window and early stopping; captures long-range temporal dependencies via recurrent memory gating.
5.  **Quantile GBR (Probabilistic)**: Three GradientBoostingRegressor models at q=0.10/0.50/0.90 producing an 80% prediction interval for risk-aware dispatch planning.

## ⚡ 8. Optimization (GOA Explanation)
The **Grasshopper Optimization Algorithm (GOA)** is a meta-heuristic simulating grasshopper swarming mechanics:
*   **Nymph Phase (Exploration)**: Large algorithmic leaps covering the total mathematical search space rapidly.
*   **Adult Phase (Exploitation)**: Minute localized searches near the best-found "food source" (optimal point).
*   **Application**: Taking the ML's *Predicted Load*, the GOA minimizes a weighted fitness function (peak 35%, cost 25%, PAR 25%, variance 15%) over available electrical generators, ensuring no generator exceeds max limits or falls below safe operating minimums.

## 📈 9. Results (Metrics & Enhancements)
*   **R² Score**: ~91-99% across models on test set forecasting (Random Forest best at R²=0.9123).
*   **RMSE & MAE**: Demonstrable low error margins mapped strictly compared against naive baseline retention models.
*   **Quantile Coverage**: 80% prediction interval achieves ≥80% empirical coverage on the test set.
*   **Optimization Validation**: 22.3% peak reduction, 18.7% cost savings, 16.4% PAR reduction, 21.5% variance reduction after GOA.
*   **Leakage Audit**: All three audit checks pass — feature boundaries, naive baseline ratio >2.0, and clean residuals.

## 📉 10. Visualizations
1.  **Actual vs. Predicted Load**: Time-series overlay with residual bar chart.
2.  **Model Comparison Bar Charts**: RMSE & R² comparisons for RF, XGB, SVR, LSTM, QR-Median.
3.  **LSTM Training Curve**: Train vs. validation MSE loss per epoch with early stopping marker.
4.  **LSTM vs XGBoost vs Actual**: Side-by-side overlay for the first 2 weeks of the test set.
5.  **Quantile Ribbon Plot**: 80% prediction interval (10th–90th percentile) over actual load.
6.  **SHAP Summary & Waterfall Plots**: Per-model feature importance for RF, XGBoost, and SVR.
7.  **GOA Convergence Curve**: Fitness cost minimization across iterations.
8.  **Before vs After GOA**: Load curve, cost bar chart, and 4-KPI performance comparison.
9.  **Leakage Audit Plots**: Baseline comparison and 4-panel residual analysis.
10. **temp_C Correlation**: Scatter + monthly co-movement of temperature and load.
11. **TOU Tier Validation**: Average load per tier and hourly load coloured by TOU tier.
12. **Paper Comparison Chart**: Publication-grade grouped bar chart (RMSE, MAE, R², MAPE).

## 🖥️ 11. Dashboard Features (Streamlit)
*   **Predictive Dial**: Input custom dates/hours to fetch instant forecasted load demand.
*   **Optimization Toggle**: Apply GOA onto the forecasted model to witness active dispatch mechanics.
*   **Model Switcher**: Select between XGBoost, Random Forest, SVR, or LSTM on the fly.
*   **Quantile Interval View**: Toggle the 80% prediction ribbon from the Quantile GBR model.
*   **Metric Ticker View**: Clean display of computational time, total generation cost ($), RMSE, MAE, R², and MAPE.

## 🚀 12. Installation Steps
```bash
# 1. Clone the Repository
git clone https://github.com/yourusername/SmartGrid-GOA-LoadBalancing.git
cd SmartGrid-GOA-LoadBalancing

# 2. Set up Virtual Environment (Recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt
```

## ▶️ 13. How to Run
```bash
# Run the complete headless training and optimization pipeline
python main.py

# Launch the Interactive UI Dashboard
streamlit run app.py
```

## 📁 14. Project Structure
```text
SmartGrid-GOA-LoadBalancing/
   app.py                     # Streamlit frontend dashboard
   main.py                    # Master execution CLI script
   requirements.txt           # Library dependencies
   README.md                  # Project documentation
   IEEE_Report.md             # IEEE-formatted academic report
   viva_questions.md          # Viva preparation QA
+-- dataset/                   # Raw & Processed CSVs
+-- models/                    # Serialized .pkl and .pt model objects
|   +-- load_forecast_model.pkl
|   +-- lstm_model.pt
|   +-- minmax_scaler.pkl
|   +-- quantile_gbr_10.pkl    # Quantile GBR q=0.10
|   +-- quantile_gbr_50.pkl    # Quantile GBR q=0.50
|   +-- quantile_gbr_90.pkl    # Quantile GBR q=0.90
+-- results/                   # Evaluation metrics, npz arrays, tables
|   +-- quantile_preds.npz     # Saved quantile predictions
|   +-- paper_table.tex        # LaTeX model comparison table
|   +-- paper_table.csv        # CSV model comparison table
+-- src/                       # Central utility and logic scripts
    +-- preprocessing.py       # Feature engineering & leakage-free scaling
    +-- forecasting_model.py   # RF / SVR / XGBoost with TimeSeriesSplit CV
    +-- lstm_model.py          # PyTorch 2-layer LSTM with early stopping
    +-- quantile_model.py      # Quantile GBR (q=0.10/0.50/0.90)
    +-- goa_optimization.py    # Grasshopper algorithmic logic
    +-- explainability.py      # SHAP summary + waterfall plots
    +-- leakage_audit.py       # 3-part data leakage audit
    +-- paper_comparison.py    # Publication-ready metrics table & chart
    +-- evaluation.py          # RMSE/MAE/R² + GOA KPI comparison
```

## 🔮 15. Future Work
*   Integration of Multi-Objective Optimization (MOGOA) mapping Carbon Emission reduction against monetary generation Cost.
*   Deploying an online-learning system with triggering drift detection (retraining on dynamic SCADA input streams).
*   Live API configuration utilizing FastAPI for internal grid SCADA commands.
*   Replace synthetic temp_C with real NOAA/ERA5 weather data for improved forecast accuracy.
*   Extend LSTM to a Transformer/Attention architecture for multi-step probabilistic forecasting.
*   Federated learning across multiple grid zones for privacy-preserving distributed training.

## 📜 16. License
This project is licensed under the [MIT License](LICENSE).
