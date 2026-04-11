# Smart Grid Load Forecasting and Optimization using Machine Learning and Grasshopper Optimization Algorithm (GOA)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)

## ?? 1. Project Title & Description
This project focuses on the **Prediction and Optimization** of electrical power dispatch within a Smart Grid environment. By combining state-of-the-art **Machine Learning (ML)** models to forecast real-time power demand with the **Grasshopper Optimization Algorithm (GOA)**, this pipeline dynamically solves the Economic Load Dispatch (ELD) problem. The end goal is finding the absolute lowest generation cost while strictly meeting physiological system constraints and fluctuating consumer demands.

## ?? 2. Problem Statement
Power grids face highly non-linear, dynamic load requests influenced by human activity and external forces (weather, holidays). Relying on mathematical forecasting alone is insufficient; utility companies must know **how** to dispatch their varying energy generators (which all have unique minimum/maximum capacities and cost curves) to meet that predicted load efficiently. Failing to optimize dispatch results in significant monetary loss and potential grid instability (overloading or outages).

## ?? 3. Key Features
*   **Dual-End Pipeline**: Unified ML Forecasting + Meta-heuristic Optimization framework.
*   **Cyclical & Temporal Engineering**: Implements sine/cosine encodings for time-based features preventing dimensional sparsity.
*   **Model Benchmarking**: Benchmarks statistical trees (XGBoost, Random Forest) against Deep Learning (LSTM).
*   **Explainable AI (XAI)**: Integrates SHAP values to decode the model's logic for utility administrators.
*   **Probabilistic Forecasting**: Employs Quantile Regression for boundary/interval predictions (Risk Management).
*   **Interactive UI**: A fully functional Streamlit dashboard allowing users to interact with prediction horizons and dispatch algorithms actively.

## ??? 4. Architecture Diagram
`	ext
[ Raw Grid Data (DUQ) ] ? [ Data Preprocessing & Leakage Audit ]
                                      ?
           (Lag Features, Rolling Means, Cyclical Encodings, Scaling)
                                      ?
[ Machine Learning Layer ] ? (Model Comparison: LSTM, XGBoost, SVR, RF)
                                      ?
                            (Predicted Megawatt Load)
                                      ?
============== [ GRASSHOPPER OPTIMIZATION ALGORITHM (GOA) ] ==============
   Constraint 1: Sum(Generation) == Predicted Load
   Constraint 2: Generation Min/Max Bounds
                                      ?
               [ Optimal Generator Dispatch Setpoints ]
                                      ?
[ Streamlit UI + SHAP Explainability Dashboard + Financial Cost Reports ]
`

## ?? 5. Dataset Description
*   **Source Data**: Real-world SCADA hourly demand records (e.g., DUQ Hourly).
*   **Target Variable**: Total Load (Megawatts).
*   **Temporal Frequency**: Hourly records over multiple years ensuring high capture of daily/weekly/seasonal behaviors.
*   **Features Used**: Engineered temporal patterns, moving averages, and autoregressive lag indicators.

## ?? 6. Feature Engineering Explanation
*   **Cyclical Encoding**: Hour (0-23) and Month (1-12) are translated into Sine/Cosine pairs to reflect continuous temporal boundaries.
*   **Lag-24 / Lag-1**: Captures grid inertia. Current usage strongly correlates with usage exactly 24 hours ago.
*   **Rolling Means**: Moving average to encapsulate broader trend variations, abstracting short-term random noise.
*   **Strict Split Scaling**: Data is rigorously chronologically split *before* MinMaxScaler is fitted on training to eliminate data leakage.

## ?? 7. Models Used (Comparison)
1.  **LSTM (Deep Learning)**: Ideal for extracting highly complex, long-term sequential temporal relationships via recurrent memory gating.
2.  **XGBoost (Gradient Boosting)**: Rapid, highly regularized mathematically advanced tree-modeling preventing overfitting dynamically.
3.  **Random Forest**: Baseline ensemble model generating excellent feature importance metrics.
4.  **Support Vector Regression (SVR)**: Utilizes the kernel trick to map non-linear loads with defined epsilon margins.

## ?? 8. Optimization (GOA Explanation)
The **Grasshopper Optimization Algorithm (GOA)** is a meta-heuristic simulating grasshopper swarming mechanics:
*   **Nymph Phase (Exploration)**: Large algorithmic leaps covering the total mathematical search space rapidly.
*   **Adult Phase (Exploitation)**: Minute localized searches near the best-found "food source" (optimal point).
*   **Application**: Taking the ML's *Predicted Load*, the GOA minimizes the quadratic cost formulas of available electrical generators (ensuring no generator exceeds max limits or falls below safe operating minimums).

## ?? 9. Results (Metrics & Enhancements)
*   **R� Score**: ~99% on 1-step ahead forecasting (highlighting massive autocorrelation logic).
*   **RMSE & MAE**: Demonstrable low error margins mapped strictly compared against naive baseline retention models.
*   **Optimization Validation**: Complete alignment with strict demand constraint boundaries preventing "Loss of Load" simulations.

## ??? 10. Visualizations
1.  **Actual vs. Predicted Load**: Time-series overlay graphs.
2.  **Model Comparison Bar Charts**: RMSE & R� comparisons for RF, XGB, SVR, LSTM.
3.  **SHAP Feature Importance**: Force plots identifying driver variables (Hour, Lag-Load).
4.  **GOA Convergence Curve**: Showcasing fitness cost minimization across iterations.

## ??? 11. Dashboard Features (Streamlit)
*   **Predictive Dial**: Input custom dates/hours to fetch instant forecasted load demand.
*   **Optimization Toggle**: Apply GOA onto the forecasted model to witness active dispatch mechanics.
*   **Model Switcher**: Select between XGBoost, Random Forest, or LSTM on the fly.
*   **Metric Ticker View**: Clean display of computational time, total generation cost ($), and error metrics.

## ?? 12. Installation Steps
\\\ash
# 1. Clone the Repository
git clone https://github.com/yourusername/SmartGrid-GOA-LoadBalancing.git
cd SmartGrid-GOA-LoadBalancing

# 2. Set up Virtual Environment (Recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt
\\\

## ?? 13. How to Run
\\\ash
# Run the complete headless training and optimization pipeline
python main.py

# Launch the Interactive UI Dashboard
streamlit run app.py
\\\

## ?? 14. Project Structure
\\\	ext
SmartGrid-GOA-LoadBalancing/
�   app.py                     # Streamlit frontend dashboard
�   main.py                    # Master execution CLI script
�   requirements.txt           # Library dependencies
�   README.md                  # Project documentation
�   viva_questions.md          # Viva preparation QA
+-- dataset/                   # Raw & Processed CSVs
+-- models/                    # Serialized .pkl and .pt model objects
+-- results/                   # Evaluation metrics, npz arrays, tables
+-- src/                       # Central utility and logic scripts
    +-- preprocessing.py       # Eng/Scaling logic
    +-- forecasting_model.py   # Tree/SVR implementations
    +-- lstm_model.py          # PyTorch Recurrent logic
    +-- goa_optimization.py    # Grasshopper algorithmic logic
    +-- explainability.py      # SHAP integration
\\\

## ?? 15. Future Work
*   Integration of Multi-Objective Optimization (MOGOA) mapping Carbon Emission reduction against monetary generation Cost.
*   Deploying an online-learning system with triggering drift detection (retraining on dynamic SCADA input streams).
*   Live API configuration utilizing FastAPI for internal grid SCADA commands.

## ?? 16. License
This project is licensed under the [MIT License](LICENSE).
