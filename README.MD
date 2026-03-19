# Grasshopper Optimization for Energy Load Balancing in Smart Grid

## Problem Statement

Modern smart grids face the challenge of **peak load demand**, which increases
electricity costs, stresses infrastructure, and reduces grid stability.
This project uses **Machine Learning (Random Forest)** to forecast energy load
and applies the **Grasshopper Optimization Algorithm (GOA)** to produce an
optimised load schedule that reduces peak demand, cost, and variance.

---

## Project Architecture

```
Raw Dataset (CSV)
      │
      ▼
┌─────────────────┐
│  Preprocessing  │  ← handle_missing, extract_time_features, normalize
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Random Forest   │  ← train, evaluate (RMSE / MAE / R²), save .pkl
│  Forecasting    │
└────────┬────────┘
         │  predicted load
         ▼
┌─────────────────┐
│      GOA        │  ← swarm optimises load schedule
│  Optimisation   │     Fitness = w1·Peak + w2·Cost + w3·Variance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │  ← Peak Load, Cost, PAR, Variance (before vs after)
└────────┬────────┘
         │
         ▼
   results/*.png   +   models/*.pkl
```

---

## Folder Structure

```
SmartGrid-GOA-LoadBalancing/
├── dataset/
│   └── smartgrid.csv               ← synthetic hourly smart-grid data
├── models/
│   └── load_forecast_model.pkl     ← trained Random Forest model
├── notebooks/
│   └── smartgrid_load_prediction.ipynb
├── results/
│   ├── before_optimization.png
│   ├── after_optimization.png
│   └── cost_comparison.png
├── src/
│   ├── preprocessing.py
│   ├── forecasting_model.py
│   ├── goa_optimization.py
│   └── evaluation.py
├── generate_dataset.py             ← run once to create the CSV
├── README.md
└── requirements.txt
```

---

## Installation

```bash
# 1. Clone / download the project
cd SmartGrid-GOA-LoadBalancing

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Option A — Jupyter Notebook (recommended)

```bash
jupyter notebook notebooks/smartgrid_load_prediction.ipynb
```

Run all cells top-to-bottom. The notebook:
1. Generates the dataset (if missing)
2. Preprocesses data
3. Trains and evaluates the Random Forest model
4. Runs GOA optimisation
5. Saves all result plots to `results/`

### Option B — Individual scripts (VS Code / terminal)

```bash
# Step 1 – generate dataset
python generate_dataset.py

# Step 2 – preprocess
python src/preprocessing.py

# Step 3 – train model
python src/forecasting_model.py

# Step 4 – run GOA
python src/goa_optimization.py

# Step 5 – evaluate
python src/evaluation.py
```

---

## Dataset Description

| Column      | Description                        | Unit   |
|-------------|------------------------------------|--------|
| datetime    | Hourly timestamp                   | –      |
| load        | Energy consumption                 | kWh    |
| price       | Electricity price                  | $/kWh  |
| temperature | Ambient temperature                | °C     |

1 000 hourly records (≈ 42 days) with realistic daily and seasonal patterns.

---

## Results

| Metric     | Before GOA | After GOA | Improvement |
|------------|-----------|-----------|-------------|
| Peak Load  | ~380 kWh  | ~310 kWh  | ↓ ~18 %     |
| Total Cost | ~$42      | ~$35      | ↓ ~17 %     |
| PAR        | ~1.65     | ~1.35     | ↓ ~18 %     |
| Variance   | ~3 200    | ~2 100    | ↓ ~34 %     |

> Exact numbers depend on the random seed and GOA hyper-parameters.

### Visualisations

- **before_optimization.png** – original predicted load curve
- **after_optimization.png** – GOA-optimised load curve
- **cost_comparison.png** – bar chart comparing cost before vs after

---

## Key Technologies

| Component        | Library / Algorithm          |
|------------------|------------------------------|
| Data handling    | pandas, numpy                |
| ML model         | scikit-learn RandomForest    |
| Optimisation     | GOA (custom implementation)  |
| Visualisation    | matplotlib                   |
| Model persistence| joblib                       |

---

## References

- Saremi, S., Mirjalili, S., & Lewis, A. (2017). *Grasshopper Optimisation
  Algorithm: Theory and Application.* Advances in Engineering Software, 105, 30-47.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.
