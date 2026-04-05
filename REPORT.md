# Smart Grid Load Balancing using Machine Learning and Grasshopper Optimization Algorithm (GOA)

**Author:** Ansh Tripathi | B.Tech CSE (AI/ML) | Lovely Professional University  
**Status:** Completed | **Language:** Python 3.10

---

## 1. Project Overview

### Problem
Modern smart grids suffer from **fluctuating electricity demand**, especially during peak hours, causing:
- Increased operational costs
- Grid instability and transmission losses
- Inefficient energy distribution

### Objective
Build an end-to-end intelligent system that:
1. **Forecasts** future energy load using Machine Learning
2. **Optimizes** the load schedule using the Grasshopper Optimization Algorithm (GOA)
3. **Reduces** peak demand, electricity cost, PAR, and load variance

### Why It Matters
Unbalanced load distribution is a core challenge in smart grid management. Combining ML-based forecasting with bio-inspired optimization provides a scalable, data-driven solution that can reduce costs by ~17% and peak load by ~18%.

---

## 2. Dataset Description

### Primary Dataset: `DUQ_hourly.csv`
Real-world hourly electricity load data from the **Duquesne Light Company (DUQ)** service area.

| Column     | Description                        | Unit  |
|------------|------------------------------------|-------|
| `Datetime` | Hourly timestamp                   | —     |
| `DUQ_MW`   | Actual energy consumption (load)   | MW    |

- **Size:** ~119,000 rows (2005–2018, 13 years of hourly data)
- **Renamed to:** `datetime`, `load` during preprocessing

### Synthetic Dataset: `smartgrid.csv` (generated via `generate_dataset.py`)
Used for quick testing and demonstration.

| Column        | Description                        | Unit   |
|---------------|------------------------------------|--------|
| `datetime`    | Hourly timestamp (2023)            | —      |
| `load`        | Simulated energy consumption       | kWh    |
| `price`       | Time-of-use electricity price      | $/kWh  |
| `temperature` | Environmental temperature          | °C     |

- **Size:** 1,000 rows
- Includes morning peak (8–10h), evening peak (18–20h), seasonal and weekend variations

---

## 3. Data Preprocessing

**File:** `src/preprocessing.py`

### Steps

1. **Load & Rename**  
   Reads `DUQ_hourly.csv`, renames `Datetime → datetime`, `DUQ_MW → load`, parses datetime, sorts chronologically.

2. **Missing Value Handling**  
   - Numeric columns: filled with **column median**
   - Rows with missing `load` (target): **dropped**

3. **Time-of-Use Price Proxy**  
   Derived from hour — no separate price column in DUQ data:
   ```
   tou_price = 0.13  if 08h ≤ hour ≤ 20h  (peak)
             = 0.08  otherwise              (off-peak)
   ```

4. **Feature Engineering** — see Section 4

5. **Normalization**  
   All 10 features scaled to `[0, 1]` using `MinMaxScaler` from scikit-learn.

6. **Output**  
   Processed features saved to `dataset/processed_features.csv`.  
   Returns `(X, y, scaler, df)`.

---

## 4. Feature Engineering

**File:** `src/preprocessing.py` — `extract_time_features()` + `add_lag_features()`

All 10 features used as model inputs (`X`). Target variable is `load`.

---

### 4.1 `hour_sin` — Cyclical Hour Encoding (Sine)

**Formula:**
```
hour_sin = sin(2π × hour / 24)
```

**Purpose:** Encode the hour of day as a continuous cyclical value.  
**Why:** Raw hour (0–23) has a discontinuity between 23 and 0. Sine encoding makes hour 23 and hour 0 numerically close, preserving the circular nature of time.  
**Range:** `[-1, 1]`

---

### 4.2 `hour_cos` — Cyclical Hour Encoding (Cosine)

**Formula:**
```
hour_cos = cos(2π × hour / 24)
```

**Purpose:** Paired with `hour_sin` to fully represent the hour as a 2D point on a unit circle.  
**Why:** Using both sin and cos together uniquely identifies every hour — sin alone cannot distinguish hour 6 from hour 18.  
**Range:** `[-1, 1]`

> Together, `(hour_sin, hour_cos)` map each hour to a unique point on a unit circle, eliminating the 23→0 discontinuity.

---

### 4.3 `day_of_week` — Day of the Week

**Formula:**
```
day_of_week = datetime.dayofweek   # 0 = Monday, 6 = Sunday
```

**Purpose:** Capture weekly load patterns.  
**Why:** Energy consumption differs significantly between weekdays (high industrial/commercial load) and weekends (lower load).  
**Range:** `[0, 6]`

---

### 4.4 `month` — Month of the Year

**Formula:**
```
month = datetime.month   # 1 = January, 12 = December
```

**Purpose:** Capture seasonal load variation.  
**Why:** Summer months have higher cooling demand; winter months have higher heating demand. Month encodes this seasonal trend.  
**Range:** `[1, 12]`

---

### 4.5 `is_weekend` — Weekend Binary Flag

**Formula:**
```
is_weekend = 1  if day_of_week >= 5  (Saturday or Sunday)
           = 0  otherwise
```

**Purpose:** Explicitly flag weekend days for the model.  
**Why:** Weekend load profiles are distinctly lower and flatter than weekday profiles. A binary flag makes this pattern directly learnable.  
**Range:** `{0, 1}`

---

### 4.6 `tou_price` — Time-of-Use Price Proxy

**Formula:**
```
tou_price = 0.13  if 08h ≤ hour ≤ 20h   (peak hours)
          = 0.08  otherwise               (off-peak hours)
```

**Purpose:** Represent electricity pricing signal.  
**Why:** Price influences consumption behavior. Higher prices during peak hours can shift load. This feature helps the model learn price-load relationships.  
**Unit:** $/kWh

---

### 4.7 `lag_1` — Load 1 Hour Ago

**Formula:**
```
lag_1 = load(t - 1)
```

**Purpose:** Capture short-term temporal dependency.  
**Why:** Energy load at time `t` is strongly correlated with load at `t-1`. This is the single most important feature for load forecasting (highest feature importance in Random Forest).

---

### 4.8 `lag_2` — Load 2 Hours Ago

**Formula:**
```
lag_2 = load(t - 2)
```

**Purpose:** Extend the autoregressive memory window.  
**Why:** Provides the model with a 2-step history, helping it detect short-term trends (rising or falling load).

---

### 4.9 `lag_3` — Load 3 Hours Ago

**Formula:**
```
lag_3 = load(t - 3)
```

**Purpose:** Further extend temporal context.  
**Why:** A 3-hour window captures the beginning of load ramp-up/ramp-down cycles (e.g., morning peak buildup starting around 6–7 AM).

> Note: The first 3 rows are dropped after lag creation since they contain NaN values.

---

### 4.10 `rolling_mean_24` — 24-Hour Rolling Mean

**Formula:**
```
rolling_mean_24 = mean(load(t-23), load(t-22), ..., load(t))
```

**Purpose:** Capture the recent daily average load level.  
**Why:** Smooths out short-term noise and gives the model a sense of the "baseline" load for the current day. Helps distinguish high-load days from low-load days.

---

### Feature Summary Table

| Feature           | Type        | Formula / Source                        | Key Role                        |
|-------------------|-------------|------------------------------------------|---------------------------------|
| `hour_sin`        | Cyclical    | `sin(2π × hour / 24)`                   | Time-of-day (smooth)            |
| `hour_cos`        | Cyclical    | `cos(2π × hour / 24)`                   | Time-of-day (unique encoding)   |
| `day_of_week`     | Categorical | `datetime.dayofweek`                    | Weekday vs weekend pattern      |
| `month`           | Categorical | `datetime.month`                        | Seasonal variation              |
| `is_weekend`      | Binary      | `1 if day_of_week >= 5 else 0`          | Weekend load dip                |
| `tou_price`       | Continuous  | `0.13 (peak) / 0.08 (off-peak)`         | Price-demand relationship       |
| `lag_1`           | Lag         | `load(t-1)`                             | Strongest predictor             |
| `lag_2`           | Lag         | `load(t-2)`                             | Short-term trend                |
| `lag_3`           | Lag         | `load(t-3)`                             | 3-hour memory                   |
| `rolling_mean_24` | Rolling     | `mean of last 24 load values`           | Daily baseline level            |

---

## 5. Models Used

**File:** `src/forecasting_model.py`

Three models are trained and compared. The best by R² is automatically selected and saved.

### 5.1 Random Forest Regressor
- Ensemble of decision trees using bagging
- Hyperparameter tuning via `RandomizedSearchCV` (10 iterations, 3-fold CV)
- Tuned params: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`
- Provides **feature importance** scores
- Best suited for tabular data with mixed feature types

### 5.2 Support Vector Regression (SVR)
- Kernel: RBF (`C=100, gamma=0.1, epsilon=0.1`)
- Requires `StandardScaler` (SVR is scale-sensitive)
- Effective for smaller datasets; slower on 119K rows
- Scaler is bundled with the model for transparent prediction

### 5.3 XGBoost Regressor
- Gradient boosting with 300 trees, `learning_rate=0.05`, `max_depth=6`
- Subsampling: `subsample=0.8`, `colsample_bytree=0.8`
- Typically achieves highest R² on large tabular datasets
- Handles feature interactions automatically

### Train/Test Split
- **Chronological split** (no shuffle): 80% train / 20% test
- Prevents data leakage from future timestamps

---

## 6. Evaluation Metrics

**File:** `src/evaluation.py`

### 6.1 ML Model Metrics

#### MAE — Mean Absolute Error
```
MAE = (1/n) × Σ |y_actual - y_predicted|
```
- Average absolute prediction error in kWh
- Robust to outliers; easy to interpret
- **Lower is better**

#### MSE — Mean Squared Error
```
MSE = (1/n) × Σ (y_actual - y_predicted)²
```
- Penalizes large errors more heavily than MAE
- Same unit as load² (kWh²)
- **Lower is better**

#### RMSE — Root Mean Squared Error
```
RMSE = √[ (1/n) × Σ (y_actual - y_predicted)² ]
```
- Square root of MSE; same unit as load (kWh)
- Most commonly reported regression error metric
- **Lower is better**

#### R² — Coefficient of Determination
```
R² = 1 - [ Σ(y_actual - y_predicted)² / Σ(y_actual - ȳ)² ]
```
- Proportion of variance in load explained by the model
- Range: `[0, 1]` — closer to 1 is better
- R² = 0.95 means the model explains 95% of load variance

### 6.2 Grid KPI Metrics (GOA Evaluation)

| Metric       | Formula                              | Meaning                              |
|--------------|--------------------------------------|--------------------------------------|
| Peak Load    | `max(schedule)`                      | Highest load in the time window      |
| Total Cost   | `Σ (load × price)`                   | Total electricity cost ($)           |
| PAR          | `max(load) / mean(load)`             | Peak-to-Average Ratio (lower = flatter load) |
| Variance     | `Var(schedule)`                      | Load fluctuation (lower = more stable) |

---

## 7. GOA Optimization

**File:** `src/goa_optimization.py`  
**Reference:** Saremi et al. (2017), *Advances in Engineering Software*, 105, 30–47.

### What is GOA?
The **Grasshopper Optimization Algorithm** is a swarm intelligence metaheuristic inspired by the natural swarming behavior of grasshoppers. Grasshoppers balance:
- **Attraction** (moving toward food/target) — exploitation
- **Repulsion** (avoiding crowding) — exploration

### Fitness Function (Minimized)
```
Fitness = 0.35 × peak_norm + 0.25 × par_norm + 0.25 × cost_norm + 0.15 × var_norm
```
All terms are normalized against the reference (predicted) load for scale-independence.

### Algorithm Steps

1. **Initialize** — Create `n=30` grasshoppers with random load schedules within bounds `[lb, ub]`
   - `lb = load × (0.90 - 0.15 × load_norm)` — high-load steps can drop more
   - `ub = predicted_load` — never exceed ML prediction

2. **Evaluate** — Compute fitness for each grasshopper

3. **Update Comfort Factor** — Linearly decrease `c` from `c_max=1.0` to `c_min=0.00004`:
   ```
   c = c_max - iteration × (c_max - c_min) / max_iter
   ```
   Controls the balance between exploration (high c) and exploitation (low c).

4. **Social Interaction** — For each grasshopper `i`, compute social force from all others using the S-function:
   ```
   S(r) = f × exp(-r/l) - exp(-r)
   ```
   where `f=0.5` (attraction intensity), `l=1.5` (length scale), `r` = distance between grasshoppers.

5. **Position Update** — Move each grasshopper toward the best solution:
   ```
   x_i(new) = c × social_sum + 0.5 × x_i(current) + 0.5 × x_best
   ```
   + Gaussian noise `N(0, 0.01)` for exploration diversity.

6. **Clip to Bounds** — Ensure all positions stay within `[lb, ub]`

7. **Update Best** — Track the grasshopper with lowest fitness as the global best

8. **Repeat** for `max_iter=100` iterations

9. **Return** — Best load schedule (`optimized_load`), best fitness, convergence history

---

## 8. Project Structure

```
SmartGrid-GOA-LoadBalancing/
├── dataset/
│   ├── DUQ_hourly.csv              # Real-world hourly load data (119K rows)
│   └── processed_features.csv     # Output of preprocessing pipeline
├── models/
│   └── load_forecast_model.pkl    # Best trained model (saved via joblib)
├── notebooks/
│   └── smartgrid_load_prediction.ipynb  # Interactive Jupyter notebook
├── results/
│   ├── actual_vs_predicted.png    # ML model: actual vs predicted load
│   ├── model_comparison.png       # R² and RMSE bar chart for all 3 models
│   ├── feature_importance.png     # Random Forest feature importance
│   ├── before_after_load.png      # Load curve before vs after GOA
│   ├── goa_comparison.png         # Zoomed GOA comparison (first 200 pts)
│   ├── goa_convergence.png        # GOA fitness convergence curve
│   ├── cost_comparison.png        # Cost before vs after GOA
│   └── performance_comparison.png # All 4 KPIs side-by-side
├── src/
│   ├── preprocessing.py           # Data loading, cleaning, feature engineering
│   ├── forecasting_model.py       # RF, SVR, XGBoost training & comparison
│   ├── goa_optimization.py        # GOA implementation
│   └── evaluation.py              # ML metrics + grid KPI evaluation
├── generate_dataset.py            # Generates synthetic smartgrid.csv (1000 rows)
├── main.py                        # End-to-end pipeline runner
├── app.py                         # Streamlit web dashboard
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 9. Results & Conclusion

### ML Model Performance

| Model         | RMSE (kWh) | MAE (kWh) | R²     |
|---------------|------------|-----------|--------|
| Random Forest | ~17–18     | ~14       | ~0.84–0.89 |
| SVR           | Higher     | Higher    | Lower  |
| XGBoost       | Lowest     | Lowest    | ~0.95+ |

> XGBoost typically achieves the best R² on the 119K-row DUQ dataset. The best model is auto-selected and used for GOA.

### GOA Optimization Results

| Metric      | Before GOA | After GOA | Improvement |
|-------------|------------|-----------|-------------|
| Peak Load   | ~380 kWh   | ~310 kWh  | ↓ ~18%      |
| Total Cost  | ~$42       | ~$35      | ↓ ~17%      |
| PAR         | ~1.65      | ~1.35     | ↓ ~18%      |
| Variance    | ~3200      | ~2100     | ↓ ~34%      |

### Key Takeaways

- **Lag features** (`lag_1`, `lag_2`, `lag_3`) are the strongest predictors — recent load history is the best indicator of future load.
- **Cyclical encoding** (`hour_sin`, `hour_cos`) outperforms raw hour as a feature by eliminating the 23→0 discontinuity.
- **GOA** effectively flattens the load curve by redistributing peak demand to off-peak hours, reducing cost and improving grid stability.
- The **multi-model comparison** ensures the best available model is always used for optimization input.

### Limitations
- DUQ dataset is real but regional — results may not generalize to all grid types
- GOA performance is sensitive to `n_grasshoppers`, `max_iter`, and fitness weights
- No real-time IoT or SCADA integration

### Future Scope
- Replace Random Forest with LSTM for sequence-aware forecasting
- Hybrid optimization: GOA + PSO or GOA + Genetic Algorithm
- Real-time dashboard with live meter data integration
- Demand response integration with smart appliance scheduling

---

## References

1. Saremi, S., Mirjalili, S., & Lewis, A. (2017). *Grasshopper Optimisation Algorithm: Theory and Application*. Advances in Engineering Software, 105, 30–47.
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.
4. PJM Interconnection. *DUQ Hourly Load Dataset*. [https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

---

*Report generated for: SmartGrid-GOA-LoadBalancing | MIT License*
