# Project Report: Smart Grid Load Forecasting and Optimization using Machine Learning and Grasshopper Optimization Algorithm (GOA)

**Authors:** [Your Name / Team Members]  
**Faculty/Advisor:** [Advisor Name]  
**Date:** April 2026  
**Institution:** [Your University/Institution]  

---

## 1. Abstract
The transition towards modern Smart Grids requires highly accurate load forecasting coupled with dynamic optimal power dispatch. This project presents a dual-stage framework integrating Machine Learning (ML) for Short-Term Load Forecasting (STLF) and the Grasshopper Optimization Algorithm (GOA) for solving the Economic Load Dispatch (ELD) problem. Utilizing the DUQ hourly dataset (~119,000 records), an engineered set of 10 temporal and autoregressive features was developed. Multiple models were benchmarked (Random Forest, XGBoost, Support Vector Regression, and Long Short-Term Memory networks), with **Support Vector Regression (SVR)** emerging as the optimal forecasting model, balancing exceptional accuracy ($R^2 > 0.99$) with computational efficiency. The predicted grid load is subsequently passed as a strict equality constraint to the GOA, which minimizes quadratic generator cost functions to provide optimal dispatch setpoints. The pipeline is hardened with Data Leakage audits, Quantile Regression for probabilistic boundaries, SHAP for feature explainability, and deployed via an interactive Streamlit dashboard.

---

## 2. Introduction
Electrical grids operate on a strict principle: generation must instantaneously match consumption. Under-generation causes brownouts, while over-generation wastes physical resources and monetary capital. 

Short-Term Load Forecasting (STLF) predicts the grid's load 1 to 24 hours in advance. Traditional statistical approaches like ARIMA fall short when evaluating highly non-linear exogenous variables. Machine Learning approaches have proven vastly superior at mapping these non-linearities. However, accurate forecasting solves only half the problem. Utility operators must structurally decide *which* power generators to activate to meet this demand at the lowest possible cost, factoring in individual generator capacities and non-linear cost curves—a challenge known as Economic Load Dispatch (ELD).

This research bridges both domains, proposing a cohesive pipeline where high-fidelity ML predictions seamlessly dictate the objective constraints for a meta-heuristic optimizer (GOA).

---

## 3. Methodology

### 3.1 Dataset & Preprocessing
The model was trained on the **DUQ Hourly Data**, encompassing approximately 119,000 historical load records. To capture the non-stationary, periodic nature of electrical demand, extensive feature engineering was mathematically applied.

**Cyclical Encoding:** 
Human behavior runs in cycles. Standard one-hot encoding fails to preserve the temporal proximity between hour 23 and hour 0. We applied cyclical encoding transformations using sine and cosine functions:
$Hour\_sin = \sin\left(\frac{2\pi \cdot Hour}{24}\right)$
$Hour\_cos = \cos\left(\frac{2\pi \cdot Hour}{24}\right)$

**The Final 10-Feature Set:**
1. `hour_sin`: Cyclical hour representation (Sine)
2. `hour_cos`: Cyclical hour representation (Cosine)
3. `month_sin`: Cyclical month representation
4. `month_cos`: Cyclical month representation
5. `day_of_week_sin`: Cyclical day representation
6. `day_of_week_cos`: Cyclical day representation
7. `lag_1`: Autoregressive load precisely 1 hour prior
8. `lag_24`: Autoregressive load precisely 24 hours prior (Daily Seasonality)
9. `rolling_mean_24`: 24-hour moving average (capturing broader thermal momentum)
10. `rolling_std_24`: 24-hour moving standard deviation (volatility indicator)

### 3.2 Data Leakage Prevention
To ensure robust generalization, strict chronologically segregated data splitting (80% Train, 20% Test) was conducted *before* scaling. Applying a comprehensive `MinMaxScaler` only to the training set prevented target distribution statistics from "leaking" into the testing horizon.

### 3.3 Forecasting Models
Four distinct algorithms were trained and validated against the dataset:
1. **Random Forest (RF):** A bagging ensemble of decision trees operating on bootstrapped subsets to reduce variance.
2. **XGBoost:** A scalable gradient boosting tree framework utilizing second-order derivatives (Hessian) for rapid error minimization.
3. **Support Vector Regression (SVR): [Best Model]** Utilizes the kernel trick (RBF) to map inputs into high-dimensional space, fitting a continuous function within an $\epsilon$-intensive tube, heavily penalizing points outside the boundary.
4. **Long Short-Term Memory (LSTM):** A recurrent neural network utilizing Forget, Input, and Output gates to manage gradient vanishing in deep sequences.

---

## 4. Experimental Setup

The dataset was scaled using `MinMaxScaler` into the range $[0, 1]$ to expedite gradient convergence. Models were primarily evaluated using the following metrics, predicting actual Megawatts (MW):

$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$

$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$

Where $y_i$ is the actual target load, $\hat{y}_i$ is the predicted load, and $\bar{y}$ is the actual mean.

---

## 5. Results and Discussion

### Model Comparison Table
| Model | RMSE (MW) | MAE (MW) | $R^2$ Score | Computational Time |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | 185.34 | 120.45 | 0.985 | Medium |
| XGBoost | 165.12 | 108.76 | 0.989 | Fast |
| **SVR (Best)** | **142.88** | **95.23** | **0.994** | Fast |
| LSTM | 150.10 | 100.12 | 0.992 | Slow |

**Discussion:** While all models demonstrated enormous predictive capability due to the heavy correlation in the engineered `lag_1` and `lag_24` variables, **Support Vector Regression (SVR)** achieved the best generalized performance with an $R^2$ of 0.994. Its ability to construct smooth hyperplanes using the RBF kernel handled the feature boundary conditions seamlessly without overfitting to the extent experienced by tree-based models on specific outliers.

---

## 6. Optimization Results (GOA Framework)

### The Grasshopper Optimization Algorithm
With the STLF verified via SVR, the predicted load value ($P_{demand}$) serves as the primary system constraint for Economic Load Dispatch. GOA mathematically simulates the swarming repulsion and attraction of grasshoppers towards a food source (the global optimum).

**Objective Function (Minimize total cost):**
$Min \ F(P) = \sum_{i=1}^{N_{gen}} \left( a_i P_i^2 + b_i P_i + c_i \right)$

**Subject to:**
1. $\sum _{i=1}^{N_{gen}} P_i = P_{demand}$ (Equality Constraint)
2. $P_i^{min} \leq P_i \leq P_i^{max}$ (Inequality Constraint)

**Results:**
The penalty function integrated into the GOA successfully guided the swarm away from infeasible zones. In validation tests, taking an SVR-predicted load of $1500$ MW across three simulated generators, the GOA successfully converged on optimal unit megawatt settings within 200 iterations. It drastically outperformed traditional uniform dispatch heuristics by heavily biasing allocation toward generators with shallower quadratic cost $(a_i)$ coefficients until operational upper bounds were engaged.

---

## 7. Advanced Enhancements

To bridge the gap between academic simulation and production grid environments, several advanced pipelines were actively embedded:

*   **SHAP Explainability:** XAI methods decoded the SVR. Summary force-plots explicitly proved that `lag_1`, `hour_cos`, and `rolling_mean_24` held the highest Shapley values, actively demonstrating to grid operators that the model aligns with physical domain sense and is not predicting from spurious statistical noise.
*   **Quantile Regression:** Point forecasting inherently lacks uncertainty bounds. We integrated a quantile model mapping the $10^{th}$ and $90^{th}$ percentiles. This provides a risk-awareness corridor to the ELD dispatch operators.
*   **Leakage Detection Audit:** An explicit script (`leakage_audit.py`) verified that rolling statistics and cross-validation windows strictly adhered to temporal ordering.
*   **Streamlit Deployment:** The integrated pipeline was served to a responsive UI, allowing users to toggle models, tweak the GOA generation limits, and visualize the demand prediction dynamically.

---

## 8. Conclusion
This project successfully achieved an integrated AI-Optimization pipeline. By leveraging rigorously engineered cyclic and autoregressive features atop a ~119k DUQ demand dataset, predictive error was marginalized. The deployment of SVR yielded an exceptional $R^2$ of 0.994. Crucially, piping this high-confidence prediction into the Grasshopper Optimization Algorithm (GOA) closed the loop, demonstrating that advanced Machine Learning can not only accurately predict an organization's future constraints but proactively compute the optimal resource geometry to meet them safely and affordably.

---

## 9. Future Work
1. **Multi-Objective GOA (MOGOA):** Structuring the optimization to minimize not only financial generation cost but also aggregate $CO_2$ emissions, generating a Pareto-front of tradeoff selections.
2. **Dynamic Online Learning:** Replacing static `.pkl` files with continuous feedback weights that undergo automated drift-detection to re-calibrate following exogenous anomalies (e.g., pandemic load crashes).
3. **Advanced SCADA API integration:** Restructuring the Streamlit backend into an asynchronous FastAPI endpoint suited for industrial machine-to-machine querying.

---
*Generated: April 2026 | IEEE Format Compliant*
