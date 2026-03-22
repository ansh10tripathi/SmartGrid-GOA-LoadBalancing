# ⚡ Grasshopper Optimization for Energy Load Balancing in Smart Grid

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Optimization](https://img.shields.io/badge/Optimization-GOA-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![GitHub repo size](https://img.shields.io/github/repo-size/ansh10tripathi/SmartGrid-GOA-LoadBalancing)
![GitHub last commit](https://img.shields.io/github/last-commit/ansh10tripathi/SmartGrid-GOA-LoadBalancing)
![GitHub stars](https://img.shields.io/github/stars/ansh10tripathi/SmartGrid-GOA-LoadBalancing?style=social)

## 📌 Overview

Modern smart grids face critical challenges due to **fluctuating electricity demand**, especially during peak hours. This leads to:

* Increased operational costs
* Grid instability
* Higher transmission losses

This project presents an intelligent solution that combines:

* 🤖 **Machine Learning (Random Forest)** for load forecasting
* 🦗 **Grasshopper Optimization Algorithm (GOA)** for load balancing

The system predicts future energy demand and optimizes load distribution to **reduce peak load, cost, and variance**, ensuring efficient smart grid operation.

---

## 🎯 Objectives

* Forecast energy load using Machine Learning
* Optimize load distribution using GOA
* Reduce peak demand and electricity cost
* Improve grid stability and efficiency
* Evaluate performance using key metrics (RMSE, MAE, R², PAR)

---

## 🏗️ System Architecture

```
Raw Dataset (CSV)
      │
      ▼
┌────────────────────────┐
│   Data Preprocessing   │
│  - Missing handling    │
│  - Feature extraction  │
│  - Normalization       │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│  ML Load Forecasting   │
│  (Random Forest Model) │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│   GOA Optimization     │
│  (Load Scheduling)     │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│   Evaluation Metrics   │
│ (PAR, Cost, Variance)  │
└──────────┬─────────────┘
           │
           ▼
     Results & Graphs
```

---

## 📁 Project Structure

```
SmartGrid-GOA-LoadBalancing/
├── dataset/
│   └── smartgrid.csv
├── models/
│   └── load_forecast_model.pkl
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
├── generate_dataset.py
├── main.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/SmartGrid-GOA-LoadBalancing.git
cd SmartGrid-GOA-LoadBalancing

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🟢 Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/smartgrid_load_prediction.ipynb
```

Run all cells sequentially to:

* Load and preprocess data
* Train ML model
* Evaluate performance
* Apply GOA optimization
* Visualize results

---

### 🔵 Option 2: Full Pipeline (main.py)

```bash
python main.py
```

---

### 🟡 Option 3: Step-by-Step Execution

```bash
python generate_dataset.py
python src/preprocessing.py
python src/forecasting_model.py
python src/goa_optimization.py
python src/evaluation.py
```

---

## 📊 Dataset Description

| Feature     | Description               | Unit  |
| ----------- | ------------------------- | ----- |
| datetime    | Timestamp (hourly)        | —     |
| load        | Energy consumption        | kWh   |
| price       | Electricity price         | $/kWh |
| temperature | Environmental temperature | °C    |

* ~1000 records
* Includes daily and seasonal variations

---

## 📈 Model Performance

| Metric | Value        |
| ------ | ------------ |
| RMSE   | ~17–18 kWh   |
| MAE    | ~14 kWh      |
| R²     | ~0.84 – 0.89 |

> R² score indicates that the model explains ~85–90% of variance in energy load.

---

## ⚡ Optimization Results (GOA)

| Metric     | Before GOA | After GOA | Improvement |
| ---------- | ---------- | --------- | ----------- |
| Peak Load  | ~380 kWh   | ~310 kWh  | ↓ ~18%      |
| Total Cost | ~$42       | ~$35      | ↓ ~17%      |
| PAR        | ~1.65      | ~1.35     | ↓ ~18%      |
| Variance   | ~3200      | ~2100     | ↓ ~34%      |

---

## 📊 Visual Outputs

* 📉 Load curve before optimization
* 📈 Load curve after optimization
* 💰 Cost comparison chart

(All saved in `results/` directory)

---

## 🛠️ Tech Stack

| Component       | Technology                         |
| --------------- | ---------------------------------- |
| Programming     | Python                             |
| Data Processing | pandas, numpy                      |
| ML Model        | Random Forest (scikit-learn)       |
| Optimization    | Grasshopper Optimization Algorithm |
| Visualization   | matplotlib                         |
| Model Storage   | joblib                             |

---

## 🔍 Key Features

* Modular project structure
* End-to-end ML + optimization pipeline
* Realistic synthetic dataset generation
* Advanced evaluation metrics
* Visual performance comparison

---

## ⚠️ Limitations

* Synthetic dataset (not real-time grid data)
* GOA performance depends on parameter tuning
* No real-time IoT integration

---

## 🚀 Future Scope

* Integration with real-time smart meters
* Use of deep learning models (LSTM, XGBoost)
* Hybrid optimization techniques (GOA + PSO)
* Web-based dashboard deployment

---

## 📚 References

* Saremi et al. (2017) – Grasshopper Optimization Algorithm
* Breiman (2001) – Random Forests

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Ansh Tripathi**
B.Tech CSE (AI/ML)
Lovely Professional University

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share your feedback!
