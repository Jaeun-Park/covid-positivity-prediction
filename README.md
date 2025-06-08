# covid-positivity-prediction
 

This project explores spatial and temporal modeling techniques to predict county-level COVID-19 test positivity rates in the United States. We compare the performance of multiple models, including Ridge Regression, Spatial Lag 1 Regression, Multi-Layer Perceptron (MLP), and Random Forest, using survey-based behavioral and health data.

---

## Project Goals

- Predict county-level **COVID-19 positivity rates**
- Evaluate effectiveness of different models in capturing **spatio-temporal dynamics**
- Provide insights for **public health policy** and early intervention

---

## Dataset

- **Source**: COVID-19 Trends and Impact Survey (CTIS) by Carnegie Mellon University and Facebook
- **Time Period**: January 7 – February 12, 2021
- **Scope**: 25,626 county-level daily records from across the U.S.
- **Features**: 19 variables covering health behaviors, beliefs, and outcomes

---

## Models Compared

| Model                      | Highlights                                        |
|---------------------------|---------------------------------------------------|
| Ridge Regression          | Linear baseline with regularization               |
| Spatial Lag 1 Regression  | Captures spatial correlation within states        |
| Multi-Layer Perceptron    | Deep learning model for nonlinear relationships   |
| Random Forest             | Tree-based ensemble model with strong performance |

---

## Performance Summary

**Best Model (Test Set):**  
- **Random Forest**
  - R²: 0.998  
  - RMSE: 0.09  
  - MAE: 0.02  
  - Strengths: Nonlinear modeling, high accuracy, robust to noise

**Runner-Up:**  
- **Spatial Lag Regression** (R² = 0.999) — More interpretable, but assumes linearity

---

## Key Techniques

- **Spatial Modeling**: Used a row-normalized spatial weights matrix (same-state adjacency)
- **Feature Engineering**: Lag features, rolling averages, behavioral interaction terms
- **Imputation**: Compared moving average vs. KNN imputation
- **Evaluation**: 5-fold CV, RMSE, MAE, and R²

---

## File Structure
covid-positivity-prediction/
│
├── data/ # CSVs used in the analysis
├── notebooks/ # Jupyter notebooks per model
├── results/ # Charts, metrics, and outputs
├── README.md # Project documentation
├── requirements.txt # Python dependencies
└── .gitignore # Ignore data checkpoints, logs, etc.

---

## Policy Implications

- **Vaccination momentum** and **public activity** were strong predictors of both vaccine uptake and COVID spread.
- **Spatial spillovers** matter: counties influence each other across borders.
- Suggests **coordinated regional policies** and monitoring **lagged behavioral indicators**.

---

## Future Work

- Integrate real-time data streams
- Use **LSTM** or **Transformer** architectures for dynamic sequence modeling
- Expand spatial granularity and refine adjacency definitions

---

## Citation

COVID-19 Trends and Impact Survey (CTIS), Carnegie Mellon University and Facebook, 2021  

---

## 🛠 Built With

- Python, Pandas, NumPy
- scikit-learn, PyTorch
- statsmodels, linearmodels

