# AirSight AI — AI-Powered Air Quality Analytics & Pollution Prediction System

**BharatCares – AICTE Internship Project in Collaboration with IBM**  
*Track: Data Analytics, Environmental Informatics & Applied Machine Learning*

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00.svg)](https://colab.research.google.com/)

---

## 1. Project Overview & Executive Summary

**AirSight AI** is a production-grade, end-to-end environmental data analytics and machine learning system engineered to analyze historical ambient air pollution dynamics, discover spatial and seasonal patterns across Indian metropolitan centers, and forecast the **24-Hour Ahead National Air Quality Index (AQI $t+24$)**.

The system addresses public health vulnerabilities across urban India by transforming raw ambient sensor recordings into actionable intelligence, interactive visual dashboards, and scientifically grounded predictive horizons without data leakage.

```
       ┌────────────────────────────────────────────────────────┐
       │     Central Pollution Control Board (CPCB) Data        │
       │        (26 Indian Cities, 2015 - 2020 Records)         │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     Automated Data Ingestion & Physical Validation     │
       │    (Negatives Capped, Outlier Diagnostics Preserved)   │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │   Zero-Leakage Temporal Feature Engineering (t <= now) │
       │ (Calendar Cycles, Historical Lags, Past Rolling Means) │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │      Strict Chronological Split (Train/Val/Test)       │
       │   Train (70% <= 2018) | Val (15% 2019) | Test (15% 2020)│
       └───────────────────────────┬────────────────────────────┘
                                   │
             ┌─────────────────────┼─────────────────────┐
             ▼                     ▼                     ▼
     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
     │  Persistence  │     │    Linear     │     │ Random Forest │
     │   Baseline    │     │  Regression   │     │   & XGBoost   │
     └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
             └─────────────────────┼─────────────────────┘
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │   Model Selection (Validation RMSE) & Test Evaluation  │
       │ Champion: Random Forest Regressor (Test R²: 0.855)     │
       └───────────────────────────┬────────────────────────────┘
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│  AirSight AI Web Dashboard   │        │ Google Colab Research Study  │
│   (5-Page Streamlit App)     │        │ (Air_Quality_Analytics_AI)   │
└──────────────────────────────┘        └──────────────────────────────┘
```

---

## 2. Problem Statement & Research Questions

Air pollution in Indian cities fluctuates with extreme non-linearity driven by seasonal meteorology (e.g., winter atmospheric inversions, monsoon precipitation washout) and anthropogenic source spikes (e.g., post-harvest stubble burning, festival emissions, heavy vehicular density).

This project empirically answers seven key environmental research questions:
1. **Highest Pollution Cities**: Which metropolitan centers endure the highest historical average AQI?
2. **Seasonal Critical Periods**: Which months and seasons record the most dangerous pollution surges?
3. **Seasonal Divergence**: How does air quality differ between the monsoon washout and winter trapping periods?
4. **Chemical Pollutant Drivers**: Which specific chemical species ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$) exhibit the strongest correlation with overall AQI?
5. **Multi-Year Trajectory**: How has ambient air quality evolved across the 2015–2020 horizon?
6. **Recurrence of Spikes**: Are extreme pollution episodes recurring seasonal cycles or random events?
7. **Atmospheric Volatility**: Which urban centers suffer from the highest coefficient of variation (CV) in air quality?

---

## 3. Technology Stack

* **Language**: Python 3.10+ / 3.13
* **Interactive Web Platform**: Streamlit, Plotly Express & Graph Objects
* **Interactive Notebook**: Google Colab / Jupyter Notebook
* **Data Ingestion & Preprocessing**: Pandas, NumPy, SciPy
* **Statistical Analysis & Visualization**: Matplotlib, Seaborn, Plotly
* **Machine Learning**: Scikit-Learn, Random Forest Regressor, Linear Regression, XGBoost
* **Model Serialization & Pipeline**: Joblib, JSON
* **Version Control**: Git, GitHub

---

## 4. Repository Structure

```
air-quality-analytics-ai/
│
├── app.py                     # Streamlit multi-page web platform (AirSight AI)
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive technical documentation
├── .gitignore                 # Cache and artifact filters
│
├── notebooks/
│   ├── Air_Quality_Analytics_AI.ipynb  # Complete Google Colab notebook (18 sections)
│   └── generate_notebook.py            # Automated notebook compilation script
│
├── src/
│   ├── data_loader.py         # CPCB downloader, custom buffer parser & synthetic generator
│   ├── preprocessing.py       # Deduplication, physical bounds, imputation & outlier policy
│   ├── feature_engineering.py # Lags, cyclical calendar features, past rolling statistics
│   ├── eda.py                 # Core analytical functions answering 7 research questions
│   ├── train_model.py         # Chronological train/val/test ML pipeline & persistence
│   ├── predict.py             # 24-hr inference engine with 95% error bands & health advisory
│   └── insights.py            # Deterministic, non-hallucinated analytical insights generator
│
├── models/
│   ├── best_aqi_model.joblib  # Serialized champion model bundle & scaler
│   └── model_metadata.json    # Complete benchmark metrics, split dates & feature importances
│
├── data/
│   ├── README.md              # Dataset provenance, data dictionary & CPCB NAQI scales
│   ├── city_day.csv           # Real historical CPCB dataset (2015-2020, 26 cities)
│   └── synthetic_sample_air_quality.csv # Small synthetic dataset for smoke tests
│
├── reports/
│   └── project_report.md      # Full academic internship report (AICTE/IBM format)
│
└── assets/                    # Static visual assets
```

---

## 5. Dataset Provenance & Data Quality Policy

* **Source**: Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change (MoEFCC), Government of India. Standardized via Kaggle Indian Air Quality Data (2015–2020) by Rohan Rao.
* **Volume**: 29,531 daily records across 26 major metropolitan centers.
* **Attributes**: Date, City, $PM_{2.5}$, $PM_{10}$, $NO$, $NO_2$, $NO_x$, $NH_3$, $CO$, $SO_2$, $O_3$, Benzene, Toluene, Xylene, AQI, AQI_Bucket.
* **Outlier Policy**: Outliers are documented via IQR bounds but **retained**, because extreme episodic peaks (e.g., $AQI > 400$ during post-harvest agricultural burning and Diwali) are genuine environmental realities rather than sensor glitches.
* **Synthetic Data Disclaimer**: A small 180-row synthetic dataset (`data/synthetic_sample_air_quality.csv`) is provided strictly for offline pipeline testing and is explicitly labeled. Model evaluation and research conclusions are derived exclusively from the real CPCB dataset.

---

## 6. Machine Learning Methodology & Zero-Leakage Architecture

### A. Target Definition
$$\text{Target\_AQI\_t24} = \text{AQI}_{t+24\text{ hours}} = \text{AQI}(t+1\text{ day})$$
Derived via forward `shift(-1)` partitioned per city.

### B. Input Features (Strictly at Time $t$ or Earlier)
1. **Calendar & Cyclical Indicators**: Year, Month, Day, Day of Week, Is_Weekend, $\sin/\cos(\text{Month})$, $\sin/\cos(\text{DayOfWeek})$.
2. **Meteorological Season**: Winter, Summer, Monsoon, Post-Monsoon.
3. **Autoregressive Lags**: Current observed $\text{AQI}_t$, $\text{AQI}_{t-1}$, $\text{AQI}_{t-2}$, $\text{AQI}_{t-3}$, and weekly lag $\text{AQI}_{t-7}$.
4. **Pollutant Historical Lags**: $PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$ at time $t$ and $t-1$.
5. **Past Rolling Windows**: 7-day and 14-day rolling mean, rolling standard deviation (volatility), rolling maximum, and rolling minimum strictly using historical observations $[t-k, t]$.
6. **Chemical Domain Ratios**: $PM_{2.5}/PM_{10}$ ratio (combustion vs crustal dust) and $NO_2/SO_2$ ratio (vehicular vs industrial marker).

### C. Chronological Train-Validation-Test Splitting
* **Train Partition (70%)**: 2015-01-01 to 2018-11-05 (16,431 samples)
* **Validation Partition (15%)**: 2018-11-06 to 2019-09-02 (5,708 samples)
* **Test Partition (15%)**: 2019-09-03 to 2020-06-30 (7,366 samples)
* *Anti-Leakage Control*: Preprocessing scalers are fit **only** on the training partition. No temporal lookahead or random cross-validation shuffling is permitted.

---

## 7. Model Evaluation & Real CPCB Benchmark Results

All models were evaluated on the chronological Validation set to select the champion architecture, and final generalization performance was verified on the untouched Test set:

| Model | Val MAE | Val RMSE | Val $R^2$ Score | Train Time (s) | Evaluation Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **28.50** | **63.30** | **0.8349** | 16.16s | **Selected Champion** |
| **XGBoost Regressor** | 28.54 | 63.96 | 0.8315 | 1.46s | High Performance Candidate |
| **Linear Regression** | 28.56 | 64.61 | 0.8280 | 0.05s | Parametric Baseline |
| **Persistence Baseline** ($y_t$) | 32.65 | 79.25 | 0.7413 | 0.001s | Naive Benchmark |

### Final Untouched Test Set Performance (Champion: Random Forest)
* **Mean Absolute Error (MAE)**: **19.76 AQI points**
* **Root Mean Squared Error (RMSE)**: **40.22 AQI points**
* **Coefficient of Determination ($R^2$)**: **0.8546**

---

## 8. Verified Data-Driven AI Insights

All observations below are mathematically computed from the real dataset (zero hallucinations):
1. **Seasonal Inversion Surge**: Winter recorded an average AQI of **223.47**, representing a **92.8% elevation** compared to the Monsoon average of **115.93** ($p < 0.001$ via Mann-Whitney U test).
2. **Regional Urban Hotspot**: Ahmedabad recorded the highest historical average AQI (**433.60**), exceeding the national city median (**117.00**) by **270.6%** and **12.47x higher** than the lowest urban center (Aizawl: 34.78).
3. **Primary Chemical Drivers**: Carbon Monoxide ($r = 0.674$) and fine particulate matter $PM_{2.5}$ ($r = 0.619$) show the strongest linear correlation with the composite National AQI score.
4. **Multi-Year Trajectory**: Mean annual AQI decreased by **42.4%** between 2015 (196.78) and 2020 (113.40), attributable to monitoring network expansion and the Q2-2020 nationwide COVID-19 lockdown.
5. **Atmospheric Volatility**: Guwahati exhibited the highest relative variability with a Coefficient of Variation (CV) of **80.54%** (Standard Deviation: 112.28).

---

## 9. Installation & Local Execution

### Step 1: Clone or Navigate to the Repository
```bash
git clone https://github.com/your-username/air-quality-analytics-ai.git
cd air-quality-analytics-ai
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Machine Learning Training Pipeline
```bash
python src/train_model.py
```
*Outputs: Generates `models/best_aqi_model.joblib` and `models/model_metadata.json`.*

### Step 5: Launch Interactive Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view **AirSight AI**.

---

## 10. Google Colab Instructions

To execute the project in Google Colab:
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `notebooks/Air_Quality_Analytics_AI.ipynb`.
3. Set runtime hardware accelerator: **Runtime > Change runtime type > Python 3**.
4. Run all cells sequentially (**Runtime > Run all** or `Ctrl + F9`).
5. The notebook automatically downloads the official CPCB dataset, performs preprocessing, displays visualizations, trains models, and prints insights.

---

## 11. Ethical Notice & Limitations

* **Station vs. Microclimate Resolution**: Ambient monitoring records represent regional background concentrations and cannot account for localized street-canyon traffic dynamics.
* **Exogenous Weather Shocks**: Unseasonal rainfall, sudden boundary-layer temperature inversions, or unannounced biomass burning cannot be fully anticipated solely from historical lags without real-time numerical weather prediction (NWP) inputs.
* **Academic Disclaimer**: Predictions are produced by a student academic research model for educational evaluation under the BharatCares–AICTE program. They do **not** constitute official regulatory advisories issued by the Central Pollution Control Board (CPCB).
