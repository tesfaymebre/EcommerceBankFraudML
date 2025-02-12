# ⚙️ Scripts for EcommerceBankFraudML

This folder contains Python scripts for data preprocessing, feature engineering, and visualization.

## 📌 Scripts Overview

### **1. `data_cleaning.py`**
- Handles **missing values**, **outlier detection**, and **data type conversions**.
- Cleans raw data before analysis.

### **2. `eda.py`**
- Performs **Exploratory Data Analysis (EDA)** on fraud and transaction datasets.
- Includes **summary statistics, duplicate checks, and outlier detection**.

### **3. `data_visualizer.py`**
- Generates **visualizations** for better data understanding.
- Includes plots for **numerical distributions, categorical trends, fraud rates, and geolocation analysis**.

### **4. `feature_engineering.py`**
- Creates **derived features** such as:
  - **Time-based features** (`hour_of_day`, `day_of_week`).
  - **Geolocation features** (`high_risk_country` flag).
  - **Normalization and categorical encoding**.

### **5. `geolocation_analysis.py`**
- Merges `Fraud_Data.csv` with `IpAddress_to_Country.csv`.
- Converts **IP addresses to integers** for matching.
- Identifies **fraud patterns based on geography**.

## 📌 How to Use
1. **Preprocess data** → Run `data_cleaning.py`.
2. **Explore & visualize** → Use `eda.py` and `data_visualizer.py`.
3. **Engineer new features** → Run `feature_engineering.py`.
4. **Merge geolocation data** → Execute `geolocation_analysis.py`.

---

✅ **Processed Dataset:** `../data/preprocessed/Final_Fraud_Data.csv`  
✅ **Notebooks Location:** `../notebooks/`
