# **EcommerceBankFraudML**

## **Project Overview**

EcommerceBankFraudML is a machine learning project aimed at improving fraud detection for e-commerce and bank credit transactions. This project utilizes advanced data preprocessing, feature engineering, and geolocation analysis to identify fraudulent activities effectively. The ultimate goal is to build, evaluate, and deploy robust fraud detection models.

---

## **Project Structure**

```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       ├── unittests.yml
├── .dvc/
├── data/
│   ├── raw/                # Raw datasets
│   ├── preprocessed/       # Cleaned and processed datasets
├── .gitignore
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── dvc.yml                 # DVC pipeline file
├── src/
│   ├── __init__.py
├── notebooks/
│   ├── __init__.py
│   ├── data_exploration.ipynb  # EDA notebook
├── tests/
│   ├── __init__.py
└── scripts/
    ├── eda.py              # Exploratory Data Analysis script
    ├── data_cleaning.py    # Data cleaning script
    ├── data_merging.py     # Dataset merging script
    ├── __init__.py
```

---

## **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/EcommerceBankFraudML.git
   cd EcommerceBankFraudML
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up DVC** (if not already configured):
   ```bash
   dvc init
   dvc pull
   ```

---

## **Usage**

### **Data Preprocessing**
1. **Exploratory Data Analysis (EDA)**:
   - Use the `eda.py` script or the `data_exploration.ipynb` notebook to analyze datasets.
   ```bash
   python scripts/eda.py
   ```

2. **Data Cleaning**:
   - Clean datasets using `data_cleaning.py`.
   ```bash
   python scripts/data_cleaning.py
   ```

3. **Dataset Merging**:
   - Merge datasets for geolocation analysis using `data_merging.py`.
   ```bash
   python scripts/data_merging.py
   ```

### **Pipeline Execution**
- Run the DVC pipeline:
   ```bash
   dvc repro
   ```

---

## **Key Features**
1. **Exploratory Data Analysis**:
   - Detailed dataset analysis, including summary statistics, missing value detection, and visualizations.
   
2. **Data Cleaning**:
   - Handles outliers, normalizes numerical features, encodes categorical variables, and creates engineered features.

3. **Geolocation Analysis**:
   - Merges e-commerce transaction data with IP address mappings for location-based fraud analysis.

---

## **Datasets**

### **1. Fraud_Data.csv**
- Includes transaction details such as `user_id`, `purchase_value`, `device_id`, and `ip_address`.

### **2. IpAddress_to_Country.csv**
- Maps IP address ranges to countries.

### **3. creditcard.csv**
- Contains credit card transaction details, with anonymized features and a `Class` column indicating fraud (1) or non-fraud (0).

---

## **Current Progress**
- **EDA**: Completed for all datasets.
- **Data Cleaning**: Outliers handled, numerical columns normalized, categorical columns encoded.
- **Dataset Merging**: `Fraud_Data.csv` successfully merged with `IpAddress_to_Country.csv`.

---

## **Next Steps**
1. Feature engineering for transaction frequency and velocity.
2. Handling class imbalance in `creditcard.csv`.
3. Building and evaluating machine learning models.

---