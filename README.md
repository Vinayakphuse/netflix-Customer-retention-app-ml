# 🎬 Netflix Customer Churn Prediction System

An end-to-end Machine Learning project that predicts Netflix customer churn using advanced ML models and explainable AI techniques. The project includes a complete ML pipeline with data ingestion, feature engineering, model training, SHAP explainability, FastAPI deployment, and an interactive Streamlit dashboard.

---

# 🚀 Project Overview

<img width="1919" height="935" alt="Screenshot 2026-05-20 161625" src="https://github.com/user-attachments/assets/a9b40bd9-5fa2-4044-844c-6059c69b87e0" />


Customer churn is one of the biggest challenges for subscription-based platforms like Netflix. This project helps predict which customers are likely to cancel their subscription so businesses can take proactive retention actions.

The system uses historical customer behavior data to generate churn predictions and provide explainable insights using SHAP.

---

<img width="3106" height="4096" alt="IMG_20260520_170735" src="https://github.com/user-attachments/assets/2fb2bd70-704f-473d-9865-fe0400b8e2e7" />

<img width="3106" height="4096" alt="IMG_20260520_170812" src="https://github.com/user-attachments/assets/b75b5f84-8eff-457e-8e14-2a62c5156cf8" />




# ✨ Features

- End-to-End ML Pipeline
- Customer Churn Prediction
- Data Cleaning & Preprocessing
- Feature Engineering
- XGBoost Model Training
- SHAP Explainability
- FastAPI Model Serving
- Streamlit Interactive Dashboard
- PostgreSQL Database Integration
- Real-Time Prediction API
- Business KPI Visualization

---

# 🛠️ Tech Stack

## Languages & Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP

## Backend & Deployment
- FastAPI
- Uvicorn

## Frontend
- Streamlit

## Database
- PostgreSQL

## Visualization
- Matplotlib
- Seaborn
- Plotly

---

# 📂 Project Structure

```bash
Netflix-Churn-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── models/
│
├── api/
│   └── app.py
│
├── streamlit_app/
│   └── app.py
│
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── explainability.py
│
├── requirements.txt
├── README.md
└── main.py
```

---

# ⚙️ ML Pipeline

## 1️⃣ Data Ingestion
- Load Netflix customer dataset
- Read data from CSV/PostgreSQL
- Validate and clean raw data

## 2️⃣ Data Preprocessing
- Handle missing values
- Encode categorical features
- Scale numerical data

## 3️⃣ Feature Engineering
- Create customer engagement features
- Generate behavioral metrics
- Select important features

## 4️⃣ Model Training
- Train XGBoost classifier
- Hyperparameter tuning
- Cross-validation

## 5️⃣ Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score

## 6️⃣ Explainable AI
- SHAP summary plots
- Feature importance visualization
- Individual prediction explanations

## 7️⃣ Deployment
- FastAPI REST API
- Streamlit interactive dashboard

---

# 📊 Dashboard Features

- Upload customer data
- View churn predictions
- Real-time analytics
- SHAP explanation graphs
- Customer risk analysis
- KPI monitoring dashboard

---

# 📈 Business Impact

This project helps businesses:

- Reduce customer churn
- Improve customer retention
- Increase revenue
- Understand customer behavior
- Build data-driven retention strategies

---

# 🔥 Model Explainability with SHAP

SHAP values are used to explain:
- Why a customer may churn
- Which features influence predictions
- Customer risk factors

Example:
> High monthly charges and low engagement increased churn probability.

---

# 🚀 FastAPI Endpoints

## Predict Churn

```python
POST /predict
```

### Example Request

```json
{
  "MonthlyCharges": 89.5,
  "Tenure": 12,
  "SubscriptionType": "Premium"
}
```

### Example Response

```json
{
  "prediction": "Churn",
  "probability": 0.91
}
```

---

# ▶️ Run Locally

## Clone Repository

```bash
git clone https://github.com/your-username/netflix-churn-prediction.git
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run FastAPI Server

```bash
uvicorn api.app:app --reload
```

## Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

---

# 📌 Future Improvements

- Docker Deployment
- AWS/GCP Deployment
- CI/CD Integration
- Real-Time Streaming Predictions
- Deep Learning Models
- Automated Retraining Pipeline

---

# 👨‍💻 Author

## Vinayak Anil Phuse

Aspiring Data Scientist | Machine Learning Engineer | Data Analyst

---

# ⭐ If you like this project, give it a star on GitHub!
