# Customer Churn Analytics Dashboard

A machine learning-powered web application that predicts customer churn risk for a telecom company and provides actionable business insights through an interactive dashboard.

🔗 **Live Demo:** [ Local URL: http://localhost:8501,
  Network URL: http://192.168.0.108:8501]
📊 **Dataset:** 
[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📌 Overview

Customer churn — when a customer stops using a company's service — is one of the most critical metrics for subscription-based businesses. This project analyzes historical customer data to identify patterns behind churn and builds a predictive model to flag at-risk customers before they leave.

The final model is deployed as an interactive **Streamlit dashboard**, allowing business users to explore churn drivers and get risk predictions for individual customers.

---

## ✨ Features

- **Exploratory Data Analysis (EDA)** — visualizes churn patterns across demographics, contract types, tenure, and billing methods
- **Predictive Model** — Logistic Regression model trained to classify customers as likely to churn or stay
- **Three-Tier Risk Scoring** — customers are segmented into **Low, Medium, and High risk** categories based on predicted churn probability
- **Interactive Dashboard** — built with Streamlit for real-time exploration and predictions
- **Model Performance Metrics** — accuracy, ROC-AUC, confusion matrix, and feature importance visualizations

---

## 🧠 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 80.8% |
| ROC-AUC | 0.842 |

The model was trained using **Logistic Regression** on the **Telco Customer Churn dataset** (7,043 customers), with preprocessing steps including encoding categorical variables, feature scaling, and handling class imbalance.

---

## 🗂️ Dataset

- **Source:** IBM/Telco Customer Churn dataset
- **Size:** 7,043 customer records
- **Features:** Demographics (gender, senior citizen status), account info (tenure, contract type, payment method), and services subscribed (internet, phone, streaming, etc.)
- **Target Variable:** `Churn` (Yes/No)

---

## 🛠️ Tech Stack

- **Language:** Python 3.13
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Logistic Regression, preprocessing pipelines)
- **Visualization:** Plotly, Matplotlib/Seaborn
- **Deployment:** Streamlit

---

## 📁 Project Structure

```
customer-churn-dashboard/
│
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── eda_and_model_training.ipynb
├── app.py                  # Streamlit dashboard
├── model.pkl                # Trained model
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/customer-churn-dashboard.git
   cd customer-churn-dashboard
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   python -m streamlit run app.py
   ```

---

## 📊 How It Works

1. Raw customer data is cleaned and preprocessed (handling missing values, encoding categorical features, scaling numerical features)
2. A Logistic Regression model is trained to predict churn probability
3. Customers are bucketed into **Low / Medium / High risk** tiers based on their predicted probability
4. Results are visualized in an interactive Streamlit dashboard for business decision-making

---

## 🔮 Future Improvements

- Experiment with ensemble models (Random Forest, XGBoost) for improved accuracy
- Add SHAP values for model explainability
- Integrate customer retention recommendations based on risk tier
- Deploy with a database backend for real-time data updates

---

## 👤 Author

**Vinod Kumar**
B.E. Computer Science & Engineering, Vemana Institute of Technology-Bengaluru

---

## 📄 License

This project is licensed under the MIT License.
