[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_small.svg)](https://alzheimer-diagnosis-system.streamlit.app/)
# 🧠 Alzheimer's Disease Diagnosis System

This project is an AI-powered system designed to predict and diagnose Alzheimer's Disease using clinical and lifestyle data. It compares multiple machine learning models to find the most accurate diagnostic tool.

## 🚀 Project Overview
The goal is to help healthcare providers identify potential Alzheimer's cases early by analyzing 35 different medical features including cognitive tests (MMSE), lifestyle habits, and medical history.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib.
* **Deployment:** Streamlit (Web Interface).

## 📊 Models Evaluated
We implemented and compared four major models:
1. **Random Forest** (Best Performing - Used in the Web App).
2. **Logistic Regression**.
3. **Decision Tree**.
4. **Naïve Bayes**.

## 📈 Key Findings
* **MMSE** and **Functional Assessment** scores were the most significant predictors of the disease.
* The system achieved an accuracy of approximately **95%** using the Random Forest model.

## 💻 How to Run
1. Clone the repository.
2. Install requirements: `pip install -r requirements.txt`.
3. Run the app: `streamlit run app.py`.

## 👥 Contributors
* [Khaled Mohamed]
* [Khaled Hamada]
