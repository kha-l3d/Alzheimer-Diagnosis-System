import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# 1. إعدادات الصفحة
st.set_page_config(page_title="Alzheimer Diagnosis System", layout="wide")

st.title("🧠 Alzheimer's Disease Diagnosis System")
st.markdown("---")

# 2. تحميل وتجهيز البيانات 
@st.cache_resource
def load_and_train():
    df = pd.read_csv("uncleaned_alzheimers_disease_data.csv")
    
    # التنظيفد
    df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True, errors='ignore')
    df.dropna(subset=['Diagnosis'], inplace=True)
    
    # تعبئة القيم المفقودة
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

model, scaler, feature_names = load_and_train()

# 3. القائمة الجانبية لإدخال بيانات المريض
st.sidebar.header("📋 Patient Data Entry")

def user_input_features():
    mapping = {"No": 0, "Yes": 1}
    inputs = {}
    # أهم الميزات اللي بتأثر في التشخيص 
    inputs['Age'] = st.sidebar.slider('Age', 60, 90, 75)
    inputs['MMSE'] = st.sidebar.slider('MMSE Score', 0, 30, 20)
    inputs['FunctionalAssessment'] = st.sidebar.slider('Functional Assessment', 0, 10, 5)
    inputs['MemoryComplaints'] = mapping[st.sidebar.selectbox('Memory Complaints', ["No", "Yes"])]
    inputs['BehavioralProblems'] = mapping[st.sidebar.selectbox('Behavioral Problems', ["No", "Yes"])]

    inputs['ADL'] = st.sidebar.slider('Activities of Daily Living (ADL)', 0, 10, 5)
    
    # باقي الأعمدة بقيم افتراضية (لتبسيط الـ UI)
    full_data = {col: 0 for col in feature_names}
    full_data.update(inputs)
    return pd.DataFrame([full_data])

input_df = user_input_features()

# 4. عرض النتائج والتشخيص
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Diagnosis Result")
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error("🚨 Potential Alzheimer's Disease Detected")
    else:
        st.success("✅ Patient is Healthy")
    
    st.write(f"**Confidence Level:** {np.max(probability)*100:.2f}%")

# 5. عرض الرسوم البيانية (Visuals)
with col2:
    st.subheader("📊 Model Insight")
    # رسم أهمية الميزاد
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(5)
    
    fig, ax = plt.subplots()
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis', ax=ax)
    plt.title("Top Factors Influencing Diagnosis")
    st.pyplot(fig)

