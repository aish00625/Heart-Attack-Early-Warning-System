# ❤️ Heart Attack Early Warning System (Explainable AI Dashboard)

An end-to-end **Explainable AI (XAI)** project that predicts the likelihood of heart disease using a **Random Forest classifier** and provides transparent model interpretations through **SHAP (Shapley Additive Explanations)**.  
The entire system is deployed as an interactive **Streamlit dashboard**.

---

## 🚀 Features

### ✔ Machine Learning  
- Random Forest model trained on Kaggle Heart Disease Dataset  
- Evaluation metrics: Accuracy, ROC-AUC, Confusion Matrix, ROC Curve  

### ✔ Explainable AI (XAI)  
- **Global SHAP**: Feature importance using mean absolute SHAP values  
- **Local SHAP**: Patient-level explanation (why this prediction?)  
- Static beeswarm-style visualization  

### ✔ Interactive Dashboard (Streamlit)
- Fixed top navigation bar  
- Dataset preview  
- EDA visualizations  
- Correlation heatmap  
- Model performance section  
- SHAP global + local interpretation  
- Patient input form for personalized risk prediction  

---

## 📂 Project Structure

HeartAttackProject/
│── app.py # Streamlit dashboard


│── heart.csv # Dataset


│── requirements.txt # Project dependencies


│── README.md # Documentation


│── screenshots/ # Optional: UI screenshots


│── report/ # Optional: PDF documentation


---

## 📊 Dataset

The project uses the **Heart Disease UCI dataset** (Kaggle version).  
It contains 14 clinical input features such as:

- Age  
- Sex  
- Chest Pain Type (cp)  
- Resting BP (trtbps)  
- Cholesterol (chol)  
- Fasting Blood Sugar (fbs)  
- Rest ECG  
- Max Heart Rate (thalachh)  
- Exercise Induced Angina (exng)  
- Oldpeak  
- Slope  
- Major vessels (caa)  
- Thal  
- **Output (target: 0 = no disease, 1 = disease)**  

---

## 🔍 Exploratory Data Analysis (EDA)

The dashboard visualizes:

- Age distribution  
- Cholesterol distribution  
- Correlation heatmap  
- Target balance  

EDA helps identify outliers, patterns, and clinically meaningful relationships.

---

## 🤖 Model Details

### **Algorithm:** RandomForestClassifier (200 trees)

### Why Random Forest?
- Handles non-linear relationships  
- Robust to outliers  
- Minimal preprocessing  
- Works well with SHAP  

### Evaluation Metrics
- **Accuracy**
- **ROC-AUC**
- **Confusion Matrix**
- **ROC Curve**

---

## 🧠 Explainability with SHAP

### ⭐ Global SHAP
Shows which features affect predictions most on average.

### ⭐ Local SHAP
Explains *why the model predicted risk for a specific patient*.

This is crucial in medical AI where trust and transparency are required.

---

## 🖥 Running the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt

###2. Run the Streamlit app
streamlit run app.py

3. The app will open at:
http://localhost:8501

<img width="1770" height="766" alt="Screenshot 2025-11-27 003621" src="https://github.com/user-attachments/assets/13d9c2ed-7870-4a44-8350-19c0fed15aff" />
<img width="1743" height="812" alt="Screenshot 2025-11-27 004016" src="https://github.com/user-attachments/assets/634d799d-8df7-4680-9f49-50ca046e63e7" />
<img width="1824" height="809" alt="Screenshot 2025-11-27 004535" src="https://github.com/user-attachments/assets/8a076247-a41c-4b3d-9d36-382af2a4cfdb" />
