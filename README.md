# Heart Failure Prediction Using Artificial Neural Networks (ANN)

## 📖 Overview
This project aims to predict **heart failure** in patients using an **Artificial Neural Network (ANN)**. The dataset consists of **299 patients**, including various clinical factors that contribute to heart failure. 

By leveraging **machine learning** and **deep learning (ANNs)**, we aim to provide a reliable prediction model to assist in early diagnosis and intervention.

---

## 📊 Dataset Information
The dataset contains **13 features** including:

- **Demographics:** Age, Sex
- **Health Conditions:** Diabetes, Anaemia, High Blood Pressure, Smoking
- **Clinical Parameters:** Serum Sodium, Serum Creatinine, Platelets, Ejection Fraction
- **Other Medical Data:** Creatinine Phosphokinase, Time (Follow-up period)
- **Target Variable:** `DEATH_EVENT` (0 = Survived, 1 = Death)

---

## ⚙️ **Model Development Process**
### 1️⃣ Data Preprocessing  
✔ Handled missing values (if any)  
✔ Removed outliers using **IQR method**  
✔ Scaled selected numerical features using **MinMaxScaler**  

### 2️⃣ Handling Class Imbalance  
✔ Used **SMOTE** to balance the dataset (50% minority class)  

### 3️⃣ ANN Model Architecture  
- **Input Layer:** 16 Neurons (ReLU Activation)  
- **Hidden Layer:** 8 Neurons (ReLU Activation)  
- **Output Layer:** 1 Neuron (Sigmoid Activation for Binary Classification)  

### 4️⃣ Model Training  
✔ Optimizer: `Adam`  
✔ Loss Function: `Binary Crossentropy`  
✔ Evaluation Metrics: `Accuracy`  

### 5️⃣ Model Performance  
✔ Training Accuracy: **81.48%**  
✔ Test Accuracy: **78.33%**  

---

## 📌 **Insights & Recommendations (15 Marks)**

### 🔍 **Key Insights (10 Marks)**  
1️⃣ **Age & Ejection Fraction are Strong Predictors:**  
   - Older patients with **low ejection fraction (<30%)** had a **high risk of death**.  

2️⃣ **Serum Creatinine and Sodium Levels Matter:**  
   - **High serum creatinine (>2.0)** and **low serum sodium (<135)** increased the risk of heart failure.  

3️⃣ **Chronic Conditions Contribute to Mortality:**  
   - Patients with **anaemia, diabetes, or high blood pressure** showed a higher probability of death.  

4️⃣ **Time (Follow-up Period) Affects Survival:**  
   - Patients with **short follow-up times (<10 days)** had a higher risk of heart failure, indicating critical conditions at admission.  

5️⃣ **Imbalanced Data Biased Initial Model Predictions:**  
   - Before SMOTE, the model was biased toward predicting `DEATH_EVENT = 0`. Balancing the data helped improve the prediction accuracy.  

---

## 📌 **Model Improvement Recommendations (5 Marks)**  
💡 **1. Use More Features**  
   - Including **ECG data, cholesterol levels, and heart rate** could improve accuracy.  

💡 **2. Experiment with Different Architectures**  
   - Adding **more hidden layers** or using a **dropout layer** (to prevent overfitting) may improve performance.  

💡 **3. Hyperparameter Tuning**  
   - Testing different optimizers (`SGD`, `RMSprop`), batch sizes, and learning rates can refine model accuracy.  

💡 **4. Use Other Machine Learning Models for Comparison**  
   - Trying **Random Forest, XGBoost, or SVM** alongside ANN would help compare performance.  

💡 **5. Increase Dataset Size**  
   - More training data will reduce overfitting and help ANN generalize better.  

---

## 📈 **Final Thoughts**  
This project successfully built a **heart failure prediction model using ANN**. The model can help doctors assess **high-risk patients** based on clinical data.  
Future improvements, such as **feature engineering, advanced architectures, and larger datasets**, can enhance its predictive power.  

---

### 🚀 **Author: Arjun**
✅ **Technologies Used:** Python, TensorFlow/Keras, Scikit-Learn, SMOTE, Pandas, Seaborn  
✅ **Contact:** [arjunmenon21102003@gmail.com]
