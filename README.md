# 🧠 Task 5: Decision Trees and Random Forests - Heart Disease Prediction

## 📌 Objective
To build and evaluate tree-based models for classification using Decision Trees and Random Forests, and interpret the results using SHAP for model explainability.

---

## 🛠️ Tools & Libraries Used
- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- SHAP (SHapley Additive exPlanations)
- Graphviz (optional, for tree visualization)

---

## 📊 Dataset
**Heart Disease Dataset**  
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) or directly use `kagglehub`.

---

## 🔍 Steps Performed

### 1. **Data Preprocessing**
- Loaded the dataset and explored basic info
- Handled missing values (if any)
- Encoded categorical variables using Label Encoding / OneHotEncoding
- Standardized numerical features using `StandardScaler`

### 2. **Model Training**
- Trained a **Decision Tree Classifier**
- Visualized the tree structure
- Controlled for overfitting using `max_depth`, `min_samples_split`, etc.

### 3. **Model Evaluation**
- Evaluated using accuracy, confusion matrix, precision, recall, and F1-score
- Used **cross-validation** to assess model robustness

### 4. **Random Forest**
- Trained a **Random Forest Classifier**
- Compared accuracy with the Decision Tree
- Extracted and visualized **feature importances**

### 5. **Explainability with SHAP**
- Applied `shap.TreeExplainer` to interpret model predictions
- Used `shap.summary_plot()` to show feature contributions
- Explained why certain features influence the target variable

---

## 📈 Visualizations Included
- Correlation heatmap
- Decision tree plot (if Graphviz is installed)
- Feature importance bar plot
- SHAP summary beeswarm plot

---

## 🧠 Key Learnings
- How tree-based models make decisions
- Controlling model complexity to avoid overfitting
- Importance of interpretability in ML using SHAP
- Comparative performance of single vs ensemble models
