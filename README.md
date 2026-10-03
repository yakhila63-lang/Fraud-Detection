# Fraud Detection System - Credit Card

> Credit Card Fraud Detection with 95% Recall & AUC 1.0 - ML Project

### 📊 Project Overview
This project detects fraudulent credit card transactions using Machine Learning. The dataset is highly imbalanced (1.7% fraud).

### 📁 Files
- `main.py` - Full model training & evaluation code
- `roc_curve.png` - ROC Curve (AUC 1.0)
- `class_imbalance.png` - Class distribution
- `eda_amount_time.png` - EDA analysis
- `feature_importance.png` - Top features

### 📈 Results

| Model | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 1.00 | 0.95 | 0.97 | 1.00 |
| Random Forest | 1.00 | 0.951 | 0.9749 | 1.00 |

**Best Metric:** Recall is KING - We must catch max frauds. Threshold 0.3 recommended over 0.5.

### 🔍 Key Insights
1.  **Imbalance:** Only 1.7% fraud - Accuracy misleading (98% accuracy even if predict all Normal)
2.  **Amount:** Fraud avg amount 3x higher
3.  **Top Features:** V2, V3, V14, V1, V7 are most important
4.  **ROC:** Perfect curve near top-left corner

### 🚀 Scalability to 1M transactions/hour
For production need <50ms latency:
`Kafka -> Spark Streaming -> Redis Feature Store -> ONNX Model -> FastAPI -> Alert System`
- Model quantization
- Auto-scaling
- Drift monitoring

### 🛠️ Tech Stack
Python, Scikit-learn, Pandas, Matplotlib, Seaborn, SMOTE, StandardScaler

### 👤 Author
[yakhila63-lang](https://github.com/yakhila63-lang)

---
⭐ If you like this project, give a star!
