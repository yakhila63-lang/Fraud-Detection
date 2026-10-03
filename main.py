import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # VS Code lo save avadaniki idi important
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score

print("=== Fraud Detection Project ===")
print(f"Files save ayye location: {os.getcwd()}")

# Demo data with pattern
np.random.seed(42)
n_total, n_fraud = 20000, 340
n_normal = n_total - n_fraud
data = {}
for i in range(1, 29):
    normal_vals = np.random.normal(0, 1, n_normal)
    if i in [1,2,3,7,14]:
        fraud_vals = np.random.normal(2.5, 0.8, n_fraud)
    elif i in [4,10,12]:
        fraud_vals = np.random.normal(-2.0, 0.8, n_fraud)
    else:
        fraud_vals = np.random.normal(0, 1, n_fraud)
    data[f'V{i}'] = np.concatenate([normal_vals, fraud_vals])

df = pd.DataFrame(data)
df['Time'] = np.random.uniform(0, 172792, n_total)
df['Amount'] = np.concatenate([np.random.lognormal(3.5, 0.8, n_normal), np.random.lognormal(5.0, 0.5, n_fraud)])
df['Class'] = [0]*n_normal + [1]*n_fraud
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Shape: {df.shape}, Fraud %: {df['Class'].mean()*100:.4f}%")

# 1. Class imbalance
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title('Class Imbalance')
plt.savefig('class_imbalance.png', dpi=150)
plt.close()
print("✅ Saved class_imbalance.png")

# 2. EDA
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(df[df.Class==0].Amount, bins=50, label='Non-Fraud', alpha=0.6)
sns.histplot(df[df.Class==1].Amount, bins=50, label='Fraud', alpha=0.7)
plt.legend(); plt.title('Amount Distribution'); plt.xlim(0,500)
plt.subplot(1,2,2)
df['Hour'] = (df['Time']/3600) % 24
sns.histplot(data=df, x='Hour', hue='Class', bins=24, alpha=0.7)
plt.title('Time-of-Day')
plt.tight_layout()
plt.savefig('eda_amount_time.png', dpi=150)
plt.close()
print("✅ Saved eda_amount_time.png")

# Models
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])
X = df.drop(['Class','Amount','Time','Hour'], axis=1, errors='ignore')
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

lr = LogisticRegression(class_weight='balanced', max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train); rf.fit(X_train, y_train)

for name, model in [('Logistic Regression', lr), ('Random Forest', rf)]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

# 3. ROC
plt.figure(figsize=(6,5))
for name, model in [('Logistic Regression', lr), ('Random Forest', rf)]:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr, label=f'{name} AUC={auc(fpr,tpr):.3f}')
plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.title('AUC-ROC Curve')
plt.savefig('roc_curve.png', dpi=150)
plt.close()
print("✅ Saved roc_curve.png")

# 4. Feature importance
importances = rf.feature_importances_
idx = np.argsort(importances)[-15:]
plt.figure(figsize=(8,6))
plt.barh(range(15), importances[idx])
plt.yticks(range(15), [X.columns[i] for i in idx])
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("✅ Saved feature_importance.png")

print(f"\n=== DONE MAMA! Check folder: {os.getcwd()} ===")
print("4 pics akkada unayi!")