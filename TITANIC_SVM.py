# ================================================
# Nama: Nicholas
# Dataset: Titanic
# UTS Machine Learning
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    roc_curve, roc_auc_score
)

# ---------------------------------------------------
# 1. Memuat Dataset Titanic
# ---------------------------------------------------
data = sns.load_dataset('titanic')

print("\n--- Info Dataset ---")
print(data.info())
print("\n--- 5 Data Teratas ---")
print(data.head())
print("\n--- Statistik Deskriptif ---")
print(data.describe())

# ---------------------------------------------------
# 2. Visualisasi Awal
# ---------------------------------------------------
sns.countplot(x='survived', data=data, palette='mako')
plt.title('Distribusi Penumpang Selamat (0 = Tidak, 1 = Ya)', fontsize=12)
plt.show(block=True)

# ---------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

X = data[features]
y = data[target]

# Tangani missing value
age_imputer = SimpleImputer(strategy='median')
X['age'] = age_imputer.fit_transform(X[['age']])

embarked_imputer = SimpleImputer(strategy='most_frequent')
X['embarked'] = embarked_imputer.fit_transform(X[['embarked']])[:, 0]

# One-hot encoding
X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)

print("\n--- Data Setelah Preprocessing ---")
print(X.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# 4. Pelatihan Model
# ---------------------------------------------------
print("\n--- Melatih Model Regresi Logistik ---")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
y_prob_log = log_reg.predict_proba(X_test)[:, 1]
print("Model Regresi Logistik selesai dilatih.")

print("\n--- Melatih Model Decision Tree ---")
dec_tree = DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train, y_train)
y_pred_tree = dec_tree.predict(X_test)
y_prob_tree = dec_tree.predict_proba(X_test)[:, 1]
print("Model Decision Tree selesai dilatih.")

# ---------------------------------------------------
# 5. Evaluasi Model
# ---------------------------------------------------
print("\n==============================")
print("HASIL EVALUASI MODEL")
print("==============================\n")

# --- Regresi Logistik ---
print("\n--- 1. Regresi Logistik ---")
cm_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm_log, annot=True, fmt='d', cmap='cool', 
            xticklabels=['Tidak Selamat', 'Selamat'],
            yticklabels=['Tidak Selamat', 'Selamat'])
plt.title('Confusion Matrix - Regresi Logistik', fontsize=11)
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.show(block=True)

print("\nLaporan Klasifikasi (Regresi Logistik):")
print(classification_report(y_test, y_pred_log, target_names=['Tidak Selamat', 'Selamat']))

# --- Decision Tree ---
print("\n--- 2. Decision Tree ---")
cm_tree = confusion_matrix(y_test, y_pred_tree)
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='crest', 
            xticklabels=['Tidak Selamat', 'Selamat'],
            yticklabels=['Tidak Selamat', 'Selamat'])
plt.title('Confusion Matrix - Decision Tree', fontsize=11)
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.show(block=True)

print("\nLaporan Klasifikasi (Decision Tree):")
print(classification_report(y_test, y_pred_tree, target_names=['Tidak Selamat', 'Selamat']))

# --- ROC Curve ---
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
auc_log = roc_auc_score(y_test, y_prob_log)
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
auc_tree = roc_auc_score(y_test, y_prob_tree)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC={auc_log:.2f})', color='#4E79A7', linewidth=2.5)
plt.plot(fpr_tree, tpr_tree, label=f'Decision Tree (AUC={auc_tree:.2f})', color='#F28E2B', linewidth=2.5)
plt.plot([0, 1], [0, 1], 'k--', label='Garis Acak (AUC=0.50)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Kurva ROC - Perbandingan Model', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show(block=True)

# ---------------------------------------------------
# 6. Tabel Perbandingan
# ---------------------------------------------------
df_metrics = pd.DataFrame({
    'Model': ['Regresi Logistik', 'Decision Tree'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_tree)
    ],
    'Precision (Selamat)': [
        precision_score(y_test, y_pred_log),
        precision_score(y_test, y_pred_tree)
    ],
    'Recall (Selamat)': [
        recall_score(y_test, y_pred_log),
        recall_score(y_test, y_pred_tree)
    ],
    'F1-Score (Selamat)': [
        f1_score(y_test, y_pred_log),
        f1_score(y_test, y_pred_tree)
    ],
    'AUC': [auc_log, auc_tree]
})

print("\n--- Tabel Perbandingan Metrik ---")
print(df_metrics.to_string())

print("\n=== Selesai ===")
input("\nTekan Enter untuk menutup semua grafik...")
