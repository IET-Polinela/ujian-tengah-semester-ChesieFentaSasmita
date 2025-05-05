import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = 'healthcare-dataset-stroke-data.csv'
df = pd.read_csv(file_path)

# Drop kolom yang tidak dibutuhkan
df.drop(columns=['id'], inplace=True)

# Imputasi nilai NaN pada 'bmi' secara aman
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# One-hot encoding pada kolom kategorikal
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

# Pisahkan fitur dan target
X = df.drop(columns=['stroke'])
y = df['stroke']

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Prediksi
y_pred = clf.predict(X_test)

# Evaluasi metrik
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Print metrik
print("=== METRIK EVALUASI MODEL ===")
print(f"Akurasi  : {accuracy:.4f}")
print(f"Presisi  : {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Analisis berbasis metrik
print("\n=== ANALISIS HASIL ===")
if f1 >= 0.6:
    print("- Model menunjukkan performa yang cukup baik dalam mendeteksi kasus stroke.")
else:
    print("- Model masih kurang optimal, terutama dalam mendeteksi kasus stroke.")

print("- Recall tinggi penting agar sebanyak mungkin kasus stroke bisa dideteksi.")
print("- Precision tinggi penting untuk menghindari false positive yang bikin pasien panik.")

if recall < 0.5:
    print("- Recall masih rendah, artinya banyak kasus stroke belum terdeteksi.")
if precision < 0.5:
    print("- Precision rendah, artinya banyak kasus yang dikira stroke padahal bukan.")

print("- Disarankan untuk balancing data (SMOTE/undersampling) dan tuning hyperparameter model.\n")

print("Data preprocessing dan modeling selesai.")

