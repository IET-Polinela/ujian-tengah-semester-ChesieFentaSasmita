
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = 'healthcare-dataset-stroke-data.csv'
df = pd.read_csv(file_path)

# Drop kolom 'id' (tidak relevan)
df.drop(columns=['id'], inplace=True)

# Imputasi missing value di kolom 'bmi' dengan mean
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# One-Hot Encoding untuk kolom kategorikal
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

# Pisahkan fitur dan target
X = df.drop(columns=['stroke'])
y = df['stroke']

# Balancing data pakai SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Inisialisasi model RandomForest dengan class_weight='balanced'
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Threshold tuning: prediksi pakai probabilitas
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.3).astype(int)  # threshold diturunin biar lebih berani deteksi stroke

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualisasi Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.close()

# Visualisasi Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[:10], y=feature_importance.index[:10])
plt.title("Top 10 Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Simpan dataset hasil preprocessing
pd.DataFrame(X_resampled, columns=X.columns).assign(stroke=y_resampled).to_csv('preprocessed_stroke_data.csv', index=False)

# Simpan hasil evaluasi dan analisis ke file teks
with open("model_metrics.txt", "w") as f:
    f.write("=== METRIK EVALUASI SETELAH BALANCING DAN TUNING ===\n")
    f.write(f"Akurasi  : {accuracy:.4f}\n")
    f.write(f"Presisi  : {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n\n")
    
print("Model selesai dilatih dengan data seimbang. Semua file hasil telah disimpan.")
