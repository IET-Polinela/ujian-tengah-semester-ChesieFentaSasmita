
import pandas as pd

# Load dataset dari file lokal di Colab
file_path = 'healthcare-dataset-stroke-data.csv'
df = pd.read_csv(file_path)

# Informasi umum dataset
print("Jumlah baris (observasi):", df.shape[0])
print("Jumlah kolom (fitur):", df.shape[1])
print("\nNama-nama fitur:")
print(df.columns.tolist())

# Statistik deskriptif untuk fitur numerik
print("\nStatistik deskriptif (numerik):")
print(df.describe())

# Statistik deskriptif untuk fitur kategorikal
print("\nStatistik deskriptif (kategorikal):")
print(df.describe(include=['object']))

# Jumlah data kosong per kolom
print("\nJumlah data kosong (null) per kolom:")
print(df.isnull().sum())

# Fitur yang dianggap paling relevan untuk prediksi stroke
relevant_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
print("\nFitur relevan untuk prediksi stroke:")
print(relevant_features)
