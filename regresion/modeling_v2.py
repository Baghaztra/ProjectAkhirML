import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error
)
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv('../datafinal.csv')

# Konversi kolom menjadi numerik
bulan_mapping = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6,
    "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, "November": 11, "Desember": 12
}
df["Bulan"] = df["Bulan"].map(bulan_mapping)
df["Jenis"] = df["Jenis"].map({"Domestik": 1, "Internasional": 0})

# Pastikan data sudah numerik
df.iloc[:, 3:] = df.iloc[:, 3:].apply(pd.to_numeric, errors='coerce')

# Tangani data hilang
df.fillna(0, inplace=True)

# Siapkan fitur
bandara_list = df.columns[3:]  # Daftar semua bandara
for bandara in bandara_list:
    df[f"{bandara}_lag"] = df[bandara].shift(1).fillna(0)

# Pilih fitur dan target
X = df[["Tahun", "Bulan", "Jenis"] + [f"{bandara}_lag" for bandara in bandara_list]]
y = df[bandara_list[:5]]  # Target untuk semua bandara

# =====================================================Korelasi

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Gabungkan X dan Y untuk analisis korelasi
# data = pd.concat([X, y], axis=1)
# correlation_matrix = data.corr()

# # Plot heatmap
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title("Korelasi antara Variabel")
# plt.show()


# =============================================================

# Normalisasi data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Random Forest
rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)

# Prediksi
y_pred = rf_model.predict(X_test)

# Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

# Cetak hasil evaluasi
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")
print(f"Explained Variance Score: {evs}")
print(f"Median Absolute Error: {medae}")

# Akurasi berdasarkan toleransi
for tolerance in [0.05, 0.1, 0.15, 0.2]:  # Toleransi 5%, 10%, 15%, 20%
    accuracy = np.mean(np.abs((y_test - y_pred) / y_test) <= tolerance) * 100
    print(f"Akurasi untuk toleransi ±{tolerance*100}%: {accuracy:.2f}%")

# Feature Importance
feature_importances = rf_model.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(feature_importances)[::-1]

print("\nFitur Terpenting dalam Model:")
for idx in sorted_indices[:10]:  # Menampilkan 10 fitur teratas
    print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")


# ================================================================= Export Model
# import pickle

# # Menyimpan model ke dalam file
# with open('model.pkl', 'wb') as f:
#     pickle.dump(rf_model, f)

# # Menyimpan scaler yang digunakan untuk normalisasi
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)

# =================================================================Nama field

# print("Nama field untuk fitur (features):")
# print(X.columns.tolist())
# print("Nama field untuk target:")
# print(y.columns.tolist())

# # =================================================================Visualisasi

# import pandas as pd
# import matplotlib.pyplot as plt

# # Pilih bandara untuk divisualisasikan
# bandara_pilihan = "Soekarno Hatta-Jakarta"

# # Ambil data aktual dan prediksi untuk bandara tersebut
# y_test_bandara = y_test[bandara_pilihan].values
# y_pred_bandara = y_pred[:, list(y.columns).index(bandara_pilihan)]

# # Buat label Tahun-Bulan untuk sumbu X
# tahun_bulan = df.iloc[X_test[:, 0].argsort()][["Tahun", "Bulan"]]
# tahun_bulan_labels = tahun_bulan.apply(lambda row: f"{int(row['Tahun'])}-{int(row['Bulan'])}", axis=1).values

# # Gabungkan data ke dalam DataFrame
# data = pd.DataFrame({
#     "Tahun-Bulan": tahun_bulan_labels,
#     "y_test": y_test_bandara,
#     "y_pred": y_pred_bandara
# })

# # Hapus duplikat berdasarkan kolom Tahun-Bulan, simpan yang pertama
# data_unique = data.drop_duplicates(subset=["Tahun-Bulan"], keep="last")

# # Plot data aktual dan prediksi
# plt.figure(figsize=(12, 6))
# plt.plot(data_unique["Tahun-Bulan"], data_unique["y_test"], label="Aktual", marker='o', linestyle='-')
# plt.plot(data_unique["Tahun-Bulan"], data_unique["y_pred"], label="Prediksi", marker='x', linestyle='--')

# # Tambahkan detail ke grafik
# plt.title(f"Perbandingan Aktual vs Prediksi ({bandara_pilihan})")
# plt.xlabel("Tahun-Bulan")
# plt.ylabel("Jumlah Penumpang")
# plt.xticks(rotation=45)  # Rotasi label sumbu X untuk keterbacaan
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
