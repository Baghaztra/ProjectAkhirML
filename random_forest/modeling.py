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
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

# Siapkan fitur
bandara_list = df.columns[3:]  # Daftar semua bandara
for bandara in bandara_list:
    df[f"{bandara}_lag"] = df[bandara].shift(1).fillna(0)

# Pilih fitur untuk semua bandara
X = df[["Tahun", "Bulan", "Jenis"] + [f"{bandara}_lag" for bandara in bandara_list]]
y = df[bandara_list[:5]]  # Menggunakan target untuk semua bandara

# Split data menjadi training dan testing set
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# ======================== bagian ini akan diubah ubah untuk mengganti model

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Latih model
rf_model.fit(X_train, y_train)

# Prediksi
y_pred = rf_model.predict(X_test)

#================================================================

# Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

# Cetak hasil
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")
print(f"Explained Variance Score: {evs}")
print(f"Median Absolute Error: {medae}")

# Definisikan toleransi
for tolerance in [0.05, 0.1, 0.15]:  # Toleransi 5%, 10%, 15%
    accuracy = np.mean(np.abs((y_test - y_pred) / y_test) <= tolerance) * 100
    print(f"Akurasi untuk toleransi ±{tolerance*100}%: {accuracy:.2f}%")

# print("Nama field untuk fitur (features):")
# print(X.columns.tolist())
# print("Nama field untuk target:")
# print(y.columns.tolist())
