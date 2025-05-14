import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Memuat model dan scaler dari file
with open('model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Definisikan bandara_list dengan nama yang sesuai dengan yang digunakan saat pelatihan
bandara_list = ["Hasanudin-Makassar", "Juanda-Surabaya", "Kualanamu-Medan", "Ngurah Rai-Bali", "Soekarno Hatta-Jakarta"]  # Gantilah dengan bandara yang sesuai

# Mapping bulan dan jenis penerbangan
bulan_mapping = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6,
    "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, "November": 11, "Desember": 12
}

jenis_mapping = {"Domestik": 1, "Internasional": 0}

# Fungsi untuk menerima input dan memprediksi jumlah penumpang
def prediksi_penumpang(bandara, tahun, bulan, jenis):
    # Memetakan bulan dan jenis penerbangan
    bulan_angka = bulan_mapping.get(bulan, None)
    jenis_angka = jenis_mapping.get(jenis, None)

    if bulan_angka is None or jenis_angka is None:
        print("Input bulan atau jenis penerbangan tidak valid!")
        return

    # Membuat data input untuk prediksi
    data_input = {
        "Tahun": [tahun],
        "Bulan": [bulan_angka],
        "Jenis": [jenis_angka]
    }

    # Menambahkan kolom lag untuk bandara yang dipilih
    for bandara in bandara_list:
        if f"{bandara}_lag" not in data_input:
            data_input[f"{bandara}_lag"] = [0]  # Jika tidak ada data lag, set ke 0 (atau nilai lain)

    # Membuat DataFrame untuk input
    input_df = pd.DataFrame(data_input)

    # Daftar kolom yang dibutuhkan berdasarkan pelatihan model
    required_columns = ["Tahun", "Bulan", "Jenis"] + [f"{bandara}_lag" for bandara in bandara_list]

    # Menambahkan kolom lag lainnya (jika diperlukan) agar sesuai dengan fitur yang digunakan saat pelatihan
    for col in required_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Menambahkan kolom yang hilang dengan nilai 0

    # Menyusun ulang kolom agar sesuai dengan urutan yang dibutuhkan
    input_df = input_df[required_columns]

    # Menormalisasi data input menggunakan scaler yang sudah disimpan
    input_scaled = scaler.transform(input_df)

    # Melakukan prediksi
    prediksi = rf_model.predict(input_scaled)

    # Menampilkan hasil prediksi
    print(f"Prediksi jumlah penumpang yang berangkat dari {bandara} pada {bulan} {tahun} ({jenis}): {prediksi[0][0]:,.0f}")


# Contoh penggunaan
bandara = "Ngurah Rai-Bali"
tahun = 2020
bulan = "Februari"
jenis = "Internasional"

prediksi_penumpang(bandara, tahun, bulan, jenis)
