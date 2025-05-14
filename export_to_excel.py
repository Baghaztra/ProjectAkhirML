import pandas as pd
from glob import glob

# Ambil semua file CSV
file_paths = glob("data_*.csv")  # Pastikan nama file sesuai pola ini
data_domestik = []
data_internasional = []

for file in file_paths:
    # Baca file CSV
    df = pd.read_csv(file, skiprows=3)  # Header mulai dari baris ke-4

    # Ambil tahun dari nama file
    tahun = file.split("_")[1].split(".")[0]

    # Tambahkan nama kolom
    domestik_cols = ["Bandara"] + [f"Domestik_{bulan}" for bulan in ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember", "Tahunan"]]
    internasional_cols = ["Bandara"] + [f"Internasional_{bulan}" for bulan in ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember", "Tahunan"]]

    df.columns = domestik_cols + internasional_cols[1:]

    # Hapus baris TOTAL dan Bandara Lainnya
    df = df[~df["Bandara"].isin(["TOTAL", "Bandara Lainnya"])]

    # Tambahkan kolom Tahun
    df["Tahun"] = tahun

    # Split menjadi dua DataFrame
    df_domestik = df[["Bandara", "Tahun"] + domestik_cols[1:]]
    df_internasional = df[["Bandara", "Tahun"] + internasional_cols[1:]]

    # Tambahkan ke list
    data_domestik.append(df_domestik)
    data_internasional.append(df_internasional)

# Gabungkan semua data
df_domestik_all = pd.concat(data_domestik, ignore_index=True)
df_internasional_all = pd.concat(data_internasional, ignore_index=True)

# Simpan ke file Excel
with pd.ExcelWriter("gabungan_data.xlsx", engine="openpyxl") as writer:
    df_domestik_all.to_excel(writer, index=False, sheet_name="Domestik")
    df_internasional_all.to_excel(writer, index=False, sheet_name="Internasional")

print("Data berhasil digabungkan dan disimpan ke gabungan_data.xlsx")
