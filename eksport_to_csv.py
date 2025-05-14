import pandas as pd
from glob import glob

file_paths = glob("data_*.csv")
data_internasional = []
data_domestik = []

bulan_list = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember", "Tahunan"]

for file in file_paths:
    df = pd.read_csv(file, skiprows=4) 
    df.rename(columns={"Unnamed: 0": "Bandara"}, inplace=True)

    # Ambil tahun dari nama file
    tahun = file.split("_")[1].split(".")[0]

    # Hapus kolom yang tidak dibutuhkan
    df = df[~df["Bandara"].isin(["TOTAL", "Bandara Lainnya"])]

    # Atur kolom domestik dan internasional
    domestik_cols = ["Bandara"] + bulan_list
    internasional_cols = ["Bandara"] + [f"{bulan}.1" for bulan in bulan_list]

    # Proses data domestik
    df_domestik = df[domestik_cols].copy()
    df_domestik = df_domestik.melt(id_vars=["Bandara"], var_name="Bulan", value_name="Jumlah")
    df_domestik["Jenis"] = "Domestik"
    df_domestik["Tahun"] = tahun
    data_domestik.append(df_domestik)

    # Proses data internasional
    df_internasional = df[["Bandara"] + internasional_cols[1:]].copy()
    df_internasional.columns = domestik_cols  # Samakan header dengan domestik
    df_internasional = df_internasional.melt(id_vars=["Bandara"], var_name="Bulan", value_name="Jumlah")
    df_internasional["Jenis"] = "Internasional"
    df_internasional["Tahun"] = tahun
    data_internasional.append(df_internasional)

# Gabungkan semua data
df_internasional_all = pd.concat(data_internasional, ignore_index=True)
df_domestik_all = pd.concat(data_domestik, ignore_index=True)
df_all = pd.concat([df_internasional_all, df_domestik_all], ignore_index=True)

# Pivot data untuk format akhir
df_pivot_all = df_all.pivot(index=["Tahun", "Bulan", "Jenis"], columns="Bandara", values="Jumlah").reset_index()

# Penanganan missing value
df_pivot_all.replace("-", pd.NA, inplace=True)
df_pivot_all = df_pivot_all.loc[:, ~df_pivot_all.columns.isna()]

df_pivot_all = df_pivot_all.dropna(subset=["Hasanudin-Makassar","Juanda-Surabaya","Kualanamu-Medan","Ngurah Rai-Bali","Soekarno Hatta-Jakarta"], how="all")

df_pivot_all['Hasanudin-Makassar'] = df_pivot_all['Hasanudin-Makassar'].fillna(0)

print(df_pivot_all)


# Simpan ke file CSV
df_pivot_all.to_csv("datafinal.csv", index=False)

print("Data berhasil disusun dan disimpan ke datafinal.csv")
