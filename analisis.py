import pandas as pd
from glob import glob

file_paths = glob("data_*.csv")
dataframes = []

for file in file_paths:
    df = pd.read_csv(file)
    
    tahun = file.split("_")[1].split(".")[0] 
    df['Tahun'] = tahun

    dataframes.append(df)

# Gabungkan semua DataFrame
df_all = pd.concat(dataframes, ignore_index=True)

print(df_all)
# Simpan data ke file Excel
df_all.to_excel("gabungan_data.xlsx", index=False, sheet_name="Data Gabungan")
