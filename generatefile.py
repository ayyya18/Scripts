import pandas as pd
import numpy as np

np.random.seed(100)
jumlah_sampel = 100
ndvi = np.random.uniform(0.4, 0.9, jumlah_sampel)
ndre = np.random.uniform(0.2, 0.6, jumlah_sampel)
gndvi = np.random.uniform(0.3, 0.8, jumlah_sampel)

serapan_k = (15 * ndvi) + (20 * ndre) + (10 * gndvi) + np.random.normal(0, 1.5, jumlah_sampel)

df_latihan = pd.DataFrame({
    'NDVI': ndvi,
    'NDRE': ndre,
    'GNDVI': gndvi,
    'Serapan_K': serapan_k
})
df_latihan = df_latihan.round(4)
nama_file = 'data_tanaman.csv'
df_latihan.to_csv(nama_file, index=False)

print(f"Sukses! File '{nama_file}' berisi {jumlah_sampel} data telah dibuat.")
print("Berikut 5 baris pertama data Anda:")
print(df_latihan.head())

joblib.dump(rf_model, 'model_rf_hara.joblib')
print("\nModel berhasil disimpan sebagai 'model_rf_hara.joblib'")