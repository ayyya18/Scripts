import rasterio
import numpy as np
import joblib
import matplotlib.pyplot as plt

PATH_MODEL = 'model_rf_hara.joblib'
PATH_FOTO_UDARA = 'orthomosaic_multispectral.tif'
PATH_HASIL = 'peta_prediksi_serapan_k.tif'

BAND_GREEN = 2
BAND_RED = 3
BAND_NIR = 4
BAND_RED_EDGE = 5

def hitung_indeks(red, green, nir, red_edge):
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
        ndre = (nir - red_edge) / (nir + red_edge)
        gndvi = (nir - green) / (nir + green)
    return ndvi, ndre, gndvi


print("1. Memuat Model...")
model = joblib.load(PATH_MODEL)
print("   Model berhasil dimuat.")

print(f"2. Membaca File GeoTIFF: {PATH_FOTO_UDARA}")
try:
    with rasterio.open(PATH_FOTO_UDARA) as src:
        profile = src.profile
        red = src.read(BAND_RED).astype('float32')
        green = src.read(BAND_GREEN).astype('float32')
        nir = src.read(BAND_NIR).astype('float32')
        re = src.read(BAND_RED_EDGE).astype('float32')
        height, width = red.shape
        red[red == 0] = np.nan

    print("3. Menghitung Indeks Vegetasi (NDVI, NDRE, GNDVI)...")
    ndvi, ndre, gndvi = hitung_indeks(red, green, nir, re)

    print("4. Menyiapkan Data untuk Prediksi...")
    df_prediksi = np.column_stack((ndvi.flatten(), ndre.flatten(), gndvi.flatten()))
    mask_valid = ~np.isnan(df_prediksi).any(axis=1)
    hasil_prediksi_flat = np.full(df_prediksi.shape[0], np.nan)

    print("5. Melakukan Prediksi (Ini mungkin memakan waktu)...")
    if np.sum(mask_valid) > 0:
        pixel_valid = df_prediksi[mask_valid]
        prediksi = model.predict(pixel_valid)
        hasil_prediksi_flat[mask_valid] = prediksi
    else:
        print("   Warning: Tidak ada pixel valid untuk diprediksi.")
    hasil_map = hasil_prediksi_flat.reshape(height, width)

    print(f"6. Menyimpan Hasil ke: {PATH_HASIL}")
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )

    with rasterio.open(PATH_HASIL, 'w', **profile) as dst:
        dst.write(hasil_map.astype(rasterio.float32), 1)

    print("SELESAI! Peta serapan hara siap dibuka di QGIS/ArcGIS.")
    plt.figure(figsize=(10, 8))
    plt.imshow(hasil_map, cmap='YlGn')  
    plt.colorbar(label='Prediksi Serapan K')
    plt.title('Preview Hasil Prediksi')
    plt.show()

except FileNotFoundError:
    print("ERROR: File GeoTIFF tidak ditemukan. Pastikan nama file benar.")
except Exception as e:
    print(f"ERROR Terjadi: {e}")