import rasterio
from rasterio.transform import from_origin
import numpy as np
import matplotlib.pyplot as plt

# ================= KONFIGURASI =================
NAMA_FILE = 'dummy_sawah_realistik.tif'
WIDTH = 600
HEIGHT = 600
JUMLAH_BAND = 5
DTYPE = rasterio.float32

# Ukuran Petak Sawah (dalam pixel)
UKURAN_PETAK_X = 50  # Lebar satu kotak sawah
UKURAN_PETAK_Y = 60  # Tinggi satu kotak sawah
TEBAL_GALENGAN = 4  # Tebal garis pematang

# Koordinat (Semarang, Jawa Tengah)
WEST = 110.4
NORTH = -7.0
PIXEL_SIZE = 0.00005  # ~5 meter


# ================= FUNGSI LOGIKA =================

def buat_peta_kesuburan_sawah(h, w):
    """
    Membuat peta dasar dengan pola kotak-kotak (petak sawah).
    Setiap kotak memiliki tingkat kesuburan acak yang berbeda.
    """
    # 1. Buat kanvas kosong
    map_base = np.zeros((h, w), dtype=np.float32)

    # 2. Loop untuk membuat kotak-kotak
    num_rows = h // UKURAN_PETAK_Y
    num_cols = w // UKURAN_PETAK_X

    for r in range(num_rows + 1):
        for c in range(num_cols + 1):
            # Tentukan batas pixel untuk petak ini
            y_start = r * UKURAN_PETAK_Y
            y_end = min((r + 1) * UKURAN_PETAK_Y, h)
            x_start = c * UKURAN_PETAK_X
            x_end = min((c + 1) * UKURAN_PETAK_X, w)

            if y_start >= h or x_start >= w:
                continue

            # Tentukan kesuburan acak untuk SATU PETAK INI
            # 0.2 (Sakit/Gundul) s/d 0.9 (Sangat Subur)
            kesuburan_petak = np.random.uniform(0.2, 0.95)

            # Isi petak dengan nilai tersebut
            map_base[y_start:y_end, x_start:x_end] = kesuburan_petak

    # 3. Tambahkan Galengan (Pematang Sawah) - Nilai rendah (Tanah)
    # Garis Horizontal
    for y in range(0, h, UKURAN_PETAK_Y):
        map_base[y:min(y + TEBAL_GALENGAN, h), :] = 0.15  # Nilai 0.15 anggap tanah liat

    # Garis Vertikal
    for x in range(0, w, UKURAN_PETAK_X):
        map_base[:, x:min(x + TEBAL_GALENGAN, w)] = 0.15

    # 4. Tambahkan Noise (Agar tidak terlalu mulus seperti plastik)
    noise = np.random.normal(0, 0.03, (h, w))
    map_final = map_base + noise

    # Clip agar tetap di range 0-1
    return np.clip(map_final, 0, 1)


# ================= PROSES GENERASI =================
print("Sedang mengenerate pola sawah...")

# Buat pola dasar kesuburan (Vigor Map)
vigor = buat_peta_kesuburan_sawah(HEIGHT, WIDTH)

print("Mensimulasikan pantulan cahaya multispektral...")

# Rumus Pantulan Cahaya (Reflectance Simulation)
# Ingat: Tanaman sehat -> Red Rendah, NIR Tinggi

# Band 1: Blue (Tanah agak terang, Tanaman agak gelap)
b1_blue = np.clip(0.1 + (vigor * -0.05) + np.random.normal(0, 0.01, (HEIGHT, WIDTH)), 0, 1)

# Band 2: Green (Tanaman memantulkan hijau)
b2_green = np.clip(0.1 + (vigor * 0.15) + np.random.normal(0, 0.01, (HEIGHT, WIDTH)), 0, 1)

# Band 3: Red (Diserap klorofil -> Makin subur makin gelap)
# Jika vigor 0.9 (subur), Red jadi rendah (misal 0.05)
# Jika vigor 0.1 (tanah), Red jadi tinggi (misal 0.25)
b3_red = np.clip(0.3 - (vigor * 0.25) + np.random.normal(0, 0.01, (HEIGHT, WIDTH)), 0, 1)

# Band 4: NIR (Dipantulkan struktur sel -> Makin subur makin terang)
# Jika vigor tinggi, NIR sangat tinggi
b4_nir = np.clip(0.2 + (vigor * 0.6) + np.random.normal(0, 0.02, (HEIGHT, WIDTH)), 0, 1)

# Band 5: RedEdge (Transisi)
b5_re = np.clip(0.15 + (vigor * 0.3) + np.random.normal(0, 0.01, (HEIGHT, WIDTH)), 0, 1)

# Gabung jadi tumpukan array
data_bands = np.stack([b1_blue, b2_green, b3_red, b4_nir, b5_re])

# Simpan ke GeoTIFF
transform = from_origin(WEST, NORTH, PIXEL_SIZE, PIXEL_SIZE)
crs = rasterio.crs.CRS.from_epsg(4326)

meta = {
    'driver': 'GTiff',
    'height': HEIGHT,
    'width': WIDTH,
    'count': JUMLAH_BAND,
    'dtype': DTYPE,
    'crs': crs,
    'transform': transform,
    'compress': 'lzw'
}

print(f"Menyimpan file: {NAMA_FILE}...")
with rasterio.open(NAMA_FILE, 'w', **meta) as dst:
    for i, band in enumerate(data_bands, start=1):
        dst.write(band.astype(DTYPE), i)
        dst.set_band_description(i, f"Band {i}")

print("SELESAI! File dummy sawah siap.")

# ================= PREVIEW =================
plt.figure(figsize=(10, 8))
# Tampilkan False Color (NIR-Red-Green) agar vegetasi terlihat merah terang
img_display = np.dstack((b4_nir, b3_red, b2_green))
img_display = img_display / img_display.max()  # Normalisasi agar bisa ditampilkan
plt.imshow(img_display)
plt.title("Preview Dummy Sawah (False Color Infrared)\nMerah = Tanaman Subur, Kotak Gelap = Kurang Subur/Air")
plt.axis('off')
plt.show()