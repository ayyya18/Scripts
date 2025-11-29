import rasterio
from rasterio.transform import from_origin
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import random

# ================= KONFIGURASI =================
NAMA_FILE = 'dummy_sawah_realistik.tif'
WIDTH = 800
HEIGHT = 800
JUMLAH_BAND = 5
DTYPE = rasterio.float32

# Variasi ukuran petak sawah untuk realisme
UKURAN_PETAK_MIN_X = 45
UKURAN_PETAK_MAX_X = 65
UKURAN_PETAK_MIN_Y = 55
UKURAN_PETAK_MAX_Y = 75
TEBAL_GALENGAN = 3  # Tebal garis pematang

# Koordinat (Area persawahan di Jawa Tengah)
WEST = 110.320
NORTH = -6.950
PIXEL_SIZE = 0.00004  # ~4 meter


# ================= FUNGSI LOGIKA YANG LEBIH REALISTIK =================

def generate_noise_octave(shape, frequency, octaves=4, persistence=0.5):
    """Generate Perlin-like noise dengan multiple octaves"""
    noise = np.zeros(shape)
    amplitude = 1.0
    total_amplitude = 0.0

    for _ in range(octaves):
        # Generate base noise
        base_noise = np.random.random((int(shape[0] * frequency), int(shape[1] * frequency)))
        # Resize to target shape
        base_noise = ndimage.zoom(base_noise, (shape[0] / (shape[0] * frequency),
                                               shape[1] / (shape[1] * frequency)))

        noise += base_noise * amplitude
        total_amplitude += amplitude
        frequency *= 2
        amplitude *= persistence

    return noise / total_amplitude


def buat_peta_kesuburan_sawah(h, w):
    """
    Membuat peta dasar dengan pola kotak-kotak sawah yang realistik.
    Dengan variasi ukuran petak, tekstur, dan gradasi kesuburan.
    """
    # 1. Buat kanvas kosong dengan noise dasar
    map_base = generate_noise_octave((h, w), 0.01) * 0.1 + 0.3

    # 2. Generate pola petak sawah dengan variasi ukuran
    y_pos = 0
    row_heights = []

    # Generate tinggi baris yang bervariasi
    while y_pos < h:
        height_var = random.randint(UKURAN_PETAK_MIN_Y, UKURAN_PETAK_MAX_Y)
        row_heights.append(height_var)
        y_pos += height_var

    x_pos = 0
    col_widths = []

    # Generate lebar kolom yang bervariasi
    while x_pos < w:
        width_var = random.randint(UKURAN_PETAK_MIN_X, UKURAN_PETAK_MAX_X)
        col_widths.append(width_var)
        x_pos += width_var

    # 3. Isi setiap petak dengan karakteristik unik
    y_start = 0
    for row_idx, petak_height in enumerate(row_heights):
        if y_start >= h:
            break

        y_end = min(y_start + petak_height, h)
        x_start = 0

        for col_idx, petak_width in enumerate(col_widths):
            if x_start >= w:
                break

            x_end = min(x_start + petak_width, w)

            # Nilai kesuburan dasar untuk petak ini
            base_fertility = np.random.uniform(0.3, 0.85)

            # Tambahkan gradasi kesuburan dalam petak (lebih subur di tengah)
            y_center, x_center = (y_start + y_end) / 2, (x_start + x_end) / 2
            y_grid, x_grid = np.ogrid[y_start:y_end, x_start:x_end]

            # Distance dari tengah petak
            dist_from_center = np.sqrt(((x_grid - x_center) / (petak_width / 2)) ** 2 +
                                       ((y_grid - y_center) / (petak_height / 2)) ** 2)

            # Gradasi kesuburan (tengah lebih subur)
            fertility_gradient = np.clip(1.0 - dist_from_center * 0.3, 0.5, 1.0)

            # Tambahkan tekstur noise lokal
            local_noise = np.random.normal(0, 0.08, (y_end - y_start, x_end - x_start))

            # Gabungkan semua faktor
            petak_fertility = base_fertility * fertility_gradient + local_noise

            # Terapkan ke peta dasar
            map_base[y_start:y_end, x_start:x_end] = np.clip(petak_fertility, 0.1, 0.95)

            x_start += petak_width

        y_start += petak_height

    # 4. Buat galengan (pematang) yang lebih natural
    # Horizontal galengan
    y_start = 0
    for petak_height in row_heights:
        if y_start >= h:
            break
        # Tebal galengan bervariasi
        galengan_thickness = TEBAL_GALENGAN + random.randint(-1, 1)
        y_galengan = min(y_start, h)
        y_end_galengan = min(y_start + galengan_thickness, h)

        # Beri tekstur pada galengan
        galengan_texture = np.random.uniform(0.12, 0.18, (y_end_galengan - y_galengan, w))
        map_base[y_galengan:y_end_galengan, :] = galengan_texture
        y_start += petak_height

    # Vertical galengan
    x_start = 0
    for petak_width in col_widths:
        if x_start >= w:
            break
        galengan_thickness = TEBAL_GALENGAN + random.randint(-1, 1)
        x_galengan = min(x_start, w)
        x_end_galengan = min(x_start + galengan_thickness, w)

        galengan_texture = np.random.uniform(0.12, 0.18, (h, x_end_galengan - x_galengan))
        map_base[:, x_galengan:x_end_galengan] = galengan_texture
        x_start += petak_width

    # 5. Tambahkan fitur realistik tambahan

    # Saluran irigasi (garis biru gelap)
    irrigation_channel_width = 8
    irrigation_y = h // 2
    map_base[irrigation_y:irrigation_y + irrigation_channel_width, :] = 0.08

    # Area tergenang air (nilai sangat rendah)
    water_patches = []
    for _ in range(5):
        water_y = random.randint(100, h - 100)
        water_x = random.randint(100, w - 100)
        water_size = random.randint(15, 30)
        water_patches.append((water_y, water_x, water_size))

    for water_y, water_x, water_size in water_patches:
        y1, y2 = max(0, water_y - water_size), min(h, water_y + water_size)
        x1, x2 = max(0, water_x - water_size), min(w, water_x + water_size)
        water_mask = np.sqrt(((np.ogrid[y1:y2, x1:x2][0] - water_y) / water_size) ** 2 +
                             ((np.ogrid[y1:y2, x1:x2][1] - water_x) / water_size) ** 2) <= 1
        map_base[y1:y2, x1:x2][water_mask] = 0.05

    # 6. Smoothing akhir untuk natural look
    map_base = ndimage.gaussian_filter(map_base, sigma=0.8)

    return np.clip(map_base, 0, 1)


def simulate_spectral_bands(vigor_map):
    """
    Simulasi pantulan spektral yang lebih realistik berdasarkan karakteristik vegetasi
    """
    h, w = vigor_map.shape

    # Band 1: Blue (0.45-0.51µm)
    # - Tanah: reflectance sedang (~0.15)
    # - Vegetasi: absorpsi tinggi (~0.05)
    # - Air: absorpsi sangat tinggi (~0.03)
    blue_base = np.where(vigor_map < 0.1, 0.03,  # Air
                         np.where(vigor_map < 0.2, 0.15,  # Tanah
                                  0.05 + vigor_map * 0.1))  # Vegetasi

    # Tambahkan noise dan variasi atmosfer
    blue_band = blue_base + np.random.normal(0, 0.008, (h, w))
    blue_band = ndimage.gaussian_filter(blue_band, sigma=0.5)

    # Band 2: Green (0.53-0.59µm)
    # - Puncak refleksi vegetasi sehat
    green_base = np.where(vigor_map < 0.1, 0.04,  # Air
                          np.where(vigor_map < 0.2, 0.16,  # Tanah
                                   0.1 + vigor_map * 0.25))  # Vegetasi

    green_band = green_base + np.random.normal(0, 0.01, (h, w))
    green_band = ndimage.gaussian_filter(green_band, sigma=0.5)

    # Band 3: Red (0.64-0.67µm)
    # - Absorpsi kuat oleh klorofil
    red_base = np.where(vigor_map < 0.1, 0.02,  # Air
                        np.where(vigor_map < 0.2, 0.25,  # Tanah
                                 0.05 - vigor_map * 0.2))  # Vegetasi (nilai rendah untuk vegetasi sehat)

    # Pastikan tidak negatif untuk vegetasi sangat sehat
    red_base = np.where(red_base < 0.02, 0.02, red_base)
    red_band = red_base + np.random.normal(0, 0.008, (h, w))
    red_band = ndimage.gaussian_filter(red_band, sigma=0.5)

    # Band 4: Near Infrared - NIR (0.85-0.88µm)
    # - Refleksi sangat tinggi oleh vegetasi sehat
    nir_base = np.where(vigor_map < 0.1, 0.01,  # Air
                        np.where(vigor_map < 0.2, 0.22,  # Tanah
                                 0.15 + vigor_map * 0.5))  # Vegetasi

    nir_band = nir_base + np.random.normal(0, 0.015, (h, w))
    nir_band = ndimage.gaussian_filter(nir_band, sigma=0.5)

    # Band 5: Red Edge (0.70-0.74µm)
    # - Transisi antara red dan NIR
    re_base = np.where(vigor_map < 0.1, 0.015,  # Air
                       np.where(vigor_map < 0.2, 0.18,  # Tanah
                                0.1 + vigor_map * 0.3))  # Vegetasi

    re_band = re_base + np.random.normal(0, 0.01, (h, w))
    re_band = ndimage.gaussian_filter(re_band, sigma=0.5)

    # Final clipping dan adjustment
    bands = [blue_band, green_band, red_band, nir_band, re_band]

    # Normalisasi dan koreksi radiometrik sederhana
    final_bands = []
    for band in bands:
        # Tambahkan koreksi atmosfer (haze correction)
        band_corrected = band * 0.9 + 0.02
        band_clipped = np.clip(band_corrected, 0.01, 0.9)
        final_bands.append(band_clipped)

    return final_bands


# ================= PROSES GENERASI =================
print("Membuat peta dasar kesuburan sawah...")
vigor_map = buat_peta_kesuburan_sawah(HEIGHT, WIDTH)

print("Simulasi pantulan spektral multispektral...")
bands = simulate_spectral_bands(vigor_map)
b1_blue, b2_green, b3_red, b4_nir, b5_re = bands

# Gabung jadi tumpukan array
data_bands = np.stack([b1_blue, b2_green, b3_red, b4_nir, b5_re])

# Simpan ke GeoTIFF dengan metadata lengkap
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
    'compress': 'lzw',
    'nodata': 0
}

print(f"Menyimpan file: {NAMA_FILE}...")
with rasterio.open(NAMA_FILE, 'w', **meta) as dst:
    for i, band in enumerate(data_bands, start=1):
        dst.write(band.astype(DTYPE), i)

    # Set band descriptions
    dst.set_band_description(1, "Blue (0.45-0.51µm)")
    dst.set_band_description(2, "Green (0.53-0.59µm)")
    dst.set_band_description(3, "Red (0.64-0.67µm)")
    dst.set_band_description(4, "NIR (0.85-0.88µm)")
    dst.set_band_description(5, "Red Edge (0.70-0.74µm)")

    # Tambahkan metadata
    dst.update_tags(
        AREA="Persawahan Jawa Tengah",
        DESCRIPTION="Data dummy multispektral sawah realistik",
        GENERATED_BY="Python Synthetic Generator",
        APPLICATION="Agricultural Analysis"
    )

print("SELESAI! File dummy sawah realistik siap.")

# ================= VISUALISASI KOMPREHENSIF =================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Peta Vigor/Kesuburan
im1 = axes[0, 0].imshow(vigor_map, cmap='YlGn', vmin=0, vmax=1)
axes[0, 0].set_title('Peta Vigor/Kesuburan Vegetasi')
axes[0, 0].set_xlabel('Piksel')
axes[0, 0].set_ylabel('Piksel')
plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

# 2. False Color Infrared (NIR-Red-Green)
img_false_color = np.dstack((b4_nir, b3_red, b2_green))
img_false_color = img_false_color / np.percentile(img_false_color, 95)  # Normalisasi robust
img_false_color = np.clip(img_false_color, 0, 1)
axes[0, 1].imshow(img_false_color)
axes[0, 1].set_title('False Color Infrared (NIR-Red-Green)\nMerah = Vegetasi Sehat')
axes[0, 1].set_xlabel('Piksel')
axes[0, 1].set_ylabel('Piksel')

# 3. True Color (Red-Green-Blue)
img_true_color = np.dstack((b3_red, b2_green, b1_blue))
img_true_color = img_true_color / np.percentile(img_true_color, 95)
img_true_color = np.clip(img_true_color, 0, 1)
axes[0, 2].imshow(img_true_color)
axes[0, 2].set_title('True Color (Red-Green-Blue)')
axes[0, 2].set_xlabel('Piksel')
axes[0, 2].set_ylabel('Piksel')

# 4. NDVI
with np.errstate(divide='ignore', invalid='ignore'):
    ndvi = (b4_nir - b3_red) / (b4_nir + b3_red)
    ndvi = np.nan_to_num(ndvi, nan=0, posinf=0, neginf=0)

im4 = axes[1, 0].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
axes[1, 0].set_title('NDVI - Normalized Difference Vegetation Index')
axes[1, 0].set_xlabel('Piksel')
axes[1, 0].set_ylabel('Piksel')
plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)

# 5. GNDVI
with np.errstate(divide='ignore', invalid='ignore'):
    gndvi = (b4_nir - b2_green) / (b4_nir + b2_green)
    gndvi = np.nan_to_num(gndvi, nan=0, posinf=0, neginf=0)

im5 = axes[1, 1].imshow(gndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.7)
axes[1, 1].set_title('GNDVI - Green NDVI')
axes[1, 1].set_xlabel('Piksel')
axes[1, 1].set_ylabel('Piksel')
plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)

# 6. NDRE
with np.errstate(divide='ignore', invalid='ignore'):
    ndre = (b4_nir - b5_re) / (b4_nir + b5_re)
    ndre = np.nan_to_num(ndre, nan=0, posinf=0, neginf=0)

im6 = axes[1, 2].imshow(ndre, cmap='RdYlGn', vmin=-0.2, vmax=0.6)
axes[1, 2].set_title('NDRE - Normalized Difference Red Edge')
axes[1, 2].set_xlabel('Piksel')
axes[1, 2].set_ylabel('Piksel')
plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)

plt.tight_layout()
plt.savefig('visualisasi_sawah_realistik.png', dpi=150, bbox_inches='tight')
plt.show()

# ================= INFORMASI STATISTIK =================
print("\n=== STATISTIK DATA DUMMY ===")
print(f"Ukuran peta: {WIDTH} x {HEIGHT} piksel")
print(f"Resolusi: {PIXEL_SIZE * 111320:.1f} meter/piksel")  # Konversi derajat ke meter
print(f"Luas area: {(WIDTH * PIXEL_SIZE * 111320) * (HEIGHT * PIXEL_SIZE * 111320) / 10000:.2f} hektar")

print("\N{leaf fluttering in wind} STATISTIK VEGETASI:")
print(f"NDVI - Min: {ndvi.min():.3f}, Max: {ndvi.max():.3f}, Mean: {ndvi.mean():.3f}")
print(f"GNDVI - Min: {gndvi.min():.3f}, Max: {gndvi.max():.3f}, Mean: {gndvi.mean():.3f}")
print(f"NDRE - Min: {ndre.min():.3f}, Max: {ndre.max():.3f}, Mean: {ndre.mean():.3f}")

# Klasifikasi sederhana berdasarkan NDVI
veg_healthy = np.sum(ndvi > 0.6) / (WIDTH * HEIGHT) * 100
veg_moderate = np.sum((ndvi > 0.3) & (ndvi <= 0.6)) / (WIDTH * HEIGHT) * 100
veg_stressed = np.sum((ndvi > 0.1) & (ndvi <= 0.3)) / (WIDTH * HEIGHT) * 100
non_veg = np.sum(ndvi <= 0.1) / (WIDTH * HEIGHT) * 100

print(f"\N{herb} KLASIFIKASI VEGETASI:")
print(f"Sehat (NDVI > 0.6): {veg_healthy:.1f}%")
print(f"Sedang (NDVI 0.3-0.6): {veg_moderate:.1f}%")
print(f"Stres (NDVI 0.1-0.3): {veg_stressed:.1f}%")
print(f"Non-vegetasi (NDVI ≤ 0.1): {non_veg:.1f}%")

print(f"\nFile output: {NAMA_FILE}")
print("File visualisasi: visualisasi_sawah_realistik.png")