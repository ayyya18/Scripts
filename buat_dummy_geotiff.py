import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt

NAMA_FILE_DUMMY = 'orthomosaic_multispectral.tif'
WIDTH = 500
HEIGHT = 500
NUM_BANDS = 5
DTYPE = rasterio.float32

WEST = 110.4
NORTH = -7.5
PIXEL_SIZE = 0.0001

def buat_pola_vegetasi(height, width):
    """
    Membuat pola gradien diagonal.
    Nilai 0 (kiri-atas) = Stress/Tanah, Nilai 1 (kanan-bawah) = Subur.
    """
    y = np.linspace(0, 1, height)
    x = np.linspace(0, 1, width)
    X, Y = np.meshgrid(x, y)
    base_pattern = (X + Y) / 2 + np.random.normal(0, 0.05, (height, width))
    base_pattern = np.clip(base_pattern, 0, 1)
    return base_pattern

print("Sedang membuat data dummy multispektral...")

vigor_map = buat_pola_vegetasi(HEIGHT, WIDTH)
band1_blue = np.clip(0.1 + np.random.normal(0, 0.02, (HEIGHT, WIDTH)), 0, 1)
band2_green = np.clip(0.1 + (vigor_map * 0.1) + np.random.normal(0, 0.02, (HEIGHT, WIDTH)), 0, 1)
band3_red = np.clip(0.3 - (vigor_map * 0.25) + np.random.normal(0, 0.02, (HEIGHT, WIDTH)), 0, 1)
band4_nir = np.clip(0.2 + (vigor_map * 0.5) + np.random.normal(0, 0.03, (HEIGHT, WIDTH)), 0, 1)
band5_re = np.clip(0.15 + (vigor_map * 0.3) + np.random.normal(0, 0.02, (HEIGHT, WIDTH)), 0, 1)
dummy_data = np.stack([band1_blue, band2_green, band3_red, band4_nir, band5_re]).astype(DTYPE)
transform = from_origin(WEST, NORTH, PIXEL_SIZE, PIXEL_SIZE)
crs = rasterio.crs.CRS.from_epsg(4326)

meta = {
    'driver': 'GTiff',
    'height': HEIGHT,
    'width': WIDTH,
    'count': NUM_BANDS,
    'dtype': DTYPE,
    'crs': crs,
    'transform': transform,
    'nodata': None,
    'compress': 'lzw'
}

print(f"Menulis file ke: {NAMA_FILE_DUMMY}...")
with rasterio.open(NAMA_FILE_DUMMY, 'w', **meta) as dst:
    for id, band_data in enumerate(dummy_data, start=1):
        dst.write(band_data, id)
        dst.set_band_description(id, f"Band {id}")

print("SELESAI. File dummy siap digunakan.")

plt.figure(figsize=(8, 8))
false_color = np.stack([band4_nir, band3_red, band2_green], axis=-1)
false_color = np.clip(false_color * 1.5, 0, 1)
plt.imshow(false_color)
plt.title("Preview Dummy False Color (R=NIR, G=Red, B=Green)\nKiri-Atas: Stress, Kanan-Bawah: Subur")
plt.axis('off')
plt.show()