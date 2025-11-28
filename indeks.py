import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


class MultiBandApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Agri-Tech: Multi-File Band Calculator")
        self.root.geometry("1000x750")

        # Variabel Path File (Setiap band punya variabel sendiri)
        self.path_red = tk.StringVar()
        self.path_green = tk.StringVar()
        self.path_nir = tk.StringVar()
        self.path_re = tk.StringVar()

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Header
        lbl_title = ttk.Label(root, text="Pengolahan Band Terpisah (Separate Files)", font=("Segoe UI", 16, "bold"))
        lbl_title.pack(pady=10)

        lbl_desc = ttk.Label(root, text="Masukkan file .tif yang sesuai untuk masing-masing kanal warna di bawah ini:",
                             foreground="#555")
        lbl_desc.pack(pady=(0, 10))

        # === BAGIAN 1: INPUT FILE TERPISAH ===
        frame_input = ttk.LabelFrame(root, text="1. Input File Per Band")
        frame_input.pack(fill="x", padx=10, pady=5)

        # Grid Layout untuk Input
        # --- Row 0: Red ---
        ttk.Label(frame_input, text="File Band MERAH (Red):", font=("Arial", 9, "bold")).grid(row=0, column=0,
                                                                                              sticky="e", padx=5,
                                                                                              pady=5)
        ttk.Entry(frame_input, textvariable=self.path_red, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(frame_input, text="📂 Browse Red", command=lambda: self.browse_file(self.path_red)).grid(row=0,
                                                                                                           column=2,
                                                                                                           padx=5)

        # --- Row 1: Green ---
        ttk.Label(frame_input, text="File Band HIJAU (Green):", font=("Arial", 9, "bold")).grid(row=1, column=0,
                                                                                                sticky="e", padx=5,
                                                                                                pady=5)
        ttk.Entry(frame_input, textvariable=self.path_green, width=60).grid(row=1, column=1, padx=5)
        ttk.Button(frame_input, text="📂 Browse Green", command=lambda: self.browse_file(self.path_green)).grid(row=1,
                                                                                                               column=2,
                                                                                                               padx=5)

        # --- Row 2: NIR ---
        ttk.Label(frame_input, text="File Band NIR (Infrared):", font=("Arial", 9, "bold")).grid(row=2, column=0,
                                                                                                 sticky="e", padx=5,
                                                                                                 pady=5)
        ttk.Entry(frame_input, textvariable=self.path_nir, width=60).grid(row=2, column=1, padx=5)
        ttk.Button(frame_input, text="📂 Browse NIR", command=lambda: self.browse_file(self.path_nir)).grid(row=2,
                                                                                                           column=2,
                                                                                                           padx=5)

        # --- Row 3: RedEdge ---
        ttk.Label(frame_input, text="File Band RED-EDGE:", font=("Arial", 9, "bold")).grid(row=3, column=0, sticky="e",
                                                                                           padx=5, pady=5)
        ttk.Entry(frame_input, textvariable=self.path_re, width=60).grid(row=3, column=1, padx=5)
        ttk.Button(frame_input, text="📂 Browse RedEdge", command=lambda: self.browse_file(self.path_re)).grid(row=3,
                                                                                                              column=2,
                                                                                                              padx=5)

        # === BAGIAN 2: PILIH INDEKS ===
        frame_action = ttk.LabelFrame(root, text="2. Pilih Indeks & Proses")
        frame_action.pack(fill="x", padx=10, pady=10)

        self.selected_index = tk.StringVar(value="NDVI")

        ttk.Radiobutton(frame_action, text="NDVI (Butuh: NIR + Red)", variable=self.selected_index, value="NDVI").pack(
            side="left", padx=20)
        ttk.Radiobutton(frame_action, text="NDRE (Butuh: NIR + RedEdge)", variable=self.selected_index,
                        value="NDRE").pack(side="left", padx=20)
        ttk.Radiobutton(frame_action, text="GNDVI (Butuh: NIR + Green)", variable=self.selected_index,
                        value="GNDVI").pack(side="left", padx=20)

        btn_process = ttk.Button(frame_action, text="⚙️ HITUNG INDEKS", command=self.process_multiband)
        btn_process.pack(side="right", padx=15, pady=10)

        # === BAGIAN 3: PREVIEW ===
        self.frame_preview = ttk.LabelFrame(root, text="Preview Hasil")
        self.frame_preview.pack(fill="both", expand=True, padx=10, pady=5)

        lbl_info = ttk.Label(self.frame_preview,
                             text="Tips: Pastikan semua file memiliki ukuran dimensi (pixel) yang sama.")
        lbl_info.pack(pady=10)

    # --- FUNGSI LOGIKA ---

    def browse_file(self, string_var):
        """Helper function untuk mengisi variable path"""
        filename = filedialog.askopenfilename(filetypes=[("GeoTIFF", "*.tif *.tiff")])
        if filename:
            string_var.set(filename)

    def process_multiband(self):
        idx_type = self.selected_index.get()

        # 1. Cek Kelengkapan File berdasarkan rumus
        p_nir = self.path_nir.get()
        p_red = self.path_red.get()
        p_green = self.path_green.get()
        p_re = self.path_re.get()

        if not p_nir:
            messagebox.showerror("Error", "Band NIR wajib diisi untuk semua indeks!")
            return

        target_path = ""
        target_name = ""

        if idx_type == "NDVI":
            if not p_red:
                messagebox.showerror("Error", "Untuk NDVI, Band MERAH wajib diisi!")
                return
            target_path = p_red
            target_name = "Red"

        elif idx_type == "NDRE":
            if not p_re:
                messagebox.showerror("Error", "Untuk NDRE, Band RED-EDGE wajib diisi!")
                return
            target_path = p_re
            target_name = "RedEdge"

        elif idx_type == "GNDVI":
            if not p_green:
                messagebox.showerror("Error", "Untuk GNDVI, Band HIJAU wajib diisi!")
                return
            target_path = p_green
            target_name = "Green"

        # 2. Proses Perhitungan
        try:
            # Buka File NIR (Utama)
            with rasterio.open(p_nir) as src_nir:
                arr_nir = src_nir.read(1).astype('float32')  # Biasanya file terpisah hanya punya 1 band
                meta = src_nir.meta
                shape_nir = src_nir.shape

            # Buka File Target (Red/Green/RE)
            with rasterio.open(target_path) as src_target:
                arr_target = src_target.read(1).astype('float32')
                shape_target = src_target.shape

            # 3. Validasi Ukuran (PENTING!)
            if shape_nir != shape_target:
                messagebox.showerror("Ukuran Tidak Cocok",
                                     f"Dimensi gambar tidak sama!\nNIR: {shape_nir}\n{target_name}: {shape_target}\n\nPastikan foto sudah di-align/rectify.")
                return

            # 4. Hitung Rumus
            np.seterr(divide='ignore', invalid='ignore')

            numerator = arr_nir - arr_target
            denominator = arr_nir + arr_target

            denominator[denominator == 0] = np.nan
            result = numerator / denominator

            # 5. Simpan Hasil
            # Kita gunakan nama file NIR sebagai dasar nama output
            output_path = p_nir.replace(".tif", f"_{idx_type}_Result.tif")

            # Update metadata (karena outputnya float 1 band)
            meta.update(dtype=rasterio.float32, count=1, compress='lzw')

            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(result.astype(rasterio.float32), 1)

            messagebox.showinfo("Sukses", f"Berhasil!\nFile disimpan di: {os.path.basename(output_path)}")

            # 6. Tampilkan Preview
            self.show_preview(result, idx_type)

        except Exception as e:
            messagebox.showerror("Error System", f"Terjadi kesalahan:\n{str(e)}")

    def show_preview(self, data, title):
        for widget in self.frame_preview.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 4))

        # Plotting
        im = ax.imshow(data, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(f"Peta {title} (Dari File Terpisah)")
        ax.axis('off')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Nilai Indeks")

        canvas = FigureCanvasTkAgg(fig, master=self.frame_preview)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiBandApp(root)
    root.mainloop()