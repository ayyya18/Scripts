import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import rasterio
import os
import threading  # Agar aplikasi tidak macet (Not Responding) saat proses berat


class AplikasiMachineLearning:
    def __init__(self, root):
        self.root = root
        self.root.title("Agri-Tech: Analisis Serapan Hara (Random Forest)")
        self.root.geometry("700x600")

        # Variabel untuk menyimpan path file
        self.path_data_latih = tk.StringVar()
        self.path_foto_udara = tk.StringVar()
        self.model_path = "model_rf_hara.joblib"  # Lokasi simpan model default

        # --- MEMBUAT TAMPILAN (GUI) ---

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Judul Utama
        lbl_judul = ttk.Label(root, text="Sistem Cerdas Pemetaan Unsur Hara", font=("Helvetica", 16, "bold"))
        lbl_judul.pack(pady=10)

        # Tab Control (Menu Tab)
        tab_control = ttk.Notebook(root)

        self.tab_training = ttk.Frame(tab_control)
        self.tab_prediksi = ttk.Frame(tab_control)

        tab_control.add(self.tab_training, text='1. Pelatihan Model (Training)')
        tab_control.add(self.tab_prediksi, text='2. Prediksi Peta (GeoTIFF)')
        tab_control.pack(expand=1, fill="both", padx=10, pady=5)

        # === ISI TAB 1: TRAINING ===
        frame_tr = ttk.LabelFrame(self.tab_training, text="Input Data Latihan")
        frame_tr.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame_tr, text="Pilih File (.csv atau .xlsx):").pack(anchor="w", padx=5)

        frame_file_tr = ttk.Frame(frame_tr)
        frame_file_tr.pack(fill="x", padx=5, pady=5)

        entry_tr = ttk.Entry(frame_file_tr, textvariable=self.path_data_latih, width=50)
        entry_tr.pack(side="left", padx=5)
        ttk.Button(frame_file_tr, text="Browse", command=self.browse_data_latih).pack(side="left")

        btn_train = ttk.Button(self.tab_training, text="MULAI TRAINING MODEL", command=self.start_training_thread)
        btn_train.pack(pady=10)

        # === ISI TAB 2: PREDIKSI ===
        frame_pr = ttk.LabelFrame(self.tab_prediksi, text="Input Foto Udara")
        frame_pr.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame_pr, text="Pilih File GeoTIFF (.tif):").pack(anchor="w", padx=5)

        frame_file_pr = ttk.Frame(frame_pr)
        frame_file_pr.pack(fill="x", padx=5, pady=5)

        entry_pr = ttk.Entry(frame_file_pr, textvariable=self.path_foto_udara, width=50)
        entry_pr.pack(side="left", padx=5)
        ttk.Button(frame_file_pr, text="Browse", command=self.browse_foto_udara).pack(side="left")

        btn_predict = ttk.Button(self.tab_prediksi, text="PROSES PETAAKAN WILAYAH",
                                 command=self.start_prediction_thread)
        btn_predict.pack(pady=10)

        # === LOG WINDOW (Jendela Status) ===
        lbl_log = ttk.Label(root, text="Log Proses:", font=("Arial", 10, "bold"))
        lbl_log.pack(anchor="w", padx=10)

        self.log_area = scrolledtext.ScrolledText(root, width=80, height=15, state='disabled')
        self.log_area.pack(padx=10, pady=5)

    # --- FUNGSI LOGIKA ---

    def log(self, text):
        """Menampilkan teks di kotak log"""
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, text + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def browse_data_latih(self):
        filename = filedialog.askopenfilename(filetypes=[("Data Excel/CSV", "*.csv *.xlsx")])
        self.path_data_latih.set(filename)

    def browse_foto_udara(self):
        filename = filedialog.askopenfilename(filetypes=[("GeoTIFF", "*.tif *.tiff")])
        self.path_foto_udara.set(filename)

    # --- THREADING (Agar GUI tidak macet) ---
    def start_training_thread(self):
        threading.Thread(target=self.proses_training).start()

    def start_prediction_thread(self):
        threading.Thread(target=self.proses_prediksi).start()

    # --- PROSES INTI ---
    def proses_training(self):
        path = self.path_data_latih.get()
        if not path:
            messagebox.showwarning("Peringatan", "Pilih file data latihan dulu!")
            return

        self.log("-" * 40)
        self.log(f"Memulai Training dengan data: {os.path.basename(path)}")

        try:
            # 1. Baca Data
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            self.log(f"Data dimuat. Jumlah baris: {len(df)}")
            self.log(f"Kolom ditemukan: {list(df.columns)}")

            # Validasi Kolom (Sesuaikan dengan nama kolom Excel Anda)
            required_cols = ['NDVI', 'NDRE', 'GNDVI', 'Serapan_K']
            if not all(col in df.columns for col in required_cols):
                self.log(f"ERROR: Kolom wajib tidak lengkap! Harus ada: {required_cols}")
                return

            X = df[['NDVI', 'NDRE', 'GNDVI']]
            y = df['Serapan_K']

            # 2. Training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

            self.log("Sedang melatih Random Forest...")
            rf_model.fit(X_train, y_train)

            # 3. Evaluasi
            y_pred = rf_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            self.log(f"Training Selesai! R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")

            # 4. Simpan Model
            joblib.dump(rf_model, self.model_path)
            self.log(f"Model berhasil disimpan sebagai: {self.model_path}")
            messagebox.showinfo("Sukses", "Model berhasil dilatih dan disimpan!")

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", str(e))

    def proses_prediksi(self):
        path_tif = self.path_foto_udara.get()
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", "Model belum ada! Lakukan Training dulu.")
            return
        if not path_tif:
            messagebox.showwarning("Peringatan", "Pilih file GeoTIFF dulu!")
            return

        self.log("-" * 40)
        self.log(f"Memulai Prediksi pada peta: {os.path.basename(path_tif)}")

        try:
            model = joblib.load(self.model_path)
            self.log("Model berhasil dimuat.")

            # Lokasi Simpan Output
            output_path = path_tif.replace(".tif", "_PREDIKSI_K.tif")

            with rasterio.open(path_tif) as src:
                profile = src.profile
                self.log(f"Dimensi Peta: {src.width} x {src.height} pixel")

                # Baca Band (Sesuaikan Index Band kamera Anda disini!)
                # Asumsi: 2=Green, 3=Red, 4=NIR, 5=RedEdge
                green = src.read(2).astype('float32')
                red = src.read(3).astype('float32')
                nir = src.read(4).astype('float32')
                re = src.read(5).astype('float32')  # RedEdge

                # Handling Background (Nilai 0)
                nir[nir == 0] = np.nan

                # Hitung Indeks
                self.log("Menghitung Indeks Vegetasi...")
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi = (nir - red) / (nir + red)
                    ndre = (nir - re) / (nir + re)
                    gndvi = (nir - green) / (nir + green)

                # Persiapan Data
                df_pred = np.column_stack((ndvi.flatten(), ndre.flatten(), gndvi.flatten()))
                mask_valid = ~np.isnan(df_pred).any(axis=1)
                hasil_flat = np.full(df_pred.shape[0], np.nan)

                self.log("Melakukan prediksi pixel (tunggu sebentar)...")
                if np.sum(mask_valid) > 0:
                    hasil_flat[mask_valid] = model.predict(df_pred[mask_valid])

                hasil_map = hasil_flat.reshape(src.height, src.width)

                # Simpan
                profile.update(dtype=rasterio.float32, count=1)
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(hasil_map.astype(rasterio.float32), 1)

            self.log(f"SUKSES! Peta tersimpan di: {output_path}")
            messagebox.showinfo("Sukses", f"Prediksi selesai!\nFile: {os.path.basename(output_path)}")

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = AplikasiMachineLearning(root)
    root.mainloop()