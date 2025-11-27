import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Toplevel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import rasterio
import os
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Agar resolusi tinggi di layar tajam (Windows 10/11)
try:
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


class AplikasiMachineLearning:
    def __init__(self, root):
        self.root = root
        self.root.title("Agri-Tech: Sistem Analisis Hara & Pemetaan")
        self.root.geometry("900x700")

        # Variabel
        self.path_data_latih = tk.StringVar()
        self.path_foto_udara = tk.StringVar()
        self.model_path = "model_rf_hara.joblib"
        self.df_data = None  # Menyimpan data yang diupload

        # --- STYLE ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", rowheight=25)
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))

        # --- JUDUL ---
        lbl_judul = ttk.Label(root, text="Sistem Cerdas Pemetaan Unsur Hara", font=("Helvetica", 18, "bold"))
        lbl_judul.pack(pady=10)

        # --- TAB CONTROL ---
        tab_control = ttk.Notebook(root)
        self.tab_training = ttk.Frame(tab_control)
        self.tab_prediksi = ttk.Frame(tab_control)

        tab_control.add(self.tab_training, text='  1. Pelatihan Model & Analisis Data  ')
        tab_control.add(self.tab_prediksi, text='  2. Prediksi Peta (GeoTIFF)  ')
        tab_control.pack(expand=1, fill="both", padx=10, pady=5)

        # =================================================================
        # TAB 1: TRAINING & ANALISIS
        # =================================================================
        frame_tr = ttk.LabelFrame(self.tab_training, text="Input Data Latihan (.csv / .xlsx)")
        frame_tr.pack(fill="x", padx=10, pady=10)

        # Input File
        frame_file_tr = ttk.Frame(frame_tr)
        frame_file_tr.pack(fill="x", padx=5, pady=5)

        entry_tr = ttk.Entry(frame_file_tr, textvariable=self.path_data_latih, font=("Arial", 10))
        entry_tr.pack(side="left", fill="x", expand=True, padx=5)

        btn_browse = ttk.Button(frame_file_tr, text="📂 Cari File", command=self.browse_data_latih)
        btn_browse.pack(side="left", padx=5)

        btn_train = ttk.Button(frame_file_tr, text="🚀 MULAI TRAINING", command=self.start_training_thread)
        btn_train.pack(side="left", padx=5)

        # PREVIEW TABLE (DATA INPUT)
        lbl_preview = ttk.Label(self.tab_training, text="Preview Data Input:", font=("Arial", 10, "bold"))
        lbl_preview.pack(anchor="w", padx=15, pady=(10, 0))

        frame_table = ttk.Frame(self.tab_training)
        frame_table.pack(fill="both", expand=True, padx=10, pady=5)

        # Treeview untuk tabel
        self.tree_input = ttk.Treeview(frame_table, show='headings')

        # Scrollbars
        scroll_y = ttk.Scrollbar(frame_table, orient="vertical", command=self.tree_input.yview)
        scroll_x = ttk.Scrollbar(frame_table, orient="horizontal", command=self.tree_input.xview)
        self.tree_input.configure(yscroll=scroll_y.set, xscroll=scroll_x.set)

        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        self.tree_input.pack(fill="both", expand=True)

        # Log Area (Kecil di bawah)
        frame_log = ttk.LabelFrame(self.tab_training, text="Log Proses")
        frame_log.pack(fill="x", padx=10, pady=5)
        self.log_area = scrolledtext.ScrolledText(frame_log, height=6, font=("Consolas", 9))
        self.log_area.pack(fill="both", padx=5, pady=5)

        # =================================================================
        # TAB 2: PREDIKSI (Sama seperti sebelumnya)
        # =================================================================
        frame_pr = ttk.LabelFrame(self.tab_prediksi, text="Input Foto Udara (.tif)")
        frame_pr.pack(fill="x", padx=10, pady=10)

        entry_pr = ttk.Entry(frame_pr, textvariable=self.path_foto_udara, width=50)
        entry_pr.pack(side="left", padx=10, pady=10, fill='x', expand=True)
        ttk.Button(frame_pr, text="📂 Browse", command=self.browse_foto_udara).pack(side="left", padx=10)

        btn_predict = ttk.Button(self.tab_prediksi, text="⚙️ PROSES PETAAKAN WILAYAH",
                                 command=self.start_prediction_thread)
        btn_predict.pack(pady=10)

        self.log_pred = scrolledtext.ScrolledText(self.tab_prediksi, height=15)
        self.log_pred.pack(fill="both", padx=10, pady=10, expand=True)

    # --- FUNGSI PENDUKUNG ---

    def log(self, text):
        self.log_area.insert(tk.END, f">> {text}\n")
        self.log_area.see(tk.END)

    def log_p(self, text):
        self.log_pred.insert(tk.END, f">> {text}\n")
        self.log_pred.see(tk.END)

    def load_table(self, tree, df):
        """Fungsi helper untuk mengisi Treeview dari DataFrame"""
        tree.delete(*tree.get_children())  # Bersihkan tabel lama
        tree["columns"] = list(df.columns)

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")

        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))

    def browse_data_latih(self):
        filename = filedialog.askopenfilename(filetypes=[("Data Table", "*.csv *.xlsx")])
        if filename:
            self.path_data_latih.set(filename)
            try:
                if filename.endswith('.csv'):
                    self.df_data = pd.read_csv(filename)
                else:
                    self.df_data = pd.read_excel(filename)

                # Tampilkan di tabel preview
                self.load_table(self.tree_input, self.df_data)
                self.log(f"File dimuat: {os.path.basename(filename)} ({len(self.df_data)} baris)")

            except Exception as e:
                messagebox.showerror("Error", f"Gagal membaca file: {e}")

    def browse_foto_udara(self):
        filename = filedialog.askopenfilename(filetypes=[("GeoTIFF", "*.tif *.tiff")])
        self.path_foto_udara.set(filename)

    # --- LOGIKA TRAINING & HASIL ---

    def start_training_thread(self):
        threading.Thread(target=self.proses_training).start()

    def proses_training(self):
        if self.df_data is None:
            messagebox.showwarning("Warning", "Silakan pilih file data terlebih dahulu!")
            return

        self.log("Memulai proses training...")

        try:
            df = self.df_data

            # Cek kolom
            target_col = 'Serapan_K'
            if target_col not in df.columns:
                messagebox.showerror("Error", f"Kolom '{target_col}' tidak ditemukan di data!")
                return

            # Pisahkan X dan y
            X = df.drop(columns=[target_col])
            # Hapus kolom non-numerik (misal kolom 'No' atau 'Nama') jika ada
            X = X.select_dtypes(include=[np.number])
            y = df[target_col]

            self.log(f"Fitur: {list(X.columns)}")

            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)

            # Evaluasi
            y_pred = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            self.log(f"Training Selesai! R2: {r2:.4f}, RMSE: {rmse:.4f}")
            joblib.dump(rf, self.model_path)

            # --- MENAMPILKAN POPUP HASIL ---
            self.root.after(0, lambda: self.show_result_window(rf, X.columns, y_test, y_pred, r2, rmse))

        except Exception as e:
            self.log(f"Error: {e}")
            messagebox.showerror("Error Training", str(e))

    def show_result_window(self, model, feature_names, y_test, y_pred, r2, rmse):
        """Membuat jendela baru untuk menampilkan tabel hasil"""
        win = Toplevel(self.root)
        win.title("Hasil Analisis & Regresi")
        win.geometry("1000x600")

        # Tab di dalam jendela hasil
        tabs = ttk.Notebook(win)
        tab_metrics = ttk.Frame(tabs)
        tab_importance = ttk.Frame(tabs)
        tab_prediction = ttk.Frame(tabs)

        tabs.add(tab_metrics, text='Ringkasan & Grafik')
        tabs.add(tab_importance, text='Feature Importance (Pengaruh Variabel)')
        tabs.add(tab_prediction, text='Tabel Prediksi vs Aktual')
        tabs.pack(expand=True, fill="both", padx=10, pady=10)

        # --- TAB 1: METRICS & PLOT ---
        lbl_res = ttk.Label(tab_metrics,
                            text=f"Akurasi Model (R-Squared): {r2:.4f}\nError Rata-rata (RMSE): {rmse:.4f}",
                            font=("Arial", 14, "bold"), foreground="blue")
        lbl_res.pack(pady=10)

        # Embed Matplotlib Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # Plot Scatter
        ax[0].scatter(y_test, y_pred, color='blue', alpha=0.6)
        ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax[0].set_xlabel("Aktual")
        ax[0].set_ylabel("Prediksi")
        ax[0].set_title("Aktual vs Prediksi")

        # Plot Bar Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        ax[1].bar(range(len(importances)), importances[indices], align="center")
        ax[1].set_xticks(range(len(importances)))
        ax[1].set_xticklabels([feature_names[i] for i in indices], rotation=45)
        ax[1].set_title("Feature Importance")

        canvas = FigureCanvasTkAgg(fig, master=tab_metrics)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- TAB 2: FEATURE IMPORTANCE TABLE ---
        tree_imp = ttk.Treeview(tab_importance, columns=("Fitur", "Score"), show='headings')
        tree_imp.heading("Fitur", text="Nama Variabel (Indeks)")
        tree_imp.heading("Score", text="Tingkat Kepentingan (0-1)")
        tree_imp.pack(fill="both", expand=True)

        # Isi data importance
        df_imp = pd.DataFrame({'Fitur': feature_names, 'Score': model.feature_importances_})
        df_imp = df_imp.sort_values(by='Score', ascending=False)
        for _, row in df_imp.iterrows():
            tree_imp.insert("", "end", values=(row['Fitur'], f"{row['Score']:.4f}"))

        # --- TAB 3: PREDICTION TABLE ---
        tree_pred = ttk.Treeview(tab_prediction, columns=("No", "Aktual", "Prediksi", "Selisih"), show='headings')
        tree_pred.heading("No", text="No Sampel")
        tree_pred.heading("Aktual", text="Nilai Aktual (Lab)")
        tree_pred.heading("Prediksi", text="Prediksi Model")
        tree_pred.heading("Selisih", text="Selisih (Error)")
        tree_pred.pack(fill="both", expand=True)

        # Isi data prediksi
        y_test_arr = np.array(y_test)
        for i in range(len(y_test)):
            act = y_test_arr[i]
            pred = y_pred[i]
            diff = abs(act - pred)
            tree_pred.insert("", "end", values=(i + 1, f"{act:.2f}", f"{pred:.2f}", f"{diff:.2f}"))

    def start_prediction_thread(self):
        threading.Thread(target=self.proses_prediksi).start()

    def proses_prediksi(self):
        # (Logika prediksi sama dengan kode sebelumnya, disederhanakan untuk ringkas)
        path_tif = self.path_foto_udara.get()
        if not path_tif or not os.path.exists(self.model_path):
            messagebox.showerror("Error", "Data/Model belum siap.")
            return

        try:
            self.log_p("Memuat model & Membaca GeoTIFF...")
            model = joblib.load(self.model_path)

            with rasterio.open(path_tif) as src:
                profile = src.profile
                # Asumsi urutan band: Green=2, Red=3, NIR=4, RE=5
                green = src.read(2).astype('float32')
                red = src.read(3).astype('float32')
                nir = src.read(4).astype('float32')
                re = src.read(5).astype('float32')

                # Hitung Indeks
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi = (nir - red) / (nir + red)
                    ndre = (nir - re) / (nir + re)
                    gndvi = (nir - green) / (nir + green)

                # Prediksi
                df_pred = np.column_stack((ndvi.flatten(), ndre.flatten(), gndvi.flatten()))
                mask = ~np.isnan(df_pred).any(axis=1)
                res = np.full(df_pred.shape[0], np.nan)

                if mask.sum() > 0:
                    res[mask] = model.predict(df_pred[mask])

                # Simpan
                out_path = path_tif.replace(".tif", "_PREDIKSI.tif")
                profile.update(dtype=rasterio.float32, count=1)
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(res.reshape(src.height, src.width).astype(rasterio.float32), 1)

            self.log_p(f"Selesai! Disimpan di: {out_path}")
            messagebox.showinfo("Sukses", "Peta berhasil dibuat!")

        except Exception as e:
            self.log_p(f"Error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AplikasiMachineLearning(root)
    root.mainloop()