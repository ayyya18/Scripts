import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
import seaborn as sns


class AgriApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Training & Validasi Model Hara Kalium (K)")
        self.root.geometry("1400x900")  # Diperbesar untuk menampung lebih banyak kolom
        self.root.state('zoomed')

        # Model paths
        self.model_rf_path = "model_rf_hara_final.joblib"
        self.model_ndvi_path = "model_ndvi.joblib"
        self.model_ndre_path = "model_ndre.joblib"
        self.model_gndvi_path = "model_gndvi.joblib"

        # Variabel Path
        self.path_train = tk.StringVar()
        self.path_input_pred = tk.StringVar()
        self.path_actual = tk.StringVar()

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Header
        ttk.Label(root,
                  text="Workflow: Training Model -> Simpan Model -> Validasi Eksternal -> Bandingkan Akurasi Indeks",
                  font=("Segoe UI", 14, "bold")).pack(pady=10)

        # Tabs
        tabs = ttk.Notebook(root)
        self.tab_train = ttk.Frame(tabs)
        self.tab_predict = ttk.Frame(tabs)
        self.tab_compare = ttk.Frame(tabs)  # New tab for comparison

        tabs.add(self.tab_train, text='  1. Training & Analisis Model  ')
        tabs.add(self.tab_predict, text='  2. Prediksi & Validasi  ')
        tabs.add(self.tab_compare, text='  3. Perbandingan Akurasi Indeks  ')
        tabs.pack(expand=True, fill="both", padx=10, pady=5)

        self.setup_tab_training()
        self.setup_tab_prediction()
        self.setup_tab_comparison()

    # =========================================================================
    # TAB 1: TRAINING
    # =========================================================================
    def setup_tab_training(self):
        # Frame Input
        frame_in = ttk.LabelFrame(self.tab_train, text="Input Data Training (File 1: 1_data_training.csv)")
        frame_in.pack(fill="x", padx=10, pady=10)

        ttk.Entry(frame_in, textvariable=self.path_train, width=80).pack(side="left", fill="x", expand=True, padx=5,
                                                                         pady=5)
        ttk.Button(frame_in, text="📂 Buka File Training",
                   command=lambda: self.browse(self.path_train)).pack(side="left", padx=5)

        # Tombol Aksi
        btn_frame = ttk.Frame(self.tab_train)
        btn_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(btn_frame, text="🚀 LATIH SEMUA MODEL & ANALISIS INDEKS",
                   command=self.process_training).pack(side="left", pady=10)

        # Area Hasil (Grafik & Log)
        self.frame_results_train = ttk.Frame(self.tab_train)
        self.frame_results_train.pack(fill="both", expand=True, padx=10)

        # Split kiri (Log) kanan (Grafik)
        self.log_train = scrolledtext.ScrolledText(self.frame_results_train, width=50, height=20)
        self.log_train.pack(side="left", fill="both", padx=5, pady=5)

        self.frame_graph_train = ttk.Frame(self.frame_results_train)
        self.frame_graph_train.pack(side="right", fill="both", expand=True, padx=5)

    # =========================================================================
    # TAB 2: PREDIKSI & VALIDASI
    # =========================================================================
    def setup_tab_prediction(self):
        # Frame 1: Input Data Baru
        frame_new = ttk.LabelFrame(self.tab_predict,
                                   text="A. Input Data Yang Akan Diduga (File 2: 2_data_input_prediksi.csv)")
        frame_new.pack(fill="x", padx=10, pady=5)
        ttk.Entry(frame_new, textvariable=self.path_input_pred, width=80).pack(side="left", fill="x", expand=True,
                                                                               padx=5)
        ttk.Button(frame_new, text="📂 Buka File Input Prediksi",
                   command=lambda: self.browse(self.path_input_pred)).pack(side="left")

        # Frame 2: Input Kunci Jawaban
        frame_ans = ttk.LabelFrame(self.tab_predict,
                                   text="B. Input Data Aktual / Kunci Jawaban (File 3: 3_data_aktual_jawaban.csv)")
        frame_ans.pack(fill="x", padx=10, pady=5)
        ttk.Entry(frame_ans, textvariable=self.path_actual, width=80).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(frame_ans, text="📂 Buka File Jawaban Aktual",
                   command=lambda: self.browse(self.path_actual)).pack(side="left")

        # Tombol Eksekusi
        ttk.Button(self.tab_predict, text="⚙️ JALANKAN PREDIKSI & HITUNG AKURASI SEMUA MODEL",
                   command=self.process_validation).pack(pady=10)

        # Frame untuk hasil prediksi
        result_frame = ttk.Frame(self.tab_predict)
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Notebook untuk menampilkan hasil prediksi masing-masing model
        self.prediction_notebook = ttk.Notebook(result_frame)
        self.prediction_notebook.pack(fill="both", expand=True)

        # Frame untuk masing-masing model
        self.frame_rf = ttk.Frame(self.prediction_notebook)
        self.frame_ndvi = ttk.Frame(self.prediction_notebook)
        self.frame_ndre = ttk.Frame(self.prediction_notebook)
        self.frame_gndvi = ttk.Frame(self.prediction_notebook)

        self.prediction_notebook.add(self.frame_rf, text="Model Gabungan (RF)")
        self.prediction_notebook.add(self.frame_ndvi, text="Model NDVI")
        self.prediction_notebook.add(self.frame_ndre, text="Model NDRE")
        self.prediction_notebook.add(self.frame_gndvi, text="Model GNDVI")

        # Setup tabel untuk masing-masing model
        self.setup_prediction_tables()

    def setup_prediction_tables(self):
        # Tabel untuk Model Gabungan (RF)
        header_frame_rf = ttk.Frame(self.frame_rf)
        header_frame_rf.pack(fill="x", padx=5, pady=5)

        ttk.Label(header_frame_rf, text="Hasil Prediksi Model Gabungan (Random Forest):",
                  font=("Arial", 10, "bold")).pack(anchor="w")

        self.lbl_mae_rf = ttk.Label(header_frame_rf, text="Rata-rata Selisih: -",
                                    font=("Arial", 9, "bold"), foreground="blue")
        self.lbl_mae_rf.pack(anchor="w", pady=2)

        tree_frame_rf = ttk.Frame(self.frame_rf)
        tree_frame_rf.pack(fill="both", expand=True, pady=5)

        tree_scroll_rf_y = ttk.Scrollbar(tree_frame_rf)
        tree_scroll_rf_y.pack(side="right", fill="y")

        tree_scroll_rf_x = ttk.Scrollbar(tree_frame_rf, orient="horizontal")
        tree_scroll_rf_x.pack(side="bottom", fill="x")

        self.tree_rf = ttk.Treeview(tree_frame_rf,
                                    yscrollcommand=tree_scroll_rf_y.set,
                                    xscrollcommand=tree_scroll_rf_x.set,
                                    columns=("No", "NDVI", "NDRE", "GNDVI", "Prediksi", "Aktual", "Selisih"),
                                    show='headings', height=12)

        tree_scroll_rf_y.config(command=self.tree_rf.yview)
        tree_scroll_rf_x.config(command=self.tree_rf.xview)

        cols_rf = ["No", "NDVI", "NDRE", "GNDVI", "Prediksi", "Aktual", "Selisih"]
        for col in cols_rf:
            self.tree_rf.heading(col, text=col)
            self.tree_rf.column(col, width=100, anchor="center")
        self.tree_rf.pack(fill="both", expand=True)

        # Tabel untuk Model NDVI
        header_frame_ndvi = ttk.Frame(self.frame_ndvi)
        header_frame_ndvi.pack(fill="x", padx=5, pady=5)

        ttk.Label(header_frame_ndvi, text="Hasil Prediksi Model NDVI:",
                  font=("Arial", 10, "bold")).pack(anchor="w")

        self.lbl_mae_ndvi = ttk.Label(header_frame_ndvi, text="Rata-rata Selisih: -",
                                      font=("Arial", 9, "bold"), foreground="blue")
        self.lbl_mae_ndvi.pack(anchor="w", pady=2)

        tree_frame_ndvi = ttk.Frame(self.frame_ndvi)
        tree_frame_ndvi.pack(fill="both", expand=True, pady=5)

        tree_scroll_ndvi_y = ttk.Scrollbar(tree_frame_ndvi)
        tree_scroll_ndvi_y.pack(side="right", fill="y")

        tree_scroll_ndvi_x = ttk.Scrollbar(tree_frame_ndvi, orient="horizontal")
        tree_scroll_ndvi_x.pack(side="bottom", fill="x")

        self.tree_ndvi = ttk.Treeview(tree_frame_ndvi,
                                      yscrollcommand=tree_scroll_ndvi_y.set,
                                      xscrollcommand=tree_scroll_ndvi_x.set,
                                      columns=("No", "NDVI", "Prediksi", "Aktual", "Selisih"),
                                      show='headings', height=12)

        tree_scroll_ndvi_y.config(command=self.tree_ndvi.yview)
        tree_scroll_ndvi_x.config(command=self.tree_ndvi.xview)

        cols_ndvi = ["No", "NDVI", "Prediksi", "Aktual", "Selisih"]
        for col in cols_ndvi:
            self.tree_ndvi.heading(col, text=col)
            self.tree_ndvi.column(col, width=120, anchor="center")
        self.tree_ndvi.pack(fill="both", expand=True)

        # Tabel untuk Model NDRE
        header_frame_ndre = ttk.Frame(self.frame_ndre)
        header_frame_ndre.pack(fill="x", padx=5, pady=5)

        ttk.Label(header_frame_ndre, text="Hasil Prediksi Model NDRE:",
                  font=("Arial", 10, "bold")).pack(anchor="w")

        self.lbl_mae_ndre = ttk.Label(header_frame_ndre, text="Rata-rata Selisih: -",
                                      font=("Arial", 9, "bold"), foreground="blue")
        self.lbl_mae_ndre.pack(anchor="w", pady=2)

        tree_frame_ndre = ttk.Frame(self.frame_ndre)
        tree_frame_ndre.pack(fill="both", expand=True, pady=5)

        tree_scroll_ndre_y = ttk.Scrollbar(tree_frame_ndre)
        tree_scroll_ndre_y.pack(side="right", fill="y")

        tree_scroll_ndre_x = ttk.Scrollbar(tree_frame_ndre, orient="horizontal")
        tree_scroll_ndre_x.pack(side="bottom", fill="x")

        self.tree_ndre = ttk.Treeview(tree_frame_ndre,
                                      yscrollcommand=tree_scroll_ndre_y.set,
                                      xscrollcommand=tree_scroll_ndre_x.set,
                                      columns=("No", "NDRE", "Prediksi", "Aktual", "Selisih"),
                                      show='headings', height=12)

        tree_scroll_ndre_y.config(command=self.tree_ndre.yview)
        tree_scroll_ndre_x.config(command=self.tree_ndre.xview)

        cols_ndre = ["No", "NDRE", "Prediksi", "Aktual", "Selisih"]
        for col in cols_ndre:
            self.tree_ndre.heading(col, text=col)
            self.tree_ndre.column(col, width=120, anchor="center")
        self.tree_ndre.pack(fill="both", expand=True)

        # Tabel untuk Model GNDVI
        header_frame_gndvi = ttk.Frame(self.frame_gndvi)
        header_frame_gndvi.pack(fill="x", padx=5, pady=5)

        ttk.Label(header_frame_gndvi, text="Hasil Prediksi Model GNDVI:",
                  font=("Arial", 10, "bold")).pack(anchor="w")

        self.lbl_mae_gndvi = ttk.Label(header_frame_gndvi, text="Rata-rata Selisih: -",
                                       font=("Arial", 9, "bold"), foreground="blue")
        self.lbl_mae_gndvi.pack(anchor="w", pady=2)

        tree_frame_gndvi = ttk.Frame(self.frame_gndvi)
        tree_frame_gndvi.pack(fill="both", expand=True, pady=5)

        tree_scroll_gndvi_y = ttk.Scrollbar(tree_frame_gndvi)
        tree_scroll_gndvi_y.pack(side="right", fill="y")

        tree_scroll_gndvi_x = ttk.Scrollbar(tree_frame_gndvi, orient="horizontal")
        tree_scroll_gndvi_x.pack(side="bottom", fill="x")

        self.tree_gndvi = ttk.Treeview(tree_frame_gndvi,
                                       yscrollcommand=tree_scroll_gndvi_y.set,
                                       xscrollcommand=tree_scroll_gndvi_x.set,
                                       columns=("No", "GNDVI", "Prediksi", "Aktual", "Selisih"),
                                       show='headings', height=12)

        tree_scroll_gndvi_y.config(command=self.tree_gndvi.yview)
        tree_scroll_gndvi_x.config(command=self.tree_gndvi.xview)

        cols_gndvi = ["No", "GNDVI", "Prediksi", "Aktual", "Selisih"]
        for col in cols_gndvi:
            self.tree_gndvi.heading(col, text=col)
            self.tree_gndvi.column(col, width=120, anchor="center")
        self.tree_gndvi.pack(fill="both", expand=True)

    # =========================================================================
    # TAB 3: PERBANDINGAN AKURASI
    # =========================================================================
    def setup_tab_comparison(self):
        # Frame untuk tombol
        btn_frame = ttk.Frame(self.tab_compare)
        btn_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(btn_frame, text="📊 TAMPILKAN PERBANDINGAN AKURASI SEMUA MODEL",
                   command=self.show_model_comparison).pack(pady=5)

        # Frame untuk hasil perbandingan
        self.comparison_frame = ttk.Frame(self.tab_compare)
        self.comparison_frame.pack(fill="both", expand=True, padx=10, pady=5)

    # =========================================================================
    # LOGIKA UTAMA
    # =========================================================================

    def browse(self, tk_var):
        f = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if f:
            tk_var.set(f)
            self.log(f"File dipilih: {os.path.basename(f)}")

    def log(self, text, tab="train"):
        if tab == "train":
            self.log_train.insert(tk.END, text + "\n")
            self.log_train.see(tk.END)
        print(text)  # Also print to console

    def process_training(self):
        """Melatih Model Utama + Model Individual + Membandingkan kekuatan tiap Indeks"""
        f_train = self.path_train.get()
        if not f_train:
            return messagebox.showerror("Error", "File Training belum dipilih!")

        try:
            self.log("=== MEMULAI PROSES TRAINING ===")
            df = pd.read_csv(f_train)
            X = df[['NDVI', 'NDRE', 'GNDVI']]
            y = df['Serapan_K']

            # 1. LATIH MODEL UTAMA (Random Forest Gabungan)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            joblib.dump(rf_model, self.model_rf_path)
            self.log(f"✓ Model Gabungan (RF) berhasil dilatih & disimpan")

            # 2. LATIH MODEL INDIVIDUAL UNTUK SETIAP INDEKS
            # Model NDVI
            model_ndvi = LinearRegression().fit(df[['NDVI']], y)
            joblib.dump(model_ndvi, self.model_ndvi_path)
            self.log(f"✓ Model NDVI berhasil dilatih & disimpan")

            # Model NDRE
            model_ndre = LinearRegression().fit(df[['NDRE']], y)
            joblib.dump(model_ndre, self.model_ndre_path)
            self.log(f"✓ Model NDRE berhasil dilatih & disimpan")

            # Model GNDVI
            model_gndvi = LinearRegression().fit(df[['GNDVI']], y)
            joblib.dump(model_gndvi, self.model_gndvi_path)
            self.log(f"✓ Model GNDVI berhasil dilatih & disimpan")

            # 3. EVALUASI MODEL PADA DATA TRAINING
            self.log("\n--- Evaluasi Model pada Data Training ---")

            scores_train = {}

            # Evaluasi model individual
            scores_train['NDVI'] = r2_score(y, model_ndvi.predict(df[['NDVI']]))
            scores_train['NDRE'] = r2_score(y, model_ndre.predict(df[['NDRE']]))
            scores_train['GNDVI'] = r2_score(y, model_gndvi.predict(df[['GNDVI']]))
            scores_train['Gabungan (RF)'] = r2_score(y, rf_model.predict(X))

            for model_name, score in scores_train.items():
                self.log(f"{model_name} - R2 Score: {score:.4f}")

            # 4. VISUALISASI PERBANDINGAN
            self.plot_training_comparison(scores_train)

            messagebox.showinfo("Sukses", "Training semua model selesai! Silakan lanjut ke tab Prediksi & Validasi.")

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat training: {str(e)}")

    def plot_training_comparison(self, scores):
        # Bersihkan plot lama
        for widget in self.frame_graph_train.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Plot 1: Bar chart perbandingan R2 Score
        models = list(scores.keys())
        r2_scores = list(scores.values())

        bars = ax1.bar(models, r2_scores, color=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71'])
        ax1.set_title("Perbandingan R2 Score Model pada Data Training")
        ax1.set_ylabel("R2 Score")
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)

        # Label di atas bar
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom')

        # Plot 2: Pie chart kontribusi relatif
        ax2.pie(r2_scores, labels=models, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Distribusi Kinerja Model")

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.frame_graph_train)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def process_validation(self):
        """Workflow: Load semua model -> Predict File 2 -> Compare with File 3"""
        f_input = self.path_input_pred.get()
        f_actual = self.path_actual.get()

        if not f_input or not f_actual:
            return messagebox.showerror("Error", "Harap pilih kedua file (Input & Aktual)!")

        # Cek apakah semua model sudah ada
        model_files = [self.model_rf_path, self.model_ndvi_path, self.model_ndre_path, self.model_gndvi_path]
        for model_file in model_files:
            if not os.path.exists(model_file):
                return messagebox.showerror("Error",
                                            f"Model {os.path.basename(model_file)} belum ada! Lakukan Training dulu.")

        try:
            # 1. Load Data
            df_input = pd.read_csv(f_input)
            df_actual = pd.read_csv(f_actual)

            if len(df_input) != len(df_actual):
                return messagebox.showerror("Error", "Jumlah baris data input dan data aktual tidak sama!")

            # 2. Load semua Model
            model_rf = joblib.load(self.model_rf_path)
            model_ndvi = joblib.load(self.model_ndvi_path)
            model_ndre = joblib.load(self.model_ndre_path)
            model_gndvi = joblib.load(self.model_gndvi_path)

            # 3. Prediksi dengan semua model
            X_pred = df_input[['NDVI', 'NDRE', 'GNDVI']]
            y_true = df_actual['Serapan_K']

            # Prediksi masing-masing model
            y_pred_rf = model_rf.predict(X_pred)
            y_pred_ndvi = model_ndvi.predict(df_input[['NDVI']])
            y_pred_ndre = model_ndre.predict(df_input[['NDRE']])
            y_pred_gndvi = model_gndvi.predict(df_input[['GNDVI']])

            # 4. Hitung metrik untuk semua model
            model_results = {}

            # Model Gabungan (RF)
            model_results['Gabungan (RF)'] = {
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_rf)),
                'MAE': mean_absolute_error(y_true, y_pred_rf),
                'R2': r2_score(y_true, y_pred_rf)
            }

            # Model NDVI
            model_results['NDVI'] = {
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_ndvi)),
                'MAE': mean_absolute_error(y_true, y_pred_ndvi),
                'R2': r2_score(y_true, y_pred_ndvi)
            }

            # Model NDRE
            model_results['NDRE'] = {
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_ndre)),
                'MAE': mean_absolute_error(y_true, y_pred_ndre),
                'R2': r2_score(y_true, y_pred_ndre)
            }

            # Model GNDVI
            model_results['GNDVI'] = {
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_gndvi)),
                'MAE': mean_absolute_error(y_true, y_pred_gndvi),
                'R2': r2_score(y_true, y_pred_gndvi)
            }

            # 5. Tampilkan hasil prediksi semua model di tabel masing-masing
            self.display_prediction_results(df_input, y_true, y_pred_rf, y_pred_ndvi, y_pred_ndre, y_pred_gndvi,
                                            model_results)

            # 6. Simpan hasil perbandingan untuk ditampilkan di tab 3
            self.model_comparison_results = model_results

            messagebox.showinfo("Validasi Selesai",
                                f"Prediksi dan validasi selesai!\n"
                                f"Model Gabungan (RF) - R2: {model_results['Gabungan (RF)']['R2']:.3f}\n"
                                f"Model NDVI - R2: {model_results['NDVI']['R2']:.3f}\n"
                                f"Model NDRE - R2: {model_results['NDRE']['R2']:.3f}\n"
                                f"Model GNDVI - R2: {model_results['GNDVI']['R2']:.3f}\n"
                                f"Silakan buka tab 'Perbandingan Akurasi' untuk melihat detail.")

        except Exception as e:
            messagebox.showerror("Error Validasi", f"Terjadi kesalahan: {str(e)}")

    def display_prediction_results(self, df_input, y_true, y_pred_rf, y_pred_ndvi, y_pred_ndre, y_pred_gndvi,
                                   model_results):
        """Menampilkan hasil prediksi semua model di tabel masing-masing"""

        # Bersihkan semua tabel terlebih dahulu
        for tree in [self.tree_rf, self.tree_ndvi, self.tree_ndre, self.tree_gndvi]:
            tree.delete(*tree.get_children())

        # Hitung selisih untuk setiap model
        selisih_rf_list = []
        selisih_ndvi_list = []
        selisih_ndre_list = []
        selisih_gndvi_list = []

        # Tampilkan hasil untuk Model Gabungan (RF)
        for i in range(len(df_input)):
            selisih_rf = abs(y_true.iloc[i] - y_pred_rf[i])
            selisih_rf_list.append(selisih_rf)
            self.tree_rf.insert("", "end", values=(
                i + 1,
                f"{df_input['NDVI'].iloc[i]:.4f}",
                f"{df_input['NDRE'].iloc[i]:.4f}",
                f"{df_input['GNDVI'].iloc[i]:.4f}",
                f"{y_pred_rf[i]:.2f}",
                f"{y_true.iloc[i]:.2f}",
                f"{selisih_rf:.2f}"
            ))

        # Tampilkan hasil untuk Model NDVI
        for i in range(len(df_input)):
            selisih_ndvi = abs(y_true.iloc[i] - y_pred_ndvi[i])
            selisih_ndvi_list.append(selisih_ndvi)
            self.tree_ndvi.insert("", "end", values=(
                i + 1,
                f"{df_input['NDVI'].iloc[i]:.4f}",
                f"{y_pred_ndvi[i]:.2f}",
                f"{y_true.iloc[i]:.2f}",
                f"{selisih_ndvi:.2f}"
            ))

        # Tampilkan hasil untuk Model NDRE
        for i in range(len(df_input)):
            selisih_ndre = abs(y_true.iloc[i] - y_pred_ndre[i])
            selisih_ndre_list.append(selisih_ndre)
            self.tree_ndre.insert("", "end", values=(
                i + 1,
                f"{df_input['NDRE'].iloc[i]:.4f}",
                f"{y_pred_ndre[i]:.2f}",
                f"{y_true.iloc[i]:.2f}",
                f"{selisih_ndre:.2f}"
            ))

        # Tampilkan hasil untuk Model GNDVI
        for i in range(len(df_input)):
            selisih_gndvi = abs(y_true.iloc[i] - y_pred_gndvi[i])
            selisih_gndvi_list.append(selisih_gndvi)
            self.tree_gndvi.insert("", "end", values=(
                i + 1,
                f"{df_input['GNDVI'].iloc[i]:.4f}",
                f"{y_pred_gndvi[i]:.2f}",
                f"{y_true.iloc[i]:.2f}",
                f"{selisih_gndvi:.2f}"
            ))

        # Update label rata-rata selisih untuk setiap model
        self.lbl_mae_rf.config(text=f"Rata-rata Selisih: {model_results['Gabungan (RF)']['MAE']:.2f}")
        self.lbl_mae_ndvi.config(text=f"Rata-rata Selisih: {model_results['NDVI']['MAE']:.2f}")
        self.lbl_mae_ndre.config(text=f"Rata-rata Selisih: {model_results['NDRE']['MAE']:.2f}")
        self.lbl_mae_gndvi.config(text=f"Rata-rata Selisih: {model_results['GNDVI']['MAE']:.2f}")

    def show_model_comparison(self):
        """Menampilkan perbandingan akurasi semua model di tab 3"""
        if not hasattr(self, 'model_comparison_results'):
            messagebox.showwarning("Peringatan", "Harap jalankan validasi di tab Prediksi terlebih dahulu!")
            return

        # Bersihkan frame sebelumnya
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()

        # Buat visualisasi perbandingan
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        models = list(self.model_comparison_results.keys())
        metrics = ['RMSE', 'MAE', 'R2']

        # Data untuk plotting
        rmse_values = [self.model_comparison_results[model]['RMSE'] for model in models]
        mae_values = [self.model_comparison_results[model]['MAE'] for model in models]
        r2_values = [self.model_comparison_results[model]['R2'] for model in models]

        # Plot 1: RMSE Comparison
        bars1 = ax1.bar(models, rmse_values, color=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71'])
        ax1.set_title('Perbandingan RMSE (Root Mean Square Error)')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom')

        # Plot 2: MAE Comparison
        bars2 = ax2.bar(models, mae_values, color=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71'])
        ax2.set_title('Perbandingan MAE (Mean Absolute Error)')
        ax2.set_ylabel('MAE')
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom')

        # Plot 3: R2 Score Comparison
        bars3 = ax3.bar(models, r2_values, color=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71'])
        ax3.set_title('Perbandingan R² Score')
        ax3.set_ylabel('R² Score')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom')

        # Plot 4: Metrik Gabungan
        x = np.arange(len(models))
        width = 0.25

        # Normalisasi RMSE dan MAE untuk plotting bersama
        rmse_norm = [v / max(rmse_values) for v in rmse_values]
        mae_norm = [v / max(mae_values) for v in mae_values]

        ax4.bar(x - width, rmse_norm, width, label='RMSE (norm)', color='#e74c3c')
        ax4.bar(x, mae_norm, width, label='MAE (norm)', color='#e67e22')
        ax4.bar(x + width, r2_values, width, label='R²', color='#2ecc71')

        ax4.set_title('Perbandingan Semua Metrik (Normalisasi)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45)
        ax4.legend()

        plt.tight_layout()

        # Tampilkan di GUI
        canvas = FigureCanvasTkAgg(fig, master=self.comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Tambahkan tabel ringkasan
        summary_frame = ttk.Frame(self.comparison_frame)
        summary_frame.pack(fill="x", pady=10)

        ttk.Label(summary_frame, text="Ringkasan Akurasi Model pada Data Validasi:",
                  font=("Arial", 11, "bold")).pack(pady=5)

        # Buat treeview untuk ringkasan
        summary_tree = ttk.Treeview(summary_frame,
                                    columns=("Model", "RMSE", "MAE", "R2"),
                                    show='headings', height=5)

        summary_tree.heading("Model", text="Model")
        summary_tree.heading("RMSE", text="RMSE")
        summary_tree.heading("MAE", text="MAE")
        summary_tree.heading("R2", text="R² Score")

        summary_tree.column("Model", width=120)
        summary_tree.column("RMSE", width=100)
        summary_tree.column("MAE", width=100)
        summary_tree.column("R2", width=100)

        # Isi data
        for model in models:
            results = self.model_comparison_results[model]
            summary_tree.insert("", "end", values=(
                model,
                f"{results['RMSE']:.4f}",
                f"{results['MAE']:.4f}",
                f"{results['R2']:.4f}"
            ))

        summary_tree.pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = AgriApp(root)
    root.mainloop()