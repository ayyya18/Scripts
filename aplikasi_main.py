import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Toplevel, colorchooser
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import joblib
import rasterio
from rasterio.plot import show
import os
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import warnings
from datetime import datetime
import seaborn as sns

warnings.filterwarnings('ignore')

# Agar resolusi tinggi di layar tajam (Windows 10/11)
try:
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


class ComprehensiveAgriAnalytics:
    def __init__(self, root):
        self.root = root
        self.root.title("🌱 Comprehensive Agri Analytics - Nutrient Prediction System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f5f5f7')

        self.center_window()

        # Variabel utama
        self.path_data_latih = tk.StringVar()
        self.path_data_prediksi = tk.StringVar()
        self.path_foto_udara = tk.StringVar()
        self.model_path = "best_nutrient_model.joblib"
        self.df_data = None
        self.df_prediksi = None
        self.best_model = None
        self.best_index = None
        self.comparison_results = None
        self.prediction_results = None

        # Konfigurasi band
        self.band_files = {
            'Blue': tk.StringVar(),
            'Green': tk.StringVar(),
            'Red': tk.StringVar(),
            'NIR': tk.StringVar(),
            'RedEdge': tk.StringVar()
        }

        # Model machine learning
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'SVM': SVR(kernel='rbf')
        }

        # Indeks vegetasi
        self.vegetation_indices = {
            'NDVI': lambda b, g, r, nir, re: (nir - r) / (nir + r + 1e-8),
            'GNDVI': lambda b, g, r, nir, re: (nir - g) / (nir + g + 1e-8),
            'NDRE': lambda b, g, r, nir, re: (nir - re) / (nir + re + 1e-8),
            'SAVI': lambda b, g, r, nir, re: (1.5 * (nir - r)) / (nir + r + 0.5 + 1e-8),
            'EVI': lambda b, g, r, nir, re: (2.5 * (nir - r)) / (nir + 6 * r - 7.5 * b + 1 + 1e-8),
            'OSAVI': lambda b, g, r, nir, re: (1.16 * (nir - r)) / (nir + r + 0.16 + 1e-8),
            'MSAVI': lambda b, g, r, nir, re: (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - r))) / 2,
            'GCI': lambda b, g, r, nir, re: (nir / g) - 1,
            'RECI': lambda b, g, r, nir, re: (nir / re) - 1,
            'NDWI': lambda b, g, r, nir, re: (g - nir) / (g + nir + 1e-8)
        }

        # Setup UI
        self.setup_modern_style()
        self.create_ui()

    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def setup_modern_style(self):
        """Setup macOS-like modern style"""
        style = ttk.Style()

        if self.root.tk.call('tk', 'windowingsystem') == 'aqua':
            style.theme_use('aqua')
        else:
            style.theme_use('clam')

        style.configure('Modern.TFrame', background='#f5f5f7')
        style.configure('Modern.TLabelframe', background='#ffffff', relief='flat', borderwidth=1)
        style.configure('Modern.TLabelframe.Label', background='#ffffff', font=('SF Pro Text', 11, 'bold'))

    def create_modern_label(self, parent, text, font_size=12, bold=False, color='#000000'):
        """Create a modern label"""
        font_family = 'SF Pro Text' if self.root.tk.call('tk', 'windowingsystem') == 'aqua' else 'Arial'
        font_weight = 'bold' if bold else 'normal'
        return ttk.Label(parent, text=text,
                         font=(font_family, font_size, font_weight),
                         background='#f5f5f7',
                         foreground=color)

    def create_ui(self):
        """Create the main user interface"""
        # Header
        header_frame = ttk.Frame(self.root, style='Modern.TFrame')
        header_frame.pack(fill='x', padx=20, pady=(20, 10))

        title_label = self.create_modern_label(header_frame, "🌱 Comprehensive Nutrient Prediction System",
                                               font_size=20, bold=True, color='#007AFF')
        title_label.pack(side='left')

        # Main tab control
        tab_control = ttk.Notebook(self.root, style='Modern.TNotebook')

        self.tab_training = ttk.Frame(tab_control, style='Modern.TFrame')
        self.tab_prediction = ttk.Frame(tab_control, style='Modern.TFrame')
        self.tab_comparison = ttk.Frame(tab_control, style='Modern.TFrame')
        self.tab_mapping = ttk.Frame(tab_control, style='Modern.TFrame')

        tab_control.add(self.tab_training, text='   1. 📊 Model Training   ')
        tab_control.add(self.tab_prediction, text='   2. 🔮 Data Prediction   ')
        tab_control.add(self.tab_comparison, text='   3. 📈 Index Comparison   ')
        tab_control.add(self.tab_mapping, text='   4. 🗺️  Spatial Mapping   ')
        tab_control.pack(expand=1, fill="both", padx=20, pady=10)

        self.build_training_tab()
        self.build_prediction_tab()
        self.build_comparison_tab()
        self.build_mapping_tab()

    def build_training_tab(self):
        """Build the training tab"""
        # File input section
        file_frame = ttk.LabelFrame(self.tab_training, text=" Training Data Input ", style='Modern.TLabelframe')
        file_frame.pack(fill='x', padx=20, pady=10)

        file_input_frame = ttk.Frame(file_frame, style='Modern.TFrame')
        file_input_frame.pack(fill='x', padx=15, pady=15)

        entry_file = ttk.Entry(file_input_frame, textvariable=self.path_data_latih,
                               style='Modern.TEntry', font=('SF Pro Text', 11))
        entry_file.pack(side='left', fill='x', expand=True, padx=(0, 10))

        btn_browse = ttk.Button(file_input_frame, text="📂 Browse",
                                command=self.browse_data_latih, style='Secondary.TButton')
        btn_browse.pack(side='left', padx=(0, 10))

        btn_train = ttk.Button(file_input_frame, text="🚀 Train & Compare Models",
                               command=self.start_training_thread, style='Accent.TButton')
        btn_train.pack(side='left')

        # Model selection
        model_frame = ttk.LabelFrame(self.tab_training, text=" Model Selection ", style='Modern.TLabelframe')
        model_frame.pack(fill='x', padx=20, pady=10)

        model_grid = ttk.Frame(model_frame, style='Modern.TFrame')
        model_grid.pack(fill='x', padx=15, pady=10)

        self.model_vars = {}
        row, col = 0, 0
        for model_name in self.models.keys():
            var = tk.BooleanVar(value=True)
            self.model_vars[model_name] = var
            cb = ttk.Checkbutton(model_grid, text=model_name, variable=var,
                                 style='Modern.TCheckbutton')
            cb.grid(row=row, column=col, sticky='w', padx=10, pady=5)
            col += 1
            if col > 2:
                col = 0
                row += 1

        # Preview section
        preview_frame = ttk.LabelFrame(self.tab_training, text=" Data Preview ", style='Modern.TLabelframe')
        preview_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.tree_input = ttk.Treeview(preview_frame, show='headings', style='Modern.Treeview')

        scroll_y = ttk.Scrollbar(preview_frame, orient="vertical", command=self.tree_input.yview)
        scroll_x = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.tree_input.xview)
        self.tree_input.configure(yscroll=scroll_y.set, xscroll=scroll_x.set)

        scroll_y.pack(side='right', fill='y')
        scroll_x.pack(side='bottom', fill='x')
        self.tree_input.pack(fill='both', expand=True, padx=15, pady=15)

        # Log section
        log_frame = ttk.LabelFrame(self.tab_training, text=" Training Log ", style='Modern.TLabelframe')
        log_frame.pack(fill='x', padx=20, pady=10)

        self.log_area = scrolledtext.ScrolledText(log_frame, height=8,
                                                  font=('SF Mono', 10),
                                                  background='#ffffff',
                                                  relief='flat',
                                                  borderwidth=1)
        self.log_area.pack(fill='both', padx=15, pady=15)

    def build_prediction_tab(self):
        """Build the prediction tab for new data"""
        # File input section
        file_frame = ttk.LabelFrame(self.tab_prediction, text=" Prediction Data Input ", style='Modern.TLabelframe')
        file_frame.pack(fill='x', padx=20, pady=10)

        file_input_frame = ttk.Frame(file_frame, style='Modern.TFrame')
        file_input_frame.pack(fill='x', padx=15, pady=15)

        entry_file = ttk.Entry(file_input_frame, textvariable=self.path_data_prediksi,
                               style='Modern.TEntry', font=('SF Pro Text', 11))
        entry_file.pack(side='left', fill='x', expand=True, padx=(0, 10))

        btn_browse = ttk.Button(file_input_frame, text="📂 Browse",
                                command=self.browse_data_prediksi, style='Secondary.TButton')
        btn_browse.pack(side='left', padx=(0, 10))

        btn_predict = ttk.Button(file_input_frame, text="🔮 Predict Nutrient Content",
                                 command=self.start_prediction_thread, style='Accent.TButton')
        btn_predict.pack(side='left')

        # Results display
        results_frame = ttk.LabelFrame(self.tab_prediction, text=" Prediction Results ", style='Modern.TLabelframe')
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Best model info
        best_model_frame = ttk.Frame(results_frame, style='Modern.TFrame')
        best_model_frame.pack(fill='x', padx=15, pady=10)

        self.prediction_info_label = self.create_modern_label(best_model_frame,
                                                              "Load prediction data and trained model to see results",
                                                              font_size=12, color='#8E8E93')
        self.prediction_info_label.pack(anchor='w')

        # Prediction results table
        table_frame = ttk.Frame(results_frame, style='Modern.TFrame')
        table_frame.pack(fill='both', expand=True, padx=15, pady=10)

        self.tree_prediction = ttk.Treeview(table_frame, show='headings', style='Modern.Treeview')

        scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree_prediction.yview)
        scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree_prediction.xview)
        self.tree_prediction.configure(yscroll=scroll_y.set, xscroll=scroll_x.set)

        scroll_y.pack(side='right', fill='y')
        scroll_x.pack(side='bottom', fill='x')
        self.tree_prediction.pack(fill='both', expand=True)

        # Accuracy metrics
        metrics_frame = ttk.LabelFrame(self.tab_prediction, text=" Prediction Accuracy ", style='Modern.TLabelframe')
        metrics_frame.pack(fill='x', padx=20, pady=10)

        self.accuracy_text = scrolledtext.ScrolledText(metrics_frame, height=6,
                                                       font=('SF Mono', 10),
                                                       background='#ffffff',
                                                       relief='flat',
                                                       borderwidth=1)
        self.accuracy_text.pack(fill='both', padx=15, pady=15)

    def build_comparison_tab(self):
        """Build the comparison tab"""
        # Results display
        results_frame = ttk.LabelFrame(self.tab_comparison, text=" Model & Index Comparison Results ",
                                       style='Modern.TLabelframe')
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Best model info
        best_model_frame = ttk.Frame(results_frame, style='Modern.TFrame')
        best_model_frame.pack(fill='x', padx=15, pady=10)

        self.best_model_label = self.create_modern_label(best_model_frame,
                                                         "Train models to see comparison results",
                                                         font_size=14, bold=True, color='#8E8E93')
        self.best_model_label.pack(anchor='w')

        # Comparison table
        table_frame = ttk.Frame(results_frame, style='Modern.TFrame')
        table_frame.pack(fill='both', expand=True, padx=15, pady=10)

        self.tree_comparison = ttk.Treeview(table_frame, show='headings', style='Modern.Treeview')

        scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree_comparison.yview)
        scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree_comparison.xview)
        self.tree_comparison.configure(yscroll=scroll_y.set, xscroll=scroll_x.set)

        scroll_y.pack(side='right', fill='y')
        scroll_x.pack(side='bottom', fill='x')
        self.tree_comparison.pack(fill='both', expand=True)

        # Visualization frame
        viz_frame = ttk.LabelFrame(self.tab_comparison, text=" Performance Visualizations ", style='Modern.TLabelframe')
        viz_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.viz_canvas_frame = ttk.Frame(viz_frame, style='Modern.TFrame')
        self.viz_canvas_frame.pack(fill='both', expand=True, padx=15, pady=15)

    def build_mapping_tab(self):
        """Build the spatial mapping tab"""
        # File input section
        file_frame = ttk.LabelFrame(self.tab_mapping, text=" Multispectral Image Input ", style='Modern.TLabelframe')
        file_frame.pack(fill='x', padx=20, pady=10)

        # Single file option
        single_file_frame = ttk.Frame(file_frame, style='Modern.TFrame')
        single_file_frame.pack(fill='x', padx=15, pady=10)

        self.create_modern_label(single_file_frame, "Single GeoTIFF (All Bands):", 11, True).pack(anchor='w')

        single_input_frame = ttk.Frame(single_file_frame, style='Modern.TFrame')
        single_input_frame.pack(fill='x', pady=5)

        entry_single = ttk.Entry(single_input_frame, textvariable=self.path_foto_udara,
                                 style='Modern.TEntry', font=('SF Pro Text', 11))
        entry_single.pack(side='left', fill='x', expand=True, padx=(0, 10))

        ttk.Button(single_input_frame, text="📂 Browse",
                   command=self.browse_foto_udara, style='Secondary.TButton').pack(side='left')

        # Band selector
        band_frame = ttk.LabelFrame(self.tab_mapping, text=" Band Configuration ", style='Modern.TLabelframe')
        band_frame.pack(fill='x', padx=20, pady=10)

        band_grid = ttk.Frame(band_frame, style='Modern.TFrame')
        band_grid.pack(fill='x', padx=15, pady=10)

        bands = ['Blue', 'Green', 'Red', 'NIR', 'RedEdge']
        for i, band in enumerate(bands):
            band_row = ttk.Frame(band_grid, style='Modern.TFrame')
            band_row.grid(row=i, column=0, sticky='ew', padx=10, pady=5)
            band_grid.columnconfigure(0, weight=1)

            self.create_modern_label(band_row, f"{band} Band:", 10).pack(side='left')

            entry_band = ttk.Entry(band_row, textvariable=self.band_files[band],
                                   style='Modern.TEntry', font=('SF Pro Text', 10))
            entry_band.pack(side='left', fill='x', expand=True, padx=(5, 5))

            ttk.Button(band_row, text="📁",
                       command=lambda b=band: self.browse_band_file(b),
                       style='Secondary.TButton', width=3).pack(side='left')

        # Action buttons
        action_frame = ttk.Frame(self.tab_mapping, style='Modern.TFrame')
        action_frame.pack(fill='x', padx=20, pady=10)

        btn_predict = ttk.Button(action_frame, text="⚙️ Generate Spatial Nutrient Map",
                                 command=self.start_mapping_thread, style='Accent.TButton')
        btn_predict.pack(pady=5)

        # Log area
        log_frame = ttk.LabelFrame(self.tab_mapping, text=" Mapping Log ", style='Modern.TLabelframe')
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.log_mapping = scrolledtext.ScrolledText(log_frame, height=15,
                                                     font=('SF Mono', 10),
                                                     background='#ffffff',
                                                     relief='flat',
                                                     borderwidth=1)
        self.log_mapping.pack(fill='both', padx=15, pady=15)

    def browse_data_latih(self):
        """Browse for training data file"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        if filename:
            self.path_data_latih.set(filename)
            try:
                if filename.endswith('.csv'):
                    self.df_data = pd.read_csv(filename)
                else:
                    self.df_data = pd.read_excel(filename)

                self.load_table(self.tree_input, self.df_data)
                self.log(f"✅ Training data loaded: {os.path.basename(filename)}")
                self.log(f"📊 Data shape: {len(self.df_data)} rows, {len(self.df_data.columns)} columns")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file: {e}")

    def browse_data_prediksi(self):
        """Browse for prediction data file"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        if filename:
            self.path_data_prediksi.set(filename)
            try:
                if filename.endswith('.csv'):
                    self.df_prediksi = pd.read_csv(filename)
                else:
                    self.df_prediksi = pd.read_excel(filename)

                self.log_p(f"✅ Prediction data loaded: {os.path.basename(filename)}")
                self.log_p(f"📊 Data shape: {len(self.df_prediksi)} rows, {len(self.df_prediksi.columns)} columns")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file: {e}")

    def browse_foto_udara(self):
        """Browse for multispectral image file"""
        filename = filedialog.askopenfilename(
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if filename:
            self.path_foto_udara.set(filename)
            self.log_m(f"📁 Image selected: {os.path.basename(filename)}")

    def browse_band_file(self, band_name):
        """Browse for individual band files"""
        filename = filedialog.askopenfilename(
            title=f"Select {band_name} Band File",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if filename:
            self.band_files[band_name].set(filename)
            self.log_m(f"{band_name} band: {os.path.basename(filename)}")

    def load_table(self, tree, df):
        """Load data into treeview"""
        tree.delete(*tree.get_children())
        tree["columns"] = list(df.columns)

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")

        sample_df = df.head(100)
        for index, row in sample_df.iterrows():
            tree.insert("", "end", values=list(row))

    def log(self, text):
        """Log to training area"""
        self.log_area.insert(tk.END, f"{text}\n")
        self.log_area.see(tk.END)
        self.root.update()

    def log_p(self, text):
        """Log to prediction area"""
        self.accuracy_text.insert(tk.END, f"{text}\n")
        self.accuracy_text.see(tk.END)
        self.root.update()

    def log_m(self, text):
        """Log to mapping area"""
        self.log_mapping.insert(tk.END, f"{text}\n")
        self.log_mapping.see(tk.END)
        self.root.update()

    def start_training_thread(self):
        """Start training in separate thread"""
        if self.df_data is None:
            messagebox.showwarning("Warning", "Please select training data file first!")
            return

        selected_models = [name for name, var in self.model_vars.items() if var.get()]
        if not selected_models:
            messagebox.showwarning("Warning", "Please select at least one model!")
            return

        threading.Thread(target=self.proses_training_comparison, daemon=True).start()

    def proses_training_comparison(self):
        """Comprehensive training and comparison"""
        self.log("🎯 Starting comprehensive model training and comparison...")

        try:
            df = self.df_data.copy()

            # Find target column
            target_col = self.find_target_column(df)
            if not target_col:
                return

            # Find vegetation index columns
            index_columns = self.find_vegetation_index_columns(df, target_col)
            if not index_columns:
                messagebox.showerror("Error", "No vegetation index columns found!")
                return

            self.log(f"🎯 Target column: {target_col}")
            self.log(f"📊 Vegetation indices: {index_columns}")

            # Prepare data
            X = df[index_columns]
            y = df[target_col]

            # Handle missing values
            if X.isnull().any().any() or y.isnull().any():
                self.log("🔧 Handling missing values...")
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Compare models for each index
            comparison_results = []
            selected_models = [name for name, var in self.model_vars.items() if var.get()]

            for index_name in index_columns:
                self.log(f"🔍 Analyzing index: {index_name}")

                # Use only this index
                X_train_idx = X_train[[index_name]]
                X_test_idx = X_test[[index_name]]

                for model_name in selected_models:
                    try:
                        model = self.models[model_name]
                        model.fit(X_train_idx, y_train)

                        # Predict and evaluate
                        y_pred = model.predict(X_test_idx)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)

                        # Cross-validation
                        cv_scores = cross_val_score(model, X_train_idx, y_train, cv=5, scoring='r2')
                        cv_mean = cv_scores.mean()

                        comparison_results.append({
                            'Index': index_name,
                            'Model': model_name,
                            'R2': r2,
                            'RMSE': rmse,
                            'MAE': mae,
                            'CV_Score': cv_mean,
                            'Model_Object': model
                        })

                        self.log(f"   ✅ {model_name}: R²={r2:.4f}, RMSE={rmse:.4f}")

                    except Exception as e:
                        self.log(f"   ❌ {model_name} failed: {str(e)}")

            # Find best combination
            if comparison_results:
                best_result = max(comparison_results, key=lambda x: x['R2'])
                self.best_model = best_result['Model_Object']
                self.best_index = best_result['Index']
                self.comparison_results = comparison_results

                self.log("✅ Training completed!")
                self.log(f"🏆 Best combination: {best_result['Model']} with {best_result['Index']}")
                self.log(f"📊 Best R²: {best_result['R2']:.4f}")

                # Save best model
                joblib.dump({
                    'model': self.best_model,
                    'index': self.best_index,
                    'all_results': comparison_results
                }, self.model_path)

                # Update UI
                self.root.after(0, self.update_comparison_results)

            else:
                messagebox.showerror("Error", "No successful model training!")

        except Exception as e:
            self.log(f"❌ Training error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def find_target_column(self, df):
        """Find the target nutrient column"""
        target_variations = ['serapan_k', 'serapan_kalium', 'kalium', 'k', 'nutrient', 'target', 'hara']

        for col in df.columns:
            col_str = str(col).lower()  # Konversi ke string dan lowercase
            for variation in target_variations:
                if variation in col_str:
                    return col

        # Jika tidak ditemukan, show column selection
        return self.ask_target_column(df.columns)

    def ask_target_column(self, columns):
        """Ask user to select target column"""
        win = Toplevel(self.root)
        win.title("Select Target Column")
        win.geometry("400x200")
        win.configure(bg='#f5f5f7')

        ttk.Label(win, text="Please select the target nutrient column:",
                  style='Modern.TLabel').pack(pady=20)

        selected_col = tk.StringVar()
        col_combo = ttk.Combobox(win, textvariable=selected_col, values=list(columns))
        col_combo.pack(pady=10)
        col_combo.set(columns[0] if columns else "")

        def confirm():
            win.destroy()

        ttk.Button(win, text="Confirm", command=confirm, style='Accent.TButton').pack(pady=20)

        win.transient(self.root)
        win.grab_set()
        self.root.wait_window(win)

        return selected_col.get()

    def find_vegetation_index_columns(self, df, target_col):
        """Find columns that are likely vegetation indices"""
        index_patterns = ['ndvi', 'gndvi', 'ndre', 'savi', 'evi', 'osavi', 'msavi', 'gci', 'reci', 'ndwi', 'index']

        index_columns = []
        for col in df.columns:
            col_str = str(col)  # Konversi ke string untuk menghindari masalah tipe data

            # Skip target column jika ada
            if target_col and col_str.lower() == str(target_col).lower():
                continue

            col_lower = col_str.lower()
            for pattern in index_patterns:
                if pattern in col_lower:
                    index_columns.append(col_str)
                    break

        # Jika tidak ditemukan, kembalikan semua kolom kecuali target
        if not index_columns:
            all_cols = [str(col) for col in df.columns]
            if target_col:
                target_str = str(target_col)
                index_columns = [col for col in all_cols if col != target_str]
            else:
                index_columns = all_cols

        return index_columns

    def update_comparison_results(self):
        """Update comparison tab with results"""
        if not self.comparison_results:
            return

        # Update best model label
        best_result = max(self.comparison_results, key=lambda x: x['R2'])
        best_text = f"🏆 Best: {best_result['Model']} with {best_result['Index']} (R²: {best_result['R2']:.4f}, RMSE: {best_result['RMSE']:.4f})"
        self.best_model_label.configure(text=best_text, foreground='#34C759')

        # Update comparison table
        self.update_comparison_table()

        # Create visualizations
        self.create_comparison_visualizations()

    def update_comparison_table(self):
        """Update comparison table"""
        for item in self.tree_comparison.get_children():
            self.tree_comparison.delete(item)

        columns = ('Index', 'Model', 'R2', 'RMSE', 'MAE', 'CV_Score')
        self.tree_comparison['columns'] = columns

        for col in columns:
            self.tree_comparison.heading(col, text=col)
            self.tree_comparison.column(col, width=100, anchor='center')

        # Sort by R2 score
        sorted_results = sorted(self.comparison_results, key=lambda x: x['R2'], reverse=True)

        for result in sorted_results:
            values = (
                result['Index'],
                result['Model'],
                f"{result['R2']:.4f}",
                f"{result['RMSE']:.4f}",
                f"{result['MAE']:.4f}",
                f"{result['CV_Score']:.4f}"
            )
            self.tree_comparison.insert("", "end", values=values)

    def create_comparison_visualizations(self):
        """Create comparison visualizations"""
        for widget in self.viz_canvas_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(12, 8), facecolor='#f5f5f7')

        # 1. R2 Score comparison
        ax1 = fig.add_subplot(221)
        self.plot_model_comparison(ax1, 'R2', 'R² Score Comparison', 'R² Score')

        # 2. RMSE comparison
        ax2 = fig.add_subplot(222)
        self.plot_model_comparison(ax2, 'RMSE', 'RMSE Comparison', 'RMSE', lower_better=True)

        # 3. Performance heatmap
        ax3 = fig.add_subplot(223)
        self.plot_performance_heatmap(ax3)

        # 4. Best model details
        ax4 = fig.add_subplot(224)
        self.plot_best_model_details(ax4)

        fig.tight_layout(pad=3.0)

        canvas = FigureCanvasTkAgg(fig, master=self.viz_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_model_comparison(self, ax, metric, title, ylabel, lower_better=False):
        """Plot model comparison"""
        df_plot = pd.DataFrame(self.comparison_results)
        pivot_data = df_plot.pivot(index='Index', columns='Model', values=metric)

        if lower_better:
            pivot_data = pivot_data.reindex(pivot_data.mean().sort_values().index, axis=1)
        else:
            pivot_data = pivot_data.reindex(pivot_data.mean().sort_values(ascending=False).index, axis=1)

        pivot_data.plot(kind='bar', ax=ax, colormap='Set3')
        ax.set_title(title, fontsize=12, color='#333333')
        ax.set_ylabel(ylabel, color='#333333')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#ffffff')

    def plot_performance_heatmap(self, ax):
        """Create performance heatmap"""
        df_plot = pd.DataFrame(self.comparison_results)
        heatmap_data = df_plot.pivot(index='Index', columns='Model', values='R2')

        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns, rotation=45)
        ax.set_yticklabels(heatmap_data.index)
        ax.set_title('R² Score Heatmap', fontsize=12, color='#333333')

        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                text = ax.text(j, i, f'{heatmap_data.iloc[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)

    def plot_best_model_details(self, ax):
        """Plot best model details"""
        best_result = max(self.comparison_results, key=lambda x: x['R2'])

        metrics = ['R2', 'RMSE', 'MAE', 'CV_Score']
        values = [best_result[metric] for metric in metrics]
        labels = ['R²', 'RMSE', 'MAE', 'CV Score']

        bars = ax.bar(labels, values, color=['#34C759', '#FF3B30', '#FF9500', '#007AFF'])
        ax.set_title(f'Best: {best_result["Model"]} + {best_result["Index"]}',
                     fontsize=12, color='#333333')
        ax.set_ylabel('Score', color='#333333')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#ffffff')

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.4f}', ha='center', va='bottom')

    def start_prediction_thread(self):
        """Start prediction thread"""
        if not os.path.exists(self.model_path):
            messagebox.showwarning("Warning", "No trained model found! Please train models first.")
            return

        if self.df_prediksi is None:
            messagebox.showwarning("Warning", "Please select prediction data file first!")
            return

        threading.Thread(target=self.proses_prediksi_data, daemon=True).start()

    def proses_prediksi_data(self):
        """Process data prediction"""
        try:
            self.log_p("🎯 Starting nutrient content prediction...")

            # Load model
            model_data = joblib.load(self.model_path)
            best_model = model_data['model']
            best_index = model_data['index']
            all_results = model_data['all_results']

            self.log_p(f"📊 Using best model: {type(best_model).__name__}")
            self.log_p(f"🌿 Best index: {best_index}")

            # Validasi dan konversi best_index ke string jika perlu
            if hasattr(best_index, 'item'):
                best_index = best_index.item() if hasattr(best_index, 'item') else str(best_index)
            best_index = str(best_index)

            self.log_p(f"🔧 Best index (processed): {best_index}")

            # Find target column in prediction data
            target_col = self.find_target_column(self.df_prediksi)
            if not target_col:
                self.log_p("⚠️ No target column found. Only predictions will be generated.")
                has_target = False
            else:
                has_target = True
                self.log_p(f"🎯 Target column found: {target_col}")

            # Find vegetation index columns - PERBAIKAN DI SINI
            index_columns = self.find_vegetation_index_columns(self.df_prediksi, target_col if has_target else "")

            # Debug: Tampilkan kolom yang tersedia
            self.log_p(f"📋 Available columns: {list(self.df_prediksi.columns)}")
            self.log_p(f"🌿 Found index columns: {index_columns}")

            # Validasi best_index ada dalam data prediksi
            if best_index not in self.df_prediksi.columns:
                self.log_p(f"⚠️ Warning: Best index '{best_index}' not found in prediction data!")
                self.log_p("🔍 Looking for similar columns...")

                # Cari kolom yang mirip
                similar_cols = [col for col in self.df_prediksi.columns if best_index.lower() in col.lower()]
                if similar_cols:
                    self.log_p(f"💡 Similar columns found: {similar_cols}")
                    best_index = similar_cols[0]  # Gunakan kolom pertama yang mirip
                    self.log_p(f"🔄 Using similar column: {best_index}")
                else:
                    # Jika tidak ada yang mirip, gunakan kolom pertama yang tersedia
                    available_cols = [col for col in index_columns if col in self.df_prediksi.columns]
                    if available_cols:
                        best_index = available_cols[0]
                        self.log_p(f"🔄 Using available column: {best_index}")
                    else:
                        messagebox.showerror("Error",
                                             f"Best index '{best_index}' not found in prediction data and no suitable alternatives!")
                        return

            # Prepare prediction data - PASTIKAN best_index adalah string
            try:
                X_pred = self.df_prediksi[[best_index]]
                self.log_p(f"✅ Using index column: {best_index}")
            except KeyError as e:
                self.log_p(f"❌ KeyError: {e}")
                self.log_p(f"📊 Available columns: {list(self.df_prediksi.columns)}")
                messagebox.showerror("Error", f"Column '{best_index}' not found in data!")
                return

            # Handle missing values
            if X_pred.isnull().any().any():
                X_pred = X_pred.fillna(X_pred.mean())
                self.log_p("🔧 Handled missing values")

            # Pastikan data tidak kosong
            if X_pred.empty:
                messagebox.showerror("Error", "Prediction data is empty after preprocessing!")
                return

            # Make predictions
            predictions = best_model.predict(X_pred)

            # Prepare results
            results_df = self.df_prediksi.copy()
            results_df['Predicted_Nutrient'] = predictions

            if has_target:
                actual_values = self.df_prediksi[target_col]

                # Calculate accuracy metrics
                r2 = r2_score(actual_values, predictions)
                rmse = np.sqrt(mean_squared_error(actual_values, predictions))
                mae = mean_absolute_error(actual_values, predictions)

                # Compare all indices if target available
                index_accuracies = []
                for index_name in index_columns:
                    if index_name in self.df_prediksi.columns:
                        X_idx = self.df_prediksi[[index_name]].fillna(self.df_prediksi[[index_name]].mean())

                        # Find the best model for this index from training results
                        index_results = [r for r in all_results if str(r['Index']) == str(index_name)]
                        if index_results:
                            best_index_model = max(index_results, key=lambda x: x['R2'])['Model_Object']
                            try:
                                pred_idx = best_index_model.predict(X_idx)
                                r2_idx = r2_score(actual_values, pred_idx)
                                index_accuracies.append({
                                    'Index': index_name,
                                    'R2': r2_idx,
                                    'Model': type(best_index_model).__name__
                                })
                            except Exception as e:
                                self.log_p(f"⚠️ Prediction failed for {index_name}: {e}")

                # Sort by accuracy
                index_accuracies.sort(key=lambda x: x['R2'], reverse=True)

                # Display accuracy results
                self.log_p("\n📊 PREDICTION ACCURACY RESULTS:")
                self.log_p("=" * 50)
                self.log_p(f"Overall Accuracy with Best Index ({best_index}):")
                self.log_p(f"  R² Score: {r2:.4f}")
                self.log_p(f"  RMSE: {rmse:.4f}")
                self.log_p(f"  MAE: {mae:.4f}")

                self.log_p("\n🌿 INDEX ACCURACY COMPARISON:")
                for i, acc in enumerate(index_accuracies, 1):
                    status = "🏆 BEST" if i == 1 else "✓" if acc['R2'] > 0.5 else "⚠️"
                    self.log_p(f"  {i:2d}. {acc['Index']:8} - R²: {acc['R2']:.4f} {status}")

            # Update prediction table
            self.root.after(0, lambda: self.update_prediction_results(results_df, has_target))

            self.log_p("✅ Prediction completed successfully!")

        except Exception as e:
            self.log_p(f"❌ Prediction error: {str(e)}")
            import traceback
            self.log_p(f"🔍 Detailed error: {traceback.format_exc()}")
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def update_prediction_results(self, results_df, has_target):
        """Update prediction results table"""
        try:
            for item in self.tree_prediction.get_children():
                self.tree_prediction.delete(item)

            # Determine columns to show
            columns = list(results_df.columns)

            # Tambahkan kolom Error jika ada target
            if has_target:
                target_col = self.find_target_column(results_df)
                if target_col and 'Predicted_Nutrient' in results_df.columns:
                    # Hitung error untuk setiap row
                    errors = abs(results_df[target_col] - results_df['Predicted_Nutrient'])
                    results_df['Error'] = errors
                    columns = list(results_df.columns)

            self.tree_prediction['columns'] = columns

            for col in columns:
                col_str = str(col)  # Pastikan string
                self.tree_prediction.heading(col_str, text=col_str)
                self.tree_prediction.column(col_str, width=100, anchor='center')

            # Populate table
            for index, row in results_df.iterrows():
                values = []
                for col in columns:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        values.append(f"{val:.4f}")
                    else:
                        values.append(str(val))

                self.tree_prediction.insert("", "end", values=values)

            # Update info label
            if has_target:
                self.prediction_info_label.configure(
                    text="✅ Prediction completed with accuracy assessment",
                    foreground='#34C759'
                )
            else:
                self.prediction_info_label.configure(
                    text="✅ Prediction completed (no target for accuracy assessment)",
                    foreground='#FF9500'
                )

        except Exception as e:
            self.log_p(f"❌ Error updating prediction results: {str(e)}")

    def start_mapping_thread(self):
        """Start mapping thread"""
        if not os.path.exists(self.model_path):
            messagebox.showwarning("Warning", "No trained model found! Please train models first.")
            return

        threading.Thread(target=self.proses_pemetaan, daemon=True).start()

    def proses_pemetaan(self):
        """Process spatial mapping"""
        single_file = self.path_foto_udara.get()
        band_files_provided = any(var.get() for var in self.band_files.values())

        if not single_file and not band_files_provided:
            messagebox.showerror("Error", "Please select either multispectral file or individual band files!")
            return

        try:
            self.log_m("🎯 Starting spatial nutrient mapping...")

            # Load model
            model_data = joblib.load(self.model_path)
            model = model_data['model']
            best_index = model_data['index']

            self.log_m(f"📊 Using model: {type(model).__name__}")
            self.log_m(f"🌿 Best index: {best_index}")

            # Read bands
            if single_file:
                self.log_m("🖼️ Reading multispectral file...")
                bands_data = self.read_single_file(single_file)
            else:
                self.log_m("🖼️ Reading individual band files...")
                bands_data = self.read_band_files()

            if bands_data is None:
                return

            blue, green, red, nir, re = bands_data

            # Calculate vegetation indices
            self.log_m("📊 Calculating vegetation indices...")
            index_maps = {}
            for index_name, index_func in self.vegetation_indices.items():
                index_map = index_func(blue, green, red, nir, re)
                index_maps[index_name] = index_map

            # Predict using best index
            if best_index in index_maps:
                self.log_m(f"🔮 Predicting using best index: {best_index}")
                pred_map = self.predict_with_model(model, index_maps[best_index])

                # Save results
                output_path = self.save_prediction_map(pred_map,
                                                       single_file if single_file else list(self.band_files.values())[
                                                           0].get(),
                                                       best_index)

                # Show preview
                self.root.after(0, lambda: self.tampilkan_preview_peta(output_path, index_maps, best_index))

                self.log_m("✅ Spatial mapping completed successfully!")

            else:
                messagebox.showerror("Error", f"Best index {best_index} not available!")

        except Exception as e:
            self.log_m(f"❌ Mapping error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def read_single_file(self, file_path):
        """Read single multispectral file"""
        try:
            with rasterio.open(file_path) as src:
                self.log_m(f"📐 Image: {src.width} x {src.height} pixels, {src.count} bands")

                bands = []
                for i in range(1, 6):
                    try:
                        band_data = src.read(i).astype('float32')
                        bands.append(band_data)
                    except:
                        if i == 5:
                            self.log_m("⚠️ Using NIR as RedEdge substitute")
                            bands.append(bands[3].copy())
                        else:
                            raise Exception(f"Failed to read band {i}")

                return bands

        except Exception as e:
            self.log_m(f"❌ Error reading file: {e}")
            return None

    def read_band_files(self):
        """Read individual band files"""
        bands_data = []
        band_order = ['Blue', 'Green', 'Red', 'NIR', 'RedEdge']

        for band_name in band_order:
            file_path = self.band_files[band_name].get()
            if not file_path:
                if band_name == 'RedEdge':
                    self.log_m("⚠️ Using NIR as RedEdge substitute")
                    bands_data.append(bands_data[3].copy())
                else:
                    messagebox.showerror("Error", f"Missing {band_name} band file!")
                    return None
            else:
                try:
                    with rasterio.open(file_path) as src:
                        band_data = src.read(1).astype('float32')
                        bands_data.append(band_data)
                except Exception as e:
                    self.log_m(f"❌ Error reading {band_name} band: {e}")
                    return None

        return bands_data

    def predict_with_model(self, model, index_map):
        """Predict with model"""
        flat_data = index_map.flatten().reshape(-1, 1)
        flat_data = np.nan_to_num(flat_data, nan=0, posinf=0, neginf=0)

        predictions = model.predict(flat_data)
        pred_map = predictions.reshape(index_map.shape)

        # Normalize
        pred_min, pred_max = np.min(pred_map), np.max(pred_map)
        if pred_max > pred_min:
            pred_map_normalized = (pred_map - pred_min) / (pred_max - pred_min) * 100
        else:
            pred_map_normalized = pred_map

        self.log_m(f"📊 Prediction range: {pred_min:.2f} - {pred_max:.2f}")
        return pred_map_normalized

    def save_prediction_map(self, pred_map, input_path, index_name):
        """Save prediction map"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(input_path)[0]
        out_path = f"{base_name}_NUTRIENT_MAP_{timestamp}.tif"

        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'count': 1,
            'compress': 'lzw'
        }

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(pred_map.astype(rasterio.float32), 1)
            dst.update_tags(
                Title=f"Nutrient Map - {index_name}",
                Model=type(self.best_model).__name__,
                Index=index_name,
                Units="Relative Index (0-100)"
            )

        self.log_m(f"💾 Map saved: {out_path}")
        return out_path

    def tampilkan_preview_peta(self, tif_path, index_maps, best_index):
        """Show map preview"""
        try:
            with rasterio.open(tif_path) as src:
                data = src.read(1)

                preview_win = Toplevel(self.root)
                preview_win.title("🗺️ Nutrient Map Preview")
                preview_win.geometry("1000x700")
                preview_win.configure(bg='#f5f5f7')

                # Center window
                preview_win.update_idletasks()
                x = (preview_win.winfo_screenwidth() // 2) - (1000 // 2)
                y = (preview_win.winfo_screenheight() // 2) - (700 // 2)
                preview_win.geometry(f'1000x700+{x}+{y}')

                fig = Figure(figsize=(10, 8), facecolor='#f5f5f7')
                ax = fig.add_subplot(111)

                cmap = LinearSegmentedColormap.from_list('custom_nutrient',
                                                         ['#FF6B6B', '#FFD93D', '#6BCF7F', '#4D96FF'])
                im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100)
                ax.set_title(f"Nutrient Absorption Map\nBest Index: {best_index}",
                             fontsize=14, color='#333333', pad=20)
                ax.set_facecolor('#ffffff')

                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("Nutrient Level", color='#333333')

                stats_text = f"""Statistics:
Min: {np.nanmin(data):.2f}
Max: {np.nanmax(data):.2f}
Mean: {np.nanmean(data):.2f}
Std: {np.nanstd(data):.2f}"""

                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='#ffffff', alpha=0.9),
                        fontsize=10, color='#333333')

                canvas = FigureCanvasTkAgg(fig, master=preview_win)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

        except Exception as e:
            self.log_m(f"❌ Preview error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ComprehensiveAgriAnalytics(root)
    root.mainloop()