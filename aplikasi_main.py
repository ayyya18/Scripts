import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Toplevel, colorchooser
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
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

warnings.filterwarnings('ignore')

# Agar resolusi tinggi di layar tajam (Windows 10/11)
try:
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


class ModernAplikasiMachineLearning:
    def __init__(self, root):
        self.root = root
        self.root.title("🌱 AgriSmart - Sistem Analisis Hara Kalium")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f5f5f7')

        # Center window on screen
        self.center_window()

        # Variabel
        self.path_data_latih = tk.StringVar()
        self.path_foto_udara = tk.StringVar()
        self.model_path = "model_rf_hara.joblib"
        self.df_data = None
        self.model = None

        # Band configuration
        self.band_files = {
            'Blue': tk.StringVar(),
            'Green': tk.StringVar(),
            'Red': tk.StringVar(),
            'NIR': tk.StringVar(),
            'RedEdge': tk.StringVar()
        }

        # Color settings for plots
        self.plot_colors = {
            'scatter': '#007AFF',
            'feature_importance': '#34C759',
            'actual_line': '#FF3B30',
            'grid': '#E5E5EA'
        }

        # Setup modern style
        self.setup_modern_style()

        # Create UI
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

        # Try to use aqua theme on macOS, clam on others
        if self.root.tk.call('tk', 'windowingsystem') == 'aqua':
            style.theme_use('aqua')
        else:
            style.theme_use('clam')

        # Configure modern colors
        style.configure('Modern.TFrame', background='#f5f5f7')
        style.configure('Modern.TLabelframe', background='#ffffff', relief='flat', borderwidth=1)
        style.configure('Modern.TLabelframe.Label', background='#ffffff', font=('SF Pro Text', 11, 'bold'))

        # Modern button styles
        style.configure('Accent.TButton',
                        background='#007AFF',
                        foreground='white',
                        borderwidth=0,
                        focuscolor='none',
                        font=('SF Pro Text', 11))
        style.map('Accent.TButton',
                  background=[('active', '#0056D6'), ('pressed', '#0040B2')])

        style.configure('Secondary.TButton',
                        background='#8E8E93',
                        foreground='white',
                        borderwidth=0,
                        font=('SF Pro Text', 11))

        # Entry style
        style.configure('Modern.TEntry',
                        borderwidth=1,
                        relief='flat',
                        padding=8,
                        fieldbackground='#ffffff')

        # Tab style
        style.configure('Modern.TNotebook', background='#f5f5f7', borderwidth=0)
        style.configure('Modern.TNotebook.Tab',
                        background='#e5e5ea',
                        padding=[20, 8],
                        font=('SF Pro Text', 10))
        style.map('Modern.TNotebook.Tab',
                  background=[('selected', '#ffffff')],
                  expand=[('selected', [1, 1, 1, 0])])

    def create_modern_label(self, parent, text, font_size=12, bold=False, color='#000000'):
        """Create a modern label with consistent styling"""
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

        title_label = self.create_modern_label(header_frame, "🌱 AgriSmart", font_size=24, bold=True, color='#007AFF')
        title_label.pack(side='left')

        subtitle_label = self.create_modern_label(header_frame, "Sistem Analisis Hara Kalium & Pemetaan Cerdas",
                                                  font_size=14, color='#8E8E93')
        subtitle_label.pack(side='left', padx=(10, 0))

        # Main tab control
        tab_control = ttk.Notebook(self.root, style='Modern.TNotebook')

        # Create tabs
        self.tab_training = ttk.Frame(tab_control, style='Modern.TFrame')
        self.tab_prediksi = ttk.Frame(tab_control, style='Modern.TFrame')

        tab_control.add(self.tab_training, text='   📊 Training Model  ')
        tab_control.add(self.tab_prediksi, text='   🗺️  Prediksi Peta  ')
        tab_control.pack(expand=1, fill="both", padx=20, pady=10)

        # Build tab contents
        self.build_training_tab()
        self.build_prediction_tab()

    def build_training_tab(self):
        """Build the training tab with modern UI"""
        # File input section
        file_frame = ttk.LabelFrame(self.tab_training, text=" Input Data Training ", style='Modern.TLabelframe')
        file_frame.pack(fill='x', padx=20, pady=10)

        file_input_frame = ttk.Frame(file_frame, style='Modern.TFrame')
        file_input_frame.pack(fill='x', padx=15, pady=15)

        entry_file = ttk.Entry(file_input_frame, textvariable=self.path_data_latih,
                               style='Modern.TEntry', font=('SF Pro Text', 11))
        entry_file.pack(side='left', fill='x', expand=True, padx=(0, 10))

        btn_browse = ttk.Button(file_input_frame, text="📂 Browse",
                                command=self.browse_data_latih, style='Secondary.TButton')
        btn_browse.pack(side='left', padx=(0, 10))

        btn_train = ttk.Button(file_input_frame, text="🚀 Train Model",
                               command=self.start_training_thread, style='Accent.TButton')
        btn_train.pack(side='left')

        # Preview section
        preview_frame = ttk.LabelFrame(self.tab_training, text=" Data Preview ", style='Modern.TLabelframe')
        preview_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.tree_input = ttk.Treeview(preview_frame, show='headings', style='Modern.Treeview')

        # Scrollbars for treeview
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
        """Build the prediction tab with band selector and modern UI"""
        # File input section
        file_frame = ttk.LabelFrame(self.tab_prediksi, text=" Input Citra Multispektral ", style='Modern.TLabelframe')
        file_frame.pack(fill='x', padx=20, pady=10)

        # Single file option
        single_file_frame = ttk.Frame(file_frame, style='Modern.TFrame')
        single_file_frame.pack(fill='x', padx=15, pady=10)

        self.create_modern_label(single_file_frame, "Single GeoTIFF (Semua Band):", 11, True).pack(anchor='w')

        single_input_frame = ttk.Frame(single_file_frame, style='Modern.TFrame')
        single_input_frame.pack(fill='x', pady=5)

        entry_single = ttk.Entry(single_input_frame, textvariable=self.path_foto_udara,
                                 style='Modern.TEntry', font=('SF Pro Text', 11))
        entry_single.pack(side='left', fill='x', expand=True, padx=(0, 10))

        ttk.Button(single_input_frame, text="📂 Browse",
                   command=self.browse_foto_udara, style='Secondary.TButton').pack(side='left')

        # Multiple files option
        multi_file_frame = ttk.Frame(file_frame, style='Modern.TFrame')
        multi_file_frame.pack(fill='x', padx=15, pady=15)

        self.create_modern_label(multi_file_frame, "Multiple Files (Band Terpisah):", 11, True).pack(anchor='w')

        # Band selector grid
        band_grid = ttk.Frame(multi_file_frame, style='Modern.TFrame')
        band_grid.pack(fill='x', pady=10)

        bands = ['Blue', 'Green', 'Red', 'NIR', 'RedEdge']
        for i, band in enumerate(bands):
            band_frame = ttk.Frame(band_grid, style='Modern.TFrame')
            band_frame.grid(row=i // 3, column=i % 3, sticky='ew', padx=10, pady=5)
            band_grid.columnconfigure(i % 3, weight=1)

            self.create_modern_label(band_frame, f"{band}:", 10).pack(side='left')

            entry_band = ttk.Entry(band_frame, textvariable=self.band_files[band],
                                   style='Modern.TEntry', font=('SF Pro Text', 10), width=20)
            entry_band.pack(side='left', fill='x', expand=True, padx=(5, 5))

            ttk.Button(band_frame, text="📁",
                       command=lambda b=band: self.browse_band_file(b),
                       style='Secondary.TButton', width=3).pack(side='left')

        # Color customization section
        color_frame = ttk.LabelFrame(self.tab_prediksi, text=" Customization ", style='Modern.TLabelframe')
        color_frame.pack(fill='x', padx=20, pady=10)

        color_grid = ttk.Frame(color_frame, style='Modern.TFrame')
        color_grid.pack(fill='x', padx=15, pady=10)

        color_options = [
            ('Scatter Points', 'scatter'),
            ('Feature Importance', 'feature_importance'),
            ('Actual Line', 'actual_line')
        ]

        for i, (label, key) in enumerate(color_options):
            color_btn_frame = ttk.Frame(color_grid, style='Modern.TFrame')
            color_btn_frame.grid(row=i // 2, column=i % 2, sticky='w', padx=10, pady=5)

            self.create_modern_label(color_btn_frame, label, 10).pack(side='left')

            color_btn = ttk.Button(color_btn_frame, text="🎨",
                                   command=lambda k=key: self.choose_color(k),
                                   style='Secondary.TButton', width=3)
            color_btn.pack(side='left', padx=(5, 0))

            # Color preview
            color_preview = tk.Frame(color_btn_frame, background=self.plot_colors[key],
                                     width=20, height=20, relief='solid', borderwidth=1)
            color_preview.pack(side='left', padx=(5, 0))
            setattr(self, f'{key}_preview', color_preview)

        # Action buttons
        action_frame = ttk.Frame(self.tab_prediksi, style='Modern.TFrame')
        action_frame.pack(fill='x', padx=20, pady=10)

        btn_predict = ttk.Button(action_frame, text="⚙️ Generate Potassium Map",
                                 command=self.start_prediction_thread, style='Accent.TButton')
        btn_predict.pack(pady=5)

        # Log area
        log_frame = ttk.LabelFrame(self.tab_prediksi, text=" Prediction Log ", style='Modern.TLabelframe')
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.log_pred = scrolledtext.ScrolledText(log_frame, height=15,
                                                  font=('SF Mono', 10),
                                                  background='#ffffff',
                                                  relief='flat',
                                                  borderwidth=1)
        self.log_pred.pack(fill='both', padx=15, pady=15)

    def choose_color(self, color_key):
        """Open color chooser dialog"""
        color = colorchooser.askcolor(title=f"Choose {color_key} color",
                                      initialcolor=self.plot_colors[color_key])[1]
        if color:
            self.plot_colors[color_key] = color
            # Update preview
            preview_widget = getattr(self, f'{color_key}_preview')
            preview_widget.configure(background=color)

    def browse_band_file(self, band_name):
        """Browse for individual band files"""
        filename = filedialog.askopenfilename(
            title=f"Select {band_name} Band File",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if filename:
            self.band_files[band_name].set(filename)
            self.log_p(f"{band_name} band: {os.path.basename(filename)}")

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
                self.log(f"✅ File loaded: {os.path.basename(filename)}")
                self.log(f"📊 Data shape: {len(self.df_data)} rows, {len(self.df_data.columns)} columns")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file: {e}")

    def browse_foto_udara(self):
        """Browse for multispectral image file"""
        filename = filedialog.askopenfilename(
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if filename:
            self.path_foto_udara.set(filename)
            self.log_p(f"📁 Image selected: {os.path.basename(filename)}")

    def load_table(self, tree, df):
        """Load data into treeview"""
        tree.delete(*tree.get_children())
        tree["columns"] = list(df.columns)

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")

        # Show sample data for performance
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
        self.log_pred.insert(tk.END, f"{text}\n")
        self.log_pred.see(tk.END)
        self.root.update()

    def start_training_thread(self):
        """Start training in separate thread"""
        if self.df_data is None:
            messagebox.showwarning("Warning", "Please select training data file first!")
            return
        threading.Thread(target=self.proses_training, daemon=True).start()

    def proses_training(self):
        """Training process"""
        self.log("🎯 Starting training process...")

        try:
            df = self.df_data.copy()

            # Find target column
            target_col = 'Serapan_K'
            if target_col not in df.columns:
                possible_cols = [col for col in df.columns if 'K' in col or 'kalium' in col.lower()]
                if possible_cols:
                    target_col = possible_cols[0]
                    self.log(f"⚠️ Using '{target_col}' as target column")
                else:
                    messagebox.showerror("Error",
                                         f"Target column '{target_col}' not found! Available: {list(df.columns)}")
                    return

            # Prepare features and target
            X = df.drop(columns=[target_col])
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                self.log(f"🗑️ Removing non-numeric columns: {list(non_numeric_cols)}")
                X = X.select_dtypes(include=[np.number])

            if X.empty:
                messagebox.showerror("Error", "No numeric features available for training!")
                return

            y = df[target_col]

            # Handle missing values
            if X.isnull().any().any() or y.isnull().any():
                self.log("🔧 Handling missing values...")
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())

            self.log(f"📈 Features: {list(X.columns)}")
            self.log(f"🔢 Samples: {len(X)}")
            self.log(f"🎯 Target: {target_col}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            self.log("🌲 Training Random Forest model...")
            rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
            rf.fit(X_train, y_train)

            # Evaluate
            y_pred = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            self.log("✅ Training completed!")
            self.log(f"📊 R² Score: {r2:.4f}")
            self.log(f"📉 RMSE: {rmse:.4f}")

            # Save model
            joblib.dump(rf, self.model_path)
            self.model = rf
            self.log(f"💾 Model saved as: {self.model_path}")

            # Show results
            self.root.after(0, lambda: self.show_result_window(rf, X.columns, y_test, y_pred, r2, rmse))

        except Exception as e:
            self.log(f"❌ Training error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def show_result_window(self, model, feature_names, y_test, y_pred, r2, rmse):
        """Show results in modern window"""
        win = Toplevel(self.root)
        win.title("📈 Analysis Results")
        win.geometry("1400x800")
        win.configure(bg='#f5f5f7')

        # Center the window
        win.update_idletasks()
        x = (win.winfo_screenwidth() // 2) - (1400 // 2)
        y = (win.winfo_screenheight() // 2) - (800 // 2)
        win.geometry(f'1400x800+{x}+{y}')

        # Create modern tabs
        tabs = ttk.Notebook(win, style='Modern.TNotebook')

        tab_metrics = ttk.Frame(tabs, style='Modern.TFrame')
        tab_importance = ttk.Frame(tabs, style='Modern.TFrame')
        tab_prediction = ttk.Frame(tabs, style='Modern.TFrame')

        tabs.add(tab_metrics, text='   📊 Summary & Charts  ')
        tabs.add(tab_importance, text='   🌟 Feature Importance  ')
        tabs.add(tab_prediction, text='   📋 Predictions  ')
        tabs.pack(expand=True, fill="both", padx=20, pady=20)

        # Tab 1: Metrics & Charts
        metrics_frame = ttk.Frame(tab_metrics, style='Modern.TFrame')
        metrics_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Results header
        result_text = f"Model Accuracy (R²): {r2:.4f}\nPrediction Error (RMSE): {rmse:.4f}"
        result_label = self.create_modern_label(metrics_frame, result_text, 16, True,
                                                '#34C759' if r2 > 0.7 else '#FF9500')
        result_label.pack(pady=20)

        # Modern plots
        fig = Figure(figsize=(12, 5), facecolor='#f5f5f7')

        # Plot 1: Scatter with custom colors
        ax1 = fig.add_subplot(121)
        ax1.scatter(y_test, y_pred, alpha=0.7, color=self.plot_colors['scatter'], s=60)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 color=self.plot_colors['actual_line'], linewidth=2, linestyle='--')
        ax1.set_xlabel("Actual K Absorption", fontsize=12, color='#333333')
        ax1.set_ylabel("Predicted K Absorption", fontsize=12, color='#333333')
        ax1.set_title("Actual vs Predicted", fontsize=14, color='#333333', pad=20)
        ax1.grid(True, alpha=0.3, color=self.plot_colors['grid'])
        ax1.set_facecolor('#ffffff')

        # Plot 2: Feature importance with custom colors
        ax2 = fig.add_subplot(122)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        features_sorted = [feature_names[i] for i in indices]
        importances_sorted = importances[indices]

        bars = ax2.barh(range(len(importances)), importances_sorted, align="center",
                        color=self.plot_colors['feature_importance'], alpha=0.8)
        ax2.set_yticks(range(len(importances)))
        ax2.set_yticklabels(features_sorted, color='#333333')
        ax2.set_xlabel("Feature Importance", fontsize=12, color='#333333')
        ax2.set_title("Feature Importance Ranking", fontsize=14, color='#333333', pad=20)
        ax2.grid(True, alpha=0.3, color=self.plot_colors['grid'])
        ax2.set_facecolor('#ffffff')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=9, color='#333333')

        fig.tight_layout(pad=3.0)
        canvas = FigureCanvasTkAgg(fig, master=metrics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Tab 2: Feature Importance Table
        self.create_feature_importance_tab(tab_importance, model, feature_names)

        # Tab 3: Prediction Table
        self.create_prediction_tab(tab_prediction, y_test, y_pred)

    def create_feature_importance_tab(self, parent, model, feature_names):
        """Create feature importance tab"""
        frame = ttk.Frame(parent, style='Modern.TFrame')
        frame.pack(fill='both', expand=True, padx=20, pady=20)

        tree = ttk.Treeview(frame, columns=("Rank", "Feature", "Importance"), show='headings', height=20)
        tree.heading("Rank", text="Rank")
        tree.heading("Feature", text="Feature Name")
        tree.heading("Importance", text="Importance Score")

        tree.column("Rank", width=80, anchor="center")
        tree.column("Feature", width=300, anchor="w")
        tree.column("Importance", width=150, anchor="center")

        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Populate data
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
        df_imp = df_imp.sort_values('Importance', ascending=False)

        for i, (_, row) in enumerate(df_imp.iterrows(), 1):
            tree.insert("", "end", values=(i, row['Feature'], f"{row['Importance']:.4f}"))

    def create_prediction_tab(self, parent, y_test, y_pred):
        """Create prediction results tab"""
        frame = ttk.Frame(parent, style='Modern.TFrame')
        frame.pack(fill='both', expand=True, padx=20, pady=20)

        tree = ttk.Treeview(frame, columns=("Sample", "Actual", "Predicted", "Difference", "Error%"),
                            show='headings', height=20)

        columns = {
            "Sample": ("Sample #", 80),
            "Actual": ("Actual Value", 120),
            "Predicted": ("Predicted Value", 120),
            "Difference": ("Difference", 100),
            "Error%": ("Error %", 100)
        }

        for col, (text, width) in columns.items():
            tree.heading(col, text=text)
            tree.column(col, width=width, anchor="center")

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Populate data
        y_test_arr = np.array(y_test)
        for i in range(len(y_test)):
            act = y_test_arr[i]
            pred_val = y_pred[i]
            diff = abs(act - pred_val)
            error_pct = (diff / act) * 100 if act != 0 else 0

            tree.insert("", "end", values=(
                i + 1,
                f"{act:.2f}",
                f"{pred_val:.2f}",
                f"{diff:.2f}",
                f"{error_pct:.1f}%"
            ))

    def start_prediction_thread(self):
        """Start prediction in separate thread"""
        if not os.path.exists(self.model_path):
            messagebox.showwarning("Warning", "Model not trained! Please train model first.")
            return
        threading.Thread(target=self.proses_prediksi, daemon=True).start()

    def proses_prediksi(self):
        """Prediction process with band selection support"""
        # Check input method
        single_file = self.path_foto_udara.get()
        band_files_provided = any(var.get() for var in self.band_files.values())

        if not single_file and not band_files_provided:
            messagebox.showerror("Error", "Please select either a multispectral file or individual band files!")
            return

        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"Model {self.model_path} not found!")
            return

        try:
            self.log_p("🎯 Starting potassium map generation...")
            self.log_p("📥 Loading model...")

            model = joblib.load(self.model_path)

            # Read bands based on input method
            if single_file:
                self.log_p("🖼️ Reading multispectral file...")
                bands_data = self.read_single_file(single_file)
            else:
                self.log_p("🖼️ Reading individual band files...")
                bands_data = self.read_band_files()

            if bands_data is None:
                return

            blue, green, red, nir, re = bands_data

            # Calculate vegetation indices
            self.log_p("📊 Calculating vegetation indices...")
            indices = self.calculate_vegetation_indices(blue, green, red, nir, re)

            # Prepare prediction data
            self.log_p("🔄 Preparing prediction data...")
            X_pred = self.prepare_prediction_data(indices)

            # Perform prediction
            self.log_p("🔮 Predicting potassium absorption...")
            pred_map = self.predict_potassium(model, X_pred, blue.shape)

            # Save results
            self.log_p("💾 Saving results...")
            output_path = self.save_prediction_map(pred_map,
                                                   single_file if single_file else list(self.band_files.values())[
                                                       0].get())

            # Show preview
            self.root.after(0, lambda: self.tampilkan_preview_peta(output_path))

            messagebox.showinfo("Success",
                                f"Potassium map successfully generated!\n\n"
                                f"File: {os.path.basename(output_path)}\n"
                                f"Size: {pred_map.shape[1]} x {pred_map.shape[0]} pixels")

        except Exception as e:
            self.log_p(f"❌ Prediction error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def read_single_file(self, file_path):
        """Read all bands from single multispectral file"""
        try:
            with rasterio.open(file_path) as src:
                profile = src.profile
                self.log_p(f"📐 Image profile: {src.width} x {src.height} pixels, {src.count} bands")

                # Read bands with fallbacks
                bands = []
                for i in range(1, 6):  # Try to read bands 1-5
                    try:
                        band_data = src.read(i).astype('float32')
                        bands.append(band_data)
                        self.log_p(f"📷 Band {i} loaded successfully")
                    except:
                        if i == 5:  # Red Edge
                            self.log_p("⚠️ Red Edge band not available, using NIR as substitute")
                            bands.append(bands[3].copy())  # Use NIR as substitute
                        else:
                            raise Exception(f"Failed to read band {i}")

                return bands

        except Exception as e:
            self.log_p(f"❌ Error reading file: {e}")
            return None

    def read_band_files(self):
        """Read bands from individual files"""
        bands_data = []
        band_order = ['Blue', 'Green', 'Red', 'NIR', 'RedEdge']

        for band_name in band_order:
            file_path = self.band_files[band_name].get()
            if not file_path:
                if band_name == 'RedEdge':
                    self.log_p("⚠️ Red Edge not provided, using NIR as substitute")
                    bands_data.append(bands_data[3].copy())  # Use NIR as substitute for RedEdge
                else:
                    messagebox.showerror("Error", f"Missing {band_name} band file!")
                    return None
            else:
                try:
                    with rasterio.open(file_path) as src:
                        band_data = src.read(1).astype('float32')
                        bands_data.append(band_data)
                        self.log_p(f"📷 {band_name} band loaded: {os.path.basename(file_path)}")
                except Exception as e:
                    self.log_p(f"❌ Error reading {band_name} band: {e}")
                    return None

        return bands_data

    def calculate_vegetation_indices(self, blue, green, red, nir, re):
        """Calculate various vegetation indices"""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red), 0)
            ndre = np.where((nir + re) != 0, (nir - re) / (nir + re), 0)
            gndvi = np.where((nir + green) != 0, (nir - green) / (nir + green), 0)
            savi = np.where((nir + red + 0.5) != 0, (1.5 * (nir - red)) / (nir + red + 0.5), 0)
            evi = np.where((nir + 6 * red - 7.5 * blue + 1) != 0,
                           2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1), 0)

        self.log_p("✅ Vegetation indices calculated: NDVI, NDRE, GNDVI, SAVI, EVI")
        return ndvi, ndre, gndvi, savi, evi

    def prepare_prediction_data(self, indices):
        """Prepare data for prediction"""
        ndvi, ndre, gndvi, savi, evi = indices

        # Stack all indices
        features_list = [index.flatten() for index in indices]
        X_pred = np.column_stack(features_list)

        # Clean data
        X_pred = np.nan_to_num(X_pred, nan=0, posinf=0, neginf=0)

        self.log_p(f"📊 Prediction data: {X_pred.shape[0]} pixels, {X_pred.shape[1]} features")
        return X_pred

    def predict_potassium(self, model, X_pred, original_shape):
        """Perform potassium prediction"""
        batch_size = 100000
        predictions = np.zeros(X_pred.shape[0])

        for i in range(0, X_pred.shape[0], batch_size):
            end_idx = min(i + batch_size, X_pred.shape[0])
            batch = X_pred[i:end_idx]

            valid_mask = np.any(batch != 0, axis=1)
            if np.any(valid_mask):
                predictions[i:end_idx][valid_mask] = model.predict(batch[valid_mask])

            if i % 500000 == 0:
                self.log_p(f"📈 Processed: {end_idx}/{X_pred.shape[0]} pixels")

        # Reshape and normalize
        pred_map = predictions.reshape(original_shape)
        pred_min, pred_max = np.min(pred_map), np.max(pred_map)

        if pred_max > pred_min:
            pred_map_normalized = (pred_map - pred_min) / (pred_max - pred_min) * 100
        else:
            pred_map_normalized = pred_map

        self.log_p(f"📊 Prediction range: {pred_min:.2f} - {pred_max:.2f}")
        return pred_map_normalized

    def save_prediction_map(self, pred_map, input_path):
        """Save prediction map to file"""
        base_name = os.path.splitext(input_path)[0]
        out_path = f"{base_name}_POTASSIUM_MAP.tif"

        # Create simple profile for output
        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'count': 1,
            'compress': 'lzw'
        }

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(pred_map.astype(rasterio.float32), 1)
            dst.update_tags(
                Title="Potassium Absorption Map",
                Model="Random Forest",
                Features="NDVI, NDRE, GNDVI, SAVI, EVI",
                Units="Relative Index (0-100)"
            )

        self.log_p(f"✅ Map saved: {out_path}")
        self.log_p(f"📈 Statistics - Min: {np.min(pred_map):.2f}, "
                   f"Max: {np.max(pred_map):.2f}, Mean: {np.mean(pred_map):.2f}")

        return out_path

    def tampilkan_preview_peta(self, tif_path):
        """Show map preview"""
        try:
            with rasterio.open(tif_path) as src:
                data = src.read(1)

                preview_win = Toplevel(self.root)
                preview_win.title("🗺️ Potassium Map Preview")
                preview_win.geometry("900x700")
                preview_win.configure(bg='#f5f5f7')

                # Center window
                preview_win.update_idletasks()
                x = (preview_win.winfo_screenwidth() // 2) - (900 // 2)
                y = (preview_win.winfo_screenheight() // 2) - (700 // 2)
                preview_win.geometry(f'900x700+{x}+{y}')

                # Create modern plot
                fig = Figure(figsize=(10, 8), facecolor='#f5f5f7')
                ax = fig.add_subplot(111)

                # Use custom colormap
                cmap = LinearSegmentedColormap.from_list('custom_yellow_green',
                                                         ['#FFD600', '#34C759', '#007AFF'])
                im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100)
                ax.set_title("Potassium Absorption Map\n(Relative Index 0-100)",
                             fontsize=14, color='#333333', pad=20)
                ax.set_xlabel("Pixel X", color='#333333')
                ax.set_ylabel("Pixel Y", color='#333333')
                ax.set_facecolor('#ffffff')

                # Colorbar
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("Potassium Level\n(Relative Value)",
                               color='#333333', fontsize=10)

                # Statistics
                stats_text = f"""Statistics:
Min: {np.nanmin(data):.2f}
Max: {np.nanmax(data):.2f}
Mean: {np.nanmean(data):.2f}
Std: {np.nanstd(data):.2f}"""

                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='#ffffff', alpha=0.9,
                                  edgecolor='#E5E5EA'),
                        fontsize=10, color='#333333')

                canvas = FigureCanvasTkAgg(fig, master=preview_win)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

        except Exception as e:
            self.log_p(f"❌ Preview error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernAplikasiMachineLearning(root)
    root.mainloop()