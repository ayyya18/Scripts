"""
Random Forest GUI Application for Vegetation Index and Lab Test Data Analysis
This application provides a graphical interface for:
- Loading vegetation datasets (CSV files)
- Training Random Forest models for classification
- Visualizing results (confusion matrix, feature importance, etc.)
- Making predictions on new data
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import threading
import os
import kagglehub


class RandomForestVegetationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Random Forest - Vegetation Index & Lab Test Analysis")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Data variables
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.target_column = None
        
        # Create main UI
        self.create_menu()
        self.create_main_layout()
        
    def create_menu(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load CSV Dataset", command=self.load_csv)
        file_menu.add_command(label="Download Kaggle Dataset", command=self.download_kaggle_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Export Model Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_main_layout(self):
        """Create the main application layout"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Data Loading & Preview
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="📊 Data")
        self.create_data_tab()
        
        # Tab 2: Model Configuration & Training
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="🌲 Random Forest Model")
        self.create_model_tab()
        
        # Tab 3: Results & Visualization
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="📈 Results")
        self.create_results_tab()
        
        # Tab 4: Prediction
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="🔮 Prediction")
        self.create_prediction_tab()
        
    def create_data_tab(self):
        """Create the data loading and preview tab"""
        # Top frame for controls
        control_frame = ttk.LabelFrame(self.data_tab, text="Data Loading", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Load CSV File", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Download Kaggle Dataset", command=self.download_kaggle_dataset).pack(side=tk.LEFT, padx=5)
        
        self.data_status_label = ttk.Label(control_frame, text="No data loaded")
        self.data_status_label.pack(side=tk.LEFT, padx=20)
        
        # Data preview frame
        preview_frame = ttk.LabelFrame(self.data_tab, text="Data Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for data preview with scrollbars
        tree_frame = ttk.Frame(preview_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.data_tree = ttk.Treeview(tree_frame, show="headings")
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Data info frame
        info_frame = ttk.LabelFrame(self.data_tab, text="Dataset Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.data_info_text = scrolledtext.ScrolledText(info_frame, height=6, wrap=tk.WORD)
        self.data_info_text.pack(fill=tk.X)
        
    def create_model_tab(self):
        """Create the model configuration and training tab"""
        # Left frame for configuration
        config_frame = ttk.LabelFrame(self.model_tab, text="Model Configuration", padding=10)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        
        # Target column selection
        ttk.Label(config_frame, text="Target Column (Y):").pack(anchor=tk.W, pady=5)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(config_frame, textvariable=self.target_var, width=25)
        self.target_combo.pack(anchor=tk.W, pady=5)
        
        # Feature selection
        ttk.Label(config_frame, text="Select Features (X):").pack(anchor=tk.W, pady=5)
        
        feature_list_frame = ttk.Frame(config_frame)
        feature_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.feature_listbox = tk.Listbox(feature_list_frame, selectmode=tk.MULTIPLE, height=15, width=30)
        feature_scrollbar = ttk.Scrollbar(feature_list_frame, orient=tk.VERTICAL, command=self.feature_listbox.yview)
        self.feature_listbox.configure(yscrollcommand=feature_scrollbar.set)
        
        self.feature_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        feature_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(config_frame, text="Select All Features", command=self.select_all_features).pack(pady=5)
        ttk.Button(config_frame, text="Clear Selection", command=self.clear_feature_selection).pack(pady=5)
        
        # Right frame for hyperparameters
        param_frame = ttk.LabelFrame(self.model_tab, text="Hyperparameters", padding=10)
        param_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Number of trees
        ttk.Label(param_frame, text="Number of Trees (n_estimators):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.n_estimators_var = tk.IntVar(value=100)
        ttk.Entry(param_frame, textvariable=self.n_estimators_var, width=10).grid(row=0, column=1, pady=5)
        
        # Max depth
        ttk.Label(param_frame, text="Max Depth (None = unlimited):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.max_depth_var = tk.StringVar(value="None")
        ttk.Entry(param_frame, textvariable=self.max_depth_var, width=10).grid(row=1, column=1, pady=5)
        
        # Min samples split
        ttk.Label(param_frame, text="Min Samples Split:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.min_samples_split_var = tk.IntVar(value=2)
        ttk.Entry(param_frame, textvariable=self.min_samples_split_var, width=10).grid(row=2, column=1, pady=5)
        
        # Min samples leaf
        ttk.Label(param_frame, text="Min Samples Leaf:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.min_samples_leaf_var = tk.IntVar(value=1)
        ttk.Entry(param_frame, textvariable=self.min_samples_leaf_var, width=10).grid(row=3, column=1, pady=5)
        
        # Max features
        ttk.Label(param_frame, text="Max Features:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.max_features_var = tk.StringVar(value="sqrt")
        max_features_combo = ttk.Combobox(param_frame, textvariable=self.max_features_var, 
                                          values=["sqrt", "log2", "None"], width=10)
        max_features_combo.grid(row=4, column=1, pady=5)
        
        # Test size
        ttk.Label(param_frame, text="Test Size (0.1 - 0.5):").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Entry(param_frame, textvariable=self.test_size_var, width=10).grid(row=5, column=1, pady=5)
        
        # Random state
        ttk.Label(param_frame, text="Random State:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.random_state_var = tk.IntVar(value=42)
        ttk.Entry(param_frame, textvariable=self.random_state_var, width=10).grid(row=6, column=1, pady=5)
        
        # Cross-validation folds
        ttk.Label(param_frame, text="Cross-Validation Folds:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.cv_folds_var = tk.IntVar(value=5)
        ttk.Entry(param_frame, textvariable=self.cv_folds_var, width=10).grid(row=7, column=1, pady=5)
        
        # Train button
        ttk.Button(param_frame, text="🚀 Train Model", command=self.train_model, 
                   style="Accent.TButton").grid(row=8, column=0, columnspan=2, pady=20)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(param_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=9, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        # Training log
        ttk.Label(param_frame, text="Training Log:").grid(row=10, column=0, sticky=tk.W, pady=5)
        self.training_log = scrolledtext.ScrolledText(param_frame, height=10, width=50, wrap=tk.WORD)
        self.training_log.grid(row=11, column=0, columnspan=2, sticky=tk.NSEW, pady=5)
        
    def create_results_tab(self):
        """Create the results and visualization tab"""
        # Metrics frame
        metrics_frame = ttk.LabelFrame(self.results_tab, text="Model Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, height=8, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.X)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(self.results_tab, text="Visualizations", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Buttons for different visualizations
        button_frame = ttk.Frame(viz_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Confusion Matrix", 
                   command=self.show_confusion_matrix).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Feature Importance", 
                   command=self.show_feature_importance).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Top 20 Features", 
                   command=self.show_top_features).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Class Distribution", 
                   command=self.show_class_distribution).pack(side=tk.LEFT, padx=5)
        
        # Canvas for matplotlib figures
        self.fig_frame = ttk.Frame(viz_frame)
        self.fig_frame.pack(fill=tk.BOTH, expand=True)
        
    def create_prediction_tab(self):
        """Create the prediction tab for making predictions on new data"""
        # Input frame
        input_frame = ttk.LabelFrame(self.prediction_tab, text="Input Data for Prediction", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Instructions
        ttk.Label(input_frame, text="Enter values for each feature (comma-separated) or load from CSV:").pack(anchor=tk.W)
        
        # Load from CSV button
        ttk.Button(input_frame, text="Load Prediction Data from CSV", 
                   command=self.load_prediction_data).pack(anchor=tk.W, pady=5)
        
        # Manual input
        ttk.Label(input_frame, text="Or enter values manually:").pack(anchor=tk.W, pady=5)
        
        self.prediction_input = scrolledtext.ScrolledText(input_frame, height=10, wrap=tk.WORD)
        self.prediction_input.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Predict button
        ttk.Button(input_frame, text="🔮 Make Prediction", command=self.make_prediction).pack(pady=10)
        
        # Results frame
        result_frame = ttk.LabelFrame(self.prediction_tab, text="Prediction Results", padding=10)
        result_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.prediction_result = scrolledtext.ScrolledText(result_frame, height=8, wrap=tk.WORD)
        self.prediction_result.pack(fill=tk.X)
        
    def load_csv(self):
        """Load a CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.update_data_display()
                messagebox.showinfo("Success", f"Loaded {len(self.df)} rows from {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                
    def download_kaggle_dataset(self):
        """Download the vegetation dataset from Kaggle"""
        try:
            self.log_message("Downloading vegetation dataset from Kaggle...")
            path = kagglehub.dataset_download("saurabhshahane/vegetation")
            self.log_message(f"Dataset downloaded to: {path}")
            
            # Find CSV files
            csv_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if csv_files:
                # Let user choose which file to load
                file_window = tk.Toplevel(self.root)
                file_window.title("Select Dataset File")
                file_window.geometry("500x300")
                
                ttk.Label(file_window, text="Select a CSV file to load:").pack(pady=10)
                
                file_listbox = tk.Listbox(file_window, height=10)
                file_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                
                for f in csv_files:
                    file_listbox.insert(tk.END, os.path.basename(f))
                
                def select_file():
                    selection = file_listbox.curselection()
                    if selection:
                        selected_file = csv_files[selection[0]]
                        self.df = pd.read_csv(selected_file)
                        self.update_data_display()
                        file_window.destroy()
                        messagebox.showinfo("Success", f"Loaded {len(self.df)} rows")
                
                ttk.Button(file_window, text="Load Selected File", command=select_file).pack(pady=10)
            else:
                messagebox.showwarning("Warning", "No CSV files found in the downloaded dataset")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download dataset: {str(e)}")
            
    def update_data_display(self):
        """Update the data preview and column selections"""
        if self.df is None:
            return
            
        # Clear existing data
        self.data_tree.delete(*self.data_tree.get_children())
        
        # Set up columns (show first 20 columns max for performance)
        display_cols = list(self.df.columns[:20])
        self.data_tree["columns"] = display_cols
        
        for col in display_cols:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100, minwidth=50)
        
        # Insert data (first 100 rows)
        for idx, row in self.df.head(100).iterrows():
            values = [row[col] for col in display_cols]
            self.data_tree.insert("", tk.END, values=values)
        
        # Update status
        self.data_status_label.config(text=f"Loaded: {len(self.df)} rows × {len(self.df.columns)} columns")
        
        # Update data info
        self.data_info_text.delete(1.0, tk.END)
        info_text = f"Dataset Shape: {self.df.shape}\n"
        info_text += f"Columns: {', '.join(self.df.columns[:10])}...\n"
        info_text += f"\nData Types:\n{self.df.dtypes.value_counts().to_string()}\n"
        info_text += f"\nMissing Values:\n{self.df.isnull().sum().sum()} total missing values"
        self.data_info_text.insert(tk.END, info_text)
        
        # Update column selections
        self.target_combo["values"] = list(self.df.columns)
        self.feature_listbox.delete(0, tk.END)
        for col in self.df.columns:
            self.feature_listbox.insert(tk.END, col)
            
        # Try to auto-select CLASS as target if it exists
        if "CLASS" in self.df.columns:
            self.target_var.set("CLASS")
            
    def select_all_features(self):
        """Select all features in the listbox"""
        self.feature_listbox.select_set(0, tk.END)
        
    def clear_feature_selection(self):
        """Clear all feature selections"""
        self.feature_listbox.selection_clear(0, tk.END)
        
    def log_message(self, message):
        """Add a message to the training log"""
        self.training_log.insert(tk.END, message + "\n")
        self.training_log.see(tk.END)
        self.root.update_idletasks()
        
    def train_model(self):
        """Train the Random Forest model"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
            
        target = self.target_var.get()
        if not target:
            messagebox.showwarning("Warning", "Please select a target column!")
            return
            
        selected_indices = self.feature_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select at least one feature!")
            return
            
        # Get selected features (excluding target)
        self.feature_names = [self.feature_listbox.get(i) for i in selected_indices]
        if target in self.feature_names:
            self.feature_names.remove(target)
            
        if not self.feature_names:
            messagebox.showwarning("Warning", "Please select features other than the target column!")
            return
            
        # Start training in a thread to keep UI responsive
        threading.Thread(target=self._train_model_thread, args=(target,), daemon=True).start()
        
    def _train_model_thread(self, target):
        """Train model in a separate thread"""
        try:
            self.log_message("=" * 50)
            self.log_message("Starting Random Forest Training...")
            self.progress_var.set(10)
            
            # Prepare data
            self.log_message(f"Target column: {target}")
            self.log_message(f"Features selected: {len(self.feature_names)}")
            
            # Get X and y
            self.X = self.df[self.feature_names].copy()
            self.y = self.df[target].copy()
            
            # Handle missing values
            if self.X.isnull().sum().sum() > 0:
                self.log_message("Handling missing values (filling with median)...")
                self.X = self.X.fillna(self.X.median())
                
            # Encode target if it's categorical
            if self.y.dtype == 'object':
                self.log_message("Encoding categorical target variable...")
                self.y = self.label_encoder.fit_transform(self.y)
                
            self.progress_var.set(30)
            
            # Split data
            test_size = self.test_size_var.get()
            random_state = self.random_state_var.get()
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
            )
            
            self.log_message(f"Training set size: {len(self.X_train)}")
            self.log_message(f"Test set size: {len(self.X_test)}")
            
            self.progress_var.set(50)
            
            # Get hyperparameters
            n_estimators = self.n_estimators_var.get()
            max_depth_str = self.max_depth_var.get()
            max_depth = None if max_depth_str.lower() == "none" else int(max_depth_str)
            min_samples_split = self.min_samples_split_var.get()
            min_samples_leaf = self.min_samples_leaf_var.get()
            max_features_str = self.max_features_var.get()
            max_features = None if max_features_str.lower() == "none" else max_features_str
            
            self.log_message(f"\nHyperparameters:")
            self.log_message(f"  n_estimators: {n_estimators}")
            self.log_message(f"  max_depth: {max_depth}")
            self.log_message(f"  min_samples_split: {min_samples_split}")
            self.log_message(f"  min_samples_leaf: {min_samples_leaf}")
            self.log_message(f"  max_features: {max_features}")
            
            # Create and train model
            self.log_message("\nTraining Random Forest model...")
            
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=-1
            )
            
            self.model.fit(self.X_train, self.y_train)
            self.progress_var.set(80)
            
            # Evaluate model
            self.log_message("Evaluating model...")
            
            train_accuracy = self.model.score(self.X_train, self.y_train)
            test_accuracy = self.model.score(self.X_test, self.y_test)
            
            # Cross-validation
            cv_folds = self.cv_folds_var.get()
            cv_scores = cross_val_score(self.model, self.X, self.y, cv=cv_folds)
            
            self.log_message(f"\n📊 Results:")
            self.log_message(f"  Training Accuracy: {train_accuracy:.4f}")
            self.log_message(f"  Test Accuracy: {test_accuracy:.4f}")
            self.log_message(f"  Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            
            self.progress_var.set(100)
            self.log_message("\n✅ Training completed successfully!")
            
            # Update metrics display
            self.update_metrics_display()
            
            # Switch to results tab
            self.root.after(0, lambda: self.notebook.select(2))
            
        except Exception as e:
            self.log_message(f"\n❌ Error during training: {str(e)}")
            messagebox.showerror("Training Error", str(e))
            
    def update_metrics_display(self):
        """Update the metrics text widget"""
        if self.model is None:
            return
            
        y_pred = self.model.predict(self.X_test)
        
        # Get class labels
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
            target_names = self.label_encoder.classes_
        else:
            target_names = None
            
        # Classification report
        report = classification_report(self.y_test, y_pred, target_names=target_names)
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Update metrics text
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "=" * 60 + "\n")
        self.metrics_text.insert(tk.END, "RANDOM FOREST MODEL METRICS\n")
        self.metrics_text.insert(tk.END, "=" * 60 + "\n\n")
        self.metrics_text.insert(tk.END, f"Overall Accuracy: {accuracy:.4f}\n\n")
        self.metrics_text.insert(tk.END, "Classification Report:\n")
        self.metrics_text.insert(tk.END, "-" * 60 + "\n")
        self.metrics_text.insert(tk.END, report)
        
    def show_confusion_matrix(self):
        """Display confusion matrix visualization"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return
            
        # Clear previous figure
        for widget in self.fig_frame.winfo_children():
            widget.destroy()
            
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get class labels
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
            labels = self.label_encoder.classes_
        else:
            labels = np.unique(self.y_test)
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        canvas = FigureCanvasTkAgg(fig, self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def show_feature_importance(self):
        """Display all feature importances"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return
            
        # Clear previous figure
        for widget in self.fig_frame.winfo_children():
            widget.destroy()
            
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show all features
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title('Random Forest Feature Importance (All Features)')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def show_top_features(self):
        """Display top 20 feature importances"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return
            
        # Clear previous figure
        for widget in self.fig_frame.winfo_children():
            widget.destroy()
            
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
        ax.barh(range(len(indices)), importances[indices], color=colors)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 20 Most Important Features')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def show_class_distribution(self):
        """Display class distribution"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        target = self.target_var.get()
        if not target:
            messagebox.showwarning("Warning", "Please select a target column!")
            return
            
        # Clear previous figure
        for widget in self.fig_frame.winfo_children():
            widget.destroy()
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original distribution
        class_counts = self.df[target].value_counts()
        axes[0].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%')
        axes[0].set_title('Class Distribution (Original Data)')
        
        # Bar chart
        sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[1], palette='viridis')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Class Frequency')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def load_prediction_data(self):
        """Load data for prediction from CSV"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select CSV File for Prediction",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                pred_df = pd.read_csv(file_path)
                
                # Check if it has required features
                missing_features = set(self.feature_names) - set(pred_df.columns)
                if missing_features:
                    messagebox.showwarning("Warning", 
                        f"Missing features in prediction data:\n{', '.join(missing_features)}")
                    return
                    
                # Display data in input field
                self.prediction_input.delete(1.0, tk.END)
                self.prediction_input.insert(tk.END, pred_df[self.feature_names].to_string())
                
                # Store prediction data
                self.pred_data = pred_df[self.feature_names]
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load prediction data: {str(e)}")
                
    def make_prediction(self):
        """Make predictions on new data"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return
            
        try:
            # Check if we have loaded prediction data
            if hasattr(self, 'pred_data') and self.pred_data is not None:
                X_pred = self.pred_data
            else:
                # Try to parse manual input
                input_text = self.prediction_input.get(1.0, tk.END).strip()
                if not input_text:
                    messagebox.showwarning("Warning", "Please enter prediction data or load from CSV!")
                    return
                    
                # Parse comma-separated values
                values = [float(v.strip()) for v in input_text.split(',')]
                if len(values) != len(self.feature_names):
                    messagebox.showwarning("Warning", 
                        f"Expected {len(self.feature_names)} values, got {len(values)}")
                    return
                    
                X_pred = pd.DataFrame([values], columns=self.feature_names)
                
            # Make predictions
            predictions = self.model.predict(X_pred)
            probabilities = self.model.predict_proba(X_pred)
            
            # Decode predictions if using label encoder
            if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
                decoded_predictions = self.label_encoder.inverse_transform(predictions)
            else:
                decoded_predictions = predictions
                
            # Display results
            self.prediction_result.delete(1.0, tk.END)
            self.prediction_result.insert(tk.END, "=" * 50 + "\n")
            self.prediction_result.insert(tk.END, "PREDICTION RESULTS\n")
            self.prediction_result.insert(tk.END, "=" * 50 + "\n\n")
            
            for i, (pred, prob) in enumerate(zip(decoded_predictions, probabilities)):
                self.prediction_result.insert(tk.END, f"Sample {i+1}:\n")
                self.prediction_result.insert(tk.END, f"  Predicted Class: {pred}\n")
                self.prediction_result.insert(tk.END, f"  Confidence: {max(prob)*100:.2f}%\n")
                self.prediction_result.insert(tk.END, f"  Class Probabilities:\n")
                
                if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
                    classes = self.label_encoder.classes_
                else:
                    classes = range(len(prob))
                    
                for cls, p in zip(classes, prob):
                    self.prediction_result.insert(tk.END, f"    {cls}: {p*100:.2f}%\n")
                self.prediction_result.insert(tk.END, "\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            
    def export_results(self):
        """Export model results to file"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Random Forest Model Results\n")
                    f.write("=" * 60 + "\n\n")
                    
                    # Model parameters
                    f.write("Model Parameters:\n")
                    f.write(f"  n_estimators: {self.model.n_estimators}\n")
                    f.write(f"  max_depth: {self.model.max_depth}\n")
                    f.write(f"  min_samples_split: {self.model.min_samples_split}\n")
                    f.write(f"  min_samples_leaf: {self.model.min_samples_leaf}\n")
                    f.write(f"  max_features: {self.model.max_features}\n\n")
                    
                    # Metrics
                    y_pred = self.model.predict(self.X_test)
                    f.write("Classification Report:\n")
                    f.write("-" * 60 + "\n")
                    
                    if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
                        target_names = self.label_encoder.classes_
                    else:
                        target_names = None
                        
                    f.write(classification_report(self.y_test, y_pred, target_names=target_names))
                    
                    # Feature importance
                    f.write("\n\nFeature Importances:\n")
                    f.write("-" * 60 + "\n")
                    importances = self.model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    for i in indices:
                        f.write(f"  {self.feature_names[i]}: {importances[i]:.4f}\n")
                        
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
                
    def show_about(self):
        """Show about dialog"""
        about_text = """Random Forest Vegetation Analysis Tool

Version: 1.0

This application provides a GUI for analyzing vegetation index 
and lab test data using the Random Forest machine learning algorithm.

Features:
• Load CSV datasets or download from Kaggle
• Configure Random Forest hyperparameters
• Train and evaluate classification models
• Visualize confusion matrix and feature importance
• Make predictions on new data

Developed for vegetation classification research."""

        messagebox.showinfo("About", about_text)


def main():
    root = tk.Tk()
    
    # Set theme
    style = ttk.Style()
    style.theme_use('clam')
    
    app = RandomForestVegetationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
