import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from tkinter import PhotoImage
import json


class ModernMultiBandApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SpectraCalc Pro - Vegetation Index Analyzer")
        self.root.geometry("1200x900")
        self.root.configure(bg='#f5f6fa')

        # Set minimum window size
        self.root.minsize(1100, 800)

        # Application variables
        self.path_red = tk.StringVar()
        self.path_green = tk.StringVar()
        self.path_nir = tk.StringVar()
        self.path_re = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.output_filename = tk.StringVar(value="vegetation_index_result")

        # Set initial output directory to user's documents
        documents_path = os.path.expanduser("~/Documents")
        self.output_dir.set(documents_path)

        # Configure styles for macOS-like appearance
        self.setup_styles()

        # Create main container with modern layout
        self.setup_ui()

        # Initialize band counters
        self.band_status = {
            "Red": False,
            "Green": False,
            "NIR": False,
            "RedEdge": False
        }

    def setup_styles(self):
        """Configure modern macOS-like styles"""
        style = ttk.Style()

        # Configure colors for macOS Tahoe-like theme
        style.configure('Modern.TFrame', background='#f5f6fa')
        style.configure('Header.TFrame', background='#ffffff')
        style.configure('Card.TFrame', background='#ffffff', relief='flat', borderwidth=0)

        # Modern button styles
        style.configure('Accent.TButton',
                        background='#007AFF',
                        foreground='white',
                        borderwidth=0,
                        focuscolor='none',
                        padding=(20, 10))

        style.map('Accent.TButton',
                  background=[('active', '#0056D6'), ('pressed', '#0040B2')])

        style.configure('Secondary.TButton',
                        background='#E9E9EB',
                        foreground='#000000',
                        borderwidth=0,
                        padding=(15, 8))

        style.map('Secondary.TButton',
                  background=[('active', '#D1D1D6'), ('pressed', '#C5C5C9')])

        # Label styles
        style.configure('Title.TLabel',
                        background='#ffffff',
                        foreground='#1C1C1E',
                        font=('SF Pro Display', 20, 'bold'),
                        padding=(0, 10))

        style.configure('Subtitle.TLabel',
                        background='#ffffff',
                        foreground='#8E8E93',
                        font=('SF Pro Text', 12),
                        padding=(0, 5))

        style.configure('Section.TLabel',
                        background='#ffffff',
                        foreground='#1C1C1E',
                        font=('SF Pro Display', 16, 'bold'),
                        padding=(0, 15))

        style.configure('CardTitle.TLabel',
                        background='#ffffff',
                        foreground='#1C1C1E',
                        font=('SF Pro Text', 14, 'bold'))

        # Entry styles
        style.configure('Modern.TEntry',
                        borderwidth=1,
                        relief='flat',
                        padding=(10, 8),
                        fieldbackground='#FFFFFF',
                        foreground='#1C1C1E')

        # Radiobutton styles
        style.configure('Modern.TRadiobutton',
                        background='#ffffff',
                        foreground='#1C1C1E',
                        font=('SF Pro Text', 11))

    def setup_ui(self):
        """Create modern macOS-like user interface"""

        # Main container with subtle gradient background
        main_container = ttk.Frame(self.root, style='Modern.TFrame')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)

        # Header section with app icon and title
        header_frame = ttk.Frame(main_container, style='Header.TFrame')
        header_frame.pack(fill='x', pady=(0, 20))

        # App title and subtitle
        title_label = ttk.Label(header_frame, text="SpectraCalc Pro", style='Title.TLabel')
        title_label.pack(anchor='w')

        subtitle_label = ttk.Label(header_frame,
                                   text="Advanced Vegetation Index Analysis with Multi-Spectral Imaging",
                                   style='Subtitle.TLabel')
        subtitle_label.pack(anchor='w')

        # Create notebook for organized sections
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill='both', expand=True)

        # Band Input Tab
        input_tab = ttk.Frame(notebook, style='Card.TFrame')
        notebook.add(input_tab, text="📁 Band Input")

        # Index Calculation Tab
        calc_tab = ttk.Frame(notebook, style='Card.TFrame')
        notebook.add(calc_tab, text="📊 Analysis")

        # Setup each tab
        self.setup_input_tab(input_tab)
        self.setup_calculation_tab(calc_tab)

        # Status bar
        self.setup_status_bar(main_container)

    def setup_input_tab(self, parent):
        """Setup the band input tab with modern file selectors"""

        # Input section
        input_section = ttk.LabelFrame(parent, text="SPECTRAL BAND FILES", style='Card.TFrame')
        input_section.pack(fill='x', padx=10, pady=10)

        # Band input grid
        bands_grid = ttk.Frame(input_section, style='Card.TFrame')
        bands_grid.pack(fill='x', padx=15, pady=15)

        # NIR Band (Most important)
        self.create_band_row(bands_grid, "Near Infrared (NIR)", "Required for all indices",
                             self.path_nir, 0, "#007AFF", "🌿")

        # Red Band
        self.create_band_row(bands_grid, "Red Band", "Required for NDVI",
                             self.path_red, 1, "#FF3B30", "🔴")

        # Green Band
        self.create_band_row(bands_grid, "Green Band", "Required for GNDVI",
                             self.path_green, 2, "#34C759", "🟢")

        # Red Edge Band
        self.create_band_row(bands_grid, "Red Edge Band", "Required for NDRE",
                             self.path_re, 3, "#FF9500", "🟠")

        # Output configuration section
        output_section = ttk.LabelFrame(parent, text="OUTPUT CONFIGURATION", style='Card.TFrame')
        output_section.pack(fill='x', padx=10, pady=10)

        output_frame = ttk.Frame(output_section, style='Card.TFrame')
        output_frame.pack(fill='x', padx=15, pady=15)

        # Output directory
        ttk.Label(output_frame, text="Output Directory:", style='CardTitle.TLabel').grid(row=0, column=0, sticky='w',
                                                                                         pady=5)
        dir_frame = ttk.Frame(output_frame, style='Card.TFrame')
        dir_frame.grid(row=0, column=1, sticky='ew', pady=5, padx=(10, 0))
        dir_frame.columnconfigure(0, weight=1)

        ttk.Entry(dir_frame, textvariable=self.output_dir, style='Modern.TEntry').grid(row=0, column=0, sticky='ew',
                                                                                       padx=(0, 5))
        ttk.Button(dir_frame, text="📁 Browse", command=self.browse_output_dir, style='Secondary.TButton').grid(row=0,
                                                                                                               column=1)

        # Output filename
        ttk.Label(output_frame, text="Output Filename:", style='CardTitle.TLabel').grid(row=1, column=0, sticky='w',
                                                                                        pady=5)
        ttk.Entry(output_frame, textvariable=self.output_filename, style='Modern.TEntry').grid(row=1, column=1,
                                                                                               sticky='ew', pady=5,
                                                                                               padx=(10, 0))

        output_frame.columnconfigure(1, weight=1)

    def create_band_row(self, parent, title, description, variable, row, color, emoji):
        """Create a modern band input row"""
        row_frame = ttk.Frame(parent, style='Card.TFrame')
        row_frame.grid(row=row, column=0, sticky='ew', pady=8)
        row_frame.columnconfigure(1, weight=1)

        # Icon and title
        icon_frame = ttk.Frame(row_frame, style='Card.TFrame')
        icon_frame.grid(row=0, column=0, sticky='w', padx=(0, 10))

        ttk.Label(icon_frame, text=emoji, font=('Segoe UI Emoji', 14), background='#ffffff').grid(row=0, column=0)

        title_frame = ttk.Frame(row_frame, style='Card.TFrame')
        title_frame.grid(row=0, column=1, sticky='w')

        ttk.Label(title_frame, text=title, style='CardTitle.TLabel').grid(row=0, column=0, sticky='w')
        ttk.Label(title_frame, text=description, font=('SF Pro Text', 10), foreground='#8E8E93',
                  background='#ffffff').grid(row=1, column=0, sticky='w')

        # File path and browse button
        file_frame = ttk.Frame(row_frame, style='Card.TFrame')
        file_frame.grid(row=0, column=2, sticky='ew', padx=(20, 0))
        file_frame.columnconfigure(0, weight=1)

        entry = ttk.Entry(file_frame, textvariable=variable, style='Modern.TEntry')
        entry.grid(row=0, column=0, sticky='ew', padx=(0, 8))

        browse_btn = ttk.Button(file_frame, text="Select File",
                                command=lambda: self.browse_file(variable, title),
                                style='Secondary.TButton')
        browse_btn.grid(row=0, column=1)

        # Status indicator
        self.status_indicator = ttk.Label(row_frame, text="●", font=('SF Pro Text', 12), foreground='#FF3B30',
                                          background='#ffffff')
        self.status_indicator.grid(row=0, column=3, padx=(15, 0))

    def setup_calculation_tab(self, parent):
        """Setup the calculation and visualization tab"""

        # Left panel for controls
        control_frame = ttk.Frame(parent, style='Card.TFrame')
        control_frame.pack(side='left', fill='y', padx=(0, 10), pady=10)

        # Index selection section
        index_section = ttk.LabelFrame(control_frame, text="VEGETATION INDICES", style='Card.TFrame')
        index_section.pack(fill='x', padx=10, pady=10)

        # Index descriptions
        indices_info = {
            "NDVI": "Normalized Difference Vegetation Index\n- Best for general vegetation health\n- Uses: NIR + Red bands",
            "NDRE": "Normalized Difference Red Edge Index\n- Better for dense vegetation\n- Uses: NIR + Red Edge bands",
            "GNDVI": "Green Normalized Difference Vegetation Index\n- Sensitive to chlorophyll content\n- Uses: NIR + Green bands"
        }

        self.selected_index = tk.StringVar(value="NDVI")

        for idx, (name, desc) in enumerate(indices_info.items()):
            radio_frame = ttk.Frame(index_section, style='Card.TFrame')
            radio_frame.pack(fill='x', padx=15, pady=8)

            rb = ttk.Radiobutton(radio_frame, text=name, variable=self.selected_index,
                                 value=name, style='Modern.TRadiobutton')
            rb.pack(anchor='w')

            desc_label = ttk.Label(radio_frame, text=desc, font=('SF Pro Text', 9),
                                   foreground='#8E8E93', background='#ffffff', justify='left')
            desc_label.pack(anchor='w', padx=(20, 0))

        # Process button
        btn_frame = ttk.Frame(control_frame, style='Card.TFrame')
        btn_frame.pack(fill='x', padx=10, pady=20)

        self.process_btn = ttk.Button(btn_frame, text="🚀 Calculate Vegetation Index",
                                      command=self.process_multiband, style='Accent.TButton')
        self.process_btn.pack(fill='x', pady=5)

        # Right panel for visualization
        viz_frame = ttk.Frame(parent, style='Card.TFrame')
        viz_frame.pack(side='right', fill='both', expand=True, pady=10)

        # Results display area
        self.results_frame = ttk.LabelFrame(viz_frame, text="ANALYSIS RESULTS", style='Card.TFrame')
        self.results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Initial placeholder
        self.setup_placeholder()

    def setup_status_bar(self, parent):
        """Setup modern status bar"""
        status_frame = ttk.Frame(parent, style='Header.TFrame')
        status_frame.pack(fill='x', pady=(10, 0))

        self.status_var = tk.StringVar(value="Ready to analyze vegetation indices")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                 font=('SF Pro Text', 9), foreground='#8E8E93', background='#ffffff')
        status_label.pack(side='left')

        # Band status indicators
        status_indicators = ttk.Frame(status_frame, style='Header.TFrame')
        status_indicators.pack(side='right')

        self.nir_status = ttk.Label(status_indicators, text="NIR: ●", font=('SF Pro Text', 9),
                                    foreground='#FF3B30', background='#ffffff')
        self.nir_status.pack(side='left', padx=(10, 5))

    def setup_placeholder(self):
        """Setup initial placeholder in results frame"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        placeholder = ttk.Frame(self.results_frame, style='Card.TFrame')
        placeholder.pack(expand=True, fill='both')

        ttk.Label(placeholder, text="🌿", font=('Segoe UI Emoji', 48),
                  background='#ffffff').pack(pady=(50, 10))

        ttk.Label(placeholder, text="No Analysis Results",
                  font=('SF Pro Display', 16, 'bold'), background='#ffffff').pack(pady=5)

        ttk.Label(placeholder, text="Select spectral band files and calculate a vegetation index to see results here",
                  font=('SF Pro Text', 11), foreground='#8E8E93', background='#ffffff').pack(pady=5)

    def browse_file(self, string_var, band_name):
        """Modern file browser with feedback"""
        filename = filedialog.askopenfilename(
            title=f"Select {band_name} Band File",
            filetypes=[("GeoTIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if filename:
            string_var.set(filename)
            self.update_band_status(band_name, True)
            self.update_status(f"Loaded {band_name} band: {os.path.basename(filename)}")

    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            self.update_status(f"Output directory set to: {directory}")

    def update_band_status(self, band_name, loaded):
        """Update band loading status"""
        self.band_status[band_name] = loaded
        color = "#34C759" if loaded else "#FF3B30"
        # Update status indicators here

    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update()

    def calculate_vegetation_index(self, nir_array, target_array):
        """
        Calculate vegetation index with robust zero-division handling
        """
        # Convert to float for precision
        nir = nir_array.astype('float64')
        target = target_array.astype('float64')

        # Calculate numerator and denominator
        numerator = nir - target
        denominator = nir + target

        # Initialize result array with NaN
        result = np.full_like(numerator, np.nan, dtype=np.float64)

        # Division only where denominator is not zero
        valid_mask = denominator != 0
        result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

        return result.astype('float32')

    def process_multiband(self):
        """Main processing function with enhanced features"""
        idx_type = self.selected_index.get()

        # Update status
        self.update_status(f"Calculating {idx_type}...")

        # Check file requirements
        p_nir = self.path_nir.get()
        p_red = self.path_red.get()
        p_green = self.path_green.get()
        p_re = self.path_re.get()

        if not p_nir:
            messagebox.showerror("Missing Data", "Near Infrared (NIR) band is required for all vegetation indices!")
            return

        target_path = ""
        target_name = ""

        if idx_type == "NDVI":
            if not p_red:
                messagebox.showerror("Missing Data", "Red band is required for NDVI calculation!")
                return
            target_path = p_red
            target_name = "Red"

        elif idx_type == "NDRE":
            if not p_re:
                messagebox.showerror("Missing Data", "Red Edge band is required for NDRE calculation!")
                return
            target_path = p_re
            target_name = "RedEdge"

        elif idx_type == "GNDVI":
            if not p_green:
                messagebox.showerror("Missing Data", "Green band is required for GNDVI calculation!")
                return
            target_path = p_green
            target_name = "Green"

        try:
            # Read NIR band
            with rasterio.open(p_nir) as src_nir:
                arr_nir = src_nir.read(1)
                meta = src_nir.meta.copy()
                shape_nir = src_nir.shape
                crs_nir = src_nir.crs
                transform_nir = src_nir.transform

            # Read target band
            with rasterio.open(target_path) as src_target:
                arr_target = src_target.read(1)
                shape_target = src_target.shape
                crs_target = src_target.crs

            # Validate dimensions
            if shape_nir != shape_target:
                messagebox.showerror("Dimension Mismatch",
                                     f"Band dimensions do not match!\n\n"
                                     f"NIR Band: {shape_nir}\n"
                                     f"{target_name} Band: {shape_target}\n\n"
                                     f"Please ensure all bands are properly aligned and have the same dimensions.")
                return

            # Calculate vegetation index
            self.update_status("Computing vegetation index...")
            result = self.calculate_vegetation_index(arr_nir, arr_target)

            # Calculate statistics
            valid_mask = ~np.isnan(result)
            valid_pixels = np.sum(valid_mask)
            total_pixels = result.size
            valid_percentage = (valid_pixels / total_pixels) * 100

            if valid_pixels > 0:
                min_val = np.nanmin(result)
                max_val = np.nanmax(result)
                mean_val = np.nanmean(result)
                std_val = np.nanstd(result)
            else:
                min_val = max_val = mean_val = std_val = 0

            # Prepare output path with user-selected directory and filename
            output_filename = f"{self.output_filename.get()}_{idx_type}.tif"
            output_path = os.path.join(self.output_dir.get(), output_filename)

            # Ensure output directory exists
            os.makedirs(self.output_dir.get(), exist_ok=True)

            # Update metadata for UNCOMPRESSED output with maximum quality
            meta.update({
                'dtype': 'float32',
                'count': 1,
                'compress': 'none',  # No compression for maximum quality
                'nodata': np.nan,
                'tiled': False,  # Disable tiling for simpler structure
                'interleave': 'band'
            })

            # Save UNCOMPRESSED GeoTIFF
            self.update_status("Saving high-quality GeoTIFF...")
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(result, 1)

            # Show success message with detailed information
            success_msg = (
                f"✅ {idx_type} Calculation Complete!\n\n"
                f"📊 Results Summary:\n"
                f"• File: {output_filename}\n"
                f"• Location: {self.output_dir.get()}\n"
                f"• Valid Pixels: {valid_pixels:,} ({valid_percentage:.1f}%)\n"
                f"• Value Range: {min_val:.3f} to {max_val:.3f}\n"
                f"• Mean Index: {mean_val:.3f} ± {std_val:.3f}\n\n"
                f"💾 File saved in full quality (uncompressed)"
            )

            messagebox.showinfo("Analysis Complete", success_msg)

            # Update status
            self.update_status(f"{idx_type} analysis complete - {valid_pixels:,} pixels processed")

            # Display results
            self.show_enhanced_preview(result, idx_type, {
                'valid_pixels': valid_pixels,
                'total_pixels': total_pixels,
                'valid_percentage': valid_percentage,
                'min_value': min_val,
                'max_value': max_val,
                'mean_value': mean_val,
                'std_value': std_val,
                'output_path': output_path
            })

        except Exception as e:
            error_msg = f"Processing Error:\n\n{str(e)}"
            messagebox.showerror("Calculation Failed", error_msg)
            self.update_status("Calculation failed - check console for details")

    def show_enhanced_preview(self, data, title, stats):
        """Display enhanced results visualization"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Create main results layout
        main_viz_frame = ttk.Frame(self.results_frame, style='Card.TFrame')
        main_viz_frame.pack(fill='both', expand=True, padx=15, pady=15)

        # Left: Visualization
        viz_left = ttk.Frame(main_viz_frame, style='Card.TFrame')
        viz_left.pack(side='left', fill='both', expand=True, padx=(0, 15))

        # Right: Statistics panel
        viz_right = ttk.Frame(main_viz_frame, style='Card.TFrame')
        viz_right.pack(side='right', fill='y', padx=(15, 0))

        # Create matplotlib figure with modern style
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#ffffff')

        # Create the visualization with enhanced colormap
        im = ax.imshow(data, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

        # Customize the plot
        ax.set_title(f'{title} Vegetation Index\n',
                     fontsize=16, fontweight='bold', pad=20, color='#1C1C1E')
        ax.axis('off')

        # Add subtle grid for reference
        ax.grid(False)

        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Index Value', rotation=270, labelpad=15, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)

        # Add statistics annotation
        stats_text = (
            f"Valid Pixels: {stats['valid_pixels']:,}\n"
            f"Coverage: {stats['valid_percentage']:.1f}%\n"
            f"Range: [{stats['min_value']:.3f}, {stats['max_value']:.3f}]\n"
            f"Mean: {stats['mean_value']:.3f} ± {stats['std_value']:.3f}"
        )

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                                                   facecolor='white', alpha=0.9, edgecolor='#E5E5EA'),
                fontfamily='sans-serif')

        fig.tight_layout()

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=viz_left)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        # Statistics panel
        stats_frame = ttk.LabelFrame(viz_right, text="DETAILED STATISTICS", style='Card.TFrame')
        stats_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create modern statistics display
        stats_data = [
            ("📊 Index Type", title),
            ("✅ Valid Pixels", f"{stats['valid_pixels']:,}"),
            ("📈 Coverage", f"{stats['valid_percentage']:.1f}%"),
            ("📉 Minimum", f"{stats['min_value']:.4f}"),
            ("📈 Maximum", f"{stats['max_value']:.4f}"),
            ("📊 Mean", f"{stats['mean_value']:.4f}"),
            ("📋 Std Dev", f"{stats['std_value']:.4f}"),
            ("💾 File Status", "✓ Saved" if os.path.exists(stats['output_path']) else "✗ Failed")
        ]

        for i, (label, value) in enumerate(stats_data):
            row_frame = ttk.Frame(stats_frame, style='Card.TFrame')
            row_frame.pack(fill='x', padx=10, pady=6)

            ttk.Label(row_frame, text=label, font=('SF Pro Text', 10, 'bold'),
                      foreground='#8E8E93', background='#ffffff').pack(anchor='w')

            ttk.Label(row_frame, text=value, font=('SF Pro Text', 11),
                      foreground='#1C1C1E', background='#ffffff').pack(anchor='w', pady=(2, 0))

        # Add action buttons in stats panel
        action_frame = ttk.Frame(stats_frame, style='Card.TFrame')
        action_frame.pack(fill='x', padx=10, pady=(20, 5))

        ttk.Button(action_frame, text="📁 Show in Folder",
                   command=lambda: os.startfile(os.path.dirname(stats['output_path'])),
                   style='Secondary.TButton').pack(fill='x', pady=2)

        ttk.Button(action_frame, text="🔄 New Analysis",
                   command=self.setup_placeholder,
                   style='Secondary.TButton').pack(fill='x', pady=2)


if __name__ == "__main__":
    root = tk.Tk()

    # Set window icon (if you have an icon file)
    try:
        root.iconphoto(True, PhotoImage(file='icon.png'))
    except:
        pass

    app = ModernMultiBandApp(root)
    root.mainloop()