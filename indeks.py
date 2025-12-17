import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import rasterio
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from tkinter import PhotoImage
import json
from shapely.geometry import shape
import fiona
import geopandas as gpd
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors


class ModernMultiBandApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SpectraCalc Pro - Advanced Vegetation Index Analyzer")
        self.root.geometry("1400x1000")
        self.root.configure(bg='#f5f6fa')

        # Set minimum window size
        self.root.minsize(1300, 900)

        # Application variables
        self.path_red = tk.StringVar()
        self.path_green = tk.StringVar()
        self.path_nir = tk.StringVar()
        self.path_re = tk.StringVar()
        self.path_blue = tk.StringVar()
        self.path_rgb = tk.StringVar()
        self.path_shapefile = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.output_filename = tk.StringVar(value="vegetation_index_result")

        # RGB band selection variables
        self.rgb_red_band = tk.IntVar(value=1)
        self.rgb_green_band = tk.IntVar(value=2)
        self.rgb_blue_band = tk.IntVar(value=3)
        self.rgb_nir_band = tk.IntVar(value=4)

        # Crop parameters
        self.crop_enabled = tk.BooleanVar(value=False)
        self.crop_buffer = tk.DoubleVar(value=0.0)

        # Set initial output directory to user's documents
        documents_path = os.path.expanduser("~/Documents")
        self.output_dir.set(documents_path)

        # Band metadata storage
        self.band_metadata = {}

        # Configure styles for macOS-like appearance
        self.setup_styles()

        # Create main container with modern layout
        self.setup_ui()

        # Initialize band counters
        self.band_status = {
            "Red": False,
            "Green": False,
            "NIR": False,
            "RedEdge": False,
            "Blue": False,
            "RGB": False,
            "Shapefile": False
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

        # Metadata Tab
        metadata_tab = ttk.Frame(notebook, style='Card.TFrame')
        notebook.add(metadata_tab, text="📋 Metadata")

        # Setup each tab
        self.setup_input_tab(input_tab)
        self.setup_calculation_tab(calc_tab)
        self.setup_metadata_tab(metadata_tab)

        # Status bar
        self.setup_status_bar(main_container)

    def setup_input_tab(self, parent):
        """Setup the band input tab with modern file selectors"""

        # Main container with scrollbar
        canvas = tk.Canvas(parent, bg='#f5f6fa', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Modern.TFrame')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Input section for separate bands
        input_section = ttk.LabelFrame(scrollable_frame, text="INDIVIDUAL SPECTRAL BAND FILES", style='Card.TFrame')
        input_section.pack(fill='x', padx=10, pady=10, ipady=5)

        # Band input grid
        bands_grid = ttk.Frame(input_section, style='Card.TFrame')
        bands_grid.pack(fill='x', padx=15, pady=15)

        # NIR Band (Most important)
        self.create_band_row(bands_grid, "Near Infrared (NIR)", "Required for all indices",
                             self.path_nir, 0, "#007AFF", "🌿")

        # Red Band
        self.create_band_row(bands_grid, "Red Band", "Required for NDVI, SAVI, EVI",
                             self.path_red, 1, "#FF3B30", "🔴")

        # Green Band
        self.create_band_row(bands_grid, "Green Band", "Required for GNDVI, GCI",
                             self.path_green, 2, "#34C759", "🟢")

        # Red Edge Band
        self.create_band_row(bands_grid, "Red Edge Band", "Required for NDRE, RECI",
                             self.path_re, 3, "#FF9500", "🟠")

        # Blue Band
        self.create_band_row(bands_grid, "Blue Band", "Required for EVI, ARVI, MCARI",
                             self.path_blue, 4, "#5856D6", "🔵")

        # RGB Multi-band File Section
        rgb_section = ttk.LabelFrame(scrollable_frame, text="RGB MULTI-BAND FILE", style='Card.TFrame')
        rgb_section.pack(fill='x', padx=10, pady=10, ipady=5)

        rgb_frame = ttk.Frame(rgb_section, style='Card.TFrame')
        rgb_frame.pack(fill='x', padx=15, pady=15)

        # RGB file selection
        ttk.Label(rgb_frame, text="RGB Multi-band File:", style='CardTitle.TLabel').grid(row=0, column=0, sticky='w',
                                                                                         pady=5)
        rgb_file_frame = ttk.Frame(rgb_frame, style='Card.TFrame')
        rgb_file_frame.grid(row=0, column=1, sticky='ew', pady=5, padx=(10, 0), columnspan=2)
        rgb_file_frame.columnconfigure(0, weight=1)

        ttk.Entry(rgb_file_frame, textvariable=self.path_rgb, style='Modern.TEntry').grid(row=0, column=0, sticky='ew',
                                                                                          padx=(0, 5))
        ttk.Button(rgb_file_frame, text="📁 Browse",
                   command=lambda: self.browse_file(self.path_rgb, "RGB"),
                   style='Secondary.TButton').grid(row=0, column=1)

        # RGB band selection
        ttk.Label(rgb_frame, text="Band Assignment:", style='CardTitle.TLabel').grid(row=1, column=0, sticky='w',
                                                                                     pady=10)

        band_frame = ttk.Frame(rgb_frame, style='Card.TFrame')
        band_frame.grid(row=1, column=1, sticky='w', pady=10, padx=(10, 0))

        ttk.Label(band_frame, text="Red Band:", font=('SF Pro Text', 10), background='#ffffff').grid(row=0, column=0,
                                                                                                     sticky='w',
                                                                                                     padx=(0, 5))
        ttk.Combobox(band_frame, textvariable=self.rgb_red_band, values=list(range(1, 11)),
                     width=5, state='readonly').grid(row=0, column=1, padx=(0, 10))

        ttk.Label(band_frame, text="Green Band:", font=('SF Pro Text', 10), background='#ffffff').grid(row=0, column=2,
                                                                                                       sticky='w',
                                                                                                       padx=(0, 5))
        ttk.Combobox(band_frame, textvariable=self.rgb_green_band, values=list(range(1, 11)),
                     width=5, state='readonly').grid(row=0, column=3, padx=(0, 10))

        ttk.Label(band_frame, text="Blue Band:", font=('SF Pro Text', 10), background='#ffffff').grid(row=0, column=4,
                                                                                                      sticky='w',
                                                                                                      padx=(0, 5))
        ttk.Combobox(band_frame, textvariable=self.rgb_blue_band, values=list(range(1, 11)),
                     width=5, state='readonly').grid(row=0, column=5, padx=(0, 10))

        ttk.Label(band_frame, text="NIR Band:", font=('SF Pro Text', 10), background='#ffffff').grid(row=0, column=6,
                                                                                                     sticky='w',
                                                                                                     padx=(0, 5))
        ttk.Combobox(band_frame, textvariable=self.rgb_nir_band, values=list(range(1, 11)),
                     width=5, state='readonly').grid(row=0, column=7)

        # Shapefile Cropping Section
        crop_section = ttk.LabelFrame(scrollable_frame, text="SHAPEFILE CROPPING", style='Card.TFrame')
        crop_section.pack(fill='x', padx=10, pady=10, ipady=5)

        crop_frame = ttk.Frame(crop_section, style='Card.TFrame')
        crop_frame.pack(fill='x', padx=15, pady=15)

        # Crop enable checkbox
        ttk.Checkbutton(crop_frame, text="Enable Cropping", variable=self.crop_enabled,
                        style='Modern.TRadiobutton').grid(row=0, column=0, sticky='w', pady=5)

        # Shapefile selection
        ttk.Label(crop_frame, text="Shapefile:", style='CardTitle.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        shape_frame = ttk.Frame(crop_frame, style='Card.TFrame')
        shape_frame.grid(row=1, column=1, sticky='ew', pady=5, padx=(10, 0))
        shape_frame.columnconfigure(0, weight=1)

        ttk.Entry(shape_frame, textvariable=self.path_shapefile, style='Modern.TEntry').grid(row=0, column=0,
                                                                                             sticky='ew', padx=(0, 5))
        ttk.Button(shape_frame, text="📁 Browse",
                   command=lambda: self.browse_shapefile(),
                   style='Secondary.TButton').grid(row=0, column=1)

        # Buffer distance
        ttk.Label(crop_frame, text="Buffer (meters):", style='CardTitle.TLabel').grid(row=2, column=0, sticky='w',
                                                                                      pady=5)
        ttk.Spinbox(crop_frame, from_=0, to=1000, textvariable=self.crop_buffer,
                    width=10).grid(row=2, column=1, sticky='w', pady=5, padx=(10, 0))

        # Output configuration section
        output_section = ttk.LabelFrame(scrollable_frame, text="OUTPUT CONFIGURATION", style='Card.TFrame')
        output_section.pack(fill='x', padx=10, pady=10, ipady=5)

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

        # Metadata button
        if title != "RGB":
            metadata_btn = ttk.Button(file_frame, text="📋 Metadata",
                                      command=lambda: self.show_file_metadata(variable.get(), title),
                                      style='Secondary.TButton')
            metadata_btn.grid(row=0, column=2, padx=(5, 0))

    def setup_calculation_tab(self, parent):
        """Setup the calculation and visualization tab"""

        # Left panel for controls
        control_frame = ttk.Frame(parent, style='Card.TFrame')
        control_frame.pack(side='left', fill='y', padx=(0, 10), pady=10)

        # Index selection section
        index_section = ttk.LabelFrame(control_frame, text="VEGETATION INDICES", style='Card.TFrame')
        index_section.pack(fill='x', padx=10, pady=10)

        # Create a listbox for vegetation indices
        index_list_frame = ttk.Frame(index_section, style='Card.TFrame')
        index_list_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Scrollbar for index list
        scrollbar = ttk.Scrollbar(index_list_frame)
        scrollbar.pack(side='right', fill='y')

        self.index_listbox = tk.Listbox(index_list_frame,
                                        yscrollcommand=scrollbar.set,
                                        font=('SF Pro Text', 11),
                                        bg='white',
                                        selectbackground='#007AFF',
                                        selectforeground='white',
                                        height=15)
        self.index_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.index_listbox.yview)

        # Define vegetation indices with descriptions
        self.indices_info = {
            "NDVI": "Normalized Difference Vegetation Index\n- Best for general vegetation health\n- Uses: NIR + Red bands",
            "NDRE": "Normalized Difference Red Edge Index\n- Better for dense vegetation\n- Uses: NIR + Red Edge bands",
            "GNDVI": "Green Normalized Difference Vegetation Index\n- Sensitive to chlorophyll content\n- Uses: NIR + Green bands",
            "SAVI": "Soil Adjusted Vegetation Index\n- Corrects for soil brightness\n- Uses: NIR + Red bands (L=0.5)",
            "EVI": "Enhanced Vegetation Index\n- Minimizes atmospheric effects\n- Uses: NIR + Red + Blue bands",
            "ARVI": "Atmospherically Resistant Vegetation Index\n- Reduces atmospheric effects\n- Uses: NIR + Red + Blue bands",
            "OSAVI": "Optimized Soil Adjusted Vegetation Index\n- Optimized soil adjustment\n- Uses: NIR + Red bands",
            "GCI": "Green Chlorophyll Index\n- Estimates chlorophyll content\n- Uses: NIR + Green bands",
            "RECI": "Red Edge Chlorophyll Index\n- Chlorophyll in dense vegetation\n- Uses: NIR + Red Edge bands",
            "MSAVI": "Modified Soil Adjusted Vegetation Index\n- Self-adjusting soil factor\n- Uses: NIR + Red bands",
            "MCARI": "Modified Chlorophyll Absorption Ratio\n- Sensitive to chlorophyll\n- Uses: Red Edge + Red + Green bands",
            "TCARI": "Transformed Chlorophyll Absorption Ratio\n- Chlorophyll with soil adjustment\n- Uses: Red Edge + Red + Green bands",
            "MTVI": "Modified Triangular Vegetation Index\n- Leaf chlorophyll content\n- Uses: NIR + Red + Green bands",
            "NDWI": "Normalized Difference Water Index\n- Water content in vegetation\n- Uses: NIR + SWIR bands (Green as proxy)",
            "DVI": "Difference Vegetation Index\n- Simple vegetation difference\n- Uses: NIR - Red",
            "RVI": "Ratio Vegetation Index\n- Simple vegetation ratio\n- Uses: NIR / Red"
        }

        # Populate listbox
        for idx_name in self.indices_info.keys():
            self.index_listbox.insert(tk.END, idx_name)

        self.index_listbox.select_set(0)  # Select first item
        self.index_listbox.bind('<<ListboxSelect>>', self.on_index_select)

        # Index description label
        self.index_desc_var = tk.StringVar(value="Select an index to view details")
        desc_label = ttk.Label(index_section, textvariable=self.index_desc_var,
                               font=('SF Pro Text', 9), foreground='#8E8E93',
                               background='#ffffff', justify='left', wraplength=300)
        desc_label.pack(fill='x', padx=5, pady=(5, 10))

        # Band requirements indicator
        self.requirements_var = tk.StringVar(value="Required bands: None")
        req_label = ttk.Label(index_section, textvariable=self.requirements_var,
                              font=('SF Pro Text', 9), foreground='#FF3B30',
                              background='#ffffff', justify='left')
        req_label.pack(fill='x', padx=5, pady=(0, 10))

        # Process button
        btn_frame = ttk.Frame(control_frame, style='Card.TFrame')
        btn_frame.pack(fill='x', padx=10, pady=20)

        self.process_btn = ttk.Button(btn_frame, text="🚀 Calculate Vegetation Index",
                                      command=self.process_multiband, style='Accent.TButton')
        self.process_btn.pack(fill='x', pady=5)

        # Batch processing checkbox
        self.batch_process = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_frame, text="Batch Process All Indices",
                        variable=self.batch_process,
                        style='Modern.TRadiobutton').pack(pady=5)

        # Right panel for visualization
        viz_frame = ttk.Frame(parent, style='Card.TFrame')
        viz_frame.pack(side='right', fill='both', expand=True, pady=10)

        # Results display area
        self.results_frame = ttk.LabelFrame(viz_frame, text="ANALYSIS RESULTS", style='Card.TFrame')
        self.results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Initial placeholder
        self.setup_placeholder()

    def setup_metadata_tab(self, parent):
        """Setup the metadata display tab"""
        metadata_container = ttk.Frame(parent, style='Card.TFrame')
        metadata_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Top control frame
        control_frame = ttk.Frame(metadata_container, style='Card.TFrame')
        control_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(control_frame, text="Select File to View Metadata:",
                  style='CardTitle.TLabel').pack(side='left', padx=(0, 10))

        # File selector dropdown
        self.metadata_file_var = tk.StringVar()
        file_dropdown = ttk.Combobox(control_frame, textvariable=self.metadata_file_var,
                                     state='readonly', width=50)
        file_dropdown.pack(side='left', padx=(0, 10))

        # Refresh button
        ttk.Button(control_frame, text="🔄 Refresh",
                   command=self.refresh_metadata_list,
                   style='Secondary.TButton').pack(side='left')

        # Metadata display area
        metadata_display_frame = ttk.Frame(metadata_container, style='Card.TFrame')
        metadata_display_frame.pack(fill='both', expand=True)

        # Text widget for metadata display
        self.metadata_text = tk.Text(metadata_display_frame, wrap='word',
                                     font=('SF Mono', 10), bg='white',
                                     relief='flat', height=30)

        scrollbar = ttk.Scrollbar(metadata_display_frame, command=self.metadata_text.yview)
        self.metadata_text.configure(yscrollcommand=scrollbar.set)

        self.metadata_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Initialize file list
        self.refresh_metadata_list()

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

    def on_index_select(self, event):
        """Handle index selection from listbox"""
        selection = self.index_listbox.curselection()
        if selection:
            idx_name = self.index_listbox.get(selection[0])
            self.index_desc_var.set(self.indices_info.get(idx_name, "No description available"))

            # Update requirements
            requirements = self.get_index_requirements(idx_name)
            self.requirements_var.set(f"Required bands: {requirements}")

    def get_index_requirements(self, index_name):
        """Return required bands for a given index"""
        requirements = {
            "NDVI": "NIR, Red",
            "NDRE": "NIR, Red Edge",
            "GNDVI": "NIR, Green",
            "SAVI": "NIR, Red",
            "EVI": "NIR, Red, Blue",
            "ARVI": "NIR, Red, Blue",
            "OSAVI": "NIR, Red",
            "GCI": "NIR, Green",
            "RECI": "NIR, Red Edge",
            "MSAVI": "NIR, Red",
            "MCARI": "Red Edge, Red, Green",
            "TCARI": "Red Edge, Red, Green",
            "MTVI": "NIR, Red, Green",
            "NDWI": "NIR, Green",
            "DVI": "NIR, Red",
            "RVI": "NIR, Red"
        }
        return requirements.get(index_name, "Unknown")

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

            # Read and store metadata
            if band_name != "RGB":
                self.read_file_metadata(filename, band_name)

            # Refresh metadata file list
            self.refresh_metadata_list()

    def browse_shapefile(self):
        """Browse for shapefile"""
        filename = filedialog.askopenfilename(
            title="Select Shapefile",
            filetypes=[("Shapefiles", "*.shp"), ("All files", "*.*")]
        )
        if filename:
            self.path_shapefile.set(filename)
            self.update_band_status("Shapefile", True)
            self.update_status(f"Loaded shapefile: {os.path.basename(filename)}")

    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            self.update_status(f"Output directory set to: {directory}")

    def update_band_status(self, band_name, loaded):
        """Update band loading status"""
        self.band_status[band_name] = loaded

    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update()

    def read_file_metadata(self, filepath, band_name):
        """Read and store metadata from a raster file"""
        try:
            with rasterio.open(filepath) as src:
                metadata = {
                    'filename': os.path.basename(filepath),
                    'band_name': band_name,
                    'driver': src.driver,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs),
                    'transform': str(src.transform),
                    'bounds': str(src.bounds),
                    'nodata': src.nodata,
                    'resolution': (src.res[0], src.res[1]),
                    'profile': dict(src.profile)
                }

                # Add band-specific metadata if available
                if src.tags():
                    metadata['tags'] = dict(src.tags())

                self.band_metadata[band_name] = metadata

        except Exception as e:
            self.band_metadata[band_name] = {'error': str(e)}

    def show_file_metadata(self, filepath, band_name):
        """Display metadata for a specific file"""
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", f"No file selected for {band_name}")
            return

        try:
            with rasterio.open(filepath) as src:
                metadata = f"=== {band_name} Band Metadata ===\n\n"
                metadata += f"File: {os.path.basename(filepath)}\n"
                metadata += f"Size: {src.width} x {src.height} pixels\n"
                metadata += f"Bands: {src.count}\n"
                metadata += f"Data Type: {src.dtypes[0]}\n"
                metadata += f"CRS: {src.crs}\n"
                metadata += f"Resolution: {src.res[0]:.4f}, {src.res[1]:.4f}\n"
                metadata += f"NoData Value: {src.nodata}\n"
                metadata += f"Bounds: {src.bounds}\n\n"

                metadata += "=== Profile ===\n"
                for key, value in src.profile.items():
                    metadata += f"{key}: {value}\n"

                if src.tags():
                    metadata += "\n=== Tags ===\n"
                    for key, value in src.tags().items():
                        metadata += f"{key}: {value}\n"

                # Show in a new window
                self.show_metadata_window(metadata, f"{band_name} Metadata")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read metadata: {str(e)}")

    def show_metadata_window(self, metadata, title):
        """Display metadata in a new window"""
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("600x700")
        window.configure(bg='#f5f6fa')

        # Text widget for metadata
        text_widget = tk.Text(window, wrap='word', font=('SF Mono', 10),
                              bg='white', relief='flat')
        text_widget.insert('1.0', metadata)
        text_widget.config(state='disabled')

        # Scrollbar
        scrollbar = ttk.Scrollbar(window, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        # Pack widgets
        text_widget.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y')

    def refresh_metadata_list(self):
        """Refresh the list of files for metadata display"""
        files = []

        # Add individual band files
        for var, name in [(self.path_nir, "NIR"), (self.path_red, "Red"),
                          (self.path_green, "Green"), (self.path_re, "RedEdge"),
                          (self.path_blue, "Blue"), (self.path_rgb, "RGB")]:
            if var.get():
                files.append(f"{name}: {os.path.basename(var.get())}")

        # Update dropdown
        if hasattr(self, 'metadata_file_var'):
            # Find the combobox widget
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Toplevel):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Combobox):
                            child['values'] = files
                            break

        # Also update metadata text if a file is selected
        if hasattr(self, 'metadata_text') and self.metadata_file_var.get():
            self.display_selected_metadata()

    def display_selected_metadata(self):
        """Display metadata for selected file"""
        selection = self.metadata_file_var.get()
        if not selection:
            return

        # Parse selection to get band name and filename
        try:
            band_name = selection.split(":")[0].strip()
            filename = selection.split(":", 1)[1].strip()

            # Find the actual file path
            filepath = None
            band_vars = {
                "NIR": self.path_nir,
                "Red": self.path_red,
                "Green": self.path_green,
                "RedEdge": self.path_re,
                "Blue": self.path_blue,
                "RGB": self.path_rgb
            }

            if band_name in band_vars:
                filepath = band_vars[band_name].get()

            if filepath and os.path.exists(filepath):
                self.show_file_metadata(filepath, band_name)

        except Exception as e:
            self.metadata_text.delete('1.0', tk.END)
            self.metadata_text.insert('1.0', f"Error: {str(e)}")

    def load_band_data(self, band_name, band_path=None, rgb_path=None, rgb_band=None):
        """Load band data from either individual file or RGB file"""
        try:
            if rgb_path and rgb_band:
                # Load from RGB file
                with rasterio.open(rgb_path) as src:
                    if rgb_band <= src.count:
                        data = src.read(rgb_band)
                        meta = src.meta.copy()
                        meta.update({'count': 1})
                        return data, meta
                    else:
                        raise ValueError(f"Band {rgb_band} not found in RGB file")
            elif band_path:
                # Load from individual file
                with rasterio.open(band_path) as src:
                    data = src.read(1)
                    meta = src.meta.copy()
                    return data, meta
            else:
                raise ValueError(f"No source provided for {band_name} band")

        except Exception as e:
            raise Exception(f"Failed to load {band_name} band: {str(e)}")

    def crop_with_shapefile(self, data, meta, shapefile_path):
        """Crop raster data using shapefile"""
        try:
            # Read shapefile
            with fiona.open(shapefile_path, "r") as shapefile:
                shapes = [feature["geometry"] for feature in shapefile]

            # Apply buffer if specified
            if self.crop_buffer.get() > 0:
                gdf = gpd.read_file(shapefile_path)
                gdf['geometry'] = gdf.buffer(self.crop_buffer.get())
                shapes = [shape(geom) for geom in gdf.geometry]

            # Crop the raster
            cropped_data, cropped_transform = mask(data, shapes, crop=True)

            # Update metadata
            meta.update({
                "height": cropped_data.shape[1],
                "width": cropped_data.shape[2],
                "transform": cropped_transform
            })

            return cropped_data[0], meta

        except Exception as e:
            raise Exception(f"Failed to crop with shapefile: {str(e)}")

    def calculate_index(self, index_name, bands):
        """Calculate various vegetation indices"""
        try:
            # Convert all bands to float64 for calculation
            bands = {k: v.astype('float64') for k, v in bands.items()}

            # Define index calculations
            if index_name == "NDVI":
                nir, red = bands.get('NIR'), bands.get('Red')
                if nir is None or red is None:
                    raise ValueError("NDVI requires NIR and Red bands")
                denominator = nir + red
                valid_mask = denominator != 0
                result = np.full_like(nir, np.nan)
                result[valid_mask] = (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]

            elif index_name == "NDRE":
                nir, re = bands.get('NIR'), bands.get('RedEdge')
                if nir is None or re is None:
                    raise ValueError("NDRE requires NIR and RedEdge bands")
                denominator = nir + re
                valid_mask = denominator != 0
                result = np.full_like(nir, np.nan)
                result[valid_mask] = (nir[valid_mask] - re[valid_mask]) / denominator[valid_mask]

            elif index_name == "GNDVI":
                nir, green = bands.get('NIR'), bands.get('Green')
                if nir is None or green is None:
                    raise ValueError("GNDVI requires NIR and Green bands")
                denominator = nir + green
                valid_mask = denominator != 0
                result = np.full_like(nir, np.nan)
                result[valid_mask] = (nir[valid_mask] - green[valid_mask]) / denominator[valid_mask]

            elif index_name == "SAVI":
                nir, red = bands.get('NIR'), bands.get('Red')
                if nir is None or red is None:
                    raise ValueError("SAVI requires NIR and Red bands")
                L = 0.5  # Soil adjustment factor
                denominator = nir + red + L
                valid_mask = denominator != 0
                result = np.full_like(nir, np.nan)
                result[valid_mask] = (1 + L) * (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]

            elif index_name == "EVI":
                nir, red, blue = bands.get('NIR'), bands.get('Red'), bands.get('Blue')
                if nir is None or red is None or blue is None:
                    raise ValueError("EVI requires NIR, Red, and Blue bands")
                denominator = nir + 6 * red - 7.5 * blue + 1
                valid_mask = denominator != 0
                result = np.full_like(nir, np.nan)
                result[valid_mask] = 2.5 * (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]

            elif index_name == "ARVI":
                nir, red, blue = bands.get('NIR'), bands.get('Red'), bands.get('Blue')
                if nir is None or red is None or blue is None:
                    raise ValueError("ARVI requires NIR, Red, and Blue bands")
                rb = red - 2 * (red - blue)
                denominator = nir + rb
                valid_mask = denominator != 0
                result = np.full_like(nir, np.nan)
                result[valid_mask] = (nir[valid_mask] - rb[valid_mask]) / denominator[valid_mask]

            elif index_name == "OSAVI":
                nir, red = bands.get('NIR'), bands.get('Red')
                if nir is None or red is None:
                    raise ValueError("OSAVI requires NIR and Red bands")
                denominator = nir + red + 0.16
                valid_mask = denominator != 0
                result = np.full_like(nir, np.nan)
                result[valid_mask] = (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]

            elif index_name == "GCI":
                nir, green = bands.get('NIR'), bands.get('Green')
                if nir is None or green is None:
                    raise ValueError("GCI requires NIR and Green bands")
                valid_mask = green != 0
                result = np.full_like(nir, np.nan)
                result[valid_mask] = (nir[valid_mask] / green[valid_mask]) - 1

            elif index_name == "RECI":
                nir, re = bands.get('NIR'), bands.get('RedEdge')
                if nir is None or re is None:
                    raise ValueError("RECI requires NIR and RedEdge bands")
                valid_mask = re != 0
                result = np.full_like(nir, np.nan)
                result[valid_mask] = (nir[valid_mask] / re[valid_mask]) - 1

            elif index_name == "MSAVI":
                nir, red = bands.get('NIR'), bands.get('Red')
                if nir is None or red is None:
                    raise ValueError("MSAVI requires NIR and Red bands")
                result = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2

            elif index_name == "MCARI":
                re, red, green = bands.get('RedEdge'), bands.get('Red'), bands.get('Green')
                if re is None or red is None or green is None:
                    raise ValueError("MCARI requires RedEdge, Red, and Green bands")
                result = ((re - red) - 0.2 * (re - green)) * (re / red)

            elif index_name == "TCARI":
                re, red, green = bands.get('RedEdge'), bands.get('Red'), bands.get('Green')
                if re is None or red is None or green is None:
                    raise ValueError("TCARI requires RedEdge, Red, and Green bands")
                result = 3 * ((re - red) - 0.2 * (re - green) * (re / red))

            elif index_name == "MTVI":
                nir, red, green = bands.get('NIR'), bands.get('Red'), bands.get('Green')
                if nir is None or red is None or green is None:
                    raise ValueError("MTVI requires NIR, Red, and Green bands")
                result = 1.2 * (1.2 * (nir - green) - 2.5 * (red - green))

            elif index_name == "NDWI":
                nir, green = bands.get('NIR'), bands.get('Green')
                if nir is None or green is None:
                    raise ValueError("NDWI requires NIR and Green bands")
                denominator = nir + green
                valid_mask = denominator != 0
                result = np.full_like(nir, np.nan)
                result[valid_mask] = (green[valid_mask] - nir[valid_mask]) / denominator[valid_mask]

            elif index_name == "DVI":
                nir, red = bands.get('NIR'), bands.get('Red')
                if nir is None or red is None:
                    raise ValueError("DVI requires NIR and Red bands")
                result = nir - red

            elif index_name == "RVI":
                nir, red = bands.get('NIR'), bands.get('Red')
                if nir is None or red is None:
                    raise ValueError("RVI requires NIR and Red bands")
                valid_mask = red != 0
                result = np.full_like(nir, np.nan)
                result[valid_mask] = nir[valid_mask] / red[valid_mask]

            else:
                raise ValueError(f"Unknown index: {index_name}")

            return result.astype('float32')

        except Exception as e:
            raise Exception(f"Failed to calculate {index_name}: {str(e)}")

    def process_multiband(self):
        """Main processing function with enhanced features"""
        try:
            # Get selected index
            selection = self.index_listbox.curselection()
            if not selection:
                messagebox.showerror("Selection Error", "Please select a vegetation index!")
                return

            indices_to_process = []
            if self.batch_process.get():
                # Process all indices
                indices_to_process = list(self.indices_info.keys())
            else:
                # Process single index
                idx_name = self.index_listbox.get(selection[0])
                indices_to_process = [idx_name]

            # Check for RGB file or individual files
            rgb_path = self.path_rgb.get() if self.path_rgb.get() else None

            # Load required bands
            bands_to_load = {}
            meta = None

            for idx_name in indices_to_process:
                requirements = self.get_index_requirements(idx_name)
                required_bands = [band.strip() for band in requirements.split(',')]

                for band_name in required_bands:
                    if band_name not in bands_to_load:
                        # Determine source for this band
                        if rgb_path:
                            # Try to load from RGB file
                            rgb_band = None
                            if band_name == "Red":
                                rgb_band = self.rgb_red_band.get()
                            elif band_name == "Green":
                                rgb_band = self.rgb_green_band.get()
                            elif band_name == "Blue":
                                rgb_band = self.rgb_blue_band.get()
                            elif band_name == "NIR":
                                rgb_band = self.rgb_nir_band.get()

                            if rgb_band:
                                data, band_meta = self.load_band_data(
                                    band_name,
                                    rgb_path=rgb_path,
                                    rgb_band=rgb_band
                                )
                                bands_to_load[band_name] = data
                                if meta is None:
                                    meta = band_meta
                        else:
                            # Try to load from individual file
                            band_var = None
                            if band_name == "Red":
                                band_var = self.path_red
                            elif band_name == "Green":
                                band_var = self.path_green
                            elif band_name == "Blue":
                                band_var = self.path_blue
                            elif band_name == "NIR":
                                band_var = self.path_nir
                            elif band_name == "RedEdge":
                                band_var = self.path_re

                            if band_var and band_var.get():
                                data, band_meta = self.load_band_data(
                                    band_name,
                                    band_path=band_var.get()
                                )
                                bands_to_load[band_name] = data
                                if meta is None:
                                    meta = band_meta
                            else:
                                raise ValueError(f"Required {band_name} band not found")

            # Check if we have all required bands
            if not bands_to_load:
                messagebox.showerror("Data Error", "No band data loaded!")
                return

            # Apply cropping if enabled
            if self.crop_enabled.get() and self.path_shapefile.get():
                self.update_status("Applying shapefile cropping...")
                for band_name in bands_to_load.keys():
                    bands_to_load[band_name], meta = self.crop_with_shapefile(
                        bands_to_load[band_name],
                        meta,
                        self.path_shapefile.get()
                    )

            # Process each index
            results = {}
            for idx_name in indices_to_process:
                try:
                    self.update_status(f"Calculating {idx_name}...")

                    # Calculate index
                    result = self.calculate_index(idx_name, bands_to_load)

                    # Calculate statistics
                    valid_mask = ~np.isnan(result)
                    valid_pixels = np.sum(valid_mask)
                    total_pixels = result.size

                    if valid_pixels > 0:
                        stats = {
                            'valid_pixels': valid_pixels,
                            'total_pixels': total_pixels,
                            'valid_percentage': (valid_pixels / total_pixels) * 100,
                            'min_value': np.nanmin(result),
                            'max_value': np.nanmax(result),
                            'mean_value': np.nanmean(result),
                            'std_value': np.nanstd(result),
                            'median_value': np.nanmedian(result)
                        }
                    else:
                        stats = {
                            'valid_pixels': 0,
                            'total_pixels': total_pixels,
                            'valid_percentage': 0,
                            'min_value': 0,
                            'max_value': 0,
                            'mean_value': 0,
                            'std_value': 0,
                            'median_value': 0
                        }

                    # Save result
                    output_filename = f"{self.output_filename.get()}_{idx_name}.tif"
                    output_path = os.path.join(self.output_dir.get(), output_filename)

                    # Ensure output directory exists
                    os.makedirs(self.output_dir.get(), exist_ok=True)

                    # Update metadata for output
                    output_meta = meta.copy()
                    output_meta.update({
                        'dtype': 'float32',
                        'count': 1,
                        'compress': 'none',
                        'nodata': np.nan,
                        'tiled': False,
                        'interleave': 'band'
                    })

                    # Save GeoTIFF
                    with rasterio.open(output_path, 'w', **output_meta) as dst:
                        dst.write(result, 1)

                    results[idx_name] = {
                        'data': result,
                        'stats': stats,
                        'output_path': output_path
                    }

                    self.update_status(f"{idx_name} calculation complete")

                except Exception as e:
                    messagebox.showwarning(f"{idx_name} Error",
                                           f"Failed to calculate {idx_name}: {str(e)}")

            # Show success message
            if results:
                success_msg = f"✅ Processing Complete!\n\n"
                success_msg += f"📊 Results Summary:\n"
                for idx_name, result_data in results.items():
                    stats = result_data['stats']
                    success_msg += f"• {idx_name}: {stats['valid_pixels']:,} valid pixels ({stats['valid_percentage']:.1f}%)\n"
                    success_msg += f"  Range: {stats['min_value']:.3f} to {stats['max_value']:.3f}\n"

                success_msg += f"\n💾 Files saved to: {self.output_dir.get()}"

                messagebox.showinfo("Analysis Complete", success_msg)

                # Display first result
                first_idx = list(results.keys())[0]
                self.show_enhanced_preview(
                    results[first_idx]['data'],
                    first_idx,
                    results[first_idx]['stats']
                )

        except Exception as e:
            error_msg = f"Processing Error:\n\n{str(e)}"
            messagebox.showerror("Calculation Failed", error_msg)
            self.update_status("Calculation failed")

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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                       gridspec_kw={'width_ratios': [3, 1]},
                                       facecolor='#ffffff')

        # Main visualization
        im1 = ax1.imshow(data, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        ax1.set_title(f'{title} Vegetation Index\n',
                      fontsize=16, fontweight='bold', pad=20, color='#1C1C1E')
        ax1.axis('off')

        # Enhanced colorbar
        cbar = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
        cbar.set_label('Index Value', rotation=270, labelpad=15, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)

        # Histogram
        valid_data = data[~np.isnan(data)].flatten()
        if len(valid_data) > 0:
            ax2.hist(valid_data, bins=50, color='#34C759', alpha=0.7, edgecolor='black')
            ax2.axvline(stats['mean_value'], color='#FF3B30', linestyle='--',
                        label=f"Mean: {stats['mean_value']:.3f}")
            ax2.axvline(stats['median_value'], color='#5856D6', linestyle=':',
                        label=f"Median: {stats['median_value']:.3f}")
            ax2.set_xlabel('Index Value', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.set_title('Value Distribution', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

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
            ("🎯 Median", f"{stats['median_value']:.4f}"),
            ("💾 File Status", "✓ Saved" if os.path.exists(stats.get('output_path', '')) else "✗ Failed")
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

        if stats.get('output_path'):
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