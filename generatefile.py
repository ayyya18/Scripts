import pandas as pd
import numpy as np
import os
import random

def generate_dataset():
    # Seed untuk reproducibility
    np.random.seed(42)
    random.seed(42)
    
    n_samples_train = 5000  # Total data training
    
    print("=== MEMULAI GENERASI DATA ===")
    
    # =========================================================================
    # 1. GENERATE DATA TRAINING (File 1)
    # =========================================================================
    print("Generating data training...")
    
    # Variasi parameter untuk membuat data training yang kaya
    n_clusters = 8  # Jumlah cluster untuk variasi pattern
    cluster_centers_ndvi = np.linspace(0.3, 0.9, n_clusters)
    
    train_data = []
    
    for cluster_idx in range(n_clusters):
        # Setiap cluster memiliki karakteristik berbeda
        cluster_size = n_samples_train // n_clusters
        remaining = n_samples_train % n_clusters
        current_size = cluster_size + (1 if cluster_idx < remaining else 0)
        
        # Center untuk cluster ini
        center_ndvi = cluster_centers_ndvi[cluster_idx]
        
        # Generate NDVI dengan variasi berdasarkan cluster
        ndvi = np.random.normal(center_ndvi, 0.15, current_size)
        
        # Setiap cluster memiliki hubungan yang berbeda antara indeks
        if cluster_idx % 3 == 0:
            # Pattern 1: NDRE dan GNDVI sangat terkorelasi dengan NDVI
            ndre = ndvi * 0.85 + np.random.normal(0, 0.03, current_size)
            gndvi = ndvi * 0.95 + np.random.normal(0, 0.02, current_size)
        elif cluster_idx % 3 == 1:
            # Pattern 2: NDRE lebih independen
            ndre = ndvi * 0.7 + np.random.normal(0.1, 0.08, current_size)
            gndvi = ndvi * 0.8 + np.random.normal(0.05, 0.06, current_size)
        else:
            # Pattern 3: GNDVI lebih dominan
            ndre = ndvi * 0.9 + np.random.normal(-0.05, 0.04, current_size)
            gndvi = ndvi * 1.1 + np.random.normal(-0.1, 0.05, current_size)
        
        # Setiap cluster memiliki rumus serapan K yang berbeda
        if cluster_idx % 4 == 0:
            # Pattern A: NDRE dominan
            base_k = 8 + (30 * ndre) + (10 * ndvi) + (3 * gndvi)
        elif cluster_idx % 4 == 1:
            # Pattern B: NDVI dominan
            base_k = 12 + (15 * ndre) + (25 * ndvi) + (8 * gndvi)
        elif cluster_idx % 4 == 2:
            # Pattern C: GNDVI dominan
            base_k = 5 + (20 * ndre) + (12 * ndvi) + (18 * gndvi)
        else:
            # Pattern D: Seimbang
            base_k = 10 + (22 * ndre) + (18 * ndvi) + (10 * gndvi)
        
        # Noise yang berbeda untuk setiap cluster
        if cluster_idx < n_clusters // 2:
            noise = np.random.normal(0, 1.0, current_size)  # Low noise
        else:
            noise = np.random.normal(0, 2.5, current_size)  # High noise
        
        serapan_k = base_k + noise
        
        # Clip values
        ndvi = np.clip(ndvi, 0.1, 0.95)
        ndre = np.clip(ndre, 0.1, 0.95)
        gndvi = np.clip(gndvi, 0.1, 0.95)
        serapan_k = np.clip(serapan_k, 5, 60)
        
        for i in range(current_size):
            train_data.append({
                'NDVI': round(ndvi[i], 4),
                'NDRE': round(ndre[i], 4),
                'GNDVI': round(gndvi[i], 4),
                'Serapan_K': round(serapan_k[i], 2)
            })
    
    # Acak data training
    random.shuffle(train_data)
    df_train = pd.DataFrame(train_data)
    
    # Simpan data training
    file_train = '1_data_training.csv'
    df_train.to_csv(file_train, index=False)
    print(f"✓ {file_train} berhasil dibuat ({len(df_train)} sampel)")
    
    # =========================================================================
    # 2. GENERATE 50 DATA INPUT PREDIKSI & JAWABAN
    # =========================================================================
    
    # Buat folder untuk organisasi yang lebih baik
    os.makedirs('data_prediksi', exist_ok=True)
    os.makedirs('data_jawaban', exist_ok=True)
    
    # Berbagai skenario untuk data testing yang beragam
    scenarios = [
        # (nama_scenario, base_samples, ndvi_range, base_noise, pattern_type)
        ("normal", 100, (0.3, 0.9), 1.5, "balanced"),
        ("high_ndvi", 80, (0.7, 0.95), 1.0, "ndvi_dominant"),
        ("low_ndvi", 80, (0.3, 0.5), 2.0, "ndre_dominant"),
        ("high_variability", 120, (0.2, 0.95), 3.0, "mixed"),
        ("precise", 60, (0.5, 0.8), 0.5, "gndvi_dominant"),
        ("extreme_low", 70, (0.1, 0.4), 1.8, "ndvi_dominant"),
        ("extreme_high", 70, (0.8, 1.0), 1.8, "ndre_dominant"),
        ("uniform", 90, (0.4, 0.9), 1.2, "balanced")
    ]
    
    total_test_files = 50
    files_per_scenario = max(1, total_test_files // len(scenarios))
    
    file_counter = 1
    index_data = []
    
    for scenario_name, base_samples, ndvi_range, base_noise, pattern_type in scenarios:
        for scenario_file in range(files_per_scenario):
            if file_counter > total_test_files:
                break
                
            print(f"Generating test data {file_counter}/50 - Scenario: {scenario_name}_{scenario_file+1}")
            
            # Variasi dalam scenario yang sama
            n_samples = base_samples + random.randint(-20, 20)
            current_noise = base_noise * random.uniform(0.8, 1.2)
            
            # Generate data dengan karakteristik unik
            if scenario_name == "high_variability":
                # Mixed distribution - pastikan total sampel sesuai
                part1 = n_samples // 4
                part2 = n_samples // 2
                part3 = n_samples - part1 - part2  # Sisa untuk bagian ketiga
                
                ndvi = np.concatenate([
                    np.random.uniform(0.2, 0.4, part1),
                    np.random.uniform(0.4, 0.7, part2),
                    np.random.uniform(0.7, 0.95, part3)
                ])
            elif scenario_name == "uniform":
                ndvi = np.random.uniform(ndvi_range[0], ndvi_range[1], n_samples)
            else:
                # Normal distribution around center
                center = (ndvi_range[0] + ndvi_range[1]) / 2
                span = ndvi_range[1] - ndvi_range[0]
                ndvi = np.random.normal(center, span/4, n_samples)
            
            # Pastikan n_samples sesuai dengan panjang ndvi
            n_samples = len(ndvi)
            
            # Acak urutan
            np.random.shuffle(ndvi)
            
            # Terapkan pattern type yang berbeda
            if pattern_type == "ndvi_dominant":
                ndre = ndvi * 0.75 + np.random.normal(0.1, 0.1, n_samples)
                gndvi = ndvi * 0.85 + np.random.normal(0.05, 0.08, n_samples)
                base_k = 12 + (15 * ndre) + (28 * ndvi) + (5 * gndvi)
            elif pattern_type == "ndre_dominant":
                ndre = ndvi * 0.9 + np.random.normal(0, 0.05, n_samples)
                gndvi = ndvi * 0.7 + np.random.normal(0.15, 0.1, n_samples)
                base_k = 8 + (32 * ndre) + (12 * ndvi) + (8 * gndvi)
            elif pattern_type == "gndvi_dominant":
                ndre = ndvi * 0.8 + np.random.normal(0.05, 0.06, n_samples)
                gndvi = ndvi * 1.05 + np.random.normal(0, 0.04, n_samples)
                base_k = 6 + (18 * ndre) + (14 * ndvi) + (22 * gndvi)
            else:  # balanced
                ndre = ndvi * 0.82 + np.random.normal(0.02, 0.07, n_samples)
                gndvi = ndvi * 0.92 + np.random.normal(0.03, 0.05, n_samples)
                base_k = 10 + (24 * ndre) + (16 * ndvi) + (10 * gndvi)
            
            # Tambahkan outlier untuk beberapa dataset
            if file_counter % 7 == 0:  # Setiap dataset ke-7 memiliki outlier
                outlier_count = max(1, n_samples // 20)
                outlier_indices = random.sample(range(n_samples), outlier_count)
                for idx in outlier_indices:
                    ndvi[idx] = random.uniform(0.1, 1.0)
                    ndre[idx] = random.uniform(0.1, 1.0)
                    gndvi[idx] = random.uniform(0.1, 1.0)
            
            # Clip values
            ndvi = np.clip(ndvi, 0.1, 1.0)
            ndre = np.clip(ndre, 0.1, 1.0)
            gndvi = np.clip(gndvi, 0.1, 1.0)
            
            # Generate serapan K dengan noise yang berbeda
            noise = np.random.normal(0, current_noise, n_samples)
            serapan_k = base_k + noise
            serapan_k = np.clip(serapan_k, 5, 65)
            
            # Buat DataFrame untuk input prediksi
            df_input = pd.DataFrame({
                'NDVI': np.round(ndvi, 4),
                'NDRE': np.round(ndre, 4),
                'GNDVI': np.round(gndvi, 4)
            })
            
            # Buat DataFrame untuk jawaban
            df_actual = pd.DataFrame({
                'Serapan_K': np.round(serapan_k, 2)
            })
            
            # Simpan file
            file_input = f'data_prediksi/2_data_input_prediksi_{file_counter}.csv'
            file_actual = f'data_jawaban/3_data_aktual_jawaban_{file_counter}.csv'
            
            df_input.to_csv(file_input, index=False)
            df_actual.to_csv(file_actual, index=False)
            
            # Kumpulkan statistik untuk file indeks
            index_data.append({
                'No': file_counter,
                'File_Input': file_input,
                'File_Actual': file_actual,
                'Jumlah_Sampel': len(df_input),
                'NDVI_Rata2': round(df_input['NDVI'].mean(), 3),
                'NDVI_Std': round(df_input['NDVI'].std(), 3),
                'Serapan_K_Rata2': round(df_actual['Serapan_K'].mean(), 2),
                'Serapan_K_Std': round(df_actual['Serapan_K'].std(), 2)
            })
            
            file_counter += 1
    
    # Jika masih kurang dari 50 file, tambahkan dengan skenario random
    while file_counter <= total_test_files:
        print(f"Generating additional test data {file_counter}/50")
        
        # Pilih skenario random
        scenario = random.choice(scenarios)
        scenario_name, base_samples, ndvi_range, base_noise, pattern_type = scenario
        
        n_samples = base_samples + random.randint(-20, 20)
        current_noise = base_noise * random.uniform(0.8, 1.2)
        
        # Generate simple uniform distribution untuk file tambahan
        ndvi = np.random.uniform(ndvi_range[0], ndvi_range[1], n_samples)
        n_samples = len(ndvi)
        
        # Pattern balanced untuk file tambahan
        ndre = ndvi * 0.82 + np.random.normal(0.02, 0.07, n_samples)
        gndvi = ndvi * 0.92 + np.random.normal(0.03, 0.05, n_samples)
        base_k = 10 + (24 * ndre) + (16 * ndvi) + (10 * gndvi)
        
        # Clip values
        ndvi = np.clip(ndvi, 0.1, 1.0)
        ndre = np.clip(ndre, 0.1, 1.0)
        gndvi = np.clip(gndvi, 0.1, 1.0)
        
        # Generate serapan K
        noise = np.random.normal(0, current_noise, n_samples)
        serapan_k = base_k + noise
        serapan_k = np.clip(serapan_k, 5, 65)
        
        # Buat dan simpan DataFrame
        df_input = pd.DataFrame({
            'NDVI': np.round(ndvi, 4),
            'NDRE': np.round(ndre, 4),
            'GNDVI': np.round(gndvi, 4)
        })
        
        df_actual = pd.DataFrame({
            'Serapan_K': np.round(serapan_k, 2)
        })
        
        file_input = f'data_prediksi/2_data_input_prediksi_{file_counter}.csv'
        file_actual = f'data_jawaban/3_data_aktual_jawaban_{file_counter}.csv'
        
        df_input.to_csv(file_input, index=False)
        df_actual.to_csv(file_actual, index=False)
        
        # Kumpulkan statistik
        index_data.append({
            'No': file_counter,
            'File_Input': file_input,
            'File_Actual': file_actual,
            'Jumlah_Sampel': len(df_input),
            'NDVI_Rata2': round(df_input['NDVI'].mean(), 3),
            'NDVI_Std': round(df_input['NDVI'].std(), 3),
            'Serapan_K_Rata2': round(df_actual['Serapan_K'].mean(), 2),
            'Serapan_K_Std': round(df_actual['Serapan_K'].std(), 2)
        })
        
        file_counter += 1
    
    # =========================================================================
    # 3. GENERATE FILE INDEKS UNTUK KEMUDAHAN
    # =========================================================================
    
    df_index = pd.DataFrame(index_data)
    df_index.to_csv('0_index_file_testing.csv', index=False)
    
    # =========================================================================
    # 4. SUMMARY
    # =========================================================================
    
    print("\n" + "="*50)
    print("=== GENERATE DATA SUKSES ===")
    print("="*50)
    print(f"1. {file_train}")
    print(f"   - Jumlah sampel: {len(df_train)}")
    print(f"   - Range NDVI: {df_train['NDVI'].min():.3f} - {df_train['NDVI'].max():.3f}")
    print(f"   - Range Serapan K: {df_train['Serapan_K'].min():.1f} - {df_train['Serapan_K'].max():.1f}")
    
    print(f"\n2. Data Input Prediksi:")
    print(f"   - 50 file di folder 'data_prediksi/'")
    print(f"   - Beragam skenario: normal, high_ndvi, low_ndvi, high_variability, dll.")
    
    print(f"\n3. Data Jawaban Aktual:")
    print(f"   - 50 file di folder 'data_jawaban/'")
    
    print(f"\n4. File Tambahan:")
    print(f"   - 0_index_file_testing.csv (Indeks semua file testing)")
    
    print(f"\nStatistik Training Data:")
    print(f"   - Korelasi NDVI-SerapanK: {df_train['NDVI'].corr(df_train['Serapan_K']):.3f}")
    print(f"   - Korelasi NDRE-SerapanK: {df_train['NDRE'].corr(df_train['Serapan_K']):.3f}")
    print(f"   - Korelasi GNDVI-SerapanK: {df_train['GNDVI'].corr(df_train['Serapan_K']):.3f}")
    
    # Hitung variasi antar file testing
    ndvi_means = [data['NDVI_Rata2'] for data in index_data]
    k_means = [data['Serapan_K_Rata2'] for data in index_data]
    
    print(f"\nVariasi Data Testing:")
    print(f"   - Rata2 NDVI: {min(ndvi_means):.3f} - {max(ndvi_means):.3f}")
    print(f"   - Rata2 Serapan K: {min(k_means):.1f} - {max(k_means):.1f}")
    print(f"   - Total sampel testing: {sum([data['Jumlah_Sampel'] for data in index_data])}")


if __name__ == "__main__":
    generate_dataset()