import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Set seed untuk reproducibility
np.random.seed(100)

# Daftar fitur yang konsisten untuk training dan prediksi
FEATURES = ['NDVI', 'NDRE', 'GNDVI', 'SAVI', 'EVI', 'OSAVI', 'MSAVI', 'GCI', 'RECI', 'NDWI']


def generate_training_data(jumlah_sampel=200):
    """Generate data training dengan berbagai indeks vegetasi"""
    print("📊 Generating training data...")

    # Generate nilai dasar spektral yang lebih realistis
    blue = np.random.uniform(0.05, 0.25, jumlah_sampel)  # Blue reflectance
    green = np.random.uniform(0.1, 0.4, jumlah_sampel)  # Green reflectance
    red = np.random.uniform(0.05, 0.3, jumlah_sampel)  # Red reflectance
    nir = np.random.uniform(0.3, 0.8, jumlah_sampel)  # NIR reflectance
    red_edge = np.random.uniform(0.2, 0.6, jumlah_sampel)  # Red Edge reflectance

    # Hitung berbagai indeks vegetasi
    ndvi = (nir - red) / (nir + red + 1e-8)
    gndvi = (nir - green) / (nir + green + 1e-8)
    ndre = (nir - red_edge) / (nir + red_edge + 1e-8)
    savi = (1.5 * (nir - red)) / (nir + red + 0.5 + 1e-8)
    evi = (2.5 * (nir - red)) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
    osavi = (1.16 * (nir - red)) / (nir + red + 0.16 + 1e-8)
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
    gci = (nir / green) - 1
    reci = (nir / red_edge) - 1
    ndwi = (green - nir) / (green + nir + 1e-8)

    # Simulasi serapan kalium dengan hubungan yang lebih realistis
    base_serapan = 25  # Base level serapan kalium

    # Kontribusi masing-masing indeks (dalam persentase)
    serapan_k = (
            base_serapan +
            (12 * ndvi) +  # NDVI memberikan kontribusi sedang
            (15 * ndre) +  # NDRE penting untuk nitrogen yang berhubungan dengan K
            (8 * gndvi) +  # GNDVI untuk klorofil
            (10 * savi) +  # SAVI koreksi tanah
            (6 * evi) +  # EVI koreksi atmosfer
            (9 * osavi) +  # OSAVI optimized
            (7 * msavi) +  # MSAVI modified
            (5 * gci) +  # GCI green chlorophyll
            (13 * reci) +  # RECI penting untuk red edge
            (4 * ndwi)  # NDWI untuk kandungan air
    )

    # Tambahkan noise yang realistis
    noise = np.random.normal(0, 2.5, jumlah_sampel)
    serapan_k += noise

    # Pastikan nilai positif
    serapan_k = np.maximum(serapan_k, 5)

    # Buat DataFrame dengan urutan fitur yang konsisten
    data_dict = {
        'NDVI': ndvi,
        'NDRE': ndre,
        'GNDVI': gndvi,
        'SAVI': savi,
        'EVI': evi,
        'OSAVI': osavi,
        'MSAVI': msavi,
        'GCI': gci,
        'RECI': reci,
        'NDWI': ndwi,
        'Serapan_K': serapan_k
    }

    # Pastikan urutan kolom konsisten
    df_latihan = pd.DataFrame(data_dict)
    df_latihan = df_latihan[FEATURES + ['Serapan_K']]  # Urutan yang konsisten

    # Round values
    df_latihan = df_latihan.round(4)

    return df_latihan


def generate_prediction_data(jumlah_sampel=50):
    """Generate data prediksi dengan variasi yang lebih luas"""
    print("🔮 Generating prediction data...")

    # Generate nilai dengan range yang lebih luas untuk testing
    blue = np.random.uniform(0.03, 0.3, jumlah_sampel)
    green = np.random.uniform(0.08, 0.45, jumlah_sampel)
    red = np.random.uniform(0.03, 0.35, jumlah_sampel)
    nir = np.random.uniform(0.25, 0.85, jumlah_sampel)
    red_edge = np.random.uniform(0.15, 0.7, jumlah_sampel)

    # Hitung indeks vegetasi - menggunakan rumus yang sama dengan training
    ndvi = (nir - red) / (nir + red + 1e-8)
    gndvi = (nir - green) / (nir + green + 1e-8)
    ndre = (nir - red_edge) / (nir + red_edge + 1e-8)
    savi = (1.5 * (nir - red)) / (nir + red + 0.5 + 1e-8)
    evi = (2.5 * (nir - red)) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
    osavi = (1.16 * (nir - red)) / (nir + red + 0.16 + 1e-8)
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
    gci = (nir / green) - 1
    reci = (nir / red_edge) - 1
    ndwi = (green - nir) / (green + nir + 1e-8)

    # Buat DataFrame dengan urutan fitur yang sama seperti training
    data_dict = {
        'NDVI': ndvi,
        'NDRE': ndre,
        'GNDVI': gndvi,
        'SAVI': savi,
        'EVI': evi,
        'OSAVI': osavi,
        'MSAVI': msavi,
        'GCI': gci,
        'RECI': reci,
        'NDWI': ndwi
    }

    # Pastikan urutan kolom sama dengan training (tanpa Serapan_K)
    df_prediksi = pd.DataFrame(data_dict)
    df_prediksi = df_prediksi[FEATURES]  # Urutan yang konsisten

    df_prediksi = df_prediksi.round(4)

    return df_prediksi


def validate_features(df_training, df_prediction, model=None):
    """Validasi konsistensi fitur antara data training dan prediksi"""
    print("🔍 Validating feature consistency...")

    training_features = [col for col in df_training.columns if col != 'Serapan_K']
    prediction_features = df_prediction.columns.tolist()

    print(f"📊 Training features ({len(training_features)}): {sorted(training_features)}")
    print(f"📈 Prediction features ({len(prediction_features)}): {sorted(prediction_features)}")

    # Cek perbedaan
    missing_in_prediction = set(training_features) - set(prediction_features)
    extra_in_prediction = set(prediction_features) - set(training_features)

    if missing_in_prediction:
        print(f"❌ Missing in prediction: {missing_in_prediction}")
    if extra_in_prediction:
        print(f"❌ Extra in prediction: {extra_in_prediction}")

    if model is not None:
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_.tolist()
            print(f"🤖 Model features ({len(model_features)}): {sorted(model_features)}")

            missing_in_model = set(training_features) - set(model_features)
            if missing_in_model:
                print(f"⚠️  Features in training but not in model: {missing_in_model}")

    return len(missing_in_prediction) == 0 and len(extra_in_prediction) == 0


def train_and_save_model(df_latihan):
    """Train model dan simpan"""
    print("🎯 Training machine learning model...")

    # Pisahkan features dan target
    X = df_latihan[FEATURES]  # Gunakan urutan fitur yang konsisten
    y = df_latihan['Serapan_K']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    # Evaluasi model
    from sklearn.metrics import r2_score, mean_squared_error
    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"✅ Model training completed!")
    print(f"📊 R² Score: {r2:.4f}")
    print(f"📉 RMSE: {rmse:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n🌿 Feature Importance:")
    print(feature_importance.to_string(index=False))

    return rf_model


def predict_with_trained_model():
    """Fungsi untuk melakukan prediksi menggunakan model yang sudah dilatih"""
    try:
        # Load model
        print("📥 Loading trained model...")
        rf_model = joblib.load('model_rf_hara.joblib')

        # Load prediction data
        df_prediksi = pd.read_csv('data_prediksi_indeks_hara.csv')

        # Validasi fitur sebelum prediksi
        df_training_template = pd.read_csv('data_training_indeks_hara.csv')
        is_valid = validate_features(df_training_template, df_prediksi, rf_model)

        if not is_valid:
            print("⚠️  Feature mismatch detected! Adjusting prediction data...")
            # Pastikan hanya fitur yang ada di training yang digunakan
            training_features = [col for col in df_training_template.columns if col != 'Serapan_K']
            df_prediksi = df_prediksi[training_features]  # Pilih hanya fitur yang sesuai

        # Pastikan urutan fitur sama dengan saat training
        if hasattr(rf_model, 'feature_names_in_'):
            df_prediksi = df_prediksi[rf_model.feature_names_in_]

        # Lakukan prediksi
        print("🔮 Making predictions...")
        predictions = rf_model.predict(df_prediksi)

        # Buat DataFrame hasil prediksi
        df_hasil = df_prediksi.copy()
        df_hasil['Predicted_Serapan_K'] = predictions.round(2)

        # Simpan hasil prediksi
        nama_file_hasil = 'hasil_prediksi_serapan_k.csv'
        df_hasil.to_csv(nama_file_hasil, index=False)

        print(f"✅ Predictions saved as '{nama_file_hasil}'")
        print(
            f"📊 Prediction range: {df_hasil['Predicted_Serapan_K'].min():.1f} - {df_hasil['Predicted_Serapan_K'].max():.1f}")
        print(f"📈 Average prediction: {df_hasil['Predicted_Serapan_K'].mean():.1f}")

        print("\n📋 Sample predictions:")
        print(df_hasil.head(10).to_string(index=False))

        return df_hasil

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None


def main():
    """Main function untuk generate semua data dan model"""
    print("🌱 Starting comprehensive data generation...")
    print("=" * 50)

    # 1. Generate training data
    df_latihan = generate_training_data(200)
    nama_file_latihan = 'data_training_indeks_hara.csv'
    df_latihan.to_csv(nama_file_latihan, index=False)

    print(f"\n✅ Training data saved as '{nama_file_latihan}'")
    print(f"📊 Shape: {df_latihan.shape}")
    print("\nSample training data:")
    print(df_latihan.head())

    # 2. Generate prediction data
    df_prediksi = generate_prediction_data(50)
    nama_file_prediksi = 'data_prediksi_indeks_hara.csv'
    df_prediksi.to_csv(nama_file_prediksi, index=False)

    print(f"\n✅ Prediction data saved as '{nama_file_prediksi}'")
    print(f"📊 Shape: {df_prediksi.shape}")
    print("\nSample prediction data:")
    print(df_prediksi.head())

    # 3. Validasi konsistensi fitur
    print("\n🔍 Feature Consistency Check:")
    validate_features(df_latihan, df_prediksi)

    # 4. Train and save model
    rf_model = train_and_save_model(df_latihan)

    # 5. Save model
    joblib.dump(rf_model, 'model_rf_hara.joblib')
    print(f"\n💾 Model saved as 'model_rf_hara.joblib'")

    # 6. Generate summary statistics
    print("\n" + "=" * 50)
    print("📈 DATA SUMMARY STATISTICS")
    print("=" * 50)

    print("\n📊 TRAINING DATA STATISTICS:")
    print(df_latihan.describe().round(4))

    print(f"\n📈 Serapan_K Range: {df_latihan['Serapan_K'].min():.1f} - {df_latihan['Serapan_K'].max():.1f}")
    print(f"📊 Average Serapan_K: {df_latihan['Serapan_K'].mean():.1f}")

    print("\n🔮 PREDICTION DATA STATISTICS:")
    print(df_prediksi.describe().round(4))

    # 7. Correlation analysis
    print("\n🔍 CORRELATION WITH Serapan_K (Training Data):")
    correlations = df_latihan.corr()['Serapan_K'].sort_values(ascending=False)
    for idx, (feature, corr) in enumerate(correlations.items(), 1):
        if feature != 'Serapan_K':
            stars = "***" if abs(corr) > 0.7 else "**" if abs(corr) > 0.5 else "*" if abs(corr) > 0.3 else ""
            print(f"  {idx:2d}. {feature:8}: {corr:7.4f} {stars}")

    # 8. Jalankan prediksi untuk testing
    print("\n" + "=" * 50)
    print("🧪 TESTING PREDICTION")
    print("=" * 50)
    df_hasil_prediksi = predict_with_trained_model()

    print("\n🎉 All files generated successfully!")
    print("\n📁 Generated Files:")
    print(f"  1. {nama_file_latihan} - Data training dengan target Serapan_K")
    print(f"  2. {nama_file_prediksi} - Data prediksi (tanpa target)")
    print(f"  3. model_rf_hara.joblib - Trained Random Forest model")
    if df_hasil_prediksi is not None:
        print(f"  4. hasil_prediksi_serapan_k.csv - Hasil prediksi serapan K")


if __name__ == "__main__":
    main()