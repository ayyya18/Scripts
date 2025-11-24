import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

try:
    df = pd.read_csv('data_tanaman.csv')
    print("Data berhasil dimuat!")
    print(df.head())
except FileNotFoundError:
    print("Error: File tidak ditemukan. Pastikan file csv ada di folder project.")
    exit()

X = df[['NDVI', 'NDRE', 'GNDVI']]
y = df['Serapan_K']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

print("\nSedang melatih model...")
rf_model.fit(X_train, y_train)
print("Pelatihan selesai.")
y_pred = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Hasil Evaluasi Model ---")
print(f"R-Squared (R2): {r2:.4f} (Semakin mendekati 1 semakin baik)")
print(f"RMSE: {rmse:.4f} (Semakin kecil semakin baik)")
feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n--- Tingkat Kepentingan Indeks Vegetasi ---")
print(feature_imp)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Garis diagonal lurus
plt.xlabel('Serapan K Aktual (Uji Lab)')
plt.ylabel('Serapan K Prediksi (Model)')
plt.title('Aktual vs Prediksi')

plt.subplot(1, 2, 2)
sns.barplot(x=feature_imp, y=feature_imp.index, palette="viridis")
plt.xlabel('Skor Kepentingan')
plt.title('Indeks Mana yang Paling Berpengaruh?')

plt.tight_layout()
plt.show()

joblib.dump(rf_model, 'model_rf_hara.joblib')
print("\nModel berhasil disimpan sebagai 'model_rf_hara.joblib'")