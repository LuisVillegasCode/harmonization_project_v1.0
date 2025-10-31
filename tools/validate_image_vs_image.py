import rasterio
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# --- 1. CONFIGURACIÓN ---
# Tu imagen corregida con seis_s_paper.py
PREDICTED_TIF = r"C:\Users\51926\Desktop\Proyecto de Investigación\IA\proyecto_deforestacion\results\harmonization_run_002\PS1_harmonized_to_S2.tif"
# Tu imagen de "verdad" (Sentinel-2 SR, ya alineada)
TRUTH_TIF = r"C:\Users\51926\Downloads\s2_recortado.tif"

# Define el mapeo de bandas (Índice en el TIF)
# DEBES AJUSTAR ESTO. Las bandas de Sentinel-2 SR son:
# B1(Aerosol), B2(Azul), B3(Verde), B4(Rojo), B8(NIR)
# Asumiendo que tu PeruSat-1 es [B0, B1, B2, B3]
BAND_MAP = {
    # Asumiendo que tu PeruSat-1 (pred) es [B0, B1, B2, B3]
    # Asumiendo que tu Sentinel (truth) es [B2, B3, B4, B8]
    'Azul':  {'pred_idx': 0, 'truth_idx': 0}, # PeruSat B0 vs S2 (Banda 1 del TIF)
    'Verde': {'pred_idx': 1, 'truth_idx': 1}, # PeruSat B1 vs S2 (Banda 2 del TIF)
    'Rojo':  {'pred_idx': 2, 'truth_idx': 2}, # PeruSat B2 vs S2 (Banda 3 del TIF)
    'NIR':   {'pred_idx': 3, 'truth_idx': 3}  # PeruSat B3 vs S2 (Banda 4 del TIF)
}

N_SAMPLES = 50000  # Número de píxeles aleatorios a muestrear
N_SPLITS_K_FOLD = 10 # K-fold
NODATA_VALUE = 0     # Valor de Nodata a ignorar

print(f"Iniciando validación Imagen-vs-Imagen (K={N_SPLITS_K_FOLD})")
print(f"  Predicción: {PREDICTED_TIF}")
print(f"  Verdad:     {TRUTH_TIF}")

# --- 2. Generar Puntos de Muestreo Aleatorios ---
try:
    with rasterio.open(PREDICTED_TIF) as src:
        bounds = src.bounds
        crs = src.crs
        # Genera coordenadas aleatorias (x, y) DENTRO de los límites de la imagen
        xs = np.random.uniform(bounds.left, bounds.right, N_SAMPLES)
        ys = np.random.uniform(bounds.bottom, bounds.top, N_SAMPLES)
        coords = list(zip(xs, ys))
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo de predicción: {PREDICTED_TIF}")
    exit()

# --- 3. Extraer Valores de Ambas Imágenes ---
try:
    with rasterio.open(PREDICTED_TIF) as src_pred:
        # 'sample' retorna un generador.
        pred_samples = np.array([val for val in src_pred.sample(coords)])

    with rasterio.open(TRUTH_TIF) as src_truth:
        # Validar que estén alineadas
        if src_truth.crs != crs:
            raise ValueError("Error: Los CRS de las imágenes no coinciden.")
        
        # Muestrear los mismos puntos exactos
        truth_samples = np.array([val for val in src_truth.sample(coords)])
        
    print(f"Muestreo completado. Obtenidos {pred_samples.shape[0]} píxeles.")

except Exception as e:
    print(f"ERROR: Falló la lectura de los rasters: {e}")
    exit()

# --- 4. Preparar DataFrame y Limpiar Datos ---
all_data = []
for band_name, mapping in BAND_MAP.items():
    pred_vals = pred_samples[:, mapping['pred_idx']]
    truth_vals = truth_samples[:, mapping['truth_idx']]
    
    df_band = pd.DataFrame({
        'banda': band_name,
        'y_pred': pred_vals,
        'y_true': truth_vals
    })
    all_data.append(df_band)

df = pd.concat(all_data)

# Limpiar datos no válidos (Nodata, nubes, ceros)
initial_count = len(df)
# Ignorar advertencias de división por cero o NaN
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df = df[
        (df['y_pred'] > NODATA_VALUE) & (df['y_true'] > NODATA_VALUE) &
        (df['y_pred'] < 10000) & (df['y_true'] < 10000) & # Asumiendo reflectancia 0-10000
        (np.isfinite(df['y_pred'])) & (np.isfinite(df['y_true']))
    ]
print(f"Limpieza de Nodata: {initial_count - len(df)} píxeles eliminados. Quedan {len(df)} válidos.")

# --- 5. Ejecutar Validación Cruzada K-Fold ---
kf = KFold(n_splits=N_SPLITS_K_FOLD, shuffle=True, random_state=42)
results = {}

for band in BAND_MAP.keys():
    band_df = df[df['banda'] == band]
    if band_df.empty:
        print(f"Advertencia: No hay datos válidos para la banda {band}. Saltando.")
        continue
        
    X = band_df['y_pred']
    y = band_df['y_true']
    
    fold_rmse = []
    fold_r2 = []

    for train_index, test_index in kf.split(X):
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        
        if len(X_test) < 2:
            continue # No se puede calcular R2 con un solo punto

        rmse = np.sqrt(mean_squared_error(y_test, X_test))
        r2 = r2_score(y_test, X_test)
        
        fold_rmse.append(rmse)
        fold_r2.append(r2)

    results[band] = {
        "RMSE_mean": np.mean(fold_rmse),
        "RMSE_std": np.std(fold_rmse),
        "R2_mean": np.mean(fold_r2),
        "R2_std": np.std(fold_r2),
        "N_pixels": len(band_df)
    }

# --- 6. Mostrar Resultados (Listos para tu Paper) ---
print("\n--- Resultados de Validación Cruzada (Imagen-vs-Imagen) ---")
for band, metrics in results.items():
    print(f"Banda: {band} (N={metrics['N_pixels']})")
    print(f"  R² (media):   {metrics['R2_mean']:.4f} (std: {metrics['R2_std']:.4f})")
    print(f"  RMSE (media): {metrics['RMSE_mean']:.2f} (std: {metrics['RMSE_std']:.2f})")
    print("-" * 20)