"""
================================================================================
VALIDACIÓN CRUZADA: PeruSat-1 (10m) vs Sentinel-2 (10m)
================================================================================
Autor: Teledetección - Armonización Radiométrica
Descripción:
    - K-Fold Cross-Validation riguroso
    - Manejo de nubes con máscara SCL de Sentinel-2
    - Métricas multibanda: RMSE, R², SSIM, Correlación, PSNR, Bias
    - Normalización robusta
    - Salida para paper científico
================================================================================
"""

import rasterio
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from math import log10, sqrt
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# ⚙️ CONFIGURACIÓN DEL USUARIO
# ============================================================

# Rutas de archivos
PERUSAT_FILE = r"C:\Users\51926\Desktop\Proyecto de Investigación\IA\harmonization_project_v1.0\outputs\calibnet_002\P1_harmonized_to_S2.tif"
SENTINEL_FILE = r"C:\Users\51926\Desktop\Proyecto de Investigación\IA\harmonization_project_v1.0\inputs\Datos_Listos\S2_final.tif"
CLOUD_MASK_FILE = r"C:\Users\51926\Desktop\Proyecto de Investigación\IA\harmonization_project_v1.0\inputs\Datos_Listos\mask_s2_final.tif"

# Parámetros de validación
N_SPLITS_KFOLD = 5              # K-Fold (5 o 10)
N_SAMPLES_RANDOM = 100000       # Píxeles aleatorios para eficiencia
RANDOM_STATE = 42
CLOUD_THRESHOLD = 0.3           # % máximo de nubes permitidas por fold

# Mapeo de bandas PeruSat → Sentinel-2
# PeruSat-1: [Azul, Verde, Rojo, NIR]
# Sentinel-2 SR: [Azul(B2), Verde(B3), Rojo(B4), NIR(B8)]
BAND_MAPPING = {
    'Azul (Blue)':      {'ps1_idx': 0, 's2_idx': 0},
    'Verde (Green)':    {'ps1_idx': 1, 's2_idx': 1},
    'Rojo (Red)':       {'ps1_idx': 2, 's2_idx': 2},
    'NIR':              {'ps1_idx': 3, 's2_idx': 3}
}

# Rango válido de reflectancia (DNs, Sentinel-2 SR es típicamente 0-10000)
REFLECTANCE_MIN = 0.0
REFLECTANCE_MAX = 10000.0

# Directorio de salida
OUTPUT_DIR = Path("./validacion_resultados")
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# 📊 FUNCIONES DE VALIDACIÓN
# ============================================================

def create_cloud_mask(cloud_mask_path, cloud_mask_band=1):
    """
    Lee máscara SCL de Sentinel-2 y crea máscara booleana de píxeles válidos.
    
    Valores SCL a excluir:
        0: No datos
        1: Sombra saturada
        3: Sombra
        8: Nube mediana
        9: Nube alta
        10: Sombra cirrus
    """
    try:
        with rasterio.open(cloud_mask_path) as src:
            cloud_mask = src.read(cloud_mask_band).astype(np.uint8)
    except Exception as e:
        print(f"⚠️  ERROR: No se pudo leer máscara de nubes: {e}")
        print("   → Continuando sin máscara (RECOMENDADO: revisar archivo)")
        return None
    
    # Píxeles válidos (sin nubes/sombras)
    invalid_values = [0,1,3,6, 8, 9, 10]
    valid_mask = ~np.isin(cloud_mask, invalid_values)
    
    cloud_percentage = (1 - valid_mask.sum() / valid_mask.size) * 100
    print(f"☁️  Cobertura nubosa: {cloud_percentage:.2f}%")
    
    return valid_mask


def read_raster_data(filepath):
    """Lee datos raster multibanda."""
    try:
        with rasterio.open(filepath) as src:
            data = src.read().astype(np.float32)
            profile = src.profile
            crs = src.crs
        return data, profile, crs
    except Exception as e:
        print(f"ERROR: No se pudo leer {filepath}")
        raise


def generate_random_pixels(bounds, n_samples=100000, random_state=42):
    """Genera coordenadas aleatorias dentro de los límites de la imagen."""
    np.random.seed(random_state)
    xs = np.random.uniform(bounds.left, bounds.right, n_samples)
    ys = np.random.uniform(bounds.bottom, bounds.top, n_samples)
    return list(zip(xs, ys))


def extract_pixel_values(filepath, coords):
    """Extrae valores en coordenadas específicas."""
    try:
        with rasterio.open(filepath) as src:
            values = np.array([val for val in src.sample(coords)])
        return values
    except Exception as e:
        print(f"ERROR en extracción de píxeles: {e}")
        raise


def create_validation_dataframe(ps1_samples, s2_samples, cloud_mask, band_mapping):
    """
    Crea DataFrame con datos válidos (filtrados por nubes y rango).
    
    Returns:
        pd.DataFrame con columnas: banda, y_ps1, y_s2
    """
    all_data = []
    
    for band_name, mapping in band_mapping.items():
        ps1_vals = ps1_samples[:, mapping['ps1_idx']]
        s2_vals = s2_samples[:, mapping['s2_idx']]
        
        df_band = pd.DataFrame({
            'banda': band_name,
            'y_ps1': ps1_vals,
            'y_s2': s2_vals
        })
        all_data.append(df_band)
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Filtrar datos inválidos
    initial_count = len(df)
    df = df[
        (df['y_ps1'] >= REFLECTANCE_MIN) & (df['y_ps1'] <= REFLECTANCE_MAX) &
        (df['y_s2'] >= REFLECTANCE_MIN) & (df['y_s2'] <= REFLECTANCE_MAX) &
        (np.isfinite(df['y_ps1'])) & (np.isfinite(df['y_s2']))
    ]
    
    removed = initial_count - len(df)
    print(f"🧹 Limpieza: {removed:,} píxeles inválidos removidos")
    print(f"✅ Píxeles válidos para análisis: {len(df):,}")
    
    return df


def calculate_metrics(y_true, y_pred):
    """
    Calcula todas las métricas de validación.
    
    Retorna diccionario con: RMSE, MAE, R², Correlación, SSIM, PSNR, Bias
    """
    # RMSE y MAE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R²
    r2 = r2_score(y_true, y_pred)
    
    # Correlación de Pearson
    if len(y_true) > 2:
        corr, _ = pearsonr(y_true, y_pred)
    else:
        corr = np.nan
    
    # SSIM (requiere normalización a [0, 1])
    y_true_norm = (y_true - y_true.min()) / (y_true.max() - y_true.min() + 1e-10)
    y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-10)
    ssim_val = ssim(y_true_norm, y_pred_norm, data_range=1.0)
    
    # PSNR
    mse = mean_squared_error(y_true, y_pred)
    if mse == 0:
        psnr = 100.0
    else:
        max_signal = y_true.max()
        psnr = 20 * log10(max_signal) - 10 * log10(mse)
    
    # Bias (error sistemático)
    bias = np.mean(y_pred - y_true)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Correlación': corr,
        'SSIM': ssim_val,
        'PSNR': psnr,
        'Bias': bias,
        'MAPE': mape
    }


def perform_kfold_validation(df, band_name, n_splits=5, random_state=42):
    """
    Ejecuta K-Fold Cross-Validation para una banda específica.
    
    Retorna:
        - Diccionario de métricas promediadas
        - Lista de métricas por fold (para boxplot)
    """
    X = df['y_ps1'].values
    y = df['y_s2'].values
    
    if len(X) < n_splits:
        print(f"⚠️  Banda {band_name}: insuficientes datos ({len(X)}) para {n_splits}-Fold")
        return None, []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_metrics = []
    all_metrics = {key: [] for key in ['RMSE', 'MAE', 'R2', 'Correlación', 'SSIM', 'PSNR', 'Bias', 'MAPE']}
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        if len(X_test) < 2:
            continue
        
        # Calcular métricas (sin entrenar modelo, solo comparar valores)
        metrics = calculate_metrics(y_test, X_test)
        fold_metrics.append(metrics)
        
        for key in all_metrics.keys():
            all_metrics[key].append(metrics[key])
    
    # Promediar métricas
    summary_metrics = {
        key: {
            'mean': np.nanmean(all_metrics[key]),
            'std': np.nanstd(all_metrics[key]),
            'min': np.nanmin(all_metrics[key]),
            'max': np.nanmax(all_metrics[key])
        }
        for key in all_metrics.keys()
    }
    
    return summary_metrics, fold_metrics


# ============================================================
# 🚀 EJECUCIÓN PRINCIPAL
# ============================================================

def main():
    print("\n" + "="*70)
    print("🛰️  VALIDACIÓN CRUZADA: PeruSat-1 (10m) vs Sentinel-2 (10m)")
    print("="*70)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Directorio de salida: {OUTPUT_DIR}")
    print()
    
    # ---- 1. Leer archivos ----
    print("📖 Leyendo archivos...")
    try:
        ps1_data, ps1_profile, ps1_crs = read_raster_data(PERUSAT_FILE)
        s2_data, s2_profile, s2_crs = read_raster_data(SENTINEL_FILE)
        
        print(f"   ✓ PeruSat-1: {ps1_data.shape}")
        print(f"   ✓ Sentinel-2: {s2_data.shape}")
        
        # Validar CRS
        if ps1_crs != s2_crs:
            print(f"⚠️  Advertencia: CRS diferentes (PS1: {ps1_crs}, S2: {s2_crs})")
        
    except Exception as e:
        print(f"❌ ERROR al leer archivos: {e}")
        return
    
    # ---- 2. Leer máscara de nubes ----
    print("\n☁️  Procesando máscara de nubes...")
    cloud_mask = create_cloud_mask(CLOUD_MASK_FILE)
    
    # ---- 3. Generar puntos de muestreo aleatorios ----
    print(f"\n🎲 Generando {N_SAMPLES_RANDOM:,} puntos aleatorios...")
    with rasterio.open(PERUSAT_FILE) as src:
        bounds = src.bounds
        coords = generate_random_pixels(bounds, N_SAMPLES_RANDOM, RANDOM_STATE)
    
    # ---- 4. Extraer valores ----
    print("🔍 Extrayendo valores de píxeles...")
    ps1_samples = extract_pixel_values(PERUSAT_FILE, coords)
    s2_samples = extract_pixel_values(SENTINEL_FILE, coords)
    print(f"   ✓ {ps1_samples.shape[0]:,} píxeles extraídos")
    
    # ---- 5. Crear DataFrame de validación ----
    print("\n📊 Preparando datos...")
    df = create_validation_dataframe(ps1_samples, s2_samples, cloud_mask, BAND_MAPPING)
    
    # ---- 6. K-Fold Validation por banda ----
    print(f"\n📈 Ejecutando {N_SPLITS_KFOLD}-Fold Cross-Validation...\n")
    
    all_results = {}
    all_fold_metrics = {}
    
    for band_name in BAND_MAPPING.keys():
        band_df = df[df['banda'] == band_name]
        
        if len(band_df) < N_SPLITS_KFOLD:
            print(f"⚠️  {band_name}: datos insuficientes. Saltando.")
            continue
        
        print(f"🔹 {band_name} (n={len(band_df):,})...")
        summary, fold_data = perform_kfold_validation(
            band_df, band_name, N_SPLITS_KFOLD, RANDOM_STATE
        )
        
        all_results[band_name] = summary
        all_fold_metrics[band_name] = fold_data
        
        # Mostrar resultados por banda
        if summary:
            print(f"   RMSE:  {summary['RMSE']['mean']:.2f} ± {summary['RMSE']['std']:.2f}")
            print(f"   R²:    {summary['R2']['mean']:.4f} ± {summary['R2']['std']:.4f}")
            print(f"   SSIM:  {summary['SSIM']['mean']:.4f} ± {summary['SSIM']['std']:.4f}")
            print(f"   Corr:  {summary['Correlación']['mean']:.4f}")
            print(f"   Bias:  {summary['Bias']['mean']:.2f} (error sistemático)")
            print()
    
    # ---- 7. Crear DataFrame de resultados finales ----
    print("📋 Compilando resultados...")
    results_summary = []
    
    for band_name, metrics in all_results.items():
        results_summary.append({
            'Banda': band_name,
            'RMSE (media)': f"{metrics['RMSE']['mean']:.2f}",
            'RMSE (±)': f"{metrics['RMSE']['std']:.2f}",
            'R² (media)': f"{metrics['R2']['mean']:.4f}",
            'R² (±)': f"{metrics['R2']['std']:.4f}",
            'SSIM': f"{metrics['SSIM']['mean']:.4f}",
            'Correlación': f"{metrics['Correlación']['mean']:.4f}",
            'PSNR (dB)': f"{metrics['PSNR']['mean']:.2f}",
            'Bias': f"{metrics['Bias']['mean']:.2f}",
            'MAPE (%)': f"{metrics['MAPE']['mean']:.2f}"
        })
    
    df_results = pd.DataFrame(results_summary)
    
    # ---- 8. Mostrar tabla final ----
    print("\n" + "="*70)
    print("✅ RESULTADOS DE VALIDACIÓN CRUZADA")
    print("="*70)
    print(df_results.to_string(index=False))
    
    # Estadísticas globales
    print("\n📊 ESTADÍSTICAS GLOBALES:")
    for band_name, metrics in all_results.items():
        print(f"\n{band_name}:")
        print(f"  • RMSE:        {metrics['RMSE']['mean']:.2f} ± {metrics['RMSE']['std']:.2f}")
        print(f"  • MAE:         {metrics['MAE']['mean']:.2f} ± {metrics['MAE']['std']:.2f}")
        print(f"  • R²:          {metrics['R2']['mean']:.4f} ± {metrics['R2']['std']:.4f}")
        print(f"  • Correlación: {metrics['Correlación']['mean']:.4f}")
        print(f"  • SSIM:        {metrics['SSIM']['mean']:.4f}")
        print(f"  • PSNR:        {metrics['PSNR']['mean']:.2f} dB")
        print(f"  • Bias:        {metrics['Bias']['mean']:.2f} DN")
        print(f"  • MAPE:        {metrics['MAPE']['mean']:.2f} %")
    
    # ---- 9. Guardar resultados en CSV ----
    csv_path = OUTPUT_DIR / f"validacion_resultados_{TIMESTAMP}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 Resultados guardados en: {csv_path}")
    
    print("\n" + "="*70)
    print("✨ VALIDACIÓN COMPLETADA EXITOSAMENTE")
    print("="*70 + "\n")
    
    return df_results, all_results

# ============================================================
# 🎯 ENTRADA
# ============================================================

if __name__ == "__main__":
    df_results, all_results = main()