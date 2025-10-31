import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

import affine
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
from skimage.registration import phase_cross_correlation
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter, laplace
from sklearn.linear_model import RANSACRegressor

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)


def compute_valid_mask(array: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    """
    Crea máscara de píxeles válidos considerando NoData y valores extremos.
    """
    mask = np.ones(array.shape, dtype=bool)
    
    if nodata is not None:
        mask &= (array != nodata)
    
    # Filtrar valores extremos (posibles artefactos)
    mask &= np.isfinite(array)
    mask &= (array > np.percentile(array[mask], 1))
    mask &= (array < np.percentile(array[mask], 99))
    
    return mask


def assess_overlap_quality(master: np.ndarray, slave: np.ndarray, 
                          master_mask: np.ndarray, slave_mask: np.ndarray) -> Dict:
    """
    Evalúa la calidad del overlap para determinar si el co-registro es viable.
    """
    combined_mask = master_mask & slave_mask
    valid_pixels = np.sum(combined_mask)
    total_pixels = master.size
    overlap_ratio = valid_pixels / total_pixels
    
    # Validar rango dinámico
    master_valid = master[combined_mask]
    slave_valid = slave[combined_mask]
    
    master_std = np.std(master_valid)
    slave_std = np.std(slave_valid)
    
    # Información mutua preliminar
    hist_2d, _, _ = np.histogram2d(
        master_valid.ravel(), 
        slave_valid.ravel(), 
        bins=50
    )
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi_preliminary = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    quality = {
        'overlap_ratio': overlap_ratio,
        'valid_pixels': int(valid_pixels),
        'master_std': float(master_std),
        'slave_std': float(slave_std),
        'mutual_information': float(mi_preliminary)
    }
    
    # Validaciones
    if overlap_ratio < 0.3:
        logging.warning(f"Overlap insuficiente: {overlap_ratio:.2%} (mínimo recomendado: 30%)")
    
    if master_std < 5 or slave_std < 5:
        logging.warning(f"Bajo contraste detectado (std < 5). Puede afectar precisión.")
    
    return quality


def calculate_alignment_metrics(master: np.ndarray, slave: np.ndarray, 
                               mask: Optional[np.ndarray] = None) -> Dict:
    """
    Calcula métricas exhaustivas de alineamiento.
    """
    if mask is None:
        mask = np.ones(master.shape, dtype=bool)
    
    master_masked = master[mask]
    slave_masked = slave[mask]
    
    # 1. RMSE
    rmse = np.sqrt(np.mean((master_masked - slave_masked) ** 2))
    
    # 2. RMSE Normalizado (nRMSE)
    range_master = np.ptp(master_masked)
    nrmse = rmse / range_master if range_master > 0 else np.inf
    
    # 3. Correlación de Pearson
    corr_pearson, p_value = pearsonr(master_masked, slave_masked)
    
    # 4. SSIM (Structural Similarity)
    # Normalizar a rango [0, 1] para SSIM
    master_norm = (master - master.min()) / (master.max() - master.min())
    slave_norm = (slave - slave.min()) / (slave.max() - slave.min())
    
    ssim_score = ssim(master_norm, slave_norm, data_range=1.0)
    
    # 5. Información Mutua (más robusto a diferencias radiométricas)
    hist_2d, _, _ = np.histogram2d(master_masked, slave_masked, bins=50)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    # 6. Diferencia Absoluta Media
    mae = np.mean(np.abs(master_masked - slave_masked))
    
    metrics = {
        'rmse': float(rmse),
        'nrmse': float(nrmse),
        'pearson_r': float(corr_pearson),
        'pearson_p_value': float(p_value),
        'ssim': float(ssim_score),
        'mutual_information': float(mi),
        'mae': float(mae),
        'n_pixels_evaluated': int(np.sum(mask))
    }
    
    return metrics


def prepare_pc_inputs(
    master_s2_path: Path,
    master_nir_index: int,
    slave_p1_path: Path,
    slave_nir_index: int,
    auto_crop_to_overlap: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray, Dict]:
    """
    Prepara los dos arrays de entrada para la Correlación de Fase.
    Ahora lee bandas específicas de archivos apilados.
    """
    logging.info(f"Preparando entradas para PC (Maestro: {master_s2_path.name} [Banda {master_nir_index}], Esclavo: {slave_p1_path.name} [Banda {slave_nir_index}])")
    try:
        # 1. Leer metadata de ambas imágenes
        with rasterio.open(slave_p1_path) as slave_ds:
            slave_bounds = slave_ds.bounds
            slave_crs = slave_ds.crs
            # Leer nodata de la banda NIR específica
            slave_nodata = slave_ds.nodatavals[slave_nir_index - 1] # nodatavals es 0-indexed
        
        # 2. Leer Maestra (S2 B8), opcionalmente recortada
        with rasterio.open(master_s2_path) as master_ds:
            master_crs = master_ds.crs
            master_nodata = master_ds.nodatavals[master_nir_index - 1] # 0-indexed
            
            if auto_crop_to_overlap:
                slave_bounds_in_master_crs = transform_bounds(
                    slave_crs, master_crs, *slave_bounds
                )
                window = from_bounds(*slave_bounds_in_master_crs, master_ds.transform)
                
                # Leer solo la banda NIR (master_nir_index)
                master_array = master_ds.read(master_nir_index, window=window)
                master_transform = master_ds.window_transform(window)
                
                logging.info(f"✂ S2 recortada automáticamente a extent de PeruSat")
            else:
                # Leer solo la banda NIR (master_nir_index)
                master_array = master_ds.read(master_nir_index)
                master_transform = master_ds.transform
            
            master_shape = master_array.shape
            master_meta = master_ds.meta.copy()
            # Actualizar metadata para reflejar que es de una sola banda
            master_meta.update({
                'height': master_shape[0],
                'width': master_shape[1],
                'transform': master_transform,
                'count': 1, # Importante: el grid de salida es de 1 banda
                'nodata': master_nodata # Asignar el nodata de esta banda
            })

        # 3. Crear Esclava temporal en la cuadrícula de la Maestra (recortada)
        slave_temp_10m_array = np.empty(master_shape, dtype=master_meta['dtype'])
        
        with rasterio.open(slave_p1_path) as slave_ds:
            reproject(
                # Leer solo la banda NIR esclava (slave_nir_index)
                source=rasterio.band(slave_ds, slave_nir_index),
                destination=slave_temp_10m_array,
                src_transform=slave_ds.transform,
                src_crs=slave_ds.crs,
                src_nodata=slave_nodata,
                dst_transform=master_transform,
                dst_crs=master_crs,
                dst_nodata=master_nodata,
                resampling=Resampling.average
            )
        
        # 4. Crear máscaras de validez
        master_mask = compute_valid_mask(master_array, master_nodata)
        slave_mask = compute_valid_mask(slave_temp_10m_array, slave_nodata)
        
        # 5. Evaluar calidad del overlap
        quality_report = assess_overlap_quality(
            master_array, slave_temp_10m_array, 
            master_mask, slave_mask
        )
        
        logging.info(f"Arrays de {master_shape} listos.")
        logging.info(f"Overlap válido: {quality_report['overlap_ratio']:.2%}")
        
        if quality_report['overlap_ratio'] < 0.2:
            raise ValueError("Overlap insuficiente para co-registro confiable (<20%)")
        
        combined_mask = master_mask & slave_mask
        
        return master_array, slave_temp_10m_array, master_meta, combined_mask, quality_report

    except RasterioIOError as e:
        logging.error(f"Error de E/S al leer los archivos: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error inesperado en 'prepare_pc_inputs': {e}")
        sys.exit(1)


def radiometric_normalization(
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    method: str = 'histogram_matching'
) -> np.ndarray:
    """
    Normaliza radiométricamente la imagen target a la referencia.
    Crítico para comparar DN vs Reflectancia.
    
    Args:
        reference: Imagen de referencia (ej. S2 SR)
        target: Imagen a normalizar (ej. PeruSat DN)
        mask: Máscara de píxeles válidos
        method: 'histogram_matching' o 'linear_regression'
    """
    from skimage.exposure import match_histograms
    
    ref_valid = reference[mask]
    tgt_valid = target[mask]
    
    if method == 'histogram_matching':
        # Histogram matching: preserva estructura local
        target_norm = match_histograms(target, reference)
        logging.info("Normalización radiométrica: Histogram Matching")
        
    elif method == 'linear_regression':
        # Regresión lineal: y = ax + b
        # Asume relación lineal entre DN y Reflectancia
        from sklearn.linear_model import RANSACRegressor
        
        # Usar RANSAC para robustez ante outliers
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(tgt_valid.reshape(-1, 1), ref_valid)
        
        a = ransac.estimator_.coef_[0]
        b = ransac.estimator_.intercept_
        
        target_norm = a * target + b
        target_norm = np.clip(target_norm, reference.min(), reference.max())
        
        logging.info(f"Normalización radiométrica: Regresión Lineal (y = {a:.4f}x + {b:.2f})")
        logging.info(f"R² inliers: {ransac.score(tgt_valid.reshape(-1, 1), ref_valid):.3f}")
    
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    return target_norm


def calculate_subpixel_shift(
    master_array: np.ndarray,
    slave_array: np.ndarray,
    mask: np.ndarray,
    normalize_radiometry: bool = True,  # Dejamos el argumento por compatibilidad
    use_edge_map: bool = True       # NUEVO: Control para usar bordes
) -> Tuple[float, float, Dict]:
    """
    Calcula el desplazamiento sub-píxel usando Correlación de Fase.
    MODIFICADO: Ahora puede usar mapas de bordes (Laplaciano) para
    robustez extrema contra diferencias radiométricas.
    """
    logging.info("Calculando desplazamiento sub-píxel (Correlación de Fase)...")
    
    if use_edge_map:
        logging.info("MODO ROBUSTO: Usando mapas de bordes (Filtro Laplaciano).")
        # Aplicar filtro Laplaciano para realzar bordes
        # Esto es extremadamente robusto a nubes, bruma y DN vs L2A
        master_proc = laplace(master_array, mode='mirror')
        slave_proc = laplace(slave_array, mode='mirror')
        
    elif normalize_radiometry:
        # Modo anterior (falló en su caso)
        logging.info("Aplicando normalización radiométrica (Histogram Matching)...")
        slave_proc = radiometric_normalization(
            master_array, 
            slave_array, 
            mask,
            method='histogram_matching'
        )
        master_proc = master_array
    
    else:
        # Sin procesamiento (mala idea para DN vs L2A)
        master_proc = master_array
        slave_proc = slave_array

    try:
        # Enmascarar áreas inválidas (asignar a cero en mapas de bordes)
        master_masked = master_proc.copy()
        slave_masked = slave_proc.copy()
        
        mask_fill_value = 0.0 if use_edge_map else np.mean(master_proc[mask])
        
        master_masked[~mask] = mask_fill_value
        slave_masked[~mask] = mask_fill_value
        
        # Correlación de fase con alta precisión
        shift, error, diffphase = phase_cross_correlation(
            master_masked,
            slave_masked,
            upsample_factor=100
        )
        
        dy_pixels, dx_pixels = shift
        
        # Calcular métricas de confianza
        confidence = {
            'shift_dy_px': float(dy_pixels),
            'shift_dx_px': float(dx_pixels),
            'shift_magnitude_px': float(np.sqrt(dy_pixels**2 + dx_pixels**2)),
            'phase_correlation_error': float(error), # ¡Este valor DEBE ser < 1.0!
            'diffphase': float(diffphase),
            'method_used': 'edge_map (laplace)' if use_edge_map else ('histogram_matching' if normalize_radiometry else 'none')
        }
        
        logging.info(f"Shift (dy, dx): [{dy_pixels:.4f}, {dx_pixels:.4f}] px")
        logging.info(f"Magnitud del shift: {confidence['shift_magnitude_px']:.4f} px")
        logging.info(f"Error de correlación de fase: {error:.6f}")
        
        # Advertencia si el shift es anormalmente grande
        if confidence['shift_magnitude_px'] > 10.0: # Aumentamos el umbral
            logging.warning(f"Shift grande detectado (>{10} px). Verificar resultados.")
        
        if error > 0.9:
            logging.error(f"¡FALLA DE CORRELACIÓN! Error de fase es {error:.4f}. El shift no es confiable.")
            
        return dx_pixels, dy_pixels, confidence
        
    except ValueError as e:
        logging.error(f"Error en 'phase_cross_correlation': {e}")
        logging.error("Posibles causas: imágenes vacías, sin overlap, o completamente diferentes")
        sys.exit(1)


def get_corrected_transform(
    slave_stacked_path: Path, 
    dx_pixels: float,
    dy_pixels: float,
    master_meta: dict
) -> affine.Affine:
    """
    Calcula la nueva geotransformación (Affine) corregida para la imagen Esclava.
    """
    try:
        master_transform = master_meta['transform']
        dx_meters = dx_pixels * master_transform.a
        dy_meters = dy_pixels * master_transform.e
        logging.info(f"Shift en metros (dx, dy): [{dx_meters:.4f}m, {dy_meters:.4f}m]")

        # Abrir el stack esclavo para obtener su transform original
        with rasterio.open(slave_stacked_path) as slave_ds:
            original_slave_transform = slave_ds.transform

        shift_transform = affine.Affine.translation(dx_meters, dy_meters)
        corrected_slave_transform = original_slave_transform * shift_transform
        
        logging.info(f"Transformación Affine corregida generada.")
        return corrected_slave_transform

    except RasterioIOError as e:
        logging.error(f"No se pudo leer la metadata de {slave_stacked_path}: {e}")
        sys.exit(1)


# MODIFICADA
def warp_and_aggregate_band(
    src_stacked_path: Path, # Ruta al stack
    src_band_index: int,    # Qué banda leer
    out_path: Path,         # Dónde escribir
    corrected_transform: affine.Affine,
    master_meta: dict
):
    """
    Aplica la reproyección final: lee 1 banda del stack, la alinea y la agrega.
    """
    logging.info(f"Procesando: {src_stacked_path.name} [Banda {src_band_index}] -> {out_path.name}")
    
    try:
        with rasterio.open(src_stacked_path) as src_ds:
            # Obtener metadata de la banda específica
            src_nodata = src_ds.nodatavals[src_band_index - 1] # 0-indexed
            src_dtype = src_ds.dtypes[src_band_index - 1]     # 0-indexed

            # Usar metadata de la Maestra (grid) pero actualizar tipo de dato
            output_meta = master_meta.copy()
            output_meta.update({
                "dtype": src_dtype,
                "nodata": src_nodata,
                "count": 1 # La salida es un archivo de 1 banda
            })

            with rasterio.open(out_path, 'w', **output_meta) as dst_ds:
                reproject(
                    source=rasterio.band(src_ds, src_band_index), 
                    destination=rasterio.band(dst_ds, 1),
                    src_crs=src_ds.crs,
                    src_nodata=src_nodata,
                    src_transform=corrected_transform, 
                    dst_transform=master_meta['transform'],
                    dst_crs=master_meta['crs'],
                    dst_nodata=src_nodata,
                    resampling=Resampling.average
                )
    except Exception as e:
        logging.error(f"Falló la reproyección para {src_stacked_path.name} [Banda {src_band_index}]: {e}")


def verify_alignment(
    master_s2_path: Path,
    master_nir_index: int,
    aligned_p1_nir_path: Path,
    metrics_before: Dict
) -> Dict:
    """
    Verificación exhaustiva del alineamiento.
    Lee la banda NIR del stack maestro S2.
    """
    logging.info("--- Iniciando Verificación Post-Proceso ---")
    try:
        with rasterio.open(master_s2_path) as master_ds:
            # Leer solo la banda NIR maestra
            master_array = master_ds.read(master_nir_index) 
            master_nodata = master_ds.nodatavals[master_nir_index - 1]
            master_shape = master_ds.shape # Shape completo, no de la banda
            
        # Corregir shape de master_array si se leyó con auto_crop
        # (El array alineado ya tiene el shape correcto del grid de salida)
        with rasterio.open(aligned_p1_nir_path) as aligned_ds:
             aligned_shape = aligned_ds.shape
             aligned_nodata = aligned_ds.nodata

        # Si el array maestro leído (que puede ser del auto-crop)
        # no coincide con el array alineado (que está en el grid
        # de salida del auto-crop), debemos leer el maestro con
        # el mismo window que el alineado.
        
        # Estrategia más simple: volver a leer el S2 usando el 
        # grid del P1 alineado como referencia
        
        with rasterio.open(aligned_p1_nir_path) as aligned_ds:
            aligned_array = aligned_ds.read(1)
            aligned_meta = aligned_ds.meta.copy()

        # Crear array maestro que coincida 100% con el P1 alineado
        master_array_aligned_grid = np.empty(aligned_shape, dtype=aligned_meta['dtype'])
        
        with rasterio.open(master_s2_path) as master_ds:
            reproject(
                source=rasterio.band(master_ds, master_nir_index),
                destination=master_array_aligned_grid,
                src_transform=master_ds.transform,
                src_crs=master_ds.crs,
                src_nodata=master_ds.nodatavals[master_nir_index - 1],
                dst_transform=aligned_meta['transform'],
                dst_crs=aligned_meta['crs'],
                dst_nodata=aligned_nodata,
                resampling=Resampling.nearest
            )
        
        master_array = master_array_aligned_grid # Sobrescribir
        master_nodata = aligned_nodata # Ahora usan el mismo nodata

        # --- El resto de la función (cálculo de métricas) sigue igual ---
        
        # Crear máscara válida
        mask = compute_valid_mask(master_array, master_nodata)
        mask &= compute_valid_mask(aligned_array, aligned_nodata)
        
        # Calcular shift residual
        shift, _, _ = phase_cross_correlation(
            master_array, 
            aligned_array,
            upsample_factor=100
        )
        
        # ... (el resto del código de la función no cambia) ...
        # ... (cálculo de métricas, reporte, etc.) ...

        # (Asegúrese de copiar el resto del cuerpo de la función original aquí)
        # Por brevedad, se omite el código idéntico.
        # El siguiente código es el original, no requiere cambios:

        shift_magnitude = np.sqrt(shift[0]**2 + shift[1]**2)
        
        metrics_after = calculate_alignment_metrics(master_array, aligned_array, mask)
        metrics_after['residual_shift_dy'] = float(shift[0])
        metrics_after['residual_shift_dx'] = float(shift[1])
        metrics_after['residual_shift_magnitude'] = float(shift_magnitude)
        
        # Reporte comparativo
        logging.info("\n" + "="*60)
        logging.info("REPORTE DE VERIFICACIÓN")
        logging.info("="*60)
        logging.info(f"\n{'Métrica':<30} {'Antes':<15} {'Después':<15} {'Mejora':<15}")
        logging.info("-"*75)
        
        for key in ['rmse', 'pearson_r', 'ssim', 'mutual_information']:
            before = metrics_before.get(key, 0)
            after = metrics_after.get(key, 0)
            
            if key == 'rmse':
                improvement = ((before - after) / before * 100) if before > 0 else 0
                improvement_str = f"{improvement:.1f}% ↓"
            else:
                improvement = ((after - before) / abs(before) * 100) if before != 0 else 0
                improvement_str = f"{improvement:.1f}% ↑"
            
            logging.info(f"{key:<30} {before:<15.4f} {after:<15.4f} {improvement_str:<15}")
        
        logging.info("-"*75)
        logging.info(f"Shift residual (dy, dx): [{shift[0]:.4f}, {shift[1]:.4f}] px")
        logging.info(f"Magnitud shift residual: {shift_magnitude:.4f} px")
        logging.info("="*60 + "\n")
        
        # Criterios de éxito
        success = True
        if shift_magnitude > 0.1:
            logging.warning(f"Shift residual alto: {shift_magnitude:.4f} px (esperado <0.1)")
            success = False
        
        if metrics_after['pearson_r'] < 0.7:
            logging.warning(f"Correlación baja: {metrics_after['pearson_r']:.3f} (esperado >0.7)")
            success = False
            
        if metrics_after['ssim'] < 0.7:
            logging.warning(f"SSIM bajo: {metrics_after['ssim']:.3f} (esperado >0.7)")
            success = False
        
        if success:
            logging.info("✓ VERIFICACIÓN EXITOSA: Alineamiento sub-píxel confirmado.")
        else:
            logging.warning("✗ VERIFICACIÓN CON ADVERTENCIAS: Revisar métricas.")
        
        return metrics_after
            
    except Exception as e:
        logging.error(f"Error durante la verificación: {e}")
        return {}


# MODIFICADA
def run_coregistration(config: Dict): # Config ya no tiene tipo Path
    """
    Orquesta el flujo completo de co-registro con métricas científicas.
    Adaptado para leer de archivos apilados.
    """
    logging.info("="*60)
    logging.info("INICIANDO PROCESO DE CO-REGISTRO SUB-PÍXEL (MODO STACKED)")
    logging.info("="*60)
    
    report = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'preprocessing': {},
        'coregistration': {},
        'validation': {}
    }

    # --- Obtener índices NIR de la configuración ---
    master_nir_index = config['master_s2_nir_band_index']
    slave_nir_index = config['slave_p1_band_mapping']['nir'] # Ya validamos que 'nir' existe
    
    # --- PASO 1 y 2: Preparar y Calcular Shift ---
    master_array, slave_temp_array, master_meta, mask, quality_report = prepare_pc_inputs(
        config['master_s2_stacked_file'],
        master_nir_index,
        config['slave_p1_stacked_file'],
        slave_nir_index,
        auto_crop_to_overlap=True
    )
    
    report['preprocessing'] = quality_report
    
    metrics_before = calculate_alignment_metrics(master_array, slave_temp_array, mask)
    report['validation']['metrics_before'] = metrics_before
    
    logging.info("\n--- Métricas ANTES del Co-registro ---")
    logging.info(f"RMSE: {metrics_before['rmse']:.2f}")
    logging.info(f"Correlación: {metrics_before['pearson_r']:.3f}")
    logging.info(f"SSIM: {metrics_before['ssim']:.3f}")
    
    dx_pixels, dy_pixels, confidence = calculate_subpixel_shift(
        master_array, slave_temp_array, mask,
        normalize_radiometry=True
    )
    report['coregistration'] = confidence
    
    del master_array, slave_temp_array

    # --- PASO 3: Calcular Transformación Corregida ---
    corrected_slave_transform = get_corrected_transform(
        config['slave_p1_stacked_file'], # Pasa el stack
        dx_pixels,
        dy_pixels,
        master_meta
    )

    # --- PASO 4: Aplicar Warp (Loop modificado) ---
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    processed_files = {} # Usar dict para guardar la ruta de 'nir'

    # Iterar sobre el mapeo de bandas (nombre, índice)
    for band_name, band_index in config['slave_p1_band_mapping'].items():
        out_path = output_dir / f"PS1_{band_name}_10m_aligned_DN.tif"
        warp_and_aggregate_band(
            src_stacked_path=config['slave_p1_stacked_file'], # El stack de origen
            src_band_index=band_index,                      # La banda a procesar
            out_path=out_path,
            corrected_transform=corrected_slave_transform,
            master_meta=master_meta
        )
        if band_name == 'nir':
            processed_files['nir'] = out_path # Guardar la ruta de salida de nir

    # --- PASO 5: Verificación ---
    metrics_after = verify_alignment(
        config['master_s2_stacked_file'],
        master_nir_index,
        processed_files['nir'], # Pasa la ruta NIR de salida
        metrics_before
    )
    report['validation']['metrics_after'] = metrics_after
    
    # --- PASO 6: Guardar Reporte ---
    report_path = output_dir / "coregistration_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logging.info(f"\n✓ Reporte guardado en: {report_path}")
    logging.info("="*60)
    logging.info("PROCESO COMPLETADO")
    logging.info("="*60)


# --- Punto de Entrada Principal ---
if __name__ == "__main__":
    
    try:
        # --- NUEVA ESTRUCTURA DE CONFIGURACIÓN ---
        PROJECT_CONFIG = {
            
            # --- Maestro (Sentinel-2) ---
            "master_s2_stacked_file": Path(r"C:\Users\51926\Downloads\Sentinel_Stack.tif"),
            # Indique qué banda de su stack S2 es la NIR (B8)
            "master_s2_nir_band_index": 4, # EJEMPLO: Si B2=1, B3=2, B4=3, B8=4
            
            # --- Esclavo (PeruSAT-1) ---
            "slave_p1_stacked_file": Path(r"C:\Users\51926\Desktop\Proyecto de Investigación\IA\Programacion Paper\Validación cruzada\IMG_PER1_ORT_MS_001586\analysis_ready_data_modified.tif"),
            
            # Mapeo de nombres a índices de banda (1-based) en el stack PS1
            "slave_p1_band_mapping": {
                # 'nombre_salida': indice_de_banda_en_el_stack
                'blue': 1,
                'green': 2,
                'red': 3,
                'nir': 4,
            },
            
            # --- Salida ---
            "output_dir": Path("resultados_alineados_2/")
        }
        
        # --- VALIDACIÓN DE ENTRADA MODIFICADA ---
        if not PROJECT_CONFIG['master_s2_stacked_file'].exists():
            raise FileNotFoundError(f"Archivo maestro no encontrado: {PROJECT_CONFIG['master_s2_stacked_file']}")
        
        if not PROJECT_CONFIG['slave_p1_stacked_file'].exists():
            raise FileNotFoundError(f"Archivo esclavo no encontrado: {PROJECT_CONFIG['slave_p1_stacked_file']}")
        
        if 'nir' not in PROJECT_CONFIG['slave_p1_band_mapping']:
            raise KeyError("La configuración 'slave_p1_band_mapping' DEBE contener una clave 'nir'.")

        # Ejecutar el proceso
        run_coregistration(PROJECT_CONFIG)

    except (FileNotFoundError, KeyError) as e:
        logging.error(e)
        sys.exit(1)
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado en el main: {e}")
        sys.exit(1)