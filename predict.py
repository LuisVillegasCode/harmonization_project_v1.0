# /harmonization_project/predict.py

"""
Script de inferencia para BCNet de armonización de imágenes satelitales.

Este script aplica el modelo entrenado a imágenes PeruSat-1 completas
para generar reflectancias armonizadas con Sentinel-2.

Metodología [cite: 191, 991]:
- Usa SOLO CalibNet (MLP pixel-wise) en inferencia
- Omite InputModule para preservar resolución espacial
- Procesamiento eficiente por bloques (tiling)

Uso:
    python predict.py \\
        --input /data/perusat1_scene.tif \\
        --output /outputs/perusat1_harmonized.tif \\
        --checkpoint /models/best_model.pth \\
        --block-size 1024
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
from tqdm import tqdm

# Importar arquitectura
try:
    from src.architecture import BCNet
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.architecture import BCNet


# ============================================================================
# EXCEPCIONES PERSONALIZADAS
# ============================================================================

class PredictionException(Exception):
    """Excepción base para errores de predicción."""
    pass


class InvalidInputException(PredictionException):
    """Excepción para inputs inválidos."""
    pass


# ============================================================================
# CARGA DE MODELO
# ============================================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """
    Carga el checkpoint de BCNet y extrae SOLO el módulo CalibNet (MLP)
    [cite_start]para la inferencia pixel-wise, como exige la metodología[cite: 191, 991].
    
    Args:
        checkpoint_path: Ruta al checkpoint (.pth).
        device: Device de PyTorch.
        
    Returns:
        Módulo CalibNet (el MLP) en modo eval.
        
    Raises:
        FileNotFoundError: Si checkpoint no existe.
        PredictionException: Si hay error al cargar.
    """
    log = logging.getLogger(__name__)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
    
    log.info(f"Cargando checkpoint desde: {checkpoint_path}")
    
    try:
        # 1. Instanciar el esqueleto completo de BCNet
        full_model = BCNet()
        
        # 2. Cargar el checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        full_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 3. EXTRAER SOLO EL MLP (CalibNet)
        # [cite_start]Esto es metodológicamente correcto para la inferencia [cite: 191, 991]
        mlp_model = full_model.downstream_calibnet
        
        # 4. Mover a device y modo eval
        mlp_model = mlp_model.to(device)
        mlp_model.eval()
        
        # Log de información
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('best_val_loss', 'unknown')
        
        log.info(f"OK Modelo CalibNet (MLP) extraído exitosamente")
        log.info(f"  Época del checkpoint: {epoch}")
        log.info(f"  Val loss: {val_loss}")
        
        return mlp_model
        
    except Exception as e:
        log.error(f"Error al cargar modelo: {e}", exc_info=True)
        raise PredictionException(f"Fallo al cargar modelo: {e}") from e

# ============================================================================
# PROCESAMIENTO DE BLOQUES
# ============================================================================

def create_nodata_mask(
    block: np.ndarray,
    nodata_values: List[float],
) -> np.ndarray:
    """
    Crea máscara booleana para píxeles nodata.
    
    Args:
        block: Array [C, H, W] con datos del bloque.
        nodata_values: Lista de valores que representan nodata.
        
    Returns:
        Máscara booleana [H, W] donde True = nodata.
    """
    mask = np.zeros(block.shape[1:], dtype=bool)
    
    for nodata_val in nodata_values:
        # Píxel es nodata si CUALQUIER banda contiene el valor
        mask |= np.any(block == nodata_val, axis=0)
    
    return mask


def process_block(
    block: np.ndarray,
    mlp_model: nn.Module,
    nodata_values: List[float],
    scale_factor: float,
    device: torch.device,
    gpu_batch_size: int = 10000,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Aplica el modelo MLP (CalibNet) a un bloque de imagen (pixel-wise).
    
    Args:
        block: Array [C, H, W] con reflectancias.
        mlp_model: Modelo CalibNet (MLP) en modo eval.
        (El resto de argumentos permanecen igual)
        
    Returns:
        Tupla (processed_block, stats)
    """
    C, H, W = block.shape
    
    # Estadísticas del bloque
    stats = {
        'total_pixels': H * W,
        'nodata_pixels': 0,
        'valid_pixels': 0,
        'out_of_range_pixels': 0,
    }
    
    # 1. Crear máscara de nodata
    nodata_mask = create_nodata_mask(block, nodata_values)
    stats['nodata_pixels'] = nodata_mask.sum()
    
    # 2. Escalar y reformatear: [C,H,W] → [H*W, C]
    block_permuted = np.moveaxis(block.astype(np.float32), 0, -1)
    block_scaled = block_permuted / scale_factor
    block_flat = block_scaled.reshape(H * W, C)
    
    # 3. Extraer píxeles válidos
    valid_mask = ~nodata_mask.flatten()
    valid_pixels = block_flat[valid_mask]
    stats['valid_pixels'] = len(valid_pixels)
    
    # 4. Crear output (inicializar con nodata)
    output_flat = np.full_like(block_flat, nodata_values[0])
    
    if len(valid_pixels) > 0:
        # 5. Procesar en mini-batches (para GPU)
        predictions = []
        
        for i in range(0, len(valid_pixels), gpu_batch_size):
            batch = valid_pixels[i:i + gpu_batch_size]
            batch_tensor = torch.from_numpy(batch).to(device)
            
            with torch.no_grad():
                # Usar training=False para inferencia pixel-wise
                # (omite InputModule)
                pred = mlp_model(batch_tensor)
            
            predictions.append(pred.cpu().numpy())
        
        predicted_pixels = np.concatenate(predictions)
        
        # 6. Insertar predicciones en output
        output_flat[valid_mask] = predicted_pixels
    
    # 7. Reformatear: [H*W, C] → [C, H, W]
    output_spatial = output_flat.reshape(H, W, C).transpose(2, 0, 1)
    
    # 8. Re-escalar
    output_unscaled = output_spatial * scale_factor
    
    # 9. Verificar y clampear valores fuera de rango
    out_of_range = (
        (output_unscaled < 0) | (output_unscaled > 65535)
    ) & (~nodata_mask)
    
    stats['out_of_range_pixels'] = out_of_range.sum()
    
    np.clip(output_unscaled, 0, 65535, out=output_unscaled)
    
    # 10. Reinsertar nodata
    output_unscaled[:, nodata_mask] = nodata_values[0]
    
    return output_unscaled.astype(np.uint16), stats


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def setup_logging():
    """Configura el logging para la predicción (solo consola)."""
    # (Esta función es requerida por main.py)
    log = logging.getLogger(__name__)
    if log.hasHandlers():
        return log # Ya configurado por main.py
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    log.info("Logging de predicción (standalone) iniciado.")
    return log


def run_prediction(args: argparse.Namespace) -> Dict:
    """
    Ejecuta predicción completa sobre imagen de entrada (llamado desde main.py).
    
    Args:
        args: Namespace de argparse proveniente de main.py.
              Debe contener (mínimo):
              - checkpoint, input, output, device
    
    Returns:
        Diccionario con estadísticas de procesamiento.
    """
    log = logging.getLogger(__name__)
    log.info("="*70)
    log.info("INICIANDO PREDICCIÓN (Lógica de predict.py)")
    log.info("="*70)
    
    # --- 1. Definir Configuración (Defaults + Args) ---
    config = {
        'input_path': args.input,
        'output_path': args.output,
        'checkpoint_path': args.checkpoint,
        'device': args.device,
        'nodata_values': getattr(args, 'nodata', [0]),
        'scale_factor': getattr(args, 'scale_factor', 10000.0),
        'block_size': getattr(args, 'block_size', 1024),
        'gpu_batch_size': getattr(args, 'gpu_batch_size', 10000),
    }
    
    log.info(f"Configuración de predicción: {config}")

    # (El resto de esta función es idéntico a la run_prediction
    # original, pero leyendo desde 'config')
    
    global_stats = {
        'total_pixels': 0, 'nodata_pixels': 0, 'valid_pixels': 0,
        'out_of_range_pixels': 0, 'total_blocks': 0, 'processing_time': 0,
    }
    
    start_time = time.time()
    
    # 1. Cargar modelo (MLP extraído)
    model = load_model(config['checkpoint_path'], config['device'])
    
    # 2. Abrir imagen de entrada
    log.info(f"\nAbriendo imagen: {config['input_path']}")
    
    try:
        with rasterio.open(config['input_path']) as src:
            if src.count != 4:
                raise InvalidInputException(f"Imagen debe tener 4 bandas, tiene {src.count}")
            
            meta = src.meta.copy()
            meta.update(
                dtype=rasterio.uint16,
                nodata=int(config['nodata_values'][0]),
                compress='lzw',
            )
            
            # 3. Calcular ventanas
            block_size = config['block_size']
            windows = []
            for i in range(0, src.height, block_size):
                for j in range(0, src.width, block_size):
                    height = min(block_size, src.height - i)
                    width = min(block_size, src.width - j)
                    windows.append(Window(j, i, width, height))
            
            global_stats['total_blocks'] = len(windows)
            log.info(f"OK {len(windows)} bloques generados para procesar.")
            
            # 4. Procesar bloques
            log.info(f"\nCreando archivo de salida: {config['output_path']}")
            config['output_path'].parent.mkdir(parents=True, exist_ok=True)
            
            with rasterio.open(config['output_path'], 'w', **meta) as dst:
                pbar = tqdm(windows, desc="Procesando bloques", unit="bloque")
                
                for window in pbar:
                    block_data = src.read(window=window)
                    
                    processed_block, block_stats = process_block(
                        block=block_data,
                        mlp_model=model, # Pasar el MLP
                        nodata_values=config['nodata_values'],
                        scale_factor=config['scale_factor'],
                        device=config['device'],
                        gpu_batch_size=config['gpu_batch_size'],
                    )
                    
                    dst.write(processed_block, window=window)
                    
                    for key in ['total_pixels', 'nodata_pixels', 'valid_pixels', 'out_of_range_pixels']:
                        global_stats[key] += block_stats[key]
                        
    except rasterio.errors.RasterioIOError as e:
        raise InvalidInputException(f"Error al leer imagen: {e}") from e
    
    # 5. Estadísticas finales
    global_stats['processing_time'] = time.time() - start_time
    
    log.info("\n" + "="*70)
    log.info("OK PREDICCIÓN COMPLETADA")
    log.info("="*70)
    log.info(f"Estadísticas: {global_stats}")
    log.info(f"\nImagen guardada: {config['output_path']}")
    
    return global_stats