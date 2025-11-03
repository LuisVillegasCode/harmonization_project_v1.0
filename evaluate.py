# /harmonization_project/evaluate.py

"""
Script de evaluación para BCNet de armonización de imágenes satelitales.

Este script calcula las métricas de evaluación descritas en el paper
[cite: 73] y genera visualizaciones cualitativas (Figuras 7, 8, 9).

Métricas calculadas:
- RMSE (Root Mean Square Error)
- R² Score (Coefficient of Determination)
- Maximum Absolute Error
- Explained Variance Score

Visualizaciones generadas:
- Scatter plots 2D (histogramas) por banda
- Mapas de error de predicción (Figura 7)
- Mapas de alteración espacial (Figura 8)

Uso:
    # Evaluación básica
    python evaluate.py \\
        --checkpoint /models/best_model.pth \\
        --p1-path /data/perusat1.tif \\
        --s2-path /data/sentinel2.tif \\
        --cloud-mask /data/mask.tif \\
        --output-dir /outputs/evaluation
    
    # Con configuración personalizada
    python evaluate.py \\
        --checkpoint /models/best_model.pth \\
        --p1-path /data/perusat1.tif \\
        --s2-path /data/sentinel2.tif \\
        --cloud-mask /data/mask.tif \\
        --output-dir /outputs/evaluation \\
        --batch-size 100 \\
        --seed 42 \\
        --split-ratio 0.9
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    max_error,
    explained_variance_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importar módulos del proyecto
try:
    from src.data_loader import HarmonizationPatchDataset, create_train_test_split
    from src.architecture import BCNet
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.data_loader import HarmonizationPatchDataset, create_train_test_split
    from src.architecture import BCNet


# ============================================================================
# EXCEPCIONES PERSONALIZADAS
# ============================================================================

class EvaluationException(Exception):
    """Excepción base para errores de evaluación."""
    pass


class InvalidConfigurationException(EvaluationException):
    """Excepción para configuraciones inválidas."""
    pass


# ============================================================================
# CARGA DE MODELO
# ============================================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    """
    Carga modelo BCNet desde checkpoint.
    
    Args:
        checkpoint_path: Ruta al checkpoint (.pth).
        device: Device de PyTorch.
        
    Returns:
        Tupla (modelo, metadata_checkpoint).
        
    Raises:
        FileNotFoundError: Si checkpoint no existe.
        EvaluationException: Si hay error al cargar.
    """
    log = logging.getLogger(__name__)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
    
    log.info(f"Cargando modelo desde: {checkpoint_path}")
    
    try:
        # Cargar checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Instanciar modelo
        model = BCNet()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Mover a device y modo eval
        model = model.to(device)
        model.eval()
        
        # Extraer metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
            'total_iterations': checkpoint.get('total_iterations', 'unknown'),
        }
        
        log.info(
            f"OK Modelo cargado exitosamente\n"
            f"  Época: {metadata['epoch']}\n"
            f"  Val loss: {metadata['best_val_loss']}\n"
            f"  Iteraciones: {metadata['total_iterations']}"
        )
        
        return model, metadata
        
    except Exception as e:
        log.error(f"Error al cargar modelo: {e}", exc_info=True)
        raise EvaluationException(f"Fallo al cargar modelo: {e}") from e


# ============================================================================
# GENERACIÓN DE PREDICCIONES
# ============================================================================

def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """
    Genera predicciones sobre el conjunto de datos.
    
    Args:
        model: Modelo BCNet en modo eval.
        dataloader: DataLoader con datos de evaluación.
        device: Device de PyTorch.
        
    Returns:
        Tupla (predictions, targets, sample_batch):
            - predictions: Array [N, 4] con predicciones aplanadas.
            - targets: Array [N, 4] con targets aplanados.
            - sample_batch: Tupla (p1, s2, pred) del primer batch para viz.
    """
    log = logging.getLogger(__name__)
    log.info("Generando predicciones sobre conjunto de evaluación...")
    
    all_predictions = []
    all_targets = []
    sample_batch = None
    
    with torch.no_grad():
        for p1_batch, s2_batch in tqdm(dataloader, desc="Evaluando"):
            p1_batch = p1_batch.to(device)
            s2_batch = s2_batch.to(device)
            
            # Forward pass (modo training=True para evaluar modelo completo)
            # Output: [B, 4, 32, 32] @ 10m (con arquitectura corregida)
            pred_batch = model(p1_batch, training=True)
            
            # Validar dimensiones
            if pred_batch.shape != s2_batch.shape:
                raise EvaluationException(
                    f"Shape mismatch: pred {pred_batch.shape} vs "
                    f"target {s2_batch.shape}. Verifique la arquitectura."
                )
            
            # Guardar primer batch para visualización
            if sample_batch is None:
                sample_batch = (
                    p1_batch.cpu(),
                    s2_batch.cpu(),
                    pred_batch.cpu()
                )
            
            # Aplanar: [B, C, H, W] → [B*H*W, C]
            pred_flat = pred_batch.permute(0, 2, 3, 1).reshape(-1, 4)
            target_flat = s2_batch.permute(0, 2, 3, 1).reshape(-1, 4)
            
            all_predictions.append(pred_flat.cpu())
            all_targets.append(target_flat.cpu())
    
    # Concatenar todos los batches
    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()
    
    log.info(
        f"OK Predicciones generadas:\n"
        f"  Shape: {predictions.shape}\n"
        f"  Rango pred: [{predictions.min():.4f}, {predictions.max():.4f}]\n"
        f"  Rango target: [{targets.min():.4f}, {targets.max():.4f}]"
    )
    
    return predictions, targets, sample_batch


# ============================================================================
# CÁLCULO DE MÉTRICAS
# ============================================================================

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict:
    """
    Calcula métricas de evaluación del paper [cite: 73].
    
    Métricas calculadas por banda:
    - RMSE (Root Mean Square Error)
    - R² Score (Coefficient of Determination)
    - Maximum Absolute Error
    - Explained Variance Score
    
    Args:
        predictions: Array [N, 4] con predicciones.
        targets: Array [N, 4] con targets.
        
    Returns:
        Diccionario con métricas por banda y globales.
    """
    log = logging.getLogger(__name__)
    log.info("Calculando métricas de evaluación...")
    
    band_names = ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B8 (NIR)']
    
    results = {
        'bands': {},
        'global': {},
    }
    
    # Métricas por banda
    for i, band_name in enumerate(band_names):
        pred_band = predictions[:, i]
        target_band = targets[:, i]
        
        # Calcular métricas [cite: 73]
        rmse = math.sqrt(mean_squared_error(target_band, pred_band))
        r2 = r2_score(target_band, pred_band)
        max_err = max_error(target_band, pred_band)
        exp_var = explained_variance_score(target_band, pred_band)
        mae = np.mean(np.abs(target_band - pred_band))
        bias = np.mean(pred_band - target_band)
        
        results['bands'][band_name] = {
            'rmse': float(rmse),
            'r2_score': float(r2),
            'max_absolute_error': float(max_err),
            'explained_variance': float(exp_var),
            'mae': float(mae),
            'bias': float(bias),
        }
        
        log.info(
            f"\n{band_name}:\n"
            f"  RMSE: {rmse:.4f}\n"
            f"  R²: {r2:.4f}\n"
            f"  MAE: {mae:.4f}\n"
            f"  Max Error: {max_err:.4f}\n"
            f"  Explained Variance: {exp_var:.4f}\n"
            f"  Bias: {bias:.4f}"
        )
    
    # Métricas globales (todas las bandas concatenadas)
    pred_all = predictions.flatten()
    target_all = targets.flatten()
    
    results['global'] = {
        'rmse': float(math.sqrt(mean_squared_error(target_all, pred_all))),
        'r2_score': float(r2_score(target_all, pred_all)),
        'max_absolute_error': float(max_error(target_all, pred_all)),
        'explained_variance': float(explained_variance_score(target_all, pred_all)),
        'mae': float(np.mean(np.abs(target_all - pred_all))),
        'bias': float(np.mean(pred_all - target_all)),
    }
    
    log.info(
        f"\nMétricas Globales:\n"
        f"  RMSE: {results['global']['rmse']:.4f}\n"
        f"  R²: {results['global']['r2_score']:.4f}\n"
        f"  MAE: {results['global']['mae']:.4f}\n"
        f"  Explained Variance: {results['global']['explained_variance']:.4f}"
    )
    
    return results


# ============================================================================
# VISUALIZACIONES
# ============================================================================

def plot_scatter_plots(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Path,
) -> None:
    """
    Genera scatter plots 2D (histogramas) por banda (Figura 5 del paper).
    
    Args:
        predictions: Array [N, 4] con predicciones.
        targets: Array [N, 4] con targets.
        output_path: Ruta para guardar figura.
    """
    log = logging.getLogger(__name__)
    log.info(f"Generando scatter plots: {output_path}")
    
    band_names = ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B8 (NIR)']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        'Scatter Plots 2D: Predicción vs Referencia\n(Histogramas Bidimensionales)',
        fontsize=16,
        fontweight='bold'
    )
    
    for i, (ax, band_name) in enumerate(zip(axes.flat, band_names)):
        pred_band = predictions[:, i]
        target_band = targets[:, i]
        
        # Determinar rango común
        vmin = min(target_band.min(), pred_band.min())
        vmax = max(target_band.max(), pred_band.max())
        
        # Crear histograma 2D
        h = ax.hist2d(
            target_band,
            pred_band,
            bins=100,
            cmap='viridis',
            cmin=1,
            range=[[vmin, vmax], [vmin, vmax]]  # OK Rango explícito
        )
        
        # Línea 1:1 (predicción perfecta)
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2, label='1:1')
        
        # Calcular R² para el título
        r2 = r2_score(target_band, pred_band)
        
        ax.set_xlabel('Referencia S2 (Reflectancia)', fontsize=11)
        ax.set_ylabel('Predicción Modelo (Reflectancia)', fontsize=11)
        ax.set_title(f'{band_name} (R²={r2:.4f})', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')
        
        # Colorbar
        plt.colorbar(h[3], ax=ax, label='Densidad')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    log.info(f"OK Scatter plots guardados: {output_path}")


def plot_error_maps(
    model: nn.Module,
    sample_batch: Tuple,
    device: torch.device,
    output_path: Path,
) -> None:
    """
    Genera mapas de error (Figuras 7 y 8 del paper).
    
    Figura 7: Mapa de error de predicción |S2 - Pred|
    Figura 8: Mapa de alteración espacial (InputModule vs pixel-wise)
    
    Args:
        model: Modelo BCNet.
        sample_batch: Tupla (p1, s2, pred) del primer batch.
        device: Device de PyTorch.
        output_path: Ruta para guardar figura.
    """
    log = logging.getLogger(__name__)
    log.info(f"Generando mapas de error: {output_path}")
    
    p1_batch, s2_batch, pred_batch = sample_batch
    
    # Tomar primera imagen del batch
    p1_img = p1_batch[0].numpy()      # [4, 160, 160] @ 2m
    s2_img = s2_batch[0].numpy()      # [4, 32, 32] @ 10m
    pred_img = pred_batch[0].numpy()  # [4, 32, 32] @ 10m
    
    # --- Mapa de Error de Predicción (Figura 7) ---
    error_pred = np.abs(s2_img - pred_img)
    
    # --- Mapa de Alteración Espacial (Figura 8) ---
    # Comparar InputModule output vs pixel-wise MLP output
    
    with torch.no_grad():
        # Predicción con InputModule (training=True)
        pred_with_input = model(
            p1_batch[0:1].to(device),
            training=True
        ).cpu().squeeze(0).numpy()  # [4, 32, 32]
        
        # Predicción pixel-wise (training=False)
        pred_pixelwise = model(
            p1_batch[0:1].to(device),
            training=False
        ).cpu().squeeze(0).numpy()  # [4, 160, 160]
        
        # Para comparar, necesitamos downsample pred_pixelwise a 32x32
        # usando average pooling
        import torch.nn.functional as F
        pred_pixelwise_down = F.avg_pool2d(
            torch.from_numpy(pred_pixelwise).unsqueeze(0),
            kernel_size=5,
            stride=5
        ).squeeze(0).numpy()  # [4, 32, 32]
    
    alteration_map = np.abs(pred_with_input - pred_pixelwise_down)
    
    # --- Normalización para visualización RGB ---
    def normalize_for_viz(img, percentile=98):
        """Normaliza imagen usando percentiles para mejor visualización."""
        vmin = np.percentile(img, 2)
        vmax = np.percentile(img, percentile)
        return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    # Seleccionar bandas RGB (R=B4, G=B3, B=B2 = índices 2,1,0)
    rgb_indices = [2, 1, 0]
    
    s2_rgb = np.stack([normalize_for_viz(s2_img[i]) for i in rgb_indices], axis=-1)
    pred_rgb = np.stack([normalize_for_viz(pred_img[i]) for i in rgb_indices], axis=-1)
    error_rgb = np.stack([normalize_for_viz(error_pred[i]) for i in rgb_indices], axis=-1)
    alteration_rgb = np.stack([normalize_for_viz(alteration_map[i]) for i in rgb_indices], axis=-1)
    
    # --- Crear figura ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        'Análisis Visual de Errores (Muestra del Primer Batch)',
        fontsize=16,
        fontweight='bold'
    )
    
    axes[0, 0].imshow(s2_rgb)
    axes[0, 0].set_title('Referencia S2 @ 10m', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_rgb)
    axes[0, 1].set_title('Predicción Modelo @ 10m', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(error_rgb)
    axes[1, 0].set_title(
        'Mapa de Error de Predicción (Fig 7)\n|S2 - Pred|',
        fontsize=12,
        fontweight='bold'
    )
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(alteration_rgb)
    axes[1, 1].set_title(
        'Mapa de Alteración Espacial (Fig 8)\n|InputModule - Pixel-wise|',
        fontsize=12,
        fontweight='bold'
    )
    axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    log.info(f"OK Mapas de error guardados: {output_path}")


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_evaluation(args: argparse.Namespace) -> Dict:
    """
    Ejecuta el pipeline completo de evaluación (llamado desde main.py).
    
    Args:
        args: Namespace de argparse proveniente de main.py.
              Debe contener (mínimo):
              - checkpoint, p1_path, s2_path, cloud_mask, output_dir, device
              
    Returns:
        Diccionario con métricas de evaluación.
    """
    log = logging.getLogger(__name__)
    
    log.info("="*70)
    log.info("INICIANDO EVALUACIÓN (Lógica de evaluate.py)")
    log.info("="*70)
    
    # --- 1. Definir Configuración (Defaults + Args) ---
    # (Usamos getattr para tomar defaults si main.py no los proveyó)
    config = {
        'checkpoint_path': args.checkpoint,
        'p1_path': args.p1_path,
        's2_path': args.s2_path,
        'cloud_mask_path': args.cloud_mask,
        'output_dir': args.output_dir,
        'device': args.device,
        'p1_nodata': getattr(args, 'p1_nodata', [0]),
        's2_nodata': getattr(args, 's2_nodata', [0]),
        'cloud_values': getattr(args, 'cloud_values', [0,3, 8, 9, 10]),
        'scale_factor': getattr(args, 'scale_factor', 10000.0),
        'batch_size': getattr(args, 'batch_size', 100),
        'split_ratio': getattr(args, 'split_ratio', 0.9),
        'seed': getattr(args, 'seed', 42),
    }
    
    log.info(f"Configuración de evaluación: {config}")
    
    start_time = time.time()
    
    # 1. Configurar seeds
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    log.info(f"Seeds configuradas: {config['seed']}")
    
    # 2. Cargar modelo
    model, model_metadata = load_model(config['checkpoint_path'], config['device'])
    
    # 3. Cargar dataset
    log.info("\n" + "-"*70 + "\nCARGANDO DATASET\n" + "-"*70)
    
    try:
        dataset = HarmonizationPatchDataset(
            p1_path=config['p1_path'],
            s2_path=config['s2_path'],
            s2_cloud_mask_path=config['cloud_mask_path'],
            p1_nodata_values=config['p1_nodata'],
            s2_nodata_values=config['s2_nodata'],
            s2_cloud_values=config['cloud_values'],
            scale_factor=config['scale_factor'],
            shuffle_patches=True,
            validate_range=True,
        )
        log.info("Extrayendo patches del dataset...")
        dataset.load()
        log.info(f"OK Dataset cargado: {len(dataset)} patches")
        
    except Exception as e:
        log.error(f"Error al cargar dataset: {e}", exc_info=True)
        raise EvaluationException(f"Fallo al cargar dataset: {e}") from e
    
    # 4. Split train/val
    log.info("\n" + "-"*70 + "\nDIVISIÓN TRAIN/VAL\n" + "-"*70)
    
    _, val_dataset = create_train_test_split(
        dataset,
        test_ratio=1.0 - config['split_ratio'],
        seed=config['seed']
    )
    log.info(f"Conjunto de validación: {len(val_dataset)} patches")
    
    # 5. Crear DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4, # (Podríamos añadir 'num_workers' a args)
        pin_memory=(config['device'].type == 'cuda')
    )
    
    # 6. Generar predicciones
    log.info("\n" + "-"*70 + "\nGENERANDO PREDICCIONES\n" + "-"*70)
    
    predictions, targets, sample_batch = get_predictions(
        model, val_loader, config['device']
    )
    
    # 7. Calcular métricas
    log.info("\n" + "-"*70 + "\nCALCULANDO MÉTRICAS\n" + "-"*70)
    
    metrics = calculate_metrics(predictions, targets)
    
    # 8. Generar visualizaciones
    log.info("\n" + "-"*70 + "\nGENERANDO VISUALIZACIONES\n" + "-"*70)
    
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    plot_scatter_plots(
        predictions,
        targets,
        config['output_dir'] / 'scatter_plots.png'
    )
    
    plot_error_maps(
        model,
        sample_batch,
        config['device'],
        config['output_dir'] / 'error_maps.png'
    )
    
    # 9. Guardar métricas
    metrics_path = config['output_dir'] / 'metrics.json'
    
    evaluation_results = {
        'metrics': metrics,
        'model_metadata': model_metadata,
        'dataset_info': {
            'total_patches': len(dataset),
            'val_patches': len(val_dataset),
            'split_ratio': config['split_ratio'],
        },
        'evaluation_time': time.time() - start_time,
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    log.info(f"\nOK Métricas guardadas: {metrics_path}")
    
    # 10. Resumen final
    log.info("\n" + "="*70)
    log.info("OK EVALUACIÓN COMPLETADA")
    log.info("="*70)
    
    return evaluation_results

def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """
    Configura logging para archivo y consola (llamado desde main.py).
    
    Args:
        output_dir: Directorio donde guardar el log.
        verbose: Si True, usa nivel DEBUG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'evaluation.log'
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Limpiar handlers existentes
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # 'w' para sobrescribir logs antiguos
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    log = logging.getLogger(__name__)
    log.info(f"Logging de evaluación configurado: {log_file}")