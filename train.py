# /harmonization_project/train.py

"""
Script de entrenamiento para BCNet de armonización de imágenes satelitales.

Este script orquesta el pipeline completo de entrenamiento:
1. Carga y validación de datos
2. División train/val (90/10)
3. Configuración de modelo y optimizador
4. Entrenamiento con ~5000 iteraciones [cite: 106]
5. Guardado de checkpoints y logs

Uso:
    # Configuración básica
    python train.py \\
        --p1-path /data/perusat1.tif \\
        --s2-path /data/sentinel2.tif \\
        --cloud-mask /data/mask.tif \\
        --output-dir /outputs/exp_001
    
    # Configuración avanzada
    python train.py \\
        --p1-path /data/perusat1.tif \\
        --s2-path /data/sentinel2.tif \\
        --cloud-mask /data/mask.tif \\
        --output-dir /outputs/exp_001 \\
        --batch-size 100 \\
        --num-iterations 5000 \\
        --learning-rate 0.0002 \\
        --num-workers 4 \\
        --resume-from /outputs/exp_001/checkpoints/last_checkpoint.pth
    
    # PeruSat @ 2.8m (nativo)
    python train.py \\
        --p1-path /data/perusat1_native.tif \\
        --s2-path /data/sentinel2.tif \\
        --cloud-mask /data/mask.tif \\
        --output-dir /outputs/exp_002 \\
        --perusat-resolution 2.8 \\
        --perusat-variant native
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, random_split

# Importar módulos del proyecto
try:
    from src.data_loader import HarmonizationPatchDataset, create_train_test_split
    from src.architecture import BCNet, create_bcnet_for_perusat
    from src.losses import RelativeErrorLoss
    from src.trainer import HarmonizationTrainer
except ImportError as e:
    # Fallback: permitir ejecutar desde raíz del proyecto
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from src.data_loader import HarmonizationPatchDataset, create_train_test_split
        from src.architecture import BCNet, create_bcnet_for_perusat
        from src.losses import RelativeErrorLoss
        from src.trainer import HarmonizationTrainer
    except ImportError:
        print(
            "ERROR: No se pudieron importar los módulos del proyecto.\n"
            "Asegúrese de que los archivos en src/ estén disponibles:\n"
            "  - src/data_loader.py\n"
            "  - src/architecture.py\n"
            "  - src/losses.py\n"
            "  - src/trainer.py"
        )
        sys.exit(1)


# ============================================================================
# EXCEPCIONES PERSONALIZADAS
# ============================================================================

class TrainingException(Exception):
    """Excepción base para errores durante entrenamiento."""
    pass


class ConfigurationException(TrainingException):
    """Excepción para errores de configuración."""
    pass


# ============================================================================
# CONFIGURACIÓN Y ARGUMENTOS
# ============================================================================

def validate_configuration(config: Dict[str, Any]) -> None:
    """
    Valida que la configuración (ya fusionada con defaults) sea correcta.
    
    Args:
        config: Diccionario de configuración (no args).
        
    Raises:
        ConfigurationException: Si la configuración es inválida.
    """
    # Validar que archivos de entrada existen
    if not config['p1_path'].exists():
        raise ConfigurationException(
            f"Archivo PeruSat-1 no encontrado: {config['p1_path']}"
        )
    
    if not config['s2_path'].exists():
        raise ConfigurationException(
            f"Archivo Sentinel-2 no encontrado: {config['s2_path']}"
        )
    
    if not config['cloud_mask'].exists():
        raise ConfigurationException(
            f"Máscara de nubes no encontrada: {config['cloud_mask']}"
        )
    
    # Validar parámetros numéricos (ahora leyendo desde 'config')
    if config['batch_size'] <= 0:
        raise ConfigurationException(
            f"batch_size debe ser > 0, pero es {config['batch_size']}"
        )
    
    if config['num_iterations'] <= 0:
        raise ConfigurationException(
            f"num_iterations debe ser > 0, pero es {config['num_iterations']}"
        )
    
    if not (0.0 < config['split_ratio'] < 1.0):
        raise ConfigurationException(
            f"split_ratio debe estar en (0, 1), pero es {config['split_ratio']}"
        )
    
    if config['learning_rate'] <= 0:
        raise ConfigurationException(
            f"learning_rate debe ser > 0, pero es {config['learning_rate']}"
        )
    
    # Validar combinación PeruSat resolution/variant
    if config['perusat_resolution'] == 2.0 and config['perusat_variant'] != 'resampled':
        raise ConfigurationException(
            "Con resolution=2.0m, debe usar variant='resampled'"
        )
    
    if config['perusat_resolution'] == 2.8 and config['perusat_variant'] != 'native':
        raise ConfigurationException(
            "Con resolution=2.8m, debe usar variant='native'"
        )
    
    # Validar checkpoint si se especifica
    if config['resume_from'] is not None:
        if not config['resume_from'].exists():
            raise ConfigurationException(
                f"Checkpoint no encontrado: {config['resume_from']}"
            )


def save_configuration(config: Dict[str, Any], output_dir: Path) -> None:
    """
    Guarda la configuración usada en formato JSON.
    
    Args:
        config: Diccionario de configuración (no args).
        output_dir: Directorio de salida.
    """
    config_dict = config.copy()
    
    # Convertir Paths y Device a strings para JSON
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
        elif isinstance(value, torch.device):
            config_dict[key] = str(value)
    
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logging.info(f"Configuración guardada: {config_path}")


# ============================================================================
# SETUP
# ============================================================================

def set_random_seeds(seed: int) -> None:
    """
    Configura seeds para reproducibilidad.
    
    Args:
        seed: Semilla aleatoria.
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Para reproducibilidad completa (puede reducir performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"Seeds configuradas: {seed}")


def test_gpu_memory(batch_size: int, device: torch.device) -> bool:
    """
    Prueba si hay suficiente memoria GPU para el batch_size.
    
    Args:
        batch_size: Tamaño de batch a probar.
        device: Device de PyTorch.
        
    Returns:
        True si el test pasa, False si hay OOM.
    """
    if device.type == 'cpu':
        return True
    
    log = logging.getLogger(__name__)
    log.info(f"Probando memoria GPU con batch_size={batch_size}...")
    
    try:
        # Crear modelo y datos ficticios
        model = BCNet().to(device)
        dummy_input = torch.rand(batch_size, 4, 160, 160).to(device)
        dummy_target = torch.rand(batch_size, 4, 32, 32).to(device)
        
        # Forward + backward pass
        output = model(dummy_input, training=True)
        loss = torch.nn.functional.mse_loss(output, dummy_target)
        loss.backward()
        
        # Limpiar
        del model, dummy_input, dummy_target, output, loss
        torch.cuda.empty_cache()
        
        log.info(" Test de memoria GPU pasado")
        return True
        
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            log.error(f"X GPU sin memoria suficiente para batch_size={batch_size}")
            return False
        else:
            raise


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de entrenamiento (llamado desde main.py).
    
    Args:
        args: Namespace de argparse proveniente de main.py.
              
    Returns:
        Diccionario con el historial de entrenamiento.
        
    Raises:
        TrainingException: Si ocurre un error durante el entrenamiento.
        ConfigurationException: Si la validación de args falla.
    """
    log = logging.getLogger(__name__)
    
    log.info("="*70)
    log.info("INICIANDO PIPELINE DE ENTRENAMIENTO (Lógica de train.py)")
    log.info("="*70)
    
    # --- 1. Definir Configuración (Defaults + Args) ---
    # (Usamos getattr para tomar defaults si main.py no los proveyó)
    config = {
        'p1_path': args.p1_path,
        's2_path': args.s2_path,
        'cloud_mask': args.cloud_mask,
        'output_dir': args.output_dir,
        'device': args.device,
        'resume_from': getattr(args, 'resume_from', None),
        'p1_nodata': getattr(args, 'p1_nodata', [0]),
        's2_nodata': getattr(args, 's2_nodata', [0]),
        'cloud_values': getattr(args, 'cloud_values', [0,3, 8, 9, 10]),
        'scale_factor': getattr(args, 'scale_factor', 10000.0),
        'perusat_resolution': getattr(args, 'perusat_resolution', 2.0),
        'perusat_variant': getattr(args, 'perusat_variant', 'resampled'),
        'batch_size': getattr(args, 'batch_size', 100),
        'num_iterations': getattr(args, 'num_iterations', 5000),
        'learning_rate': getattr(args, 'learning_rate', 0.0002),
        'split_ratio': getattr(args, 'split_ratio', 0.9),
        'seed': getattr(args, 'seed', 42),
        'num_workers': getattr(args, 'num_workers', 0),
        'gradient_clip': getattr(args, 'gradient_clip', None),
        'save_filters_every': getattr(args, 'save_filters_every', None),
        'dry_run': getattr(args, 'dry_run', False),
    }

    log.info(f"Configuración de entrenamiento (combinada): {config}")

    # --- 2. Validar y Guardar Configuración (CORREGIDO) ---
    try:
        log.info("Validando configuración...")
        # (Llamamos a validate_configuration *DESPUÉS* de crear el 'config' dict)
        validate_configuration(config) 
        log.info("[OK] Configuración válida") # (Reemplazado OK)
        
        # (Guardamos el 'config' dict, no 'args')
        save_configuration(config, config['output_dir'])
        
    except ConfigurationException as e:
        log.error(f"Error de configuración: {e}", exc_info=True)
        raise

    # 3. Configurar seeds
    set_random_seeds(config['seed'])
    
    # 4. Obtener device
    device = config['device']
    log.info(f"Device seleccionado: {device}")
    
    if device.type == 'cuda':
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 5. Test de memoria GPU
    if device.type == 'cuda':
        if not test_gpu_memory(config['batch_size'], device):
            log.warning(
                f"Batch size {config['batch_size']} puede causar OOM."
            )

    # 6. Cargar y validar dataset
    log.info("\n" + "-"*70) # (Reemplazado -)
    log.info("CARGANDO DATASET")
    log.info("-"*70) # (Reemplazado -)
    
    try:
        dataset = HarmonizationPatchDataset(
            p1_path=config['p1_path'],
            s2_path=config['s2_path'],
            s2_cloud_mask_path=config['cloud_mask'],
            p1_nodata_values=config['p1_nodata'],
            s2_nodata_values=config['s2_nodata'],
            s2_cloud_values=config['cloud_values'],
            scale_factor=config['scale_factor'],
            expected_resolution_factor=5 if config['perusat_resolution'] == 2.0 else None,
            shuffle_patches=True,
            validate_range=True,
        )
        
        log.info("Extrayendo patches del dataset...")
        dataset.load()
        log.info(f"[OK] Dataset cargado: {len(dataset)} patches válidos") # (Reemplazado OK)
        
        stats = dataset.get_statistics()
        log.info(
            f"Estadísticas del dataset:\n"
            f"  Patches P1: {stats['p1_shape']}\n"
            f"  Patches S2: {stats['s2_shape']}\n"
            f"  Rango P1: [{stats['p1_min']:.4f}, {stats['p1_max']:.4f}]\n"
            f"  Rango S2: [{stats['s2_min']:.4f}, {stats['s2_max']:.4f}]"
        )
        
    except Exception as e:
        log.error(f"Error al cargar dataset: {e}", exc_info=True)
        raise TrainingException(f"Fallo al cargar dataset: {e}") from e
    
    if len(dataset) < 10:
        raise TrainingException(
            f"Dataset demasiado pequeño: {len(dataset)} patches."
        )
    
    # 7. Split train/val
    log.info("\n" + "-"*70 + "\nDIVISIÓN TRAIN/VAL\n" + "-"*70) # (Reemplazado -)
    
    train_dataset, val_dataset = create_train_test_split(
        dataset,
        test_ratio=1.0 - config['split_ratio'],
        seed=config['seed']
    )
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    log.info(f"Train: {train_size} patches ({config['split_ratio']*100:.0f}%)")
    log.info(f"Val:   {val_size} patches ({(1-config['split_ratio'])*100:.0f}%)")
    
    # 8. Crear DataLoaders
    log.info("\n" + "-"*70 + "\nCONFIGURANDO DATALOADERS\n" + "-"*70) # (Reemplazado -)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )
    
    # 9. Calcular épocas
    batches_per_epoch = len(train_loader)
    if batches_per_epoch == 0:
        raise TrainingException("El DataLoader de entrenamiento está vacío.")
        
    num_epochs = max(1, math.ceil(config['num_iterations'] / batches_per_epoch))
    total_iterations = num_epochs * batches_per_epoch
    log.info(f"Épocas calculadas: {num_epochs} (Total iteraciones: ~{total_iterations})")

    # 10. Crear modelo
    log.info("\n" + "-"*70 + "\nCONFIGURANDO MODELO\n" + "-"*70) # (Reemplazado -)
    
    model = create_bcnet_for_perusat(
        resolution=config['perusat_resolution'],
        variant=config['perusat_variant']
    )
    params = model.count_parameters()
    log.info(
        f"Modelo BCNet creado (Variante: {config['perusat_variant']})\n"
        f"  Parámetros totales: {params['total']:,}"
    )
    
    # 11. Crear función de pérdida
    loss_fn = RelativeErrorLoss(
        epsilon=1e-6,
        clamp_prediction=True,
        validate_range=True,
        log_statistics=False
    )
    log.info(f"Función de pérdida: {type(loss_fn).__name__}")
    
    # 12. Crear trainer
    log.info("\n" + "-"*70 + "\nCONFIGURANDO TRAINER\n" + "-"*70) # (Reemplazado -)
    
    trainer = HarmonizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        output_dir=config['output_dir'],
        learning_rate=config['learning_rate'],
        target_iterations=config['num_iterations'],
        gradient_clip_norm=config['gradient_clip'],
        save_filters_every=config['save_filters_every'],
    )
    log.info("Trainer configurado correctamente")
    
    # 13. Modo dry-run
    if config['dry_run']:
        log.info("\n" + "="*70)
        log.info("MODO DRY-RUN ACTIVADO. Configuración validada.")
        log.info("No se ejecutará entrenamiento real.")
        log.info("="*70)
        return {"status": "dry-run", "epochs": 0, "best_val_loss": -1}
    
    # 14. Ejecutar entrenamiento
    log.info("\n" + "="*70)
    log.info("EJECUTANDO ENTRENAMIENTO")
    log.info("="*70)
    
    try:
            history = trainer.train(resume_from=config['resume_from'])
            
            log.info("\n" + "="*70)
            log.info("[OK] ENTRENAMIENTO COMPLETADO EXITOSAMENTE") # (Caracter ASCII)
            log.info("="*70)
            
            return history # Retorna el historial a main.py
            
    except Exception as e:
            log.error(f"[X] Error durante entrenamiento: {e}", exc_info=True) # (Caracter ASCII)
            raise TrainingException(f"Entrenamiento falló: {e}") from e