# /harmonization_project/main.py

"""
Orquestador principal del pipeline BCNet (v2.0 - Refactorizado).

Este script proporciona una interfaz de CLI unificada y limpia para
ejecutar todas las etapas del pipeline. Los hiperparámetros
(batch size, lr, etc.) se definen como valores por defecto en los
módulos de lógica (train.py, predict.py) para mantener este
orquestador limpio.

Modos de ejecución:
  train    - Entrenar modelo
  predict  - Aplicar modelo a imagen
  evaluate - Evaluar modelo en test set
  all      - Pipeline completo (train -> predict -> evaluate)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Importar las funciones de LÓGICA (ya no los 'main' de los scripts)
try:
    # Asumimos que train/predict/evaluate están refactorizados
    from train import run_training
    from predict import run_prediction
    from evaluate import run_evaluation
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from train import run_training
        from predict import run_prediction
        from evaluate import run_evaluation
    except ImportError as e:
        print(f"ERROR: No se pudieron importar los módulos de lógica. {e}", file=sys.stderr)
        sys.exit(1)

# --- Configuración de Logging ---
def setup_logging(log_path: Path, verbose: bool = False) -> logging.Logger:
    """Configura logging global para el pipeline."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    
    # Limpiar handlers existentes para evitar duplicados
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logger = logging.getLogger('pipeline')
    logger.info(f"Logging configurado: {log_path}")
    return logger

# --- Definición de Argumentos (Mínima) ---
def create_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos con subcomandos."""
    
    parser = argparse.ArgumentParser(
        description="Pipeline BCNet para armonización PeruSat-1 / Sentinel-2.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # --- Argumentos Globales ---
    parser.add_argument(
        '--verbose', action='store_true', help='Logging verboso (nivel DEBUG)'
    )
    parser.add_argument(
        '--cpu', action='store_true', help='Forzar uso de CPU'
    )
    
    subparsers = parser.add_subparsers(
        dest='mode', required=True, help='Modo de ejecución'
    )
    
    # --- Parser Padre: DATOS COMUNES (p1, s2, mask) ---
    common_data_parser = argparse.ArgumentParser(add_help=False)
    common_data_parser.add_argument(
        "--p1-path", type=Path, required=True,
        help="Ruta al archivo GeoTIFF de PeruSat-1 (alta res, 2m)."
    )
    common_data_parser.add_argument(
        "--s2-path", type=Path, required=True,
        help="Ruta al archivo GeoTIFF de Sentinel-2 (baja res, 10m)."
    )
    common_data_parser.add_argument(
        "--cloud-mask", type=Path, required=True,
        help="Ruta al GeoTIFF de la máscara de nubes de S2."
    )
    common_data_parser.add_argument(
        '--p1-nodata', type=float, nargs='+', default=[0], help='Valor nodata P1'
    )
    common_data_parser.add_argument(
        '--s2-nodata', type=float, nargs='+', default=[0], help='Valor nodata S2'
    )
    common_data_parser.add_argument(
        '--cloud-values', type=int, nargs='+', default=[3, 8, 9, 10], help='Valores de nube'
    )
    common_data_parser.add_argument(
        '--scale-factor', type=float, default=10000.0, help='Factor de escala'
    )
    
    # --- NUEVO Parser Padre: PARÁMETROS DE ENTRENAMIENTO ---
    # (Argumentos que train y all necesitan, pero predict y evaluate no)
    training_params_parser = argparse.ArgumentParser(add_help=False)
    training_params_parser.add_argument(
        '--perusat-resolution', type=float, choices=[2.0, 2.8], default=2.0
    )
    training_params_parser.add_argument(
        '--perusat-variant', type=str, choices=['resampled', 'native'], default='resampled'
    )
    training_params_parser.add_argument(
        '--batch-size', type=int, default=100
    )
    training_params_parser.add_argument(
        '--num-iterations', type=int, default=5000
    )
    training_params_parser.add_argument(
        '--learning-rate', type=float, default=0.0002
    )
    training_params_parser.add_argument(
        '--seed', type=int, default=42
    )
    training_params_parser.add_argument(
        '--num-workers', type=int, default=0
    )
    training_params_parser.add_argument(
        '--split-ratio', type=float, default=0.9
    )
    training_params_parser.add_argument(
        '--gradient-clip', type=float, default=None
    )
    training_params_parser.add_argument(
        '--resume-from', type=Path, default=None
    )
    training_params_parser.add_argument(
        '--save-filters-every', type=int, default=None
    )
    training_params_parser.add_argument(
        '--dry-run', action='store_true'
    )
    
    # --- Sub-comando: train ---
    # MODIFICADO: Ahora hereda de ambos parsers padre
    train_parser = subparsers.add_parser(
        'train', 
        help='Entrenar modelo BCNet',
        parents=[common_data_parser, training_params_parser], # <-- HERENCIA
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument(
        '--output-dir', type=Path, required=True, 
        help='Directorio de salida para checkpoints y logs'
    )

    # --- Sub-comando: predict ---
    # (Sin cambios)
    predict_parser = subparsers.add_parser(
        'predict', help='Aplicar modelo a imagen',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    predict_required = predict_parser.add_argument_group('Argumentos requeridos')
    predict_required.add_argument('--checkpoint', type=Path, required=True)
    predict_required.add_argument('--input', type=Path, required=True)
    predict_required.add_argument('--output', type=Path, required=True)
    predict_params = predict_parser.add_argument_group('Parámetros')
    predict_params.add_argument('--nodata', type=float, nargs='+', default=[0])
    predict_params.add_argument('--scale-factor', type=float, default=10000.0)
    predict_params.add_argument('--block-size', type=int, default=1024)
    predict_params.add_argument('--gpu-batch-size', type=int, default=10000)

    # --- Sub-comando: evaluate ---
    # (Sin cambios)
    evaluate_parser = subparsers.add_parser(
        'evaluate', help='Evaluar modelo en test set',
        parents=[common_data_parser], # Solo hereda datos
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    eval_required = evaluate_parser.add_argument_group('Argumentos requeridos')
    eval_required.add_argument('--checkpoint', type=Path, required=True)
    eval_required.add_argument('--output-dir', type=Path, required=True)
    eval_params = evaluate_parser.add_argument_group('Parámetros')
    eval_params.add_argument('--batch-size', type=int, default=100)
    eval_params.add_argument('--seed', type=int, default=42)
    eval_params.add_argument('--split-ratio', type=float, default=0.9)
    
    # --- Sub-comando: all ---
    # MODIFICADO: Ahora hereda de ambos parsers padre
    all_parser = subparsers.add_parser(
        'all', 
        help='Ejecutar pipeline completo (train -> predict -> evaluate)',
        parents=[common_data_parser, training_params_parser], # <-- HERENCIA
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    all_required = all_parser.add_argument_group('Argumentos (requeridos para \'all\')')
    all_required.add_argument('--predict-input', type=Path, required=True, help='Imagen P1 completa para predicción')
    all_required.add_argument('--output-dir', type=Path, required=True, help='Directorio raíz de salida')
    # Argumentos específicos de predicción (si son diferentes al default)
    all_predict = all_parser.add_argument_group('Parámetros (opcionales para \'all\')')
    all_predict.add_argument('--block-size', type=int, default=1024)
    all_predict.add_argument('--gpu-batch-size', type=int, default=10000)
    
    return parser

# --- Punto de Entrada Principal ---
def main():
    """Punto de entrada principal del orquestador."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Determinar directorio de log
    if args.mode == 'train' or args.mode == 'all' or args.mode == 'evaluate':
        log_dir = args.output_dir
    elif args.mode == 'predict':
        log_dir = args.output.parent
    else:
        log_dir = Path.cwd()
    
    log = setup_logging(log_dir / "pipeline.log", verbose=args.verbose)
    
    log.info(f"Iniciando modo: {args.mode}")
    log.info(f"PyTorch version: {torch.__version__}")
    
    # Seleccionar device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device seleccionado: {device}")
    
    # Inyectar device en args
    args.device = device
    
    try:
        # Ejecutar modo seleccionado
        if args.mode == 'train':
            run_training(args)
            
        elif args.mode == 'predict':
            run_prediction(args)
            
        elif args.mode == 'evaluate':
            run_evaluation(args)
            
        elif args.mode == 'all':
            # --- Pipeline Secuencial ---
            log.info("="*70)
            log.info("MODO: PIPELINE COMPLETO")
            log.info("="*70)

            # Paso 1: Entrenar
            log.info("\n" + "-"*70 + "\nPASO 1/3: ENTRENAMIENTO\n" + "-"*70)
            run_training(args)
            
            # Paso 2: Predecir
            log.info("\n" + "-"*70 + "\nPASO 2/3: PREDICCIÓN\n" + "-"*70)
            checkpoint_path = args.output_dir / 'checkpoints' / 'best_model.pth'
            predict_output = args.output_dir / 'prediction' / f'{args.predict_input.stem}_harmonized.tif'
            predict_output.parent.mkdir(parents=True, exist_ok=True)
            
            # Crear args para predicción
            predict_args = argparse.Namespace(
                checkpoint=checkpoint_path,
                input=args.predict_input,
                output=predict_output,
                device=device
                # (El resto de args se tomarán de los defaults en predict.py)
            )
            run_prediction(predict_args)
            
            # Paso 3: Evaluar
            log.info("\n" + "-"*70 + "\nPASO 3/3: EVALUACIÓN\n" + "-"*70)
            eval_output_dir = args.output_dir / 'evaluation'
            
            # Crear args para evaluación
            eval_args = argparse.Namespace(
                checkpoint=checkpoint_path,
                p1_path=args.p1_path,
                s2_path=args.s2_path,
                cloud_mask=args.cloud_mask,
                output_dir=eval_output_dir,
                device=device
                # (El resto de args se tomarán de los defaults en evaluate.py)
            )
            run_evaluation(eval_args)
            
            log.info("\n" + "="*70 + "\nPIPELINE COMPLETO FINALIZADO\n" + "="*70)

    except KeyboardInterrupt:
        log.warning("\n Ejecución interrumpida por el usuario (Ctrl+C)")
        return 130
        
    except Exception as e:
        log.error(f"\n Error fatal en el pipeline: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())