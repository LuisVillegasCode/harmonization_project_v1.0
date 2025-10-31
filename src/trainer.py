# /harmonization_project/src/trainer.py

"""
Trainer para modelo BCNet de armonización de imágenes satelitales.

Este módulo implementa el bucle de entrenamiento descrito en 
Michel & Inglada (2021): "Learning Harmonised Pleiades and 
Sentinel-2 Surface Reflectances".

Características:
- Optimizador Adam con LR=0.0002 [cite: 108]
- ~5000 iteraciones de entrenamiento [cite: 106]
- Re-shuffle de datos cada época [cite: 108]
- Logging de curvas de training/validation (Figuras 5, 6, 9)
- Guardado de filtros aprendidos (Figura 10)
- Cálculo de métricas múltiples (Figura 3)
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Importaciones locales
from .architecture import BCNet
from .losses import RelativeErrorLoss

# Configurar logger
log = logging.getLogger(__name__)


# ============================================================================
# EXCEPCIONES PERSONALIZADAS
# ============================================================================

class TrainerException(Exception):
    """Excepción base para errores del Trainer."""
    pass


class InvalidConfigurationException(TrainerException):
    """Excepción para configuraciones inválidas."""
    pass


# ============================================================================
# TRAINER PRINCIPAL
# ============================================================================

class HarmonizationTrainer:
    """
    Trainer para modelo BCNet de armonización.
    
    Implementa la metodología del paper [cite: 106-109]:
    - Adam optimizer con LR=0.0002
    - ~5000 iteraciones de entrenamiento
    - Re-shuffle cada época
    - Guardado de mejor modelo
    - Logging de historial de entrenamiento
    
    Args:
        model: Instancia de BCNet.
        train_loader: DataLoader de entrenamiento (debe tener shuffle=True).
        val_loader: DataLoader de validación.
        loss_fn: Función de pérdida (RelativeErrorLoss).
        device: Device de PyTorch (cpu o cuda).
        output_dir: Directorio para guardar checkpoints y logs.
        learning_rate: Tasa de aprendizaje [cite: 108] (default: 0.0002).
        target_iterations: Número objetivo de iteraciones [cite: 106] (default: 5000).
        gradient_clip_norm: Norma máxima de gradientes (None=sin clipping).
        save_filters_every: Guardar filtros cada N épocas (None=solo al final).
        
    Example:
        >>> trainer = HarmonizationTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     loss_fn=RelativeErrorLoss(),
        ...     device=torch.device('cuda'),
        ...     output_dir=Path('./outputs/experiment_001'),
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: BCNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: RelativeErrorLoss,
        device: torch.device,
        output_dir: Path,
        learning_rate: float = 0.0002,  # [cite: 108]
        target_iterations: int = 5000,   # [cite: 106]
        gradient_clip_norm: Optional[float] = None,
        save_filters_every: Optional[int] = None,
    ):
        # Validaciones
        self._validate_configuration(train_loader, val_loader)
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.target_iterations = target_iterations
        self.gradient_clip_norm = gradient_clip_norm
        self.save_filters_every = save_filters_every
        
        # [cite: 108] Optimizador Adam con LR=0.0002
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # Historial de entrenamiento (para Figuras 5, 6, 9)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': [],
        }
        
        # Estado del entrenamiento
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.total_iterations = 0
        
        # Crear directorios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'filters').mkdir(exist_ok=True)
        
        # Calcular número de épocas para ~5000 iteraciones [cite: 106]
        self.num_epochs = self._calculate_epochs()
        
        log.info(
            f"HarmonizationTrainer inicializado:\n"
            f"  Optimizador: Adam (LR={self.learning_rate})\n"
            f"  Épocas calculadas: {self.num_epochs} "
            f"({self.target_iterations} iteraciones objetivo)\n"
            f"  Iteraciones por época: {len(self.train_loader)}\n"
            f"  Gradient clipping: {self.gradient_clip_norm}\n"
            f"  Output dir: {self.output_dir}"
        )
    
    def _validate_configuration(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        """Valida la configuración del Trainer."""
        # Validar que train_loader tiene shuffle=True [cite: 108]
        if hasattr(train_loader, 'sampler'):
            if hasattr(train_loader.sampler, 'shuffle'):
                if not train_loader.sampler.shuffle:
                    log.warning(
                        "train_loader no tiene shuffle=True. "
                        "El paper requiere re-shuffle cada época [cite: 108]."
                    )
        
        # Validar tamaños mínimos
        if len(train_loader) == 0:
            raise InvalidConfigurationException(
                "train_loader está vacío. Verifique el dataset."
            )
        
        if len(val_loader) == 0:
            raise InvalidConfigurationException(
                "val_loader está vacío. Verifique el dataset."
            )
    
    def _calculate_epochs(self) -> int:
        """
        Calcula número de épocas para ~5000 iteraciones [cite: 106].
        
        Returns:
            Número de épocas calculadas.
        """
        iterations_per_epoch = len(self.train_loader)
        epochs = max(1, self.target_iterations // iterations_per_epoch)
        
        actual_iterations = epochs * iterations_per_epoch
        
        log.info(
            f"Épocas calculadas: {epochs} épocas x {iterations_per_epoch} "
            f"iteraciones/época = {actual_iterations} iteraciones totales "
            f"(objetivo: {self.target_iterations})"
        )
        
        return epochs
    
    # ========================================================================
    # ENTRENAMIENTO
    # ========================================================================
    
    def _train_epoch(self) -> float:
        """
        Ejecuta una época de entrenamiento.
        
        Returns:
            Pérdida promedio de la época.
        """
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Época {self.current_epoch}/{self.num_epochs} [Train]",
            unit="batch"
        )
        
        for batch_idx, (p1_batch, s2_batch) in enumerate(pbar):
            p1_batch = p1_batch.to(self.device)
            s2_batch = s2_batch.to(self.device)
            
            # Forward pass (EXPLÍCITAMENTE en modo training)
            # Output: [B, 4, 32, 32] @ 10m
            prediction = self.model(p1_batch, training=True)
            
            # NOTA: Ya NO hay crop porque InputModule produce 32x32 correctamente
            # con kernel=21, stride=5, padding=8
            
            # Calcular pérdida [cite: 107, 109]
            loss = self.loss_fn(prediction, s2_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (opcional, para estabilidad)
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_norm
                )
            
            # Optimization step
            self.optimizer.step()
            
            # Logging
            running_loss += loss.item()
            self.total_iterations += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{running_loss / (batch_idx + 1):.6f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss
    
    def _validate_epoch(self) -> float:
        """
        Ejecuta una época de validación.
        
        Returns:
            Pérdida promedio de validación.
        """
        self.model.eval()
        running_loss = 0.0
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Época {self.current_epoch}/{self.num_epochs} [Val]",
            unit="batch"
        )
        
        with torch.no_grad():
            for p1_batch, s2_batch in pbar:
                p1_batch = p1_batch.to(self.device)
                s2_batch = s2_batch.to(self.device)
                
                # Forward pass en modo training (para evaluar modelo completo)
                prediction = self.model(p1_batch, training=True)
                
                # Calcular pérdida
                loss = self.loss_fn(prediction, s2_batch)
                
                running_loss += loss.item()
                
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        epoch_loss = running_loss / len(self.val_loader)
        return epoch_loss
    
    # ========================================================================
    # CHECKPOINTS Y LOGGING
    # ========================================================================
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """
        Guarda checkpoint del modelo.
        
        Args:
            is_best: Si True, también guarda como best_model.pth.
            is_final: Si True, guarda como final_model.pth.
        """
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'total_iterations': self.total_iterations,
        }
        
        # Guardar último checkpoint
        last_path = self.output_dir / 'checkpoints' / 'last_checkpoint.pth'
        torch.save(state, last_path)
        log.debug(f"Checkpoint guardado: {last_path}")
        
        # Guardar mejor modelo
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best_model.pth'
            torch.save(state, best_path)
            log.info(
                f"OK Nuevo mejor modelo guardado: {best_path} "
                f"(val_loss={self.best_val_loss:.6f})"
            )
        
        # Guardar modelo final
        if is_final:
            final_path = self.output_dir / 'checkpoints' / 'final_model.pth'
            torch.save(state, final_path)
            log.info(f"Modelo final guardado: {final_path}")
    
    def save_learned_filters(self):
        """
        Guarda filtros MTF aprendidos del InputModule (Figura 10) [cite: Fig 10].
        """
        try:
            filters = self.model.get_learned_mtf_filters()  # [4, 1, 21, 21]
            
            filename = (
                self.output_dir / 'filters' / 
                f'learned_filters_epoch_{self.current_epoch}.pt'
            )
            torch.save(filters, filename)
            
            log.info(f"Filtros aprendidos guardados: {filename}")
            
        except AttributeError:
            log.warning(
                "Modelo no tiene método get_learned_mtf_filters(). "
                "Saltando guardado de filtros."
            )
    
    def save_training_history(self):
        """
        Guarda historial de entrenamiento para graficar (Figuras 5, 6, 9).
        """
        history_path = self.output_dir / 'logs' / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        log.info(f"Historial de entrenamiento guardado: {history_path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Carga checkpoint para resumir entrenamiento.
        
        Args:
            checkpoint_path: Ruta al archivo .pth.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
        
        log.info(f"Cargando checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        self.total_iterations = checkpoint.get('total_iterations', 0)
        
        log.info(
            f"Checkpoint cargado: época {self.current_epoch}, "
            f"best_val_loss={self.best_val_loss:.6f}"
        )
    
    # ========================================================================
    # BUCLE PRINCIPAL
    # ========================================================================
    
    def train(
        self,
        resume_from: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """
        Bucle de entrenamiento principal.
        
        Args:
            resume_from: Path a checkpoint para resumir entrenamiento.
            
        Returns:
            Diccionario con historial de entrenamiento.
            
        Raises:
            TrainerException: Si ocurre un error durante entrenamiento.
        """
        # Resumir desde checkpoint si se especifica
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        log.info(
            f"\n{'='*70}\n"
            f"INICIANDO ENTRENAMIENTO\n"
            f"{'='*70}\n"
            f"Épocas: {self.num_epochs}\n"
            f"Iteraciones objetivo: ~{self.target_iterations}\n"
            f"Device: {self.device}\n"
            f"{'='*70}"
        )
        
        try:
            for epoch in range(self.current_epoch + 1, self.num_epochs + 1):
                self.current_epoch = epoch
                
                log.info(f"\n{'-'*70}")
                log.info(f"ÉPOCA {epoch}/{self.num_epochs}")
                log.info(f"{'-'*70}")
                
                # Entrenamiento
                train_loss = self._train_epoch()
                log.info(f"  Train Loss: {train_loss:.6f}")
                
                # Validación
                val_loss = self._validate_epoch()
                log.info(f"  Val Loss:   {val_loss:.6f}")
                
                # Actualizar historial
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rates'].append(
                    self.optimizer.param_groups[0]['lr']
                )
                self.history['epochs'].append(epoch)
                
                # Verificar si es mejor modelo
                is_best = val_loss < self.best_val_loss
                if is_best:
                    improvement = self.best_val_loss - val_loss
                    self.best_val_loss = val_loss
                    log.info(
                        f"  OK Mejor validación! Mejora: {improvement:.6f}"
                    )
                
                # Guardar checkpoint
                self.save_checkpoint(is_best=is_best)
                
                # Guardar filtros (periódicamente o al final)
                if self.save_filters_every is not None:
                    if epoch % self.save_filters_every == 0:
                        self.save_learned_filters()
                elif epoch == self.num_epochs:  # Solo al final
                    self.save_learned_filters()
                
                # Guardar historial
                self.save_training_history()
            
            # Entrenamiento completado
            log.info(
                f"\n{'='*70}\n"
                f"ENTRENAMIENTO COMPLETADO\n"
                f"{'='*70}\n"
                f"Épocas: {self.num_epochs}\n"
                f"Iteraciones totales: {self.total_iterations}\n"
                f"Mejor val_loss: {self.best_val_loss:.6f}\n"
                f"{'='*70}"
            )
            
            # Guardar modelo final
            self.save_checkpoint(is_final=True)
            
            return self.history
        
        except KeyboardInterrupt:
            log.warning(
                "\n Entrenamiento interrumpido por el usuario (Ctrl+C). "
                "Guardando checkpoint..."
            )
            self.save_checkpoint()
            self.save_training_history()
            log.info("Checkpoint guardado. El entrenamiento puede resumirse.")
            
            return self.history
        
        except Exception as e:
            log.error(
                f"X Error fatal durante entrenamiento: {e}",
                exc_info=True
            )
            # Intentar guardar checkpoint de emergencia
            try:
                emergency_path = (
                    self.output_dir / 'checkpoints' / 'emergency_checkpoint.pth'
                )
                state = {
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(state, emergency_path)
                log.info(f"Checkpoint de emergencia guardado: {emergency_path}")
            except:
                log.error("No se pudo guardar checkpoint de emergencia.")
            
            raise TrainerException(f"Entrenamiento falló: {e}") from e


# ============================================================================
# UTILIDADES
# ============================================================================

def plot_training_curves(history_path: Path, output_path: Optional[Path] = None):
    """
    Genera gráfico de curvas de entrenamiento (Figuras 5, 6, 9).
    
    Args:
        history_path: Ruta al training_history.json.
        output_path: Ruta para guardar gráfico. Si None, muestra interactivo.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("matplotlib no disponible. Instalar: pip install matplotlib")
        return
    
    # Cargar historial
    with open(history_path) as f:
        history = json.load(f)
    
    epochs = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    # Crear gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    # Marcar mejor validación
    best_epoch = np.argmin(val_loss) + 1
    best_val = np.min(val_loss)
    plt.plot(best_epoch, best_val, 'r*', markersize=15, label=f'Best ({best_epoch})')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        log.info(f"Gráfico guardado: {output_path}")
    else:
        plt.show()


# ============================================================================
# TESTS Y DEMOSTRACIÓN
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import numpy as np
    import rasterio
    from torch.utils.data import TensorDataset
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*70)
    print("TESTS: HarmonizationTrainer")
    print("="*70)
    
    # Test con datos sintéticos
    print("\n[Test 1/2] Entrenamiento con datos sintéticos")
    
    try:
        # Crear datos ficticios
        n_samples = 100
        p1_data = torch.rand(n_samples, 4, 160, 160)
        s2_data = torch.rand(n_samples, 4, 32, 32)
        
        dataset = TensorDataset(p1_data, s2_data)
        
        # Split 90/10
        train_size = int(0.9 * n_samples)
        val_size = n_samples - train_size
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=10, shuffle=False)
        
        # Importar componentes (asumiendo están disponibles)
        from architecture import BCNet
        from losses import RelativeErrorLoss
        
        # Crear modelo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BCNet()
        loss_fn = RelativeErrorLoss()
        
        # Crear trainer
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = HarmonizationTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                output_dir=Path(tmpdir) / 'test_output',
                target_iterations=100,  # Pocas iteraciones para test
                gradient_clip_norm=1.0,
            )
            
            print(f"OK Trainer inicializado")
            print(f"  Épocas calculadas: {trainer.num_epochs}")
            
            # Entrenar por 2 épocas
            history = trainer.train()
            
            print(f"OK Entrenamiento completado")
            print(f"  Train losses: {history['train_loss']}")
            print(f"  Val losses: {history['val_loss']}")
            
            # Verificar que se guardaron archivos
            assert (Path(tmpdir) / 'test_output' / 'checkpoints' / 'last_checkpoint.pth').exists()
            assert (Path(tmpdir) / 'test_output' / 'logs' / 'training_history.json').exists()
            
            print(f"OK Checkpoints y logs guardados correctamente")
        
    except Exception as e:
        print(f"X FALLO: {e}")
        raise
    
    print("\n[Test 2/2] Resumption desde checkpoint")
    
    try:
        # Simular interrupción y resumption
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test_resume'
            
            # Primera sesión de entrenamiento
            trainer1 = HarmonizationTrainer(
                model=BCNet(),
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=RelativeErrorLoss(),
                device=device,
                output_dir=output_dir,
                target_iterations=50,
            )
            
            # Entrenar 1 época y "interrumpir"
            trainer1.current_epoch = 0
            trainer1._train_epoch()
            trainer1.save_checkpoint()
            
            print(f"OK Primera sesión: 1 época completada")
            
            # Segunda sesión: resumir
            trainer2 = HarmonizationTrainer(
                model=BCNet(),
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=RelativeErrorLoss(),
                device=device,
                output_dir=output_dir,
                target_iterations=50,
            )
            
            checkpoint_path = output_dir / 'checkpoints' / 'last_checkpoint.pth'
            trainer2.load_checkpoint(checkpoint_path)
            
            assert trainer2.current_epoch == 1, "Epoch no restaurada"
            
            print(f"OK Segunda sesión: checkpoint cargado correctamente")
            print(f"  Época restaurada: {trainer2.current_epoch}")
        
    except Exception as e:
        print(f"X FALLO: {e}")
        raise
    
    print("\n" + "="*70)
    print("OK TODOS LOS TESTS PASARON")
    print("="*70)