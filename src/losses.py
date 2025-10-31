# /harmonization_project/src/losses.py

"""
Funciones de pérdida para armonización de imágenes satelitales.

Este módulo implementa las funciones de pérdida descritas en 
Michel & Inglada (2021): "Learning Harmonised Pleiades and 
Sentinel-2 Surface Reflectances".

Funciones incluidas:
- RelativeErrorLoss: Error relativo promedio (Ecuación 1)
- (Futuro) VarianceAwareLoss: Para predicción con varianza (Ecuación 2)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

# Configurar logger
log = logging.getLogger(__name__)


# ============================================================================
# EXCEPCIONES PERSONALIZADAS
# ============================================================================

class LossException(Exception):
    """Excepción base para errores en funciones de pérdida."""
    pass


class InvalidTensorException(LossException):
    """Excepción para tensores inválidos (NaN, Inf, shapes incorrectas)."""
    pass


# ============================================================================
# RELATIVE ERROR LOSS (Ecuación 1)
# ============================================================================

class RelativeErrorLoss(nn.Module):
    """
    Error Relativo Promedio (Ecuación 1 del paper) [cite: 107, 109].
    
    Implementa la función de pérdida:
        L(t, r) = mean( |t - r| / (ε + r) )
    
    Donde:
        - t: Target (reflectancias Sentinel-2 de referencia)
        - r: Predicción del modelo
        - ε: Constante pequeña para estabilidad numérica
    
    Características:
    ----------------
    1. **Pérdida relativa:** Normaliza el error por la magnitud de la predicción.
       Esto hace que errores en regiones oscuras (r≈0) se penalicen más que
       en regiones brillantes (r≈1).
       
    2. **Robustez:** Maneja valores negativos de predicción (posibles con
       Tanh + skip connection) mediante clamping a [0, 1].
       
    3. **Validación:** Verifica NaN, Inf, y rangos de reflectancias.
    
    Args:
        epsilon: Constante para evitar división por cero (default: 1e-6).
        clamp_prediction: Si True, clampea predicción a [0, 1] (default: True).
                          Esto garantiza que r sea una reflectancia válida.
        validate_range: Si True, valida que target esté en [0, 1] (default: True).
        log_statistics: Si True, loggea estadísticas de pérdida (default: False).
        
    Raises:
        InvalidTensorException: Si inputs contienen NaN, Inf, o shapes incorrectas.
        
    Example:
        >>> loss_fn = RelativeErrorLoss(epsilon=1e-6)
        >>> prediction = model(input_batch)  # [B, 4, 32, 32]
        >>> target = target_batch            # [B, 4, 32, 32]
        >>> loss = loss_fn(prediction, target)
        >>> loss.backward()
    """
    
    def __init__(
        self,
        epsilon: float = 1e-6,
        clamp_prediction: bool = True,
        validate_range: bool = True,
        log_statistics: bool = False,
    ):
        super().__init__()
        
        if epsilon <= 0:
            raise ValueError(f"Epsilon debe ser > 0, pero es {epsilon}")
        
        self.epsilon = epsilon
        self.clamp_prediction = clamp_prediction
        self.validate_range = validate_range
        self.log_statistics = log_statistics
        
        # Contadores para logging
        self._call_count = 0
        self._total_loss = 0.0
        
        log.info(
            f"RelativeErrorLoss inicializada: epsilon={self.epsilon}, "
            f"clamp_prediction={self.clamp_prediction}, "
            f"validate_range={self.validate_range}"
        )
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula el error relativo promedio.
        
        Args:
            prediction: Reflectancias predichas por el modelo [B, C, H, W].
            target: Reflectancias de referencia S2 [B, C, H, W].
            
        Returns:
            Tensor escalar con el valor de pérdida.
            
        Raises:
            InvalidTensorException: Si inputs son inválidos.
        """
        # Validación exhaustiva
        self._validate_inputs(prediction, target)
        
        # Clampear predicción a rango válido de reflectancias
        if self.clamp_prediction:
            prediction = prediction.clamp(min=0.0, max=1.0)
        
        # [cite: 109] Numerador: |t - r|
        abs_diff = torch.abs(target - prediction)
        
        # [cite: 109] Denominador: ε + r
        # CRÍTICO: Después de clamp, prediction ≥ 0, por lo tanto
        # denominator ≥ epsilon > 0 siempre (no hay división por 0)
        denominator = prediction + self.epsilon
        
        # Pérdida por píxel
        loss_per_pixel = abs_diff / denominator
        
        # [cite: 107] Promedio sobre todos los píxeles
        loss = loss_per_pixel.mean()
        
        # Logging de estadísticas (opcional)
        if self.log_statistics:
            self._log_statistics(loss, prediction, target, abs_diff)
        
        return loss
    
    def _validate_inputs(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """
        Valida que los tensores de entrada sean válidos.
        
        Verificaciones:
        1. Shapes coinciden
        2. No contienen NaN
        3. No contienen Inf
        4. Target está en rango [0, 1] (si validate_range=True)
        """
        # Validación 1: Shapes
        if prediction.shape != target.shape:
            raise InvalidTensorException(
                f"Shape mismatch: prediction {prediction.shape} vs "
                f"target {target.shape}"
            )
        
        # Validación 2: NaN
        if torch.isnan(prediction).any():
            raise InvalidTensorException(
                "Prediction contiene valores NaN. Verifique el modelo."
            )
        
        if torch.isnan(target).any():
            raise InvalidTensorException(
                "Target contiene valores NaN. Verifique los datos."
            )
        
        # Validación 3: Inf
        if torch.isinf(prediction).any():
            raise InvalidTensorException(
                "Prediction contiene valores Inf. Verifique el modelo."
            )
        
        if torch.isinf(target).any():
            raise InvalidTensorException(
                "Target contiene valores Inf. Verifique los datos."
            )
        
        # Validación 4: Rango de reflectancias
        if self.validate_range:
            if target.min() < 0.0 or target.max() > 1.0:
                log.warning(
                    f"Target fuera de rango [0, 1]: "
                    f"min={target.min():.4f}, max={target.max():.4f}. "
                    f"Verifique el escalado de datos."
                )
            
            # No validamos prediction aquí porque será clampeada
    
    def _log_statistics(
        self,
        loss: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        abs_diff: torch.Tensor,
    ) -> None:
        """Loggea estadísticas de la pérdida para debugging."""
        self._call_count += 1
        self._total_loss += loss.item()
        
        if self._call_count % 100 == 0:  # Log cada 100 llamadas
            avg_loss = self._total_loss / self._call_count
            
            log.info(
                f"Loss Statistics (call #{self._call_count}):\n"
                f"  Current loss: {loss.item():.6f}\n"
                f"  Average loss: {avg_loss:.6f}\n"
                f"  Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]\n"
                f"  Target range: [{target.min():.4f}, {target.max():.4f}]\n"
                f"  Mean abs error: {abs_diff.mean():.6f}\n"
                f"  Max abs error: {abs_diff.max():.6f}"
            )
    
    def reset_statistics(self) -> None:
        """Reinicia contadores de estadísticas."""
        self._call_count = 0
        self._total_loss = 0.0
        log.debug("Loss statistics reset.")