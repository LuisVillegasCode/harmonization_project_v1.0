# /harmonization_project/src/architecture.py

"""
Arquitecturas de Redes Neuronales para Armonización de Imágenes Satelitales.

Este módulo implementa rigurosamente las arquitecturas descritas en 
Michel & Inglada (2021): "Learning Harmonised Pleiades and Sentinel-2 
Surface Reflectances".

Arquitecturas incluidas:
- CalibNet: MLP pixel-wise (Figura 2a)
- InputModule: Módulo convolucional de entrada para BCNet (Figura 2c)
- BCNet: Modelo completo para entrenamiento/inferencia (Figura 2c)

Adaptaciones para PeruSat-1:
- Soporte para stride=5 (PeruSat @ 2m) o stride=3 (PeruSat @ 2.8m nativo)
- Configuración flexible mediante parámetros
"""

import logging
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configurar logger
log = logging.getLogger(__name__)


# ============================================================================
# EXCEPCIONES PERSONALIZADAS
# ============================================================================

class ArchitectureException(Exception):
    """Excepción base para errores de arquitectura."""
    pass


class InvalidInputShapeException(ArchitectureException):
    """Excepción para dimensiones de entrada inválidas."""
    pass


class InvalidModeException(ArchitectureException):
    """Excepción para modos de operación inválidos."""
    pass


# ============================================================================
# CALIBNET - MLP PIXEL-WISE (Figura 2a)
# ============================================================================

class CalibNet(nn.Module):
    """
    Implementación del Perceptrón Multicapa (MLP) 'CalibNet' (Figura 2a).
    
    Esta red opera a nivel de píxel (pixel-wise) y realiza el mapeo 
    radiométrico entre sensores. Es el componente downstream de BCNet
    y se usa directamente en inferencia.
    
    Arquitectura [cite: 87]:
    -----------------------
    Input [4] 
      → BatchNorm1d 
      → Linear(4→320) + LeakyReLU 
      → Linear(320→320) + LeakyReLU
      → Linear(320→4) + Tanh 
      → Skip Connection (+ Input)
      → Output [4]
    
    Args:
        n_features: Número de bandas espectrales (default: 4).
        n_hidden: Número de unidades en capas ocultas (default: 320).
        
    Raises:
        InvalidInputShapeException: Si input no tiene shape [N, 4].
    """
    
    def __init__(self, n_features: int = 4, n_hidden: int = 320):
        super().__init__()
        
        self.n_features = n_features
        self.n_hidden = n_hidden
        
        # [cite: 87] 1. Batch Normalization (1D para vectores de píxeles)
        self.batch_norm = nn.BatchNorm1d(n_features)
        
        # [cite: 87] 2. Dos capas ocultas con LeakyReLU
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        
        # [cite: 87] 3. Capa de salida
        self.fc3 = nn.Linear(n_hidden, n_features)
        
        # Activaciones
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        
        log.info(
            f"CalibNet inicializado: {n_features} features, "
            f"{n_hidden} hidden units."
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass pixel-wise.
        
        Args:
            x: Tensor [N_pixels, 4_bandas] con reflectancias normalizadas.
            
        Returns:
            Tensor [N_pixels, 4_bandas] con reflectancias armonizadas.
            
        Raises:
            InvalidInputShapeException: Si dimensiones son incorrectas.
        """
        self._validate_input(x)
        
        # [cite: 87] Guardar input para skip connection
        identity = x
        
        # Forward a través de las capas
        x = self.batch_norm(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))  # OK LeakyReLU agregada
        x = self.tanh(self.fc3(x))
        
        # [cite: 87, 97] Skip connection (suma)
        output = x + identity
        
        return output
    
    def _validate_input(self, x: torch.Tensor) -> None:
        """Valida dimensiones del input."""
        if x.dim() != 2:
            raise InvalidInputShapeException(
                f"CalibNet requiere input 2D [N, {self.n_features}], "
                f"pero recibió {x.dim()}D: {x.shape}"
            )
        
        if x.size(1) != self.n_features:
            raise InvalidInputShapeException(
                f"CalibNet requiere {self.n_features} features, "
                f"pero input tiene {x.size(1)}: {x.shape}"
            )


# ============================================================================
# INPUT MODULE - CONVOLUCIÓN ESPACIAL (Figura 2c)
# ============================================================================

class InputModule(nn.Module):
    """
    Módulo convolucional de entrada para BCNet (Figura 2c).
    
    Este módulo aprende y compensa discrepancias espaciales entre sensores:
    - Diferencias en MTF (Modulation Transfer Function)
    - Desregistro espacial
    - Diferencias en frecuencias espaciales
    
    CRÍTICO: Este módulo solo se usa en ENTRENAMIENTO. Durante inferencia,
    se omite para preservar la resolución espacial original [cite: 191, 991].
    
    Características [cite: 99-101]:
    - Filtro dedicado por banda (groups=in_channels)
    - Inicialización con MTF Gaussiano de Sentinel-2
    - Normalización de filtros (suma=1) via Softmax
    
    Args:
        in_channels: Número de bandas de entrada (default: 4).
        out_channels: Número de bandas de salida (default: 4).
        kernel_size: Tamaño del kernel [cite: Fig 2c: 21] (default: 21).
        stride: Stride de la convolución (default: 5 para factor 2m→10m).
        padding: Padding para preservar dimensiones (default: 8 para output 32x32).
        mtf_sigma: Diccionario de sigmas por banda para inicialización MTF.
                   Si None, usa valores por defecto.
                   
    Example:
        # Para PeruSat @ 2m (como Pleiades)
        >>> module = InputModule(kernel_size=21, stride=5, padding=8)
        
        # Para PeruSat @ 2.8m (resolución nativa)
        >>> module = InputModule(kernel_size=21, stride=3, padding=0)
    """
    
    # Valores MTF aproximados de Sentinel-2 (B2, B3, B4, B8)
    # NOTA: Estos son valores de ejemplo basados en literatura
    # Para producción, usar valores oficiales de ESA
    DEFAULT_MTF_SIGMA = {
        'B2': 2.5,   # Blue (490nm)
        'B3': 2.5,   # Green (560nm)
        'B4': 2.5,   # Red (665nm)
        'B8': 2.2,   # NIR (842nm) - típicamente más nítido
    }
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        kernel_size: int = 21,  # OK CORRECTO según paper [cite: Fig 2c]
        stride: int = 5,
        padding: int = 8,       # OK Para output 32x32 desde 160x160
        mtf_sigma: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        if in_channels != 4 or out_channels != 4:
            raise ValueError(
                "InputModule debe tener 4 canales de entrada/salida "
                "(bandas B2, B3, B4, B8)."
            )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # [cite: 99] Convolución agrupada: un filtro dedicado por banda
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # CRÍTICO: filtros independientes
            bias=False           # Sin bias (filtros normalizados)
        )
        
        # Usar sigmas por defecto o los proporcionados
        self.mtf_sigma = mtf_sigma or self.DEFAULT_MTF_SIGMA
        
        # [cite: 100] Inicializar con MTF Gaussiano
        self._initialize_mtf_filters()
        
        # Calcular dimensiones de output para validación
        self._calculate_output_size()
        
        log.info(
            f"InputModule inicializado: kernel={kernel_size}, "
            f"stride={stride}, padding={padding}, "
            f"output_size={self.output_height}x{self.output_width}"
        )
    
    def _create_gaussian_kernel(
        self, kernel_size: int, sigma: float
    ) -> torch.Tensor:
        """
        Crea un kernel Gaussiano 2D normalizado.
        
        Args:
            kernel_size: Tamaño del kernel (NxN).
            sigma: Desviación estándar del Gaussiano.
            
        Returns:
            Tensor [1, 1, kernel_size, kernel_size] normalizado (suma=1).
        """
        # Crear grid de coordenadas
        coords = torch.arange(kernel_size, dtype=torch.float32)
        x_grid = coords.repeat(kernel_size, 1)
        y_grid = x_grid.t()
        
        # Centro del kernel
        center = (kernel_size - 1) / 2.0
        
        # Fórmula Gaussiana 2D
        variance = sigma ** 2
        gaussian = torch.exp(
            -((x_grid - center) ** 2 + (y_grid - center) ** 2) / (2 * variance)
        )
        
        # Normalizar (suma = 1)
        gaussian = gaussian / gaussian.sum()
        
        return gaussian.view(1, 1, kernel_size, kernel_size)
    
    def _initialize_mtf_filters(self) -> None:
        """
        Inicializa pesos con MTF Gaussiano específico por banda [cite: 100].
        
        Cada banda (B2, B3, B4, B8) tiene características MTF ligeramente
        diferentes, que se modelan con diferentes sigmas Gaussianos.
        """
        log.info("Inicializando filtros con MTF Gaussiano por banda...")
        
        band_names = ['B2', 'B3', 'B4', 'B8']
        
        for i, band_name in enumerate(band_names):
            sigma = self.mtf_sigma[band_name]
            kernel = self._create_gaussian_kernel(self.kernel_size, sigma)
            
            # Asignar kernel a la banda correspondiente
            # Shape: [out_channels, in_channels/groups, H, W]
            # Con groups=4: [4, 1, 21, 21]
            self.conv.weight.data[i] = kernel.squeeze(0)
            
            log.debug(
                f"Banda {band_name}: kernel {self.kernel_size}x{self.kernel_size}, "
                f"sigma={sigma:.2f}, sum={self.conv.weight.data[i].sum():.6f}"
            )
        
        log.info(f"Filtros MTF inicializados: {self.mtf_sigma}")
    
    def _calculate_output_size(self, input_size: int = 160) -> None:
        """
        Calcula dimensiones de output para validación.
        
        Args:
            input_size: Tamaño del input (default: 160 para PeruSat @ 2m).
        """
        self.output_height = (
            input_size + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        self.output_width = self.output_height
        
        log.debug(
            f"Output calculado: {input_size}x{input_size} → "
            f"{self.output_height}x{self.output_width} "
            f"(kernel={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding})"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass con normalización de filtros [cite: 101].
        
        CRÍTICO: Los filtros se normalizan (Softmax) ANTES de cada convolución
        para garantizar que cada filtro suma exactamente 1.
        
        Args:
            x: Tensor [B, 4, H, W] (típicamente [B, 4, 160, 160] @ 2m).
            
        Returns:
            Tensor [B, 4, H', W'] (típicamente [B, 4, 32, 32] @ 10m).
        """
        # [cite: 101] Normalización de filtros via Softmax
        # Shape: [4, 1, 21, 21]
        num_filters = self.conv.weight.size(0)
        kernel_elements = self.conv.weight.size(2) * self.conv.weight.size(3)
        
        # Aplanar cada filtro: [4, 1, 21, 21] → [4, 441]
        weights_flat = self.conv.weight.view(num_filters, -1)
        
        # Softmax sobre elementos del kernel (dim=1)
        # Garantiza que sum(kernel[i]) = 1 para cada i
        normalized_flat = F.softmax(weights_flat, dim=1)
        
        # Reshape de vuelta: [4, 441] → [4, 1, 21, 21]
        normalized_weights = normalized_flat.view_as(self.conv.weight)
        
        # Aplicar convolución con pesos normalizados
        output = F.conv2d(
            x,
            normalized_weights,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=self.conv.groups
        )
        
        return output
    
    def get_learned_filters(self) -> torch.Tensor:
        """
        Retorna los filtros aprendidos (para visualización, Figura 10).
        
        Returns:
            Tensor [4, 1, kernel_size, kernel_size] con filtros aprendidos.
        """
        return self.conv.weight.data.clone()


# ============================================================================
# BCNET - MODELO COMPLETO (Figura 2c)
# ============================================================================

class BCNet(nn.Module):
    """
    Modelo BCNet completo para entrenamiento e inferencia (Figura 2c).
    
    BCNet combina un módulo convolucional (InputModule) que aprende
    discrepancias espaciales, con un MLP pixel-wise (CalibNet) que
    realiza el mapeo radiométrico.
    
    Modos de Operación:
    -------------------
    1. **Entrenamiento (training=True)**:
       - Usa InputModule + CalibNet
       - Input: [B, 4, 160, 160] @ 2m (alta resolución)
       - Output: [B, 4, 32, 32] @ 10m (baja resolución)
       - Aprende compensar MTF y desregistro
       
    2. **Inferencia (training=False)** [cite: 191, 991]:
       - OMITE InputModule (preserva resolución espacial)
       - Usa solo CalibNet (pixel-wise mapping)
       - Input: [B, 4, H, W] @ 2m (cualquier tamaño)
       - Output: [B, 4, H, W] @ 2m (mismo tamaño)
       - Aplica solo corrección radiométrica
    
    Args:
        input_module_config: Diccionario de configuración para InputModule.
                             Si None, usa configuración por defecto.
                             
    Example:
        # Configuración para PeruSat @ 2m (como Pleiades)
        >>> model = BCNet(input_module_config={
        ...     'kernel_size': 21,
        ...     'stride': 5,
        ...     'padding': 8
        ... })
        
        # Entrenamiento
        >>> model.train()
        >>> output = model(input_160x160, training=True)  # [B,4,32,32]
        
        # Inferencia
        >>> model.eval()
        >>> harmonized = model(input_fullres, training=False)  # [B,4,H,W]
    """
    
    def __init__(
        self,
        input_module_config: Optional[Dict] = None,
    ):
        super().__init__()
        
        # Configuración por defecto para PeruSat @ 2m
        default_config = {
            'kernel_size': 21,
            'stride': 5,
            'padding': 8,
        }
        
        config = input_module_config or default_config
        
        # [cite: 99] Módulo convolucional de entrada
        self.input_module = InputModule(**config)
        
        # [cite: 95] MLP downstream
        self.downstream_calibnet = CalibNet()
        
        # Guardar configuración para validación
        self.expected_training_size = 160  # Para PeruSat @ 2m
        self.expected_output_size = self.input_module.output_height
        
        log.info(
            f"BCNet inicializado. Configuración InputModule: {config}. "
            f"Expected training input: {self.expected_training_size}x"
            f"{self.expected_training_size}, "
            f"output: {self.expected_output_size}x{self.expected_output_size}."
        )
    
    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """
        Forward pass con control de modo entrenamiento/inferencia.
        
        Args:
            x: Tensor de entrada [B, 4, H, W].
            training: Si True, usa InputModule+CalibNet.
                     Si False, usa solo CalibNet (inferencia).
                     
        Returns:
            Tensor procesado:
                - Si training=True: [B, 4, 32, 32] @ 10m
                - Si training=False: [B, 4, H, W] @ 2m (mismo tamaño)
                
        Raises:
            InvalidInputShapeException: Si dimensiones son incorrectas.
            InvalidModeException: Si modo no es válido.
        """
        if training:
            return self._forward_training(x)
        else:
            return self._forward_inference(x)
    
    def _forward_training(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de entrenamiento [cite: 94-97].
        
        Args:
            x: [B, 4, 160, 160] @ 2m (alta resolución).
            
        Returns:
            [B, 4, 32, 32] @ 10m (baja resolución).
        """
        self._validate_training_input(x)
        
        # 1. Módulo convolucional de entrada
        # [B, 4, 160, 160] → [B, 4, 32, 32]
        x_spatial = self.input_module(x)
        
        # 2. Reformatear para MLP (pixel-wise)
        B, C, H, W = x_spatial.shape
        # [B, 4, 32, 32] → [B, 32, 32, 4] → [B*32*32, 4]
        x_flat = x_spatial.permute(0, 2, 3, 1).reshape(B * H * W, C)
        
        # 3. MLP pixel-wise
        # [B*32*32, 4] → [B*32*32, 4]
        x_mapped = self.downstream_calibnet(x_flat)
        
        # 4. Reformatear de vuelta a espacial
        # [B*32*32, 4] → [B, 32, 32, 4] → [B, 4, 32, 32]
        x_output = x_mapped.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return x_output
    
    def _forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de inferencia [cite: 191, 991].
        
        CRÍTICO: Omite InputModule para preservar detalles espaciales.
        Solo aplica CalibNet pixel-wise.
        
        Args:
            x: [B, 4, H, W] @ 2m (cualquier tamaño).
            
        Returns:
            [B, 4, H, W] @ 2m (mismo tamaño, armonizado).
        """
        self._validate_inference_input(x)
        
        B, C, H, W = x.shape
        
        # Aplicar MLP a cada píxel (preserva dimensiones espaciales)
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x_mapped = self.downstream_calibnet(x_flat)
        x_output = x_mapped.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        log.debug(
            f"Inferencia aplicada: {B} imágenes de {H}x{W} píxeles procesadas."
        )
        
        return x_output
    
    def _validate_training_input(self, x: torch.Tensor) -> None:
        """Valida dimensiones para modo entrenamiento."""
        if x.dim() != 4:
            raise InvalidInputShapeException(
                f"Input de entrenamiento debe ser 4D [B,C,H,W], "
                f"pero tiene {x.dim()}D: {x.shape}"
            )
        
        B, C, H, W = x.shape
        
        if C != 4:
            raise InvalidInputShapeException(
                f"Input debe tener 4 bandas, pero tiene {C}: {x.shape}"
            )
        
        if H != self.expected_training_size or W != self.expected_training_size:
            raise InvalidInputShapeException(
                f"Input de entrenamiento debe ser "
                f"{self.expected_training_size}x{self.expected_training_size}, "
                f"pero es {H}x{W}. Verifique que los patches P1 tengan "
                f"el tamaño correcto."
            )
    
    def _validate_inference_input(self, x: torch.Tensor) -> None:
        """Valida dimensiones para modo inferencia."""
        if x.dim() != 4:
            raise InvalidInputShapeException(
                f"Input debe ser 4D [B,C,H,W], pero tiene {x.dim()}D: {x.shape}"
            )
        
        B, C, H, W = x.shape
        
        if C != 4:
            raise InvalidInputShapeException(
                f"Input debe tener 4 bandas, pero tiene {C}: {x.shape}"
            )
    
    def get_learned_mtf_filters(self) -> torch.Tensor:
        """
        Retorna los filtros MTF aprendidos del InputModule (Figura 10).
        
        Útil para:
        - Visualización de filtros aprendidos
        - Análisis de MTF y desregistro aprendido
        - Validación del entrenamiento
        
        Returns:
            Tensor [4, 1, 21, 21] con filtros aprendidos.
        """
        return self.input_module.get_learned_filters()
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Cuenta parámetros del modelo.
        
        Returns:
            Diccionario con conteo de parámetros por componente.
        """
        input_params = sum(
            p.numel() for p in self.input_module.parameters()
        )
        calib_params = sum(
            p.numel() for p in self.downstream_calibnet.parameters()
        )
        total_params = input_params + calib_params
        
        return {
            'input_module': input_params,
            'calibnet': calib_params,
            'total': total_params,
        }


# ============================================================================
# UTILIDADES
# ============================================================================

def create_bcnet_for_perusat(
    resolution: float,
    variant: str = 'resampled'
) -> BCNet:
    """
    Factory function para crear BCNet configurado para PeruSat-1.
    
    Args:
        resolution: Resolución de PeruSat en metros (2.0 o 2.8).
        variant: Variante de configuración:
                 - 'resampled': PeruSat remuestreado a 2m (stride=5)
                 - 'native': PeruSat a resolución nativa (stride adaptado)
                 
    Returns:
        Instancia de BCNet configurada.
        
    Raises:
        ValueError: Si resolución o variante son inválidos.
        
    Example:
        # PeruSat @ 2m (remuestreado, como Pleiades)
        >>> model = create_bcnet_for_perusat(2.0, variant='resampled')
        
        # PeruSat @ 2.8m (nativo)
        >>> model = create_bcnet_for_perusat(2.8, variant='native')
    """
    if resolution == 2.0 and variant == 'resampled':
        # Configuración idéntica al paper (Pleiades @ 2m)
        config = {
            'kernel_size': 21,
            'stride': 5,
            'padding': 8,
        }
        log.info("Creando BCNet para PeruSat @ 2m (variante resampled).")
        
    elif resolution == 2.8 and variant == 'native':
        # Configuración adaptada para PeruSat nativo
        # Patches de 114x114 @ 2.8m → 32x32 @ 10m con stride=3
        config = {
            'kernel_size': 21,
            'stride': 3,
            'padding': 0,
        }
        log.info("Creando BCNet para PeruSat @ 2.8m (variante native).")
        log.warning(
            "ATENCIÓN: Usando stride=3 (desviación del paper stride=5). "
            "Esta es una adaptación metodológica para PeruSat nativo."
        )
        
    else:
        raise ValueError(
            f"Combinación inválida: resolution={resolution}, variant={variant}. "
            f"Opciones válidas:\n"
            f"  - (2.0, 'resampled'): PeruSat remuestreado a 2m\n"
            f"  - (2.8, 'native'): PeruSat a resolución nativa"
        )
    
    model = BCNet(input_module_config=config)
    
    # Log de configuración
    params = model.count_parameters()
    log.info(
        f"BCNet creado. Parámetros totales: {params['total']:,} "
        f"(InputModule: {params['input_module']:,}, "
        f"CalibNet: {params['calibnet']:,})"
    )
    
    return model
