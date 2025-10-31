# /harmonization_project/src/data_loader.py

"""
Dataset de PyTorch para Armonización de Imágenes Satelitales.

Este módulo implementa rigurosamente la metodología de preparación de datos
descrita en Michel & Inglada (2021): "Learning Harmonised Pleiades and 
Sentinel-2 Surface Reflectances".

Fidelidad Metodológica:
- Patches no superpuestos: 32x32 @ 10m (S2), 160x160 @ 2m (P1)
- Filtrado de nubes según máscara S2
- Descarte de patches con valores no-data
- Shuffle aleatorio después de extracción
- Split 90% entrenamiento / 10% prueba

Mejoras de Ingeniería:
- Lectura por ventanas (memoria eficiente)
- Validación exhaustiva de alineación geográfica
- Manejo robusto de errores
- Logging comprehensivo
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioIOError
import torch
from torch.utils.data import Dataset

# Configurar logger para este módulo
log = logging.getLogger(__name__)


# ============================================================================
# EXCEPCIONES PERSONALIZADAS
# ============================================================================

class DataValidationException(Exception):
    """Excepción para errores en la validación de datos geoespaciales."""
    pass


class GeographicAlignmentException(DataValidationException):
    """Excepción específica para errores de alineación geográfica."""
    pass


class ResolutionFactorException(DataValidationException):
    """Excepción específica para errores en el factor de resolución."""
    pass


# ============================================================================
# DATASET PRINCIPAL
# ============================================================================

class HarmonizationPatchDataset(Dataset):
    """
    Dataset de PyTorch para pares de patches PeruSat-1 / Sentinel-2.
    
    Implementa la metodología de BCNet del paper de Michel & Inglada (2021).
    
    Características Clave:
    ----------------------
    1. **Patches no superpuestos**: Extracción en grid regular sin overlap.
    2. **Filtrado riguroso**: Descarta patches con nubes o no-data.
    3. **Shuffle metodológico**: Mezcla patches después de extracción [cite: 81].
    4. **Lectura eficiente**: Windowed reading para minimizar uso de memoria.
    5. **Validación geoespacial**: Verifica CRS, alineación, y resoluciones.
    
    Args:
        p1_path: Ruta al GeoTIFF de PeruSat-1 (alta resolución, ej. 2m).
        s2_path: Ruta al GeoTIFF de Sentinel-2 L2A (10m, 4 bandas: B2,B3,B4,B8).
        s2_cloud_mask_path: Ruta al GeoTIFF de máscara de nubes de S2.
        p1_nodata_values: Valor(es) que representan no-data en P1.
        s2_nodata_values: Valor(es) que representan no-data en S2.
        s2_cloud_values: Valores en la máscara que indican nube/inválido.
        scale_factor: Factor de escalado de reflectancias (ej. 10000.0).
        expected_resolution_factor: Factor de resolución esperado (ej. 5 para 10m/2m).
                                    Si es None, se calcula automáticamente.
        shuffle_patches: Si True, mezcla patches después de extraerlos [cite: 81].
        validate_range: Si True, valida rangos físicos de reflectancias.
        
    Raises:
        FileNotFoundError: Si algún archivo de entrada no existe.
        DataValidationException: Si la validación geoespacial falla.
        ValueError: Si no se encuentran patches válidos.
    """
    
    def __init__(
        self,
        p1_path: Path,
        s2_path: Path,
        s2_cloud_mask_path: Path,
        p1_nodata_values: Union[float, List[float]],
        s2_nodata_values: Union[float, List[float]],
        s2_cloud_values: List[int],
        scale_factor: float = 10000.0,
        expected_resolution_factor: Optional[int] = None,
        shuffle_patches: bool = True,
        validate_range: bool = True,
    ):
        super().__init__()
        
        # Guardar rutas (necesarias para lectura por ventanas)
        self.p1_path = Path(p1_path)
        self.s2_path = Path(s2_path)
        self.s2_cloud_mask_path = Path(s2_cloud_mask_path)
        
        # Validar existencia de archivos
        self._validate_file_existence()
        
        # Parámetros de procesamiento
        self.scale_factor = scale_factor
        self.shuffle_patches = shuffle_patches
        self.validate_range = validate_range
        
        # Convertir nodata a listas para procesamiento uniforme
        self.p1_nodata = self._to_list(p1_nodata_values)
        self.s2_nodata = self._to_list(s2_nodata_values)
        self.s2_cloud_values = set(s2_cloud_values)  # Set para búsqueda O(1)
        
        # Contenedor de patches (se llena durante load())
        self.patch_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._is_loaded = False
        
        # Validar datos geoespaciales y calcular parámetros
        try:
            self._load_and_validate_metadata(expected_resolution_factor)
        except (RasterioIOError, DataValidationException) as e:
            log.error(f"Error en validación de metadatos: {e}")
            raise
        
        log.info(
            f"Dataset inicializado. Configuración: "
            f"P1 patch={self.p1_patch_size}x{self.p1_patch_size}@{self.p1_resolution}m, "
            f"S2 patch={self.s2_patch_size}x{self.s2_patch_size}@{self.s2_resolution}m, "
            f"factor={self.resolution_factor}."
        )
    
    # ========================================================================
    # MÉTODOS PÚBLICOS
    # ========================================================================
    
    def load(self) -> None:
        """
        Carga y procesa patches de las imágenes.
        
        Este método debe ser llamado explícitamente después de la construcción
        para iniciar la extracción de patches. Esto permite una inicialización
        lazy y facilita el testing.
        
        Raises:
            RuntimeError: Si el dataset ya está cargado.
            DataValidationException: Si hay errores durante la extracción.
        """
        if self._is_loaded:
            log.warning("Dataset ya está cargado. Llamada a load() ignorada.")
            return
        
        log.info("Iniciando carga y extracción de patches...")
        
        try:
            self._extract_and_filter_patches()
            
            if not self.patch_pairs:
                raise ValueError(
                    "No se encontraron patches válidos. Verifique:\n"
                    "  1. Las imágenes están correctamente alineadas\n"
                    "  2. Los valores de no-data son correctos\n"
                    "  3. La máscara de nubes no cubre toda la imagen\n"
                    "  4. Las imágenes tienen al menos 32x32 píxeles @ 10m"
                )
            
            self._is_loaded = True
            log.info(
                f"Carga completada exitosamente. "
                f"{len(self.patch_pairs)} patches válidos disponibles."
            )
            
        except Exception as e:
            log.error(f"Error durante la carga de patches: {e}")
            raise
    
    def get_statistics(self) -> dict:
        """
        Retorna estadísticas descriptivas del dataset.
        
        Returns:
            Diccionario con estadísticas (número de patches, rangos, etc.)
        """
        if not self._is_loaded:
            raise RuntimeError("Dataset no cargado. Llamar .load() primero.")
        
        # Concatenar todos los patches para análisis
        all_p1 = torch.stack([p[0] for p in self.patch_pairs])
        all_s2 = torch.stack([p[1] for p in self.patch_pairs])
        
        return {
            'num_patches': len(self.patch_pairs),
            'p1_shape': tuple(all_p1.shape),
            's2_shape': tuple(all_s2.shape),
            'p1_mean': all_p1.mean(dim=(0, 2, 3)).tolist(),
            'p1_std': all_p1.std(dim=(0, 2, 3)).tolist(),
            's2_mean': all_s2.mean(dim=(0, 2, 3)).tolist(),
            's2_std': all_s2.std(dim=(0, 2, 3)).tolist(),
            'p1_min': all_p1.min().item(),
            'p1_max': all_p1.max().item(),
            's2_min': all_s2.min().item(),
            's2_max': all_s2.max().item(),
        }
    
    # ========================================================================
    # MÉTODOS DE PYTORCH DATASET
    # ========================================================================
    
    def __len__(self) -> int:
        """Retorna el número total de patches válidos."""
        if not self._is_loaded:
            raise RuntimeError(
                "Dataset no cargado. Llamar .load() antes de usar el dataset."
            )
        return len(self.patch_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna el par de patches en el índice especificado.
        
        Args:
            idx: Índice del patch (0 <= idx < len(self)).
            
        Returns:
            Tupla (p1_patch, s2_patch):
                - p1_patch: Tensor [4, 160, 160] (para BCNet)
                - s2_patch: Tensor [4, 32, 32]
        """
        if not self._is_loaded:
            raise RuntimeError(
                "Dataset no cargado. Llamar .load() antes de acceder a patches."
            )
        return self.patch_pairs[idx]
    
    # ========================================================================
    # MÉTODOS PRIVADOS - VALIDACIÓN
    # ========================================================================
    
    def _validate_file_existence(self) -> None:
        """Valida que todos los archivos de entrada existan."""
        for path, name in [
            (self.p1_path, "PeruSat-1"),
            (self.s2_path, "Sentinel-2"),
            (self.s2_cloud_mask_path, "Máscara de nubes"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Archivo {name} no encontrado: {path}"
                )
    
    def _load_and_validate_metadata(
        self, expected_resolution_factor: Optional[int]
    ) -> None:
        """
        Carga metadatos de los archivos y ejecuta todas las validaciones
        geoespaciales requeridas por la metodología del paper.
        
        Validaciones:
        1. Consistencia de CRS entre las 3 imágenes
        2. Geometría idéntica entre S2 y su máscara
        3. Número correcto de bandas (4 para P1/S2, 1 para máscara)
        4. Factor de resolución coherente
        5. Alineación geográfica (orígenes coincidentes)
        6. Ausencia de rotación/shear
        """
        log.info("Validando metadatos geoespaciales...")
        
        with rasterio.open(self.p1_path) as p1_src, \
             rasterio.open(self.s2_path) as s2_src, \
             rasterio.open(self.s2_cloud_mask_path) as mask_src:
            
            # --- Validación 1: Consistencia de CRS ---
            if not (p1_src.crs == s2_src.crs == mask_src.crs):
                raise DataValidationException(
                    f"CRS inconsistente entre imágenes:\n"
                    f"  P1: {p1_src.crs}\n"
                    f"  S2: {s2_src.crs}\n"
                    f"  Máscara: {mask_src.crs}\n"
                    f"Las tres imágenes deben estar en el mismo CRS."
                )
            
            # --- Validación 2: Geometría S2 y Máscara ---
            if not (s2_src.shape == mask_src.shape):
                raise DataValidationException(
                    f"Shape inconsistente entre S2 y máscara:\n"
                    f"  S2: {s2_src.shape}\n"
                    f"  Máscara: {mask_src.shape}"
                )
            
            if not (s2_src.transform == mask_src.transform):
                raise DataValidationException(
                    f"Transform inconsistente entre S2 y máscara:\n"
                    f"  S2: {s2_src.transform}\n"
                    f"  Máscara: {mask_src.transform}"
                )
            
            # --- Validación 3: Número de Bandas ---
            if p1_src.count != 4:
                raise DataValidationException(
                    f"P1 debe tener 4 bandas multiespectrales, "
                    f"pero tiene {p1_src.count}."
                )
            
            if s2_src.count != 4:
                raise DataValidationException(
                    f"S2 debe tener 4 bandas (B2,B3,B4,B8), "
                    f"pero tiene {s2_src.count}."
                )
            
            if mask_src.count != 1:
                raise DataValidationException(
                    f"Máscara de nubes debe tener 1 banda, "
                    f"pero tiene {mask_src.count}."
                )
            
            # --- Validación 4: Factor de Resolución ---
            p1_res_x, p1_res_y = p1_src.res
            s2_res_x, s2_res_y = s2_src.res
            
            # Verificar que las resoluciones son isotrópicas
            if not (abs(p1_res_x - p1_res_y) < 0.01):
                log.warning(
                    f"P1 tiene resolución anisotrópica: {p1_res_x} x {p1_res_y}m. "
                    f"Usando promedio."
                )
            if not (abs(s2_res_x - s2_res_y) < 0.01):
                log.warning(
                    f"S2 tiene resolución anisotrópica: {s2_res_x} x {s2_res_y}m. "
                    f"Usando promedio."
                )
            
            self.p1_resolution = (p1_res_x + p1_res_y) / 2
            self.s2_resolution = (s2_res_x + s2_res_y) / 2
            
            # Calcular factor de resolución
            calculated_factor = int(round(self.s2_resolution / self.p1_resolution))
            
            # Si el usuario especificó un factor esperado, validar
            if expected_resolution_factor is not None:
                if calculated_factor != expected_resolution_factor:
                    raise ResolutionFactorException(
                        f"Factor de resolución calculado ({calculated_factor}) "
                        f"no coincide con el esperado ({expected_resolution_factor}).\n"
                        f"  S2 resolution: {self.s2_resolution}m\n"
                        f"  P1 resolution: {self.p1_resolution}m\n"
                        f"  Ratio: {self.s2_resolution / self.p1_resolution:.2f}"
                    )
            
            self.resolution_factor = calculated_factor
            
            # Calcular tamaños de patch según metodología BCNet [cite: 82, 99]
            self.s2_patch_size = 32
            self.p1_patch_size = self.s2_patch_size * self.resolution_factor
            
            log.info(
                f"Factor de resolución: {self.resolution_factor} "
                f"(S2: {self.s2_resolution}m / P1: {self.p1_resolution}m)"
            )
            
            # --- Validación 5: Alineación Geográfica ---
            self._validate_geographic_alignment(p1_src, s2_src)
            
            # --- Validación 6: Rotación/Shear ---
            self._validate_no_rotation(p1_src, "P1")
            self._validate_no_rotation(s2_src, "S2")
            
            # Guardar dimensiones para verificación posterior
            self.s2_height, self.s2_width = s2_src.shape
            self.p1_height, self.p1_width = p1_src.shape
            
            log.info("Todas las validaciones geoespaciales pasaron exitosamente.")
    
    def _validate_geographic_alignment(
        self,
        p1_src: rasterio.DatasetReader,
        s2_src: rasterio.DatasetReader,
    ) -> None:
        """
        Valida que P1 y S2 estén geográficamente alineados.
        
        Verifica que los orígenes (esquina superior izquierda) coincidan
        dentro de una tolerancia razonable (1/10 de píxel S2).
        """
        p1_origin = (p1_src.transform.c, p1_src.transform.f)
        s2_origin = (s2_src.transform.c, s2_src.transform.f)
        
        # Tolerancia: 1/10 de la resolución S2
        tolerance = self.s2_resolution / 10.0
        
        offset_x = abs(p1_origin[0] - s2_origin[0])
        offset_y = abs(p1_origin[1] - s2_origin[1])
        
        if offset_x > tolerance or offset_y > tolerance:
            raise GeographicAlignmentException(
                f"P1 y S2 NO están geográficamente alineados:\n"
                f"  P1 origin (x, y): ({p1_origin[0]:.2f}, {p1_origin[1]:.2f})\n"
                f"  S2 origin (x, y): ({s2_origin[0]:.2f}, {s2_origin[1]:.2f})\n"
                f"  Offset (x, y): ({offset_x:.2f}m, {offset_y:.2f}m)\n"
                f"  Tolerancia: {tolerance:.2f}m\n"
                f"Reproyecte las imágenes para alinearlas correctamente."
            )
        
        log.debug(
            f"Alineación geográfica verificada. "
            f"Offset: ({offset_x:.4f}m, {offset_y:.4f}m) < {tolerance:.2f}m"
        )
    
    def _validate_no_rotation(
        self, src: rasterio.DatasetReader, name: str
    ) -> None:
        """
        Valida que la imagen no tenga rotación o shear.
        
        El código asume imágenes "north-up" (orientadas al norte) sin
        transformaciones afines complejas.
        """
        transform = src.transform
        
        # Componentes b y d deben ser 0 para imágenes north-up
        if abs(transform.b) > 1e-6 or abs(transform.d) > 1e-6:
            raise DataValidationException(
                f"{name} tiene rotación o shear:\n"
                f"  Transform: {transform}\n"
                f"  b (rotation): {transform.b}\n"
                f"  d (rotation): {transform.d}\n"
                f"Solo se soportan imágenes north-up sin rotación."
            )
    
    # ========================================================================
    # MÉTODOS PRIVADOS - EXTRACCIÓN DE PATCHES
    # ========================================================================
    
    def _extract_and_filter_patches(self) -> None:
        """
        Extrae patches no superpuestos usando lectura por ventanas.
        
        Implementa la metodología del paper:
        1. Itera sobre grid regular sin overlap
        2. Lee solo las ventanas necesarias (eficiente en memoria)
        3. Filtra patches según nubes y no-data
        4. Escala reflectancias a [0, 1]
        5. Mezcla (shuffle) los patches [cite: 81]
        """
        log.info("Extrayendo patches usando lectura por ventanas...")
        
        with rasterio.open(self.p1_path) as p1_src, \
             rasterio.open(self.s2_path) as s2_src, \
             rasterio.open(self.s2_cloud_mask_path) as mask_src:
            
            n_total_patches = 0
            n_valid_patches = 0
            n_cloud_rejected = 0
            n_nodata_rejected = 0
            
            # Calcular número total de patches para logging
            n_rows = (self.s2_height - self.s2_patch_size) // self.s2_patch_size + 1
            n_cols = (self.s2_width - self.s2_patch_size) // self.s2_patch_size + 1
            total_possible = n_rows * n_cols
            
            log.info(
                f"Dimensiones S2: {self.s2_height}x{self.s2_width}, "
                f"Grid de patches: {n_rows}x{n_cols} = {total_possible} patches máximo"
            )
            
            # Iterador no superpuesto sobre grid S2
            for y in range(0, self.s2_height - self.s2_patch_size + 1, self.s2_patch_size):
                for x in range(0, self.s2_width - self.s2_patch_size + 1, self.s2_patch_size):
                    
                    n_total_patches += 1
                    
                    # Definir ventanas de lectura
                    s2_window = Window(
                        col_off=x,
                        row_off=y,
                        width=self.s2_patch_size,
                        height=self.s2_patch_size
                    )
                    
                    # Coordenadas correspondientes en P1
                    y_p1 = y * self.resolution_factor
                    x_p1 = x * self.resolution_factor
                    
                    p1_window = Window(
                        col_off=x_p1,
                        row_off=y_p1,
                        width=self.p1_patch_size,
                        height=self.p1_patch_size
                    )
                    
                    # Leer SOLO las ventanas necesarias (memoria eficiente)
                    s2_patch = s2_src.read(window=s2_window)
                    mask_patch = mask_src.read(1, window=s2_window)
                    p1_patch = p1_src.read(window=p1_window)
                    
                    # Aplicar lógica de filtrado del paper [cite: 80]
                    is_valid, reject_reason = self._is_patch_pair_valid(
                        p1_patch, s2_patch, mask_patch
                    )
                    
                    if is_valid:
                        # Escalar y convertir a tensores
                        p1_tensor = self._to_tensor(p1_patch)
                        s2_tensor = self._to_tensor(s2_patch)
                        
                        self.patch_pairs.append((p1_tensor, s2_tensor))
                        n_valid_patches += 1
                    else:
                        # Contabilizar razones de rechazo para debugging
                        if reject_reason == 'cloud':
                            n_cloud_rejected += 1
                        elif reject_reason == 'nodata':
                            n_nodata_rejected += 1
            
            # CRÍTICO: Shuffle después de extracción [cite: 81]
            if self.shuffle_patches:
                random.shuffle(self.patch_pairs)
                log.info("Patches mezclados aleatoriamente (shuffle).")
            
            # Logging de estadísticas
            rejection_rate = (1 - n_valid_patches / n_total_patches) * 100
            log.info(
                f"Extracción completada:\n"
                f"  Total procesados: {n_total_patches}\n"
                f"  Válidos: {n_valid_patches} ({100 - rejection_rate:.1f}%)\n"
                f"  Rechazados por nubes: {n_cloud_rejected}\n"
                f"  Rechazados por no-data: {n_nodata_rejected}\n"
                f"  Tasa de rechazo: {rejection_rate:.1f}%"
            )
    
    def _is_patch_pair_valid(
        self,
        p1_patch: np.ndarray,
        s2_patch: np.ndarray,
        mask_patch: np.ndarray,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifica si un par de patches es válido según la metodología del paper.
        
        Filtros aplicados:
        1. Presencia de nubes en S2
        2. Valores no-data en S2
        3. Valores no-data en P1
        4. (Opcional) Rango físico de reflectancias
        
        Returns:
            Tupla (is_valid, reject_reason):
                - is_valid: True si el patch es válido
                - reject_reason: 'cloud', 'nodata', 'range', o None
        """
        
        # Filtro 1: Nubes en S2
        if np.isin(mask_patch, list(self.s2_cloud_values)).any():
            return False, 'cloud'
        
        # Filtro 2: No-data en S2
        if np.isin(s2_patch, self.s2_nodata).any():
            return False, 'nodata'
        
        # Filtro 3: No-data en P1
        if np.isin(p1_patch, self.p1_nodata).any():
            return False, 'nodata'
        
        # Filtro 4: Validación de rango físico (opcional)
        if self.validate_range:
            # Reflectancias S2 típicamente en [0, 10000]
            if (s2_patch < 0).any() or (s2_patch > 10000).any():
                return False, 'range'
            
            # Reflectancias P1 típicamente en [0, 10000]
            if (p1_patch < 0).any() or (p1_patch > 10000).any():
                return False, 'range'
        
        return True, None
    
    def _to_tensor(self, patch: np.ndarray) -> torch.Tensor:
        """
        Convierte un patch NumPy a tensor PyTorch y escala reflectancias.
        
        Optimizaciones:
        - Conversión directa a float32 (evita float64 intermedio)
        - Escalado en una sola operación
        
        Args:
            patch: Array NumPy [C, H, W] con valores en [0, scale_factor].
            
        Returns:
            Tensor PyTorch [C, H, W] con valores en [0, 1].
        """
        # Conversión eficiente: int16 → float32 → escalar
        return torch.from_numpy(patch.astype(np.float32)) / self.scale_factor
    
    # ========================================================================
    # MÉTODOS AUXILIARES
    # ========================================================================
    
    @staticmethod
    def _to_list(value: Union[float, List[float]]) -> List[float]:
        """Convierte un valor escalar o lista a lista."""
        return [value] if not isinstance(value, list) else value


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def create_train_test_split(
    dataset: HarmonizationPatchDataset,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.
    
    Implementa el split 90/10 descrito en el paper [cite: 81].
    
    Args:
        dataset: Dataset de HarmonizationPatchDataset cargado.
        test_ratio: Proporción del conjunto de prueba (default: 0.1 = 10%).
        seed: Semilla aleatoria para reproducibilidad.
        
    Returns:
        Tupla (train_dataset, test_dataset).
    """
    if not dataset._is_loaded:
        raise RuntimeError(
            "Dataset no cargado. Llamar dataset.load() antes de crear el split."
        )
    
    dataset_size = len(dataset)
    test_size = int(dataset_size * test_ratio)
    train_size = dataset_size - test_size
    
    # Usar random_split de PyTorch con generador para reproducibilidad
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )
    
    log.info(
        f"Dataset dividido: {train_size} entrenamiento ({100*(1-test_ratio):.0f}%), "
        f"{test_size} prueba ({100*test_ratio:.0f}%)."
    )
    
    return train_dataset, test_dataset
