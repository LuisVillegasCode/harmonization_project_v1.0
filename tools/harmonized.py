"""
Harmonization of PeruSAT-1 and Sentinel-2 Surface Reflectances using Deep Learning.

This module implements the CalibNet architecture proposed by Michel & Inglada (2021)
for cross-sensor radiometric harmonization. It addresses critical issues identified
in the original implementation:
    - Corrected skip connection implementation
    - Added MTF filtering preprocessing
    - Enhanced evaluation metrics and visualizations
    - Improved code structure following software engineering best practices

Reference:
    Michel, J., & Inglada, J. (2021). Learning harmonised Pleiades and Sentinel-2 
    surface reflectances. The International Archives of Photogrammetry, Remote Sensing 
    and Spatial Information Sciences, XLIII-B3-2021, 265-272.

Author: Refactored Implementation
Date: October 2025
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, asdict
import json

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    explained_variance_score,
    max_error
)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

@dataclass
class HarmonizationConfig:
    """Configuration parameters for the harmonization workflow."""
    
    # Input/Output Paths
    p1_aligned_stack: Path
    s2_aligned_stack: Path
    s2_cloud_mask: Path
    output_dir: Path
    
    # Model Hyperparameters (from Michel & Inglada, 2021, Section 2.3)
    learning_rate: float = 0.0002
    hidden_units: int = 320
    target_iterations: int = 5000
    batch_size: int = 2048
    
    # Data Parameters (from Section 2.1)
    test_split_ratio: float = 0.1  # Paper uses 90/10 split
    bands: List[str] = None
    data_floor_value: float = 0.0001
    data_ceiling_value: float = 1.0
    
    # MTF Filter Parameters (Sentinel-2 specific)
    # These sigma values approximate S2 MTF at 10m resolution
    mtf_sigma: Dict[str, float] = None
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.bands is None:
            self.bands = ['Blue', 'Green', 'Red', 'NIR']
        
        if self.mtf_sigma is None:
            # Default MTF sigma values for S2 bands (empirical approximations)
            # These should be calibrated based on sensor specifications
            self.mtf_sigma = {
                'Blue': 1.2,   # B2
                'Green': 1.2,  # B3
                'Red': 1.2,    # B4
                'NIR': 1.5     # B8 (slightly broader PSF)
            }
    
    def save(self, filepath: Path):
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        # Convert Path objects to strings
        config_dict['p1_aligned_stack'] = str(config_dict['p1_aligned_stack'])
        config_dict['s2_aligned_stack'] = str(config_dict['s2_aligned_stack'])
        config_dict['s2_cloud_mask'] = str(config_dict['s2_cloud_mask'])
        config_dict['output_dir'] = str(config_dict['output_dir'])
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Configure logging to both file and console.
    
    Args:
        output_dir: Directory where log file will be saved
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('Harmonization')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(output_dir / 'harmonization.log')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class CalibNet(nn.Module):
    """
    Multi-Layer Perceptron for pixel-wise radiometric calibration.
    
    Implements the architecture from Michel & Inglada (2021), Figure 2(a):
        Input (4 bands) -> BatchNorm -> FC(320) -> LeakyReLU -> 
        FC(320) -> LeakyReLU -> FC(4) -> Tanh -> Skip Connection -> Output
    
    The skip connection adds the ORIGINAL input (before normalization) to the
    output, allowing the network to learn residual corrections.
    
    Args:
        in_features: Number of input bands (default: 4)
        out_features: Number of output bands (default: 4)
        hidden_units: Number of units in hidden layers (default: 320, from paper)
    """
    
    def __init__(
        self, 
        in_features: int = 4, 
        out_features: int = 4, 
        hidden_units: int = 320
    ):
        super(CalibNet, self).__init__()
        
        # Input normalization layer
        self.batch_norm = nn.BatchNorm1d(in_features)
        
        # Hidden layers with LeakyReLU activation
        self.fc1 = nn.Linear(in_features, hidden_units)
        self.activation1 = nn.LeakyReLU(inplace=True)
        
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.activation2 = nn.LeakyReLU(inplace=True)
        
        # Output layer with Tanh activation
        self.fc_out = nn.Linear(hidden_units, out_features)
        self.output_activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with corrected skip connection.
        
        CRITICAL FIX: The skip connection now correctly adds the ORIGINAL input
        (before BatchNorm) to the processed output. This matches Figure 2(a)
        of the paper and allows the network to learn additive corrections.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Store original input for skip connection (BEFORE normalization)
        identity = x
        
        # Main processing path
        x = self.batch_norm(x)
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.output_activation(self.fc_out(x))
        
        # Skip connection: add original input
        # This allows learning: output = f(input) + input
        x = x + identity
        
        return x


class RelativeErrorLoss(nn.Module):
    """
    Relative error loss function from Michel & Inglada (2021), Equation 1.
    
    L(target, prediction) = mean(|target - prediction| / (epsilon + prediction))
    
    This loss emphasizes relative errors over absolute errors, which is
    important for reflectance values that span multiple orders of magnitude.
    
    Args:
        epsilon: Small constant to prevent division by zero (default: 1e-6)
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super(RelativeErrorLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute relative error loss.
        
        Args:
            prediction: Predicted reflectances (model output)
            target: Target reflectances (Sentinel-2 reference)
            
        Returns:
            Scalar loss value
        """
        abs_diff = torch.abs(target - prediction)
        # Clamp target to avoid negative denominators
        target_safe = torch.clamp(target, min=0.0)
        relative_error = abs_diff / (self.epsilon + target_safe)
        return torch.mean(relative_error)

    
# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class MTFFilter:
    """
    Modulation Transfer Function (MTF) filter for spatial harmonization.
    
    Applies Gaussian filtering to higher-resolution imagery to match the
    spatial frequency content of the lower-resolution reference sensor.
    This is critical preprocessing step from Section 2.1 of the paper.
    """
    
    def __init__(self, mtf_sigma: Dict[str, float], band_names: List[str]):
        """
        Initialize MTF filter with band-specific sigma values.
        
        Args:
            mtf_sigma: Dictionary mapping band names to Gaussian sigma values
            band_names: Ordered list of band names
        """
        self.mtf_sigma = mtf_sigma
        self.band_names = band_names
        self.logger = logging.getLogger('Harmonization.MTFFilter')
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply MTF filtering to multi-band image.
        
        Args:
            data: Input array of shape (n_bands, height, width)
            
        Returns:
            Filtered array of same shape
        """
        self.logger.info("Applying MTF filtering...")
        filtered_data = np.zeros_like(data, dtype=np.float32)
        
        for i, band_name in enumerate(self.band_names):
            sigma = self.mtf_sigma.get(band_name, 1.2)  # Default sigma
            self.logger.debug(f"  Band {band_name}: sigma = {sigma:.2f}")
            filtered_data[i] = gaussian_filter(data[i], sigma=sigma)
        
        self.logger.info("MTF filtering completed.")
        return filtered_data


class PixelDatabase:
    """
    Manages extraction and preparation of pixel samples for training/testing.
    
    Implements data preparation workflow from Section 2.1 of the paper:
        1. Load co-registered image pairs
        2. Apply cloud masking
        3. Filter invalid pixels (NoData, extremes)
        4. Extract valid pixel pairs
        5. Split into training/testing sets
    """
    
    def __init__(
        self, 
        config: HarmonizationConfig,
        logger: logging.Logger
    ):
        self.config = config
        self.logger = logger
    
    def load_and_validate_images(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Load input images and validate geometric consistency.
        
        Returns:
            Tuple of (p1_data, s2_data, cloud_mask, s2_metadata)
        """
        self.logger.info("Loading input images...")
        
        try:
            # Load Sentinel-2 reference
            with rasterio.open(self.config.s2_aligned_stack) as s2_ds:
                s2_data = s2_ds.read().astype(np.float32)
                s2_data /= 10000.0
                s2_meta = s2_ds.meta.copy()
                s2_shape = s2_ds.shape
            
            # Load PeruSAT-1 input
            with rasterio.open(self.config.p1_aligned_stack) as p1_ds:
                p1_data = p1_ds.read().astype(np.float32)
                p1_data /= 10000.0
                p1_shape = p1_ds.shape
            
            # Load cloud mask
            with rasterio.open(self.config.s2_cloud_mask) as cm_ds:
                cloud_mask = cm_ds.read(1)
            
            # Validate geometric consistency
            if p1_shape != s2_shape:
                self.logger.error(
                    f"Geometric mismatch: S2 {s2_shape} vs P1 {p1_shape}"
                )
                raise ValueError(
                    "Input images must have identical dimensions. "
                    "Ensure proper co-registration and resampling."
                )
            
            self.logger.info(f"Images loaded successfully: {s2_shape}")
            return p1_data, s2_data, cloud_mask, s2_meta
            
        except Exception as e:
            self.logger.error(f"Failed to load images: {e}")
            raise
    
    def create_valid_pixel_mask(
        self,
        p1_data: np.ndarray,
        s2_data: np.ndarray,
        cloud_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create boolean mask for valid training pixels.
        
        Valid pixels must satisfy:
            - No clouds (from S2 mask)
            - No NoData values
            - Within valid reflectance range [floor, ceiling]
        
        Args:
            p1_data: PeruSAT-1 data (n_bands, h, w)
            s2_data: Sentinel-2 data (n_bands, h, w)
            cloud_mask: Binary cloud mask (h, w), 0=clear
            
        Returns:
            Boolean mask (h, w) where True = valid pixel
        """
        self.logger.info("Creating valid pixel mask...")
        
        # Start with cloud-free pixels
        bad_values_list = [3, 8, 9, 10]
        valid_mask = ~np.isin(cloud_mask, bad_values_list)
        self.logger.debug(f"  Cloud-free pixels: {valid_mask.sum():,}")
        
        # Exclude NoData (using first band as proxy)
        valid_mask &= (s2_data[0] > self.config.data_floor_value)
        valid_mask &= (p1_data[0] > self.config.data_floor_value)
        self.logger.debug(f"  After NoData filter: {valid_mask.sum():,}")
        
        # Exclude extreme values (potential errors)
        valid_mask &= (s2_data[0] < self.config.data_ceiling_value)
        valid_mask &= (p1_data[0] < self.config.data_ceiling_value)
        self.logger.debug(f"  After extremes filter: {valid_mask.sum():,}")
        
        n_valid = valid_mask.sum()
        total_pixels = valid_mask.size
        valid_percent = 100 * n_valid / total_pixels
        
        self.logger.info(
            f"Valid pixels: {n_valid:,} / {total_pixels:,} ({valid_percent:.1f}%)"
        )
        
        if n_valid < 10000:
            self.logger.warning(
                "Very low number of valid pixels. Results may be unreliable."
            )
        
        return valid_mask
    
    def filter_by_relative_difference(
        self,
        p1_samples: np.ndarray,
        s2_samples: np.ndarray,
        threshold: float = 0.40
        ) -> np.ndarray:
        """
            Filtra píxeles cuya diferencia relativa supera el umbral especificado.
    
            Calcula: |S2 - P1| / S2 para cada banda. Si alguna banda excede el 
            umbral, el píxel completo se marca como inválido.
    
    Args:
        p1_samples: Muestras PeruSAT-1 (N, n_bands)
        s2_samples: Muestras Sentinel-2 (N, n_bands) - referencia
        threshold: Umbral de diferencia relativa (default: 0.40 = 40%)
        
    Returns:
        Máscara booleana (N,) donde True = píxel válido
    """
        self.logger.info(f"Aplicando filtro de diferencia relativa (umbral: {threshold*100:.0f}%)...")
        
        # Evitar división por cero
        epsilon = 1e-6
        s2_safe = np.maximum(s2_samples, epsilon)
        
        # Calcular diferencia relativa por banda: |S2 - P1| / S2
        relative_diff = np.abs(s2_samples - p1_samples) / s2_safe
        
        # Un píxel es válido si TODAS sus bandas están bajo el umbral
        valid_mask = np.all(relative_diff <= threshold, axis=1)
        
        n_total = len(valid_mask)
        n_valid = valid_mask.sum()
        n_rejected = n_total - n_valid
        rejection_rate = 100 * n_rejected / n_total
        
        self.logger.info(f"  Píxeles rechazados: {n_rejected:,} / {n_total:,} ({rejection_rate:.1f}%)")
        self.logger.info(f"  Píxeles válidos restantes: {n_valid:,}")
        
        # Advertencia si rechazo es muy alto
        if rejection_rate > 50:
            self.logger.warning(
                f"Tasa de rechazo muy alta ({rejection_rate:.1f}%). "
                "Verifique co-registro y calibración de imágenes."
            )
        
        return valid_mask
    
    def extract_pixel_samples(
        self,
        p1_data: np.ndarray,
        s2_data: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract valid pixel samples and flatten to (N, n_bands) format.
        
        Args:
            p1_data: PeruSAT-1 data (n_bands, h, w)
            s2_data: Sentinel-2 data (n_bands, h, w)
            valid_mask: Boolean mask (h, w)
            
        Returns:
            Tuple of (X_samples, y_samples), each with shape (N, n_bands)
        """
        self.logger.info("Extrayendo píxeles válidos...")
        
        # Transpose to (h, w, n_bands)
        p1_hwc = np.moveaxis(p1_data, 0, -1)
        s2_hwc = np.moveaxis(s2_data, 0, -1)
        
        # Extract valid pixels (filtro espacial previo)
        X_samples_raw = p1_hwc[valid_mask]  # PeruSAT-1 as input
        y_samples_raw = s2_hwc[valid_mask]  # Sentinel-2 as target
        
        self.logger.info(f"Píxeles extraídos (pre-filtro): {X_samples_raw.shape[0]:,}")
        
        # **NUEVO: Aplicar filtro de diferencia relativa**
        relative_diff_mask = self.filter_by_relative_difference(
            X_samples_raw, 
            y_samples_raw,
            threshold=0.90  # 40% como especificado
        )
        
        # Aplicar máscara de diferencia relativa
        X_samples = X_samples_raw[relative_diff_mask]
        y_samples = y_samples_raw[relative_diff_mask]
        
        self.logger.info(f"Píxeles finales (post-filtro): {X_samples.shape[0]:,} pares")
        
        return X_samples, y_samples
        
    def prepare(
        self
    ) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, dict]:
        """
        Complete data preparation pipeline.
        
        Returns:
            Tuple of:
                - train_loader: DataLoader for training
                - test_loader: DataLoader for validation
                - X_test: Test input samples (for metrics)
                - y_test: Test target samples (for metrics)
                - s2_meta: Metadata for output raster
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: DATA PREPARATION")
        self.logger.info("=" * 60)
        
        # Load images
        p1_data, s2_data, cloud_mask, s2_meta = self.load_and_validate_images()
        
        # Apply MTF filtering to PeruSAT-1
        mtf_filter = MTFFilter(self.config.mtf_sigma, self.config.bands)
        p1_data = mtf_filter.apply(p1_data)
        
        # Create valid pixel mask
        valid_mask = self.create_valid_pixel_mask(p1_data, s2_data, cloud_mask)
        
        # Extract samples
        X_samples, y_samples = self.extract_pixel_samples(
            p1_data, s2_data, valid_mask
        )
        
        # Split into train/test (following paper's 90/10 split)
        self.logger.info("Splitting into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_samples,
            y_samples,
            test_size=self.config.test_split_ratio,
            random_state=self.config.random_seed
        )
        
        self.logger.info(f"Training samples: {X_train.shape[0]:,}")
        self.logger.info(f"Testing samples: {X_test.shape[0]:,}")
        
        # Create PyTorch DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=2
        )
        
        self.logger.info("Data preparation completed successfully.")
        return train_loader, test_loader, X_test, y_test, s2_meta


# ============================================================================
# MODEL TRAINING
# ============================================================================

class ModelTrainer:
    """
    Handles model training, validation, and checkpoint management.
    
    Implements training procedure from Section 2.3 of the paper:
        - Adam optimizer with lr=0.0002
        - Relative error loss
        - ~5000 iterations (converted to epochs)
        - Early stopping based on validation loss
    """
    
    def __init__(
        self,
        model: CalibNet,
        config: HarmonizationConfig,
        logger: logging.Logger
    ):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        self.logger.info(f"Training on device: {self.device}")
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        self.criterion = RelativeErrorLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def calculate_epochs_from_iterations(
        self,
        n_train_samples: int
    ) -> int:
        """
        Calculate number of epochs needed for target iterations.
        
        Paper uses ~5000 iterations. We convert this to epochs based on
        the actual number of training samples and batch size.
        
        Args:
            n_train_samples: Number of training samples
            
        Returns:
            Number of epochs
        """
        iterations_per_epoch = n_train_samples / self.config.batch_size
        epochs = int(np.ceil(self.config.target_iterations / iterations_per_epoch))
        
        self.logger.info(
            f"Target iterations: {self.config.target_iterations}, "
            f"Iterations/epoch: {iterations_per_epoch:.1f}, "
            f"Required epochs: {epochs}"
        )
        
        return max(epochs, 10)  # Minimum 10 epochs
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        return epoch_loss / n_batches
    
    def validate(self, test_loader: DataLoader) -> float:
        """Validate model on test set."""
        self.model.eval()
        val_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                val_loss += loss.item()
                n_batches += 1
        
        return val_loss / n_batches
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.history['train_loss'][-1],
            'val_loss': self.history['val_loss'][-1],
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        checkpoint_path = self.config.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.config.output_dir / 'calibnet_best_model.pth'
            torch.save(self.model.state_dict(), best_path)
            self.logger.info(f"  -> New best model saved (epoch {epoch})")
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Complete training procedure.
        
        Returns:
            Training history dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: MODEL TRAINING")
        self.logger.info("=" * 60)
        
        # Calculate number of epochs
        n_train_samples = len(train_loader.dataset)
        n_epochs = self.calculate_epochs_from_iterations(n_train_samples)
        
        self.logger.info(f"Starting training for {n_epochs} epochs...")
        self.logger.info("-" * 60)
        
        for epoch in range(n_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(test_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epochs'].append(epoch + 1)
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
                f"{' <- BEST' if is_best else ''}"
            )
        
        self.logger.info("-" * 60)
        self.logger.info(
            f"Training completed. Best validation loss: {self.best_val_loss:.6f} "
            f"at epoch {self.best_epoch}"
        )
        
        return self.history


# ============================================================================
# MODEL EVALUATION
# ============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Implements evaluation procedures from the paper:
        - Metrics: RMSE, R², Explained Variance, Max Error (Figure 3)
        - Visualizations: Scatter plots with 1:1 line (Figure 4)
        - Training curves (Figures 5, 6, 9)
    """
    
    def __init__(
        self,
        config: HarmonizationConfig,
        logger: logging.Logger
    ):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute evaluation metrics for each band.
        
        Args:
            y_true: True values (n_samples, n_bands)
            y_pred: Predicted values (n_samples, n_bands)
            
        Returns:
            Dictionary with metrics per band
        """
        metrics = {}
        
        for i, band_name in enumerate(self.config.bands):
            y_true_band = y_true[:, i]
            y_pred_band = y_pred[:, i]
            
            metrics[band_name] = {
                'RMSE': float(np.sqrt(mean_squared_error(y_true_band, y_pred_band))),
                'R2_Score': float(r2_score(y_true_band, y_pred_band)),
                'Explained_Variance': float(explained_variance_score(y_true_band, y_pred_band)),
                'Max_Error': float(max_error(y_true_band, y_pred_band))
            }
        
        return metrics
    
    def print_metrics_table(self, metrics: Dict[str, Dict[str, float]]):
        """Print formatted metrics table."""
        self.logger.info("-" * 70)
        self.logger.info("EVALUATION METRICS (Test Set)")
        self.logger.info("-" * 70)
        header = f"{'Band':<10} {'RMSE':<12} {'R² Score':<12} {'Expl. Var.':<12} {'Max Error':<12}"
        self.logger.info(header)
        self.logger.info("-" * 70)
        
        for band, band_metrics in metrics.items():
            row = (
                f"{band:<10} "
                f"{band_metrics['RMSE']:<12.6f} "
                f"{band_metrics['R2_Score']:<12.6f} "
                f"{band_metrics['Explained_Variance']:<12.6f} "
                f"{band_metrics['Max_Error']:<12.6f}"
            )
            self.logger.info(row)
        
        self.logger.info("-" * 70)
    
    def plot_scatter_comparison(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, Dict[str, float]]
    ):
        """
        Create scatter plots comparing true vs predicted values.
        
        Replicates Figure 4 from the paper with 2D histograms and 1:1 line.
        """
        self.logger.info("Generating scatter plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes = axes.flatten()
        
        # Determine value range (99th percentile to avoid outliers)
        vmax = np.percentile(y_true, 99)
        
        for i, (ax, band_name) in enumerate(zip(axes, self.config.bands)):
            y_true_band = y_true[:, i]
            y_pred_band = y_pred[:, i]
            
            # 2D histogram (density plot)
            h = ax.hist2d(
                y_true_band,
                y_pred_band,
                bins=100,
                norm=LogNorm(),
                cmap='viridis',
                range=[[0, vmax], [0, vmax]]
            )
            
            # Add colorbar
            plt.colorbar(h[3], ax=ax, label='Pixel Count')
            
            # 1:1 reference line
            ax.plot([0, vmax], [0, vmax], 'r--', linewidth=2, label='1:1 Line')
            
            # Formatting
            ax.set_xlabel('Sentinel-2 (Reference)', fontsize=11)
            ax.set_ylabel('PeruSAT-1 (Harmonized)', fontsize=11)
            ax.set_title(
                f'{band_name} - R² = {metrics[band_name]["R2_Score"]:.4f}',
                fontsize=12,
                fontweight='bold'
            )
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='upper left')
            ax.set_xlim(0, vmax)
            ax.set_ylim(0, vmax)
        
        plt.suptitle(
            'Harmonization Evaluation: Sentinel-2 vs PeruSAT-1 (Test Set)',
            fontsize=16,
            fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        
        # Save figure
        plot_path = self.config.output_dir / 'evaluation_scatter_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Scatter plots saved to: {plot_path}")
        plt.close()
    
    def plot_training_curves(self, history: Dict[str, List[float]]):
        """
        Plot training and validation loss curves.
        
        Replicates training curve visualizations from Figures 5, 6, 9 of paper.
        """
        self.logger.info("Generating training curves...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = history['epochs']
        
        # Plot losses
        ax.plot(
            epochs, 
            history['train_loss'], 
            'b-', 
            linewidth=2, 
            label='Training Loss',
            marker='o',
            markersize=4,
            markevery=max(1, len(epochs) // 20)
        )
        ax.plot(
            epochs, 
            history['val_loss'], 
            'r-', 
            linewidth=2, 
            label='Validation Loss',
            marker='s',
            markersize=4,
            markevery=max(1, len(epochs) // 20)
        )
        
        # Mark best validation loss
        best_idx = np.argmin(history['val_loss'])
        best_epoch = history['epochs'][best_idx]
        best_loss = history['val_loss'][best_idx]
        
        ax.axvline(
            best_epoch, 
            color='g', 
            linestyle='--', 
            alpha=0.7,
            label=f'Best Model (Epoch {best_epoch})'
        )
        ax.plot(
            best_epoch,
            best_loss,
            'g*',
            markersize=15,
            markeredgecolor='darkgreen',
            markeredgewidth=1.5
        )
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Relative Error Loss', fontsize=12)
        ax.set_title(
            'Training Progress: CalibNet Model',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        
        # Save figure
        curves_path = self.config.output_dir / 'training_curves.png'
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Training curves saved to: {curves_path}")
        plt.close()
    
    def plot_residuals_histogram(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """
        Plot histogram of prediction residuals for each band.
        
        Additional visualization to assess error distribution.
        """
        self.logger.info("Generating residuals histogram...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, (ax, band_name) in enumerate(zip(axes, self.config.bands)):
            residuals = y_pred[:, i] - y_true[:, i]
            
            # Histogram
            ax.hist(
                residuals,
                bins=100,
                color='steelblue',
                alpha=0.7,
                edgecolor='black'
            )
            
            # Statistics
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
            
            # Add vertical lines for mean and ±1 std
            ax.axvline(mean_res, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_res:.4f}')
            ax.axvline(mean_res + std_res, color='orange', linestyle=':', linewidth=2, label=f'±1 σ: {std_res:.4f}')
            ax.axvline(mean_res - std_res, color='orange', linestyle=':', linewidth=2)
            
            # Formatting
            ax.set_xlabel('Residual (Predicted - True)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{band_name} Band', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(
            'Prediction Residuals Distribution',
            fontsize=16,
            fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        
        # Save figure
        residuals_path = self.config.output_dir / 'residuals_histogram.png'
        plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Residuals histogram saved to: {residuals_path}")
        plt.close()
    
    def evaluate(
        self,
        model: CalibNet,
        X_test: np.ndarray,
        y_test: np.ndarray,
        history: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Complete evaluation pipeline.
        
        Args:
            model: Trained CalibNet model
            X_test: Test input samples
            y_test: Test target samples
            history: Training history
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: MODEL EVALUATION")
        self.logger.info("=" * 60)
        
        # Set model to evaluation mode
        model.to(self.device)
        model.eval()
        
        # Generate predictions
        self.logger.info("Generating predictions on test set...")
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_pred = model(X_test_tensor).cpu().numpy()
        
        # Compute metrics
        self.logger.info("Computing evaluation metrics...")
        metrics = self.compute_metrics(y_test, y_pred)
        
        # Print metrics
        self.print_metrics_table(metrics)
        
        # Save metrics to JSON
        metrics_path = self.config.output_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Metrics saved to: {metrics_path}")
        
        # Generate visualizations
        self.plot_scatter_comparison(y_test, y_pred, metrics)
        self.plot_training_curves(history)
        self.plot_residuals_histogram(y_test, y_pred)
        
        self.logger.info("Evaluation completed successfully.")
        return metrics


# ============================================================================
# IMAGE HARMONIZATION (INFERENCE)
# ============================================================================

class ImageHarmonizer:
    """
    Applies trained model to full-resolution images.
    
    Handles:
        - Loading trained model
        - Processing images in batches (memory efficient)
        - Preserving NoData values
        - Writing georeferenced output
    """
    
    def __init__(
        self,
        config: HarmonizationConfig,
        logger: logging.Logger
    ):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, model_path: Path) -> CalibNet:
        """Load trained model from checkpoint."""
        self.logger.info(f"Loading model from: {model_path}")
        
        model = CalibNet(
            in_features=len(self.config.bands),
            out_features=len(self.config.bands),
            hidden_units=self.config.hidden_units
        )
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Model loaded successfully on {self.device}")
        return model
    
    def harmonize(
        self,
        model: CalibNet,
        s2_meta: dict,
        output_path: Path
    ):
        """
        Apply harmonization to full PeruSAT-1 image.
        
        Args:
            model: Trained CalibNet model
            s2_meta: Sentinel-2 metadata for output georeferencing
            output_path: Path for output harmonized image
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: FULL IMAGE HARMONIZATION")
        self.logger.info("=" * 60)
        
        # Load PeruSAT-1 image
        self.logger.info("Loading PeruSAT-1 image...")
        with rasterio.open(self.config.p1_aligned_stack) as src:
            p1_data = src.read().astype(np.float32)
            p1_data /= 10000.0
            height, width = src.shape
        
        self.logger.info(f"Image dimensions: {height} x {width}")
        
        # Apply MTF filtering (same as training)
        mtf_filter = MTFFilter(self.config.mtf_sigma, self.config.bands)
        p1_data = mtf_filter.apply(p1_data)
        
        # Prepare for batch processing
        nodata_value = s2_meta.get('nodata', 0)
        p1_hwc = np.moveaxis(p1_data, 0, -1)  # (H, W, C)
        
        # Create NoData mask
        nodata_mask = (p1_hwc[..., 0] == nodata_value)
        
        # Flatten
        p1_flat = p1_hwc.reshape(-1, len(self.config.bands))
        n_pixels = p1_flat.shape[0]
        
        self.logger.info(f"Processing {n_pixels:,} pixels...")
        
        # Initialize output
        harmonized_flat = np.zeros_like(p1_flat, dtype=np.float32)
        
        # Process in batches
        batch_size = self.config.batch_size * 4  # Larger batches for inference
        n_batches = int(np.ceil(n_pixels / batch_size))
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_pixels)
                
                # Process batch
                batch_input = torch.tensor(
                    p1_flat[start_idx:end_idx],
                    dtype=torch.float32
                ).to(self.device)
                
                batch_output = model(batch_input)
                harmonized_flat[start_idx:end_idx] = batch_output.cpu().numpy()
                
                # Progress update
                if (i + 1) % 100 == 0 or (i + 1) == n_batches:
                    progress = 100 * (i + 1) / n_batches
                    self.logger.info(f"  Progress: {progress:.1f}% ({i+1}/{n_batches} batches)")
        
        # Reshape to image
        harmonized_hwc = harmonized_flat.reshape(height, width, len(self.config.bands))
        
        # Restore NoData
        harmonized_hwc[nodata_mask] = nodata_value
        
        # Transpose to (C, H, W)
        harmonized_chw = np.moveaxis(harmonized_hwc, -1, 0)
        
        harmonized_chw *= 10000.0
        # Prepare output metadata
        output_meta = s2_meta.copy()
        output_meta.update({
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': len(self.config.bands),
            'nodata': nodata_value,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256
        })
        
        # Write output
        self.logger.info(f"Writing harmonized image to: {output_path}")
        with rasterio.open(output_path, 'w', **output_meta) as dst:
            dst.write(harmonized_chw.astype(np.uint16))
            
            # Copy band descriptions if available
            for i, band_name in enumerate(self.config.bands, start=1):
                dst.set_band_description(i, f'{band_name} (Harmonized)')
        
        self.logger.info("Image harmonization completed successfully!")


# ============================================================================
# MAIN WORKFLOW ORCHESTRATOR
# ============================================================================

class HarmonizationWorkflow:
    """
    Orchestrates the complete harmonization workflow.
    
    Pipeline:
        1. Data Preparation (pixel extraction, MTF filtering, train/test split)
        2. Model Training (CalibNet with relative error loss)
        3. Model Evaluation (metrics and visualizations)
        4. Image Harmonization (apply to full image)
    """
    
    def __init__(self, config: HarmonizationConfig):
        self.config = config
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.config.output_dir)
        
        # Log configuration
        self.logger.info("=" * 60)
        self.logger.info("PERUSAT-1 / SENTINEL-2 HARMONIZATION WORKFLOW")
        self.logger.info("=" * 60)
        self.logger.info(f"Implementation of Michel & Inglada (2021) CalibNet")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info("=" * 60)
        
        # Save configuration
        config_path = self.config.output_dir / 'config.json'
        self.config.save(config_path)
        self.logger.info(f"Configuration saved to: {config_path}")
    
    def run(self):
        """Execute complete harmonization workflow."""
        try:
            # Step 1: Prepare data
            pixel_db = PixelDatabase(self.config, self.logger)
            train_loader, test_loader, X_test, y_test, s2_meta = pixel_db.prepare()
            
            # Step 2: Train model
            model = CalibNet(
                in_features=len(self.config.bands),
                out_features=len(self.config.bands),
                hidden_units=self.config.hidden_units
            )
            
            trainer = ModelTrainer(model, self.config, self.logger)
            history = trainer.train(train_loader, test_loader)
            
            # Step 3: Evaluate model
            best_model = CalibNet(
                in_features=len(self.config.bands),
                out_features=len(self.config.bands),
                hidden_units=self.config.hidden_units
            )
            best_model_path = self.config.output_dir / 'calibnet_best_model.pth'
            best_model.load_state_dict(torch.load(best_model_path))
            
            evaluator = ModelEvaluator(self.config, self.logger)
            metrics = evaluator.evaluate(best_model, X_test, y_test, history)
            
            # Step 4: Harmonize full image
            harmonizer = ImageHarmonizer(self.config, self.logger)
            output_path = self.config.output_dir / 'PS1_harmonized_to_S2.tif'
            harmonizer.harmonize(best_model, s2_meta, output_path)
            
            # Final summary
            self.logger.info("=" * 60)
            self.logger.info("WORKFLOW COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 60)
            self.logger.info("Output files:")
            self.logger.info(f"  - Harmonized image: {output_path}")
            self.logger.info(f"  - Best model: {best_model_path}")
            self.logger.info(f"  - Metrics: {self.config.output_dir / 'evaluation_metrics.json'}")
            self.logger.info(f"  - Plots: {self.config.output_dir}/*.png")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Workflow failed with error: {e}", exc_info=True)
            raise


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
#Main entry point for the harmonization workflow.
    
    # Example configuration
    # MODIFY THESE PATHS FOR YOUR PROJECT
    config = HarmonizationConfig(
        # Input files
        p1_aligned_stack=Path(r"C:\Users\51926\Downloads\PS1_alineado_FINAL_EN.tif"),
        s2_aligned_stack=Path(r"C:\Users\51926\Downloads\s2_recortado.tif"),
        s2_cloud_mask=Path(r"C:\Users\51926\Downloads\S2_cloud_mask_ALIGNED.tif"),
        
        # Output directory
        output_dir=Path("results/harmonization_run_003"),
        
        # Model hyperparameters (from paper)
        learning_rate=0.0002,
        hidden_units=320,
        target_iterations=5000,
        batch_size=2048,
        
        # Data parameters
        test_split_ratio=0.1,  # 90/10 split as in paper
        bands=['Blue', 'Green', 'Red', 'NIR'],
        
        # MTF filter (adjust based on sensor specs)
        mtf_sigma={
            'Blue': 1.2,
            'Green': 1.2,
            'Red': 1.2,
            'NIR': 1.5
        },
        
        # Reproducibility
        random_seed=42
    )
    
    # Run workflow
    workflow = HarmonizationWorkflow(config)
    workflow.run()
    
if __name__ == "__main__":
    main()
