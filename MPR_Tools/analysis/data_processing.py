"""Data processing utilities for MPR analysis."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from typing import Tuple

if TYPE_CHECKING:
    from ..core.spectrometer import MPRSpectrometer

class DataProcessor:
    """Handles data processing operations."""
    def __init__(self, spectrometer: MPRSpectrometer) -> None:
        self.spectrometer = spectrometer
    
    def bin_hodoscope_response(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin hydron hits into hodoscope channels.
        
        Returns:
            Tuple of (channel_numbers, counts_per_channel)
        """
        channel_counts = np.zeros(self.spectrometer.hodoscope.total_channels)
        
        for x_position in self.spectrometer.output_beam[:, 0]:
            channel = self.spectrometer.hodoscope.get_channel_for_position(x_position)
            if channel is not None:
                channel_counts[channel] += 1
        
        channel_numbers = np.arange(self.spectrometer.hodoscope.total_channels)
        return channel_numbers, channel_counts
    
    def get_proton_density_map(
        self, 
        dx: float = 0.01, 
        dy: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the density of proton impact sites in the focal plane.
        
        Args:
            dx: X-direction resolution in meters
            dy: Y-direction resolution in meters
            
        Returns:
            Tuple of (density_array, X_meshgrid, Y_meshgrid)
        """
        if len(self.spectrometer.output_beam) == 0:
            raise ValueError("No output beam data available. Run apply_transfer_map() first.")
        
        x_positions = self.spectrometer.output_beam[:, 0]
        y_positions = self.spectrometer.output_beam[:, 2]
        
        # Define grid boundaries
        x_min, x_max = np.min(x_positions), np.max(x_positions)
        y_min, y_max = np.min(y_positions), np.max(y_positions)
        
        # Create coordinate arrays
        x_coords = np.linspace(x_min, x_max, int((x_max - x_min) / dx) + 1)
        y_coords = np.linspace(y_min, y_max, int((y_max - y_min) / dy) + 1)
        
        print(f"Grid bounds: X=[{x_min:.4f}, {x_max:.4f}], Y=[{y_min:.4f}, {y_max:.4f}]")
        
        # Create meshgrids
        X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)
        density = np.zeros_like(X_mesh)
        
        # Bin protons into grid cells
        total_protons = len(self.spectrometer.output_beam)
        for x_pos, y_pos in zip(x_positions, y_positions):
            # Convert coordinates to grid indices
            x_idx = int((x_pos - x_min) / dx)
            y_idx = int((y_pos - y_min) / dy)
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, density.shape[1] - 1))
            y_idx = max(0, min(y_idx, density.shape[0] - 1))
            
            density[y_idx, x_idx] += 1 / total_protons
        
        return density, X_mesh, Y_mesh