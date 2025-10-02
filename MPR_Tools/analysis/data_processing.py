"""Data processing utilities for MPR analysis."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from typing import Tuple, Optional

if TYPE_CHECKING:
    from ..core.spectrometer import MPRSpectrometer

class DataProcessor:
    """Handles data processing operations."""
    def __init__(
        self,
        spectrometer: MPRSpectrometer,
        performance_curve_file: Optional[str] = None) -> None:
        
        self.spectrometer = spectrometer
        self.performance_curve_file = performance_curve_file or 'comprehensive_performance.csv'
    
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
    
    def get_plasma_parameters(
        self,
        n_bins: int = 200,
        dsr_energy_range: Tuple[float, float] = (10, 12),
        primary_energy_range: Tuple[float, float] = (13, 15)
    ) -> Tuple[float, float, float, Tuple[float, float], Tuple[float, float], np.ndarray]:
        """
        Get plasma parameters from the spectrometer object.
        
        Args:
            n_bins:
                Number of bins to use for histogram
            dsr_energy_range:
                Energy range for DSR in MeV
            primary_energy_range:
                Energy range for primary neutrons in MeV
            
        Returns:
            Tuple of (dsr, plasma_temperature, fwhm, dsr_energy_range, primary_energy_range, energies)
        """
        energies = self._get_neutron_spectrum()
        
        # Calculate dsr
        ds_count = np.sum((energies > dsr_energy_range[0]) & (energies < dsr_energy_range[1]))
        primary_count = np.sum((energies > primary_energy_range[0]) & (energies < primary_energy_range[1]))
        dsr = ds_count / primary_count
        
        # Bin the energies into bins
        hist, edges = np.histogram(energies, bins=n_bins)
        
        # Calculate plasma temperature
        # Find FWHM of 14.1 MeV peak
        # From J A Frenje 2020 Plasma Phys. Control. Fusion 62 023001
        fwhm = self._get_fwhm(hist, edges)
        m_rat = 5.0 # sum of neutron plus alpha mass divided by neutron mass
        plasma_temperature = 9e-5 * m_rat / self.spectrometer.reference_energy * (fwhm * 1000)**2
        
        return dsr, plasma_temperature, fwhm, dsr_energy_range, primary_energy_range, energies

    def _get_fwhm(self, hist, edges):
        """
        Get full width at half maximum (FWHM) of largest peak of a histogram.
        """
        peak_idx = np.argmax(hist)
        half_max = hist[peak_idx] / 2.0
        
        # Find leftmost and rightmost crossings
        left_indices = np.where(hist[:peak_idx] >= half_max)[0]
        right_indices = np.where(hist[peak_idx:] >= half_max)[0] + peak_idx
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0.0
        left_idx = left_indices[0]
        right_idx = right_indices[-1]
        return edges[right_idx] - edges[left_idx]
        
    def _get_neutron_spectrum(self) -> np.ndarray:
        """
        Get neutron spectrum based on the x position of the output beam.
        
        Returns:
            Neutron spectrum
        """
        # Convert the x positions to neutron energies based on the offset curve
        x_positions = self.spectrometer.output_beam[:, 0]
        
        # Load comprehensive performance curve
        performance_df = pd.read_csv(f'{self.spectrometer.figure_directory}/{self.performance_curve_file}')
        input_energies = performance_df['energy [MeV]']
        position_mean = performance_df['position mean [m]']
        position_std = performance_df['position std [m]']
        
        # Interpolate to get the energies for the x positions
        energies = np.interp(x_positions, position_mean, input_energies)
        
        return energies
    
    def get_proton_density_map(
        self, 
        dx: float = 0.005, 
        dy: float = 0.005
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
        
        # Calculate cell area in cm^2
        cell_area_cm2 = (dx * 100) * (dy * 100)  # Convert m^2 to cm^2
        
        # Bin protons into grid cells
        for x_pos, y_pos in zip(x_positions, y_positions):
            # Convert coordinates to grid indices
            x_idx = int((x_pos - x_min) / dx)
            y_idx = int((y_pos - y_min) / dy)
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, density.shape[1] - 1))
            y_idx = max(0, min(y_idx, density.shape[0] - 1))
            
            density[y_idx, x_idx] += 1
            
        # Convert to protons/cm^2/source_proton
        total_protons = len(self.spectrometer.output_beam)
        density = density / (cell_area_cm2 * total_protons)
        
        return density, X_mesh, Y_mesh