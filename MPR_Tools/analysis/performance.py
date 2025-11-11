"""Performance analysis methods for MPR spectrometer."""

from __future__ import annotations
from typing import Tuple, Optional, Union, TYPE_CHECKING
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

if TYPE_CHECKING:
    from ..core.spectrometer import MPRSpectrometer

class PerformanceAnalyzer:
    """Handles performance analysis for MPR spectrometer."""
    
    def __init__(self, spectrometer: MPRSpectrometer):
        self.spectrometer = spectrometer
    
    def analyze_monoenergetic_performance(
        self,
        neutron_energy: float,
        delta_energy: float = 0.05,
        num_hydrons: int = 10000,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        verbose: bool = False
    ) -> Tuple[float, float, float, float]:
        """
        Analyze spectrometer performance for monoenergetic neutrons.
        
        Args:
            neutron_energy: Neutron energy in MeV
            delta_energy: Percentage deviation from target energy for resolution calculation
            num_hydrons: Number of hydrons to simulate
            include_kinematics: Include kinematic energy transfer
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            verbose: Print detailed results
            
        Returns:
            Tuple of (mean_position, std_deviation, fwhm, energy_resolution)
        """
        print(f'\nAnalyzing performance for {neutron_energy:.3f} MeV monoenergetic neutrons...')
        
        # Helper function for generating hydron positions mean and std
        def _get_positions(energy: float, num_hydrons: int) -> Tuple[float, float]:
            self.spectrometer.generate_monte_carlo_rays(
                np.array([energy]), 
                np.array([1.0]), 
                num_hydrons, 
                include_kinematics, 
                include_stopping_power_loss,
                save_beam=False
            )
            self.spectrometer.apply_transfer_map(map_order=5, save_beam=False)
            x_positions = self.spectrometer.output_beam[:, 0]
            mean_position, std_deviation = norm.fit(x_positions)
            return mean_position, std_deviation
        
        # Analyze focal plane distribution of target energy +/- delta
        E_low = neutron_energy * (1 - delta_energy)
        E_high = neutron_energy * (1 + delta_energy)
        # To save compute time, since we're only interested in the mean, use less hydrons
        mean_position_low, std_deviation_low = _get_positions(E_low, num_hydrons // 10)
        mean_position_high, std_deviation_high = _get_positions(E_high, num_hydrons // 10)
        
        # Analyze focal plane distribution of target energy beamlet
        mean_position_0, std_deviation_0 = _get_positions(neutron_energy, num_hydrons)
        fwhm_0 = 2 * np.sqrt(2 * np.log(2)) * std_deviation_0

        mean_positions = np.r_[mean_position_low, mean_position_0, mean_position_high]
        energies = np.r_[E_low, neutron_energy, E_high]

        dispersion = np.gradient(mean_positions, energies)[1]

        energy_resolution = 1000 / (dispersion / fwhm_0) if fwhm_0 > 0 else 0 # keV

        if verbose:
            print('Ion Optical Image Parameters:')
            print(f'  Mean position [cm]: {mean_position_0 * 100:.3f}')
            print(f'  Standard deviation [cm]: {std_deviation_0 * 100:.3f}')
            print(f'  FWHM [cm]: {fwhm_0 * 100:.3f}')
            print(f'  Energy resolution [keV]: {energy_resolution * 1000:.2f}')
        
        return mean_position_0, std_deviation_0, fwhm_0, energy_resolution
    def generate_performance_curve(
        self,
        num_energies: int = 40,
        num_hydrons_per_energy: int = 10000,
        num_efficiency_samples: int = int(1e6),
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        output_filename: Optional[str] = None,
        reset: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate comprehensive performance analysis including location, resolution, and efficiency.
        
        Args:
            num_energies: Number of energy points to simulate
            num_hydrons_per_energy: Number of hydrons per energy point for location/resolution
            num_efficiency_samples: Number of samples for efficiency calculation
            include_kinematics: Include kinematic effects
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            output_filename: Name for output data file
            reset: Whether to regenerate the dataset or load an existing one
            
        Returns:
            Tuple of (energies, positions_mean, positions_std, energy_resolutions, total_efficiencies)
        """
        print('\nGenerating comprehensive performance analysis...')
        
        # Save comprehensive data
        if output_filename == None:
            output_filename = f'{self.spectrometer.figure_directory}/comprehensive_performance.csv'
        
        if reset:
            # Energy range
            energies = np.linspace(self.spectrometer.min_energy, self.spectrometer.max_energy, num_energies)
            
            positions_mean = np.zeros_like(energies)
            positions_std = np.zeros_like(energies)
            energy_resolutions = np.zeros_like(energies)
            scattering_efficiencies = np.zeros_like(energies)
            geometric_efficiencies = np.zeros_like(energies)
            total_efficiencies = np.zeros_like(energies)
            
            for i, energy in enumerate(tqdm(energies, desc='Calculating performance for a range of energies...')):
                # Calculate location and resolution from monoenergetic analysis
                mean_pos, std_dev, fwhm, energy_res = self.analyze_monoenergetic_performance(
                    energy,
                    num_hydrons=num_hydrons_per_energy, 
                    include_kinematics=include_kinematics, 
                    include_stopping_power_loss=include_stopping_power_loss,
                    verbose=False
                )
                positions_mean[i] = mean_pos
                positions_std[i] = std_dev
                energy_resolutions[i] = energy_res
                
                # Calculate efficiency for this energy
                scattering_efficiency, geometric_efficiency, total_efficiency = self.spectrometer.conversion_foil.calculate_efficiency(
                    energy, 
                    num_samples=num_efficiency_samples
                )
                scattering_efficiencies[i] = scattering_efficiency
                geometric_efficiencies[i] = geometric_efficiency
                total_efficiencies[i] = total_efficiency
            
            # Save results to a csv
            df = pd.DataFrame({
                'energy [MeV]': energies,
                'position mean [m]': positions_mean,
                'position std [m]': positions_std,
                'resolution [MeV]': energy_resolutions,
                'scattering efficiency': scattering_efficiencies,
                'geometric efficiency': geometric_efficiencies,
                'total efficiency': total_efficiencies
            })
            df.to_csv(output_filename, index=False)
            
            print(f'Comprehensive performance data saved to {output_filename}')
        
        else:
            df = pd.read_csv(f'{output_filename}')
            energies = df['energy [MeV]'].to_numpy()
            positions_mean = df['position mean [m]'].to_numpy()
            positions_std = df['position std [m]'].to_numpy()
            energy_resolutions = df['resolution [MeV]'].to_numpy()
            scattering_efficiencies = df['scattering efficiency'].to_numpy()
            geometric_efficiencies = df['geometric efficiency'].to_numpy()
            total_efficiencies = df['total efficiency'].to_numpy()
        
        return energies, positions_mean, positions_std, energy_resolutions, total_efficiencies
    
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
    
    def _load_performance_curve(self, performance_curve_file: Optional[str] = None) -> Union[pd.DataFrame, None]:
        """
        Loads comprehensive performance curve for analysis
        """
        # Load comprehensive performance curve
        performance_curve_file = performance_curve_file or 'comprehensive_performance.csv'
        try:
            performance_df = pd.read_csv(f'{self.spectrometer.figure_directory}/{performance_curve_file}')
            return performance_df
        except:
            warnings.warn(f'Performance curve file {performance_curve_file} not found. May need to generate first.', RuntimeWarning)
            return
        
    def _get_neutron_spectrum(self) -> Union[np.ndarray, None]:
        """
        Get neutron spectrum based on the x position of the output beam.
        
        Returns:
            Neutron spectrum
        """
        # Convert the x positions to neutron energies based on the offset curve
        x_positions = self.spectrometer.output_beam[:, 0]
        
        # Load comprehensive performance curve
        performance_df = self._load_performance_curve()
        if performance_df is None:
            raise ValueError('Performance curve file not found. May need to generate first.')
        input_energies = performance_df['energy [MeV]']
        position_mean = performance_df['position mean [m]']
        position_std = performance_df['position std [m]']
        
        # Interpolate to get the energies for the x positions
        # TODO: interpolate with error
        energies = np.interp(x_positions, position_mean, input_energies)
        breakpoint()
        return energies
    
    def get_hydron_density_map(
        self, 
        dx: float = 0.5, 
        dy: float = 0.5,
        foil_distance: Optional[float] = None,
        neutron_yield: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the density of proton impact sites in the focal plane.
        
        Args:
            dx: X-direction resolution in cm
            dy: Y-direction resolution in cm
            foil_distance, optional: Distance between foil and target in meters
            neutron_yield, optional: Neutron yield
            
        Returns:
            Tuple of (density_array, X_meshgrid, Y_meshgrid)
        """
        if len(self.spectrometer.output_beam) == 0:
            raise ValueError("No output beam data available. Run apply_transfer_map() first.")
        
        x_positions = self.spectrometer.output_beam[:, 0] * 100
        y_positions = self.spectrometer.output_beam[:, 2] * 100
        
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
        cell_area_cm2 = dx*dy
        
        # Load performance curve to get foil efficiency and aperture solid angle
        performance_df = self._load_performance_curve()
        if performance_df is not None:
            performance_energies = performance_df['energy [MeV]']
            performance_efficiencies = performance_df['total efficiency']
            
            # Interpolate to get the efficiencies for the input energies
            input_energies = self.spectrometer.input_beam[:, 4]
            input_efficiencies = np.interp(input_energies, performance_energies, performance_efficiencies)
        else:
            input_efficiencies = np.ones(len(self.spectrometer.input_beam))
        
        # Bin protons into grid cells
        for i, (x_pos, y_pos) in enumerate(zip(x_positions, y_positions)):
            # Convert coordinates to grid indices
            x_idx = int((x_pos - x_min) / dx)
            y_idx = int((y_pos - y_min) / dy)
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, density.shape[1] - 1))
            y_idx = max(0, min(y_idx, density.shape[0] - 1))
            
            # Add efficiency to cell
            density[y_idx, x_idx] += input_efficiencies[i]
        
        # Convert to protons/cm^2/source_proton
        total_protons = len(self.spectrometer.output_beam)
        density /= (cell_area_cm2 * total_protons)
        
        # Calculate foil solid angle fraction
        if foil_distance:
            foil_solid_angle_fraction = self.spectrometer.conversion_foil.foil_radius**2 / (4 * foil_distance**2)
            density *= foil_solid_angle_fraction
            
        # Add neutron yield
        if neutron_yield:
            density *= neutron_yield
        
        # Save the density map as csv
        density_data = np.column_stack((X_mesh.flatten(), Y_mesh.flatten(), density.flatten()))
        density_df = pd.DataFrame(density_data, columns=['X', 'Y', 'Density'])
        density_df.to_csv(f'{self.spectrometer.figure_directory}/hydron_density_map.csv', index=False)        
        
        return density, X_mesh, Y_mesh