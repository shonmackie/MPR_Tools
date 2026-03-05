"""Performance analysis methods for MPR spectrometer."""

from __future__ import annotations

from concurrent.futures import Executor
from typing import Tuple, Optional, Union, Literal
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

from ..core.spectrometer import MPRSpectrometer
from ..core.dual_foil_spectrometer import DualFoilSpectrometer

class PerformanceAnalyzer:
    """Handles performance analysis for MPR spectrometer."""
    
    def __init__(self, spectrometer: Union[MPRSpectrometer, DualFoilSpectrometer]):
        if isinstance(spectrometer, MPRSpectrometer):
            self.spectrometer = spectrometer
        elif isinstance(spectrometer, DualFoilSpectrometer):
            self.spectrometer = spectrometer.spec_ch2
            self.dual_spectrometer = spectrometer.spec_cd2
        else:
            raise ValueError(f"Unsupported spectrometer type: {type(spectrometer)}")
    
    def fwhm(self, data: np.ndarray, bandwidth: str | float = "scott") -> float:
        """
        Estimate the FWHM of a 1D distribution using a KDE.

        Parameters
        ----------
        data      : 1D array of position samples
        bandwidth : KDE bandwidth — "scott", "silverman", or a float in data units

        Returns
        -------
        FWHM as a float
        """
        data = np.asarray(data, dtype=float)

        bw = bandwidth / np.std(data) if isinstance(bandwidth, (int, float)) else bandwidth
        kde = gaussian_kde(data, bw_method=bw)

        x = np.linspace(data.min(), data.max(), 1024)
        y = kde(x)

        half_max = y.max() / 2
        roots = UnivariateSpline(x, y - half_max, s=0).roots()

        if len(roots) < 2:
            raise RuntimeError("Could not find two half-max crossings. Try a smaller bandwidth.")

        return roots[-1] - roots[0]

    def analyze_monoenergetic_performance(
        self,
        incident_energy: float,
        delta_energy: float = 0.05,
        num_recoil_particles: int = 10000,
        spectrometer: Optional[MPRSpectrometer] = None,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        verbose: bool = False,
        executor: Optional[Executor] = None,
        max_workers: Optional[int] = None,
    ) -> Tuple[float, float, float, float, float]:
        """
        Analyze spectrometer performance for monoenergetic incident particles.
        
        Args:
            incident_energy: Incident particle energy in MeV
            delta_energy: Percentage deviation from target energy for resolution calculation
            num_recoil_particles: Number of recoil particles to simulate
            spectrometer: MPRSpectrometer to analyze (defaults to self.spectrometer)
            include_kinematics: Include kinematic energy transfer
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            verbose: Print detailed results
            executor: Pool of workers to use (if None, we will make our own)
            max_workers: Maximum number of worker processes (None for CPU count)
            
        Returns:
            Tuple of (mean_position in m, std_deviation in m, fwhm in m, energy_resolution in keV, dispersion in m/MeV)
        """
        if spectrometer is None:
            spectrometer = self.spectrometer

        foil_name = spectrometer.conversion_foil.foil_material
        print(f'\nAnalyzing {foil_name} performance for {incident_energy:.3f} MeV monoenergetic incident particles...')
        
        # Helper function for generating recoil positions mean and std
        def _get_positions(energy: float, num_recoils: int) -> Tuple[float, float]:
            self.spectrometer.generate_monte_carlo_rays(
                np.array([energy]), 
                np.array([1.0]), 
                num_recoils,
                include_kinematics, 
                include_stopping_power_loss,
                save_beam=False,
                executor=executor,
                max_workers=max_workers,
            )
            spectrometer.apply_transfer_map(
                map_order=5, save_beam=False, executor=executor, max_workers=max_workers)
            x_positions = spectrometer.output_beam[:, 0]
            mean_position, std_deviation = norm.fit(x_positions)
            return mean_position, std_deviation
        
        # Analyze focal plane distribution of target energy +/- delta
        E_low = incident_energy * (1 - delta_energy)
        E_high = incident_energy * (1 + delta_energy)
        # To save compute time, since we're only interested in the mean, use less recoils
        mean_position_low, std_deviation_low = _get_positions(E_low, num_recoil_particles // 10)
        mean_position_high, std_deviation_high = _get_positions(E_high, num_recoil_particles // 10)
        
        # Analyze focal plane distribution of target energy beamlet
        mean_position_0, std_deviation_0 = _get_positions(incident_energy, num_recoil_particles)
        fwhm_0 = 2 * np.sqrt(2 * np.log(2)) * std_deviation_0

        mean_positions = np.r_[mean_position_low, mean_position_0, mean_position_high]
        energies = np.r_[E_low, incident_energy, E_high]

        dispersion = np.gradient(mean_positions, energies)[1]

        energy_resolution = 1000 / (dispersion / fwhm_0) if fwhm_0 > 0 else 0 # keV

        if verbose:
            print('Ion Optical Image Parameters:')
            print(f'  Mean position [cm]: {mean_position_0 * 100:.3f}')
            print(f'  Standard deviation [cm]: {std_deviation_0 * 100:.3f}')
            print(f'  FWHM [cm]: {fwhm_0 * 100:.3f}')
            print(f'  Energy resolution [keV]: {energy_resolution:.2f}')
        
        return mean_position_0, std_deviation_0, fwhm_0, energy_resolution, dispersion
    
    def generate_performance_curve(
        self,
        num_energies: int = 40,
        num_recoils_per_energy: int = 10000,
        num_efficiency_samples: int = 10000,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        output_filename: Optional[str] = None,
        reset: bool = True,
        executor: Optional[Executor] = None,
        max_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate comprehensive performance analysis including location, resolution, and efficiency.
        If a dual-foil spectrometer is used, analyzes both foils.

        Args:
            num_energies: Number of energy points to simulate
            num_recoils_per_energy: Number of recoil events per energy point for location/resolution
            num_efficiency_samples: Number of samples for efficiency calculation
            include_kinematics: Include kinematic effects
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            output_filename: Name for output data file
            reset: Whether to regenerate the dataset rather than loading an existing one
            executor: Pool of workers to use (if None, we will make our own)
            max_workers: Maximum number of worker processes (None for CPU count)
            
        Returns:
            Tuple of (energies in MeV, positions_mean in m, positions_fwhm in m, energy_resolutions in keV, total_efficiencies, foil_species)
        """
        print('\nGenerating comprehensive performance analysis...')

        # Save comprehensive data
        if output_filename == None:
            output_filename = f'{self.spectrometer.data_directory}/comprehensive_performance.csv'
        else:
            output_filename = f'{self.spectrometer.data_directory}/{output_filename}'

        if reset:
            # Determine spectrometers to analyze
            spectrometers = [self.spectrometer]
            if hasattr(self, 'dual_spectrometer'):
                spectrometers.append(self.dual_spectrometer)

            all_dfs = []

            for spec in spectrometers:
                foil_name = spec.conversion_foil.foil_material
                print(f'\nAnalyzing {foil_name} foil...')

                # Energy range
                energies = np.linspace(spec.min_incident_energy, spec.max_incident_energy, num_energies)

                positions_mean = np.zeros_like(energies)
                positions_fwhm = np.zeros_like(energies)
                gradients = np.zeros_like(energies)
                energy_resolutions = np.zeros_like(energies)
                scattering_efficiencies = np.zeros_like(energies)
                geometric_efficiencies = np.zeros_like(energies)
                total_efficiencies = np.zeros_like(energies)

                for i, energy in enumerate(tqdm(energies, desc=f'Calculating {foil_name} performance...')):
                    # Calculate location and resolution from monoenergetic analysis
                    spec.generate_monte_carlo_rays(np.array([energy]), 
                        np.array([1.0]), 
                        num_recoils_per_energy,
                        include_kinematics, 
                        include_stopping_power_loss,
                        save_beam=False,
                        executor=executor,
                        max_workers=max_workers,)
                    spec.apply_transfer_map(save_beam=False,
                        executor=executor,
                        max_workers=max_workers)
                    
                    positions = spec.output_beam[:,0]
                    positions_mean[i] = np.mean(positions)
                    positions_fwhm[i] = self.fwhm(positions)
                    
                    # Calculate efficiency for this energy
                    scattering_efficiency, geometric_efficiency, total_efficiency = spec.conversion_foil.calculate_efficiency(
                        energy,
                        num_samples=num_efficiency_samples,
                        executor=executor,
                        max_workers=max_workers,
                    )
                    scattering_efficiencies[i] = scattering_efficiency
                    geometric_efficiencies[i] = geometric_efficiency
                    total_efficiencies[i] = total_efficiency
                
                # calculate dispersion gradient and energy resolution from monoenergetic beamlet results
                gradients = np.gradient(positions_mean, energies)  # m/MeV
                energy_resolutions = positions_fwhm/gradients * 1000 # keV

                # Create DataFrame for this foil
                foil_df = pd.DataFrame({
                    'foil': foil_name,
                    'energy [MeV]': energies,
                    'position mean [m]': positions_mean,
                    'position fwhm [m]': positions_fwhm,
                    'gradient [m/MeV]': gradients,
                    'resolution [keV]': energy_resolutions,
                    'scattering efficiency': scattering_efficiencies,
                    'geometric efficiency': geometric_efficiencies,
                    'total efficiency': total_efficiencies
                })
                all_dfs.append(foil_df)

            # Combine all foil DataFrames
            df = pd.concat(all_dfs, ignore_index=True)
            df.to_csv(output_filename, index=False)

            print(f'Comprehensive performance data saved to {output_filename}')

        else:
            df = pd.read_csv(output_filename)

        return df
    
    def get_plasma_parameters(
        self,
        dsr_energy_range: Tuple[float, float] = (10, 12),
        primary_energy_range: Tuple[float, float] = (13, 15)
    ) -> Tuple[float, float, float, float, Tuple[float, float], Tuple[float, float], np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Get plasma parameters from the spectrometer object, assuming the incident particles are neutrons.
        
        Args:
            n_bins:
                Number of bins to use for histogram
            dsr_energy_range:
                Energy range for DSR in MeV
            primary_energy_range:
                Energy range for primary neutrons in MeV
            
        Returns:
            Tuple of (dsr, plasma_temperature, fwhm, dsr_energy_range, primary_energy_range, energies, energies_std, response, background)
        """
        response, background, energies, energies_std = self._get_incident_spectrum()
        
        # Calculate dsr
        ds_idx = (energies > dsr_energy_range[0]) & (energies < dsr_energy_range[1])
        primary_idx = (energies > primary_energy_range[0]) & (energies < primary_energy_range[1])
        dsr = np.sum(response[ds_idx]) / np.sum(response[primary_idx])
        # TODO: Add dsr uncertainty
        
        # Calculate plasma temperature
        # Find FWHM of 14.1 MeV peak
        # From J A Frenje 2020 Plasma Phys. Control. Fusion 62 023001
        left_edge, right_edge = self._get_fwhm(response, energies)
        fwhm = (right_edge - left_edge)
        m_rat = 5.0 # sum of neutron plus alpha mass divided by neutron mass
        plasma_temperature = 9e-5 * m_rat / self.spectrometer.reference_energy * (fwhm * 1000)**2
        
        return dsr, plasma_temperature, left_edge, right_edge, dsr_energy_range, primary_energy_range, energies, energies_std, response, background

    def _get_fwhm(self, hist, edges) -> Tuple[float, float]:
        """
        Get full width at half maximum (FWHM) of largest peak of a histogram.
        """
        peak_idx = np.argmax(hist)
        half_max = hist[peak_idx] / 2.0
        
        # Find leftmost and rightmost crossings
        left_indices = np.where(hist[:peak_idx] >= half_max)[0]
        right_indices = np.where(hist[peak_idx:] >= half_max)[0] + peak_idx + 1
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0.0, 0.0
        left_idx = left_indices[0]
        right_idx = right_indices[-1]
        return edges[left_idx], edges[right_idx]
    
    def _load_performance_curve(self, performance_curve_file: Optional[str] = None) -> Union[pd.DataFrame, None]:
        """
        Loads comprehensive performance curve for analysis
        """
        # Load comprehensive performance curve
        performance_curve_file = performance_curve_file or 'comprehensive_performance.csv'
        try:
            performance_df = pd.read_csv(f'{self.spectrometer.data_directory}/{performance_curve_file}')
            return performance_df
        except:
            warnings.warn(f'Performance curve file {performance_curve_file} not found. May need to generate first.', RuntimeWarning)
            return
        
    def _get_incident_spectrum(
        self,
        dx: float = 0.5,
        dy: float = 0.5,
        foil_distance: Optional[float] = None,
        particle_yield: Optional[float] = None
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Get incident particle mean energy based on the binned response.
        
        Returns:
            response: np.ndarray of detector response values
            background: float of total background contribution
            incident_energies: np.ndarray of incident energies
            incident_energies_std: np.ndarray of incident energy uncertainties
        """
        recoil_density_map, response_map, X_mesh, Y_mesh = self.get_recoil_density_map(
            dx, dy, foil_distance, particle_yield
        )
        
        # Sum response map along y to get total response vs x
        # In units of response/cm(-source)
        response_values = np.sum(response_map, axis=0) * dy
        x_positions = X_mesh[0, :] / 100  # Convert to meters
        
        # Total background is in response/cm^2-source
        # Assume background is uniform across detector
        total_background = self.spectrometer.hodoscope.get_total_background()
        # Integrate background over y
        total_background *= X_mesh.shape[0] * dy
        if particle_yield:
            total_background *= particle_yield
        
        # Load comprehensive performance curve
        performance_df = self._load_performance_curve()
        if performance_df is None:
            raise ValueError('Performance curve file not found. May need to generate first.')
        incident_energies = performance_df['energy [MeV]']
        position_mean = performance_df['position mean [m]']
        position_std = performance_df['position std [m]']
        gradient = performance_df['gradient [m/MeV]']
        
        # Interpolate to get the energies for the x positions
        energies = np.interp(x_positions, position_mean, incident_energies)
        # Calculate energy uncertainty sigma_E = sigma_x / |dx/dE|
        energies_std = np.interp(x_positions, position_mean, position_std / np.abs(gradient))
        
        return response_values, total_background, energies, energies_std
    
    def _get_foil_efficiency(self, energies: np.ndarray) -> np.ndarray:
        """
        Get the foil efficiency for a given set of incident particle energies.
        """
        performance_df = self._load_performance_curve()
        if performance_df is not None:
            incident_energies = performance_df['energy [MeV]']
            total_efficiencies = performance_df['total efficiency']
            
            # Interpolate to get the efficiencies for the incident energies
            efficiencies = np.interp(energies, incident_energies, total_efficiencies)
        else:
            efficiencies = np.ones(len(energies))
        
        return efficiencies
    
    def get_recoil_density_map(
        self,
        dx: float = 0.5, 
        dy: float = 0.5,
        foil_distance: Optional[float] = None,
        particle_yield: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the density of recoil particle impact sites and detector response (if available) in the focal plane.
        
        Args:
            dx: X-direction resolution in cm
            dy: Y-direction resolution in cm
            foil_distance, optional: Distance between foil and target in meters
            particle_yield, optional: Input particle yield
            
        Returns:
            Tuple of (density_map, response_map, X_meshgrid, Y_meshgrid)
        """
        if len(self.spectrometer.output_beam) == 0:
            raise ValueError("No output beam data available. Run apply_transfer_map() first.")
        
        x_positions = self.spectrometer.output_beam[:, 0] * 100
        y_positions = self.spectrometer.output_beam[:, 2] * 100
        input_energies = self.spectrometer.input_beam[:, 4]
        output_energies = self.spectrometer.output_beam[:, 4]
        
        # Define grid boundaries
        x_min, x_max = np.min(x_positions), np.max(x_positions)
        y_min, y_max = np.min(y_positions), np.max(y_positions)
        
        # Create coordinate arrays
        x_coords = np.linspace(x_min, x_max, int((x_max - x_min) / dx) + 1)
        y_coords = np.linspace(y_min, y_max, int((y_max - y_min) / dy) + 1)
        
        print(f"Grid bounds: X=[{x_min:.4f}, {x_max:.4f}], Y=[{y_min:.4f}, {y_max:.4f}]")
        
        # Create meshgrids
        X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)
        density_map = np.zeros_like(X_mesh)
        
        # Calculate cell area in cm^2
        cell_area_cm2 = dx*dy
        
        # Load performance curve to get foil efficiency and aperture solid angle
        foil_efficiencies = self._get_foil_efficiency(input_energies)
            
        # If detector is used, calculate the sensitivity for the incident recoil particle
        response_map = np.zeros_like(density_map)
        sensitivity_efficiencies = self.spectrometer.hodoscope.get_detector_response(
            energies=output_energies,
            particle=self.spectrometer.conversion_foil.particle
        )
        
        # Bin recoils into grid cells
        for i, (x_pos, y_pos) in enumerate(zip(x_positions, y_positions)):
            # Convert coordinates to grid indices
            x_idx = int((x_pos - x_min) / dx)
            y_idx = int((y_pos - y_min) / dy)
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, density_map.shape[1] - 1))
            y_idx = max(0, min(y_idx, density_map.shape[0] - 1))
            
            # Add efficiency to cell
            density_map[y_idx, x_idx] += foil_efficiencies[i]
            
            # If detector is used, weight by sensitivity efficiency
            if self.spectrometer.hodoscope.detector_used:
                response_map[y_idx, x_idx] += foil_efficiencies[i] * sensitivity_efficiencies[i]
        
        # Convert to recoils/cm^2/source_proton
        total_recoils = len(self.spectrometer.output_beam)
        density_map /= (cell_area_cm2 * total_recoils)
        response_map /= (cell_area_cm2 * total_recoils)
        
        # Calculate foil solid angle fraction
        if foil_distance:
            foil_solid_angle_fraction = self.spectrometer.conversion_foil.foil_radius**2 / (4 * foil_distance**2)
            density_map *= foil_solid_angle_fraction
            response_map *= foil_solid_angle_fraction
            
        # Add yield multiplier
        if particle_yield:
            density_map *= particle_yield
            response_map *= particle_yield
        
        # Save the density map as csv
        density_data = np.column_stack((X_mesh.flatten(), Y_mesh.flatten(), density_map.flatten(), response_map.flatten()))
        density_df = pd.DataFrame(density_data, columns=['X', 'Y', 'Density', 'Sensitivity'])
        density_df.to_csv(f'{self.spectrometer.figure_directory}/particle_density_map.csv', index=False)
        
        return density_map, response_map, X_mesh, Y_mesh
    
    def analyze_response(
        self,
        dx: float = 0.5,
        dy: float = 0.5,
        foil_distance: Optional[float] = None,
        particle_yield: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze the detector response in the focal plane.
        
        Args:
            dx: X-direction resolution in cm
            dy: Y-direction resolution in cm
            foil_distance, optional: Distance between foil and target in meters
            particle_yield, optional: Particle yield
            
        Returns:
            Tuple of x_positions, response_values
        """
        density_map, response_map, X_mesh, Y_mesh = self.get_recoil_density_map(
            dx, dy, foil_distance, particle_yield
        )
        
        # Sum response map along y to get total response vs x
        response_values = np.sum(response_map, axis=0)
        x_positions = X_mesh[0, :]
        
        return x_positions, response_values
        
