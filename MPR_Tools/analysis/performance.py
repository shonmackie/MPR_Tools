"""Performance analysis methods for MPR spectrometer."""

from __future__ import annotations

from concurrent.futures import Executor
from typing import Dict, Tuple, Optional, Union
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
    
    @staticmethod
    def fwfm(data: np.ndarray, fractional_max, bandwidth_method: str | float = "scott", _recursions=0) -> tuple[float, float]:
        """
        Estimate the full-width fractional-max (FWFM) of a 1D distribution using a KDE.

        Parameters
        ----------
        data      : 1D array of position samples
        fractional_max : Fractional position at which to estimate the full width (e.g. 0.5 for FWHM)
        bandwidth_method : KDE bandwidth — "scott", "silverman", or a sigma as a float in data units
        _recursions: The number of times this function has called itself

        Returns
        -------
        FWFM as a float, and the center of the fractional-max interval as a float
        """
        data = np.asarray(data, dtype=float)

        if isinstance(bandwidth_method, (int, float)):
            bandwidth_method = bandwidth_method / np.std(data)
        kde = gaussian_kde(data, bw_method=bandwidth_method)

        bandwidth_factor = kde.factor
        bandsigma = bandwidth_factor * np.std(data)
        bandwidth = 2*np.sqrt(2*np.log(2)) * bandsigma
        
        x = np.linspace(data.min(), data.max(), round(5 * (data.max() - data.min()) / bandwidth))
        y = kde(x)

        # Find the most extreme data points where y >= y_cutoff
        y_cutoff = y.max() * fractional_max
        roots = UnivariateSpline(x, y - y_cutoff, s=0).roots()
        if y[0] >= y_cutoff:
            lower = x[0]
        else:
            lower = roots[0]
        if y[-1] >= y_cutoff:
            upper = x[-1]
        else:
            upper = roots[-1]
            
        full_width = upper - lower
        position = (lower + upper)/2
        
        # If the kernel seems like it could be smaller
        if full_width < 3*bandwidth and _recursions < 5:
            # Reduce the kernel size to aim for bandwidth < full_width/3
            reduction = 4*bandwidth/full_width
            return PerformanceAnalyzer.fwfm(
                data, fractional_max, bandwidth_method=bandsigma/reduction, _recursions=_recursions + 1)
        
        else:
            return full_width, position

    def analyze_monoenergetic_performance(
        self,
        incident_energy: float,
        delta_energy: float = 0.05,
        num_recoil_particles: int = 10000,
        fractional_max: float = 0.5,
        spectrometer: Optional[MPRSpectrometer] = None,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        map_order: int = 5,
        verbose: bool = False,
        executor: Optional[Executor] = None,
        max_workers: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Analyze spectrometer performance for monoenergetic incident particles.
        
        Args:
            incident_energy: Incident particle energy in MeV
            delta_energy: Percentage deviation from target energy for resolution calculation
            num_recoil_particles: Number of recoil particles to simulate
            fractional_max: Fractional position at which to estimate the full width (e.g. 0.5 for FWHM)
            spectrometer: MPRSpectrometer to analyze (defaults to self.spectrometer)
            include_kinematics: Include kinematic energy transfer
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            map_order: Order of transfer map to apply (1-5 typically)
            verbose: Print detailed results
            executor: Pool of workers to use (if None, we will make our own)
            max_workers: Maximum number of worker processes (None for CPU count)
            
        Returns:
            Tuple of (position_mean in m, std_deviation in m, energy_resolution in keV, dispersion in m/MeV)
        """
        if spectrometer is None:
            spectrometer = self.spectrometer

        foil_name = spectrometer.conversion_foil.foil_material
        print(f'\nAnalyzing {foil_name} performance for {incident_energy:.3f} MeV monoenergetic incident particles...')
        
        # Helper function for generating recoil positions mean and std
        def _get_positions(energy: float, num_recoils: int) -> Tuple[float, float]:
            spectrometer.generate_monte_carlo_rays(
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
                map_order=map_order, save_beam=False, executor=executor, max_workers=max_workers)
            positions = spectrometer.output_beam[:, 0]
            position_width, position_mean = PerformanceAnalyzer.fwfm(positions, fractional_max=fractional_max)
            return position_mean, position_width
        
        # Analyze focal plane distribution of target energy +/- delta
        E_low = incident_energy * (1 - delta_energy)
        E_high = incident_energy * (1 + delta_energy)
        # To save compute time, since we're only interested in the mean, use less recoils
        position_mean_low, position_width_low = _get_positions(E_low, num_recoil_particles // 10)
        position_mean_high, position_width_high = _get_positions(E_high, num_recoil_particles // 10)
        
        # Analyze focal plane distribution of target energy beamlet
        position_mean_0, position_width_0 = _get_positions(incident_energy, num_recoil_particles)

        position_means = np.r_[position_mean_low, position_mean_0, position_mean_high]
        energies = np.r_[E_low, incident_energy, E_high]

        dispersion = np.gradient(position_means, energies)[1]

        energy_resolution = 1000 / (dispersion / position_width_0) if position_width_0 > 0 else 0 # keV

        if verbose:
            print('Ion Optical Image Parameters:')
            print(f'  Mean position [cm]: {position_mean_0 * 100:.3f}')
            print(f'  fwfm [cm]: {position_width_0 * 100:.3f}')
            print(f'  Energy resolution [keV]: {energy_resolution:.2f}')
        
        return position_mean_0, position_width_0, energy_resolution, dispersion
    
    def generate_performance_curve(
        self,
        num_energies: int = 40,
        num_recoils_per_energy: int = 10000,
        num_efficiency_samples: int = 10000,
        fractional_max: float = 0.5,
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
            fractional_max: Fraction of maximum of spatial peak to use for resolution calculation (defaults to 0.5 for FWHM)
            include_kinematics: Include kinematic effects
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            output_filename: Name for output data file
            reset: Whether to regenerate the dataset rather than loading an existing one
            executor: Pool of workers to use (if None, we will make our own)
            max_workers: Maximum number of worker processes (None for CPU count)
            
        Returns:
            Pandas dataframe containing energies in MeV, position (center of fractional-max interval) in m, positions_width in m, energy_resolutions in keV, total_efficiencies, foil_species
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
                positions_width = np.zeros_like(energies)
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
                    positions_width[i], positions_mean[i] = PerformanceAnalyzer.fwfm(positions, fractional_max=fractional_max)
                    
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
                energy_resolutions = positions_width/gradients * 1000 # keV

                # Create DataFrame for this foil
                foil_df = pd.DataFrame({
                    'foil': foil_name,
                    'energy [MeV]': energies,
                    'position [m]': positions_mean,
                    'position width [m]': positions_width,
                    'fractional max': fractional_max,
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

    def build_response_matrix(
        self,
        energy_grid: np.ndarray,
        num_recoils_per_energy: int = 10000,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        output_filename: Optional[str] = None,
        reset: bool = True,
        executor: Optional[Executor] = None,
        max_workers: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Build the instrument response matrix for each foil.

        Returns a dict mapping foil material name to a response matrix R of shape
        (n_energies, n_channels).  R[i, k] is the expected signal in hodoscope channel
        k per foil-face incident particle of energy energy_grid[i] — pure instrument physics,
        independent of source geometry. Foil efficiency and y-acceptance are applied
        inside get_recoil_x_map.

        To convert to "per source particle" multiply R by foil_solid_angle_fraction.
        To recover total source yield after unfolding, divide the summed recovered
        spectrum by foil_solid_angle_fraction:
            Y = f_recovered.sum() / spectrometer.foil_solid_angle_fraction

        Energies outside a foil's acceptance range contribute zero rows for that foil.

        Files are saved as <data_directory>/response_matrix_<foil>.npy (or, if
        output_filename is supplied, <output_filename>_<foil>.npy).

        Args:
            energy_grid: 1-D array of incident particle energies [MeV] at which to evaluate R.
            num_recoils_per_energy: Monte Carlo rays per energy point.
            include_kinematics: Pass through to generate_monte_carlo_rays.
            include_stopping_power_loss: Pass through to generate_monte_carlo_rays.
            output_filename: Base path for .npy cache files (foil name is appended).
                             Defaults to <data_directory>/response_matrix.
            reset: If True, regenerate and save.  If False, load from file.
            executor: Worker pool to use (if None, a fresh pool is created).
            max_workers: Maximum worker processes (None -> CPU count).

        Returns:
            Dict mapping foil material name -> np.ndarray of shape (n_energies, n_channels).
            For dual-foil setups the dict has two keys, one per foil.
        """
        # Determine spectrometers to analyze (mirrors generate_performance_curve)
        spectrometers = [self.spectrometer]
        if hasattr(self, 'dual_spectrometer'):
            spectrometers.append(self.dual_spectrometer)

        base = output_filename if output_filename is not None else f'{self.spectrometer.data_directory}/response_matrix'

        def _path(key):
            return f'{base}_{key}.npy'

        n_energies = len(energy_grid)

        # Build a work list: each entry is (spec, tqdm_label, [(key, hodoscope), ...]).
        # The simulation runs once per (spec, energy); binnings lists which (key, hodoscope)
        # pairs to fill from that simulation result.
        work = [
            (spec, spec.conversion_foil.foil_material,
             [(spec.conversion_foil.foil_material, spec.hodoscope)])
            for spec in spectrometers
        ]

        all_keys = [key for _, _, binnings in work for key, _ in binnings]

        # Load from file if not resetting; otherwise build empty matrices to fill in
        if not reset:
            matrices = {}
            for key in all_keys:
                matrices[key] = np.load(_path(key))
                print(f'Response matrix {key} loaded from {_path(key)}')
            return matrices

        matrices = {
            key: np.zeros((n_energies, hodo.total_channels))
            for _, _, binnings in work
            for key, hodo in binnings
        }

        for spec, label, binnings in work:
            print(f'\nBuilding response matrix for {label}...')
            for i, energy in enumerate(tqdm(energy_grid, desc=label)):
                if energy < spec.min_incident_energy or energy > spec.max_incident_energy:
                    continue
                spec.generate_monte_carlo_rays(
                    np.array([energy]),
                    np.array([1.0]),
                    num_recoils_per_energy,
                    include_kinematics,
                    include_stopping_power_loss,
                    save_beam=False,
                    executor=executor,
                    max_workers=max_workers,
                )
                spec.apply_transfer_map(
                    save_beam=False,
                    executor=executor,
                    max_workers=max_workers,
                )
                for key, hodoscope in binnings:
                    signal, _, _ = self.get_recoil_x_map(spectrometer=spec, hodoscope=hodoscope)
                    matrices[key][i, :] = signal

        for key, R in matrices.items():
            np.save(_path(key), R)
            print(f'Response matrix {key} saved to {_path(key)}')
        return matrices

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
        
    def _get_foil_efficiency(self, energies: np.ndarray, spectrometer: Optional[MPRSpectrometer] = None) -> np.ndarray:
        """
        Get the foil efficiency for a given set of incident particle energies.

        Args:
            energies: Incident particle energies in MeV.
            spectrometer: Spectrometer whose foil efficiency to look up.  Defaults to
                          self.spectrometer.  When the performance CSV contains data for
                          multiple foils (dual-foil case), only the rows matching this
                          foil's material are used.
        """
        spec = spectrometer if spectrometer is not None else self.spectrometer
        performance_df = self._load_performance_curve()
        if performance_df is not None:
            if 'foil' in performance_df.columns:
                foil_name = spec.conversion_foil.foil_material
                performance_df = performance_df[performance_df['foil'] == foil_name]
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
        particle_yield: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the density of recoil particle impact sites and detector response (if available) in the focal plane.

        Args:
            dx: X-direction resolution in cm
            dy: Y-direction resolution in cm
            particle_yield, optional: Input particle yield
            
        Returns:
            Tuple of (density_map, response_map, X_meshgrid, Y_meshgrid)
        """
        if len(self.spectrometer.output_beam) == 0:
            raise ValueError("No output beam data available. Run apply_transfer_map() first.")
        
        x_positions = self.spectrometer.output_beam[:, 0] * 100
        y_positions = self.spectrometer.output_beam[:, 2] * 100
        input_energies = self.spectrometer.input_beam[:, 6]
        output_energies_MeV = self.spectrometer.reference_energy * (1 + self.spectrometer.output_beam[:, 5])

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
        sensitivities = self.spectrometer.hodoscope.get_detector_response(
            energies=output_energies_MeV,
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
                response_map[y_idx, x_idx] += foil_efficiencies[i] * sensitivities[i]
        
        # Convert to recoils/cm^2/source_proton
        total_recoils = len(self.spectrometer.output_beam)
        density_map /= (cell_area_cm2 * total_recoils)
        response_map /= (cell_area_cm2 * total_recoils)
        
        # Add source-to-foil geometric factor
        if self.spectrometer.foil_geometric_factor:
            density_map *= self.spectrometer.foil_geometric_factor
            response_map *= self.spectrometer.foil_geometric_factor
            
        # Add yield multiplier
        if particle_yield:
            density_map *= particle_yield
            response_map *= particle_yield
        
        # Save the density map as csv
        density_data = np.column_stack((X_mesh.flatten(), Y_mesh.flatten(), density_map.flatten(), response_map.flatten()))
        density_df = pd.DataFrame(density_data, columns=['X', 'Y', 'Density', 'Sensitivity'])
        density_df.to_csv(f'{self.spectrometer.figure_directory}/particle_density_map.csv', index=False)
        
        return density_map, response_map, X_mesh, Y_mesh
    
    def get_recoil_x_map(
        self,
        time_gate_percentiles: Tuple[float, float] = (0, 100),
        spectrometer: Optional[MPRSpectrometer] = None,
        hodoscope=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bin the recoil beam into 1-D x channels and compute signal, y-coverage, and per-channel
        signal arrival-time windows.

        Returns signal in [particles / foil-face neutron] — pure instrument response,
        independent of source geometry.  To obtain physical counts, multiply the result
        by foil_solid_angle_fraction * particle_yield after calling this method.

        The y-acceptance of each channel is determined entirely by the hodoscope: particles are
        accepted when |y - hodoscope.y_center| <= channel_height/2.  In a dual-foil setup each
        hodoscope's y_center and channel_height together define its physical half of the detector
        with no additional restriction parameter required.

        Per-channel time windows are computed when hodoscope.use_time_gating is True.  For each
        channel the window spans the requested percentile range of detector arrival times of
        signal particles accepted into that channel.  When use_time_gating is False, the returned
        channel_time_windows array is filled with NaN.

        Args:
            time_gate_percentiles: (low_percentile, high_percentile) pair defining the signal
                                   time window per channel.  Defaults to (0, 100) to accept
                                   all arrival times.

        Returns:
            Tuple of (signal_per_bin, coverage_per_bin, channel_time_windows) where
            signal_per_bin [particles/source or MeV/source],
            coverage_per_bin [0-1],
            channel_time_windows of shape (n_channels, 2) [s] — columns are [t_min, t_max].
        """
        spec = spectrometer if spectrometer is not None else self.spectrometer

        if len(spec.output_beam) == 0:
            raise ValueError("No output beam data available. Run apply_transfer_map() first.")

        hodoscope = hodoscope if hodoscope is not None else spec.hodoscope
        x_positions = spec.output_beam[:, 0] * 100  # m to cm
        y_positions = spec.output_beam[:, 2] * 100  # m to cm
        input_energies = spec.input_beam[:, 6]
        output_energies_MeV = spec.reference_energy * (1 + spec.output_beam[:, 5])
        total_particles = len(x_positions)

        # Determine bin edges, channel heights, and detector y-center
        bin_edges_cm = hodoscope.channel_edges * 100   # m to cm
        bin_heights_cm = hodoscope.channel_heights * 100  # m to cm
        y_center_cm = hodoscope.y_center * 100  # m to cm

        n_bins = len(bin_edges_cm) - 1

        # Per-particle weights
        foil_efficiencies = self._get_foil_efficiency(input_energies, spectrometer=spec)
        sensitivities = hodoscope.get_detector_response(
            energies=output_energies_MeV,
            particle=spec.conversion_foil.particle
        )
        weights = foil_efficiencies * sensitivities

        # Bin particles into x channels; track total and within-y-acceptance separately.
        # Simultaneously compute per-channel signal arrival-time windows when time gating is enabled.
        # np.digitize returns 1-based indices; subtract 1 to get 0-based bin indices.
        bin_indices = np.digitize(x_positions, bin_edges_cm) - 1  # -1 and n_bins are out of range
        signal_per_bin = np.zeros(n_bins)
        total_per_bin = np.zeros(n_bins)
        channel_time_windows = np.full((n_bins, 2), np.nan)

        for b in range(n_bins):
            in_bin = bin_indices == b

            # Accept particles within [y_center - height/2, y_center + height/2].
            # For dual-foil, each hodoscope's y_center and channel_height place it in its
            # physical half of the detector.
            accepted = in_bin & (np.abs(y_positions - y_center_cm) <= bin_heights_cm[b] / 2)

            total_per_bin[b] = np.sum(weights[in_bin])
            signal_per_bin[b] = np.sum(weights[accepted])

            # Compute the signal arrival-time window for this channel from the percentile range
            # of detector arrival times of all accepted signal particles.
            if hodoscope.use_time_gating:
                arrival_times = spec.output_beam[:, 4]
                times_in_channel = arrival_times[accepted]
                if len(times_in_channel) > 0:
                    channel_time_windows[b, 0] = np.percentile(times_in_channel, time_gate_percentiles[0])
                    channel_time_windows[b, 1] = np.percentile(times_in_channel, time_gate_percentiles[1])

        # Normalise to per foil-face neutron — pure instrument response.
        signal_per_bin /= total_particles
        total_per_bin /= total_particles

        if self.spectrometer.foil_geometric_factor:
            signal_per_bin *= self.spectrometer.foil_geometric_factor
            total_per_bin *= self.spectrometer.foil_geometric_factor

        # Yield scaling
        if particle_yield:
            signal_per_bin *= particle_yield
            total_per_bin *= particle_yield

        coverage_per_bin = np.where(total_per_bin > 0, signal_per_bin / total_per_bin, 0.0)

        return signal_per_bin, coverage_per_bin, channel_time_windows