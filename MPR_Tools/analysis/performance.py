"""Performance analysis methods for MPR spectrometer."""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple, Optional, Union
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

from ..config.constants import ELEMENTARY_CHARGE
from ..core.spectrometer import MPRSpectrometer
from ..core.dual_foil_spectrometer import DualFoilSpectrometer

if TYPE_CHECKING:
    from ..core.hodoscope import Hodoscope


class HodoscopeResponse:
    """Per-channel measurement bundle for one hodoscope.

    Constructed from a single ``MPRSpectrometer`` (after its recoil beam
    has been generated and the transfer map applied).  All per-channel
    quantities are computed internally in one pass over the output beam.

    Parameters
    ----------
    spectrometer : MPRSpectrometer
        Spectrometer whose output_beam is used.
    foil_efficiencies : np.ndarray, shape (n_particles,)
        Per-particle foil efficiency from ``PerformanceAnalyzer._get_foil_efficiencies()``.
    particle_yield : float or None
        Total source yield for absolute-unit scaling. When None, signal is
        in [particles/source] and signal_std is NaN.
    response_matrix : shape (n_energies, n_channels) or None
        Pre-built instrument response matrix for SpectrumFitter.
    energy_grid : shape (n_energies,) or None
        Incident energy axis [MeV] paired with response_matrix.
    time_gate_percentiles : (low, high)
        Percentile range used to define per-channel signal time windows.
    compute_density : bool
        If True, also compute the 2-D focal-plane density map.
    dx, dy : float
        Grid resolution [cm] for the 2-D density map.

    Instance attributes
    -------------------
    signal : shape (n_channels,)
    signal_std : shape (n_channels,)  – Poisson std; NaN when yield is None
    coverage : shape (n_channels,)
    channel_time_windows : shape (n_channels, 2)
    foil_material : str
    neutron_background, photon_background : shape (n_channels,)
    neutron_background_std, photon_background_std : shape (n_channels,)
    background, background_std : properties (derived sums)
    response_matrix : shape (n_energies, n_channels) or None
    energy_grid : shape (n_energies,) or None
    density_map, response_map_2d : shape (ny, nx) or None
    density_x, density_y : shape (ny, nx) meshgrids [cm] or None
    """

    def __init__(
        self,
        spectrometer: MPRSpectrometer,
        foil_efficiencies: np.ndarray,
        particle_yield: Optional[float] = None,
        response_matrix: Optional[np.ndarray] = None,
        energy_grid: Optional[np.ndarray] = None,
        time_gate_percentiles: Tuple[float, float] = (0, 100),
        compute_density: bool = False,
        dx: float = 0.5,
        dy: float = 0.5,
    ) -> None:
        self.foil_material = spectrometer.conversion_foil.foil_material
        hodoscope = spectrometer.hodoscope

        self.signal, signal_std_per_source, self.count, total, self.channel_time_windows = self._bin_to_channels(
            spectrometer, foil_efficiencies, time_gate_percentiles
        )

        n_bins = len(self.signal)
        self.coverage = np.where(total > 0, self.signal / total, 0.0)

        if particle_yield is not None:
            self.signal *= particle_yield
            self.count *= particle_yield
            self.signal_std = np.where(
                signal_std_per_source > 0,
                signal_std_per_source * np.sqrt(particle_yield),
                np.nan,
            )
        else:
            self.signal_std = np.full(n_bins, np.nan)

        n_ch = hodoscope.total_channels
        if (hodoscope.neutron_background_file and hodoscope.photon_background_file
                and hodoscope.detector_used):
            _bg_yield = particle_yield if particle_yield is not None else 1.0
            n_bg, ph_bg, n_bg_std, ph_bg_std = self._compute_background(hodoscope, _bg_yield)
            self.neutron_background = n_bg
            self.photon_background = ph_bg
            # shot std is only meaningful with a real yield
            if particle_yield is None:
                n_bg_std = np.zeros_like(n_bg)
                ph_bg_std = np.zeros_like(ph_bg)
            self.neutron_background_std = n_bg_std
            self.photon_background_std = ph_bg_std
        else:
            self.neutron_background = np.zeros(n_ch)
            self.photon_background = np.zeros(n_ch)
            self.neutron_background_std = np.zeros(n_ch)
            self.photon_background_std = np.zeros(n_ch)

        if compute_density:
            density_map, response_map_2d, density_x, density_y = self._compute_density_map(
                spectrometer, foil_efficiencies, dx, dy,
            )
            if particle_yield is not None:
                density_map = density_map * particle_yield
                response_map_2d = response_map_2d * particle_yield
            self.density_map = density_map
            self.response_map_2d = response_map_2d
            self.density_x = density_x
            self.density_y = density_y
        else:
            self.density_map = None
            self.response_map_2d = None
            self.density_x = None
            self.density_y = None

        self.response_matrix = response_matrix
        self.energy_grid = energy_grid

    @staticmethod
    def _bin_to_channels(
        spec: MPRSpectrometer,
        foil_efficiencies: np.ndarray,
        time_gate_percentiles: Tuple[float, float] = (0, 100),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bin the output beam into hodoscope channels.

        Returns (signal, count, total, channel_time_windows) per channel, normalized per source
        particle (foil_geometric_factor already applied) — no yield scaling applied.
        signal: weighted by foil_efficiency × detector_sensitivity.
                Units: MeV deposited/source particle (detector_used=True) or
                       counts/source particle (detector_used=False).
        count:  weighted by foil_efficiency only (denominator for Poisson std).
        total:  signal summed over all x-accepted particles (denominator for coverage).
        channel_time_windows: shape (n_bins, 2) arrival-time percentile bounds [s]; NaN when
                              use_time_gating is False.
        """
        hodoscope = spec.hodoscope
        x_positions = spec.output_beam[:, 0] * 100  # m → cm
        y_positions = spec.output_beam[:, 2] * 100  # m → cm
        output_energies_MeV = spec.reference_energy * (1 + spec.output_beam[:, 5])
        total_particles = len(x_positions)

        sensitivities = hodoscope.get_detector_response(
            energies=output_energies_MeV,
            particle=spec.conversion_foil.particle,
        )
        importance_weights = spec.input_beam[:, 7]
        weights = foil_efficiencies * sensitivities * importance_weights

        bin_edges_cm = hodoscope.channel_edges * 100
        bin_heights_cm = hodoscope.channel_heights * 100
        channel_y_centers_cm = hodoscope.channel_y_centers * 100
        n_bins = len(bin_edges_cm) - 1
        bin_indices = np.digitize(x_positions, bin_edges_cm) - 1

        arrival_times = spec.output_beam[:, 4]
        signal_per_bin = np.zeros(n_bins)
        count_per_bin = np.zeros(n_bins)
        total_per_bin = np.zeros(n_bins)
        channel_time_windows = np.full((n_bins, 2), np.nan)
        sq_weight_per_bin = np.zeros(n_bins)
        for b in range(n_bins):
            in_bin = bin_indices == b
            accepted = in_bin & (
                np.abs(y_positions - channel_y_centers_cm[b]) <= bin_heights_cm[b] / 2
            )
            total_per_bin[b] = np.sum(weights[in_bin]) / total_particles
            signal_per_bin[b] = np.sum(weights[accepted]) / total_particles
            count_per_bin[b] = np.sum(foil_efficiencies[accepted] * importance_weights[accepted]) / total_particles
            sq_weight_per_bin[b] = np.sum(weights[accepted] * sensitivities[accepted]) / total_particles
            if hodoscope.use_time_gating:
                times_in_channel = arrival_times[accepted]
                if len(times_in_channel) > 0:
                    channel_time_windows[b, 0] = np.percentile(times_in_channel, time_gate_percentiles[0])
                    channel_time_windows[b, 1] = np.percentile(times_in_channel, time_gate_percentiles[1])

        if spec.foil_geometric_factor:
            signal_per_bin *= spec.foil_geometric_factor
            count_per_bin *= spec.foil_geometric_factor
            total_per_bin *= spec.foil_geometric_factor
            sq_weight_per_bin *= spec.foil_geometric_factor

        signal_std_per_source = np.sqrt(sq_weight_per_bin)
        return signal_per_bin, signal_std_per_source, count_per_bin, total_per_bin, channel_time_windows

    @staticmethod
    def _compute_density_map(
        spec: MPRSpectrometer,
        foil_efficiencies: np.ndarray,
        dx: float = 0.5,
        dy: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the 2-D focal-plane density map.

        Returns (density_map, response_map_2d, density_x, density_y) — all shape (ny, nx).
        Normalized per source particle (foil_geometric_factor already applied). No yield scaling.
        density_map is weighted by foil_efficiency; response_map_2d also by detector_sensitivity.
        """
        hodoscope = spec.hodoscope
        x_positions = spec.output_beam[:, 0] * 100  # m → cm
        y_positions = spec.output_beam[:, 2] * 100  # m → cm
        output_energies_MeV = spec.reference_energy * (1 + spec.output_beam[:, 5])
        total_particles = len(x_positions)

        sensitivities = hodoscope.get_detector_response(
            energies=output_energies_MeV,
            particle=spec.conversion_foil.particle,
        )
        importance_weights = spec.input_beam[:, 7]

        x_min, x_max = float(np.min(x_positions)), float(np.max(x_positions))
        y_min, y_max = float(np.min(y_positions)), float(np.max(y_positions))
        x_coords = np.linspace(x_min, x_max, int((x_max - x_min) / dx) + 1)
        y_coords = np.linspace(y_min, y_max, int((y_max - y_min) / dy) + 1)
        X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)
        density_map = np.zeros_like(X_mesh)
        response_map_2d = np.zeros_like(X_mesh)
        cell_area_cm2 = dx * dy

        for i in range(total_particles):
            xi = min(max(int((x_positions[i] - x_min) / dx), 0), density_map.shape[1] - 1)
            yi = min(max(int((y_positions[i] - y_min) / dy), 0), density_map.shape[0] - 1)
            density_map[yi, xi] += foil_efficiencies[i] * importance_weights[i]
            if hodoscope.detector_used:
                response_map_2d[yi, xi] += foil_efficiencies[i] * sensitivities[i] * importance_weights[i]

        density_map /= (cell_area_cm2 * total_particles)
        response_map_2d /= (cell_area_cm2 * total_particles)

        if spec.foil_geometric_factor:
            density_map *= spec.foil_geometric_factor
            response_map_2d *= spec.foil_geometric_factor

        return density_map, response_map_2d, X_mesh, Y_mesh

    def _compute_background(
        self,
        hodoscope: Hodoscope,
        particle_yield: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Integrate time-resolved background into per-channel counts.

        Reads background time series from hodoscope.get_background() and
        integrates over each channel's time window (or the full window when
        time-gating is disabled).  Uses fractional bin overlap so windows
        narrower than a time bin are handled correctly at the edges.

        The compound-Poisson shot (MeV^2/cm^2/source per time bin) is integrated
        over the same time gate, then sqrt'd into a per-channel shot std [MeV].

        Returns n_bg, ph_bg, n_bg_std, ph_bg_std — each shape (n_channels,).
        """
        time_bins, n_bg_ts, ph_bg_ts, n_var_ts, ph_var_ts = hodoscope.get_background()
        channel_widths_cm = hodoscope.channel_widths * 100
        channel_heights_cm = hodoscope.channel_heights * 100
        n_ch = len(channel_widths_cm)
        dt = float(np.median(np.diff(time_bins))) if len(time_bins) > 1 else 1.0
        n_bg = np.zeros(n_ch)
        ph_bg = np.zeros(n_ch)
        n_bg_var = np.zeros(n_ch)
        ph_bg_var = np.zeros(n_ch)
        for k in range(n_ch):
            area = channel_widths_cm[k] * channel_heights_cm[k]
            if hodoscope.use_time_gating and not np.isnan(self.channel_time_windows[k, 0]):
                t_lo, t_hi = self.channel_time_windows[k]
                bin_left = time_bins - dt / 2
                bin_right = time_bins + dt / 2
                overlap = (
                    np.maximum(0.0, np.minimum(bin_right, t_hi) - np.maximum(bin_left, t_lo)) / dt
                )
            else:
                overlap = np.ones(len(time_bins))
            n_bg[k] = np.dot(n_bg_ts, overlap) * particle_yield * area
            ph_bg[k] = np.dot(ph_bg_ts, overlap) * particle_yield * area
            n_bg_var[k] = np.dot(n_var_ts, overlap) * particle_yield * area
            ph_bg_var[k] = np.dot(ph_var_ts, overlap) * particle_yield * area
        return n_bg, ph_bg, np.sqrt(n_bg_var), np.sqrt(ph_bg_var)

    @property
    def background(self) -> np.ndarray:
        return self.neutron_background + self.photon_background

    @property
    def background_std(self) -> np.ndarray:
        return np.sqrt(self.neutron_background_std ** 2 + self.photon_background_std ** 2)


@dataclass
class ChannelWaveforms:
    """Per-channel electrode current waveforms for one hodoscope (raw Allpix sign: collected electrons negative).

    Species keys: 'proton', 'deuteron', 'neutron', 'photon'.

    Attributes:
        foil_material: Foil whose hodoscope these channels belong to.
        t: Time axis [s], shape (n_t,).
        current, variance: species -> shape (n_channels, n_t) [A, A^2]; variance is compound-Poisson shot noise.
        charge, charge_var: species -> shape (n_channels,) [C, C^2], integrated over the full time axis.
        space_charge_ratio: |Total charge| / (C V) per channel (NaN for zero-area channels); large values break the linear pulse sum.
        linear_valid: space_charge_ratio below space_charge_threshold.
        space_charge_threshold: Threshold passed to get_channel_waveforms().
        gate_percentiles: Cumulative |current| percentiles of the intended species that bound each channel's gate.
        gate_windows: shape (n_channels, 2) [s], start and end of each channel's gate (NaN without intended current).
        charge_gated, charge_gated_var: species -> shape (n_channels,) [C, C^2], integrated over the gate only
            (charge_gated_var is NaN for a pulse library without qv). Gated fields are None when get_channel_waveforms() is called with gate_percentiles=None.
    """
    foil_material: str
    t: np.ndarray
    current: Dict[str, np.ndarray]
    variance: Dict[str, np.ndarray]
    charge: Dict[str, np.ndarray]
    charge_var: Dict[str, np.ndarray]
    space_charge_ratio: np.ndarray
    linear_valid: np.ndarray
    space_charge_threshold: float
    gate_percentiles: Optional[Tuple[float, float]] = None
    gate_windows: Optional[np.ndarray] = None
    charge_gated: Optional[Dict[str, np.ndarray]] = None
    charge_gated_var: Optional[Dict[str, np.ndarray]] = None

    @property
    def total_current(self) -> np.ndarray:
        return sum(self.current.values())

    @property
    def total_std(self) -> np.ndarray:
        return np.sqrt(sum(self.variance.values()))

    @property
    def total_charge(self) -> np.ndarray:
        return sum(self.charge.values())

    @property
    def total_charge_std(self) -> np.ndarray:
        return np.sqrt(sum(self.charge_var.values()))

    @property
    def total_charge_gated(self) -> np.ndarray:
        return sum(self.charge_gated.values())

    @property
    def total_charge_gated_std(self) -> np.ndarray:
        return np.sqrt(sum(self.charge_gated_var.values()))


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
    def fwfm(data: np.ndarray, fractional_max, bandwidth_method: str | float = "scott", _recursions=0) -> tuple[float, float, float, float]:
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
        Tuple of (full_width, position, lower, upper) where position is the KDE peak,
        and lower/upper are the left and right edges of the fractional-max interval.
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
        position = np.mean(data)
        
        # If the kernel seems like it could be smaller
        if full_width < 3*bandwidth and _recursions < 5:
            # Reduce the kernel size to aim for bandwidth < full_width/3
            reduction = 4*bandwidth/full_width
            return PerformanceAnalyzer.fwfm(
                data, fractional_max, bandwidth_method=bandsigma/reduction, _recursions=_recursions + 1)

        else:
            return full_width, position, lower, upper

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
            position_width, position_mean, _, _ = PerformanceAnalyzer.fwfm(positions, fractional_max=fractional_max)
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
                positions_lower = np.zeros_like(energies)
                positions_upper = np.zeros_like(energies)
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
                    positions_width[i], positions_mean[i], positions_lower[i], positions_upper[i] = PerformanceAnalyzer.fwfm(positions, fractional_max=fractional_max)
                    
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
                
                gradients = np.gradient(positions_mean, energies)
                energy_resolutions = positions_width / gradients * 1000

                # Create DataFrame for this foil
                foil_df = pd.DataFrame({
                    'foil': foil_name,
                    'energy [MeV]': energies,
                    'position [m]': positions_mean,
                    'position lower [m]': positions_lower,
                    'position upper [m]': positions_upper,
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
    
    def _load_performance_curve(self, performance_curve_file: Optional[str] = None) -> Optional[pd.DataFrame]:
        filename = performance_curve_file or 'comprehensive_performance.csv'
        try:
            return pd.read_csv(f'{self.spectrometer.data_directory}/{filename}')
        except Exception:
            warnings.warn(
                f'Performance curve {filename} not found in {self.spectrometer.data_directory}. '
                'Run generate_performance_curve() first.',
                RuntimeWarning,
            )
            return None

    @staticmethod
    def _get_foil_efficiencies(
        spec: MPRSpectrometer,
        performance_df: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """Interpolate per-particle foil efficiency from the performance curve."""
        df = performance_df
        if df is not None and 'foil' in df.columns:
            df = df[df['foil'] == spec.conversion_foil.foil_material]
        input_energies = spec.input_beam[:, 6]
        if df is not None and len(df) > 0:
            return np.interp(
                input_energies,
                df['energy [MeV]'].to_numpy(),
                df['total efficiency'].to_numpy(),
            )
        return np.ones(len(input_energies))

    def build_response_matrix(
        self,
        energy_grid: np.ndarray,
        num_recoils_per_energy: int = 10000,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        output_filename: Optional[str] = None,
        reset: bool = True,
        executor=None,
        max_workers: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Build the instrument response matrix for each foil.

        Returns a dict mapping foil material name to R of shape (n_energies, n_channels).
        R[i, k] is the expected signal in hodoscope channel k per source particle at energy
        energy_grid[i]. foil_geometric_factor is already baked in.

        Units of R[i, k]:
            detector_used=True  -> MeV deposited per source particle
            detector_used=False -> counts per source particle (dimensionless)

        Files are cached as <base>_<foil>.npz.

        Args:
            energy_grid: 1-D array of incident energies [MeV].
            num_recoils_per_energy: Monte Carlo rays per energy point.
            include_kinematics: Passed to generate_monte_carlo_rays.
            include_stopping_power_loss: Passed to generate_monte_carlo_rays.
            output_filename: Base path for .npz cache files (foil name appended).
                             Defaults to <data_directory>/response_matrix.
                             Each file contains arrays: R, energy_grid, bin_edges_cm.
            reset: If True, regenerate and save. If False, load from file.
            executor: Worker pool (if None, a fresh pool is created).
            max_workers: Maximum worker processes.

        Returns:
            Dict mapping foil material name -> np.ndarray of shape (n_energies, n_channels).
        """
        base = output_filename if output_filename is not None else f'{self.spectrometer.data_directory}/response_matrix'
        performance_df = self._load_performance_curve()

        def _build_for_spec(spec: MPRSpectrometer, y_restriction=None) -> tuple[str, np.ndarray]:
            key = spec.conversion_foil.foil_material
            cache_path = f'{base}_{key}.npz'

            if not reset:
                data = np.load(cache_path)
                R = data['R']
                print(f'Response matrix {key} loaded from {cache_path}')
                return key, R

            # Save polychromatic beams so the R-matrix loop doesn't overwrite them.
            saved_input = spec.input_beam.copy() if spec.input_beam.size > 0 else np.empty((0, 8))
            saved_output = spec.output_beam.copy() if spec.output_beam.size > 0 else np.empty((0, 6))

            n_energies = len(energy_grid)
            R = np.zeros((n_energies, spec.hodoscope.total_channels))
            print(f'\nBuilding response matrix for {key}...')

            in_range = (energy_grid >= spec.min_incident_energy) & (energy_grid <= spec.max_incident_energy)
            n_in_range = int(in_range.sum())

            spec.generate_monte_carlo_rays(
                energy_grid[in_range],
                np.ones(n_in_range),
                num_recoils_per_energy * n_in_range,
                include_kinematics,
                include_stopping_power_loss,
                save_beam=False,
                executor=executor,
                max_workers=max_workers,
                y_restriction=y_restriction,
            )
            spec.apply_transfer_map(
                save_beam=False,
                executor=executor,
                max_workers=max_workers,
            )

            foil_efficiencies = self._get_foil_efficiencies(spec, performance_df)
            bin_idx = np.searchsorted(energy_grid, spec.input_beam[:, 6])
            full_input, full_output = spec.input_beam, spec.output_beam
            for i in tqdm(range(n_energies), desc=key):
                m = bin_idx == i
                if not m.any():
                    continue
                spec.input_beam, spec.output_beam = full_input[m], full_output[m]
                signal, _, _, _, _ = HodoscopeResponse._bin_to_channels(spec, foil_efficiencies[m])
                R[i, :] = signal
            spec.input_beam, spec.output_beam = full_input, full_output

            # Restore the polychromatic beams.
            spec.input_beam = saved_input
            spec.output_beam = saved_output

            bin_edges_cm = spec.hodoscope.channel_edges * 100  # m → cm
            np.savez(cache_path, R=R, energy_grid=energy_grid, bin_edges_cm=bin_edges_cm)
            print(f'Response matrix {key} saved to {cache_path}')
            return key, R

        result = {}
        is_dual = hasattr(self, 'dual_spectrometer')
        k, R = _build_for_spec(self.spectrometer, y_restriction='positive' if is_dual else None)
        result[k] = R
        if is_dual:
            k2, R2 = _build_for_spec(self.dual_spectrometer, y_restriction='negative')
            result[k2] = R2
        return result

    def get_channel_response(
        self,
        particle_yield: Optional[float],
        response_matrices: Optional[Dict[str, np.ndarray]] = None,
        energy_grid: Optional[np.ndarray] = None,
        compute_density: bool = False,
        dx: float = 0.2,
        dy: float = 0.2,
    ) -> Union[HodoscopeResponse, Dict[str, HodoscopeResponse]]:
        """Build a HodoscopeResponse (or dict for dual-foil) for this spectrometer.

        Args:
            particle_yield: Total source yield for absolute-unit scaling.
            response_matrices: Dict mapping foil material name to R
                (n_energies, n_channels), as returned by build_response_matrix().
                Pass None to leave response_matrix unset on the result.
            energy_grid: Incident energy axis [MeV] paired with response_matrices.
            compute_density: If True, also compute the 2-D focal-plane density map.
            dx, dy: Grid resolution [cm] for the 2-D density map.

        Returns:
            Single HodoscopeResponse for a single-foil spectrometer.
            Dict[str, HodoscopeResponse] for a dual-foil spectrometer,
            keyed by foil material name.
        """
        performance_df = self._load_performance_curve()

        def _build(spec: MPRSpectrometer) -> HodoscopeResponse:
            foil = spec.conversion_foil.foil_material
            foil_efficiencies = self._get_foil_efficiencies(spec, performance_df)
            return HodoscopeResponse(
                spectrometer=spec,
                foil_efficiencies=foil_efficiencies,
                particle_yield=particle_yield,
                response_matrix=response_matrices.get(foil) if response_matrices else None,
                energy_grid=energy_grid,
                compute_density=compute_density,
                dx=dx,
                dy=dy,
            )

        primary = _build(self.spectrometer)
        if not hasattr(self, 'dual_spectrometer'):
            return primary
        secondary = _build(self.dual_spectrometer)
        return {primary.foil_material: primary, secondary.foil_material: secondary}

    def get_channel_waveforms(
        self,
        particle_yield: float,
        space_charge_threshold: float = 0.1,
        gate_percentiles: Optional[Tuple[float, float]] = (0.5, 99.5),
        chunk_size: int = 2000,
    ) -> Union[ChannelWaveforms, Dict[str, ChannelWaveforms]]:
        """Per-channel electrode current waveforms from the hodoscope's Allpix pulse library.

        Each MC recoil adds its interpolated pulse (hit pad and both neighbors), scaled by the number of real
        particles it represents and shifted to its arrival time. Recoils from every foil are binned against every
        hodoscope, so protons landing on the CD2 wafer (and deuterons on the CH2 wafer) are included.
        Background (neutron, photon) adds the pad-averaged neutral pulse for each (time, energy) fluence bin,
        spread uniformly over the time bin. Variances are compound-Poisson: sum of N * E[h^2].

        Gating models post-processing of each channel's digitized waveform: the gate spans the given cumulative
        |current| percentiles of the hodoscope's intended species (its foil's recoil, including current induced from
        neighbor hits), and every species' current is integrated over it, so pulses cut at the edges keep their
        residual charge. Gated variance is sum of N * E[Q_gate^2], each particle's second moment of the charge inside
        its window of the gate (PulseLibrary.gated_q2); NaN with a warning if the library predates qv.

        Args:
            particle_yield: Total source yield.
            space_charge_threshold: Q / (C V) above which a channel is flagged as outside the linear regime.
            gate_percentiles: (low, high) percentiles bounding each channel's gate; None skips gating.
            chunk_size: Recoils processed per vectorized batch (bounds memory).

        Returns:
            ChannelWaveforms for a single-foil spectrometer, or a dict keyed by foil material.
        """
        specs = [self.spectrometer] + ([self.dual_spectrometer] if hasattr(self, 'dual_spectrometer') else [])
        # All hodoscopes need a library on a shared time grid
        libraries = [s.hodoscope.pulse_library for s in specs]
        if any(lib is None for lib in libraries):
            raise ValueError("Every hodoscope needs a pulse_library.")
        if len({(lib.dt, lib.n_t) for lib in libraries}) > 1:
            raise ValueError("Pulse libraries must share dt and window to use a common time axis.")
        dt, n_k = libraries[0].dt, libraries[0].n_t

        # Per-foil recoils at the detector; n = real particles per MC ray
        performance_df = self._load_performance_curve()
        recoils = []
        for spec in specs:
            efficiencies = self._get_foil_efficiencies(spec, performance_df)
            geometric = spec.foil_geometric_factor or 1.0
            recoils.append(dict(
                species=spec.conversion_foil.particle,
                x=spec.output_beam[:, 0] * 100, y=spec.output_beam[:, 2] * 100,  # cm
                energy=spec.reference_energy * (1 + spec.output_beam[:, 5]),
                time=spec.output_beam[:, 4],
                n=efficiencies * spec.input_beam[:, 7] * geometric / len(spec.output_beam) * particle_yield,
            ))

        # Load each distinct (time, energy) background file once
        backgrounds = {}
        for spec in specs:
            h = spec.hodoscope
            for species, path in (('neutron', h.neutron_background_file), ('photon', h.photon_background_file)):
                if path and path not in backgrounds:
                    if 'time' not in pd.read_csv(path, nrows=0).columns:
                        raise ValueError(f"Background file needs a 'time' column for waveforms: {path}")
                    backgrounds[path] = (species, *h._load_background_2d(path))

        # Global time axis: earliest of 0, first arrival and first background bin edge, to last arrival plus one pulse window
        # Background time-bin width, used to spread each bin's arrivals
        bg_width = {p: float(np.median(np.diff(b[1]))) if len(b[1]) > 1 else 0.0 for p, b in backgrounds.items()}
        t_start = min([0.0] + [r['time'].min() for r in recoils]
                      + [b[1][0] - bg_width[p] / 2 for p, b in backgrounds.items()])
        t_last = max([r['time'].max() for r in recoils] + [b[1][-1] + bg_width[p] / 2 for p, b in backgrounds.items()])
        n_t = int(np.ceil((t_last - t_start) / dt)) + n_k + 1
        t = t_start + np.arange(n_t) * dt

        def build(spec: MPRSpectrometer) -> ChannelWaveforms:
            h = spec.hodoscope
            library = h.pulse_library  # this wafer's response, also for crossover recoils
            n_ch = h.total_channels
            # Channel geometry in cm
            edges, heights, y_centers = h.channel_edges * 100, h.channel_heights * 100, h.channel_y_centers * 100
            centers, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
            area = widths * heights  # cm^2
            current, variance, charge_var = {}, {}, {}  # electrons per bin, electrons^2 per bin, electrons^2

            def totals(species: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                if species not in current:
                    current[species], variance[species] = np.zeros((n_ch, n_t)), np.zeros((n_ch, n_t))
                    charge_var[species] = np.zeros(n_ch)
                return current[species], variance[species], charge_var[species]

            # Library pad pitch nearest to each channel width
            pitch_of = np.array([library.nearest_pitch(w * 1e-2) for w in widths])

            def recoil_chunks():
                """Yield (species, pitch, energy, offset_um, n, channel, side, start bin) per recoil chunk and pitch."""
                # Signal: every foil's recoils against this hodoscope
                for r in recoils:
                    # Keep recoils inside a channel in both x and y
                    b = np.digitize(r['x'], edges) - 1
                    inside = (b >= 0) & (b < n_ch)
                    b = np.where(inside, b, 0)
                    accepted = np.flatnonzero(inside & (np.abs(r['y'] - y_centers[b]) <= heights[b] / 2))
                    # Chunked to bound the (recoils x kernel length) arrays
                    for i in range(0, len(accepted), chunk_size):
                        sel = accepted[i:i + chunk_size]
                        bs = b[sel]
                        offset_um = (widths[bs] / 2 - np.abs(r['x'][sel] - centers[bs])) * 1e4  # distance to nearest edge (gap center)
                        side = np.where(r['x'][sel] >= centers[bs], 1, -1)  # library pad +1 is the nearest neighbor
                        start = np.rint((r['time'][sel] - t_start) / dt).astype(int)
                        # Kernel lookup grouped by pitch
                        for pitch in np.unique(pitch_of[bs]):
                            m = pitch_of[bs] == pitch
                            yield r['species'], pitch, r['energy'][sel][m], offset_um[m], r['n'][sel][m], bs[m], side[m], start[m]

            def pads(bs: np.ndarray, side: np.ndarray):
                """Yield (library pad, channel, mask of channels on the array) for the hit pad and both neighbors."""
                for k, rel in enumerate((-1, 0, 1)):
                    ch = bs + rel * side
                    ok = (ch >= 0) & (ch < n_ch)  # drop neighbors off the array
                    yield k, ch, ok

            def background_files():
                """Yield (species, energy bins, fluence, first start bin per time bin, n_box) per file of this hodoscope."""
                for path, (species, time_bins, energy_bins, fluence) in backgrounds.items():
                    # Skip files that belong to the other hodoscope
                    if path not in (h.neutron_background_file, h.photon_background_file):
                        continue
                    # Start bins of the uniform spread across each background time bin
                    n_box = max(1, int(round(bg_width[path] / dt)))
                    first = np.rint((time_bins - bg_width[path] / 2 - t_start) / dt).astype(int)
                    yield species, energy_bins, fluence, first, n_box

            for species, pitch, energy, offset_um, n, bs, side, start in recoil_chunks():
                mean, m2, q2 = library.kernels(species, pitch, energy, offset_um)
                cur, var, qv = totals(species)
                for k, ch, ok in pads(bs, side):
                    # Direct sum of n-weighted kernels into channels ch at time bins start
                    idx = ((ch[ok] * n_t + start[ok])[:, None] + np.arange(n_k)).ravel()
                    cur += np.bincount(idx, (n[ok, None] * mean[ok, k]).ravel(), n_ch * n_t).reshape(n_ch, n_t)
                    var += np.bincount(idx, (n[ok, None] * m2[ok, k]).ravel(), n_ch * n_t).reshape(n_ch, n_t)
                    qv += np.bincount(ch[ok], n[ok] * q2[ok, k], n_ch)

            # Background: response per (source x cm^2) for each pitch, then scaled by yield x area per channel
            for species, energy_bins, fluence, first, n_box in background_files():
                for pitch in np.unique(pitch_of):
                    mean, m2, q2 = library.kernels(species, pitch, energy_bins)
                    # Per-source, per-cm^2 waveform on pads [-1, 0, +1]
                    unit = np.zeros((2, 3, n_t))  # [mean, m2] x pads [-1, 0, +1] x time
                    for a, kern in enumerate((mean, m2)):
                        per_bin = np.einsum('jk,kpt->jpt', fluence, kern) / n_box
                        for j in np.flatnonzero(fluence.sum(axis=1) > 0):
                            for st in range(first[j], first[j] + n_box):  # uniform arrival within the time bin
                                unit[a, :, st:st + n_k] += per_bin[j]
                    unit_q2 = (fluence @ q2).sum(axis=0)  # (3,)
                    cur, var, qv = totals(species)
                    # Scale to each channel of this pitch, dropping neighbors off the array
                    for c in np.flatnonzero(pitch_of == pitch):
                        scale = particle_yield * area[c]
                        for k, rel in enumerate((-1, 0, 1)):
                            if 0 <= c + rel < n_ch:
                                cur[c + rel] += scale * unit[0, k]
                                var[c + rel] += scale * unit[1, k]
                                qv[c + rel] += scale * unit_q2[k]

            # Convert to A, A^2, C, C^2
            to_amp = ELEMENTARY_CHARGE / dt  # electrons per bin -> A
            current_amp = {s: v * to_amp for s, v in current.items()}
            variance_amp2 = {s: v * to_amp ** 2 for s, v in variance.items()}
            charge = {s: c.sum(axis=1) * dt for s, c in current_amp.items()}
            q_var = {s: c * ELEMENTARY_CHARGE ** 2 for s, c in charge_var.items()}
            # Space-charge ratio Q/CV per channel
            capacitance = h.channel_capacitance()
            # Q/CV is undefined (NaN, not flagged) for degenerate zero-area channels
            ratio = np.divide(np.abs(sum(charge.values())), capacitance * abs(library.bias_voltage),
                              out=np.full(n_ch, np.nan), where=capacitance > 1e-6 * np.median(capacitance))
            result = ChannelWaveforms(spec.conversion_foil.foil_material, t, current_amp, variance_amp2, charge, q_var,
                                      ratio, ~(ratio >= space_charge_threshold), space_charge_threshold)
            if gate_percentiles is None:
                return result

            # Gate per channel: time bins [lo, hi] between the intended species' cumulative |current| percentiles;
            # empty (hi < lo) for channels without intended current
            lo, hi = np.zeros(n_ch, int), np.full(n_ch, -1)
            intended = current.get(spec.conversion_foil.particle, np.zeros((n_ch, n_t)))
            cum = np.cumsum(np.abs(intended), axis=1)
            for c in np.flatnonzero(cum[:, -1] > 0):
                lo[c] = np.searchsorted(cum[c], gate_percentiles[0] / 100 * cum[c, -1], side='right')
                hi[c] = min(np.searchsorted(cum[c], gate_percentiles[1] / 100 * cum[c, -1], side='left'), n_t - 1)
            in_gate = (np.arange(n_t) >= lo[:, None]) & (np.arange(n_t) <= hi[:, None])
            result.gate_percentiles = tuple(gate_percentiles)
            result.gate_windows = np.where((hi >= lo)[:, None], np.column_stack([t[lo], t[hi] + dt]), np.nan)
            result.charge_gated = {s: (c * in_gate).sum(axis=1) * ELEMENTARY_CHARGE for s, c in current.items()}
            if not library.has_gated_moments:
                warnings.warn("Pulse library has no cumulative-charge moments (qv); gated charge variance is NaN. "
                              "Rebuild the library to get it.")
                result.charge_gated_var = {s: np.full(n_ch, np.nan) for s in current}
                return result

            # Gated variance: second pass, each particle's charge inside its target channel's gate, as pulse bins
            # [lo - start, hi + 1 - start) from its arrival
            gated_var = {s: np.zeros(n_ch) for s in current}  # electrons^2
            for species, pitch, energy, offset_um, n, bs, side, start in recoil_chunks():
                for k, ch, ok in pads(bs, side):
                    a, b = (lo[ch[ok]] - start[ok])[:, None], (hi[ch[ok]] + 1 - start[ok])[:, None]
                    q2g = library.gated_q2(species, pitch, energy[ok], a, b, offset_um[ok], pads=(k,))[:, 0]
                    gated_var[species] += np.bincount(ch[ok], n[ok] * q2g, n_ch)
            for species, energy_bins, fluence, first, n_box in background_files():
                # Nonzero (time, energy) bins, each spread uniformly over n_box start bins
                j, e = np.nonzero(fluence)
                starts = (first[j, None] + np.arange(n_box)).ravel()
                energies = np.repeat(energy_bins[e], n_box)
                per_start = np.repeat(fluence[j, e], n_box) / n_box  # per source per cm^2
                for c in range(n_ch):
                    scale = particle_yield * area[c]
                    for k, rel in enumerate((-1, 0, 1)):
                        g = c + rel
                        if not 0 <= g < n_ch or hi[g] < lo[g]:
                            continue
                        q2g = library.gated_q2(species, pitch_of[c], energies, (lo[g] - starts)[:, None],
                                               (hi[g] + 1 - starts)[:, None], pads=(k,))[:, 0]
                        gated_var[species][g] += scale * (per_start @ q2g)
            result.charge_gated_var = {s: v * ELEMENTARY_CHARGE ** 2 for s, v in gated_var.items()}
            return result

        results = [build(s) for s in specs]
        if len(results) == 1:
            return results[0]
        return {r.foil_material: r for r in results}
