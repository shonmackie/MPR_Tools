"""Plotting methods for MPR spectrometer visualization."""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata, interp1d
from labellines import labelLines
import pandas as pd

# Set default plotting parameters
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['lines.linewidth'] = 3

from ..core.spectrometer import MPRSpectrometer
from ..core.dual_foil_spectrometer import DualFoilSpectrometer
from ..analysis.performance import PerformanceAnalyzer
from ..config.constants import MASS_TO_MEV

from ..analysis.forward_fitting import compute_spectrum_data_points

if TYPE_CHECKING:
    from ..analysis.parameter_sweep import FoilSweeper
    from ..analysis.forward_fitting import ForwardFittingResult

class SpectrometerPlotter:
    """Handles all plotting functionality for MPR spectrometer."""
    
    def __init__(self, spectrometer: Union[MPRSpectrometer, DualFoilSpectrometer]) -> None:
        if isinstance(spectrometer, MPRSpectrometer):
            self.spectrometer = spectrometer
            self.dual_data = None  # Will be set for dual-foil mode
            self.performance_analyzer = PerformanceAnalyzer(spectrometer)
            self.primary_color = 'tab:red'
            self.primary_cmap = 'plasma'
            
        elif isinstance(spectrometer, DualFoilSpectrometer):
            # Dual-foil mode, primary foil is CH2, secondary foil is CD2
            self.spectrometer = spectrometer.spec_ch2
            self.performance_analyzer = PerformanceAnalyzer(self.spectrometer)
            self.dual_data = {
                'spectrometer': spectrometer.spec_cd2,
                'performance_analyzer': PerformanceAnalyzer(spectrometer.spec_cd2),
                'primary_label': 'protons',
                'secondary_label': 'deuterons',
                'secondary_color': 'tab:blue',
                'secondary_cmap': 'GnBu'
            }
            self.dual_spectrometer = spectrometer
            self.primary_color = 'tab:red'
            self.primary_cmap = 'YlOrRd'
        else:
            raise ValueError(f'Invalid spectrometer type: {type(spectrometer)}. Should be MPRSpectrometer or DualFoilSpectrometer.')
    
    @staticmethod
    def _overlay_hodoscope(ax: Axes, hodoscope, color: str = 'black') -> None:
        """Draw the full envelope and channel boundaries for one hodoscope on ax."""
        heights = hodoscope.channel_heights * 100  # cm, shape (N,)
        edges = hodoscope.channel_edges * 100       # cm, shape (N+1,)
        y_ctrs = hodoscope.channel_y_centers * 100  # cm, per-channel (N,)

        tops = y_ctrs + heights / 2
        bots = y_ctrs - heights / 2
        n = len(heights)

        x_env = np.empty(2 * n)
        x_env[0::2] = edges[:-1]
        x_env[1::2] = edges[1:]

        ax.plot(x_env, np.repeat(tops, 2), color=color, linewidth=1.0)
        ax.plot(x_env, np.repeat(bots, 2), color=color, linewidth=1.0)
        ax.plot([edges[0], edges[0]], [bots[0], tops[0]], color=color, linewidth=1.0)
        ax.plot([edges[-1], edges[-1]], [bots[-1], tops[-1]], color=color, linewidth=1.0)

        for i in range(1, n):
            y_lo = max(bots[i - 1], bots[i])
            y_hi = min(tops[i - 1], tops[i])
            if y_hi > y_lo:
                ax.vlines(edges[i], y_lo, y_hi, color=color, linestyle='--', linewidth=0.5)

    def plot_focal_plane_distribution(
        self,
        filename: Optional[str] = None,
        include_hodoscope: bool = False,
        point_size: float = 1.0
    ) -> None:
        """
        Plot focal particle distribution in the detector plane.

        Args:
            filename: Output filename
            include_hodoscope: Whether to overlay hodoscope geometry
            point_size: Size of scatter plot points
        """
        if filename == None:
            filename = f'{self.spectrometer.figure_directory}/focal_plane_distribution.png'

        fig, ax = plt.subplots(figsize=(10, 8))

        if include_hodoscope:
            self._overlay_hodoscope(ax, self.spectrometer.hodoscope)
            if self.dual_data:
                self._overlay_hodoscope(ax, self.dual_data['spectrometer'].hodoscope)

        # Scatter plot of focal particle positions
        particle_energies = self.spectrometer.input_beam[:, 5] * self.spectrometer.reference_energy + self.spectrometer.reference_energy
        scatter = ax.scatter(
            self.spectrometer.output_beam[:, 0]*100, 
            self.spectrometer.output_beam[:, 2]*100,
            c=particle_energies,
            s=point_size,
            cmap=self.primary_cmap,
            alpha=0.7
        )
        
        fig.colorbar(scatter, label=f'{self.spectrometer.conversion_foil.particle.capitalize()} Energy [MeV]')
        ax.set_xlabel('Horizontal Position [cm]')
        ax.set_ylabel('Vertical Position [cm]')
        ax.grid(True, alpha=0.3)
        
        if self.dual_data:
            spec2: MPRSpectrometer = self.dual_data['spectrometer']
            recoil_energies2 = spec2.input_beam[:, 5] * spec2.reference_energy + spec2.reference_energy
            scatter2 = ax.scatter(
                spec2.output_beam[:, 0]*100, 
                spec2.output_beam[:, 2]*100,
                c=recoil_energies2,
                s=point_size,
                cmap=self.dual_data['secondary_cmap'],
                alpha=0.7,
            )
            fig.colorbar(scatter2, label=f'{spec2.conversion_foil.particle.capitalize()} Energy [MeV]')
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Focal plane plot saved to {filename}')
    
    def plot_phase_space(self, filename: Optional[str] = None) -> None:
        """
        Generate phase space plots.
        
        Args:
            filename: Output filename for the plot
        """
        if filename == None:
            filename = f'{self.spectrometer.figure_directory}/phase_space.png'
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout='constrained')
        fig.suptitle('Phase Space')
        
        # Color by focal particle energy
        x_pos = self.spectrometer.output_beam[:, 0] * 100  # Convert to cm
        x_moment = self.spectrometer.output_beam[:, 1] * 1000  # Convert to mrad
        y_pos = self.spectrometer.output_beam[:, 2] * 100 # Convert to cm
        y_moment = self.spectrometer.output_beam[:, 3] * 1000  # Convert to mrad
        particle_energies = self.spectrometer.input_beam[:, 5] * self.spectrometer.reference_energy + self.spectrometer.reference_energy
        
        # X-Y position plot
        scatter1 = axes[0, 0].scatter(
            x_pos, y_pos, c=particle_energies,
            s=2.0, cmap=self.primary_cmap, alpha=0.7
        )
        axes[0, 0].set_xlabel('X Position [cm]')
        axes[0, 0].set_ylabel('Y Position [cm]')
        axes[0, 0].set_title('X-Y Position')
        axes[0, 0].grid(True, alpha=0.3)
        
        # X position vs normalized X momentum
        scatter2 = axes[0, 1].scatter(
            x_pos, x_moment, c=particle_energies,
            s=2.0, cmap=self.primary_cmap, alpha=0.7
        )
        axes[0, 1].set_xlabel('X Position [cm]')
        axes[0, 1].set_ylabel('X Angle [mrad]')
        axes[0, 1].set_title('X Position-Angle')
        axes[0, 1].grid(True, alpha=0.3)
        
        # X position vs energy
        scatter3 = axes[1, 0].scatter(
            x_pos, particle_energies, c=particle_energies,
            s=2.0, cmap=self.primary_cmap, alpha=0.7
        )
        axes[1, 0].set_xlabel('X Position [cm]')
        axes[1, 0].set_ylabel(f'E$_{{{self.spectrometer.conversion_foil.particle}}}$ [MeV]')
        axes[1, 0].set_title('X Position-Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Y position vs normalized Y momentum
        scatter4 = axes[1, 1].scatter(
            y_pos, y_moment, c=particle_energies,
            s=2.0, cmap=self.primary_cmap, alpha=0.7
        )
        axes[1, 1].set_xlabel('Y Position [cm]')
        axes[1, 1].set_ylabel('Y Angle [mrad]')
        axes[1, 1].set_title('Y Position-Angle')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add colorbar
        fig.colorbar(scatter1, ax=axes, label=f'{self.spectrometer.conversion_foil.particle.capitalize()} Energy [MeV]', shrink=0.8)
        
        # Plot dual data if available
        if self.dual_data:
            spec2: MPRSpectrometer = self.dual_data['spectrometer']
            x_pos2 = spec2.output_beam[:, 0] * 100 # Convert to cm
            x_moment2 = spec2.output_beam[:, 1] * 1000 # Convert to mrad
            y_pos2 = spec2.output_beam[:, 2] * 100 # Convert to cm
            y_moment2 = spec2.output_beam[:, 3] * 1000 # Convert to mrad
            recoil_energies2 = spec2.input_beam[:, 5] * spec2.reference_energy + spec2.reference_energy
            
            # X-Y position plot
            scatter1 = axes[0, 0].scatter(
                x_pos2, y_pos2, c=recoil_energies2,
                s=2.0, cmap=self.dual_data['secondary_cmap'], alpha=0.7
            )
            
            # X position vs normalized X momentum
            scatter2 = axes[0, 1].scatter(
                x_pos2, x_moment2, c=recoil_energies2,
                s=2.0, cmap=self.dual_data['secondary_cmap'], alpha=0.7
            )
            
            # X position vs energy
            scatter3 = axes[1, 0].scatter(
                x_pos2, recoil_energies2, c=recoil_energies2,
                s=2.0, cmap=self.dual_data['secondary_cmap'], alpha=0.7
            )
            
            # Y position vs normalized Y momentum
            scatter4 = axes[1, 1].scatter(
                y_pos2, y_moment2, c=recoil_energies2,
                s=2.0, cmap=self.dual_data['secondary_cmap'], alpha=0.7
            )
            
            # Add colorbar
            fig.colorbar(scatter1, ax=axes, label=f'{spec2.conversion_foil.particle.capitalize()} Energy [MeV]', shrink=0.8)
        
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Phase space portraits saved to {filename}')
    
    def plot_characteristic_rays(
        self,
        radial_points: int = 3,
        angular_points: int = 0, 
        aperture_radial_points: int = 0,
        aperture_angular_points: int = 0,
        energy_points: int = 1,
        min_energy: Optional[float] = None,
        max_energy: Optional[float] = None,
        filename: Optional[str] = None,
    ) -> None:
        """
        Generate and plot characteristic rays through the spectrometer system.
        
        This function generates characteristic rays using the generate_characteristic_rays()
        method, applies the transfer map, and visualizes both the input geometry and 
        output focal plane distribution.
        
        Args:
            radial_points: Number of radial points in foil (0 for on-axis only)
            angular_points: Number of angular points in foil
            aperture_radial_points: Number of radial points in aperture
            aperture_angular_points: Number of angular points in aperture
            energy_points: Number of energy points around reference
            min_energy: Minimum energy in MeV (defaults to class value)
            max_energy: Maximum energy in MeV (defaults to class value)
            filename: Output filename for the plot
        """
        if filename == None:
            filename = f'{self.spectrometer.figure_directory}/characteristic_rays.png'
        
        # Set default energy range if not provided
        if min_energy is None:
            min_energy = self.spectrometer.min_energy
        if max_energy is None:
            max_energy = self.spectrometer.max_energy
        
        print(f'Generating characteristic rays from {min_energy:.2f} to {max_energy:.2f} MeV...')
        
        # Generate characteristic rays
        self.spectrometer.generate_characteristic_rays(
            radial_points=radial_points,
            angular_points=angular_points,
            aperture_radial_points=aperture_radial_points,
            aperture_angular_points=aperture_angular_points,
            energy_points=energy_points,
            min_energy=min_energy,
            max_energy=max_energy
        )
        
        # Apply transfer map
        self.spectrometer.apply_transfer_map(map_order=5, save_beam=False)
        
        # Create subplots
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.suptitle('Characteristic Ray Analysis')
        
        # Focal plane distribution        
        # Scatter plot colored by energy
        output_energies = self.spectrometer.input_beam[:, 5] * self.spectrometer.reference_energy + self.spectrometer.reference_energy
        scatter = ax.scatter(
            self.spectrometer.output_beam[:, 0] * 100,  # Convert to cm
            self.spectrometer.output_beam[:, 2] * 100,  # Convert to cm
            c=output_energies,
            s=20,
            cmap=self.primary_cmap,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(f'{self.spectrometer.conversion_foil.particle.capitalize()} Energy [MeV]')

        if self.dual_data is not None:
            spec2: MPRSpectrometer = self.dual_data['spectrometer']
            print(f'Generating CD2 characteristic rays from {spec2.min_energy:.2f} to {spec2.max_energy:.2f} MeV...')
            spec2.generate_characteristic_rays(
                radial_points=radial_points,
                angular_points=angular_points,
                aperture_radial_points=aperture_radial_points,
                aperture_angular_points=aperture_angular_points,
                energy_points=energy_points,
                min_energy=spec2.min_energy,
                max_energy=spec2.max_energy,
            )
            spec2.apply_transfer_map(map_order=5, save_beam=False)
            output_energies2 = spec2.input_beam[:, 5] * spec2.reference_energy + spec2.reference_energy
            scatter2 = ax.scatter(
                spec2.output_beam[:, 0] * 100,
                spec2.output_beam[:, 2] * 100,
                c=output_energies2,
                s=20,
                cmap=self.dual_data['secondary_cmap'],
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
            )
            cbar2 = fig.colorbar(scatter2, ax=ax)
            cbar2.set_label(f'{spec2.conversion_foil.particle.capitalize()} Energy [MeV]')

        ax.set_xlabel('X Position [cm]')
        ax.set_ylabel('Y Position [cm]')
        ax.set_title('Focal Plane Distribution')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        fig.savefig(filename, dpi=150, bbox_inches='tight')
        
        # Print summary statistics
        print(f'Characteristic ray analysis complete:')
        print(f'  Total rays generated: {len(self.spectrometer.input_beam)}')
        print(f'  Energy range: {min_energy:.2f} - {max_energy:.2f} MeV')
        print(f'  X position range: {self.spectrometer.output_beam[:, 0].min()*100:.2f} - {self.spectrometer.output_beam[:, 0].max()*100:.2f} cm')
        print(f'  Y position range: {self.spectrometer.output_beam[:, 2].min()*100:.2f} - {self.spectrometer.output_beam[:, 2].max()*100:.2f} cm')
        print(f'Characteristic ray plot saved to {filename}')
    
    def plot_position_histogram(
        self,
        filename: Optional[str] = None,
        incident_particle_yield: Optional[float] = None,
        performance_curve_file: Optional[str] = None,
    ) -> None:
        """
        Plot signal and background counts per hodoscope channel, with S/B and coverage panels.

        Bins are taken from the hodoscope channel definitions by default.

        Up to four separate figures are saved, derived from the base filename:
          1. Signal (and background, if provided) counts per channel [particles/source].
          2. log10(S/B) per channel (only when background is provided).
             When hodoscope.use_time_gating is True and both background files are provided,
             two additional step lines are overlaid showing the non-gated S/B for comparison
             (dashed = no gate, solid = gated).
          3. Fraction of total y-beam captured within each channel's height [%].
          4. Per-channel signal arrival-time windows as horizontal bars [ns]. Only produced
             when hodoscope.use_time_gating is True.

        Args:
            filename: Output filename for the plot.
            incident_particle_yield: Total source yield; scales both signal and background.
                Background file paths are read from hodoscope.neutron_background_file and
                hodoscope.photon_background_file.
        """
        if filename is None:
            filename = f'{self.spectrometer.figure_directory}/counts_vs_position.png'

        if len(self.spectrometer.output_beam) == 0:
            raise ValueError("No output beam data available. Run apply_transfer_map() first.")

        hodoscope = self.spectrometer.hodoscope
        is_dual = self.dual_data is not None

        # --- Primary foil HodoscopeResponse ---
        primary_response = self.performance_analyzer.get_channel_response(incident_particle_yield)
        signal = primary_response.signal
        coverage = primary_response.coverage
        channel_time_windows = primary_response.channel_time_windows
        signal_std = primary_response.signal_std
        channel_edges = hodoscope.channel_edges * 100  # m to cm

        _nz = np.where(signal > 0)[0]
        sig_lo, sig_hi = (_nz[0], _nz[-1]) if len(_nz) else (0, len(signal) - 1)

        # Unpack per-channel background (None when no background files are configured)
        _has_bg = hodoscope.detector_used and hodoscope.neutron_background_file and hodoscope.photon_background_file
        neutron_bg_per_channel = primary_response.neutron_background if _has_bg else None
        photon_bg_per_channel = primary_response.photon_background if _has_bg else None
        neutron_bg_std_per_channel = primary_response.neutron_background_std if _has_bg else None
        photon_bg_std_per_channel = primary_response.photon_background_std if _has_bg else None

        # Time-resolved background arrays (used only for the S/B overlay in time-gating mode)
        time_bins = neutron_background_vs_time = photon_background_vs_time = None
        neutron_background_std_vs_time = photon_background_std_vs_time = None
        if _has_bg:
            (time_bins, neutron_background_vs_time, neutron_background_std_vs_time,
             photon_background_vs_time, photon_background_std_vs_time) = hodoscope.get_background()

        # --- Secondary foil HodoscopeResponse (dual-foil mode) ---
        signal2 = coverage2 = channel_time_windows2 = channel_edges2 = signal_std2 = None
        neutron_bg_per_channel2 = photon_bg_per_channel2 = None
        neutron_bg_std_per_channel2 = photon_bg_std_per_channel2 = None
        if is_dual:
            secondary_response = self.dual_data['performance_analyzer'].get_channel_response(
                incident_particle_yield
            )
            signal2 = secondary_response.signal
            coverage2 = secondary_response.coverage
            channel_time_windows2 = secondary_response.channel_time_windows
            signal_std2 = secondary_response.signal_std
            hodoscope2 = self.dual_data['spectrometer'].hodoscope
            channel_edges2 = hodoscope2.channel_edges * 100  # m to cm
            _has_bg2 = (hodoscope2.detector_used and hodoscope2.neutron_background_file
                        and hodoscope2.photon_background_file)
            neutron_bg_per_channel2 = secondary_response.neutron_background if _has_bg2 else None
            photon_bg_per_channel2 = secondary_response.photon_background if _has_bg2 else None
            neutron_bg_std_per_channel2 = secondary_response.neutron_background_std if _has_bg2 else None
            photon_bg_std_per_channel2 = secondary_response.photon_background_std if _has_bg2 else None
            _nz2 = np.where(signal2 > 0)[0]
            sig_lo2, sig_hi2 = (_nz2[0], _nz2[-1]) if len(_nz2) else (0, len(signal2) - 1)

        # --- Trim all arrays to the nonzero signal region ---
        signal = signal[sig_lo:sig_hi + 1]
        signal_std = signal_std[sig_lo:sig_hi + 1]
        coverage = coverage[sig_lo:sig_hi + 1]
        channel_time_windows = channel_time_windows[sig_lo:sig_hi + 1]
        channel_edges = channel_edges[sig_lo:sig_hi + 2]
        if neutron_bg_per_channel is not None:
            neutron_bg_per_channel = neutron_bg_per_channel[sig_lo:sig_hi + 1]
            photon_bg_per_channel = photon_bg_per_channel[sig_lo:sig_hi + 1]
            neutron_bg_std_per_channel = neutron_bg_std_per_channel[sig_lo:sig_hi + 1]
            photon_bg_std_per_channel = photon_bg_std_per_channel[sig_lo:sig_hi + 1]
        if is_dual:
            signal2 = signal2[sig_lo2:sig_hi2 + 1]
            signal_std2 = signal_std2[sig_lo2:sig_hi2 + 1]
            coverage2 = coverage2[sig_lo2:sig_hi2 + 1]
            channel_time_windows2 = channel_time_windows2[sig_lo2:sig_hi2 + 1]
            channel_edges2 = channel_edges2[sig_lo2:sig_hi2 + 2]
            if neutron_bg_per_channel2 is not None:
                neutron_bg_per_channel2 = neutron_bg_per_channel2[sig_lo2:sig_hi2 + 1]
                photon_bg_per_channel2 = photon_bg_per_channel2[sig_lo2:sig_hi2 + 1]
                neutron_bg_std_per_channel2 = neutron_bg_std_per_channel2[sig_lo2:sig_hi2 + 1]
                photon_bg_std_per_channel2 = photon_bg_std_per_channel2[sig_lo2:sig_hi2 + 1]

        # --- Derive per-plot filenames from base filename ---
        base, ext = os.path.splitext(filename)
        filename_counts = filename
        filename_sb = f'{base}_sb{ext}'
        filename_coverage = f'{base}_coverage{ext}'

        particle_label = self.spectrometer.conversion_foil.particle
        detector_used = self.spectrometer.hodoscope.detector_used
        if detector_used:
            label = f'$E_{{dep}}$ [{"MeV" if incident_particle_yield else "MeV/source"}]'
        else:
            label = f'Counts [{"particles" if incident_particle_yield else "particles/source"}]'

        # Build position→energy interpolant from the performance curve (optional).
        # x_to_en: cm → MeV,  en_to_x: MeV → cm
        _x_to_en = _en_to_x = None
        _x_to_en2 = _en_to_x2 = None
        perf_df = self.performance_analyzer._load_performance_curve(performance_curve_file)
        if perf_df is not None:
            _pos_cm = perf_df['position [m]'].values * 100  # m → cm
            _en_mev = perf_df['energy [MeV]'].values
            # keep only the CH2 foil rows if dual-foil data is present
            if 'foil' in perf_df.columns:
                primary_foil = self.spectrometer.conversion_foil.foil_material
                mask = perf_df['foil'] == primary_foil
                if mask.any():
                    _pos_cm = _pos_cm[mask.values]
                    _en_mev = _en_mev[mask.values]
            _x_to_en = interp1d(_pos_cm, _en_mev, bounds_error=False, fill_value='extrapolate')
            _en_to_x = interp1d(_en_mev, _pos_cm, bounds_error=False, fill_value='extrapolate')
            # Build a second interpolant for the CD2 (deuteron) foil in dual-foil mode
            if self.dual_data is not None and 'foil' in perf_df.columns:
                secondary_foil = self.dual_data['spectrometer'].conversion_foil.foil_material
                mask2 = perf_df['foil'] == secondary_foil
                if mask2.any():
                    _pos_cm2 = perf_df['position [m]'].values[mask2.values] * 100
                    _en_mev2 = perf_df['energy [MeV]'].values[mask2.values]
                    _x_to_en2 = interp1d(_pos_cm2, _en_mev2, bounds_error=False, fill_value='extrapolate')
                    _en_to_x2 = interp1d(_en_mev2, _pos_cm2, bounds_error=False, fill_value='extrapolate')

        def _add_energy_axis(ax, which: Literal['all', 'primary', 'secondary'] = 'all'):
            """Add twin top x-axis(es) showing incident neutron energy in MeV.

            In dual-foil mode two axes are added (one per foil), each colored to match
            the corresponding signal line.  The deuteron axis is offset outward so the
            two labels do not overlap.

            Args:
                which: 'all' adds all axes; 'primary' adds only the CH2 axis;
                       'secondary' adds only the CD2 axis (no offset since it is alone).
            """
            if _x_to_en is None or _en_to_x is None:
                return
            x_lo, x_hi = ax.get_xlim()

            def _make_twin(x_to_en, en_to_x, xlabel, color=None, offset=0, tick_step=None):
                e_lo, e_hi = float(x_to_en(x_lo)), float(x_to_en(x_hi))
                e_min, e_max = min(e_lo, e_hi), max(e_lo, e_hi)
                e_span = e_max - e_min
                step = tick_step if tick_step is not None else 10 ** np.floor(np.log10(e_span / 4))
                tick_energies = np.arange(np.ceil(e_min / step) * step,
                                          np.floor(e_max / step) * step + step * 0.5,
                                          step)
                tick_positions = en_to_x(tick_energies)
                ax_top = ax.twiny()
                ax_top.set_xlim(ax.get_xlim())
                ax_top.set_xticks(tick_positions)
                ax_top.set_xticklabels([f'{e:.3g}' for e in tick_energies])
                ax_top.set_xlabel(xlabel)
                if offset:
                    ax_top.spines['top'].set_position(('outward', offset))
                if color is not None:
                    ax_top.xaxis.label.set_color(color)
                    ax_top.tick_params(axis='x', colors=color)
                    ax_top.spines['top'].set_edgecolor(color)

            inc = self.spectrometer.conversion_foil.incident_particle.capitalize()
            if is_dual:
                if which in ('all', 'primary'):
                    _make_twin(_x_to_en, _en_to_x,
                               f'{inc} Energy [MeV] (p)',
                               color=self.primary_color)
                if which in ('all', 'secondary') and self.dual_data is not None and _x_to_en2 is not None and _en_to_x2 is not None:
                    _make_twin(_x_to_en2, _en_to_x2,
                               f'{inc} Energy [MeV] (d)',
                               color=self.dual_data['secondary_color'],
                               offset=45 if which == 'all' else 0,
                               tick_step=1.0)
            else:
                _make_twin(_x_to_en, _en_to_x, f'{inc} Energy [MeV]')

        def _step(ax, edges, values, **kwargs):
            return ax.step(edges, np.append(values, values[-1]), where='pre', **kwargs)[0]

        def _step_band(ax, edges, values, sigma, color, alpha=0.25):
            """Draw a +/-1 sigma shaded band around a step-function line."""
            if np.all(np.isnan(sigma)):
                return
            y_lo = np.append(values - sigma, (values - sigma)[-1])
            y_hi = np.append(values + sigma, (values + sigma)[-1])
            ax.fill_between(edges, y_lo, y_hi, step='pre', color=color, alpha=alpha, linewidth=0)

        # Plot 1: counts
        # In dual-foil mode background labels distinguish the CH2 and CD2 halves.
        n_label = 'neutron (p)' if is_dual else 'neutron'
        g_label = 'photon (p)' if is_dual else 'photon'
        _label_kw = dict(fontsize=18, ha='left', va='center')
        _stroke = [pe.withStroke(linewidth=3, foreground='white')]
        signal_std2 = signal_std2 if signal2 is not None else None
        if is_dual:
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            _step_band(ax_top, channel_edges, signal, signal_std, color=self.primary_color)
            _step(ax_top, channel_edges, signal,
                  color=self.primary_color, label=particle_label, linewidth=3)
            if neutron_bg_per_channel is not None:
                if neutron_bg_std_per_channel is not None:
                    _step_band(ax_top, channel_edges, neutron_bg_per_channel,
                               neutron_bg_std_per_channel, color='tab:green')
                _step(ax_top, channel_edges, neutron_bg_per_channel,
                      color='tab:green', label='neutron', linewidth=3)
            if photon_bg_per_channel is not None:
                if photon_bg_std_per_channel is not None:
                    _step_band(ax_top, channel_edges, photon_bg_per_channel,
                               photon_bg_std_per_channel, color='tab:purple')
                _step(ax_top, channel_edges, photon_bg_per_channel,
                      color='tab:purple', label='photon', linewidth=3)
            ax_top.set_yscale('log')
            ax_top.set_ylabel(label)
            ax_top.grid(True, alpha=0.3)
            ax_top.text(0.2, 0.82, particle_label, color=self.primary_color,
                        transform=ax_top.transAxes, **_label_kw).set_path_effects(_stroke)
            if neutron_bg_per_channel is not None:
                ax_top.text(0.38, 0.6, 'neutron', color='tab:green',
                            transform=ax_top.transAxes, **_label_kw).set_path_effects(_stroke)
            if photon_bg_per_channel is not None:
                ax_top.text(0.74, 0.47, 'photon', color='tab:purple',
                            transform=ax_top.transAxes, **_label_kw).set_path_effects(_stroke)
            _add_energy_axis(ax_top, which='primary')

            if signal2 is not None:
                _step_band(ax_bot, channel_edges2, signal2, signal_std2,
                           color=self.dual_data['secondary_color'])
                _step(ax_bot, channel_edges2, signal2,
                      color=self.dual_data['secondary_color'],
                      label=self.dual_data['secondary_label'], linewidth=3)
            if neutron_bg_per_channel2 is not None:
                if neutron_bg_std_per_channel2 is not None:
                    _step_band(ax_bot, channel_edges2, neutron_bg_per_channel2,
                               neutron_bg_std_per_channel2, color='tab:green')
                _step(ax_bot, channel_edges2, neutron_bg_per_channel2,
                      color='tab:green', label='neutron', linewidth=3)
            if photon_bg_per_channel2 is not None:
                if photon_bg_std_per_channel2 is not None:
                    _step_band(ax_bot, channel_edges2, photon_bg_per_channel2,
                               photon_bg_std_per_channel2, color='tab:purple')
                _step(ax_bot, channel_edges2, photon_bg_per_channel2,
                      color='tab:purple', label='photon', linewidth=3)
            ax_bot.set_yscale('log')
            ax_bot.set_xlabel('Horizontal Position [cm]')
            ax_bot.set_ylabel(label)
            ax_bot.grid(True, alpha=0.3)
            if signal2 is not None:
                ax_bot.text(0.5, 0.83, self.dual_data['secondary_label'],
                            color=self.dual_data['secondary_color'],
                            transform=ax_bot.transAxes, **_label_kw).set_path_effects(_stroke)
            if neutron_bg_per_channel2 is not None:
                ax_bot.text(0.35, 0.3, 'neutron', color='tab:green',
                            transform=ax_bot.transAxes, **_label_kw).set_path_effects(_stroke)
            if photon_bg_per_channel2 is not None:
                ax_bot.text(0.68, 0.37, 'photon', color='tab:purple',
                            transform=ax_bot.transAxes, **_label_kw).set_path_effects(_stroke)
            _add_energy_axis(ax_bot, which='secondary')

            fig.tight_layout()
            fig.savefig(filename_counts, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            fig, ax_counts = plt.subplots(figsize=(10, 4))
            _step_band(ax_counts, channel_edges, signal, signal_std, color=self.primary_color)
            _step(ax_counts, channel_edges, signal,
                  color=self.primary_color, label=particle_label, linewidth=3)
            if neutron_bg_per_channel is not None:
                if neutron_bg_std_per_channel is not None:
                    _step_band(ax_counts, channel_edges, neutron_bg_per_channel,
                               neutron_bg_std_per_channel, color='tab:green')
                _step(ax_counts, channel_edges, neutron_bg_per_channel,
                      color='tab:green', label='neutron', linewidth=3)
            if photon_bg_per_channel is not None:
                if photon_bg_std_per_channel is not None:
                    _step_band(ax_counts, channel_edges, photon_bg_per_channel,
                               photon_bg_std_per_channel, color='tab:purple')
                _step(ax_counts, channel_edges, photon_bg_per_channel,
                      color='tab:purple', label='photon', linewidth=3)
            ax_counts.set_yscale('log')
            ax_counts.set_xlabel('Horizontal Position [cm]')
            ax_counts.set_ylabel(label)
            ax_counts.grid(True, alpha=0.3)

            ax_counts.text(0.15, 0.82, particle_label, color=self.primary_color,
                           transform=ax_counts.transAxes, **_label_kw).set_path_effects(_stroke)
            if neutron_bg_per_channel is not None:
                ax_counts.text(0.55, 0.55, 'neutron', color='tab:green',
                               transform=ax_counts.transAxes, **_label_kw).set_path_effects(_stroke)
            if photon_bg_per_channel is not None:
                ax_counts.text(0.80, 0.35, 'photon', color='tab:purple',
                               transform=ax_counts.transAxes, **_label_kw).set_path_effects(_stroke)
            _add_energy_axis(ax_counts)
            fig.tight_layout()
            fig.savefig(filename_counts, dpi=150, bbox_inches='tight')
            plt.close(fig)
        print(f'Position histogram saved to {filename_counts}')

        def _sb_and_sigma(sig, bg, bg_std):
            """Return (log10_sb, sigma_log10_sb) with NaN where undefined.

            sigma_log10_sb = sqrt(1/sig + (bg_std/bg)^2) / ln(10)
            from independent Poisson signal and MC background uncertainty.
            """
            valid = (sig > 0) & (bg > 0)
            sb = np.where(valid, sig / bg, np.nan)
            sigma_rel_sq = np.where(valid, 1.0 / sig + (bg_std / np.where(bg > 0, bg, 1.0)) ** 2, np.nan)
            sigma_log10 = np.where(valid, np.sqrt(sigma_rel_sq) / np.log(10), np.nan)
            return np.log10(sb), sigma_log10

        def _step_band_linear(ax, edges, values, sigma, color, alpha=0.25):
            """Draw a +/-1 sigma band around a step-function line on a linear-scale axis."""
            y_lo = np.append(values - sigma, (values - sigma)[-1])
            y_hi = np.append(values + sigma, (values + sigma)[-1])
            ax.fill_between(edges, y_lo, y_hi, step='pre', color=color, alpha=alpha, linewidth=0)

        # Plot 2 (optional): S/B — separate lines for neutron and photon backgrounds.
        # Single-foil: gated S/B (solid) vs non-gated S/B (dashed) comparison.
        # Dual-foil: CH2 (solid) and CD2 (dashed) gated S/B; no-gate overlay omitted to
        # keep the plot readable.
        if neutron_bg_per_channel is not None and photon_bg_per_channel is not None:
            fig, ax_sb = plt.subplots(figsize=(10, 5.5 if is_dual else 4))

            _n_std = neutron_bg_std_per_channel if neutron_bg_std_per_channel is not None else np.zeros_like(neutron_bg_per_channel)
            _ph_std = photon_bg_std_per_channel if photon_bg_std_per_channel is not None else np.zeros_like(photon_bg_per_channel)

            if hodoscope.use_time_gating and neutron_background_vs_time is not None and photon_background_vs_time is not None:
                if is_dual:
                    # Gated S/B for CH2 (solid) and CD2 (dashed).
                    log10_sb_n_ch2, sigma_log10_sb_n_ch2 = _sb_and_sigma(signal, neutron_bg_per_channel, _n_std)
                    _step_band_linear(ax_sb, channel_edges, log10_sb_n_ch2, sigma_log10_sb_n_ch2, color='tab:green')
                    _step(ax_sb, channel_edges, log10_sb_n_ch2,
                          color='tab:green', linestyle='-', label='neutron (p)', linewidth=3)
                    log10_sb_ph_ch2, sigma_log10_sb_ph_ch2 = _sb_and_sigma(signal, photon_bg_per_channel, _ph_std)
                    _step_band_linear(ax_sb, channel_edges, log10_sb_ph_ch2, sigma_log10_sb_ph_ch2, color='tab:purple')
                    _step(ax_sb, channel_edges, log10_sb_ph_ch2,
                          color='tab:purple', linestyle='-', label='photon (p)', linewidth=3)
                    if neutron_bg_per_channel2 is not None and photon_bg_per_channel2 is not None and signal2 is not None:
                        _n_std2 = neutron_bg_std_per_channel2 if neutron_bg_std_per_channel2 is not None else np.zeros_like(neutron_bg_per_channel2)
                        _ph_std2 = photon_bg_std_per_channel2 if photon_bg_std_per_channel2 is not None else np.zeros_like(photon_bg_per_channel2)
                        log10_sb_n_cd2, sigma_log10_sb_n_cd2 = _sb_and_sigma(signal2, neutron_bg_per_channel2, _n_std2)
                        _step_band_linear(ax_sb, channel_edges2, log10_sb_n_cd2, sigma_log10_sb_n_cd2, color='tab:green')
                        _step(ax_sb, channel_edges2, log10_sb_n_cd2,
                              color='tab:green', linestyle='--', label='neutron (d)', linewidth=3)
                        log10_sb_ph_cd2, sigma_log10_sb_ph_cd2 = _sb_and_sigma(signal2, photon_bg_per_channel2, _ph_std2)
                        _step_band_linear(ax_sb, channel_edges2, log10_sb_ph_cd2, sigma_log10_sb_ph_cd2, color='tab:purple')
                        _step(ax_sb, channel_edges2, log10_sb_ph_cd2,
                              color='tab:purple', linestyle='--', label='photon (d)', linewidth=3)
                else:
                    # Gated S/B: use the per-channel time-windowed background.
                    log10_sb_n, sigma_log10_sb_n = _sb_and_sigma(signal, neutron_bg_per_channel, _n_std)
                    _step_band_linear(ax_sb, channel_edges, log10_sb_n, sigma_log10_sb_n, color='tab:green')
                    _step(ax_sb, channel_edges, log10_sb_n,
                          color='tab:green', linestyle='-', label='neutron', linewidth=3)
                    log10_sb_ph, sigma_log10_sb_ph = _sb_and_sigma(signal, photon_bg_per_channel, _ph_std)
                    _step_band_linear(ax_sb, channel_edges, log10_sb_ph, sigma_log10_sb_ph, color='tab:purple')
                    _step(ax_sb, channel_edges, log10_sb_ph,
                          color='tab:purple', linestyle='-', label='photon', linewidth=3)
            else:
                log10_sb_n, sigma_log10_sb_n = _sb_and_sigma(signal, neutron_bg_per_channel, _n_std)
                _step_band_linear(ax_sb, channel_edges, log10_sb_n, sigma_log10_sb_n, color='tab:green')
                _step(ax_sb, channel_edges, log10_sb_n,
                      color='tab:green', label=n_label, linewidth=3)
                log10_sb_ph, sigma_log10_sb_ph = _sb_and_sigma(signal, photon_bg_per_channel, _ph_std)
                _step_band_linear(ax_sb, channel_edges, log10_sb_ph, sigma_log10_sb_ph, color='tab:purple')
                _step(ax_sb, channel_edges, log10_sb_ph,
                      color='tab:purple', label=g_label, linewidth=3)
                if neutron_bg_per_channel2 is not None and photon_bg_per_channel2 is not None and signal2 is not None:
                    _n_std2 = neutron_bg_std_per_channel2 if neutron_bg_std_per_channel2 is not None else np.zeros_like(neutron_bg_per_channel2)
                    _ph_std2 = photon_bg_std_per_channel2 if photon_bg_std_per_channel2 is not None else np.zeros_like(photon_bg_per_channel2)
                    log10_sb_n_cd2, sigma_log10_sb_n_cd2 = _sb_and_sigma(signal2, neutron_bg_per_channel2, _n_std2)
                    _step_band_linear(ax_sb, channel_edges2, log10_sb_n_cd2, sigma_log10_sb_n_cd2, color='tab:green')
                    _step(ax_sb, channel_edges2, log10_sb_n_cd2,
                          color='tab:green', linestyle='--', label='neutron (d)', linewidth=3)
                    log10_sb_ph_cd2, sigma_log10_sb_ph_cd2 = _sb_and_sigma(signal2, photon_bg_per_channel2, _ph_std2)
                    _step_band_linear(ax_sb, channel_edges2, log10_sb_ph_cd2, sigma_log10_sb_ph_cd2, color='tab:purple')
                    _step(ax_sb, channel_edges2, log10_sb_ph_cd2,
                          color='tab:purple', linestyle='--', label='photon (d)', linewidth=3)

            ax_sb.set_xlabel('Horizontal Position [cm]')
            ax_sb.set_ylabel('log$_{10}$(S/B)')
            ax_sb.grid(True, alpha=0.3)
            valid_lines = [l for l in ax_sb.get_lines() if not np.all(np.isnan(l.get_ydata()))]
            if valid_lines:
                labelLines(valid_lines, align=False)
            _add_energy_axis(ax_sb)
            fig.tight_layout()
            fig.savefig(filename_sb, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'S/B plot saved to {filename_sb}')

        # Plot 3: signal coverage
        fig, ax_coverage = plt.subplots(figsize=(10, 5.5 if is_dual else 4))
        ax_coverage.stairs(coverage * 100, channel_edges, baseline=None,
                           color=self.primary_color, alpha=0.7, linewidth=3)
        if self.dual_data and coverage2 is not None and channel_edges2 is not None:
            ax_coverage.stairs(coverage2 * 100, channel_edges2, baseline=None,
                               color=self.dual_data['secondary_color'], alpha=0.5, linewidth=3)
        ax_coverage.set_xlabel('Horizontal Position [cm]')
        ax_coverage.set_ylabel('Signal Coverage [%]')
        ax_coverage.set_ylim(0, 105)
        ax_coverage.grid(True, alpha=0.3)
        _add_energy_axis(ax_coverage)
        fig.tight_layout()
        fig.savefig(filename_coverage, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Coverage plot saved to {filename_coverage}')

        # Plot 4 (time-gating only): ridgeline PDF of per-channel detector arrival times.
        # Pass background arrays if available so they are overlaid on the twin y-axis.
        if hodoscope.use_time_gating:
            filename_time_windows = f'{base}_time_windows{ext}'
            self._plot_time_ridgeline(
                filename_time_windows,
                time_bins=time_bins,
                neutron_background_vs_time=neutron_background_vs_time,
                neutron_background_std_vs_time=neutron_background_std_vs_time,
                photon_background_vs_time=photon_background_vs_time,
                photon_background_std_vs_time=photon_background_std_vs_time,
            )

        # Plot 5 (time-gating only, background data required): background E_dep vs time.
        if hodoscope.use_time_gating and time_bins is not None and neutron_background_vs_time is not None and photon_background_vs_time is not None:
            filename_bg_time = f'{base}_background_vs_time{ext}'
            fig, ax_bgt = plt.subplots(figsize=(10, 4))
            time_ns_bg = time_bins * 1e9
            ax_bgt.step(time_ns_bg, neutron_background_vs_time, where='mid',
                        color='tab:green', linewidth=3, label='neutron')
            ax_bgt.step(time_ns_bg, photon_background_vs_time, where='mid',
                        color='tab:purple', linewidth=3, label='photon')
            labelLines(ax_bgt.get_lines(), align=False)
            ax_bgt.set_xlabel('Time [ns]')
            ax_bgt.set_ylabel('$E_{dep}$ [MeV/cm$^2$/source]')
            ax_bgt.set_yscale('log')
            ax_bgt.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(filename_bg_time, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Background vs time plot saved to {filename_bg_time}')

    def _plot_time_ridgeline(
        self,
        filename: str,
        n_kde_points: int = 300,
        time_bins: Optional[np.ndarray] = None,
        neutron_background_vs_time: Optional[np.ndarray] = None,
        neutron_background_std_vs_time: Optional[np.ndarray] = None,
        photon_background_vs_time: Optional[np.ndarray] = None,
        photon_background_std_vs_time: Optional[np.ndarray] = None,
    ) -> None:
        """Ridgeline plot of the detector arrival-time PDF for each hodoscope channel.

        Each channel's PDF is estimated with kernel density estimation (KDE) and drawn
        as a smooth filled curve offset vertically by its channel index, so timing shifts
        and distribution shapes can be compared across the focal plane at a glance.

        Optionally overlays neutron and photon background E_dep vs time on a second
        log y-axis (right), sharing the same time x-axis.

        Args:
            filename: Output path for the saved figure.
            n_kde_points: Number of points on the evaluation grid (default 300).
            time_bins: 1-D array of background time bin centres in seconds.
            neutron_background_vs_time: Background neutron E_dep per bin [MeV/cm^2/source].
            photon_background_vs_time: Background photon E_dep per bin [MeV/cm^2/source].
        """
        hodoscope = self.spectrometer.hodoscope
        n_channels = hodoscope.total_channels
        is_dual = self.dual_data is not None

        def _collect_times(beam, hod) -> list:
            """Return per-channel accepted arrival times (ns) for one foil's output beam."""
            arr_s = beam[:, 4]
            x_cm = beam[:, 0] * 100
            y_cm = beam[:, 2] * 100
            local_y_ctrs_cm = hod.channel_y_centers * 100
            local_heights_cm = hod.channel_heights * 100
            local_edges_cm = hod.channel_edges * 100
            idx = np.digitize(x_cm, local_edges_cm) - 1
            times_per_channel = []
            for i in range(hod.total_channels):
                in_bin = idx == i
                y_ok = np.abs(y_cm - local_y_ctrs_cm[i]) <= local_heights_cm[i] / 2
                times_per_channel.append(arr_s[in_bin & y_ok] * 1e9)
            return times_per_channel

        # Collect arrival times for each foil using each hodoscope's own y_center + channel_height.
        channel_times_ch2 = _collect_times(self.spectrometer.output_beam, hodoscope)
        channel_times_cd2: list = []
        if self.dual_data is not None:
            channel_times_cd2 = _collect_times(
                self.dual_data['spectrometer'].output_beam,
                self.dual_data['spectrometer'].hodoscope,
            )

        # Build a common time grid spanning all accepted arrival times.
        all_lists = channel_times_ch2 + channel_times_cd2
        all_times = np.concatenate([t for t in all_lists if len(t) > 0])
        global_t_min, global_t_max = all_times.min(), all_times.max()
        t_grid = np.linspace(global_t_min, global_t_max, n_kde_points)

        def _build_pdfs(channel_times):
            pdfs = []
            for times in channel_times:
                if len(times) > 1:
                    pdfs.append(gaussian_kde(times)(t_grid))
                else:
                    pdfs.append(np.zeros(n_kde_points))
            return pdfs

        pdfs_ch2 = _build_pdfs(channel_times_ch2)
        pdfs_cd2 = _build_pdfs(channel_times_cd2) if is_dual else []

        max_pdf = max(
            (p.max() for p in pdfs_ch2 + pdfs_cd2 if p.max() > 0), default=1.0
        )
        # Overlap: each ridge can grow up to 3 channel-index units tall.
        ridge_scale = 3.0 / max_pdf

        # CH2 (proton) ridges in red tones; CD2 (deuteron) ridges in blue tones.
        colors_ch2 = ['darkred', 'salmon']
        colors_cd2 = ['darkblue', 'steelblue']

        def _draw_ridgelines(ax, pdfs, colors, alpha=0.5):
            for i, pdf in enumerate(pdfs):
                pdf_scaled = pdf * ridge_scale
                if pdf_scaled.max() == 0:
                    continue
                color = colors[i % 2]
                # Clip near-zero tails (KDE has infinite support).
                active = pdf_scaled > pdf_scaled.max() * 1e-3
                x_fill = np.concatenate([[t_grid[active][0]], t_grid[active], [t_grid[active][-1]]])
                y_fill = np.concatenate([[i], i + pdf_scaled[active], [i]])
                ax.fill_between(x_fill, i, y_fill, alpha=alpha, color=color)
                ax.plot(x_fill, y_fill, color=color, linewidth=0.8)

        fig, ax = plt.subplots(figsize=(10, 4) if is_dual else (8, 6))

        if is_dual:
            # Second left y-axis for CD2 so each foil's channels span the full
            # figure height independently rather than sharing one scale.
            ax2 = ax.twinx()
            ax2.yaxis.set_label_position('left')
            ax2.yaxis.tick_left()
            ax2.spines['left'].set_position(('outward', 60))
            ax2.spines['left'].set_visible(True)
            ax2.spines['right'].set_visible(False)
            ax.spines['right'].set_visible(False)

            _draw_ridgelines(ax, pdfs_ch2, colors_ch2, alpha=0.5)
            _draw_ridgelines(ax2, pdfs_cd2, colors_cd2, alpha=0.4)

            particle_ch2 = self.spectrometer.conversion_foil.particle
            particle_cd2 = self.dual_data['spectrometer'].conversion_foil.particle
            ax.set_ylabel(f'Channel index ({particle_ch2})', color=colors_ch2[0])
            ax.tick_params(axis='y', colors=colors_ch2[0])
            ax.spines['left'].set_color(colors_ch2[0])
            ax2.set_ylabel(f'Channel index ({particle_cd2})', color=colors_cd2[0])
            ax2.tick_params(axis='y', colors=colors_cd2[0])
            ax2.spines['left'].set_color(colors_cd2[0])
        else:
            _draw_ridgelines(ax, pdfs_ch2, colors_ch2, alpha=0.5)
            ax.set_ylim(-0.5, n_channels - 0.5 + ridge_scale)
            ax.set_ylabel('Channel index')

        ax.set_xlabel('Detector arrival time [ns]')
        ax.grid(True, alpha=0.3)

        _label_kw = dict(fontsize=13, ha='left', va='center')
        _stroke = [pe.withStroke(linewidth=3, foreground='white')]

        ax.text(0.27, 0.3, self.spectrometer.conversion_foil.particle,
                transform=ax.transAxes, color=colors_ch2[0],
                **_label_kw).set_path_effects(_stroke)
        if is_dual and self.dual_data is not None:
            ax2.text(0.66, 0.7, self.dual_data['spectrometer'].conversion_foil.particle,
                     transform=ax2.transAxes, color=colors_cd2[0],
                     **_label_kw).set_path_effects(_stroke)

        # Overlay background E_dep vs time on a twin log y-axis (right),
        # restricted to the signal arrival window [global_t_min, global_t_max].
        if time_bins is not None and neutron_background_vs_time is not None and photon_background_vs_time is not None:
            time_ns_bg = time_bins * 1e9
            dt = float(np.median(np.diff(time_ns_bg))) if len(time_ns_bg) > 1 else 1.0
            bin_left = time_ns_bg - dt / 2
            bin_right = time_ns_bg + dt / 2
            overlap = np.maximum(0.0, np.minimum(bin_right, global_t_max) - np.maximum(bin_left, global_t_min)) / dt
            mask = overlap > 0
            ax_bg = ax.twinx()
            n_vals = neutron_background_vs_time[mask] * overlap[mask]
            ph_vals = photon_background_vs_time[mask] * overlap[mask]
            ax_bg.step(time_ns_bg[mask], n_vals, where='mid',
                       color='tab:green', linewidth=2, label='neutron', alpha=0.8)
            ax_bg.step(time_ns_bg[mask], ph_vals, where='mid',
                       color='tab:purple', linewidth=2, label='photon', alpha=0.8)
            if neutron_background_std_vs_time is not None:
                n_std = neutron_background_std_vs_time[mask] * overlap[mask]
                ax_bg.fill_between(time_ns_bg[mask],
                                   n_vals - n_std,
                                   n_vals + n_std,
                                   step='mid', color='tab:green', alpha=0.25, linewidth=0)
            if photon_background_std_vs_time is not None:
                ph_std = photon_background_std_vs_time[mask] * overlap[mask]
                ax_bg.fill_between(time_ns_bg[mask],
                                   ph_vals - ph_std,
                                   ph_vals + ph_std,
                                   step='mid', color='tab:purple', alpha=0.25, linewidth=0)
            ax_bg.set_yscale('log')
            _sig_vals = np.concatenate([n_vals[n_vals > 0], ph_vals[ph_vals > 0]])
            if len(_sig_vals) > 0:
                ax_bg.set_ylim(bottom=_sig_vals.min() * 0.5)
            ax_bg.set_ylabel('$E_{dep}$ [MeV/cm$^2$/source]')
            ax_bg.text(0.90, 0.52, 'neutron', transform=ax_bg.transAxes,
                       color='tab:green', fontsize=13, ha='right', va='center').set_path_effects(_stroke)
            ax_bg.text(0.25, 0.93, 'photon', transform=ax_bg.transAxes,
                       color='tab:purple', fontsize=13, ha='right', va='center').set_path_effects(_stroke)

        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Time windows plot saved to {filename}')

    def plot_input_ray_geometry(self, filename: Optional[str] = None) -> None:
        """
        Draw the input beam ray geometry showing rays from foil to aperture.
        
        Args:
            filename: Output filename for the plot
        """
        if filename == None:
            filename = f'{self.spectrometer.figure_directory}/input_ray_geometry.png'
        
        if len(self.spectrometer.input_beam) == 0:
            raise ValueError("No input beam data available. Generate rays first.")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Draw foil and aperture boundaries
        # Convert all lengths to cm
        foil_radius = self.spectrometer.conversion_foil.foil_radius * 100
        aperture_distance = self.spectrometer.conversion_foil.aperture_distance * 100
        if self.spectrometer.conversion_foil.aperture_type == 'circ':
            aperture_half_height = self.spectrometer.conversion_foil.aperture_radius * 100
        else:
            aperture_half_height = self.spectrometer.conversion_foil.aperture_height * 100 / 2

        # Foil (vertical line at z=0)
        ax.vlines(0, -foil_radius, foil_radius, color='tab:purple', label='Conversion Foil')
        # Add text label for foil
        ax.text(
            0,
            foil_radius,
            'Foil',
            ha='left',
            va='bottom',
            color='tab:purple',
            fontsize=12
        )

        # Aperture (vertical line at aperture distance)
        ax.vlines(aperture_distance, -aperture_half_height, aperture_half_height,
                color='tab:orange', label='Aperture')
        # Add text label for aperture
        ax.text(
            aperture_distance,
            aperture_half_height,
            'Aperture',
            ha='right',
            va='bottom',
            color='tab:orange',
            fontsize=12
        )
        
        particle_rest_energy = self.spectrometer.conversion_foil.particle_mass * MASS_TO_MEV  # MeV
        reference_gamma = 1 + self.spectrometer.reference_energy / particle_rest_energy  # Lorentz factor of the reference particle

        # Draw sample of input rays
        num_rays_to_plot = min(len(self.spectrometer.input_beam), 200)  # Limit for clarity
        z_coords = np.linspace(0, aperture_distance, 20)
        
        for i in range(0, len(self.spectrometer.input_beam), max(1, len(self.spectrometer.input_beam) // num_rays_to_plot)):
            ray = self.spectrometer.input_beam[i]
            x0, p_x_relative, y0, p_y_relative, _, energy_relative, *_ = ray
            y0 *= 100 # cm

            # Calculate ray trajectory
            energy = self.spectrometer.reference_energy * (1 + energy_relative)  # MeV
            gamma = 1 + energy/particle_rest_energy  # Lorentz factor of the particle
            p_relative = np.sqrt((gamma**2 - 1)/(reference_gamma**2 - 1))  # the particle's momentum as a fraction of the reference particle's momentum
            slope = np.tan(np.arcsin(p_y_relative/p_relative))
            y_trajectory = slope * z_coords + y0

            ax.plot(z_coords, y_trajectory, alpha=0.4, color=self.primary_color, linewidth=0.5)

        # Plot dual data if available
        if self.dual_data:
            spec2: MPRSpectrometer = self.dual_data['spectrometer']
            particle_rest_energy_cd2 = spec2.conversion_foil.particle_mass * MASS_TO_MEV
            reference_gamma_cd2 = 1 + spec2.reference_energy / particle_rest_energy_cd2
            for i in range(0, len(spec2.input_beam), max(1, len(spec2.input_beam) // num_rays_to_plot)):
                ray = spec2.input_beam[i]
                x0, p_x_relative, y0, p_y_relative, _, energy_relative, *_ = ray
                y0 *= 100 # cm

                # Calculate ray trajectory (same rigorous relativistic calculation as CH2)
                energy2 = spec2.reference_energy * (1 + energy_relative)
                gamma2 = 1 + energy2 / particle_rest_energy_cd2
                p_relative2 = np.sqrt((gamma2**2 - 1) / (reference_gamma_cd2**2 - 1))
                slope = np.tan(np.arcsin(p_y_relative / p_relative2))
                y_trajectory = slope * z_coords + y0

                ax.plot(z_coords, y_trajectory, alpha=0.4, color=self.dual_data['secondary_color'], linewidth=0.5)
            
            # Add text labels for dual rays
            ax.text(
                aperture_distance/2,
                foil_radius,
                self.dual_data['primary_label'],
                ha='center',
                va='bottom',
                color=self.primary_color,
                fontsize=12
            )
            ax.text(
                aperture_distance/2,
                -foil_radius,
                self.dual_data['secondary_label'],
                ha='center',
                va='top',
                color=self.dual_data['secondary_color'],
                fontsize=12
            )
        
        ax.set_xlabel('Z Distance [cm]')
        ax.set_ylabel('Y Position [cm]')
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        ax.set_xlim(-0.1 * aperture_distance, 1.1 * aperture_distance)
        max_extent = 1.5 * max(foil_radius, aperture_half_height)
        ax.set_ylim(-max_extent, max_extent)
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Input ray geometry plot saved to {filename}')

    def _overlay_energy_contours(self, ax, spec, X_mesh, Y_mesh, dx, dy) -> None:
        """Overlay white contour lines of mean recoil energy on an existing heatmap axes."""
        x_cm = spec.output_beam[:, 0] * 100
        y_cm = spec.output_beam[:, 2] * 100
        energies = spec.reference_energy * (1 + spec.output_beam[:, 5])
        x_coords = X_mesh[0, :]
        y_coords = Y_mesh[:, 0]
        x_edges = np.concatenate([[x_coords[0] - dx / 2], x_coords + dx / 2])
        y_edges = np.concatenate([[y_coords[0] - dy / 2], y_coords + dy / 2])
        energy_sum, _, _ = np.histogram2d(x_cm, y_cm, bins=[x_edges, y_edges], weights=energies)
        count_hist, _, _ = np.histogram2d(x_cm, y_cm, bins=[x_edges, y_edges])
        mean_E = np.where(count_hist.T > 0, energy_sum.T / count_hist.T, np.nan)

        particle = spec.conversion_foil.particle
        if particle == 'deuteron':
            step, fmt, contour_color = 0.5, '%.1f', '#C04000'
        else:
            step, fmt, contour_color = 1.0, '%.0f', 'midnightblue'

        e_min = np.floor(np.nanmin(mean_E) / step) * step
        e_max = np.ceil(np.nanmax(mean_E) / step) * step
        levels = np.arange(e_min, e_max + step / 2, step)

        cs = ax.contour(x_coords, y_coords, mean_E,
                        levels=levels, colors=contour_color, linewidths=2.0, alpha=0.85)

        # Place each label at the midpoint of its own contour's y-range, interpolating
        # x along the contour at that y. Labels will naturally slope across levels.
        manual_positions = []
        for level_segs in cs.allsegs:
            # Find the longest segment for this level to determine target_y
            best_seg = None
            best_span = 0.0
            for seg in level_segs:
                if len(seg) < 2:
                    continue
                span = float(seg[:, 1].max() - seg[:, 1].min())
                if span > best_span:
                    best_span = span
                    best_seg = seg
            if best_seg is None:
                continue
            target_y = float((best_seg[:, 1].max() + best_seg[:, 1].min()) / 2)
            x_at_target = None
            for seg in level_segs:
                xs, ys = seg[:, 0], seg[:, 1]
                for i in range(len(xs) - 1):
                    y0, y1 = float(ys[i]), float(ys[i + 1])
                    if (y0 - target_y) * (y1 - target_y) <= 0 and abs(y1 - y0) > 1e-12:
                        t = (target_y - y0) / (y1 - y0)
                        x_at_target = float(xs[i] + t * (xs[i + 1] - xs[i]))
                        break
                if x_at_target is not None:
                    break
            if x_at_target is not None:
                manual_positions.append((x_at_target, target_y))

        clabel_kwargs = dict(fmt=fmt, fontsize=12, inline=True)
        if manual_positions:
            clabel_kwargs['manual'] = manual_positions
        labels = ax.clabel(cs, **clabel_kwargs)
        for label in labels:
            label.set_rotation(0)
            label.set_fontweight('bold')
            label.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])

    def plot_particle_density_heatmap(
        self,
        filename: Optional[str] = None,
        dx: float = 0.2,
        dy: float = 0.2,
        incident_particle_yield: Optional[float] = None,
        include_hodoscope: bool = False,
        overlay_energy: bool = False,
    ) -> None:
        """
        Plot a heatmap of focal particle density in the detector plane.

        Args:
            filename: Output filename for the plot.
            dx: X-direction resolution in cm.
            dy: Y-direction resolution in cm.
            incident_particle_yield: Total particle yield (particles/source). Scales the density map.
            include_hodoscope: Whether to overlay hodoscope channel boundaries.
            overlay_energy: If True, overlay contour lines showing mean recoil energy per spatial bin.
        """
        if filename is None:
            filename = f'{self.spectrometer.figure_directory}/particle_density_heatmap.png'

        particle = self.spectrometer.conversion_foil.particle
        primary_response = self.performance_analyzer.get_channel_response(
            incident_particle_yield, compute_density=True, dx=dx, dy=dy,
        )
        density_map = primary_response.density_map
        X_mesh = primary_response.density_x
        Y_mesh = primary_response.density_y

        fig, ax = plt.subplots(figsize=(10, 8))
        divider = make_axes_locatable(ax)

        im = ax.pcolormesh(X_mesh, Y_mesh, density_map, cmap=self.primary_cmap, shading='auto', norm=LogNorm())
        cax = divider.append_axes("bottom", size="5%", pad=0.65)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        units = f'[{particle}/cm$^2$-source]' if incident_particle_yield is None else f'[{particle}/cm$^2$]'
        cbar.set_label(f'Fluence {units}')

        if overlay_energy:
            self._overlay_energy_contours(ax, self.spectrometer, X_mesh, Y_mesh, dx, dy)

        density_map2 = response_map2 = X_mesh2 = Y_mesh2 = None
        secondary_response = None
        if self.dual_data:
            particle2 = self.dual_data['spectrometer'].conversion_foil.particle
            secondary_response = self.dual_data['performance_analyzer'].get_channel_response(
                incident_particle_yield, compute_density=True, dx=dx, dy=dy,
            )
            density_map2 = secondary_response.density_map
            X_mesh2 = secondary_response.density_x
            Y_mesh2 = secondary_response.density_y
            im2 = ax.pcolormesh(X_mesh2, Y_mesh2, density_map2, cmap=self.dual_data['secondary_cmap'], shading='auto', alpha=0.5, norm=LogNorm())
            cax2 = divider.append_axes("bottom", size="5%", pad=0.75)
            cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
            units2 = f'[{particle2}/cm$^2$-source]' if incident_particle_yield is None else f'[{particle2}/cm$^2$]'
            cbar2.set_label(f'Fluence {units2}')

            if overlay_energy:
                self._overlay_energy_contours(ax, self.dual_data['spectrometer'], X_mesh2, Y_mesh2, dx, dy)

        y_lim = max(abs(float(Y_mesh.min())), abs(float(Y_mesh.max())))
        if Y_mesh2 is not None:
            y_lim = max(y_lim, abs(float(Y_mesh2.min())), abs(float(Y_mesh2.max())))
        ax.set_ylim(-y_lim, y_lim)

        ax.grid(True, alpha=0.4)
        if include_hodoscope:
            self._overlay_hodoscope(ax, self.spectrometer.hodoscope)
            if self.dual_data:
                self._overlay_hodoscope(ax, self.dual_data['spectrometer'].hodoscope)

        ax.set_xlabel('X Position [cm]')
        ax.set_ylabel('Y Position [cm]')
        ax.set_aspect('equal')

        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Particle density heatmap saved to {filename}')

        # If detector is used, also plot response map
        if self.spectrometer.hodoscope.detector_used:
            fig, ax = plt.subplots(figsize=(10, 8))
            divider = make_axes_locatable(ax)
            im = ax.pcolormesh(X_mesh, Y_mesh, primary_response.response_map_2d, cmap=self.primary_cmap, shading='auto', norm=LogNorm())
            cax = divider.append_axes("bottom", size="5%", pad=0.65)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            response_units = '[MeV/cm$^2$-source]' if incident_particle_yield is None else '[MeV/cm$^2$]'
            cbar.set_label(f'$E_{{dep}}$ {response_units}')

            if overlay_energy:
                self._overlay_energy_contours(ax, self.spectrometer, X_mesh, Y_mesh, dx, dy)

            if self.dual_data and secondary_response is not None and secondary_response.response_map_2d is not None:
                im2 = ax.pcolormesh(X_mesh2, Y_mesh2, secondary_response.response_map_2d, cmap=self.dual_data['secondary_cmap'], shading='auto', alpha=0.5, norm=LogNorm())
                cax2 = divider.append_axes("bottom", size="5%", pad=0.75)
                cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
                cbar2.set_label(f'$E_{{dep}}$ {response_units}')

                if overlay_energy:
                    self._overlay_energy_contours(ax, self.dual_data['spectrometer'], X_mesh2, Y_mesh2, dx, dy)

            ax.set_ylim(-y_lim, y_lim)

            ax.grid(True, alpha=0.4)
            if include_hodoscope:
                self._overlay_hodoscope(ax, self.spectrometer.hodoscope)
                if self.dual_data:
                    self._overlay_hodoscope(ax, self.dual_data['spectrometer'].hodoscope)

            ax.set_xlabel('X Position [cm]')
            ax.set_ylabel('Y Position [cm]')
            ax.set_aspect('equal')
            fig.tight_layout()
            response_filename = filename.replace('.png', '_response.png')
            fig.savefig(response_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Detector response heatmap saved to {response_filename}')
        
    def plot_monoenergetic_analysis(
        self,  
        incident_energy: float,
        filename: Optional[str] = None,
    ) -> None:
        """Generate analysis plots for monoenergetic performance."""
        if filename == None:
            filename = (
                f'{self.spectrometer.figure_directory}/' 
                f'Monoenergetic_En{incident_energy:.1f}MeV_'
                f'T{self.spectrometer.conversion_foil.thickness_um:.0f}um_'
                f'E0{self.spectrometer.reference_energy:.1f}MeV.png'
            )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of x positions
        x_positions = self.spectrometer.output_beam[:, 0]*100 # cm

        axes[0].hist(x_positions, bins=30, alpha=0.7, density=True,
                     color=self.primary_color, label=self.spectrometer.conversion_foil.particle)
        axes[0].set_xlabel('X Position [cm]')
        axes[0].set_ylabel('Probability Density')
        axes[0].set_title(f'X-Position Distribution\n{incident_energy:.1f} MeV {self.spectrometer.conversion_foil.incident_particle.capitalize()}s')
        axes[0].grid(True, alpha=0.3)

        # Scatter plot
        recoil_energies = self.spectrometer.input_beam[:, 5] * self.spectrometer.reference_energy + self.spectrometer.reference_energy
        scatter = axes[1].scatter(
            self.spectrometer.output_beam[:, 0]*100,
            self.spectrometer.output_beam[:, 2]*100,
            c=recoil_energies,
            s=1.0,
            cmap=self.primary_cmap,
            alpha=0.6
        )
        fig.colorbar(scatter, ax=axes[1], label=f'{self.spectrometer.conversion_foil.particle.capitalize()} Energy [MeV]')

        if self.dual_data is not None:
            spec2: MPRSpectrometer = self.dual_data['spectrometer']
            x_positions2 = spec2.output_beam[:, 0] * 100
            axes[0].hist(x_positions2, bins=30, alpha=0.7, density=True,
                         color=self.dual_data['secondary_color'], label=spec2.conversion_foil.particle)
            recoil_energies2 = spec2.input_beam[:, 5] * spec2.reference_energy + spec2.reference_energy
            scatter2 = axes[1].scatter(
                spec2.output_beam[:, 0] * 100,
                spec2.output_beam[:, 2] * 100,
                c=recoil_energies2,
                s=1.0,
                cmap=self.dual_data['secondary_cmap'],
                alpha=0.6,
            )
            fig.colorbar(scatter2, ax=axes[1], label=f'{spec2.conversion_foil.particle.capitalize()} Energy [MeV]')

        axes[0].legend()
        axes[1].set_xlabel('X Position [cm]')
        axes[1].set_ylabel('Y Position [cm]')
        axes[1].set_title(f'Focal Plane Distribution\n{incident_energy:.1f} MeV {self.spectrometer.conversion_foil.incident_particle.capitalize()}s')
        axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        print(filename)
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def plot_performance(
        self,  
        df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> None:
        """Generate comprehensive performance plot with shared x-axis."""
        if filename == None:
            filename = f'{self.spectrometer.figure_directory}/comprehensive_performance.png'
        else:
            filename = f'{self.spectrometer.figure_directory}/{filename}'
        
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
        
        # Left y-axis: position
        color_position = 'tab:orange'
        ax1.set_xlabel(f'{self.spectrometer.conversion_foil.incident_particle.capitalize()} Energy [MeV]')
        ax1.set_ylabel(f'Position [cm]', color=color_position)
        
        # Right y-axis: resolution and efficiency
        ax2 = ax1.twinx()
        color_resolution = 'tab:purple'
        ax2.set_ylabel('Energy Resolution [keV]', color=color_resolution)
        ax2.tick_params(axis='y', labelcolor=color_resolution)
        
        ax3 = ax1.twinx()
        # Offset the third axis to the right
        ax3.spines['right'].set_position(('outward', 80))
        color_efficiency = 'tab:green'
        ax3.set_ylabel(r'Total Efficiency ($\times 10^{-6}$)', color=color_efficiency)
        ax3.tick_params(axis='y', labelcolor=color_efficiency)
        
        # Loop over foils (either one or two)
        for foil, grp in df.groupby('foil'):
            # Extract data from DataFrame
            energies = grp['energy [MeV]'].to_numpy()
            positions = grp['position [m]'].to_numpy()
            band_lower = grp['position lower [m]'].to_numpy()
            band_upper = grp['position upper [m]'].to_numpy()
            total_efficiencies = grp['total efficiency'].to_numpy()
            widths = grp['position width [m]'].to_numpy()

            # Trim edge points to remove numerical artifacts from one-sided gradient
            # estimates and poor MC statistics at extreme energies
            trim = 2
            energies = energies[trim:-trim]
            positions = positions[trim:-trim]
            band_lower = band_lower[trim:-trim]
            band_upper = band_upper[trim:-trim]
            total_efficiencies = total_efficiencies[trim:-trim]
            widths = widths[trim:-trim]

            # Compute resolution
            gradients = np.gradient(positions, energies)
            energy_resolutions = widths / gradients * 1000

            # Apply a cubic fit to smooth the resolution curve
            # coeffs = np.polyfit(energies, energy_resolutions, 3)
            # energy_resolutions = np.polyval(coeffs, energies)

            # Plot position curve (KDE peak) with asymmetric FWHM band
            position_line = ax1.plot(energies, positions * 100, color=color_position,
                    label=f'Position')
            ax1.fill_between(energies, band_lower * 100, band_upper * 100,
                            alpha=0.3, color=color_position)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='y', labelcolor=color_position)

            resolution_line = ax2.plot(energies, energy_resolutions, color=color_resolution,
                            label=f'Resolution')

            efficiency_line = ax3.plot(energies, total_efficiencies*1e6, color=color_efficiency,
                            label=f'Efficiency')
            
            # Label lines on their respective axes
            range = energies.max() - energies.min()
            labelLines(position_line, xvals=[energies.min() + 0.65 * range], align=True, fontsize=12, yoffsets=2.2)
            labelLines(resolution_line, xvals=[energies.min() + 0.85 * range], align=True, fontsize=12, yoffsets=-45)
            labelLines(efficiency_line, xvals=[energies.min() + 0.3 * range], align=True, fontsize=12, yoffsets=0.025)
            
            # Add shading and label to indicate foil energy regions
            if self.dual_data:
                color = self.primary_color if foil == 'CH2' else self.dual_data['secondary_color']
                ax1.axvspan(energies.min(), energies.max(), facecolor=color, alpha=0.3)
                ax1.text(
                    energies.mean(),
                    ax1.get_ylim()[1] * 0.9,
                    f'{foil} Foil',
                    ha='center',
                    va='top',
                    color=color,
                    fontsize=12
                )
            
        # Make sure x-axis limits are consistent
        x_min, x_max = df['energy [MeV]'].min(), df['energy [MeV]'].max()
        x_margin = (x_max - x_min) * 0.02
        ax1.set_xlim(x_min - x_margin, x_max + x_margin)
        ax2.set_ylim(bottom=0)
        ax3.set_ylim(bottom=0)
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f'Figure saved to: {filename}')
        plt.close(fig)
        
    def plot_data(
        self,
        energy_MeV: float,
        dual_energy_MeV: Optional[float] = None,
        figure_directory: Optional[str] = None,
        filename_prefix: Optional[str] = None,
        angle_range: Tuple[float, float] = (0, np.pi/2),
        num_angles: int = 100
    ) -> None:
        """
        Plot differential cross section, cross sections, and stopping power data as three separate plots.
        
        Args:
            energy_MeV: Specific energy in MeV for differential cross section plot
            dual_energy_MeV: Specific energy in MeV for secondary foil's differential cross section plot (optional, only for dual foil spectrometers)
            figure_directory: Directory to save figures (optional)
            filename_prefix: Prefix for output filenames (optional)
            angle_range: Angular range (min, max) in radians for differential cross section
            num_angles: Number of angular points for differential cross section
        """
        foil = self.spectrometer.conversion_foil
        title = f'{foil.particle} at {energy_MeV:.2f} MeV'
        
        if figure_directory is None:
            figure_directory = self.spectrometer.figure_directory
        if filename_prefix is None:
            filename_prefix = f'{figure_directory}/foil_{foil.particle}'
        else:
            filename_prefix = f'{figure_directory}/{filename_prefix}'
            
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        
        # ========== Plot 1: Differential Cross Section vs Lab Angle ==========
        angles_rad = np.linspace(angle_range[1], angle_range[0], num_angles)
        angles_deg = np.degrees(angles_rad)
        
        for interaction in foil.interactions:
            if interaction.recoil_particle == foil.particle:
                diff_xs_lab = interaction.get_angle_distribution(energy_MeV).pdf(angles_rad)
                axs[0].plot(angles_deg, diff_xs_lab, 'tab:blue')
                
        axs[0].set_xlabel('Angle [deg]')
        axs[0].set_ylabel('Angle probability density')
        axs[0].grid(True, alpha=0.3)
        
        # ========== Plot 2: Cross Sections vs Energy ==========
        energies_MeV = np.linspace(1, 20, 1901)
        for interaction in foil.interactions:
            xs_inv_m = interaction.get_cross_section(energies_MeV)
            axs[1].plot(energies_MeV, xs_inv_m, 'tab:blue',
                    label=interaction.name)
        axs[1].axvline(energy_MeV, color='k', linestyle='--', alpha=0.7, 
                    label=f'Current energy: {energy_MeV:.1f} MeV')
        
        axs[1].set_xlabel('Neutron Energy [MeV]')
        axs[1].set_ylabel('Macroscopic Cross Section [m^-1]')
        axs[1].grid(True, alpha=0.3)
        axs[1].set_yscale('log')
        axs[1].set_xscale('log')
        
        # ========== Plot 3: CSDA Range vs Energy ==========
        srim_energies_MeV, srim_range_m = foil.integrated_stopping_data
        srim_range_mm = srim_range_m/1e-3
        
        axs[2].plot(srim_energies_MeV, srim_range_mm, 'tab:blue')
        axs[2].set_xlabel(f'{self.spectrometer.conversion_foil.particle.capitalize()} Energy [MeV]')
        axs[2].set_ylabel('Range in Foil Material [mm]')
        axs[2].grid(True, alpha=0.3)
        
        # Add dual data if available
        if self.dual_data:
            spec2: MPRSpectrometer = self.dual_data['spectrometer']
            foil2 = spec2.conversion_foil
            if dual_energy_MeV is None:
                dual_energy_MeV = energy_MeV
            title = f'{foil.particle} and {foil2.particle} at {dual_energy_MeV:.2f} MeV'
            
            for interaction in foil2.interactions:
                if interaction.recoil_particle == foil2.particle:
                    diff_xs_lab2 = interaction.get_angle_distribution(dual_energy_MeV).pdf(angles_rad)
                    axs[0].plot(angles_deg, diff_xs_lab2, 'darkorange')
            
            # n-hydron cross section data
            for interaction in foil2.interactions:
                xs_inv_m2 = interaction.get_cross_section(energies_MeV)
                axs[1].plot(energies_MeV, xs_inv_m2, 'darkorange',
                            label=interaction.name)
            
            # Stopping power for dual data
            srim_energies_MeV2, srim_range_m2 = foil2.integrated_stopping_data
            srim_range_mm2 = srim_range_m2/1e-3
            
            axs[2].plot(srim_energies_MeV2, srim_range_mm2, 'darkorange')
        
        fig.legend()
        filename = f'{filename_prefix}_E{energy_MeV:.1f}MeV_data.png'
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Data plot saved to {filename}')
        
            
    def plot_combined_foil(self, filename: Optional[str] = None) -> None:
        """Plot combined foil input geometry showing y-restriction."""
        if not self.dual_data:
            raise ValueError("Dual data not available. Only applicable for dual foil spectrometers.")
        
        spec_ch2 = self.spectrometer
        spec_cd2: MPRSpectrometer = self.dual_data['spectrometer']
        
        if filename is None:
            filename = f'{spec_ch2.figure_directory}/combined_foil.png'
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # CH2 (positive y)
        x_ch2 = spec_ch2.input_beam[:, 0] * 100
        y_ch2 = spec_ch2.input_beam[:, 2] * 100
        ax.scatter(x_ch2, y_ch2, alpha=0.5, s=5, label='CH2 (Protons)', color=self.primary_color)
        
        # CD2 (negative y)
        x_cd2 = spec_cd2.input_beam[:, 0] * 100
        y_cd2 = spec_cd2.input_beam[:, 2] * 100
        ax.scatter(x_cd2, y_cd2, alpha=0.5, s=5, label='CD2 (Deuterons)', color=self.dual_data['secondary_color'])
        
        # Draw foil boundary
        theta = np.linspace(0, 2*np.pi, 100)
        foil_r = spec_ch2.conversion_foil.foil_radius_cm
        ax.plot(foil_r * np.cos(theta), foil_r * np.sin(theta), 'k-', label='Foil boundary')
        
        # Draw y=0 dividing line
        ax.axhline(0, color='black', linestyle='--', alpha=0.7, label='Y=0 divider')
        
        # Add shaded regions to show foil halves
        from matplotlib.patches import Wedge
        wedge_upper = Wedge((0, 0), foil_r, 0, 180, facecolor=self.primary_color, alpha=0.1, 
                           edgecolor='none')
        wedge_lower = Wedge((0, 0), foil_r, 180, 360, facecolor=self.dual_data['secondary_color'], alpha=0.1, 
                           edgecolor='none')
        ax.add_patch(wedge_upper)
        ax.add_patch(wedge_lower)
        
        # Add text annotation
        ax.text(0.05, 0.95, 'CH2 (Protons)', transform=ax.transAxes, ha='left', va='top', color=self.primary_color)
        ax.text(0.95, 0.05, 'CD2 (Deuterons)', transform=ax.transAxes, ha='right', va='top', color=self.dual_data['secondary_color'])
        
        ax.set_xlabel('X Position [cm]')
        ax.set_ylabel('Y Position [cm]')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Combined input geometry plot saved to {filename}')
    
    def plot_separation_analysis(self, filename: Optional[str] = None) -> None:
        """
        Plot detailed separation analysis showing crossover statistics.
        """
        if not self.dual_data:
            raise ValueError("Dual data not available. Only applicable for dual foil spectrometers.")
        
        spec_ch2 = self.spectrometer
        spec_cd2 = self.dual_data['spectrometer']
        
        if filename is None:
            filename = f'{spec_ch2.figure_directory}/separation_analysis.png'
        
        # Get separation statistics
        sep_stats = self.dual_spectrometer.calculate_physical_separation()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left plot: Y-position histograms
        y_proton = spec_ch2.output_beam[:, 2] # cm
        y_deuteron = spec_cd2.output_beam[:, 2]  # cm
        
        bins = np.linspace(min(y_proton.min(), y_deuteron.min()), 
                          max(y_proton.max(), y_deuteron.max()), 50)
        
        axes[0].hist(y_proton, bins=bins, alpha=0.6, label='Protons (CH2)', 
                    color=self.primary_color, edgecolor='black', linewidth=0.5, density=True)
        axes[0].hist(y_deuteron, bins=bins, alpha=0.6, label='Deuterons (CD2)', 
                    color=self.dual_data['secondary_color'], edgecolor='black', linewidth=0.5, density=True)
        line = axes[0].axvline(0, color='black', linestyle='--', linewidth=2, 
                       label='Y=0 divider', alpha=0.7)
        # Add label to vertical line
        labelLines([line], yoffsets=0.1, align=True)
        
        # Add shaded regions for crossovers
        axes[0].axvspan(0, bins[-1], alpha=0.1, color=self.dual_data['secondary_color'])
        axes[0].axvspan(bins[0], 0, alpha=0.1, color=self.primary_color)
        
        # Add text labels to regions
        axes[0].text(
            (np.mean(bins[bins <= 0]) - bins[0]) / (bins[-1] - bins[0]),
            0.9, 'Protons', transform=axes[0].transAxes, 
            ha='center', va='center', color=self.primary_color
        )
        axes[0].text(
            (np.mean(bins[bins >= 0]) - bins[0]) / (bins[-1] - bins[0]),
            0.9, 'Deuterons', transform=axes[0].transAxes, 
            ha='center', va='center', color=self.dual_data['secondary_color']
        )
        
        # Set x limits
        axes[0].set_xlim(bins[0], bins[-1])
        axes[0].set_xlabel('Y Position [cm]')
        axes[0].set_ylabel('Probability Density')
        axes[0].grid(True, alpha=0.3)
        
        # Right plot: Separation statistics bar chart
        categories = ['Protons\n(should be <0)', 'Deuterons\n(should be >0)', 'Overall']
        stayed = [sep_stats['proton_separation_percentage'], 
                 sep_stats['deuteron_separation_percentage'],
                 sep_stats['overall_separation_percentage']]
        crossed = [100 - sep_stats['proton_separation_percentage'],
                  100 - sep_stats['deuteron_separation_percentage'],
                  100 - sep_stats['overall_separation_percentage']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = axes[1].bar(x - width/2, stayed, width, label='Stayed in region', 
                           color='tab:green', alpha=0.7, edgecolor='black')
        bars2 = axes[1].bar(x + width/2, crossed, width, label='Crossed midline', 
                           color='tab:orange', alpha=0.7, edgecolor='black')
        
        # Add percentage labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom')
        
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].legend()
        axes[1].set_ylim(0, 105)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Separation analysis plot saved to {filename}')

    def plot_response_matrix(
        self,
        response_matrices: dict,
        energy_grid: np.ndarray,
        log_scale: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        """Plot the instrument response matrix as a heatmap (energy × channel index).

        Args:
            response_matrices: Dict mapping foil material name to R of shape
                (n_energies, n_channels), as returned by
                PerformanceAnalyzer.build_response_matrix().
            energy_grid: 1-D array of incident energies [MeV] used to build R.
            log_scale: Use logarithmic colour scale (default True).
            filename: Output file path. Defaults to
                <figure_directory>/response_matrix.png.
        """
        def energy_edges(E: np.ndarray) -> np.ndarray:
            half = np.diff(E) / 2.0
            return np.concatenate([[E[0] - half[0]], E[:-1] + half, [E[-1] + half[-1]]])

        def channel_edges(n: int) -> np.ndarray:
            return np.arange(n + 1, dtype=float) - 0.5

        def colour_norm(R: np.ndarray) -> Tuple[np.ndarray, object]:
            if log_scale:
                Z = np.ma.masked_where(R <= 0, R)
                valid = Z.compressed()
                vmin = float(valid.min()) if len(valid) else 1e-10
                vmax = float(valid.max()) if len(valid) else 1.0
                return Z, LogNorm(vmin=vmin, vmax=vmax)
            return R, Normalize(vmin=0, vmax=float(R.max()) or 1.0)

        def nice_ticks(n: int, max_labels: int = 12) -> np.ndarray:
            step = max(1, (n + max_labels - 1) // max_labels)
            return np.arange(0, n, step)

        if filename is None:
            filename = f'{self.spectrometer.figure_directory}/response_matrix.png'

        is_dual = self.dual_data is not None
        inc = self.spectrometer.conversion_foil.incident_particle.capitalize()
        cb_label = f'R [MeV / source {inc.lower()}]'
        E_edges = energy_edges(np.asarray(energy_grid, dtype=float))

        key1 = self.spectrometer.conversion_foil.foil_material
        R1 = np.asarray(response_matrices[key1], dtype=float)
        n_ch1 = R1.shape[1]
        particle1 = self.spectrometer.conversion_foil.particle
        ch_edges1 = channel_edges(n_ch1)
        Z1, norm1 = colour_norm(R1)

        if is_dual:
            key2 = self.dual_data['spectrometer'].conversion_foil.foil_material
            R2_raw = response_matrices.get(key2)
        else:
            R2_raw = None

        if R2_raw is not None:
            R2 = np.asarray(R2_raw, dtype=float)
            n_ch2 = R2.shape[1]
            particle2 = self.dual_data['spectrometer'].conversion_foil.particle
            secondary_color = self.dual_data['secondary_color']
            secondary_cmap = self.dual_data['secondary_cmap']
            ch_edges2 = channel_edges(n_ch2)
            Z2, norm2 = colour_norm(R2)

            fig = plt.figure(figsize=(14, 6))
            gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.05, 0.07, 0.07], wspace=0.1)
            ax    = fig.add_subplot(gs[0, 0])
            cax1  = fig.add_subplot(gs[0, 1])
            cax2  = fig.add_subplot(gs[0, 3])

            mesh1 = ax.pcolormesh(ch_edges1, E_edges, Z1,
                                  cmap=self.primary_cmap, norm=norm1, shading='flat')
            ax.set_xlim(-0.5, n_ch1 - 0.5)
            ax.set_xticks(nice_ticks(n_ch1))
            ax.tick_params(axis='x', colors=self.primary_color)
            ax.spines['bottom'].set_edgecolor(self.primary_color)
            ax.set_xlabel(f'Channel index ({particle1}s)', color=self.primary_color)
            ax.set_ylabel(f'{inc} energy [MeV]')

            ax_sec = ax.twiny()
            ax_sec.set_xlim(-0.5, n_ch2 - 0.5)
            mesh2 = ax_sec.pcolormesh(ch_edges2, E_edges, Z2,
                                      cmap=secondary_cmap, norm=norm2, shading='flat')
            ax_sec.set_xticks(nice_ticks(n_ch2))
            ax_sec.xaxis.set_label_position('bottom')
            ax_sec.xaxis.tick_bottom()
            ax_sec.spines['bottom'].set_position(('outward', 60))
            ax_sec.spines['bottom'].set_edgecolor(secondary_color)
            ax_sec.spines['top'].set_visible(False)
            ax_sec.tick_params(axis='x', colors=secondary_color)
            ax_sec.set_xlabel(f'Channel index ({particle2}s)', color=secondary_color)

            cb1 = fig.colorbar(mesh1, cax=cax1)
            cb1.set_label(cb_label, color=self.primary_color)
            cb1.ax.yaxis.set_tick_params(color=self.primary_color)
            cb2 = fig.colorbar(mesh2, cax=cax2)
            cb2.set_label(cb_label, color=secondary_color)
            cb2.ax.yaxis.set_tick_params(color=secondary_color)
        else:
            fig = plt.figure(figsize=(10, 6))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.15)
            ax   = fig.add_subplot(gs[0, 0])
            cax1 = fig.add_subplot(gs[0, 1])

            mesh1 = ax.pcolormesh(ch_edges1, E_edges, Z1,
                                  cmap=self.primary_cmap, norm=norm1, shading='flat')
            ax.set_xlim(-0.5, n_ch1 - 0.5)
            ax.set_xticks(nice_ticks(n_ch1))
            ax.tick_params(axis='x', colors=self.primary_color)
            ax.spines['bottom'].set_edgecolor(self.primary_color)
            ax.set_xlabel(f'Channel index ({particle1}s)', color=self.primary_color)
            ax.set_ylabel(f'{inc} energy [MeV]')

            cb1 = fig.colorbar(mesh1, cax=cax1)
            cb1.set_label(cb_label, color=self.primary_color)
            cb1.ax.yaxis.set_tick_params(color=self.primary_color)

        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Response matrix plot saved to {filename}')

    def plot_forward_fit_spectrum(
        self,
        result: 'ForwardFittingResult',
        show_uncertainties: bool = True,
        true_spectrum: Optional[np.ndarray] = None,
        true_energy_grid: Optional[np.ndarray] = None,
        components: Optional[dict] = None,
        color: str = 'tab:blue',
        filename: Optional[str] = None,
    ) -> Axes:
        """Plot a forward-fit result with per-channel data points and optional overlays.

        Per-channel data points are always plotted when result.raw_counts and
        result.response_matrix are available (populated automatically by SpectrumFitter.fit()).
        Each data point shows:
          - X error bar: rms spread of incident energies contributing to that channel
            (energy resolution, derived from the response matrix).
          - Y error bar: combined counting + background uncertainty converted to spectrum
            units.

        Parameters
        ----------
        result : ForwardFittingResult
        show_uncertainties : shade the +/-1 sigma band around the fitted spectrum.
        true_spectrum : ground-truth total spectrum to overlay.
        true_energy_grid : energy grid for ``true_spectrum``. Defaults to
            ``result.energy_grid``.
        components : dict[str, ndarray] of fitted spectral components from
            ``components_model(result.params)``. Each component is overlaid with
            a distinct colour/linestyle.
        color : line colour for the fitted spectrum.
        filename : save path.

        Returns
        -------
        Axes
        """
        component_linestyles = ['-', '--', ':', '-.']
        component_colours = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                             'tab:brown', 'tab:pink', 'tab:gray']

        fig, ax = plt.subplots(figsize=(8, 5))

        E = result.energy_grid
        bw = result.bin_widths
        E_overlay = np.asarray(true_energy_grid if true_energy_grid is not None else E)
        bw_overlay = np.empty(len(E_overlay))
        bw_overlay[:-1] = np.diff(E_overlay)
        bw_overlay[-1] = bw_overlay[-2]

        foil_geometric_factor = self.spectrometer.foil_geometric_factor
        incident_particle = self.spectrometer.conversion_foil.incident_particle

        if foil_geometric_factor is not None:
            fit_norm = bw
            def _norm_overlay(arr: np.ndarray) -> np.ndarray:
                return arr / bw_overlay
            ylabel = f'dN/dE [{incident_particle}s/MeV]'
        else:
            fit_norm = float((result.spectrum * bw).sum())
            def _norm_overlay(arr: np.ndarray) -> np.ndarray:
                integral = float((arr * bw_overlay).sum())
                return arr / integral if integral > 0 else arr
            ylabel = r'Probability density [MeV$^{-1}$]'

        f = result.spectrum / fit_norm
        sigma_spectrum = result.uncertainties / fit_norm

        # Pre-compute data points so we know the xlim before plotting.
        spectrum_data = spectrum_data_sigma = nominal_energy = energy_spread = None
        if result.raw_counts is not None and result.response_matrix is not None:
            spectrum_data, spectrum_data_sigma, nominal_energy, energy_spread = compute_spectrum_data_points(result)

        _comp_label_positions = [
            (0.58, 0.96),
            (0.25, 0.40),
            (0.33, 0.65),
            (0.39, 0.15),
        ]
        _ff_label_pos = (0.82, 0.27)
        _ts_label_pos = (0.80, 0.4)
        _data_label_pos = (0.86, 0.17)

        _label_kw = dict(transform=ax.transAxes, fontsize=13, ha='center', va='center')
        _stroke = [pe.withStroke(linewidth=3, foreground='white')]

        if components is not None:
            for i, ((label, comp), ls, col) in enumerate(zip(
                components.items(), component_linestyles, component_colours,
            )):
                comp_arr = _norm_overlay(np.asarray(comp))
                ax.plot(E_overlay, comp_arr, linestyle=ls, color=col,
                        linewidth=2.5, zorder=1, label=label)
                pos = _comp_label_positions[i] if i < len(_comp_label_positions) else (0.5, 0.5)
                ax.text(*pos, label, color=col, **_label_kw).set_path_effects(_stroke)

        if spectrum_data is not None:
            if np.ndim(fit_norm) == 0:
                data_norm = float(fit_norm)
            else:
                data_norm = np.interp(nominal_energy, E, fit_norm)
            data_norm_safe = np.where(data_norm > 0, data_norm, 1.0)
            # Exclude zero-response channels (nominal_energy == 0 is the fallback value).
            valid_ne = nominal_energy > 0
            ax.errorbar(
                nominal_energy[valid_ne], (spectrum_data / data_norm_safe)[valid_ne],
                xerr=energy_spread[valid_ne], yerr=(spectrum_data_sigma / data_norm_safe)[valid_ne],
                fmt='o', color='tab:gray', capsize=3, markersize=5, linewidth=1.5,
                zorder=5, label='Data',
            )
            ax.text(*_data_label_pos, 'Data', color='tab:gray', **_label_kw).set_path_effects(_stroke)

        if show_uncertainties:
            ax.fill_between(E, f - sigma_spectrum, f + sigma_spectrum, color=color, alpha=0.25, zorder=2)
        ax.plot(E, f, color=color, linewidth=2, zorder=3, label='Forward fit')
        ax.text(*_ff_label_pos, 'Forward fit', color=color, **_label_kw).set_path_effects(_stroke)

        ts_arr = None
        if true_spectrum is not None:
            ts_arr = _norm_overlay(np.asarray(true_spectrum))
            ax.plot(E_overlay, ts_arr, 'k--', linewidth=2, zorder=4, label='True spectrum')
            ax.text(*_ts_label_pos, 'True spectrum', color='k', **_label_kw).set_path_effects(_stroke)

        ax.set_xlabel('Incident energy [MeV]')
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')
        peak_vals = [f[f > 0].max() if np.any(f > 0) else np.nan]
        ymin = None
        if ts_arr is not None:
            peak_vals.append(ts_arr[ts_arr > 0].max() if np.any(ts_arr > 0) else np.nan)
            ymin = ts_arr[ts_arr > 0].min() if np.any(ts_arr > 0) else None
        ymax = np.nanmax(peak_vals) * 3
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if nominal_energy is not None:
            valid_ne = nominal_energy > 0
            x_lo = nominal_energy[valid_ne].min()
            x_hi = nominal_energy[valid_ne].max()
            margin = 0.04 * (x_hi - x_lo)
            ax.set_xlim(x_lo - margin, x_hi + margin)
            ax.autoscale(False)
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        return ax

    def plot_forward_fit_residuals(
        self,
        result: 'ForwardFittingResult',
        R: np.ndarray,
        counts: np.ndarray,
        sigma_counts: np.ndarray,
        color: str = 'tab:blue',
        filename: Optional[str] = None,
    ) -> Axes:
        """Plot channel-by-channel normalised residuals (model - data) / sigma.

        Parameters
        ----------
        result : ForwardFittingResult
        R : ndarray, shape (n_energies, n_channels) - response matrix.
        counts : ndarray, shape (n_channels,) - measured hodoscope counts.
        sigma_counts : ndarray, shape (n_channels,) - per-channel 1-sigma errors.
        color : bar colour.
        filename : save path.

        Returns
        -------
        Axes
        """
        fig, ax = plt.subplots(figsize=(10, 4))

        R = np.asarray(R, dtype=float)
        counts = np.asarray(counts, dtype=float)
        sigma_counts = np.asarray(sigma_counts, dtype=float)

        predicted = R.T @ result.spectrum
        residuals = (predicted - counts) / np.where(sigma_counts > 0, sigma_counts, 1.0)

        channels = np.arange(len(counts))
        ax.bar(channels, residuals, color=color, alpha=0.7)
        ax.axhline(0, color='k', linewidth=1)
        ax.axhline(1, color='grey', linewidth=0.8, linestyle='--')
        ax.axhline(-1, color='grey', linewidth=0.8, linestyle='--')

        ax.set_xlabel('Channel index')
        ax.set_ylabel('(model - data) / sigma')
        ax.set_title('Forward fit residuals')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        return ax

# =========== Contour Plotting ===============
class PlotParameter:
    """Parameter configuration for contour plotting."""
    
    def __init__(
        self,
        name: str,
        label: Optional[str] = None,
        log_scale: bool = False
    ):
        self.name = name
        self.label = label if label else name
        self.log_scale = log_scale
    
    def get_values(self, df):
        """Get values from dataframe, applying log if needed"""
        values = df[self.name].values
        if self.log_scale and np.all(values > 0):
            return np.log10(values)
        return values


class ContourParameter(PlotParameter):
    """Extended parameter class for contour lines."""
    
    def __init__(
        self,
        name: str,
        label: Optional[str] = None,
        log_scale: bool = False,
        num_levels: int = 10, 
        color: Union[str, Tuple[float, float, float]] = 'black',
        linestyle: str = 'solid',
        linewidth: float = 1.0
    ):
        """
        Args:
            name: Name of the parameter
            label: Label to display on the plot (defaults to name if None)
            log_scale: Whether to use logarithmic scale for this parameter
            num_levels: Number of contour levels to plot
            color: Color for the contour lines
            linestyle: Line style for contour lines
            linewidth: Width of contour lines
        """
        super().__init__(name, label, log_scale)
        self.num_levels = num_levels
        self.color = color
        self.linestyle = linestyle
        self.linewidth = linewidth

class SweepPlotter:
    def __init__(self, sweeper: FoilSweeper):
        self.sweeper = sweeper
        
    def plot_heatmap_grid(
        self,
        x_variable: str,
        y_variable: str,
        z_variable: str,
        heat_variable: str,
        filename: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        z_label: Optional[str] = None,
        heat_label: Optional[str] = None,
        contour_params: Optional[list[ContourParameter]] = None,
        use_grid_interpolation: bool = False,
        grid_resolution: int = 50,
        cmap: str = 'plasma'
    ) -> None:
        """
        Plot heatmap grid.
        
        Args:
            x_variable: Variable for x-axis
            y_variable: Variable for y-axis
            z_variable: Variable for z-axis
            heat_variable: Variable for heatmap
            filename: Filename for saving plot (optional)
            x_label: Label for x-axis (defaults to variable name)
            y_label: Label for y-axis (defaults to variable name)
            z_label: Label for z-axis (defaults to variable name)
            heat_label: Label for heatmap (defaults to variable name)
            contour_params: List of ContourParameter objects for additional contour lines
            use_grid_interpolation: Whether to use grid interpolation (vs triangulation)
            grid_resolution: Resolution for grid interpolation
            cmap: Colormap name
        """
        if filename is None:
            filename = f'{self.sweeper.spectrometer.figure_directory}/heatmap_grid.png'
        
        if self.sweeper.results_df is None:
            raise ValueError("No sweep results found. Please run run_sweep() first.")
        
        # Create parameter objects
        x_param = PlotParameter(x_variable, x_label)
        y_param = PlotParameter(y_variable, y_label)
        z_label = z_label or z_variable
        heat_param = PlotParameter(heat_variable, heat_label)
        
        # Calculate global min/max for consistent colorbar across all subplots
        heat_values = heat_param.get_values(self.sweeper.results_df.dropna(subset=[heat_variable]))
        vmin, vmax = np.nanmin(heat_values), np.nanmax(heat_values)
        
        # Extract z data to find grid size
        z_values = self.sweeper.results_df[z_variable].unique()
        
        # Create gridsize based on z_variable
        n_cols = int(np.ceil(np.sqrt(len(z_values))))
        n_rows = int(np.ceil(len(z_values) / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3),
                                sharex=True, sharey=True, squeeze=False, layout='constrained')
        
        # Plot heatmaps
        for i, ax in enumerate(axs.flatten()):
            if i >= len(z_values):
                ax.axis('off')
                continue
            
            z_value = z_values[i]
            data = self.sweeper.results_df[self.sweeper.results_df[z_variable] == z_value]
            
            self._plot_heatmap(ax, data, x_param, y_param, heat_param, 
                contour_params=contour_params,
                use_grid_interpolation=use_grid_interpolation,
                grid_resolution=grid_resolution,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax)
            ax.set_title(f'{z_variable} = {z_value}')
        
        fig.supxlabel(x_param.label)
        fig.supylabel(y_param.label)
        
        # Add colorbar label with log scale notation if needed
        cbar_label = heat_param.label
        if heat_param.log_scale:
            cbar_label = f"log$_{10}$({cbar_label})"
        
        # Create a ScalarMappable for the colorbar
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # Required for ScalarMappable
        fig.colorbar(sm, ax=axs, label=cbar_label, pad=0.02, shrink=0.8)
        
        # Create legend from contour_params
        if contour_params:
            legend_handles = []
            legend_labels = []
            for cp in contour_params:
                legend_line = plt.Line2D([0], [0], color=cp.color, 
                                        linestyle=cp.linestyle,
                                        linewidth=cp.linewidth)
                legend_handles.append(legend_line)
                # Use logscale notation if needed
                contour_label = cp.label
                if cp.log_scale:
                    contour_label = f"log$_{{10}}$({contour_label})"
                legend_labels.append(contour_label)
            fig.legend(legend_handles, legend_labels, framealpha=0.7, fontsize=8, loc='lower right')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    def _plot_heatmap(
        self,
        ax: Axes,
        data: pd.DataFrame,
        x_param: PlotParameter,
        y_param: PlotParameter,
        heat_param: PlotParameter,
        contour_params: Optional[list[ContourParameter]] = None,
        use_grid_interpolation: bool = True,
        grid_resolution: int = 50,
        cmap: str = 'plasma',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> None:
        """Plot heatmap with line-style contours on top."""
        
        # Clean data
        required_columns = [x_param.name, y_param.name, heat_param.name]
        if contour_params:
            required_columns.extend([cp.name for cp in contour_params])
        
        data_clean = data.dropna(subset=required_columns)
        
        if len(data_clean) == 0:
            return
        
        # Extract values
        x = x_param.get_values(data_clean)
        y = y_param.get_values(data_clean)
        z = heat_param.get_values(data_clean)
        
        # Create contour plot
        if use_grid_interpolation:
            xi = np.linspace(np.min(x), np.max(x), grid_resolution)
            yi = np.linspace(np.min(y), np.max(y), grid_resolution)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x, y), z, (X, Y), method='cubic', fill_value=np.nan)
            contour = ax.contourf(X, Y, Z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add contour lines if requested
            if contour_params:
                for cp in contour_params:
                    param_values = cp.get_values(data_clean)
                    P = griddata((x, y), param_values, (X, Y), method='cubic', fill_value=np.nan)
                    cs = ax.contour(X, Y, P, levels=cp.num_levels,
                                   colors=cp.color, linestyles=cp.linestyle,
                                   linewidths=cp.linewidth)
                    ax.clabel(cs, inline=True, fontsize=8)
        else:
            contour = ax.tricontourf(x, y, z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add contour lines if requested
            if contour_params:
                for cp in contour_params:
                    param_values = cp.get_values(data_clean)
                    cs = ax.tricontour(x, y, param_values, levels=cp.num_levels,
                                      colors=cp.color, linestyles=cp.linestyle,
                                      linewidths=cp.linewidth)
                    ax.clabel(cs, inline=True, fontsize=8)
