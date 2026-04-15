"""Plotting methods for MPR spectrometer visualization."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Union
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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

if TYPE_CHECKING:
    from ..analysis.parameter_sweep import FoilSweeper

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
                'primary_label': 'Protons (CH2)',
                'secondary_label': 'Deuterons (CD2)',
                'secondary_color': 'tab:blue',
                'secondary_cmap': 'GnBu'
            }
            self.dual_spectrometer = spectrometer
            self.primary_color = 'tab:red'
            self.primary_cmap = 'YlOrRd'
        else:
            raise ValueError(f'Invalid spectrometer type: {type(spectrometer)}. Should be MPRSpectrometer or DualFoilSpectrometer.')
    
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
        
        # Draw hodoscope if requested
        if include_hodoscope:
            def _draw_hodoscope(hod, color='black'):
                """Draw the full envelope and channel boundaries for one hodoscope."""
                heights = hod.channel_heights * 100  # cm
                edges = hod.channel_edges * 100       # cm
                y_ctr = hod.y_center * 100            # cm
                edge_lengths = np.minimum(heights[:-1], heights[1:])

                # Internal channel-edge lines (span ±half the shorter adjacent channel)
                ax.vlines(edges[1:-1],
                          y_ctr - edge_lengths / 2, y_ctr + edge_lengths / 2,
                          color=color, linestyle='--', linewidth=0.5)

                # Outer envelope (top + bottom step function + left/right edges)
                x = np.repeat(edges, 2)
                y_half = np.concatenate([[0], np.repeat(heights / 2, 2), [0]])
                ax.plot(x, y_ctr + y_half, color=color, linewidth=1.0)
                ax.plot(x, y_ctr - y_half, color=color, linewidth=1.0)

            
            _draw_hodoscope(self.spectrometer.hodoscope)
            if self.dual_data:
                _draw_hodoscope(self.dual_data['spectrometer'].hodoscope)

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
        ax.set_title(f'{self.spectrometer.conversion_foil.particle.capitalize()} Distribution in Focal Plane')
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
            ax.set_title(f'{self.spectrometer.conversion_foil.particle.capitalize()} and {spec2.conversion_foil.particle.capitalize()} Distribution in Focal Plane')
        
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
        neutron_background_file: Optional[str] = None,
        photon_background_file: Optional[str] = None,
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
            neutron_background_file: Path to a time-resolved CSV with columns 'time' [s],
                'energy' [MeV], and 'mean' [particles/cm²/source] — used when
                hodoscope.use_time_gating is True.  Also accepted in the 1-D ('energy', 'mean') format.
            photon_background_file: Same format as neutron_background_file.
            neutron_energy: Single neutron energy in MeV (scalar background input).
            neutron_flux: Neutron flux in particles/cm^2-source (scalar background input).
            photon_energy: Single photon energy in MeV (scalar background input).
            photon_flux: Photon flux in particles/cm^2-source (scalar background input).
        """
        if filename is None:
            filename = f'{self.spectrometer.figure_directory}/counts_vs_position.png'

        if len(self.spectrometer.output_beam) == 0:
            raise ValueError("No output beam data available. Run apply_transfer_map() first.")

        hodoscope = self.spectrometer.hodoscope

        # --- Signal via get_recoil_x_map ---
        # Each hodoscope's y_center and channel_height define its acceptance window.
        # channel_time_windows shape (n_channels, 2): per-channel [t_min, t_max] of signal
        # arrival times.  Filled with NaN when hodoscope.use_time_gating is False.
        is_dual = self.dual_data is not None
        signal, coverage, channel_time_windows = self.performance_analyzer.get_recoil_x_map(
            particle_yield=incident_particle_yield,
        )
        channel_edges = hodoscope.channel_edges * 100  # m to cm
        channel_widths = hodoscope.channel_widths * 100  # m to cm
        channel_heights = hodoscope.channel_heights * 100  # m to cm

        # signal units: [particles/source] (or [particles] with yield)

        # --- Dual spectrometer signal (retrieved before background so time windows are ready) ---
        signal2 = coverage2 = channel_time_windows2 = channel_edges2 = channel_widths2 = channel_heights2 = None
        if is_dual:
            signal2, coverage2, channel_time_windows2 = (
                self.dual_data['performance_analyzer'].get_recoil_x_map(
                    particle_yield=incident_particle_yield,
                )
            )
            hodoscope2 = self.dual_data['spectrometer'].hodoscope
            channel_edges2 = hodoscope2.channel_edges * 100  # m to cm
            channel_widths2 = hodoscope2.channel_widths * 100  # m to cm
            channel_heights2 = hodoscope2.channel_heights * 100  # m to cm

        # --- Background per channel (neutron and photon separately) ---
        # Each hodoscope stores its own physical height, so channel_area = width * height is
        # correct without any extra area factor.  For dual-foil, CH2 and CD2 are computed
        # independently using their respective hodoscopes and time windows.

        # Initialize per-channel background arrays and the time-resolved arrays used for
        # the gated vs. non-gated S/B overlay (only populated when use_time_gating=True).
        neutron_bg_per_channel = photon_bg_per_channel = None
        neutron_bg_per_channel2 = photon_bg_per_channel2 = None
        time_bins = neutron_background_vs_time = photon_background_vs_time = None
        scale = (incident_particle_yield if incident_particle_yield else 1.0)

        def _compute_bg(time_windows, chan_widths, chan_heights, t_bins, n_bg_ts, ph_bg_ts, use_tg):
            """Compute per-channel neutron/photon background for the given hodoscope half."""
            dt_local = float(np.median(np.diff(t_bins))) if len(t_bins) > 1 else 1.0
            neutron_bg = np.zeros(len(chan_widths))
            photon_bg = np.zeros(len(chan_widths))
            for i in range(len(chan_widths)):
                if use_tg and not np.isnan(time_windows[i, 0]):
                    # Weight each background bin by the fraction of that bin that overlaps the
                    # signal arrival window.  This correctly handles windows narrower than the
                    # bin width (which a binary mask would miss entirely) as well as partial
                    # overlaps at the edges.
                    t_min = time_windows[i, 0]
                    t_max = time_windows[i, 1]
                    bin_left = t_bins - dt_local / 2
                    bin_right = t_bins + dt_local / 2
                    overlap = np.maximum(0.0, np.minimum(bin_right, t_max) - np.maximum(bin_left, t_min)) / dt_local
                else:
                    overlap = np.ones(len(t_bins))
                channel_area = chan_widths[i] * chan_heights[i]  # cm^2 — each hodoscope carries its own height
                neutron_bg[i] = np.dot(n_bg_ts, overlap) * scale * channel_area
                photon_bg[i] = np.dot(ph_bg_ts, overlap) * scale * channel_area
            return neutron_bg, photon_bg

        if neutron_background_file and photon_background_file and hodoscope.detector_used:
            # get_background() always returns a 3-tuple of arrays regardless of use_time_gating.
            # When use_time_gating=False, time_bins=[0.0] and each background array has length 1
            # containing the total energy deposited; the time-window logic below then integrates
            # over the single bin, giving the same result as a plain scalar multiply.
            time_bins, neutron_background_vs_time, photon_background_vs_time = hodoscope.get_background(
                neutron_background_file, photon_background_file
            )
            neutron_bg_per_channel, photon_bg_per_channel = _compute_bg(
                channel_time_windows, channel_widths, channel_heights,
                time_bins, neutron_background_vs_time, photon_background_vs_time,
                hodoscope.use_time_gating,
            )

        if self.dual_data is not None and channel_time_windows2 is not None and neutron_background_file and photon_background_file:
            hodoscope2 = self.dual_data['spectrometer'].hodoscope
            if hodoscope2.detector_used:
                time_bins2, neutron_bg_vs_time2, photon_bg_vs_time2 = hodoscope2.get_background(
                    neutron_background_file, photon_background_file
                )
                neutron_bg_per_channel2, photon_bg_per_channel2 = _compute_bg(
                    channel_time_windows2, channel_widths2, channel_heights2,
                    time_bins2, neutron_bg_vs_time2, photon_bg_vs_time2,
                    hodoscope2.use_time_gating,
                )

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
            label = f'Counts [{"MeV" if incident_particle_yield else "MeV/source"}]'

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

        def _add_energy_axis(ax):
            """Add twin top x-axis(es) showing incident neutron energy in MeV.

            In dual-foil mode two axes are added (one per foil), each colored to match
            the corresponding signal line.  The deuteron axis is offset outward so the
            two labels do not overlap.
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
                _make_twin(_x_to_en, _en_to_x,
                           f'{inc} Energy [MeV] (p)',
                           color=self.primary_color)
                if self.dual_data is not None and _x_to_en2 is not None and _en_to_x2 is not None:
                    _make_twin(_x_to_en2, _en_to_x2,
                               f'{inc} Energy [MeV] (d)',
                               color=self.dual_data['secondary_color'],
                               offset=45,
                               tick_step=0.5)
            else:
                _make_twin(_x_to_en, _en_to_x, f'{inc} Energy [MeV]')

        def _step(ax, edges, values, **kwargs):
            return ax.step(edges, np.append(values, values[-1]), where='pre', **kwargs)[0]

        # Plot 1: counts
        # In dual-foil mode background labels distinguish the CH2 and CD2 halves.
        n_label = 'neutron (p)' if is_dual else 'neutron'
        g_label = 'photon (p)' if is_dual else 'photon'
        fig, ax_counts = plt.subplots(figsize=(10, 5.5 if is_dual else 4))
        _step(ax_counts, channel_edges, signal,
              color=self.primary_color, label=particle_label, linewidth=3)
        if neutron_bg_per_channel is not None:
            _step(ax_counts, channel_edges, neutron_bg_per_channel,
                  color='tab:green', label=n_label, linewidth=3)
        if photon_bg_per_channel is not None:
            _step(ax_counts, channel_edges, photon_bg_per_channel,
                  color='tab:purple', label=g_label, linewidth=3)
        if self.dual_data is not None and signal2 is not None:
            _step(ax_counts, channel_edges2, signal2,
                  color=self.dual_data['secondary_color'],
                  label=self.dual_data['secondary_label'], linewidth=3)
        if is_dual and neutron_bg_per_channel2 is not None:
            _step(ax_counts, channel_edges2, neutron_bg_per_channel2,
                  color='tab:green', linestyle='--', label='neutron (d)', linewidth=3)
        if is_dual and photon_bg_per_channel2 is not None:
            _step(ax_counts, channel_edges2, photon_bg_per_channel2,
                  color='tab:purple', linestyle='--', label='photon (d)', linewidth=3)
        ax_counts.set_yscale('log')
        ax_counts.set_xlabel('Horizontal Position [cm]')
        ax_counts.set_ylabel(label)
        ax_counts.grid(True, alpha=0.3)
        if incident_particle_yield is not None:
            ax_counts.set_title(f'Yield: {incident_particle_yield:.0e}')
        labelLines(ax_counts.get_lines(), align=False)
        _add_energy_axis(ax_counts)
        fig.tight_layout()
        fig.savefig(filename_counts, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Position histogram saved to {filename_counts}')

        # Plot 2 (optional): S/B — separate lines for neutron and photon backgrounds.
        # Single-foil: gated S/B (solid) vs non-gated S/B (dashed) comparison.
        # Dual-foil: CH2 (solid) and CD2 (dashed) gated S/B; no-gate overlay omitted to
        # keep the plot readable.
        if neutron_bg_per_channel is not None and photon_bg_per_channel is not None:
            fig, ax_sb = plt.subplots(figsize=(10, 5.5 if is_dual else 4))

            if hodoscope.use_time_gating and neutron_background_vs_time is not None and photon_background_vs_time is not None:
                if is_dual:
                    # Gated S/B for CH2 (solid) and CD2 (dashed).
                    sb_neutron_ch2 = np.where(neutron_bg_per_channel > 0, signal / neutron_bg_per_channel, np.nan)
                    _step(ax_sb, channel_edges, np.log10(sb_neutron_ch2),
                          color='tab:green', linestyle='-', label='neutron (p)', linewidth=3)
                    sb_photon_ch2 = np.where(photon_bg_per_channel > 0, signal / photon_bg_per_channel, np.nan)
                    _step(ax_sb, channel_edges, np.log10(sb_photon_ch2),
                          color='tab:purple', linestyle='-', label='photon (p)', linewidth=3)
                    if neutron_bg_per_channel2 is not None and photon_bg_per_channel2 is not None and signal2 is not None:
                        sb_neutron_cd2 = np.where(neutron_bg_per_channel2 > 0, signal2 / neutron_bg_per_channel2, np.nan)
                        _step(ax_sb, channel_edges2, np.log10(sb_neutron_cd2),
                              color='tab:green', linestyle='--', label='neutron (d)', linewidth=3)
                        sb_photon_cd2 = np.where(photon_bg_per_channel2 > 0, signal2 / photon_bg_per_channel2, np.nan)
                        _step(ax_sb, channel_edges2, np.log10(sb_photon_cd2),
                              color='tab:purple', linestyle='--', label='photon (d)', linewidth=3)
                else:
                    # Gated S/B: use the per-channel time-windowed background.
                    sb_neutron = np.where(neutron_bg_per_channel > 0, signal / neutron_bg_per_channel, np.nan)
                    _step(ax_sb, channel_edges, np.log10(sb_neutron),
                          color='tab:green', linestyle='-', label='neutron', linewidth=3)
                    sb_photon_gating = np.where(photon_bg_per_channel > 0, signal / photon_bg_per_channel, np.nan)
                    _step(ax_sb, channel_edges, np.log10(sb_photon_gating),
                          color='tab:purple', linestyle='-', label='photon', linewidth=3)
            else:
                sb_neutron = np.where(neutron_bg_per_channel > 0, signal / neutron_bg_per_channel, np.nan)
                _step(ax_sb, channel_edges, np.log10(sb_neutron),
                      color='tab:green', label=n_label, linewidth=3)
                sb_photon = np.where(photon_bg_per_channel > 0, signal / photon_bg_per_channel, np.nan)
                _step(ax_sb, channel_edges, np.log10(sb_photon),
                      color='tab:purple', label=g_label, linewidth=3)
                if neutron_bg_per_channel2 is not None and photon_bg_per_channel2 is not None and signal2 is not None:
                    sb_neutron_cd2 = np.where(neutron_bg_per_channel2 > 0, signal2 / neutron_bg_per_channel2, np.nan)
                    _step(ax_sb, channel_edges2, np.log10(sb_neutron_cd2),
                          color='tab:green', linestyle='--', label='neutron (d)', linewidth=3)
                    sb_photon_cd2 = np.where(photon_bg_per_channel2 > 0, signal2 / photon_bg_per_channel2, np.nan)
                    _step(ax_sb, channel_edges2, np.log10(sb_photon_cd2),
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
                photon_background_vs_time=photon_background_vs_time,
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
        photon_background_vs_time: Optional[np.ndarray] = None,
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
            y_ctr_cm = hod.y_center * 100
            local_heights_cm = hod.channel_heights * 100
            local_edges_cm = hod.channel_edges * 100
            idx = np.digitize(x_cm, local_edges_cm) - 1
            times_per_channel = []
            for i in range(hod.total_channels):
                in_bin = idx == i
                y_ok = np.abs(y_cm - y_ctr_cm) <= local_heights_cm[i] / 2
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

        fig, ax = plt.subplots(figsize=(8, 6))
        _draw_ridgelines(ax, pdfs_ch2, colors_ch2, alpha=0.5)
        if is_dual:
            _draw_ridgelines(ax, pdfs_cd2, colors_cd2, alpha=0.4)

        ax.set_xlabel('Detector arrival time [ns]')
        ax.set_ylabel('Channel index')
        ax.set_ylim(-0.5, n_channels - 0.5 + ridge_scale)
        ax.grid(True, alpha=0.3)
        ax.text(0.52, 0.55, self.spectrometer.conversion_foil.particle,
                transform=ax.transAxes, color=colors_ch2[0],
                va='top', ha='left')
        if is_dual and self.dual_data is not None:
            ax.text(0.52, 0.48, self.dual_data['spectrometer'].conversion_foil.particle,
                    transform=ax.transAxes, color=colors_cd2[0],
                    va='top', ha='left')

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
            ax_bg.step(time_ns_bg[mask], neutron_background_vs_time[mask] * overlap[mask], where='mid',
                       color='tab:green', linewidth=2, label='neutron', alpha=0.8)
            ax_bg.step(time_ns_bg[mask], photon_background_vs_time[mask] * overlap[mask], where='mid',
                       color='tab:purple', linewidth=2, label='photon', alpha=0.8)
            ax_bg.set_yscale('log')
            ax_bg.set_ylabel('$E_{dep}$ [MeV/cm$^2$/source]')
            labelLines(ax_bg.get_lines(), align=False)

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
            x0, p_x_relative, y0, p_y_relative, _, energy_relative, _ = ray
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
                x0, p_x_relative, y0, p_y_relative, _, energy_relative, _ = ray
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

    def plot_particle_density_heatmap(
        self,
        filename: Optional[str] = None,
        dx: float = 0.5,
        dy: float = 0.5,
        incident_particle_yield: Optional[float] = None,
    ) -> None:
        """
        Plot a heatmap of focal particle density in the detector plane.

        Args:
            filename: Output filename for the plot.
            dx: X-direction resolution in cm.
            dy: Y-direction resolution in cm.
            incident_particle_yield: Total particle yield (particles/source). Scales the density map.
        """
        if filename == None:
            filename = f'{self.spectrometer.figure_directory}/particle_density_heatmap.png'

        particle = self.spectrometer.conversion_foil.particle
        density_map, response, X_mesh, Y_mesh = self.performance_analyzer.get_recoil_density_map(dx, dy, particle_yield=incident_particle_yield)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        im = ax.pcolormesh(X_mesh, Y_mesh, np.log10(density_map), cmap=self.primary_cmap, shading='auto')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        units = f'[{particle}/cm$^2$-source]' if incident_particle_yield is None else f'[{particle}/cm$^2$]'
        cbar.set_label(f'log$_{{10}}$(Fluence {units})')

        # Add dual data if available
        if self.dual_data:
            performance_analyzer2: PerformanceAnalyzer = self.dual_data['performance_analyzer']
            particle2 = self.dual_data['spectrometer'].conversion_foil.particle
            density2, response2, X_mesh2, Y_mesh2 = performance_analyzer2.get_recoil_density_map(dx, dy, particle_yield=incident_particle_yield)
            im2 = ax.pcolormesh(X_mesh2, Y_mesh2, np.log10(density2), cmap=self.dual_data['secondary_cmap'], shading='auto', alpha=0.5)
            cbar2 = fig.colorbar(im2, ax=ax, shrink=0.6)
            units = f'[{particle2}/cm$^2$-source]' if incident_particle_yield is None else f'[{particle2}/cm$^2$]'
            cbar2.set_label(f'log$_{{10}}$(Fluence {units})')

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
            im = ax.pcolormesh(X_mesh, Y_mesh, np.log10(response), cmap=self.primary_cmap, shading='auto')
            cbar = fig.colorbar(im, ax=ax, shrink=0.6)
            response_units = '[MeV/cm$^2$-source]' if incident_particle_yield is None else '[MeV/cm$^2$]'
            cbar.set_label(f'log$_{{10}}$(Energy Deposited {response_units})')

            # Add dual data if available
            if self.dual_data:
                im2 = ax.pcolormesh(X_mesh2, Y_mesh2, np.log10(response2), cmap=self.dual_data['secondary_cmap'], shading='auto', alpha=0.5)
                cbar2 = fig.colorbar(im2, ax=ax, shrink=0.6)
                cbar2.set_label(f'log$_{{10}}$(Energy Deposited {response_units})')

            ax.set_xlabel('X Position [cm]')
            ax.set_ylabel('Y Position [cm]')
            ax.set_aspect('equal')
            fig.tight_layout()
            response_filename = filename.replace('.png', '_response.png')
            fig.savefig(response_filename, dpi=150, bbox_inches='tight')
            print(f'Detector response heatmap saved to {response_filename}')
        
    def plot_synthetic_incident_histogram(
        self,
        filename: Optional[str] = None,
    ):
        if filename == None:
            filename = f'{self.spectrometer.figure_directory}/synthetic_{self.spectrometer.conversion_foil.incident_particle}_histogram.png'
        dsr, plasma_temperature, left_edge, right_edge, dsr_energy_range, primary_energy_range, energies, energies_std, response, background = self.performance_analyzer.get_plasma_parameters()
        fwhm = right_edge - left_edge
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # Plot histogram
        ax.step(
            energies,
            response,
            where='pre',
            color=self.primary_color
        )
        ax.fill_between(
            energies,
            response - background,
            response + background,
            step='pre',
            color=self.primary_color,
            alpha=0.3,
        )
        # hist, bins, _ = ax.hist(energies, bins=n_bins, histtype='step', color='tab:blue', density=True)
        # Add energy standard deviation
        # hist_std = self._get_histogram_std(bins, energies, energies_std)
        # ax.fill_between(
        #     bins[1:],
        #     hist - hist_std,
        #     hist + hist_std,
        #     color='tab:blue',
        #     alpha=0.3,
        #     step='pre'
        # )
        
        # Add dual data if available
        if self.dual_data:
            performance_analyzer2: PerformanceAnalyzer = self.dual_data['performance_analyzer']
            dsr2, plasma_temperature2, left_edge2, right_edge2, dsr_energy_range2, primary_energy_range2, energies2, energies_std2, response2, background2 = performance_analyzer2.get_plasma_parameters()
            fwhm2 = right_edge2 - left_edge2
            ax.step(
                energies2,
                response2,
                where='pre',
                color=self.dual_data['secondary_color'],
            )
            ax.fill_between(
                energies2,
                response2 - background2,
                response2 + background2,
                step='pre',
                color=self.dual_data['secondary_color'],
                alpha=0.3,
            )
            
        
        # Highlight dsr and primary range
        ax.axvspan(dsr_energy_range[0], dsr_energy_range[1], color='tab:orange', alpha=0.2)
        ax.axvspan(primary_energy_range[0], primary_energy_range[1], color='tab:green', alpha=0.2)
        
        # Add text to indicate dsr and primary range
        ax.text(
            dsr_energy_range[0] + (dsr_energy_range[1] - dsr_energy_range[0]) / 2,
            max(response) / 1000,
            'DSR',
            ha='center',
            va='top',
            color='tab:red',
            fontsize=12
        )
        
        ax.text(
            primary_energy_range[0] + (primary_energy_range[1] - primary_energy_range[0]) / 2,
            max(response) / 1000,
            'Primary',
            ha='center',
            va='top',
            color='tab:green',
            fontsize=12
        )
        
        # Add double sided arrow to indicate fwhm
        height = max(response)
        ax.annotate(
            f'FWHM = {int(fwhm*1000):3d} keV  \n$T_i$ = {plasma_temperature:.2f} keV   ',
            xy=(right_edge, height/2),
            xytext=(left_edge, height/2),
            arrowprops=dict(
                arrowstyle='<->',
                color='black',
                shrinkA=0,
                shrinkB=0,
            ),
            ha='right',
            va='center',
            fontsize=12
        )
        
        # Add dsr text
        ax.text(
            dsr_energy_range[0] + (dsr_energy_range[1] - dsr_energy_range[0]) / 2,
            height/50,
            f'$dsr$={dsr*100:.1f}%',
            ha='center',
            va='bottom',
            color='black',
            fontsize=12
        )
        
        ax.set_xlabel(f'{self.spectrometer.conversion_foil.incident_particle.capitalize()} Energy [MeV]')
        ax.set_ylabel('PDF')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Synthetic {self.spectrometer.conversion_foil.incident_particle} histogram saved to {filename}')
        
    def _get_histogram_std(self, bins, energies, energies_std, density=True):
        """Helper function to compute histogram standard deviation."""
        # Get which bin each energy falls into
        bin_indices = np.digitize(energies, bins) - 1
        hist_std = np.zeros(len(bins) - 1)
        
        for i, energy_std in enumerate(energies_std):
            # Find the bin index for this energy
            bin_index = bin_indices[i]
            if 0 <= bin_index < len(hist_std):
                # Accumulate variance
                hist_std[bin_index] += (energy_std ** 2)
        
        # Take square root to get standard deviation
        hist_std = np.sqrt(hist_std)
        
        # TODO: Add background contribution — call hodoscope.get_background(neutron_file, photon_file)
        #       and sum per-channel arrays once background files are available here.
        
        # Normalize if density is True
        if density:
            bin_widths = np.diff(bins)
            hist_std /= (len(energies) * bin_widths)
        
        return hist_std
    
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
        fig.suptitle('Comprehensive Performance')
        
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
        ax3.spines['right'].set_position(('outward', 60))
        color_efficiency = 'tab:green'
        ax3.set_ylabel(r'Total Efficiency ($\times$1e6)', color=color_efficiency)
        ax3.tick_params(axis='y', labelcolor=color_efficiency)
        
        # Loop over foils (either one or two)
        for foil, grp in df.groupby('foil'):
            # Extract data from DataFrame
            energies = grp['energy [MeV]'].to_numpy()
            positions = grp['position [m]'].to_numpy()
            position_width = grp['position width [m]'].to_numpy()
            energy_resolutions = grp['resolution [keV]'].to_numpy()
            total_efficiencies = grp['total efficiency'].to_numpy()

            # Plot position curve (center of half-max interval) with ±width/2 band
            position_line = ax1.plot(energies, positions * 100, color=color_position,
                    label=f'Position')
            ax1.fill_between(energies, (positions - position_width / 2) * 100,
                            (positions + position_width / 2) * 100,
                            alpha=0.3, color=color_position)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='y', labelcolor=color_position)
            
            resolution_line = ax2.plot(energies, energy_resolutions, color=color_resolution, marker='o', markersize=4,
                            label=f'Resolution')
            
            efficiency_line = ax3.plot(energies, total_efficiencies*1e6, color=color_efficiency, marker='s', markersize=4,
                            label=f'Efficiency')
            
            # Label lines on their respective axes
            range = energies.max() - energies.min()
            labelLines(position_line, xvals=[energies.min() + 0.25 * range], align=True, fontsize=12)
            labelLines(resolution_line, xvals=[energies.min() + 0.25 * range], align=True, fontsize=12)
            labelLines(efficiency_line, xvals=[energies.min() + 0.75 * range], align=True, fontsize=12)
            
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
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
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
            if interaction.generates_recoil_particles:
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
                if interaction.generates_recoil_particles:
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
