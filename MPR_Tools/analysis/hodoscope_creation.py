"""Automatic hodoscope bin generation from a performance curve."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from ..core.spectrometer import MPRSpectrometer
from ..analysis.performance import PerformanceAnalyzer

# Set default plotting parameters
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16


class HodoscopeCreator:
    """
    Generates optimal hodoscope bin boundaries from a pre-computed performance curve.

    Algorithm
    ---------
    Starting at a seed energy, each bin's position width equals the full width
    at ``fractional_max`` of the peak (FWHM when ``fractional_max=0.5``).
    Graphically, this traces a staircase on the position-vs-energy curve:

    * A vertical line spans the full width of the current bin.
    * From the top of that line, a horizontal line extends to the right until it
      meets the lower edge of the next bin's width interval.
    * From the bottom of that line, a horizontal line extends to the left until it
      meets the upper edge of the previous bin's width interval.

    Bins are propagated both upward and downward from the seed energy until the
    energy range of the performance curve is exhausted.
    """

    def __init__(self, spectrometer: MPRSpectrometer) -> None:
        self.spectrometer = spectrometer
        self.performance_analyzer = PerformanceAnalyzer(spectrometer)

    def generate_bins(
        self,
        start_energy: float,
        performance_curve_file: Optional[str] = None,
        peak_energy_min: Optional[float] = None,
        peak_energy_max: Optional[float] = None,
        n_peak_bins: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate hodoscope bin boundaries using the staircase algorithm.

        Args:
            start_energy: Seed incident particle energy in MeV (e.g. 14.0).
            performance_curve_file: Path to a performance CSV produced by
                ``PerformanceAnalyzer.generate_performance_curve()``.
                Defaults to the standard location inside the spectrometer's
                data directory.
            peak_energy_min: If provided, the lower bound of a signal peak region.
            peak_energy_max: If provided, the upper bound of a signal peak region.
            n_peak_bins: If provided along with the peak energy bounds, the number of narrow bins to place within the signal peak region.

        Returns:
            bin_energies  : Center energy of each bin [MeV], shape (N,).
            bin_positions : Center position of each bin [m], shape (N,).
            bin_edges     : Position edges bounding each bin [m], shape (N+1,).
                            ``bin_edges[i]`` is the lower edge of bin ``i``;
                            ``bin_edges[i+1]`` is its upper edge.
        """
        # Load performance curve
        performance_curve_file = performance_curve_file or 'comprehensive_performance.csv'
        performance_df = pd.read_csv(
            f'{self.spectrometer.data_directory}/{performance_curve_file}'
        )

        energies = performance_df['energy [MeV]'].to_numpy()
        positions = performance_df['position [m]'].to_numpy()
        # Default width is FWHM, but can be any fractional max interval depending on how the performance curve was generated
        widths = performance_df['position width [m]'].to_numpy() 

        # Position interpolator (built first so we can use it for the signal region below)
        position_of_energy = interp1d(energies, positions, bounds_error=False,
                                      fill_value=(positions[0], positions[-1]))

        # Narrow bins in the signal peak region so it is well resolved
        if peak_energy_min and peak_energy_max and n_peak_bins:
            # Convert target energy bin width to position width using the position interpolator
            peak_range_width = (position_of_energy(peak_energy_max) - position_of_energy(peak_energy_min)) / n_peak_bins
            signal_region_mask = (energies >= peak_energy_min) & (energies <= peak_energy_max)
            widths = np.where(signal_region_mask, np.minimum(widths, peak_range_width), widths)

        # Half-width bounds as a function of energy
        lower_bounds = positions - widths / 2  # lower edge of each bin
        upper_bounds = positions + widths / 2  # upper edge of each bin

        # Width interpolator (built after widths are finalized)
        width_of_energy = interp1d(energies, widths, bounds_error=False,
                                   fill_value=(widths[0], widths[-1]))

        # Inverse interpolators for the staircase step:
        #   given a target position on the lower/upper bound curve, find energy
        energy_of_lower_bound = interp1d(lower_bounds, energies, bounds_error=False,
                                         fill_value='extrapolate')
        energy_of_upper_bound = interp1d(upper_bounds, energies, bounds_error=False,
                                         fill_value='extrapolate')

        energy_min, energy_max = energies[0], energies[-1]

        # --- Seed bin at start_energy ---
        seed_position = float(position_of_energy(start_energy))
        seed_width = float(width_of_energy(start_energy))
        seed_top = seed_position + seed_width / 2
        seed_bottom = seed_position - seed_width / 2

        upper_bin_energies = [start_energy]
        upper_bin_positions = [seed_position]

        # --- Propagate upward (increasing energy / position) ---
        # --- Propagate upward (increasing energy / position) ---
        current_top = seed_top
        while True:
            next_energy = float(energy_of_lower_bound(current_top))
            next_energy = min(next_energy, energy_max)  # clamp to range
            next_position = float(position_of_energy(next_energy))
            next_width = float(width_of_energy(next_energy))
            upper_bin_energies.append(next_energy)
            upper_bin_positions.append(next_position)
            if next_energy >= energy_max:
                break
            current_top = next_position + next_width / 2

        # --- Propagate downward (decreasing energy / position) ---
        lower_bin_energies = []
        lower_bin_positions = []
        current_bottom = seed_bottom
        while True:
            prev_energy = float(energy_of_upper_bound(current_bottom))
            prev_energy = max(prev_energy, energy_min)  # clamp to range
            prev_position = float(position_of_energy(prev_energy))
            prev_width = float(width_of_energy(prev_energy))
            lower_bin_energies.append(prev_energy)
            lower_bin_positions.append(prev_position)
            if prev_energy <= energy_min:
                break
            current_bottom = prev_position - prev_width / 2

        # Combine: lower bins reversed so array is in ascending energy order
        bin_energies = np.array(lower_bin_energies[::-1] + upper_bin_energies)
        bin_positions = np.array(lower_bin_positions[::-1] + upper_bin_positions)

        # Bin edges: lower edge of the first bin + upper edge of every bin
        bin_widths = np.array([float(width_of_energy(energy)) for energy in bin_energies])
        bottom_edge = bin_positions[0] - bin_widths[0] / 2
        upper_edges = bin_positions + bin_widths / 2
        bin_edges = np.concatenate([[bottom_edge], upper_edges])

        return bin_energies, bin_positions, bin_edges

    def plot_bins(
        self,
        bin_energies: np.ndarray,
        bin_positions: np.ndarray,
        performance_curve_file: Optional[str] = None,
        filename: Optional[str] = None,
        peak_energy_min: Optional[float] = None,
        peak_energy_max: Optional[float] = None,
    ) -> None:
        """
        Plot the staircase bin layout overlaid on the performance curve.

        Args:
            bin_energies : Center energy of each bin [MeV].
            bin_positions: Center position of each bin [m].
            performance_curve_file: Optional path to performance CSV.
            filename: Output filename.  Defaults to
                ``<figure_directory>/hodoscope_bins.png``.
            peak_energy_min: If provided, the lower bound of a signal peak region to highlight on the plot.
            peak_energy_max: If provided, the upper bound of a signal peak region to highlight on the plot.
        """
        # Load performance curve
        performance_curve_file = performance_curve_file or 'comprehensive_performance.csv'
        performance_df = pd.read_csv(
            f'{self.spectrometer.data_directory}/{performance_curve_file}'
        )

        energies = performance_df['energy [MeV]'].to_numpy()
        positions = performance_df['position [m]'].to_numpy()
        widths = performance_df['position width [m]'].to_numpy()

        if filename is None:
            filename = f'{self.spectrometer.figure_directory}/hodoscope_bins.png'
        else:
            filename = f'{self.spectrometer.figure_directory}/{filename}'

        # Interpolator for drawing the staircase
        width_of_energy = interp1d(energies, widths, bounds_error=False,
                                      fill_value=(widths[0], widths[-1]))

        fig, ax = plt.subplots(figsize=(9, 5))

        # Background: performance curve with ±width/2 band
        ax.plot(energies, positions * 100, color='tab:orange', linewidth=2, label='Position')
        ax.fill_between(
            energies,
            (positions - widths / 2) * 100,
            (positions + widths / 2) * 100,
            alpha=0.25, color='tab:orange', label='width',
        )

        # Staircase lines: each vertical at bin center, each horizontal at the junction
        # between adjacent bins. At the borders, extend by one bin spacing to close the staircase.
        staircase_color = 'tab:blue'

        # Left border: extend the first bin's bottom edge to the start of the energy range
        first_bin_width  = float(width_of_energy(bin_energies[0]))
        first_bin_bottom = (bin_positions[0] - first_bin_width / 2) * 100
        ax.plot([energies[0], bin_energies[0]], [first_bin_bottom, first_bin_bottom],
                color=staircase_color, linewidth=1.2, alpha=0.8)

        # Right border: extend the last bin's top edge to the end of the energy range
        last_bin_width = float(width_of_energy(bin_energies[-1]))
        last_bin_top   = (bin_positions[-1] + last_bin_width / 2) * 100
        ax.plot([bin_energies[-1], energies[-1]], [last_bin_top, last_bin_top],
                color=staircase_color, linewidth=1.2, alpha=0.8)
        breakpoint()
        for i, (bin_energy, bin_position) in enumerate(zip(bin_energies, bin_positions)):
            bin_width = float(width_of_energy(bin_energy))
            bin_bottom = (bin_position - bin_width / 2) * 100
            bin_top = (bin_position + bin_width / 2) * 100

            # Vertical line spanning the bin width at the bin center energy
            ax.plot([bin_energy, bin_energy], [bin_bottom, bin_top],
                    color=staircase_color, linewidth=1.2, alpha=0.8)

            # Horizontal line at the junction between this bin and the next (top of this bin)
            if i < len(bin_energies) - 1:
                next_bin_energy = bin_energies[i + 1]
                ax.plot([bin_energy, next_bin_energy], [bin_top, bin_top],
                        color=staircase_color, linewidth=1.2, alpha=0.8)

        # Mark bin centers
        ax.scatter(
            bin_energies, bin_positions * 100,
            color=staircase_color, zorder=5, s=30,
            label=f'{len(bin_energies)} bins',
        )

        # Highlight signal peak region if provided
        if peak_energy_min and peak_energy_max:
            ax.axvspan(peak_energy_min, peak_energy_max, color='tab:green', alpha=0.2, zorder=0, label='Signal Peak Region')

        incident_particle = self.spectrometer.conversion_foil.incident_particle.capitalize()
        ax.set_xlabel(f'{incident_particle} Energy [MeV]')
        ax.set_ylabel('Position [cm]')
        ax.set_title('Hodoscope Bin Layout')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Hodoscope bin plot saved to {filename}')
