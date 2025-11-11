"""Plotting methods for MPR spectrometer visualization."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import norm
from scipy.interpolate import griddata
from labellines import labelLines

# Set default plotting parameters
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

from ..core.spectrometer import MPRSpectrometer
from ..core.dual_foil_spectrometer import DualFoilSpectrometer
from ..analysis.performance import PerformanceAnalyzer

if TYPE_CHECKING:
    from ..analysis.parameter_sweep import FoilSweeper
    import pandas as pd

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
        Plot hydron distribution in the focal plane.
        
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
            # Convert to cm
            detector_width = self.spectrometer.hodoscope.detector_width * 100
            detector_height = self.spectrometer.hodoscope.detector_height * 100
            edges = self.spectrometer.hodoscope.channel_edges * 100
            
            # Draw detector boundaries
            for i, edge in enumerate(edges):                
                # Vertical lines for detector edges
                # Left-most and right-most edges
                if i == 0 or i == len(edges) - 1:
                    line_style = '-'
                    line_width = 1.0
                else:
                    line_style = '--'
                    line_width = 0.5
                ax.vlines(edge, -detector_height/2, detector_height/2, 
                          color='black', linestyle=line_style, linewidth=line_width)
            
            # Horizontal lines for detector top/bottom
            ax.hlines([-detector_height/2, detector_height/2], edges[0], 
                      edges[-1], color='black', linewidth=1.0)
        
        # Scatter plot of hydron positions
        hydron_energies = self.spectrometer.input_beam[:, 4] * self.spectrometer.reference_energy + self.spectrometer.reference_energy
        scatter = ax.scatter(
            self.spectrometer.output_beam[:, 0]*100, 
            self.spectrometer.output_beam[:, 2]*100,
            c=hydron_energies,
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
            hydron_energies2 = spec2.input_beam[:, 4] * spec2.reference_energy + spec2.reference_energy
            scatter2 = ax.scatter(
                spec2.output_beam[:, 0]*100, 
                spec2.output_beam[:, 2]*100,
                c=hydron_energies2,
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
        
        # Color by hydron energy
        x_pos = self.spectrometer.output_beam[:, 0]
        x_angle = self.spectrometer.output_beam[:, 1]
        y_pos = self.spectrometer.output_beam[:, 2]
        y_angle = self.spectrometer.output_beam[:, 3]
        hydron_energies = self.spectrometer.input_beam[:, 4] * self.spectrometer.reference_energy + self.spectrometer.reference_energy
        
        # X-Y position plot
        scatter1 = axes[0, 0].scatter(
            x_pos, y_pos, c=hydron_energies,
            s=2.0, cmap=self.primary_cmap, alpha=0.7
        )
        axes[0, 0].set_xlabel('X Position [cm]')
        axes[0, 0].set_ylabel('Y Position [cm]')
        axes[0, 0].set_title('X-Y Position')
        axes[0, 0].grid(True, alpha=0.3)
        
        # X position vs X angle
        scatter2 = axes[0, 1].scatter(
            x_pos, x_angle * 1000, c=hydron_energies, 
            s=2.0, cmap=self.primary_cmap, alpha=0.7
        )
        axes[0, 1].set_xlabel('X Position [cm]')
        axes[0, 1].set_ylabel('X Angle [mrad]')
        axes[0, 1].set_title('X Position-Angle')
        axes[0, 1].grid(True, alpha=0.3)
        
        # X position vs energy
        scatter3 = axes[1, 0].scatter(
            x_pos, hydron_energies, c=hydron_energies,
            s=2.0, cmap=self.primary_cmap, alpha=0.7
        )
        axes[1, 0].set_xlabel('X Position [cm]')
        axes[1, 0].set_ylabel('E$_{hydron}$ [MeV]')
        axes[1, 0].set_title('X Position-Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Y position vs Y angle
        scatter4 = axes[1, 1].scatter(
            y_pos, y_angle * 1000, c=hydron_energies,
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
            x_pos2 = spec2.output_beam[:, 0]
            x_angle2 = spec2.output_beam[:, 1]
            y_pos2 = spec2.output_beam[:, 2]
            y_angle2 = spec2.output_beam[:, 3]
            hydron_energies2 = spec2.input_beam[:, 4] * spec2.reference_energy + spec2.reference_energy
            
            # X-Y position plot
            scatter1 = axes[0, 0].scatter(
                x_pos2, y_pos2, c=hydron_energies2,
                s=2.0, cmap=self.dual_data['secondary_cmap'], alpha=0.7
            )
            
            # X position vs X angle
            scatter2 = axes[0, 1].scatter(
                x_pos2, x_angle2 * 1000, c=hydron_energies2,
                s=2.0, cmap=self.dual_data['secondary_cmap'], alpha=0.7
            )
            
            # X position vs energy
            scatter3 = axes[1, 0].scatter(
                x_pos2, hydron_energies2, c=hydron_energies2,
                s=2.0, cmap=self.dual_data['secondary_cmap'], alpha=0.7
            )
            
            # Y position vs Y angle
            scatter4 = axes[1, 1].scatter(
                y_pos2, y_angle2 * 1000, c=hydron_energies2,
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
        output_energies = self.spectrometer.input_beam[:, 4] * self.spectrometer.reference_energy + self.spectrometer.reference_energy
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
        cbar.set_label('Hydron Energy [MeV]')
        
        ax.set_xlabel('X Position [cm]')
        ax.set_ylabel('Y Position [cm]')
        ax.set_title('Focal Plane Distribution')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Print summary statistics
        print(f'Characteristic ray analysis complete:')
        print(f'  Total rays generated: {len(self.spectrometer.input_beam)}')
        print(f'  Energy range: {min_energy:.2f} - {max_energy:.2f} MeV')
        print(f'  X position range: {self.spectrometer.output_beam[:, 0].min()*100:.2f} - {self.spectrometer.output_beam[:, 0].max()*100:.2f} cm')
        print(f'  Y position range: {self.spectrometer.output_beam[:, 2].min()*100:.2f} - {self.spectrometer.output_beam[:, 2].max()*100:.2f} cm')
        print(f'Characteristic ray plot saved to {filename}')
    
    def plot_simple_position_histogram(
        self, 
        filename: Optional[str] = None, 
        num_bins: int = 40
    ) -> None:
        """
        Plot a simple histogram of hydron counts vs horizontal position.
        
        Args:
            filename: Output filename for the plot
            num_bins: Number of histogram bins
        """
        if filename == None:
            filename = f'{self.spectrometer.figure_directory}/counts_vs_position.png'
        
        if len(self.spectrometer.output_beam) == 0:
            raise ValueError("No output beam data available. Run apply_transfer_map() first.")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        x_positions = self.spectrometer.output_beam[:, 0]*100 # cm
        x_range = (x_positions.min(), x_positions.max())
        
        counts, bins, patches = ax.hist(
            x_positions, 
            bins=np.linspace(x_range[0], x_range[1], num_bins),
            density=True,
            alpha=0.7,
            color=self.primary_color,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add dual data if available
        if self.dual_data:
            spec2: MPRSpectrometer = self.dual_data['spectrometer']
            x_positions2 = spec2.output_beam[:, 0]*100 # cm
            x_range2 = (x_positions2.min(), x_positions2.max())
            
            counts2, bins2, patches2 = ax.hist(
                x_positions2, 
                bins=np.linspace(x_range2[0], x_range2[1], num_bins),
                density=True,
                alpha=0.7,
                color=self.dual_data['secondary_color'],
                edgecolor='black',
                linewidth=0.5,
                label=self.dual_data['secondary_label']
            )
            ax.legend()
        
        ax.set_xlabel('Horizontal Position [cm]')
        ax.set_ylabel('Counts')
        ax.set_title('Hydron Counts vs Position')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Position histogram saved to {filename}')
        
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
        aperture_radius = self.spectrometer.conversion_foil.aperture_radius * 100
        
        # Foil (vertical line at z=0)
        ax.vlines(0, -foil_radius, foil_radius, color='tab:purple', linewidth=3, label='Conversion Foil')
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
        ax.vlines(aperture_distance, -aperture_radius, aperture_radius, 
                color='tab:orange', linewidth=3, label='Aperture')
        # Add text label for aperture
        ax.text(
            aperture_distance,
            aperture_radius,
            'Aperture',
            ha='right',
            va='bottom',
            color='tab:orange',
            fontsize=12
        )
        
        # Draw sample of input rays
        num_rays_to_plot = min(len(self.spectrometer.input_beam), 200)  # Limit for clarity
        z_coords = np.linspace(0, aperture_distance, 20)
        
        for i in range(0, len(self.spectrometer.input_beam), max(1, len(self.spectrometer.input_beam) // num_rays_to_plot)):
            ray = self.spectrometer.input_beam[i]
            x0, angle_x, y0, angle_y = ray[:4]
            y0 *= 100 # cm
            
            # Calculate ray trajectory
            slope = np.tan(angle_y)
            y_trajectory = slope * z_coords + y0
            
            ax.plot(z_coords, y_trajectory, alpha=0.4, color=self.primary_color, linewidth=0.5)
                
        # Plot dual data if available
        if self.dual_data:
            spec2: MPRSpectrometer = self.dual_data['spectrometer']
            for i in range(0, len(spec2.input_beam), max(1, len(spec2.input_beam) // num_rays_to_plot)):
                ray = spec2.input_beam[i]
                x0, angle_x, y0, angle_y = ray[:4]
                y0 *= 100 # cm
                
                # Calculate ray trajectory
                slope = np.tan(angle_y)
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
        max_extent = 1.5 * max(foil_radius, aperture_radius)
        ax.set_ylim(-max_extent, max_extent)
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Input ray geometry plot saved to {filename}')

    def plot_hydron_density_heatmap(
        self, 
        filename: Optional[str] = None,
        dx: float = 0.5, 
        dy: float = 0.5,
        neutron_yield: Optional[float] = None
    ) -> None:
        """
        Plot a heatmap of hydron density in the focal plane.
        
        Args:
            filename: Output filename for the plot
            dx: X-direction resolution in cm
            dy: Y-direction resolution in cm
        """
        if filename == None:
            filename = f'{self.spectrometer.figure_directory}/hydron_density_heatmap.png'
        
        density, X_mesh, Y_mesh = self.performance_analyzer.get_hydron_density_map(dx, dy, foil_distance=6.0, neutron_yield=neutron_yield)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.pcolormesh(X_mesh, Y_mesh, np.log10(density), cmap=self.primary_cmap, shading='auto')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        particle = self.spectrometer.conversion_foil.particle
        units = f'[{particle}/cm$^2$-source]' if neutron_yield == None else f'[{particle}/cm$^2$]'
        cbar.set_label(f'log$_{{10}}$(Fluence {units})')
        
        # Add dual data if available
        if self.dual_data:
            performance_analyzer2: PerformanceAnalyzer = self.dual_data['performance_analyzer']
            density2, X_mesh2, Y_mesh2 = performance_analyzer2.get_hydron_density_map(dx, dy)
            im2 = ax.pcolormesh(X_mesh2, Y_mesh2, np.log10(density2), cmap=self.dual_data['secondary_cmap'], shading='auto', alpha=0.5)
            cbar2 = fig.colorbar(im2, ax=ax, shrink=0.6)
            particle2 = self.dual_data['spectrometer'].conversion_foil.particle
            units = f'[{particle2}/cm$^2$-source]' if neutron_yield == None else f'[{particle2}/cm$^2$]'
            cbar2.set_label(f'log$_{{10}}$(Fluence {units})')
        
        ax.set_xlabel('X Position [cm]')
        ax.set_ylabel('Y Position [cm]')
        ax.set_aspect('equal')
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Hydron density heatmap saved to {filename}')
        
    def plot_synthetic_neutron_histogram(
        self,
        n_bins = 200,
        filename: Optional[str] = None,
    ):
        if filename == None:
            filename = f'{self.spectrometer.figure_directory}/synthetic_neutron_histogram.png'
        dsr, plasma_temperature, fwhm, dsr_energy_range, primary_energy_range, energies = self.performance_analyzer.get_plasma_parameters(n_bins=n_bins)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # Plot histogram
        hist, bins, _ = ax.hist(energies, bins=n_bins, histtype='step', color='tab:blue', linewidth=3, density=True)
        
        # Add dual data if available
        if self.dual_data:
            performance_analyzer2: PerformanceAnalyzer = self.dual_data['performance_analyzer']
            dsr2, plasma_temperature2, fwhm2, dsr_energy_range2, primary_energy_range2, energies2 = performance_analyzer2.get_plasma_parameters(n_bins=n_bins)
            hist2, bins2, _ = ax.hist(energies2, bins=n_bins, histtype='step', color='tab:orange', linewidth=3, density=True)
        
        # Highlight dsr and primary range
        ax.axvspan(dsr_energy_range[0], dsr_energy_range[1], color='tab:red', alpha=0.2)
        ax.axvspan(primary_energy_range[0], primary_energy_range[1], color='tab:green', alpha=0.2)
        
        # Add text to indicate dsr and primary range
        ax.text(
            dsr_energy_range[0] + (dsr_energy_range[1] - dsr_energy_range[0]) / 2,
            max(hist) / 1000,
            'DSR',
            ha='center',
            va='top',
            color='tab:red',
            fontsize=12
        )
        
        ax.text(
            primary_energy_range[0] + (primary_energy_range[1] - primary_energy_range[0]) / 2,
            max(hist) / 1000,
            'Primary',
            ha='center',
            va='top',
            color='tab:green',
            fontsize=12
        )
        
        # Add double sided arrow to indicate fwhm
        height = max(hist)
        # TODO: fix this weird offset from the fwhm
        peak_center = bins[np.argmax(hist) + 2]
        ax.annotate(
            f'FWHM = {int(fwhm*1000):3d} keV  \n$T_i$ = {plasma_temperature:.2f} keV   ',
            xy=(peak_center + fwhm/2, height/2),
            xytext=(peak_center - fwhm/2, height/2),
            arrowprops=dict(
                arrowstyle='<->',
                color='black',
                linewidth=2,
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
        
        ax.set_xlabel('Neutron Energy [MeV]')
        ax.set_ylabel('PDF')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Synthetic neutron histogram saved to {filename}')
    
    def plot_monoenergetic_analysis(
        self,  
        neutron_energy: float, 
        mean_pos: float, 
        std_dev: float,
        filename: Optional[str] = None,
    ) -> None:
        """Generate analysis plots for monoenergetic performance."""
        if filename == None:
            filename = (
                f'{self.spectrometer.figure_directory}/' 
                f'Monoenergetic_En{neutron_energy:.1f}MeV_'
                f'T{self.spectrometer.conversion_foil.thickness_um:.0f}um_'
                f'E0{self.spectrometer.reference_energy:.1f}MeV.png'
            )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram of x positions
        x_positions = self.spectrometer.output_beam[:, 0]*100 # cm
        
        axes[0].hist(x_positions, bins=30, alpha=0.7, density=True, label='Simulation')
        
        # Gaussian fit overlay
        x_fit = np.linspace(x_positions.min(), x_positions.max(), 100)
        gaussian_fit = norm.pdf(x_fit, mean_pos * 100, std_dev * 100)
        axes[0].plot(x_fit, gaussian_fit, 'r-', label='Gaussian Fit', linewidth=2)
        
        axes[0].set_xlabel('X Position [cm]')
        axes[0].set_ylabel('Probability Density')
        axes[0].set_title(f'X-Position Distribution\n{neutron_energy:.1f} MeV Neutrons')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Scatter plot
        hydron_energies = self.spectrometer.input_beam[:, 4] * self.spectrometer.reference_energy + self.spectrometer.reference_energy
        scatter = axes[1].scatter(
            self.spectrometer.output_beam[:, 0]*100,
            self.spectrometer.output_beam[:, 2]*100,
            c=hydron_energies,
            s=1.0,
            cmap=self.primary_cmap,
            alpha=0.6
        )
        
        fig.colorbar(scatter, ax=axes[1], label=f'{self.spectrometer.conversion_foil.particle.capitalize()} Energy [MeV]')
        axes[1].set_xlabel('X Position [cm]')
        axes[1].set_ylabel('Y Position [cm]')
        axes[1].set_title(f'Focal Plane Distribution\n{neutron_energy:.1f} MeV Neutrons')
        axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        print(filename)
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def plot_performance(
        self,  
        energies: np.ndarray, 
        positions: np.ndarray, 
        position_uncertainties: np.ndarray,
        energy_resolutions: np.ndarray,
        total_efficiencies: np.ndarray,
        filename: Optional[str] = None
    ) -> None:
        """Generate comprehensive performance plot with shared x-axis."""
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        fig.suptitle('Comprehensive Performance')
        
        # Left y-axis: position
        color_position = 'tab:blue'
        ax1.set_xlabel('Neutron Energy [MeV]')
        ax1.set_ylabel('Hydron Position [cm]', color=color_position)
        
        # Plot position curve
        ax1.plot(energies, positions * 100, color=color_position, linewidth=2,
                 label=f'Position')
        ax1.fill_between(energies, (positions - position_uncertainties) * 100, 
                        (positions + position_uncertainties) * 100,
                        alpha=0.3, color=color_position)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor=color_position)
        
        # Right y-axis: Resolution and Efficiency
        ax2 = ax1.twinx()
        
        # Plot energy resolution
        color_resolution = 'tab:red'
        ax2.plot(energies, energy_resolutions, color=color_resolution, 
                        linewidth=2, marker='o', markersize=4,
                        label=f'Resolution')
        ax2.set_ylabel('Energy Resolution [keV]', color=color_resolution)
        ax2.tick_params(axis='y', labelcolor=color_resolution)
        
        # Detection Efficiency
        ax3 = ax1.twinx()
        # Offset the third axis to the right
        ax3.spines['right'].set_position(('outward', 60))
        color_efficiency = 'tab:green'
        ax3.plot(energies, total_efficiencies*1e6, color=color_efficiency, 
                        linewidth=2, marker='s', markersize=4,
                        label=f'Efficiency')
        ax3.set_ylabel(r'Efficiency[$\times$1e-6]', color=color_efficiency)
        ax3.tick_params(axis='y', labelcolor=color_efficiency)
        
        # Label lines on their respective axes
        range = energies.max() - energies.min()
        labelLines(ax1.get_lines(), xvals=[energies.min() + 0.75 * range], align=True, fontsize=12)
        labelLines(ax2.get_lines(), xvals=[energies.min() + 0.25 * range], align=True, fontsize=12)
        labelLines(ax3.get_lines(), xvals=[energies.min() + 0.75 * range], align=True, fontsize=12)
        
        # Make sure x-axis limits are consistent
        x_min, x_max = energies.min(), energies.max()
        x_margin = (x_max - x_min) * 0.02
        ax1.set_xlim(x_min - x_margin, x_max + x_margin)
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Print summary statistics
        print(f'\nPerformance Summary:')
        print(f'  Energy range: {energies.min():.2f} - {energies.max():.2f} MeV')
        print(f'  Position range: {positions.min()*100:.2f} - {positions.max()*100:.2f} cm')
        print(f'  Average resolution: {np.mean(energy_resolutions):.1f} keV')
        print(f'  Average efficiency: {np.mean(total_efficiencies):.3e}')
        print(f'  Best resolution: {np.min(energy_resolutions):.1f} keV at {energies[np.argmin(energy_resolutions)]:.2f} MeV')
        print(f'  Best efficiency: {np.max(total_efficiencies):.1e} at {energies[np.argmax(total_efficiencies)]:.2f} MeV')
        print(f'Comprehensive performance plot saved to {filename}')
        
    def plot_data(
        self,
        energy_MeV: float, 
        figure_directory: Optional[str] = None,
        filename_prefix: Optional[str] = None,
        angle_range: Tuple[float, float] = (0, np.pi/2),
        num_angles: int = 100
    ) -> None:
        """
        Plot differential cross section, cross sections, and stopping power data as three separate plots.
        
        Args:
            energy_MeV: Specific energy in MeV for differential cross section plot
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
        
        diff_xs_lab = foil.calculate_differential_xs_lab(angles_rad, energy_MeV)
        
        axs[0].plot(angles_deg, diff_xs_lab * 1e28, 'tab:blue', linewidth=2)
        axs[0].set_xlabel('Angle [deg]')
        axs[0].set_ylabel('d$\sigma$/d$\Omega$ [barns/sr]')
        axs[0].grid(True, alpha=0.3)
        
        # ========== Plot 2: Cross Sections vs Energy ==========
        # Use raw data from files (no interpolation)
        # n-hydron cross section data
        nh_energies_eV = foil.nh_cross_section_data[0]
        nh_energies_MeV = nh_energies_eV * 1e-6  # Convert eV to MeV
        nh_xs_barns = foil.nh_cross_section_data[1]  # Already in barns
        
        # n-C12 cross section data
        nc12_energies_eV = foil.nc12_cross_section_data[0]
        nc12_idx = (nc12_energies_eV >= np.min(nh_energies_eV)) & (nc12_energies_eV <= np.max(nh_energies_eV))
        nc12_energies_MeV = nc12_energies_eV[nc12_idx] * 1e-6  # Convert eV to MeV
        nc12_xs_barns = foil.nc12_cross_section_data[1, nc12_idx]  # Already in barns
        
        axs[1].plot(nh_energies_MeV, nh_xs_barns, 'tab:blue', linewidth=2, 
                label=f'n-{foil.particle[0]} elastic')
        axs[1].plot(nc12_energies_MeV, nc12_xs_barns, 'g-', linewidth=2, 
                label='n-C12 elastic')
        axs[1].axvline(energy_MeV, color='k', linestyle='--', alpha=0.7, 
                    label=f'Current energy: {energy_MeV:.1f} MeV')
        
        axs[1].set_xlabel('Neutron Energy [MeV]')
        axs[1].set_ylabel('Cross Section [barns]')
        axs[1].grid(True, alpha=0.3)
        axs[1].set_yscale('log')
        axs[1].set_xscale('log')
        
        # ========== Plot 3: Stopping Power vs Energy ==========
        # Use raw SRIM data (no interpolation)
        srim_energies_MeV = foil.srim_data[0]  # Already in MeV
        srim_stopping_power = foil.srim_data[1] + foil.srim_data[2]  # Electronic + nuclear stopping
        
        axs[2].plot(srim_energies_MeV, srim_stopping_power, 'tab:blue', linewidth=2)
        
        axs[2].set_xlabel(f'Hydron Energy [MeV]')
        axs[2].set_ylabel('Stopping Power [MeV/mm]')
        axs[2].grid(True, alpha=0.3)
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')
        
        # Add dual data if available
        if self.dual_data:
            spec2: MPRSpectrometer = self.dual_data['spectrometer']
            foil2 = spec2.conversion_foil
            title = f'{foil.particle} and {foil2.particle} at {energy_MeV:.2f} MeV'
            
            # differential cross section data
            diff_xs_lab2 = foil2.calculate_differential_xs_lab(angles_rad, energy_MeV)
            
            axs[0].plot(angles_deg, diff_xs_lab2 * 1e28, 'darkorange', linewidth=2)
            
            # n-hydron cross section data
            nh_energies_eV2 = foil2.nh_cross_section_data[0]
            nh_energies_MeV2 = nh_energies_eV2 * 1e-6  # Convert eV to MeV
            nh_xs_barns2 = foil2.nh_cross_section_data[1]  # Already in barns
            
            axs[1].plot(nh_energies_MeV2, nh_xs_barns2, 'darkorange', linewidth=2, 
                    label=f'n-{foil2.particle[0]} elastic')
            
            # Stopping power for dual data
            srim_energies_MeV2 = foil2.srim_data[0]  # Already in MeV
            srim_stopping_power2 = foil2.srim_data[1] + foil2.srim_data[2]  # Electronic + nuclear stopping
            
            axs[2].plot(srim_energies_MeV2, srim_stopping_power2, 'darkorange', linewidth=2)
        
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
        ax.plot(foil_r * np.cos(theta), foil_r * np.sin(theta), 'k-', 
               linewidth=2, label='Foil boundary')
        
        # Draw y=0 dividing line
        ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Y=0 divider')
        
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
        self.label = label if label is not None else name
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
        
        # Initialize mappable contour
        contour_obj = None
        
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