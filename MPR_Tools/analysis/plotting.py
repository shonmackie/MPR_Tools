"""Plotting methods for MPR spectrometer visualization."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from labellines import labelLines

if TYPE_CHECKING:
    from ..core.spectrometer import MPRSpectrometer

class SpectrometerPlotter:
    """Handles all plotting functionality for MPR spectrometer."""
    
    def __init__(self, spectrometer: MPRSpectrometer) -> None:
        self.spectrometer = spectrometer
    
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
            cmap='plasma',
            alpha=0.7
        )
        
        fig.colorbar(scatter, label=f'{self.spectrometer.conversion_foil.particle.capitalize()} Energy [MeV]')
        ax.set_xlabel('Horizontal Position [cm]')
        ax.set_ylabel('Vertical Position [cm]')
        ax.set_title(f'{self.spectrometer.conversion_foil.particle.capitalize()} Distribution in Focal Plane')
        ax.grid(True, alpha=0.3)
        
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
        fig.suptitle('Phase Space', fontsize=16)
        
        # Color by hydron energy
        hydron_energies = self.spectrometer.input_beam[:, 4] * self.spectrometer.reference_energy + self.spectrometer.reference_energy
        
        # X-Y position plot
        scatter1 = axes[0, 0].scatter(
            self.spectrometer.output_beam[:, 0] * 100, self.spectrometer.output_beam[:, 2] * 100,
            c=hydron_energies, s=2.0, cmap='plasma', alpha=0.7
        )
        axes[0, 0].set_xlabel('X Position [cm]')
        axes[0, 0].set_ylabel('Y Position [cm]')
        axes[0, 0].set_title('X-Y Position')
        axes[0, 0].grid(True, alpha=0.3)
        
        # X position vs X angle
        scatter2 = axes[0, 1].scatter(
            self.spectrometer.output_beam[:, 0] * 100, self.spectrometer.output_beam[:, 1] * 1000,
            c=hydron_energies, s=2.0, cmap='plasma', alpha=0.7
        )
        axes[0, 1].set_xlabel('X Position [cm]')
        axes[0, 1].set_ylabel('X Angle [mrad]')
        axes[0, 1].set_title('X Position-Angle')
        axes[0, 1].grid(True, alpha=0.3)
        
        # X position vs energy
        scatter3 = axes[1, 0].scatter(
            self.spectrometer.output_beam[:, 0] * 100, self.spectrometer.input_beam[:, 4] * self.spectrometer.reference_energy + self.spectrometer.reference_energy,
            c=hydron_energies, s=2.0, cmap='plasma', alpha=0.7
        )
        axes[1, 0].set_xlabel('X Position [cm]')
        axes[1, 0].set_ylabel('E$_{proton}$ [MeV]')
        axes[1, 0].set_title('X Position-Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Y position vs Y angle
        scatter4 = axes[1, 1].scatter(
            self.spectrometer.output_beam[:, 2] * 100, self.spectrometer.output_beam[:, 3] * 1000,
            c=hydron_energies, s=2.0, cmap='plasma', alpha=0.7
        )
        axes[1, 1].set_xlabel('Y Position [cm]')
        axes[1, 1].set_ylabel('Y Angle [mrad]')
        axes[1, 1].set_title('Y Position-Angle')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add colorbar
        fig.colorbar(scatter1, ax=axes, label=f'{self.spectrometer.conversion_foil.particle.capitalize()} Energy [MeV]', shrink=0.8)
        
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
        fig.suptitle('Characteristic Ray Analysis', fontsize=16)
        
        # Focal plane distribution        
        # Scatter plot colored by energy
        output_energies = self.spectrometer.input_beam[:, 4] * self.spectrometer.reference_energy + self.spectrometer.reference_energy
        scatter = ax.scatter(
            self.spectrometer.output_beam[:, 0] * 100,  # Convert to cm
            self.spectrometer.output_beam[:, 2] * 100,  # Convert to cm
            c=output_energies,
            s=20,
            cmap='plasma',
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Proton Energy [MeV]')
        
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
        Plot a simple histogram of proton counts vs horizontal position.
        
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
            alpha=0.7,
            color='steelblue',
            edgecolor='black',
            linewidth=0.5
        )
        
        ax.set_xlabel('Horizontal Position [cm]')
        ax.set_ylabel('Counts')
        ax.set_title('Proton Counts vs Position')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_pos = np.mean(x_positions)
        std_pos = np.std(x_positions)
        ax.axvline(mean_pos, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pos:.4f} m')
        ax.legend()
        
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
        ax.vlines(0, -foil_radius, foil_radius, color='blue', linewidth=3, label='Conversion Foil')
        
        # Aperture (vertical line at aperture distance)
        ax.vlines(aperture_distance, -aperture_radius, aperture_radius, 
                color='red', linewidth=3, label='Aperture')
        
        # Draw sample of input rays
        num_rays_to_plot = min(len(self.spectrometer.input_beam), 200)  # Limit for clarity
        z_coords = np.linspace(0, aperture_distance, 20)
        
        for i in range(0, len(self.spectrometer.input_beam), max(1, len(self.spectrometer.input_beam) // num_rays_to_plot)):
            ray = self.spectrometer.input_beam[i]
            x0, angle_x, y0, angle_y = ray[:4]
            x0 *= 100 # cm
            y0 *= 100 # cm
            
            # Calculate ray trajectory (assuming small angles)
            slope = np.tan(angle_x)
            x_trajectory = slope * z_coords + x0
            
            # Only plot rays that stay within reasonable bounds
            if np.all(np.abs(x_trajectory) < 2 * max(foil_radius, aperture_radius)):
                ax.plot(z_coords, x_trajectory, alpha=0.4, color='green', linewidth=0.5)
        
        ax.set_xlabel('Z Distance [cm]')
        ax.set_ylabel('X Position [cm]')
        ax.set_title('Input Ray Geometry (X-Z Projection)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        ax.set_xlim(-0.1 * aperture_distance, 1.1 * aperture_distance)
        max_extent = 1.5 * max(foil_radius, aperture_radius)
        ax.set_ylim(-max_extent, max_extent)
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Input ray geometry plot saved to {filename}')

    def plot_proton_density_heatmap(
        self, 
        filename: Optional[str] = None,
        dx: float = 0.005, 
        dy: float = 0.005
    ) -> None:
        """
        Plot a heatmap of proton density in the focal plane.
        
        Args:
            filename: Output filename for the plot
            dx: X-direction resolution in meters
            dy: Y-direction resolution in meters
        """
        if filename == None:
            filename = f'{self.spectrometer.figure_directory}/proton_density_heatmap.png'
        
        density, X_mesh, Y_mesh = self.spectrometer.get_proton_density_map(dx, dy)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.pcolormesh(X_mesh*100, Y_mesh*100, np.log10(density), cmap='plasma', shading='auto')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('log$_10$(Proton Density [protons/cm$^2$-source])')
        
        ax.set_xlabel('X Position [cm]')
        ax.set_ylabel('Y Position [cm]')
        ax.set_title('Proton Density in Focal Plane')
        ax.set_aspect('equal')
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Proton density heatmap saved to {filename}')
    
    def _plot_monoenergetic_analysis(
        self, 
        filename: str, 
        neutron_energy: float, 
        mean_pos: float, 
        std_dev: float
    ) -> None:
        """Generate analysis plots for monoenergetic performance."""
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
            cmap='plasma',
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
    
    def _plot_performance(
        self, 
        filename: str, 
        energies: np.ndarray, 
        positions: np.ndarray, 
        position_uncertainties: np.ndarray,
        energy_resolutions: np.ndarray,
        total_efficiencies: np.ndarray
    ) -> None:
        """Generate comprehensive performance plot with shared x-axis."""
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        fig.suptitle('Comprehensive Performance', fontsize=16)
        
        # Left y-axis: Dispersion
        color_dispersion = 'tab:blue'
        ax1.set_xlabel('Neutron Energy [MeV]')
        ax1.set_ylabel('Proton Position [cm]', color=color_dispersion)
        
        # Plot dispersion curve
        ax1.plot(energies, positions * 100, color=color_dispersion, linewidth=2,
                 label=f'Dispersion')
        ax1.fill_between(energies, (positions - position_uncertainties) * 100, 
                        (positions + position_uncertainties) * 100,
                        alpha=0.3, color=color_dispersion)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor=color_dispersion)
        
        # Right y-axis: Resolution and Efficiency
        ax2 = ax1.twinx()
        
        # Plot energy resolution
        color_resolution = 'tab:red'
        ax2.plot(energies, energy_resolutions * 1000, color=color_resolution, 
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
        labelLines(ax2.get_lines(), xvals=[energies.min() + 0.5 * range], align=True, fontsize=12)
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
        print(f'  Average resolution: {np.mean(energy_resolutions)*1000:.1f} keV')
        print(f'  Average efficiency: {np.mean(total_efficiencies):.3e}')
        print(f'  Best resolution: {np.min(energy_resolutions)*1000:.1f} keV at {energies[np.argmin(energy_resolutions)]:.2f} MeV')
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
        if figure_directory is None:
            figure_directory = self.spectrometer.figure_directory
        if filename_prefix is None:
            filename_prefix = f'{figure_directory}/foil_{self.spectrometer.conversion_foil.particle}'
        else:
            filename_prefix = f'{figure_directory}/{filename_prefix}'
        
        # ========== Plot 1: Differential Cross Section vs Lab Angle ==========
        fig, ax = plt.subplots(figsize=(5, 4))
        
        cos_angles = np.linspace(np.cos(angle_range[0]), np.cos(angle_range[1]), num_angles)
        angles_rad = np.arccos(cos_angles)
        angles_deg = np.degrees(angles_rad)
        
        diff_xs_lab = self.spectrometer.conversion_foil.calculate_differential_xs_lab(angles_rad, energy_MeV)
        
        ax.plot(angles_deg, diff_xs_lab * 1e28, 'b-', linewidth=2)
        ax.set_xlabel('Angle [deg]')
        ax.set_ylabel('d$\sigma$/d$\Omega$ [barns/sr]')
        ax.set_title(f'Differential Cross Section - {self.spectrometer.conversion_foil.particle.capitalize()} at {energy_MeV:.1f} MeV')
        ax.grid(True, alpha=0.3)
        
        filename = f'{filename_prefix}_E{energy_MeV:.1f}MeV_differential_xs.png'
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Differential cross section plot saved to {filename}')
        
        # ========== Plot 2: Cross Sections vs Energy ==========
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Use raw data from files (no interpolation)
        # n-hydron cross section data
        nh_energies_eV = self.spectrometer.conversion_foil.nh_cross_section_data[0]
        nh_energies_MeV = nh_energies_eV * 1e-6  # Convert eV to MeV
        nh_xs_barns = self.spectrometer.conversion_foil.nh_cross_section_data[1]  # Already in barns
        
        # n-C12 cross section data
        nc12_energies_eV = self.spectrometer.conversion_foil.nc12_cross_section_data[0]
        nc12_idx = (nc12_energies_eV >= np.min(nh_energies_eV)) & (nc12_energies_eV <= np.max(nh_energies_eV))
        nc12_energies_MeV = nc12_energies_eV[nc12_idx] * 1e-6  # Convert eV to MeV
        nc12_xs_barns = self.spectrometer.conversion_foil.nc12_cross_section_data[1, nc12_idx]  # Already in barns
        
        ax.plot(nh_energies_MeV, nh_xs_barns, 'r-', linewidth=2, 
                label=f'n-{self.spectrometer.conversion_foil.particle[0]} elastic')
        ax.plot(nc12_energies_MeV, nc12_xs_barns, 'g-', linewidth=2, 
                label='n-C12 elastic')
        ax.axvline(energy_MeV, color='k', linestyle='--', alpha=0.7, 
                    label=f'Current energy: {energy_MeV:.1f} MeV')
        
        ax.set_xlabel('Neutron Energy [MeV]')
        ax.set_ylabel('Cross Section [barns]')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        
        filename = f'{filename_prefix}_cross_sections.png'
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Cross sections plot saved to {filename}')
        
        # ========== Plot 3: Stopping Power vs Energy ==========
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Use raw SRIM data (no interpolation)
        srim_energies_MeV = self.spectrometer.conversion_foil.srim_data[0]  # Already in MeV
        srim_stopping_power = self.spectrometer.conversion_foil.srim_data[1] + self.spectrometer.conversion_foil.srim_data[2]  # Electronic + nuclear stopping
        
        ax.plot(srim_energies_MeV, srim_stopping_power, 'purple', linewidth=2)
        
        ax.set_title(f'{self.spectrometer.conversion_foil.particle.capitalize()} in {self.spectrometer.conversion_foil.foil_material}')
        ax.set_xlabel(f'{self.spectrometer.conversion_foil.particle.capitalize()} Energy [MeV]')
        ax.set_ylabel('Stopping Power [MeV/mm]')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        filename = f'{filename_prefix}_stopping_power.png'
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Stopping power plot saved to {filename}')