from typing import Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from .conversion_foil import ConversionFoil
from .hodoscope import Hodoscope
from .spectrometer import MPRSpectrometer


class DualFoilSpectrometer:
    """
    Dual-foil Magnetic Proton Recoil (MPR) spectrometer system.
    
    Manages two independent spectrometers with CH2 (protons, positive y) 
    and CD2 (deuterons, negative y) conversion foils for simultaneous 
    proton and deuteron spectroscopy.
    """    
    def __init__(
        self,
        foil_radius: float,
        thickness_ch2: float,
        thickness_cd2: float,
        aperture_distance: float,
        aperture_radius: float,
        transfer_map_path: str,
        reference_energy: float,
        ch2_min_energy: float,
        ch2_max_energy: float,
        cd2_min_energy: float,
        cd2_max_energy: float,
        hodoscope: Hodoscope,
        aperture_width: Optional[float] = None,
        aperture_height: Optional[float] = None,
        figure_directory: str = '.',
        aperture_type: Literal['circ', 'rect'] = 'circ',
        **shared_foil_kwargs
    ):
        """
        Initialize dual-foil spectrometer system.
        
        Args:
            foil_radius: Foil radius in cm (same for both foils)
            thickness_ch2: CH2 foil thickness in μm
            thickness_cd2: CD2 foil thickness in μm
            aperture_distance: Distance from foil to aperture in cm
            aperture_radius: Aperture radius in cm (for circular)
            transfer_map_path: Path to COSY transfer map file
            reference_energy: Reference energy in MeV
            ch2_min_energy: Minimum acceptance energy in MeV for CH2 foil
            ch2_max_energy: Maximum acceptance energy in MeV for CH2 foil
            cd2_min_energy: Minimum acceptance energy in MeV for CD2 foil
            cd2_max_energy: Maximum acceptance energy in MeV for CD2 foil
            hodoscope: Hodoscope detector system
            aperture_width: Aperture width in cm (for rectangular)
            aperture_height: Aperture height in cm (for rectangular)
            figure_directory: Directory for saving figures
            aperture_type: Type of aperture ('circ' or 'rect')
            **shared_foil_kwargs: Additional arguments passed to both ConversionFoil instances
        """
        print('='*70)
        print('Initializing Dual-Foil MPR Spectrometer...')
        print('='*70)
        
        self.figure_directory = figure_directory
        self.reference_energy = reference_energy
        self.ch2_min_energy = ch2_min_energy
        self.ch2_max_energy = ch2_max_energy
        self.cd2_min_energy = cd2_min_energy
        self.cd2_max_energy = cd2_max_energy
        
        # Create CH2 foil and spectrometer (positive y half)
        print('\n--- Initializing CH2 (Proton) Spectrometer ---')
        foil_ch2 = ConversionFoil(
            foil_radius=foil_radius,
            thickness=thickness_ch2,
            aperture_distance=aperture_distance,
            aperture_radius=aperture_radius,
            aperture_width=aperture_width,
            aperture_height=aperture_height,
            foil_material='CH2',
            aperture_type=aperture_type,
            **shared_foil_kwargs
        )
        
        self.spec_ch2 = MPRSpectrometer(
            conversion_foil=foil_ch2,
            transfer_map_path=transfer_map_path,
            reference_energy=reference_energy,
            min_energy=ch2_min_energy,
            max_energy=ch2_max_energy,
            hodoscope=hodoscope,
            figure_directory=figure_directory
        )
        
        # Create CD2 foil and spectrometer (negative y half)
        print('\n--- Initializing CD2 (Deuteron) Spectrometer ---')
        foil_cd2 = ConversionFoil(
            foil_radius=foil_radius,
            thickness=thickness_cd2,
            aperture_distance=aperture_distance,
            aperture_radius=aperture_radius,
            aperture_width=aperture_width,
            aperture_height=aperture_height,
            foil_material='CD2',
            aperture_type=aperture_type,
            **shared_foil_kwargs
        )
        
        self.spec_cd2 = MPRSpectrometer(
            conversion_foil=foil_cd2,
            transfer_map_path=transfer_map_path,
            reference_energy=reference_energy,
            min_energy=cd2_min_energy,
            max_energy=cd2_max_energy,
            hodoscope=hodoscope,
            figure_directory=figure_directory
        )
        
        # Combined beam storage
        self.combined_input_beam = np.zeros(0)
        self.combined_output_beam = np.zeros(0)
        
        # Configure CH2 plotter for dual-foil mode
        # This tells it to overlay CD2 data when plotting
        self.spec_ch2.plotter.set_dual_data(
            spectrometer_secondary=self.spec_cd2,
            primary_label='Protons (CH2)',
            secondary_label='Deuterons (CD2)',
            primary_color='blue',
            secondary_color='red'
        )
        
        # Use CH2's plotter for all combined plots
        self.plotter = self.spec_ch2.plotter
        
        print('\n' + '='*70)
        print('Dual-Foil MPR Spectrometer initialization complete!')
        print('='*70 + '\n')
    
    def generate_monte_carlo_rays(
        self,
        neutron_energies: np.ndarray,
        energy_distribution: np.ndarray,
        num_hydrons: int,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        z_sampling: Literal['exp', 'uni'] = 'exp',
        save_beam: bool = True,
        max_workers: Optional[int] = None
    ) -> None:
        """
        Generate hydron rays for both foils with y-restrictions.
        
        Args:
            neutron_energies: Array of neutron energies in MeV
            energy_distribution: Relative probability distribution
            num_hydrons: Total number of hydrons to simulate
            include_kinematics: Include kinematic energy transfer
            include_stopping_power_loss: Include SRIM energy loss
            z_sampling: Depth sampling method ('exp' or 'uni')
            save_beam: Whether to save beams to CSV
            max_workers: Maximum number of worker processes
        """
        # Split hydrons between foils based on energy_distribution
        ch2_idx = (neutron_energies >= self.ch2_min_energy) & (neutron_energies <= self.ch2_max_energy)
        ch2_fraction = np.sum(energy_distribution[ch2_idx]) / np.sum(energy_distribution)
        num_ch2 = int(num_hydrons * ch2_fraction)
        num_cd2 = num_hydrons - num_ch2
        
        
        print(f'\nGenerating {num_ch2} CH2 (proton) rays with positive y restriction...')
        self.spec_ch2.generate_monte_carlo_rays(
            neutron_energies=neutron_energies,
            energy_distribution=energy_distribution,
            num_hydrons=num_ch2,
            include_kinematics=include_kinematics,
            include_stopping_power_loss=include_stopping_power_loss,
            z_sampling=z_sampling,
            save_beam=False,
            max_workers=max_workers,
            y_restriction='positive'
        )
        
        print(f'\nGenerating {num_cd2} CD2 (deuteron) rays with negative y restriction...')
        self.spec_cd2.generate_monte_carlo_rays(
            neutron_energies=neutron_energies,
            energy_distribution=energy_distribution,
            num_hydrons=num_cd2,
            include_kinematics=include_kinematics,
            include_stopping_power_loss=include_stopping_power_loss,
            z_sampling=z_sampling,
            save_beam=False,
            max_workers=max_workers,
            y_restriction='negative',
        )
        
        # Combine beams
        self._combine_input_beams()
        
        if save_beam:
            self._save_combined_input_beam()
    
    def apply_transfer_map(
        self,
        map_order: int = 5,
        save_beam: bool = True,
        max_workers: Optional[int] = None
    ) -> None:
        """
        Apply ion optical transfer map to both spectrometers.
        
        Args:
            map_order: Order of transfer map to apply
            save_beam: Whether to save output beams to CSV
            max_workers: Maximum number of worker processes
        """
        print('\nApplying transfer map to CH2 (proton) beam...')
        self.spec_ch2.apply_transfer_map(
            map_order=map_order,
            save_beam=False,
            max_workers=max_workers
        )
        
        print('\nApplying transfer map to CD2 (deuteron) beam...')
        self.spec_cd2.apply_transfer_map(
            map_order=map_order,
            save_beam=False,
            max_workers=max_workers
        )
        
        # Combine beams
        self._combine_output_beams()
        
        if save_beam:
            self._save_combined_output_beam()
    
    def _combine_input_beams(self) -> None:
        """Combine input beams from both foils with particle type marker."""
        # Add particle type column: 1 = proton (CH2), 2 = deuteron (CD2)
        ch2_beam = np.hstack([
            self.spec_ch2.input_beam,
            np.ones((len(self.spec_ch2.input_beam), 1))
        ])
        
        cd2_beam = np.hstack([
            self.spec_cd2.input_beam,
            2 * np.ones((len(self.spec_cd2.input_beam), 1))
        ])
        
        self.combined_input_beam = np.vstack([ch2_beam, cd2_beam])
        print(f'\nCombined input beam: {len(ch2_beam)} protons + {len(cd2_beam)} deuterons = {len(self.combined_input_beam)} total')
    
    def _combine_output_beams(self) -> None:
        """Combine output beams from both foils with particle type marker."""
        # Add particle type column: 1 = proton (CH2), 2 = deuteron (CD2)
        ch2_beam = np.hstack([
            self.spec_ch2.output_beam,
            np.ones((len(self.spec_ch2.output_beam), 1))
        ])
        
        cd2_beam = np.hstack([
            self.spec_cd2.output_beam,
            2 * np.ones((len(self.spec_cd2.output_beam), 1))
        ])
        
        self.combined_output_beam = np.vstack([ch2_beam, cd2_beam])
        print(f'\nCombined output beam: {len(ch2_beam)} protons + {len(cd2_beam)} deuterons = {len(self.combined_output_beam)} total')
        
        
    def calculate_physical_separation(self) -> dict:
        """
        Calculate the physical separation statistics for the dual-foil system.
        
        Analyzes how many hydrons from each foil half end up crossing the y=0 line
        at the detector plane.
        
        Returns:
            Dictionary containing separation statistics
        """
        if len(self.combined_output_beam) == 0:
            raise ValueError("No output beam data available. Run apply_transfer_map() first.")
        
        # Extract y-positions at detector
        y_proton = self.spec_ch2.output_beam[:, 2]  # meters
        y_deuteron = self.spec_cd2.output_beam[:, 2]  # meters
        
        # Count crossovers
        protons_total = len(y_proton)
        deuterons_total = len(y_deuteron)
        
        # Protons that crossed to positive y (should be negative)
        protons_crossed = np.sum(y_proton > 0)
        protons_stayed = np.sum(y_proton <= 0)
        
        # Deuterons that crossed to negative y (should be positive)
        deuterons_crossed = np.sum(y_deuteron < 0)
        deuterons_stayed = np.sum(y_deuteron >= 0)
        
        # Calculate percentages
        proton_separation_pct = (protons_stayed / protons_total * 100) if protons_total > 0 else 0
        deuteron_separation_pct = (deuterons_stayed / deuterons_total * 100) if deuterons_total > 0 else 0
        overall_separation_pct = ((protons_stayed + deuterons_stayed) / 
                                 (protons_total + deuterons_total) * 100) if (protons_total + deuterons_total) > 0 else 0
        
        results = {
            'protons_total': protons_total,
            'protons_stayed_positive': protons_stayed,
            'protons_crossed_to_negative': protons_crossed,
            'proton_separation_percentage': proton_separation_pct,
            
            'deuterons_total': deuterons_total,
            'deuterons_stayed_negative': deuterons_stayed,
            'deuterons_crossed_to_positive': deuterons_crossed,
            'deuteron_separation_percentage': deuteron_separation_pct,
            
            'total_hydrons': protons_total + deuterons_total,
            'total_stayed_separated': protons_stayed + deuterons_stayed,
            'total_crossed': protons_crossed + deuterons_crossed,
            'overall_separation_percentage': overall_separation_pct
        }
        
        return results
    
    def _save_combined_input_beam(self, filepath: Optional[str] = None) -> None:
        """Save combined input beam to CSV."""
        if filepath is None:
            filepath = f'{self.figure_directory}/combined_input_beam.csv'
        
        df = pd.DataFrame({
            'x0': self.combined_input_beam[:, 0],
            'angle_x': self.combined_input_beam[:, 1],
            'y0': self.combined_input_beam[:, 2],
            'angle_y': self.combined_input_beam[:, 3],
            'energy_relative': self.combined_input_beam[:, 4],
            'neutron_energy': self.combined_input_beam[:, 5],
            'particle_type': self.combined_input_beam[:, 6].astype(int)  # 1=proton, 2=deuteron
        })
        df.to_csv(filepath, index=False)
        print(f'Combined input beam saved to {filepath}')
    
    def _save_combined_output_beam(self, filepath: Optional[str] = None) -> None:
        """Save combined output beam to CSV."""
        if filepath is None:
            filepath = f'{self.figure_directory}/combined_output_beam.csv'
        
        df = pd.DataFrame({
            'x0': self.combined_output_beam[:, 0],
            'angle_x': self.combined_output_beam[:, 1],
            'y0': self.combined_output_beam[:, 2],
            'angle_y': self.combined_output_beam[:, 3],
            'energy_relative': self.combined_output_beam[:, 4],
            'particle_type': self.combined_output_beam[:, 5].astype(int)  # 1=proton, 2=deuteron
        })
        df.to_csv(filepath, index=False)
        print(f'Combined output beam saved to {filepath}')
    
    def read_beams(
        self,
        combined_input_path: Optional[str] = None,
        combined_output_path: Optional[str] = None
    ) -> None:
        """
        Read combined beams from file and split into individual spectrometers.
        
        Args:
            combined_input_path: Path to combined input beam CSV
            combined_output_path: Path to combined output beam CSV
        """
        # Read combined beams
        if combined_input_path is None:
            combined_input_path = f'{self.figure_directory}/combined_input_beam.csv'
        if combined_output_path is None:
            combined_output_path = f'{self.figure_directory}/combined_output_beam.csv'
        
        input_df = pd.read_csv(combined_input_path)
        output_df = pd.read_csv(combined_output_path)
        
        self.combined_input_beam = input_df.to_numpy()
        self.combined_output_beam = output_df.to_numpy()
        
        # Split by particle type: protons = 1, deuterons = 2
        proton_mask_in = self.combined_input_beam[:, 6] == 1
        deuteron_mask_in = self.combined_input_beam[:, 6] == 2
        
        self.spec_ch2.input_beam = self.combined_input_beam[proton_mask_in, :6]
        self.spec_cd2.input_beam = self.combined_input_beam[deuteron_mask_in, :6]
        
        proton_mask_out = self.combined_output_beam[:, 5] == 1
        deuteron_mask_out = self.combined_output_beam[:, 5] == 2
        
        self.spec_ch2.output_beam = self.combined_output_beam[proton_mask_out, :5]
        self.spec_cd2.output_beam = self.combined_output_beam[deuteron_mask_out, :5]
        
        print(f'Read {len(self.spec_ch2.input_beam)} protons and {len(self.spec_cd2.input_beam)} deuterons from combined beams')
    
    def plot_focal_plane_distribution(
        self,
        include_hodoscope: bool = False,
        **kwargs
    ) -> None:
        """
        Plot focal plane distribution with both particles.
        Delegates to SpectrometerPlotter which handles dual-foil mode.
        """
        self.plotter.plot_focal_plane_distribution(
            include_hodoscope=include_hodoscope, 
            **kwargs
        )
    
    def plot_input_ray_geometry(self, **kwargs) -> None:
        """Plot input ray geometry showing both foils."""
        # This needs custom implementation since it shows foil geometry
        self.plotter.plot_input_ray_geometry(**kwargs)
        self._plot_combined_input_geometry()
    
    def plot_phase_space(self, **kwargs) -> None:
        """Plot phase space with both particles."""
        self.plotter.plot_phase_space(**kwargs)
    
    def plot_synthetic_neutron_histogram(self, **kwargs) -> None:
        """Plot neutron histogram with both particles."""
        self.plotter.plot_synthetic_neutron_histogram(**kwargs)
    
    def plot_simple_position_histogram(self, **kwargs) -> None:
        """Plot position histogram with both particles."""
        self.plotter.plot_simple_position_histogram(**kwargs)
    
    def plot_hydron_density_heatmap(self, **kwargs) -> None:
        """Plot density heatmap with both particles."""
        self.plotter.plot_hydron_density_heatmap(**kwargs)
    
    def _plot_combined_input_geometry(self, filename: Optional[str] = None) -> None:
        """Plot combined input geometry showing y-restriction."""
        if filename is None:
            filename = f'{self.figure_directory}/combined_input_geometry.png'
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # CH2 (positive y)
        x_ch2 = self.spec_ch2.input_beam[:, 0] * 100
        y_ch2 = self.spec_ch2.input_beam[:, 2] * 100
        ax.scatter(x_ch2, y_ch2, alpha=0.5, s=5, label='CH2 (Protons)', color='blue')
        
        # CD2 (negative y)
        x_cd2 = self.spec_cd2.input_beam[:, 0] * 100
        y_cd2 = self.spec_cd2.input_beam[:, 2] * 100
        ax.scatter(x_cd2, y_cd2, alpha=0.5, s=5, label='CD2 (Deuterons)', color='red')
        
        # Draw foil boundary
        theta = np.linspace(0, 2*np.pi, 100)
        foil_r = self.spec_ch2.conversion_foil.foil_radius_cm
        ax.plot(foil_r * np.cos(theta), foil_r * np.sin(theta), 'k-', 
               linewidth=2, label='Foil boundary')
        
        # Draw y=0 dividing line
        ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Y=0 divider')
        
        # Add shaded regions to show foil halves
        from matplotlib.patches import Wedge
        wedge_upper = Wedge((0, 0), foil_r, 0, 180, facecolor='blue', alpha=0.1, 
                           edgecolor='none')
        wedge_lower = Wedge((0, 0), foil_r, 180, 360, facecolor='red', alpha=0.1, 
                           edgecolor='none')
        ax.add_patch(wedge_upper)
        ax.add_patch(wedge_lower)
        
        ax.set_xlabel('X Position [cm]')
        ax.set_ylabel('Y Position [cm]')
        ax.set_title('Dual-Foil Input Ray Geometry')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Combined input geometry plot saved to {filename}')
    
    def plot_separation_analysis(self, filename: Optional[str] = None) -> None:
        """
        Plot detailed separation analysis showing crossover statistics.
        """
        if filename is None:
            filename = f'{self.figure_directory}/separation_analysis.png'
        
        # Get separation statistics
        sep_stats = self.calculate_physical_separation()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left plot: Y-position histograms
        y_proton = self.spec_ch2.output_beam[:, 2] * 100  # cm
        y_deuteron = self.spec_cd2.output_beam[:, 2] * 100  # cm
        
        bins = np.linspace(min(y_proton.min(), y_deuteron.min()), 
                          max(y_proton.max(), y_deuteron.max()), 50)
        
        axes[0].hist(y_proton, bins=bins, alpha=0.6, label='Protons (CH2)', 
                    color='blue', edgecolor='black', linewidth=0.5, density=True)
        axes[0].hist(y_deuteron, bins=bins, alpha=0.6, label='Deuterons (CD2)', 
                    color='red', edgecolor='black', linewidth=0.5, density=True)
        axes[0].axvline(0, color='black', linestyle='--', linewidth=2, 
                       label='Y=0 divider', alpha=0.7)
        
        # Add shaded regions for crossovers
        axes[0].axvspan(0, bins[-1], alpha=0.1, color='red')
        axes[0].axvspan(bins[0], 0, alpha=0.1, color='blue')
        
        # Set x limits
        axes[0].set_xlim(bins[0], bins[-1])
        
        axes[0].set_xlabel('Y Position [cm]')
        axes[0].set_ylabel('Probability Density')
        axes[0].set_title('Y-Position Distribution at Detector')
        axes[0].legend()
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
                           color='green', alpha=0.7, edgecolor='black')
        bars2 = axes[1].bar(x + width/2, crossed, width, label='Crossed midline', 
                           color='orange', alpha=0.7, edgecolor='black')
        
        # Add percentage labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9)
        
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_title('Physical Separation Statistics')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].legend()
        axes[1].set_ylim(0, 105)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Separation analysis plot saved to {filename}')