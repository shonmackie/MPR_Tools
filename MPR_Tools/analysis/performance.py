"""Performance analysis methods for MPR spectrometer."""

from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING
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
        num_hydrons: int = 10000,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        generate_figure: bool = False,
        figure_name: Optional[str] = None,
        verbose: bool = False
    ) -> Tuple[float, float, float, float]:
        """
        Analyze spectrometer performance for monoenergetic neutrons.
        
        Args:
            neutron_energy: Neutron energy in MeV
            num_hydrons: Number of hydrons to simulate
            include_kinematics: Include kinematic energy transfer
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            generate_figure: Whether to generate analysis plots
            figure_name: Name for output figure
            verbose: Print detailed results
            
        Returns:
            Tuple of (mean_position, std_deviation, fwhm, energy_resolution)
        """
        print(f'Analyzing performance for {neutron_energy:.3f} MeV monoenergetic neutrons...')
        
        # Generate and transport hydrons
        self.spectrometer.generate_monte_carlo_rays(
            np.array([neutron_energy]), 
            np.array([1.0]), 
            num_hydrons, 
            include_kinematics, 
            include_stopping_power_loss,
            save_beam=False
        )
        self.spectrometer.apply_transfer_map(map_order=5, save_beam=False)
        
        # Analyze focal plane distribution
        x_positions = self.spectrometer.output_beam[:, 0]
        #TODO - consider more sophisticated method for evaluating resolution
        mean_position, std_deviation = norm.fit(x_positions)
        fwhm = 2.355 * std_deviation
        
        # Calculate energy resolution
        dispersion = self.spectrometer.transfer_map[0, 5]  # Assuming this is the dispersion term
        energy_resolution = self.spectrometer.reference_energy / (dispersion / fwhm) if fwhm > 0 else 0
        
        # Generate figure if requested
        if generate_figure:
            if figure_name == None:
                figure_name = (
                    f'{self.spectrometer.figure_directory}/Monoenergetic_En{neutron_energy:.1f}MeV_T{self.spectrometer.conversion_foil.thickness_um:.0f}um_E0{self.spectrometer.reference_energy:.1f}MeV.png'
                )
            
            self.spectrometer._plot_monoenergetic_analysis(figure_name, neutron_energy, mean_position, std_deviation)
        
        if verbose:
            print('Ion Optical Image Parameters:')
            print(f'  Mean position [cm]: {mean_position * 100:.3f}')
            print(f'  Standard deviation [cm]: {std_deviation * 100:.3f}')
            print(f'  FWHM [cm]: {fwhm * 100:.3f}')
            print(f'  Energy resolution [keV]: {energy_resolution * 1000:.2f}')
        
        return mean_position, std_deviation, fwhm, energy_resolution
    
    def generate_performance_curve(
        self,
        num_energies: int = 40,
        num_hydrons_per_energy: int = 1000,
        num_efficiency_samples: int = int(1e6),
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        output_filename: Optional[str] = None,
        generate_figure: bool = True,
        reset: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate comprehensive performance analysis including dispersion, resolution, and efficiency.
        
        Args:
            num_energies: Number of energy points to simulate
            num_hydrons_per_energy: Number of hydrons per energy point for dispersion/resolution
            num_efficiency_samples: Number of samples for efficiency calculation
            include_kinematics: Include kinematic effects
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            output_filename: Name for output data file
            generate_figure: Whether to generate comprehensive plot
            reset: Whether to regenerate the dataset or load an existing one
            
        Returns:
            Tuple of (energies, positions_mean, positions_std, energy_resolutions, total_efficiencies)
        """
        print('Generating comprehensive performance analysis...')
        
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
                # Calculate dispersion and resolution from monoenergetic analysis
                mean_pos, std_dev, fwhm, energy_res = self.analyze_monoenergetic_performance(
                    energy, 
                    num_hydrons_per_energy, 
                    include_kinematics, 
                    include_stopping_power_loss,
                    generate_figure=False,
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
        
        # Generate performance figure if requested
        if generate_figure:
            figure_name = output_filename.replace('.csv', '.png')
            self.spectrometer._plot_performance(
                figure_name, 
                energies, positions_mean, positions_std, 
                energy_resolutions, total_efficiencies
            )
        
        return energies, positions_mean, positions_std, energy_resolutions, total_efficiencies