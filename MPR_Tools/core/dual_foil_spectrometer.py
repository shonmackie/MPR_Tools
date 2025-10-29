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
        cd2_idx = (neutron_energies >= self.cd2_min_energy) & (neutron_energies <= self.cd2_max_energy)
        ch2_fraction = np.sum(energy_distribution[ch2_idx]) / (np.sum(energy_distribution[ch2_idx]) + np.sum(energy_distribution[cd2_idx]))
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