"""File reading utilities for MPR data."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple
import pandas as pd

if TYPE_CHECKING:
    from ..core.spectrometer import MPRSpectrometer
    import numpy as np

class BeamIO:
    """Handles data export functionality."""
    
    def __init__(self, spectrometer: MPRSpectrometer) -> None:
        self.spectrometer = spectrometer
    
    def save_input_beam(self, filepath: Optional[str] = None) -> None:
        """Save input beam to CSV file."""
        if filepath is None:
            filepath = f'{self.spectrometer.figure_directory}/input_beam.csv'
        
        df = pd.DataFrame({
            'x0': self.spectrometer.input_beam[:, 0],
            'angle_x': self.spectrometer.input_beam[:, 1],
            'y0': self.spectrometer.input_beam[:, 2],
            'angle_y': self.spectrometer.input_beam[:, 3],
            'energy_relative': self.spectrometer.input_beam[:, 4],
            'neutron_energy': self.spectrometer.input_beam[:, 5]
        })
        df.to_csv(filepath, index=False)
        print(f'Input beam saved to {filepath}')
    
    def save_output_beam(self, filepath: Optional[str] = None) -> None:
        """Save output beam to CSV file."""
        if filepath is None:
            filepath = f'{self.spectrometer.figure_directory}/output_beam.csv'
        
        df = pd.DataFrame({
            'x0': self.spectrometer.output_beam[:, 0],
            'angle_x': self.spectrometer.output_beam[:, 1],
            'y0': self.spectrometer.output_beam[:, 2],
            'angle_y': self.spectrometer.output_beam[:, 3],
            'energy_relative': self.spectrometer.output_beam[:, 4]
        })
        df.to_csv(filepath, index=False)
        print(f'Output beam saved to {filepath}')
        
    def read_beams(
        self,
        input_beam_path: Optional[str] = None,
        output_beam_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read input and output beams from file
        
        Args:
            input_beam_path: Input beam path location 
            output_beam_path: Output beam path location 
        """
        # Read input beam
        if input_beam_path == None:
            input_beam_path = f'{self.spectrometer.figure_directory}/input_beam.csv'
            
        input_beam_df = pd.read_csv(input_beam_path)
        self.input_beam = input_beam_df.to_numpy()
        
        # Read output beam
        if output_beam_path == None:
            output_beam_path = f'{self.spectrometer.figure_directory}/output_beam.csv'
            
        output_beam_df = pd.read_csv(output_beam_path)
        self.output_beam = output_beam_df.to_numpy()
        
        return self.input_beam, self.output_beam