"""Parameter sweep analysis for MPR spectrometer components."""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import itertools
from copy import deepcopy

if TYPE_CHECKING:
    from ..core.spectrometer import MPRSpectrometer, ConversionFoil

class FoilSweeper:
    """
    Parameter sweep analysis for conversion foil optimization.
    
    Sweeps over foil parameters (radius, thickness, aperture distance, aperture radius)
    and evaluates performance metrics for a given neutron energy.
    """
    
    def __init__(self, spectrometer: MPRSpectrometer) -> None:
        """
        Initialize FoilSweeper.
        
        Args:
            spectrometer: MPRSpectrometer instance to use for analysis
        """
        self.spectrometer = spectrometer
        self.results_df: Optional[pd.DataFrame] = None
        self.sweep_parameters: Dict[str, np.ndarray] = {}
        
    def setup_parameter_sweep(
        self,
        foil_radius_values: np.ndarray,
        thickness_values: np.ndarray,
        aperture_distance_values: np.ndarray,
        aperture_radius_values: np.ndarray
    ) -> None:
        """
        Set up parameter ranges for sweep.
        
        Args:
            foil_radius_values: Sweep values for foil radius
            thickness_values: Sweep values for foil thickness  
            aperture_distance_values: Sweep values for aperture distance
            aperture_radius_values: Sweep values for aperture radius
        """
        self.sweep_parameters = {}
        
        self.sweep_parameters['foil_radius'] = foil_radius_values
        self.sweep_parameters['thickness'] = thickness_values
        self.sweep_parameters['aperture_distance'] = aperture_distance_values
        self.sweep_parameters['aperture_radius'] = aperture_radius_values
        
        combinations_list = itertools.product(*self.sweep_parameters.values())
        self.combinations = pd.DataFrame(combinations_list, columns=self.sweep_parameters.keys())
        
        print(f"Parameter sweep configured with {len(self.combinations)} combinations:")
    
    def run_sweep(
        self,
        neutron_energy: float,
        num_hydrons: int = 10000,
        num_neutrons: int = 100000,
        output_filename: Optional[str] = None,
        reset: bool = True
    ) -> pd.DataFrame:
        """
        Run parameter sweep analysis in serial (since Monte Carlo generation is already parallelized).
        
        Args:
            neutron_energy: Neutron energy to analyze in MeV
            num_hydrons: Number of hydrons for performance analysis
            num_neutrons: Number of neutrons for efficiency calculation
            output_filename: Output CSV filename
            reset: Whether to recalculate or load existing results
            
        Returns:
            DataFrame with sweep results
        """
        if output_filename is None:
            output_filename = (f'{self.spectrometer.figure_directory}/foil_sweep_En{neutron_energy:.1f}MeV.csv')
        
        # Check if results exist and reset=False
        if not reset and Path(output_filename).exists():
            print(f"Loading existing sweep results from {output_filename}")
            self.results_df = pd.read_csv(output_filename)
            return self.results_df
        
        # Initialize results with combinations df and add empty columns
        all_results = [{}] * len(self.combinations)
        
        # Run sweep serially with progress bar
        for i, (_, params) in enumerate(tqdm(self.combinations.iterrows(), desc='Parameter sweep progress')):
            
            result = self._evaluate_parameter_combination(params.to_dict(), neutron_energy, num_hydrons, num_neutrons)
            
            if result is not None:
                all_results[i] = result
        
        # Raise error if no results
        if len(all_results) == 0:
            raise ValueError("No results found for any parameter combination.")
        
        # Save results
        self.results_df = pd.DataFrame(all_results)
        self.results_df.to_csv(output_filename, index=False)
        print(f"Sweep results saved to {output_filename}")
        
        return self.results_df
    
    def _evaluate_parameter_combination(
        self,
        params: Dict[str, float],
        neutron_energy: float,
        num_hydrons: int,
        num_neutrons: int
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single parameter combination.
        
        Args:
            params: Dictionary of parameter values
            neutron_energy: Neutron energy in MeV
            num_hydrons: Number of hydrons for performance analysis
            num_neutrons: Number of neutrons for efficiency calculation
            
        Returns:
            Dictionary with results or None if calculation failed
        """
        try:            
            # Store original parameters
            new_foil = deepcopy(self.spectrometer.conversion_foil)
            
            # Update foil with new parameters
            for param, value in params.items():
                if param == 'foil_radius':
                    new_foil.set_foil_radius(value)
                elif param == 'thickness':
                    new_foil.set_thickness(value)
                elif param == 'aperture_distance':
                    new_foil.set_aperture_distance(value)
                elif param == 'aperture_radius':
                    new_foil.set_aperture_radius(value)
            
            self.spectrometer.conversion_foil = new_foil
            
            # Calculate efficiency
            scattering_eff, geometric_eff, total_eff = new_foil.calculate_efficiency(
                neutron_energy, 
                num_samples=num_neutrons
            )
            
            # Calculate performance metrics using spectrometer's parallelized methods
            mean_pos, std_pos, fwhm, energy_res = self.spectrometer.analyze_monoenergetic_performance(
                neutron_energy,
                num_hydrons=num_hydrons,
                include_kinematics=True,
                include_stopping_power_loss=True,
                generate_figure=False,
                verbose=False
            )
            
            # Compile results
            # Include initial parameters
            result = params
            result.update({
                'scattering_efficiency': scattering_eff,
                'geometric_efficiency': geometric_eff,
                'total_efficiency': total_eff,
                'mean_position': mean_pos,
                'std_position': std_pos,
                'fwhm': fwhm * 1000, # convert to keV
                'energy_resolution': energy_res * 1000 # convert to keV
            })
            
            return result
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def load_results(self, filename: str) -> pd.DataFrame:
        """Load sweep results from CSV file."""
        self.results_df = pd.read_csv(filename)
        return self.results_df
    
    def get_best_parameters(
        self, 
        metric: str = 'total_efficiency',
        ascending: bool = False
    ) -> Dict[str, Any]:
        """
        Get best parameter combination based on specified metric.
        
        Args:
            metric: Metric to optimize ('total_efficiency', 'energy_resolution', etc.)
            ascending: Whether lower values are better
            
        Returns:
            Dictionary with best parameters and their values
        """
        if self.results_df is None:
            raise ValueError("No results available. Run sweep first or load results.")
        
        if metric not in self.results_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
        
        # Find best row
        best_idx = self.results_df[metric].idxmin() if ascending else self.results_df[metric].idxmax()
        best_row = self.results_df.iloc[best_idx]
        
        # Extract parameter values
        param_cols = [col for col in self.results_df.columns 
                     if col in ['foil_radius', 'thickness', 'aperture_distance', 'aperture_radius']]
        
        best_params = {}
        for param in param_cols:
            if param in self.results_df.columns:
                best_params[param] = best_row[param]
        
        # Add metric value
        best_params[f'best_{metric}'] = best_row[metric]
        
        return best_params