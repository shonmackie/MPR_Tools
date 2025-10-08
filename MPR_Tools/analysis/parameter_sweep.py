"""Parameter sweep analysis for MPR spectrometer components."""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import itertools
from copy import deepcopy
import optuna

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
    
    def setup_optimization(
        self,
        foil_radius_range: Optional[Tuple[float, float]] = None,
        thickness_range: Optional[Tuple[float, float]] = None,
        aperture_radius_range: Optional[Tuple[float, float]] = None,
        aperture_distance_range: Optional[Tuple[float, float]] = None,
        study_name: Optional[str] = None,
        storage_path: Optional[str] = None
    ) -> None:
        """
        Set up optimization parameter ranges and study configuration.
        If a range is None, that parameter is fixed at its current value from the foil.
        
        Args:
            foil_radius_range: (min, max) for foil radius (None = use current value)
            thickness_range: (min, max) for foil thickness (None = use current value)
            aperture_radius_range: (min, max) for aperture radius (None = use current value)
            aperture_distance_range: (min, max) for aperture distance (None = use current value)
            study_name: Name for the Optuna study
            storage_path: Path for SQLite database storage
        """
        self.optimization_ranges = {}
        self.fixed_parameters = {}
        
        # Get current foil parameters
        current_foil = self.spectrometer.conversion_foil
        
        # Set up ranges or fixed values for each parameter
        if foil_radius_range is not None:
            self.optimization_ranges['foil_radius'] = foil_radius_range
        else:
            self.fixed_parameters['foil_radius'] = current_foil.foil_radius_cm
        
        if thickness_range is not None:
            self.optimization_ranges['thickness'] = thickness_range
        else:
            self.fixed_parameters['thickness'] = current_foil.thickness_um
        
        if aperture_radius_range is not None:
            self.optimization_ranges['aperture_radius'] = aperture_radius_range
        else:
            self.fixed_parameters['aperture_radius'] = current_foil.aperture_radius_cm
        
        if aperture_distance_range is not None:
            self.optimization_ranges['aperture_distance'] = aperture_distance_range
        else:
            self.fixed_parameters['aperture_distance'] = current_foil.aperture_distance_cm
        
        if study_name is not None:
            self.study_name = study_name
            
        if storage_path is not None:
            self.storage_path = storage_path
        else:
            self.storage_path = f'{self.spectrometer.figure_directory}/optuna.db'
        
        print(f"Optimization configured:")
        if self.optimization_ranges:
            print(f"  Trainable parameters:")
            for param, (min_val, max_val) in self.optimization_ranges.items():
                print(f"    {param}: [{min_val}, {max_val}]")
        if self.fixed_parameters:
            print(f"  Fixed parameters:")
            for param, value in self.fixed_parameters.items():
                print(f"    {param}: {value}")
    
    def run_optimization(
        self,
        neutron_energy: float,
        n_trials: int = 100,
        num_hydrons: int = 10000,
        num_neutrons: int = 100000,
        load_if_exists: bool = True
    ) -> optuna.Study:
        """
        Run multi-objective optimization using Optuna.
        
        Args:
            neutron_energy: Neutron energy to analyze in MeV
            n_trials: Number of optimization trials to run
            num_hydrons: Number of hydrons for performance analysis
            num_neutrons: Number of neutrons for efficiency calculation
            load_if_exists: If True, load existing study if it exists
            
        Returns:
            Optuna Study object with optimization results
        """
        if self.storage_path is None:
            raise ValueError("Must call setup_optimization() before run_optimization()")
        
        storage_url = f'sqlite:///{self.storage_path}'
        
        print(f"Creating new optimization study: {self.study_name}")
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_url,
            directions=['minimize', 'maximize'],  # minimize energy resolution, maximize efficiency
            load_if_exists=load_if_exists
        )
        
        # Create objective function with bound parameters
        def objective(trial: optuna.Trial) -> Tuple[float, float]:
            return self._optimization_objective(trial, neutron_energy, num_hydrons, num_neutrons)
        
        print(f"Starting optimization with {n_trials} trials...")
        self.study.optimize(objective, n_trials=n_trials)
        print(f"Optimization complete. Best trials on Pareto front: {len(self.study.best_trials)}")
        
        return self.study
    
    def _optimization_objective(
        self,
        trial: optuna.Trial,
        neutron_energy: float,
        num_hydrons: int,
        num_neutrons: int
    ) -> Tuple[float, float]:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            neutron_energy: Neutron energy in MeV
            num_hydrons: Number of hydrons for performance analysis
            num_neutrons: Number of neutrons for efficiency calculation
            
        Returns:
            Tuple of (energy_resolution, log10_efficiency)
        """
        # Build params dict with suggested values for trainable parameters
        params = {}
        for param_name, param_range in self.optimization_ranges.items():
            params[param_name] = trial.suggest_float(param_name, *param_range)
        
        # Add fixed parameters
        params.update(self.fixed_parameters)
        
        # Evaluate configuration
        result = self._evaluate_parameter_combination(params, neutron_energy, num_hydrons, num_neutrons)
        
        if result is None:
            raise ValueError(f"Optimization failed for parameters: {params}")
        
        # Return objectives: minimize energy_resolution, maximize log10_efficiency
        return result['energy_resolution'], np.log10(result['total_efficiency'])
    
    def load_study(self, study_name: Optional[str] = None) -> optuna.Study:
        """
        Load existing Optuna study from database.
        
        Args:
            study_name: Name of study to load (default: use self.study_name)
            
        Returns:
            Loaded Optuna Study object
        """
        if self.storage_path is None:
            raise ValueError("Must call setup_optimization() before load_study()")
        
        name = study_name if study_name is not None else self.study_name
        storage_url = f'sqlite:///{self.storage_path}'
        
        self.study = optuna.load_study(
            study_name=name,
            storage=storage_url
        )
        print(f"Loaded study '{name}' with {len(self.study.trials)} trials")
        
        return self.study