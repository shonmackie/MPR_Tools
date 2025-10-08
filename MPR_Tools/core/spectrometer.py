"""Main MPR spectrometer system implementation."""

from typing import Tuple, Optional, Literal, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

from .conversion_foil import ConversionFoil
from .hodoscope import Hodoscope
from ..analysis.performance import PerformanceAnalyzer
from ..analysis.plotting import SpectrometerPlotter
from ..analysis.data_processing import DataProcessor
from ..io.file_io import BeamIO

class MPRSpectrometer:
    """
    Complete Magnetic Proton Recoil (MPR) spectrometer system.
    
    Combines conversion foil, ion optics transfer map, and hodoscope detector
    for neutron spectroscopy via hydron recoil.
    """
    
    def __init__(
        self,
        conversion_foil: ConversionFoil,
        transfer_map_path: str,
        reference_energy: float,
        min_energy: float,
        max_energy: float,
        hodoscope: Hodoscope,
        figure_directory: str = '.'
    ):
        """
        Initialize complete MPR spectrometer system.
        
        Args:
            conversion_foil: ConversionFoil object
            transfer_map_path: Path to COSY transfer map file
            reference_energy: Reference energy in MeV
            min_energy: Minimum acceptance energy in MeV
            max_energy: Maximum acceptance energy in MeV
            hodoscope: Hodoscope detector system
            figure_directory: Directory for saving figures
        """
        print('Initializing Magnetic Proton Recoil Spectrometer...')
        
        self.conversion_foil = conversion_foil
        self.reference_energy = reference_energy
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.hodoscope = hodoscope
        self.figure_directory = figure_directory
        
        # Initialize helper classes
        self.performance_analyzer = PerformanceAnalyzer(self)
        self.plotter = SpectrometerPlotter(self)
        self.data_processor = DataProcessor(self)
        
        # Load transfer map
        # TODO: add functionality to check if a calibration curve exists for this map. If not, generate one!
        self.transfer_map = np.genfromtxt(transfer_map_path, unpack=True)
        print(f'Loaded COSY transfer map from {transfer_map_path}\n')
        
        # Initialize hydron beam arrays
        self.input_beam: np.ndarray = np.zeros(0)
        self.output_beam: np.ndarray = np.zeros(0)
        
        print('MPR spectrometer initialization complete.\n')
    
    def generate_characteristic_rays(
        self,
        radial_points: int,
        angular_points: int, 
        aperture_radial_points: int,
        aperture_angular_points: int,
        energy_points: int,
        min_energy: float,
        max_energy: float,
    ) -> None:
        """
        Generate characteristic rays on a phase space grid.
        
        Args:
            radial_points: Number of radial points in foil (0 for on-axis only)
            angular_points: Number of angular points in foil
            aperture_radial_points: Number of radial points in aperture
            aperture_angular_points: Number of angular points in aperture
            energy_points: Number of energy points (total = 2*energy_points + 1), equally spread around reference
            min_energy: Minimum energy in MeV
            max_energy: Maximum energy in MeV
        """
        if radial_points == 0:
            num_rays = 2 * energy_points + 1
        else:
            num_rays = ((2 * energy_points + 1) * (radial_points + 1) * angular_points * 
                       (aperture_radial_points + 1) * aperture_angular_points)
        
        # Find all energy values to generate rays for
        energy_values = np.append(
            np.linspace(min_energy, self.reference_energy, energy_points + 1),
            np.linspace(self.reference_energy, max_energy, energy_points + 1)[1:] # don't repeat reference energy
        )
        energy_offset_values = energy_values - self.reference_energy
        
        self.input_beam = np.zeros((num_rays, 6))
        print(f'Characteristic ray energy range: {min_energy:.3f}-{max_energy:.3f} MeV')
        
        ray_index = 0
        duplicates = 0
        
        # Energy loop
        for energy_offset, energy in tqdm(zip(energy_offset_values, energy_values), desc=f'Generating {num_rays} characteristic rays...'):
            
            if radial_points == 0:
                # On-axis ray only
                self.input_beam[ray_index] = [0, 0, 0, 0, energy_offset, energy]
                ray_index += 1
            else:
                # Full phase space grid
                for r_idx in range(radial_points + 1):
                    for ang_idx in range(angular_points):
                        theta = 2 * np.pi * ang_idx / angular_points
                        x_foil = (self.conversion_foil.foil_radius * np.cos(theta) * 
                                 r_idx / radial_points)
                        y_foil = (self.conversion_foil.foil_radius * np.sin(theta) * 
                                 r_idx / radial_points)
                        
                        for ar_idx in range(aperture_radial_points + 1):
                            for aang_idx in range(aperture_angular_points):
                                phi = 2 * np.pi * aang_idx / aperture_angular_points
                                x_aperture = (x_foil + self.conversion_foil.aperture_radius * 
                                            np.cos(phi) * ar_idx / aperture_radial_points)
                                y_aperture = (y_foil + self.conversion_foil.aperture_radius * 
                                            np.sin(phi) * ar_idx / aperture_radial_points)
                                
                                # Calculate angles
                                angle_x = np.arctan((x_aperture - x_foil) / self.conversion_foil.aperture_distance)
                                angle_y = np.arctan((y_aperture - y_foil) / self.conversion_foil.aperture_distance)
                                
                                # Check for duplicates
                                ray = [x_foil, -angle_x, y_foil, -angle_y, energy_offset, energy]
                                is_duplicate = False
                                
                                for prev_idx in range(ray_index):
                                    if np.allclose(self.input_beam[prev_idx], ray):
                                        duplicates += 1
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    self.input_beam[ray_index] = ray
                                    ray_index += 1
        
        print(f'Generated {ray_index} unique rays')
        print(f'Found {duplicates} duplicate rays')
        
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
        Generate hydron rays from neutron energy distribution using Monte Carlo with multiprocessing.
        
        Args:
            neutron_energies: Array of neutron energies in MeV
            energy_distribution: Relative probability distribution (normalized automatically)
            num_hydrons: Number of hydrons to simulate
            include_kinematics: Include kinematic energy transfer
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            z_sampling: Depth sampling method ('exp' or 'uni')
            save_beam: Whether or not to save input beam to csv
            max_workers: Maximum number of worker processes (None for CPU count)
        """        
        if max_workers is None:
            max_workers = mp.cpu_count()
        
        print(f'Generating {num_hydrons} Monte Carlo hydron trajectories using {max_workers} processes...')
        
        # Calculate hydrons per process
        hydrons_per_process = num_hydrons // max_workers
        remaining_hydrons = num_hydrons % max_workers
        
        # Weight energy distribution by n-h scattering cross section
        weighted_distribution = (energy_distribution * 
                            self.conversion_foil.get_nh_cross_section(neutron_energies))
        weighted_distribution /= np.sum(weighted_distribution)
        
        # Create shared counter for progress tracking
        manager = mp.Manager()
        progress_counter = manager.Value('i', 0)
        progress_lock = manager.Lock()
        
        # Initialize progress bar
        pbar = tqdm(total=num_hydrons, desc='Generating Monte Carlo hydron trajectories')
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit jobs
            for i in range(max_workers):
                batch_size = hydrons_per_process + (1 if i < remaining_hydrons else 0)
                if batch_size > 0:  # Only submit if there's work to do
                    # Package all parameters for the worker
                    worker_args = (
                        batch_size,
                        12345 + i * 1000,  # seed_offset
                        neutron_energies,
                        weighted_distribution,
                        include_kinematics,
                        include_stopping_power_loss,
                        z_sampling,
                        self.conversion_foil,
                        self.reference_energy,
                        progress_counter,
                        progress_lock
                    )
                    future = executor.submit(self._generate_batch_worker, *worker_args)
                    futures.append(future)
            
            # Monitor progress while processes run
            last_count = 0
            while any(not future.done() for future in futures):
                current_count = progress_counter.value
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                time.sleep(0.5)  # Check every x ms
            
            # Final update for any remaining progress
            final_count = progress_counter.value
            if final_count > last_count:
                pbar.update(final_count - last_count)
            
            # Collect results
            all_results = []
            total_attempts = 0
            
            for future in as_completed(futures):
                batch_results, batch_attempts = future.result()
                all_results.extend(batch_results)
                total_attempts += batch_attempts
        
        pbar.close()
        
        # Convert to numpy array and ensure we don't exceed requested count
        total_generated = min(len(all_results), num_hydrons)
        self.input_beam = np.array(all_results[:total_generated])
        
        if total_generated < num_hydrons:
            print(f"Warning: Only generated {total_generated}/{num_hydrons} hydrons due to high rejection rate")
        
        print(f'Generated {total_generated} hydrons from {total_attempts} total attempts using {max_workers} processes')
        
        # Save input beam to file
        if save_beam:
            self.save_input_beam()
            
    # Class method worker function (used with multiprocessing)
    def _generate_batch_worker(
        self,
        batch_size: int,
        seed_offset: int,
        neutron_energies: np.ndarray,
        weighted_distribution: np.ndarray,
        include_kinematics: bool,
        include_stopping_power_loss: bool,
        z_sampling: str,
        conversion_foil: ConversionFoil,
        reference_energy: float,
        progress_counter,
        progress_lock
    ) -> Tuple[List[List[float]], int]:
        """Generate a batch of hydrons in a separate process."""
        # Create independent random number generator
        rng = np.random.default_rng(seed_offset)
        
        batch_results = []
        attempts = 0
        max_attempts = batch_size * 20  # Prevent infinite loops
        
        while len(batch_results) < batch_size and attempts < max_attempts:
            try:
                attempts += 1
                
                # Sample neutron energy
                neutron_energy = rng.choice(neutron_energies, p=weighted_distribution)
                
                # Generate scattered hydron with the worker's RNG
                x0, y0, theta_s, phi_s, hydron_energy = conversion_foil.generate_scattered_hydron(
                    neutron_energy, 
                    include_kinematics, 
                    include_stopping_power_loss, 
                    z_sampling=z_sampling,
                    rng=rng  # Pass the worker's RNG
                )
                
                if conversion_foil._check_aperture_acceptance(x0, y0, theta_s, phi_s):
                    # Convert to spectrometer coordinates
                    x_aperture = x0 + conversion_foil.aperture_distance * np.tan(theta_s) * np.cos(phi_s)
                    y_aperture = y0 + conversion_foil.aperture_distance * np.tan(theta_s) * np.sin(phi_s)
                    
                    angle_x = np.arctan((x_aperture - x0) / conversion_foil.aperture_distance)
                    angle_y = np.arctan((y_aperture - y0) / conversion_foil.aperture_distance)
                    
                    energy_relative = (hydron_energy - reference_energy) / reference_energy
                    
                    batch_results.append([x0, angle_x, y0, angle_y, energy_relative, neutron_energy])
                    
                    # Update progress counter thread-safely
                    with progress_lock:
                        progress_counter.value += 1
                    
            except Exception:
                pass  # Skip failed generations
        
        return batch_results, attempts
    
    def apply_transfer_map(
        self,
        map_order: int = 5,
        save_beam: bool = True,
        max_workers: Optional[int] = None
    ) -> None:
        """
        Apply ion optical transfer map to transport hydrons through spectrometer using multiprocessing.
        
        Args:
            map_order: Order of transfer map to apply (1-5 typically)
            save_beam: Whether or not to save output beam to csv
            max_workers: Maximum number of worker processes (None for CPU count)
        """        
        if max_workers is None:
            max_workers = mp.cpu_count()
        
        num_hydrons = len(self.input_beam)
        
        if num_hydrons == 0:
            print("Warning: No input beam data available. Generate rays first.")
            return
        
        print(f'Applying order {map_order} transfer map to {num_hydrons} hydrons using {max_workers} processes...')
        
        self.output_beam = np.zeros((num_hydrons, 5))
        
        # Calculate hydrons per process
        hydrons_per_process = num_hydrons // max_workers
        remaining_hydrons = num_hydrons % max_workers
        
        # Create shared counter for progress tracking
        manager = mp.Manager()
        progress_counter = manager.Value('i', 0)
        progress_lock = manager.Lock()
        
        # Initialize progress bar
        pbar = tqdm(total=num_hydrons, desc=f'Applying order {map_order} transfer map')
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit jobs
            start_idx = 0
            for i in range(max_workers):
                batch_size = hydrons_per_process + (1 if i < remaining_hydrons else 0)
                if batch_size > 0:  # Only submit if there's work to do
                    end_idx = start_idx + batch_size
                    
                    # Package parameters for worker
                    worker_args = (
                        self.input_beam[start_idx:end_idx],
                        self.transfer_map,
                        map_order,
                        progress_counter,
                        progress_lock
                    )
                    
                    future = executor.submit(self._apply_transfer_map_worker, *worker_args)
                    futures.append((future, start_idx, end_idx))
                    start_idx = end_idx
            
            # Monitor progress while processes run
            last_count = 0
            while any(not future[0].done() for future in futures):
                current_count = progress_counter.value
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                time.sleep(0.5)  # Check every x ms
            
            # Final update for any remaining progress
            final_count = progress_counter.value
            if final_count > last_count:
                pbar.update(final_count - last_count)
            
            # Collect results and place them in the correct positions
            for future, start_idx, end_idx in futures:
                batch_output = future.result()
                self.output_beam[start_idx:end_idx] = batch_output
        
        pbar.close()
        
        print('Transfer map applied successfully!')
        
        # Save output beam to file
        if save_beam:
            self.save_output_beam()

    def _apply_transfer_map_worker(
        self,
        input_batch: np.ndarray,
        transfer_map: np.ndarray,
        map_order: int,
        progress_counter,
        progress_lock
    ) -> np.ndarray:
        """
        Worker method to apply transfer map to a batch of hydrons.
        
        Args:
            input_batch: Batch of input rays [N x 6]
            transfer_map: Transfer map coefficients
            map_order: Order of transfer map to apply
            progress_counter: Shared counter for progress tracking
            progress_lock: Lock for thread-safe progress updates
            
        Returns:
            Batch of output rays [N x 5]
        """
        import numpy as np
        
        batch_size = len(input_batch)
        output_batch = np.zeros((batch_size, 5))
        
        for i, input_ray in enumerate(input_batch):
            # Initialize output ray with input energy
            output_ray = np.array([0.0, 0.0, 0.0, 0.0, input_ray[4]])
            
            # Apply each map term
            for j, term_index in enumerate(transfer_map[-1]):
                term_powers = self._extract_digits(term_index)
                
                # Only include terms up to specified order
                if np.sum(term_powers) <= map_order:
                    # Calculate monomial term
                    monomial = np.prod([input_ray[k]**term_powers[k] for k in range(4)]) * input_ray[4]**term_powers[5]
                    
                    # Add contributions to each coordinate
                    for coord in range(4):  # x, angle_x, y, angle_y
                        output_ray[coord] += transfer_map[coord, j] * monomial
            
            output_batch[i] = output_ray
            
            # Update progress counter thread-safely
            with progress_lock:
                progress_counter.value += 1
        
        return output_batch
    
    def _extract_digits(self, number: float) -> np.ndarray:
        """
        Extract digits from a number for transfer map indexing.
        
        Args:
            number: Input number to extract digits from
            
        Returns:
            Array of 6 digits
        """
        digits = np.zeros(6, dtype=int)
        formatted_str = f"{number/1e5:.5f}"
        
        digit_idx = 0
        for char in formatted_str:
            if char not in ['.', 'e', '-'] and digit_idx < 6:
                digits[digit_idx] = int(char)
                digit_idx += 1
                
        return digits
    
    def save_input_beam(self, filepath: Optional[str] = None) -> None:
        """Save input beam to CSV file."""
        if filepath is None:
            filepath = f'{self.figure_directory}/input_beam.csv'
        
        df = pd.DataFrame({
            'x0': self.input_beam[:, 0],
            'angle_x': self.input_beam[:, 1],
            'y0': self.input_beam[:, 2],
            'angle_y': self.input_beam[:, 3],
            'energy_relative': self.input_beam[:, 4],
            'neutron_energy': self.input_beam[:, 5]
        })
        df.to_csv(filepath, index=False)
        print(f'Input beam saved to {filepath}')
    
    def save_output_beam(self, filepath: Optional[str] = None) -> None:
        """Save output beam to CSV file."""
        if filepath is None:
            filepath = f'{self.figure_directory}/output_beam.csv'
        
        df = pd.DataFrame({
            'x0': self.output_beam[:, 0],
            'angle_x': self.output_beam[:, 1],
            'y0': self.output_beam[:, 2],
            'angle_y': self.output_beam[:, 3],
            'energy_relative': self.output_beam[:, 4]
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
            input_beam_path = f'{self.figure_directory}/input_beam.csv'
            
        input_beam_df = pd.read_csv(input_beam_path)
        self.input_beam = input_beam_df.to_numpy()
        
        # Read output beam
        if output_beam_path == None:
            output_beam_path = f'{self.figure_directory}/output_beam.csv'
            
        output_beam_df = pd.read_csv(output_beam_path)
        self.output_beam = output_beam_df.to_numpy()
        
        return self.input_beam, self.output_beam
    
    # Delegate analysis methods to helper classes
    def analyze_monoenergetic_performance(self, *args, **kwargs):
        return self.performance_analyzer.analyze_monoenergetic_performance(*args, **kwargs)
    
    def generate_performance_curve(self, *args, **kwargs):
        return self.performance_analyzer.generate_performance_curve(*args, **kwargs)
    
    # Delegate plotting methods to helper classes  
    def plot_focal_plane_distribution(self, *args, **kwargs):
        return self.plotter.plot_focal_plane_distribution(*args, **kwargs)
        
    def plot_phase_space(self, *args, **kwargs):
        return self.plotter.plot_phase_space(*args, **kwargs)
    
    def _plot_monoenergetic_analysis(self, *args, **kwargs):
        return self.plotter._plot_monoenergetic_analysis(*args, **kwargs)
    
    def _plot_performance(self, *args, **kwargs):
        return self.plotter._plot_performance(*args, **kwargs)
    
    def plot_simple_position_histogram(self, *args, **kwargs):
        return self.plotter.plot_simple_position_histogram(*args, **kwargs)
    
    def plot_input_ray_geometry(self, *args, **kwargs):
        return self.plotter.plot_input_ray_geometry(*args, **kwargs)
    
    def plot_proton_density_heatmap(self, *args, **kwargs):
        return self.plotter.plot_proton_density_heatmap(*args, **kwargs)
    
    def plot_characteristic_rays(self, *args, **kwargs):
        return self.plotter.plot_characteristic_rays(*args, **kwargs)
    
    def plot_synthetic_neutron_histogram(self, *args, **kwargs):
        return self.plotter.plot_synthetic_neutron_histogram(*args, **kwargs)
    
    # Delegate data processing methods to helper classes
    def bin_hodoscope_response(self, *args, **kwargs):
        return self.data_processor.bin_hodoscope_response(*args, **kwargs)
    
    def get_proton_density_map(self, *args, **kwargs):
        return self.data_processor.get_proton_density_map(*args, **kwargs)
    
    def get_plasma_parameters(self, *args, **kwargs):
        return self.data_processor.get_plasma_parameters(*args, **kwargs)
    
    def get_system_summary(self) -> dict:
        """
        Get summary of spectrometer system parameters.
        
        Returns:
            Dictionary containing system parameters
        """
        return {
            'foil_radius_cm': self.conversion_foil.foil_radius_cm,
            'foil_thickness_um': self.conversion_foil.thickness_um,
            'aperture_distance_cm': self.conversion_foil.aperture_distance_cm,
            'aperture_radius_cm': self.conversion_foil.aperture_radius_cm,
            'aperture_type': self.conversion_foil.aperture_type,
            'reference_energy_MeV': self.reference_energy,
            'min_energy': self.min_energy,
            'max_energy': self.max_energy,
            'hodoscope_channels': self.hodoscope.total_channels,
            'detector_width_cm': self.hodoscope.detector_width_cm,
            'detector_height_cm': self.hodoscope.detector_height_cm,
            'num_input_hydrons': len(self.input_beam),
            'num_output_hydrons': len(self.output_beam)
        }