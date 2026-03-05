"""Main MPR spectrometer system implementation."""

import os
from typing import Tuple, Optional, Literal
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import Executor
import multiprocessing as mp

from .conversion_foil import ConversionFoil
from .hodoscope import Hodoscope
from .parallelization import run_concurrently


class MPRSpectrometer:
    """
    Complete Magnetic Proton Recoil (MPR) spectrometer system.
    
    Combines conversion foil, ion optics transfer map, and hodoscope detector
    for neutral particle spectroscopy via charged particle recoil.
    """
    
    def __init__(
        self,
        conversion_foil: ConversionFoil,
        transfer_map_path: str,
        reference_energy: float,
        min_energy: float,
        max_energy: float,
        hodoscope: Hodoscope,
        run_directory: str = '.'
    ):
        """
        Initialize complete MPR spectrometer system.
        
        Args:
            conversion_foil: ConversionFoil object
            transfer_map_path: Path to COSY transfer map file
            reference_energy: Reference energy in MeV
            min_energy: Minimum recoil particle acceptance energy in MeV
            max_energy: Maximum recoil particle acceptance energy in MeV
            hodoscope: Hodoscope detector system
            run_directory: Directory for saving run data and figures
        """
        print(f'Initializing Magnetic {conversion_foil.particle.capitalize()} Recoil Spectrometer...')
        
        self.conversion_foil = conversion_foil
        self.reference_energy = reference_energy
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.hodoscope = hodoscope
        self.figure_directory = f'{run_directory}/figures'
        self.data_directory = f'{run_directory}/data'
        
        # Create directories if they don't exist
        if not os.path.exists(self.figure_directory):
            os.makedirs(self.figure_directory)
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        
        # calculate approximate incident particle energy bounds
        self.min_incident_energy = conversion_foil.get_incident_energy(min_energy)
        self.max_incident_energy = conversion_foil.get_incident_energy(max_energy)
        
        # Load transfer map
        # TODO: add functionality to check if a calibration curve exists for this map. If not, generate one!
        self.transfer_map = np.genfromtxt(transfer_map_path, unpack=True)
        print(f'Loaded COSY transfer map from {transfer_map_path}\n')
        
        # Initialize recoil beam arrays
        self.input_beam: np.ndarray = np.zeros(0)
        self.output_beam: np.ndarray = np.zeros(0)
        
        print(f'M{conversion_foil.particle[0].capitalize()}R spectrometer initialization complete.\n')
    
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
        
        particle_rest_energy = self.conversion_foil.particle_mass*931.494  # MeV
        reference_gamma = 1 + self.reference_energy/particle_rest_energy  # Lorentz factor of the central ray

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
                                
                                # Calculate relative transverse momenta since that's what COSY uses instead of angle fsr
                                gamma = 1 + energy/particle_rest_energy  # lorentz factor of the ray
                                p_relative = np.sqrt((gamma**2 - 1)/(reference_gamma**2 - 1))
                                p_x_relative = p_relative * np.sin(angle_x)
                                p_y_relative = p_relative * np.sin(angle_y)

                                # Check for duplicates
                                ray = [x_foil, -p_x_relative, y_foil, -p_y_relative, energy_offset, energy]
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
        incident_energies: np.ndarray,
        energy_distribution: np.ndarray,
        num_recoil_particles: int,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        z_sampling: Literal['exp', 'uni'] = 'exp',
        save_beam: bool = True,
        executor: Optional[Executor] = None,
        max_workers: Optional[int] = None,
        y_restriction: Optional[Literal['positive', 'negative']] = None
    ) -> None:
        """
        Generate recoil rays from incident particle energy distribution using Monte Carlo with multiprocessing.
        
        Args:
            incident_energies: Array of incident particle energies in MeV
            energy_distribution: Relative probability distribution (normalized automatically)
            num_recoil_particles: Number of recoil particles to simulate
            include_kinematics: Include kinematic energy transfer
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            z_sampling: Depth sampling method ('exp' or 'uni')
            save_beam: Whether or not to save input beam to csv
            executor: Pool of workers to use (if None, we will make our own)
            max_workers: Maximum number of worker processes (None for CPU count)
        """
        if max_workers is None:
            max_workers = mp.cpu_count()
        
        print(f'Generating {num_recoil_particles} Monte Carlo {self.conversion_foil.particle} trajectories using {max_workers} processes...')
        
        # Calculate recoil events per process
        particles_per_process = num_recoil_particles // max_workers
        remaining_particles = num_recoil_particles % max_workers
        
        # Narrow energy distribution unless doing monoenergetic performance
        if len(incident_energies) > 1:
            # Only use incident energies that can possibly produce recoil energies within acceptance range
            idx = incident_energies >= self.min_energy
            incident_energies = incident_energies[idx]
            energy_distribution = energy_distribution[idx]
        
        # Weight energy distribution by scattering cross section
        interaction_probability = np.zeros_like(energy_distribution)
        for interaction in self.conversion_foil.interactions:
            interaction_probability += (interaction.get_cross_section(incident_energies) *
                                        interaction.get_recoil_probability())
        weighted_distribution = energy_distribution * interaction_probability
        weighted_distribution /= np.sum(weighted_distribution)
        
        # Execute in parallel
        worker_args = []
        for i in range(max_workers):
            batch_size = particles_per_process + (1 if i < remaining_particles else 0)
            if batch_size > 0:  # Only submit if there's work to do
                # Package all parameters for the worker
                worker_args.append((
                    batch_size,
                    12345 + i * 1000,  # seed_offset
                    incident_energies,
                    weighted_distribution,
                    include_kinematics,
                    include_stopping_power_loss,
                    z_sampling,
                    self.conversion_foil,
                    self.reference_energy,
                    y_restriction
                ))
        
        output_batches = run_concurrently(
            self._generate_batch_worker, worker_args, executor,
            progress_counter_total=num_recoil_particles,
            task_title=f'Generating Monte Carlo {self.conversion_foil.particle} trajectories',
        )
        
        # Collect results
        self.input_beam = np.concatenate(output_batches)
        
        # Save input beam to file
        if save_beam:
            self.save_input_beam()
            
    # Class method worker function (used with multiprocessing)
    def _generate_batch_worker(
        self,
        batch_size: int,
        seed_offset: int,
        incident_energies: np.ndarray,
        weighted_distribution: np.ndarray,
        include_kinematics: bool,
        include_stopping_power_loss: bool,
        z_sampling: str,
        conversion_foil: ConversionFoil,
        reference_energy: float,
        y_restriction: Optional[Literal['positive', 'negative']],
        progress_counter,
        progress_lock,
    ) -> np.ndarray:
        """Generate a batch of recoil particles in a separate process."""
        # Create independent random number generator
        rng = np.random.default_rng(seed_offset)
        
        particle_rest_energy = conversion_foil.particle_mass*931.494  # MeV
        reference_gamma = 1 + reference_energy/particle_rest_energy  # Lorentz factor of the central ray
        
        batch_results = np.empty((0, 6), dtype=float)
        
        while len(batch_results) < batch_size:
            try:                
                # Generate recoil particle with the worker's RNG
                # The recoil particles generated are already accepted by the aperture
                x0, y0, theta_s, phi_s, incident_energy, recoil_energy = conversion_foil.generate_recoil_particle(
                    incident_energies,
                    weighted_distribution,
                    include_kinematics, 
                    include_stopping_power_loss,
                    z_sampling=z_sampling,
                    rng=rng,  # Pass the worker's RNG
                    y_restriction=y_restriction
                )
                
                # Convert to spectrometer coordinates
                x_aperture = x0 + conversion_foil.aperture_distance * np.tan(theta_s) * np.cos(phi_s)
                y_aperture = y0 + conversion_foil.aperture_distance * np.tan(theta_s) * np.sin(phi_s)
                
                angle_x = np.arctan((x_aperture - x0) / conversion_foil.aperture_distance)
                angle_y = np.arctan((y_aperture - y0) / conversion_foil.aperture_distance)
                
                gamma = 1 + recoil_energy/particle_rest_energy  # Lorentz factor of the ray
                p_relative = np.sqrt((gamma**2 - 1)/(reference_gamma**2 - 1))
                p_x_relative = p_relative * np.sin(angle_x)
                p_y_relative = p_relative * np.sin(angle_y)
                
                energy_relative = (recoil_energy - reference_energy) / reference_energy
                
                batch_results = np.vstack((batch_results, np.array([x0, p_x_relative, y0, p_y_relative, energy_relative, incident_energy])))
                
                # Update progress counter thread-safely
                with progress_lock:
                    progress_counter.value += 1
                    
            except Exception as e:
                print(e)
                print(f'Failed to generate {self.conversion_foil.particle}')
                pass  # Skip failed generations
        
        return batch_results
    
    def apply_transfer_map(
        self,
        map_order: int = 5,
        save_beam: bool = True,
        executor: Optional[Executor] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Apply ion optical transfer map to transport recoil particles through spectrometer using multiprocessing.
        
        Args:
            map_order: Order of transfer map to apply (1-5 typically)
            save_beam: Whether or not to save output beam to csv
            executor: Pool of workers to use (if None, we will make our own)
            max_workers: Maximum number of worker processes (None for CPU count)
        """
        if max_workers is None:
            max_workers = mp.cpu_count()
        
        num_recoil_particles = len(self.input_beam)
        
        if num_recoil_particles == 0:
            print("Warning: No input beam data available. Generate rays first.")
            return
        
        print(f'Applying order {map_order} transfer map to {num_recoil_particles} {self.conversion_foil.particle}s using {max_workers} processes...')
        
        # Calculate recoil particles per process
        particles_per_process = num_recoil_particles // max_workers
        remaining_particles = num_recoil_particles % max_workers
        
        # Execute in parallel
        worker_args = []
        start_idx = 0
        for i in range(max_workers):
            batch_size = particles_per_process + (1 if i < remaining_particles else 0)
            if batch_size > 0:  # Only submit if there's work to do
                end_idx = start_idx + batch_size
                
                # Package parameters for worker
                worker_args.append((
                    self.input_beam[start_idx:end_idx],
                    self.transfer_map,
                    self.conversion_foil.relative_mass,
                    map_order,
                ))
                
                start_idx = end_idx
        
        output_batches = run_concurrently(
            self._apply_transfer_map_worker, worker_args, executor,
            progress_counter_total=num_recoil_particles,
            task_title=f'Applying order {map_order} transfer map',
        )
        
        self.output_beam = np.concatenate(output_batches)
        
        print('Transfer map applied successfully!')
        
        # Save output beam to file
        if save_beam:
            self.save_output_beam()

    def _apply_transfer_map_worker(
        self,
        input_batch: np.ndarray,
        transfer_map: np.ndarray,
        relative_mass: float,
        map_order: int,
        progress_counter,
        progress_lock
    ) -> np.ndarray:
        """
        Worker method to apply transfer map to a batch of recoil particles.
        
        Args:
            input_batch: Batch of input rays [N x 6]
            transfer_map: Transfer map coefficients
            relative_mass: Relative mass of recoil particle to proton, only used if mass is included in transfer map
            map_order: Order of transfer map to apply
            progress_counter: Shared counter for progress tracking
            progress_lock: Lock for thread-safe progress updates
            
        Returns:
            Batch of output rays [N x 5]
        """        
        batch_size = len(input_batch)
        output_batch = np.zeros((batch_size, 5))
        
        ### Convert last column of transfer map to monomial powers for each term ###
        # Need to convert term powers to integers
        term_indices = transfer_map[-1].astype(int)
        
        # Find maximum number of digits
        max_digits = len(str(np.max(term_indices)))
        mass_included = max_digits >= 7 # Only 6 digits if mass is not included
        
        # Convert to zero-padded strings
        term_indices_str = np.array([str(x).zfill(max_digits) for x in term_indices])
        
        # Extract digits for each term
        term_powers_array = np.array([list(s) for s in term_indices_str], dtype=int)
        
        ### Apply transfer map to each recoil ray ###
        for i, input_ray in enumerate(input_batch):
            # Initialize output ray with input energy
            output_ray = np.array([0.0, 0.0, 0.0, 0.0, input_ray[4]])
            
            # Apply each map term
            for j, term_powers in enumerate(term_powers_array):                
                # Only include terms up to specified order
                if np.sum(term_powers) <= map_order:
                    # Calculate monomial term
                    monomial = np.prod([input_ray[k]**term_powers[k] for k in range(4)]) * input_ray[4]**term_powers[5]
                    if mass_included:
                        monomial *= relative_mass**term_powers[6]
                    
                    # Add contributions to each coordinate
                    for coord in range(4):  # x, p_x, y, p_y
                        output_ray[coord] += transfer_map[coord, j] * monomial
            
            output_batch[i] = output_ray
            
            # Update progress counter thread-safely
            with progress_lock:
                progress_counter.value += 1
        
        return output_batch
    
    def save_input_beam(self, filepath: Optional[str] = None) -> None:
        """Save input beam to CSV file."""
        if filepath is None:
            filepath = f'{self.data_directory}/input_beam.csv'
        
        df = pd.DataFrame({
            'x0': self.input_beam[:, 0],
            'p_x_relative': self.input_beam[:, 1],
            'y0': self.input_beam[:, 2],
            'p_y_relative': self.input_beam[:, 3],
            'energy_relative': self.input_beam[:, 4],
            'incident_energy': self.input_beam[:, 5]
        })
        df.to_csv(filepath, index=False)
        print(f'Input beam saved to {filepath}')
    
    def save_output_beam(self, filepath: Optional[str] = None) -> None:
        """Save output beam to CSV file."""
        if filepath is None:
            filepath = f'{self.data_directory}/output_beam.csv'
        
        df = pd.DataFrame({
            'x0': self.output_beam[:, 0],
            'p_x_relative': self.output_beam[:, 1],
            'y0': self.output_beam[:, 2],
            'p_y_relative': self.output_beam[:, 3],
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
            input_beam_path = f'{self.data_directory}/input_beam.csv'
            
        input_beam_df = pd.read_csv(input_beam_path)
        self.input_beam = input_beam_df.to_numpy()
        
        # Read output beam
        if output_beam_path == None:
            output_beam_path = f'{self.data_directory}/output_beam.csv'
            
        output_beam_df = pd.read_csv(output_beam_path)
        self.output_beam = output_beam_df.to_numpy()
        
        return self.input_beam, self.output_beam
    
    def bin_hodoscope_response(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin recoil particle hits into hodoscope channels.
        
        Returns:
            Tuple of (channel_numbers, counts_per_channel)
        """
        channel_counts = np.zeros(self.hodoscope.total_channels)
        
        for x_position in self.output_beam[:, 0]:
            channel = self.hodoscope.get_channel_for_position(x_position)
            if channel is not None:
                channel_counts[channel] += 1
        
        channel_numbers = np.arange(self.hodoscope.total_channels)
        return channel_numbers, channel_counts
    
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
            'particle': self.conversion_foil.particle,
            'reference_energy_MeV': self.reference_energy,
            'min_energy': self.min_energy,
            'max_energy': self.max_energy,
            'hodoscope_channels': self.hodoscope.total_channels,
            'detector_width_cm': self.hodoscope.detector_width_cm,
            'num_input_recoil_particles': len(self.input_beam),
            'num_output_recoil_particles': len(self.output_beam)
        }
