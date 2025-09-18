"""Conversion foil implementation for neutron-hydron scattering."""

from typing import Tuple, Optional, Literal
import numpy as np
from numpy.polynomial.legendre import legval
from pathlib import Path
from scipy.interpolate import RectBivariateSpline
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
from tqdm import tqdm

from ..config.constants import AVOGADRO, NEUTRON_MASS, FOIL_MATERIALS, DEFAULT_DATA_PATHS

class ConversionFoil:
    """
    Represents a conversion foil and aperture system for neutron-hydron scattering.
    
    The foil is where neutrons impinge and scatter hydrons, while the aperture
    defines the ion optical acceptance.
    """    
    def __init__(
        self,
        foil_radius: float,
        thickness: float,
        aperture_distance: float,
        aperture_radius: float,
        srim_data_path: Optional[str] = None,
        nh_cross_section_path: Optional[str] = None,
        nc12_cross_section_path: Optional[str] = None,
        differential_xs_path: Optional[str] = None,
        z_grid_points: int = 1000000,
        foil_material: Literal['CH2', 'CD2'] = 'CH2',
        aperture_type: Literal['circ', 'rect'] = 'circ'
    ):
        """
        Initialize conversion foil system.
        
        Args:
            foil_radius: Foil radius in cm
            thickness: Foil thickness in μm
            aperture_distance: Distance from foil to aperture in cm
            aperture_radius: Aperture radius in cm
            srim_data_path: Path to SRIM stopping power data
            nh_cross_section_path: Path to n-hydron elastic scattering cross sections (either protons or deuterons)
            nc12_cross_section_path: Path to n-C12 elastic scattering cross sections
            differential_xs_path: Path to differential cross section data
            z_grid_points: Number of z-direction grid points
            foil_material: Foil material to use ('CH2' or 'CD2')
            aperture_type: Type of aperture ('circ' or 'rect')
        """
        print('Initializing convergence foil...')
        
        # Convert units and store geometry
        self.foil_radius = foil_radius * 1e-2  # cm to m
        self.thickness = thickness * 1e-6      # μm to m
        self.aperture_distance = aperture_distance * 1e-2  # cm to m
        self.aperture_radius = aperture_radius * 1e-2      # cm to m
        self.aperture_type = aperture_type
        
        # Calculate particle densities in CH2
        self.foil_material = foil_material
        self.particle = FOIL_MATERIALS[foil_material]['particle']
        foil_density = FOIL_MATERIALS[foil_material]['density'] # g/cm^3
        molecular_weight = FOIL_MATERIALS[foil_material]['molecular_weight'] # g/mol
        self.hydron_mass = FOIL_MATERIALS[foil_material]['hydron_mass'] # amu
        
        density_factor = foil_density * AVOGADRO * 1e6
        self.carbon_density = density_factor / molecular_weight # carbon/m^3
        self.hydron_density = self.carbon_density * 2 # hydrons/m^3
        
        # Initialize sampling grids
        # Exit of the foil is z=0
        self.z_grid = np.linspace(-self.thickness, 0, z_grid_points)
        
        # Load cross section and stopping power data
        module_dir = Path(__file__).parent
        if srim_data_path is None:
            self.srim_data_path = module_dir / DEFAULT_DATA_PATHS[foil_material]['srim']
        else:
            self.srim_data_path = srim_data_path
            
        if nh_cross_section_path is None:
            self.nh_cross_section_path = module_dir / DEFAULT_DATA_PATHS[foil_material]['cross_section']
        else:
            self.nh_cross_section_path = nh_cross_section_path
            
        if nc12_cross_section_path is None:
            self.nc12_cross_section_path = module_dir / DEFAULT_DATA_PATHS['nc12_cross_section']
        else:
            self.nc12_cross_section_path = nc12_cross_section_path
            
        if differential_xs_path is None:
            self.differential_xs_path = module_dir / DEFAULT_DATA_PATHS[foil_material]['differential_xs']
        else:
            self.differential_xs_path = differential_xs_path
        self._load_data_files()
        
        # Build interpolator for differential cross section data
        self._build_differential_xs_interpolator()
        
        print('Conversion foil initialization complete.\n')
    
    def _load_data_files(self) -> None:
        """Load all required data files."""
        self.srim_data = np.genfromtxt(self.srim_data_path, skip_header=2, unpack=True)
        print(f'Loaded SRIM data from {self.srim_data_path}')
        
        self.nh_cross_section_data = np.genfromtxt(self.nh_cross_section_path, skip_header=6, usecols=(0, 1), unpack=True)
        print(f'Loaded n-{self.particle} elastic scattering cross sections from {self.nh_cross_section_path}')
        
        self.nc12_cross_section_data = np.genfromtxt(self.nc12_cross_section_path, skip_header=6, usecols=(0, 1), unpack=True)
        print(f'Loaded n-C12 elastic scattering cross sections from {self.nc12_cross_section_path}')
        
        # Need to read as a pandas df and convert to numpy because some 
        # differential cross sections have different number of Legendre 
        # coefficients for each energy
        self.differential_xs_data = pd.read_csv(self.differential_xs_path, sep='\s+', comment='#').to_numpy(dtype=np.float64).T
        print(f'Loaded differential scattering data from {self.differential_xs_path}')
    
    @property
    def foil_radius_cm(self) -> float:
        """Get foil radius in cm."""
        return self.foil_radius * 1e2
    
    @property
    def thickness_um(self) -> float:
        """Get thickness in μm."""
        return self.thickness * 1e6
    
    @property
    def aperture_distance_cm(self) -> float:
        """Get aperture distance in cm."""
        return self.aperture_distance * 1e2
    
    @property
    def aperture_radius_cm(self) -> float:
        """Get aperture radius in cm."""
        return self.aperture_radius * 1e2
    
    def set_foil_radius(self, radius_cm: float, verbose: bool = False) -> None:
        """Set foil radius in cm."""
        self.foil_radius = radius_cm * 1e-2
        if verbose:
            print(f'Set conversion foil radius to {radius_cm:.2f} cm')
    
    def set_thickness(self, thickness_um: float, verbose: bool = False) -> None:
        """Set foil thickness in μm."""
        self.thickness = thickness_um * 1e-6
        self.z_grid = np.linspace(-self.thickness, 0, len(self.z_grid))
        if verbose:
            print(f'Set conversion foil thickness to {thickness_um:.1f} μm')
    
    def set_aperture_distance(self, distance_cm: float) -> None:
        """
        Set foil-aperture separation in cm.
        
        Warning: This impacts ion optical transfer maps. Ensure consistency 
        with COSY inputs for accurate results.
        """
        self.aperture_distance = distance_cm * 1e-2
    
    def set_aperture_radius(self, radius_cm: float, verbose: bool = False) -> None:
        """Set aperture radius in cm."""
        self.aperture_radius = radius_cm * 1e-2
        if verbose:
            print(f'Set aperture radius to {radius_cm:.2f} cm')
    
    def calculate_stopping_power(self, energy_MeV: float) -> float:
        """
        Calculate stopping power dE/dx [MeV/mm] for hydrons in the foil.
        
        Args:
            energy_MeV: hydron energy in MeV
            
        Returns:
            Stopping power in MeV/mm
        """
        return np.interp(
            energy_MeV, 
            self.srim_data[0], 
            self.srim_data[1] + self.srim_data[2]  # electronic + nuclear stopping
        )
    
    def calculate_stopping_power_loss(
        self, 
        initial_energy: float, 
        path_length: float, 
        num_steps: int = 1000
    ) -> float:
        """
        Calculate energy after slowing down in foil material.
        
        Neglects straggling (valid for thin foils, less accurate at low energies).
        
        Args:
            initial_energy: Initial hydron energy in MeV
            path_length: Distance traveled through material in m
            num_steps: Number of discretization steps
            
        Returns:
            Final hydron energy in MeV
        """
        step_size = path_length / num_steps
        energy = initial_energy
        
        for _ in range(num_steps):
            # stopping power is in MeV/mm, convert to MeV/m
            energy -= self.calculate_stopping_power(energy) * step_size * 1e3 
            if energy <= 0:
                break
                
        return max(energy, 0)
    
    def calculate_initial_energy(
        self, 
        final_energy: float, 
        path_length: float, 
        num_steps: int = 1000
    ) -> float:
        """
        Calculate initial energy by reversing energy loss calculation.
        
        Args:
            final_energy: Final hydron energy in MeV
            path_length: Distance traveled through material in m
            num_steps: Number of discretization steps
            
        Returns:
            Initial hydron energy in MeV
        """
        step_size = path_length / num_steps
        energy = final_energy
        
        for _ in range(num_steps):
            # stopping power is in MeV/mm, convert to MeV/m
            energy += self.calculate_stopping_power(energy) * step_size * 1e3
            
        return energy
    
    def get_nh_cross_section(self, energy_MeV: float) -> float:
        """
        Get n-h elastic scattering cross section.
        
        Args:
            energy_MeV: Incident neutron energy in MeV
            
        Returns:
            Cross section in m^2
        """
        energy_eV = energy_MeV * 1e6
        cross_section_barns = np.interp(
            energy_eV,
            self.nh_cross_section_data[0], 
            self.nh_cross_section_data[1])
        return cross_section_barns * 1e-28  # barns to m^2
    
    def get_nc12_cross_section(self, energy_MeV: float) -> float:
        """
        Get n-C12 elastic scattering cross section.
        
        Args:
            energy_MeV: Incident neutron energy in MeV
            
        Returns:
            Cross section in m^2
        """
        energy_eV = energy_MeV * 1e6
        cross_section_barns = np.interp(energy_eV, self.nc12_cross_section_data[0], 
                                       self.nc12_cross_section_data[1])
        return cross_section_barns * 1e-28  # barns to m^2
    
    def _build_differential_xs_interpolator(self):
        """
            Build interpolator for differential cross section data.
            Formulae are from section 4.1 of ENDF Manual: https://www.oecd-nea.org/dbdata/data/endf102.htm#LinkTarget_12655
        """
        # Extract energy grid
        energy_grid = self.differential_xs_data[0]
        # Cosine grid
        cos_theta_grid = np.linspace(-1, 1, 100)
        
        # Initialize lookup table
        xs_lookup = np.zeros((len(energy_grid), len(cos_theta_grid)))
        
        # Helper function to evaluate Legendre expansion
        def _evaluate_legendre(coefficients, cos_theta):
            # Build coefficient array for legval
            legendre_coeffs = np.zeros(len(coefficients))
            
            for l in range(len(legendre_coeffs)):
                # Weight factor is (2l+1)/2 for the expansion
                weight = (2*l + 1) / 2
                legendre_coeffs[l] = weight * coefficients[l]
            
            # Evaluate at cos_theta
            return legval(cos_theta, legendre_coeffs)
        
        # Fill lookup table
        for i, energy in enumerate(energy_grid):
            # Extract coefficients for this energy
            # a0 term is always 1
            coefficients = np.append(np.array([1.0]), self.differential_xs_data[1:, i])
            
            # Remove nan values
            coefficients = coefficients[(~np.isnan(coefficients)) & (coefficients != 0)]
            
            # Get energy cross section
            sigma_E = self.get_nh_cross_section(energy/1e6)
            
            # Evaluate for all cosine theta values
            for j, cos_theta in enumerate(cos_theta_grid):
                legendre_coefficient = _evaluate_legendre(coefficients, cos_theta)
                xs_lookup[i, j] = sigma_E / (2 * np.pi) * legendre_coefficient
        
        # Create interpolation object for fast lookups
        self.diff_xs_interpolator = RectBivariateSpline(
            energy_grid, cos_theta_grid, xs_lookup, 
            kx=1, ky=1, s=0  # Linear interpolation, no smoothing
        )
    
    def get_differential_xs_CM(self, energy_MeV: float, cos_theta_cm: float | np.ndarray) -> np.ndarray:
        """
        Get center-of-mass differential scattering cross section using interpolator.
        
        Args:
            energy_MeV: Incident neutron energy in MeV
            cos_theta_cm: Cosine of CM scattering angle
        Returns:
            Differential cross section
        """
        energy_eV = energy_MeV * 1e6
        diff_xs = self.diff_xs_interpolator(energy_eV, cos_theta_cm)
        if diff_xs.shape[0] == 1:
            return diff_xs.flatten()
        else:
            return diff_xs
    
    def calculate_differential_xs_lab(self, theta_lab: float | np.ndarray, energy_MeV: float) -> np.ndarray:
        """
        Calculate lab-frame differential scattering cross section.
        From https://doi.org/10.1063/1.1721536
        P. F. Zweifel; H. Hurwitz, Jr. Tranformation of Scattering Cross Sections. J. Appl. Phys. 25, 1241–1245 (1954)
        
        Args:
            theta_lab: Lab-frame scattering angle in radians
            energy_MeV: Incident neutron energy in MeV
            
        Returns:
            Lab-frame differential cross section
        """
        cos_theta_cm = 1 - 2 * np.cos(theta_lab)**2
        return 4 * np.cos(theta_lab) * self.get_differential_xs_CM(energy_MeV, cos_theta_cm)
    
    def generate_scattered_hydron(
        self, 
        neutron_energy: float, 
        include_kinematics: bool = False,
        include_stopping_power_loss: bool = False,
        num_angle_samples: int = 10000,
        z_sampling: Literal['exp', 'uni'] = 'exp',
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[float, float, float, float, float]:
        """
        Generate a scattered hydron from neutron interaction.
        
        Args:
            neutron_energy: Incident neutron energy in MeV
            include_kinematics: Include cos^2θ energy loss in scattering
            include_stopping_power_loss: Include SRIM energy loss calculation
            num_angle_samples: Number of scattering angle samples
            z_sampling: Depth sampling method ('exp' or 'uni')
            rng: Random number generator to use (for thread safety)
            
        Returns:
            Tuple of (x0, y0, theta_scatter, phi_scatter, final_energy)
        """
        # Use provided RNG or default
        if rng is None:
            rng = np.random.default_rng()
            
        # Limit scattering angles for computational efficiency
        max_angle = np.arctan((self.foil_radius + self.aperture_radius) / self.aperture_distance)
        cos_scatter_angles = np.linspace(1, np.cos(max_angle), num_angle_samples)
        scatter_angles = np.arccos(cos_scatter_angles)
        
        # Calculate differential cross section weights
        diff_xs = self.calculate_differential_xs_lab(scatter_angles, neutron_energy)
        diff_xs /= np.sum(diff_xs)  # Normalize
        
        # Set up z-sampling probability
        if z_sampling == 'exp':
            total_cross_section = (self.get_nh_cross_section(neutron_energy) * self.hydron_density +
                                self.get_nc12_cross_section(neutron_energy) * self.carbon_density)
            prob_z = np.exp(-(self.z_grid + self.thickness) * total_cross_section)
        else:  # uniform
            prob_z = np.ones_like(self.z_grid)
            
        prob_z_cdf = np.cumsum(prob_z) / np.sum(prob_z)
        
        # Generate rays until one passes through aperture
        accepted = False
        while not accepted:
            # Sample initial position in foil
            radius_sample = self.foil_radius * np.sqrt(rng.random())
            angle_sample = 2 * np.pi * rng.random()
            x0 = radius_sample * np.cos(angle_sample)
            y0 = radius_sample * np.sin(angle_sample)
            z0 = self.z_grid[np.searchsorted(prob_z_cdf, rng.random())]
            
            # Sample scattering angles
            phi_scatter = 2 * np.pi * rng.random()
            cos_scatter_angle = rng.choice(cos_scatter_angles, p=diff_xs)
            theta_scatter = np.arccos(cos_scatter_angle)
            
            # Adjust initial coordinates for transport through foil
            x0 += z0 * np.tan(theta_scatter) * np.cos(phi_scatter)
            y0 += z0 * np.tan(theta_scatter) * np.sin(phi_scatter)
            
            # Check if hydron passes through aperture
            if self._check_aperture_acceptance(x0, y0, theta_scatter, phi_scatter):
                # Convert neutron energy to recoil hydron energy
                # Not assuming any kinematic or stopping power losses (yet)
                mass_coefficient = 4 * self.hydron_mass * NEUTRON_MASS / (self.hydron_mass + NEUTRON_MASS)**2
                final_energy = mass_coefficient * neutron_energy
                
                # Apply kinematic energy loss
                if include_kinematics:
                    final_energy *= np.cos(theta_scatter)**2
                
                # Apply stopping power energy loss
                if include_stopping_power_loss:
                    path_length = (-z0) / np.cos(theta_scatter)
                    final_energy = self.calculate_stopping_power_loss(final_energy, path_length)
                
                accepted = True
                
        return x0, y0, theta_scatter, phi_scatter, final_energy
    
    def _check_aperture_acceptance(
        self, 
        x0: float, 
        y0: float, 
        theta_scatter: float, 
        phi_scatter: float
    ) -> bool:
        """
        Check if a scattered hydron passes through the aperture.
        
        Args:
            x0, y0: Initial position in foil
            theta_scatter, phi_scatter: Scattering angles
            
        Returns:
            True if hydron passes through aperture
        """
        # Calculate position at aperture
        x_aperture = x0 + self.aperture_distance * np.tan(theta_scatter) * np.cos(phi_scatter)
        y_aperture = y0 + self.aperture_distance * np.tan(theta_scatter) * np.sin(phi_scatter)
        
        if self.aperture_type == 'circ':
            return (x_aperture**2 + y_aperture**2) <= self.aperture_radius**2
        # TODO: expand rectangular aperture to have two dimensions
        elif self.aperture_type == 'rect':
            return (np.abs(x_aperture) <= self.aperture_radius and 
                   np.abs(y_aperture) <= self.aperture_radius)
        else:
            raise ValueError(f"Unsupported aperture type: {self.aperture_type}")
    
    def calculate_efficiency(
        self, 
        neutron_energy: float, 
        num_samples: int = int(1e6),
        num_angle_samples: int = 10000,
        max_workers: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """
        Estimate intrinsic efficiency (hydrons/neutron) of the spectrometer using parallel processing.
        
        Args:
            neutron_energy: Incident neutron energy in MeV
            num_samples: Number of particles to simulate
            num_angle_samples: Angular discretization steps
            max_workers: Maximum number of worker processes (None for CPU count)
            
        Returns:
            Tuple of scattering, geometric, and total efficiency as fraction of incident neutrons
        """
        if max_workers is None:
            max_workers = mp.cpu_count()
        
        print(f'Estimating intrinsic efficiency for {neutron_energy:.3f} MeV neutrons using {max_workers} processes...')
        
        # Calculate scattering probability in foil (non-parallelizable part)
        nh_xs = self.get_nh_cross_section(neutron_energy)
        nc12_xs = self.get_nc12_cross_section(neutron_energy)
        total_xs = nh_xs * self.hydron_density + nc12_xs * self.carbon_density
        
        scattering_efficiency = (self.hydron_density * nh_xs * 
                            (1 - np.exp(-total_xs * self.thickness)) / total_xs)
        
        # Prepare angular distributions
        cos_scatter_angles = np.linspace(1, 0, num_angle_samples)
        scatter_angles = np.arccos(cos_scatter_angles)
        diff_xs = self.calculate_differential_xs_lab(scatter_angles, neutron_energy)
        diff_xs /= np.sum(diff_xs)  # Normalize
        
        # Calculate samples per process
        samples_per_process = num_samples // max_workers
        remaining_samples = num_samples % max_workers
        
        # Create shared counter for progress tracking
        manager = mp.Manager()
        progress_counter = manager.Value('i', 0)
        progress_lock = manager.Lock()
        
        # Initialize progress bar
        pbar = tqdm(total=num_samples, desc='Calculating geometric acceptance')
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit jobs
            for i in range(max_workers):
                batch_size = samples_per_process + (1 if i < remaining_samples else 0)
                if batch_size > 0:  # Only submit if there's work to do
                    # Package all parameters for the worker
                    worker_args = (
                        batch_size,
                        12345 + i * 1000,  # seed_offset
                        cos_scatter_angles,
                        diff_xs,
                        self.foil_radius,
                        progress_counter,
                        progress_lock
                    )
                    future = executor.submit(self._calculate_efficiency_batch_worker, *worker_args)
                    futures.append(future)
            
            # Monitor progress while processes run
            last_count = 0
            while any(not future.done() for future in futures):
                current_count = progress_counter.value
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                time.sleep(0.5)  # Check every 500ms
            
            # Final update for any remaining progress
            final_count = progress_counter.value
            if final_count > last_count:
                pbar.update(final_count - last_count)
            
            # Collect results
            total_accepted = 0
            total_processed = 0
            
            for future in as_completed(futures):
                accepted_count, processed_count = future.result()
                total_accepted += accepted_count
                total_processed += processed_count
        
        pbar.close()
        
        # Calculate final efficiencies
        geometric_efficiency = total_accepted / total_processed if total_processed > 0 else 0.0
        total_efficiency = scattering_efficiency * geometric_efficiency
        
        print(f'Processed {total_processed} samples using {max_workers} processes')
        print(f'Scattering efficiency: {scattering_efficiency:.2e}')
        print(f'Geometric efficiency: {geometric_efficiency:.2e}') 
        print(f'Total efficiency: {total_efficiency:.2e}')
        
        return scattering_efficiency, geometric_efficiency, total_efficiency
    
    def _calculate_efficiency_batch_worker(
        self,
        batch_size: int,
        seed_offset: int,
        cos_scatter_angles: np.ndarray,
        diff_xs: np.ndarray,
        foil_radius: float,
        progress_counter,
        progress_lock
    ) -> Tuple[int, int]:
        """
        Generate a batch of efficiency samples in a separate process.
        
        Args:
            batch_size: Number of samples to process in this batch
            seed_offset: Random seed offset for this worker
            cos_scatter_angles: Array of cosine scatter angles
            diff_xs: Differential cross section weights (normalized)
            foil_radius: Foil radius in meters
            progress_counter: Shared counter for progress tracking
            progress_lock: Lock for thread-safe progress updates
            
        Returns:
            Tuple of (accepted_count, processed_count)
        """
        # Create independent random number generator
        rng = np.random.default_rng(seed_offset)
        
        accepted_count = 0
        processed_count = 0
        update_interval = max(1, batch_size // 100)  # Update progress every 1%
        
        for i in range(batch_size):
            try:
                processed_count += 1
                
                # Sample random position in foil
                radius = foil_radius * np.sqrt(rng.random())
                angle = 2 * np.pi * rng.random()
                x0 = radius * np.cos(angle)
                y0 = radius * np.sin(angle)
                
                # Sample scattering angles
                phi_scatter = 2 * np.pi * rng.random()
                cos_scatter_angle = rng.choice(cos_scatter_angles, p=diff_xs)
                theta_scatter = np.arccos(cos_scatter_angle)
                
                # Check aperture acceptance using the same logic as the original
                if self._check_aperture_acceptance(x0, y0, theta_scatter, phi_scatter):
                    accepted_count += 1
                
                # Update progress counter thread-safely at intervals
                if i % update_interval == 0:
                    with progress_lock:
                        progress_counter.value += update_interval
                        
            except Exception:
                # Skip failed calculations but still count as processed
                pass
        
        # Final update for any remaining samples
        remaining = batch_size % update_interval
        if remaining > 0:
            with progress_lock:
                progress_counter.value += remaining
        
        return accepted_count, processed_count
    
    def get_proton_energy_distribution(
        self, 
        neutron_energies: np.ndarray, 
        energy_distribution: np.ndarray, 
        num_protons: int = int(1e2)
    ) -> np.ndarray:
        """
        Calculate the proton energy distribution at foil exit for a given neutron energy distribution.
        
        Args:
            neutron_energies: Array of neutron energies in MeV
            energy_distribution: Distribution of neutron energies (will be normalized)
            num_protons: Number of protons to simulate
            
        Returns:
            Array of proton energies at foil exit
        """
        proton_energies = np.zeros(num_protons)
        
        # Weight distribution by n-p scattering cross section and normalize
        weighted_distribution = energy_distribution * self.get_nh_cross_section(neutron_energies)
        weighted_distribution = weighted_distribution / np.sum(weighted_distribution)
        
        for i in tqdm(range(num_protons), desc='Calculating proton energy distribution...'):
            # Sample neutron energy from weighted distribution
            neutron_energy = np.random.choice(neutron_energies, p=weighted_distribution)
            
            # Generate scattered proton and extract final energy
            _, _, _, _, proton_energy = self.generate_scattered_hydron(
                neutron_energy, 
                include_kinematics=True, 
                include_stopping_power_loss=True
            )
            
            proton_energies[i] = proton_energy
            
        return proton_energies
    
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
            figure_directory = '.'
        if filename_prefix is None:
            filename_prefix = f'{figure_directory}/foil_{self.particle}'
        else:
            filename_prefix = f'{figure_directory}/{filename_prefix}'
        
        # ========== Plot 1: Differential Cross Section vs Lab Angle ==========
        fig, ax = plt.subplots(figsize=(5, 4))
        
        cos_angles = np.linspace(np.cos(angle_range[0]), np.cos(angle_range[1]), num_angles)
        angles_rad = np.arccos(cos_angles)
        angles_deg = np.degrees(angles_rad)
        
        diff_xs_lab = self.calculate_differential_xs_lab(angles_rad, energy_MeV)
        
        ax.plot(angles_deg, diff_xs_lab * 1e28, 'b-', linewidth=2)
        ax.set_xlabel('Angle [deg]')
        ax.set_ylabel('d$\sigma$/d$\Omega$ [barns/sr]')
        ax.set_title(f'Differential Cross Section - {self.particle.capitalize()} at {energy_MeV:.1f} MeV')
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
        nh_energies_eV = self.nh_cross_section_data[0]
        nh_energies_MeV = nh_energies_eV * 1e-6  # Convert eV to MeV
        nh_xs_barns = self.nh_cross_section_data[1]  # Already in barns
        
        # n-C12 cross section data
        nc12_energies_eV = self.nc12_cross_section_data[0]
        nc12_idx = (nc12_energies_eV >= np.min(nh_energies_eV)) & (nc12_energies_eV <= np.max(nh_energies_eV))
        nc12_energies_MeV = nc12_energies_eV[nc12_idx] * 1e-6  # Convert eV to MeV
        nc12_xs_barns = self.nc12_cross_section_data[1, nc12_idx]  # Already in barns
        
        ax.plot(nh_energies_MeV, nh_xs_barns, 'r-', linewidth=2, 
                label=f'n-{self.particle[0]} elastic')
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
        srim_energies_MeV = self.srim_data[0]  # Already in MeV
        srim_stopping_power = self.srim_data[1] + self.srim_data[2]  # Electronic + nuclear stopping
        
        ax.plot(srim_energies_MeV, srim_stopping_power, 'purple', linewidth=2)
        
        ax.set_title(f'{self.particle.capitalize()} in {self.foil_material}')
        ax.set_xlabel(f'{self.particle.capitalize()} Energy [MeV]')
        ax.set_ylabel('Stopping Power [MeV/mm]')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        filename = f'{filename_prefix}_stopping_power.png'
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Stopping power plot saved to {filename}')