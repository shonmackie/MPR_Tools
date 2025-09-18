"""
MPR Spectrometer Analysis Toolkit

A comprehensive toolkit for analyzing the performance of Magnetic Proton Recoil (MPR) spectrometers.

Main functionalities:
- Evaluate efficiency of neutron->hydron conversion and selection
- Initialize ensembles of hydrons:
  * Characteristic rays
  * Monte Carlo generation from probability distributions
  * Full synthetic neutron-hydron conversion from phase space distributions
- Transport hydrons through ion optics using COSY transfer maps
- Analyze transported hydrons:
  * Ion optical image analysis via phase portraits
  * x-y scattering in focal plane
  * Synthetic hodoscope binning

To build an MPR system, you need: foil, aperture, ion optics, and hodoscope.

Useful resources:
- ENDF Info: https://www.oecd-nea.org/dbdata/data/endf102.htm#LinkTarget_12655
- ENDF data: https://www.nndc.bnl.gov/sigma/index.jsp?as=1&lib=endfb7.1&nsub=10
"""

from typing import Tuple, Optional, Literal, List
import numpy as np
from numpy.polynomial.legendre import legval
from pathlib import Path
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from labellines import labelLines
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

def calculate_fwhm(data: np.ndarray, domain: np.ndarray) -> float:
    """
    Calculate the full width at half maximum (FWHM) of data over a given domain.
    
    Args:
        data: Array of data values
        domain: Array of domain values corresponding to data
        
    Returns:
        Full width at half maximum
    """
    half_max = np.max(data) / 2.0
    
    # Find where function crosses half_max line (sign changes)
    diff = np.sign(half_max - data[:-1]) - np.sign(half_max - data[1:])
    
    # Find leftmost and rightmost crossings
    left_indices = np.where(diff > 0)[0]
    right_indices = np.where(diff < 0)[0]
    
    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0.0
        
    left_idx = left_indices[0]
    right_idx = right_indices[-1]
    
    return domain[right_idx] - domain[left_idx]

class ConversionFoil:
    """
    Represents a conversion foil and aperture system for neutron-hydron scattering.
    
    The foil is where neutrons impinge and scatter hydrons, while the aperture
    defines the ion optical acceptance.
    """
    
    AVOGADRO = 6.022e23
    NEUTRON_MASS = 1.00867 # amu
    
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
        if foil_material == 'CH2':
            self.particle = 'proton'
            foil_density = 0.98 # g/cm^3
            molecular_weight = 14.0266 # g/mol for CH2
            self.hydron_mass = 1.00728 # amu
        elif foil_material == 'CD2':
            self.particle = 'deuteron'
            foil_density = 1.131 # g/cm^3
            molecular_weight = 16.0395 # g/mol for CD2
            self.hydron_mass = 2.0136 # amu
        
        density_factor = foil_density * self.AVOGADRO * 1e6
        self.carbon_density = density_factor / molecular_weight # carbon/m^3
        self.hydron_density = self.carbon_density * 2 # hydrons/m^3
        
        # Initialize sampling grids
        # Exit of the foil is z=0
        self.z_grid = np.linspace(-self.thickness, 0, z_grid_points)
        
        # Load cross section and stopping power data
        module_dir = Path(__file__).parent
        if srim_data_path is None:
            if foil_material == 'CH2':
                self.srim_data_path = module_dir / 'data/CH2srimdata.txt'
            elif foil_material == 'CD2':
                self.srim_data_path = module_dir / 'data/CD2srimdata.txt'
        else:
            self.srim_data_path = srim_data_path
            
        if nh_cross_section_path is None:
            if foil_material == 'CH2':
                self.nh_cross_section_path = module_dir / 'data/np_crosssection.txt'
            elif foil_material == 'CD2':
                self.nh_cross_section_path = module_dir / 'data/nd_crosssection.txt'
        else:
            self.nh_cross_section_path = nh_cross_section_path
            
        if nc12_cross_section_path is None:
            self.nc12_cross_section_path = module_dir / 'data/nC12_crosssection.txt'
        else:
            self.nc12_cross_section_path = nc12_cross_section_path
            
        if differential_xs_path is None:
            if foil_material == 'CH2':
                self.differential_xs_path = module_dir / 'data/np_diffxs.txt'
            elif foil_material == 'CD2':
                self.differential_xs_path = module_dir / 'data/nd_diffxs.txt'
        else:
            self.differential_xs_path = differential_xs_path
        self._load_data_files()
        
        # Build interpolator for differential cross section data
        self._build_differential_xs_interpolator()
        
        print('Conversion foil initialization complete.\n')
    
    def _load_data_files(self) -> None:
        """Load all required data files."""
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
                mass_coefficient = 4 * self.hydron_mass * self.NEUTRON_MASS / (self.hydron_mass + self.NEUTRON_MASS)**2
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
        num_angle_samples: int = 10000
    ) -> Tuple[float, float, float]:
        """
        Estimate intrinsic efficiency (hydrons/neutron) of the spectrometer.
        
        Args:
            neutron_energy: Incident neutron energy in MeV
            num_samples: Number of particles to simulate
            num_angle_samples: Angular discretization steps
            
        Returns:
            Tuple of scattering, geometric, and total efficiency as fraction of incident neutrons
        """
        print(f'Estimating intrinsic efficiency for {neutron_energy:.3f} MeV neutrons...')
        
        # Calculate scattering probability in foil
        nh_xs = self.get_nh_cross_section(neutron_energy)
        nc12_xs = self.get_nc12_cross_section(neutron_energy)
        # Macroscopic cross section
        total_xs = nh_xs * self.hydron_density + nc12_xs * self.carbon_density
        
        # The carbon will cause some (though negligible) attenuation as the neutron passes through the foil, so it should be included in the efficiency calculation
        scattering_efficiency = (self.hydron_density * nh_xs * 
                               (1 - np.exp(-total_xs * self.thickness)) / total_xs)
        
        # Calculate geometric acceptance
        # From 0 to pi/2
        cos_theta_scatter = np.linspace(1, 0, num_angle_samples)
        scatter_angles = np.arccos(cos_theta_scatter)
        diff_xs = self.calculate_differential_xs_lab(scatter_angles, neutron_energy)
        diff_xs /= np.sum(diff_xs)  # Normalize
        
        accepted_count = 0
        
        for _ in tqdm(range(num_samples), desc='Calculating acceptance...'):
            # Sample random position in foil
            radius = self.foil_radius * np.sqrt(np.random.rand())
            angle = 2 * np.pi * np.random.rand()
            x0 = radius * np.cos(angle)
            y0 = radius * np.sin(angle)
            
            # Sample scattering angles
            phi_scatter = 2 * np.pi * np.random.rand()
            cos_theta_scatter = np.random.choice(cos_theta_scatter, p=diff_xs)
            theta_scatter = np.arccos(cos_theta_scatter)
            
            if self._check_aperture_acceptance(x0, y0, theta_scatter, phi_scatter):
                accepted_count += 1
        
        geometric_efficiency = accepted_count / num_samples
        total_efficiency = scattering_efficiency * geometric_efficiency
        
        print(f'Scattering efficiency: {scattering_efficiency:.2e}')
        print(f'Geometric efficiency: {geometric_efficiency:.2e}') 
        print(f'Total efficiency: {total_efficiency:.2e}')
        
        return scattering_efficiency, geometric_efficiency, total_efficiency
    
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


class Hodoscope:
    """
    Detector array at the focal plane.
    
    Detectors are assumed to be centered on the final position of the reference ray.
    TODO - add functionality to load hodo config from file
    """
    
    def __init__(
        self, 
        channels_left: int, 
        channels_right: int, 
        detector_width: float, 
        detector_height: float
    ):
        """
        Initialize hodoscope detector array.
        
        Args:
            channels_left: Number of channels to the left (low energy)
            channels_right: Number of channels to the right (high energy)  
            detector_width: Total detector width in cm
            detector_height: Total detector height in cm
        """
        self.channels_left = channels_left
        self.channels_right = channels_right
        self.total_channels = channels_left + channels_right + 1  # +1 for central channel
        
        self.detector_width = detector_width * 1e-2   # cm to m
        self.detector_height = detector_height * 1e-2  # cm to m
        
        # Calculate detector centers
        self._calculate_channel_edges()
    
    def _calculate_channel_edges(self) -> None:
        """Calculate the center positions of all channels."""        
        # Calculate individual channel width
        self.channel_width = self.detector_width / self.total_channels
        
        # The central channel (index = channels_left) should be centered at x=0
        # So its left edge is at -channel_width/2 and right edge is at +channel_width/2
        central_left_edge = -self.channel_width / 2
        
        # Calculate all channel edges starting from the leftmost
        leftmost_edge = central_left_edge - self.channels_left * self.channel_width
        
        # Create array of all channel edges (N+1 edges for N channels)
        self.channel_edges = np.linspace(leftmost_edge, leftmost_edge + self.detector_width, self.total_channels + 1)
        
        # Calculate channel centers for convenience
        self.channel_centers = (self.channel_edges[:-1] + self.channel_edges[1:]) / 2
    
    @property
    def detector_width_cm(self) -> float:
        """Get detector width in cm."""
        return self.detector_width * 1e2
    
    @property
    def detector_height_cm(self) -> float:
        """Get detector height in cm."""
        return self.detector_height * 1e2
    
    def set_detector_width(self, width_cm: float) -> None:
        """Set detector width in cm."""
        self.detector_width = width_cm * 1e-2
        self._calculate_channel_edges()
    
    def set_detector_height(self, height_cm: float) -> None:
        """Set detector height in cm."""
        self.detector_height = height_cm * 1e-2
    
    def get_channel_centers(self) -> np.ndarray:
        """Get array of detector center positions in meters."""
        return self.channel_centers
    
    def get_channel_for_position(self, x_position: float) -> Optional[int]:
        """
        Get the detector channel number for a given x position.
        
        Args:
            x_position: X position in meters
            
        Returns:
            Channel number (0-indexed) or None if outside detector array
        """
        # Check if position is within hodoscope bounds
        if x_position < self.channel_edges[0] or x_position >= self.channel_edges[-1]:
            return None
        
        # Find which channel the position falls into
        channel = np.searchsorted(self.channel_edges[1:], x_position)
        return int(channel)


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
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        import time
        
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
            df = pd.DataFrame({
                'x0': self.input_beam[:, 0],
                'angle_x': self.input_beam[:, 1],
                'y0': self.input_beam[:, 2],
                'angle_y': self.input_beam[:, 3],
                'energy_relative': self.input_beam[:, 4],
                'neutron_energy': self.input_beam[:, 5]
            })
            df.to_csv(f'{self.figure_directory}/input_beam.csv', index=False)
            print('Input beam saved!')
            
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
            df = pd.DataFrame({
                'x0': self.output_beam[:, 0],
                'angle_x': self.output_beam[:, 1],
                'y0': self.output_beam[:, 2],
                'angle_y': self.output_beam[:, 3],
                'energy_relative': self.output_beam[:, 4]
            })
            df.to_csv(f'{self.figure_directory}/output_beam.csv', index=False)
            print('Output beam saved!')

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
    
    def read_beams(
        self,
        input_beam_path: Optional[str] = None,
        output_beam_path: Optional[str] = None
    ) -> None:
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
        self.generate_monte_carlo_rays(
            np.array([neutron_energy]), 
            np.array([1.0]), 
            num_hydrons, 
            include_kinematics, 
            include_stopping_power_loss,
            save_beam=False
        )
        self.apply_transfer_map(map_order=5, save_beam=False)
        
        # Analyze focal plane distribution
        x_positions = self.output_beam[:, 0]
        #TODO - consider more sophisticated method for evaluating resolution
        mean_position, std_deviation = norm.fit(x_positions)
        fwhm = 2.355 * std_deviation
        
        # Calculate energy resolution
        dispersion = self.transfer_map[0, 5]  # Assuming this is the dispersion term
        energy_resolution = self.reference_energy / (dispersion / fwhm) if fwhm > 0 else 0
        
        # Generate figure if requested
        if generate_figure:
            if figure_name == None:
                figure_name = (
                    f'{self.figure_directory}/Monoenergetic_En{neutron_energy:.1f}MeV_T{self.conversion_foil.thickness_um:.0f}um_E0{self.reference_energy:.1f}MeV.png'
                )
            
            self._plot_monoenergetic_analysis(figure_name, neutron_energy, mean_position, std_deviation)
        
        if verbose:
            print('Ion Optical Image Parameters:')
            print(f'  Mean position [cm]: {mean_position * 100:.3f}')
            print(f'  Standard deviation [cm]: {std_deviation * 100:.3f}')
            print(f'  FWHM [cm]: {fwhm * 100:.3f}')
            print(f'  Energy resolution [keV]: {energy_resolution * 1000:.2f}')
        
        return mean_position, std_deviation, fwhm, energy_resolution
    
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
        x_positions = self.output_beam[:, 0]*100 # cm
        
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
        hydron_energies = self.input_beam[:, 4] * self.reference_energy + self.reference_energy
        scatter = axes[1].scatter(
            self.output_beam[:, 0]*100,
            self.output_beam[:, 2]*100,
            c=hydron_energies,
            s=1.0,
            cmap='plasma',
            alpha=0.6
        )
        
        fig.colorbar(scatter, ax=axes[1], label=f'{self.conversion_foil.particle.capitalize()} Energy [MeV]')
        axes[1].set_xlabel('X Position [cm]')
        axes[1].set_ylabel('Y Position [cm]')
        axes[1].set_title(f'Focal Plane Distribution\n{neutron_energy:.1f} MeV Neutrons')
        axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        print(filename)
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
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
            output_filename = f'{self.figure_directory}/comprehensive_performance.csv'
        
        if reset:
            # Energy range
            energies = np.linspace(self.min_energy, self.max_energy, num_energies)
            
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
                scattering_efficiency, geometric_efficiency, total_efficiency = self.conversion_foil.calculate_efficiency(
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
            self._plot_performance(
                figure_name, 
                energies, positions_mean, positions_std, 
                energy_resolutions, total_efficiencies
            )
        
        return energies, positions_mean, positions_std, energy_resolutions, total_efficiencies

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
            filename = f'{self.figure_directory}/focal_plane_distribution.png'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw hodoscope if requested
        if include_hodoscope:
            # Convert to cm
            detector_width = self.hodoscope.detector_width * 100
            detector_height = self.hodoscope.detector_height * 100
            edges = self.hodoscope.channel_edges * 100
            
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
        hydron_energies = self.input_beam[:, 4] * self.reference_energy + self.reference_energy
        scatter = ax.scatter(
            self.output_beam[:, 0]*100, 
            self.output_beam[:, 2]*100,
            c=hydron_energies,
            s=point_size,
            cmap='plasma',
            alpha=0.7
        )
        
        fig.colorbar(scatter, label=f'{self.conversion_foil.particle.capitalize()} Energy [MeV]')
        ax.set_xlabel('Horizontal Position [cm]')
        ax.set_ylabel('Vertical Position [cm]')
        ax.set_title(f'{self.conversion_foil.particle.capitalize()} Distribution in Focal Plane')
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
            filename = f'{self.figure_directory}/phase_space.png'
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout='constrained')
        fig.suptitle('Phase Space', fontsize=16)
        
        # Color by hydron energy
        hydron_energies = self.input_beam[:, 4] * self.reference_energy + self.reference_energy
        
        # X-Y position plot
        scatter1 = axes[0, 0].scatter(
            self.output_beam[:, 0] * 100, self.output_beam[:, 2] * 100,
            c=hydron_energies, s=2.0, cmap='plasma', alpha=0.7
        )
        axes[0, 0].set_xlabel('X Position [cm]')
        axes[0, 0].set_ylabel('Y Position [cm]')
        axes[0, 0].set_title('X-Y Position')
        axes[0, 0].grid(True, alpha=0.3)
        
        # X position vs X angle
        scatter2 = axes[0, 1].scatter(
            self.output_beam[:, 0] * 100, self.output_beam[:, 1] * 1000,
            c=hydron_energies, s=2.0, cmap='plasma', alpha=0.7
        )
        axes[0, 1].set_xlabel('X Position [cm]')
        axes[0, 1].set_ylabel('X Angle [mrad]')
        axes[0, 1].set_title('X Position-Angle')
        axes[0, 1].grid(True, alpha=0.3)
        
        # X position vs energy
        scatter3 = axes[1, 0].scatter(
            self.output_beam[:, 0] * 100, self.input_beam[:, 4] * self.reference_energy + self.reference_energy,
            c=hydron_energies, s=2.0, cmap='plasma', alpha=0.7
        )
        axes[1, 0].set_xlabel('X Position [cm]')
        axes[1, 0].set_ylabel('E$_{proton}$ [MeV]')
        axes[1, 0].set_title('X Position-Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Y position vs Y angle
        scatter4 = axes[1, 1].scatter(
            self.output_beam[:, 2] * 100, self.output_beam[:, 3] * 1000,
            c=hydron_energies, s=2.0, cmap='plasma', alpha=0.7
        )
        axes[1, 1].set_xlabel('Y Position [cm]')
        axes[1, 1].set_ylabel('Y Angle [mrad]')
        axes[1, 1].set_title('Y Position-Angle')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add colorbar
        fig.colorbar(scatter1, ax=axes, label=f'{self.conversion_foil.particle.capitalize()} Energy [MeV]', shrink=0.8)
        
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Phase space portraits saved to {filename}')
    
    def bin_hodoscope_response(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin hydron hits into hodoscope channels.
        
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
    
    def get_proton_density_map(
        self, 
        dx: float = 0.01, 
        dy: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the density of proton impact sites in the focal plane.
        
        Args:
            dx: X-direction resolution in meters
            dy: Y-direction resolution in meters
            
        Returns:
            Tuple of (density_array, X_meshgrid, Y_meshgrid)
        """
        if len(self.output_beam) == 0:
            raise ValueError("No output beam data available. Run apply_transfer_map() first.")
        
        x_positions = self.output_beam[:, 0]
        y_positions = self.output_beam[:, 2]
        
        # Define grid boundaries
        x_min, x_max = np.min(x_positions), np.max(x_positions)
        y_min, y_max = np.min(y_positions), np.max(y_positions)
        
        # Create coordinate arrays
        x_coords = np.linspace(x_min, x_max, int((x_max - x_min) / dx) + 1)
        y_coords = np.linspace(y_min, y_max, int((y_max - y_min) / dy) + 1)
        
        print(f"Grid bounds: X=[{x_min:.4f}, {x_max:.4f}], Y=[{y_min:.4f}, {y_max:.4f}]")
        
        # Create meshgrids
        X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)
        density = np.zeros_like(X_mesh)
        
        # Bin protons into grid cells
        total_protons = len(self.output_beam)
        for x_pos, y_pos in zip(x_positions, y_positions):
            # Convert coordinates to grid indices
            x_idx = int((x_pos - x_min) / dx)
            y_idx = int((y_pos - y_min) / dy)
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, density.shape[1] - 1))
            y_idx = max(0, min(y_idx, density.shape[0] - 1))
            
            density[y_idx, x_idx] += 1 / total_protons
        
        return density, X_mesh, Y_mesh

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
            filename = f'{self.figure_directory}/counts_vs_position.png'
        
        if len(self.output_beam) == 0:
            raise ValueError("No output beam data available. Run apply_transfer_map() first.")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        x_positions = self.output_beam[:, 0]*100 # cm
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
            filename = f'{self.figure_directory}/input_ray_geometry.png'
        
        if len(self.input_beam) == 0:
            raise ValueError("No input beam data available. Generate rays first.")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Draw foil and aperture boundaries
        # Convert all lengths to cm
        foil_radius = self.conversion_foil.foil_radius * 100
        aperture_distance = self.conversion_foil.aperture_distance * 100
        aperture_radius = self.conversion_foil.aperture_radius * 100
        
        # Foil (vertical line at z=0)
        ax.vlines(0, -foil_radius, foil_radius, color='blue', linewidth=3, label='Conversion Foil')
        
        # Aperture (vertical line at aperture distance)
        ax.vlines(aperture_distance, -aperture_radius, aperture_radius, 
                color='red', linewidth=3, label='Aperture')
        
        # Draw sample of input rays
        num_rays_to_plot = min(len(self.input_beam), 200)  # Limit for clarity
        z_coords = np.linspace(0, aperture_distance, 20)
        
        for i in range(0, len(self.input_beam), max(1, len(self.input_beam) // num_rays_to_plot)):
            ray = self.input_beam[i]
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
            filename = f'{self.figure_directory}/proton_density_heatmap.png'
        
        density, X_mesh, Y_mesh = self.get_proton_density_map(dx, dy)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.pcolormesh(X_mesh*100, Y_mesh*100, np.log10(density), cmap='plasma', shading='auto')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('log$_10$(Proton Density [normalized])')
        
        ax.set_xlabel('X Position [cm]')
        ax.set_ylabel('Y Position [cm]')
        ax.set_title('Proton Density in Focal Plane')
        ax.set_aspect('equal')
        
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Proton density heatmap saved to {filename}')
        
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
            filename = f'{self.figure_directory}/characteristic_rays.png'
        
        # Set default energy range if not provided
        if min_energy is None:
            min_energy = self.min_energy
        if max_energy is None:
            max_energy = self.max_energy
        
        print(f'Generating characteristic rays from {min_energy:.2f} to {max_energy:.2f} MeV...')
        
        # Generate characteristic rays
        self.generate_characteristic_rays(
            radial_points=radial_points,
            angular_points=angular_points,
            aperture_radial_points=aperture_radial_points,
            aperture_angular_points=aperture_angular_points,
            energy_points=energy_points,
            min_energy=min_energy,
            max_energy=max_energy
        )
        
        # Apply transfer map
        self.apply_transfer_map(map_order=5, save_beam=False)
        
        # Create subplots
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.suptitle('Characteristic Ray Analysis', fontsize=16)
        
        # Focal plane distribution        
        # Scatter plot colored by energy
        output_energies = self.input_beam[:, 4] * self.reference_energy + self.reference_energy
        scatter = ax.scatter(
            self.output_beam[:, 0] * 100,  # Convert to cm
            self.output_beam[:, 2] * 100,  # Convert to cm
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
        print(f'  Total rays generated: {len(self.input_beam)}')
        print(f'  Energy range: {min_energy:.2f} - {max_energy:.2f} MeV')
        print(f'  X position range: {self.output_beam[:, 0].min()*100:.2f} - {self.output_beam[:, 0].max()*100:.2f} cm')
        print(f'  Y position range: {self.output_beam[:, 2].min()*100:.2f} - {self.output_beam[:, 2].max()*100:.2f} cm')
        print(f'Characteristic ray plot saved to {filename}')
    
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