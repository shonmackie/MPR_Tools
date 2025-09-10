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

from typing import Tuple, Optional, Literal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from progress.bar import Bar

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
        srim_data_path: str,
        nh_cross_section_path: str,
        nc12_cross_section_path: str,
        differential_xs_path: str,
        foil_density: float = 0.98, # g/cm³
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
        print('Initializing acceptance geometry...')
        
        # Convert units and store geometry
        self.foil_radius = foil_radius * 1e-2  # cm to m
        self.thickness = thickness * 1e-6      # μm to m
        self.aperture_distance = aperture_distance * 1e-2  # cm to m
        self.aperture_radius = aperture_radius * 1e-2      # cm to m
        self.aperture_type = aperture_type
        
        # Calculate particle densities in CH2
        if foil_material == 'CH2':
            self.particle = 'proton'
            molecular_weight = 14.0266 # g/mol for CH2
            self.hydron_mass = 1.00728 # amu
        elif foil_material == 'CD2':
            self.particle = 'deuteron'
            molecular_weight = 16.0395 # g/mol for CD2
            self.hydron_mass = 2.0136 # amu
        
        density_factor = foil_density * self.AVOGADRO * 1e6
        self.carbon_density = density_factor / molecular_weight # carbon/m³
        self.hydron_density = self.carbon_density * 2 # hydrons/m³
        
        # Initialize sampling grids
        # Exit of the foil is z=0
        self.z_grid = np.linspace(-self.thickness, 0, z_grid_points)
        
        # Load cross section and stopping power data
        self.srim_data_path = srim_data_path
        self.nh_cross_section_path = nh_cross_section_path
        self.nc12_cross_section_path = nc12_cross_section_path
        self.differential_xs_path = differential_xs_path
        self._load_data_files()
        
        print('Conversion foil initialization complete.\n')
    
    def _load_data_files(self) -> None:
        """Load all required data files."""
        self.srim_data = np.genfromtxt(self.srim_data_path, unpack=True)
        print(f'Loaded SRIM data from {self.srim_data_path}')
        
        self.nh_cross_section_data = np.genfromtxt(self.nh_cross_section_path, unpack=True, usecols=(0, 1))
        print(f'Loaded n-{self.particle} elastic scattering cross sections from {self.nh_cross_section_path}')
        
        self.nc12_cross_section_data = np.genfromtxt(self.nc12_cross_section_path, unpack=True, usecols=(0, 1))
        print(f'Loaded n-C12 elastic scattering cross sections from {self.nc12_cross_section_path}')
        
        self.differential_xs_data = np.genfromtxt(self.differential_xs_path, unpack=True)
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
            self.srim_data[1] + self.srim_data[2]
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
            Cross section in m²
        """
        energy_eV = energy_MeV * 1e6
        cross_section_barns = np.interp(energy_eV, self.nh_cross_section_data[0], 
                                       self.nh_cross_section_data[1])
        return cross_section_barns * 1e-28  # barns to m²
    
    def get_nc12_cross_section(self, energy_MeV: float) -> float:
        """
        Get n-C12 elastic scattering cross section.
        
        Args:
            energy_MeV: Incident neutron energy in MeV
            
        Returns:
            Cross section in m²
        """
        energy_eV = energy_MeV * 1e6
        cross_section_barns = np.interp(energy_eV, self.nc12_cross_section_data[0], 
                                       self.nc12_cross_section_data[1])
        return cross_section_barns * 1e-28  # barns to m²
    
    def calculate_differential_xs_CM(self, energy_MeV: float, cos_theta_cm: float) -> float:
        """
        Calculate center-of-mass differential scattering cross section using Legendre expansion.
        
        Args:
            energy_MeV: Incident neutron energy in MeV
            cos_theta_cm: Cosine of CM scattering angle
            
        Returns:
            Differential cross section
        """
        energy_eV = energy_MeV * 1e6
        
        # Extract Legendre coefficients
        coefficients = []
        for i in range(1, 7):  # l1 through l6
            coeff = np.interp(energy_eV, self.differential_xs_data[0], 
                             self.differential_xs_data[i])
            coefficients.append(coeff)
        
        l1, l2, l3, l4, l5, l6 = coefficients
        mu = cos_theta_cm
        
        # Legendre polynomial expansion
        result = (0.5 + 
                 1.5 * l1 * mu +
                 2.5 * l2 * 0.5 * (3*mu**2 - 1) +
                 3.5 * l3 * 0.5 * (5*mu**3 - 3*mu) +
                 4.5 * l4 * 0.125 * (35*mu**4 - 30*mu**2 + 3) +
                 5.5 * l5 * 0.125 * (63*mu**5 - 70*mu**3 + 15*mu) +
                 6.5 * l6 * 0.0625 * (231*mu**6 - 315*mu**4 + 105*mu**2 - 5))
        
        return result
    
    def calculate_differential_xs_lab(self, theta_lab: float, energy_MeV: float) -> float:
        """
        Calculate lab-frame differential scattering cross section.
        
        Args:
            theta_lab: Lab-frame scattering angle in radians
            energy_MeV: Incident neutron energy in MeV
            
        Returns:
            Lab-frame differential cross section
        """
        cos_theta_cm = 1 - 2 * np.cos(theta_lab)**2
        return 4 * np.cos(theta_lab) * self.calculate_differential_xs_CM(energy_MeV, cos_theta_cm)
    
    def generate_scattered_hydron(
        self, 
        neutron_energy: float, 
        include_kinematics: bool = False,
        include_stopping_power_loss: bool = False,
        num_angle_samples: int = 10000,
        z_sampling: Literal['exp', 'uni'] = 'exp'
    ) -> Tuple[float, float, float, float, float]:
        """
        Generate a scattered hydron from neutron interaction.
        
        Args:
            neutron_energy: Incident neutron energy in MeV
            include_kinematics: Include cos²θ energy loss in scattering
            include_stopping_power_loss: Include SRIM energy loss calculation
            num_angle_samples: Number of scattering angle samples
            z_sampling: Depth sampling method ('exp' or 'uni')
            
        Returns:
            Tuple of (x0, y0, theta_scatter, phi_scatter, final_energy)
        """
        # Limit scattering angles for computational efficiency
        max_angle = np.arctan((self.foil_radius + self.aperture_radius) / self.aperture_distance)
        scatter_angles = np.linspace(0, max_angle, num_angle_samples)
        
        # Calculate differential cross section weights
        diff_xs = self.calculate_differential_xs_lab(scatter_angles, neutron_energy)
        diff_xs_cdf = np.cumsum(diff_xs) / np.sum(diff_xs)
        
        # Set up z-sampling probability
        if z_sampling == 'exp':
            total_cross_section = (self.get_nh_cross_section(neutron_energy) * self.hydron_density +
                                 self.get_nc12_cross_section(neutron_energy) * self.carbon_density)
            prob_z = np.exp(-(self.z_grid + self.thickness) * total_cross_section)
        else:  # uniform
            prob_z = np.ones_like(self.z_grid)
            
        prob_z_cdf = np.cumsum(prob_z) / np.sum(prob_z)
        
        # Generate rays until one passes through aperture
        while True:
            # Sample initial position in foil
            radius_sample = self.foil_radius * np.sqrt(np.random.rand())
            angle_sample = 2 * np.pi * np.random.rand()
            x0 = radius_sample * np.cos(angle_sample)
            y0 = radius_sample * np.sin(angle_sample)
            z0 = self.z_grid[np.searchsorted(prob_z_cdf, np.random.rand())]
            
            # Sample scattering angles
            phi_scatter = 2 * np.pi * np.random.rand()
            theta_scatter = scatter_angles[np.searchsorted(diff_xs_cdf, np.random.rand())]
            
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
        num_samples: int = int(1e5),
        num_angle_samples: int = 10000
    ) -> float:
        """
        Estimate intrinsic efficiency (hydrons/neutron) of the spectrometer.
        
        Args:
            neutron_energy: Incident neutron energy in MeV
            num_samples: Number of particles to simulate
            num_angle_samples: Angular discretization steps
            
        Returns:
            Efficiency as fraction of incident neutrons
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
        scatter_angles = np.linspace(0, np.pi/2, num_angle_samples)
        diff_xs = self.calculate_differential_xs_lab(scatter_angles, neutron_energy)
        diff_xs /= np.sum(diff_xs)  # Normalize
        
        accepted_count = 0
        progress_bar = Bar('Calculating acceptance...', max=num_samples)
        
        for _ in range(num_samples):
            # Sample random position in foil
            radius = self.foil_radius * np.sqrt(np.random.rand())
            angle = 2 * np.pi * np.random.rand()
            x0 = radius * np.cos(angle)
            y0 = radius * np.sin(angle)
            
            # Sample scattering angles
            phi_scatter = 2 * np.pi * np.random.rand()
            theta_scatter = np.random.choice(scatter_angles, p=diff_xs)
            
            if self._check_aperture_acceptance(x0, y0, theta_scatter, phi_scatter):
                accepted_count += 1
                
            progress_bar.next()
        
        progress_bar.finish()
        
        geometric_efficiency = accepted_count / num_samples
        total_efficiency = scattering_efficiency * geometric_efficiency
        
        print(f'Scattering efficiency: {scattering_efficiency:.4f}')
        print(f'Geometric efficiency: {geometric_efficiency:.4f}') 
        print(f'Total efficiency: {total_efficiency:.4f}')
        
        return total_efficiency


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
            detector_width: Detector width in cm
            detector_height: Detector height in cm
        """
        self.channels_left = channels_left
        self.channels_right = channels_right
        self.total_channels = channels_left + channels_right + 1  # +1 for central channel
        
        self.detector_width = detector_width * 1e-2   # cm to m
        self.detector_height = detector_height * 1e-2  # cm to m
        
        # Calculate detector centers
        self._calculate_detector_centers()
    
    def _calculate_detector_centers(self) -> None:
        """Calculate the center positions of all detectors."""
        start_pos = -(self.channels_left + 0.5) * self.detector_width
        end_pos = (self.channels_right + 0.5) * self.detector_width
        self.detector_centers = np.linspace(start_pos, end_pos, self.total_channels)
    
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
        self._calculate_detector_centers()
    
    def set_detector_height(self, height_cm: float) -> None:
        """Set detector height in cm."""
        self.detector_height = height_cm * 1e-2
    
    def get_detector_centers(self) -> np.ndarray:
        """Get array of detector center positions in meters."""
        return self.detector_centers
    
    def get_channel_for_position(self, x_position: float) -> Optional[int]:
        """
        Get the detector channel number for a given x position.
        
        Args:
            x_position: X position in meters
            
        Returns:
            Channel number (0-indexed) or None if outside detector array
        """
        for i, center in enumerate(self.detector_centers):
            left_edge = center - self.detector_width / 2
            right_edge = center + self.detector_width / 2
            
            if left_edge <= x_position < right_edge:
                return i
                
        return None


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
        figure_directory: str = './'
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
        
        self.input_beam = np.zeros((num_rays, 5))
        print(f'Characteristic ray energy range: {min_energy:.3f}-{max_energy:.3f} MeV')
        
        progress_bar = Bar(f'Generating {num_rays} characteristic rays...', max=num_rays)
        
        ray_index = 0
        duplicates = 0
        
        # Energy loop
        for energy_offset in energy_offset_values:
            
            if radial_points == 0:
                # On-axis ray only
                self.input_beam[ray_index] = [0, 0, 0, 0, energy_offset]
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
                                ray = [x_foil, -angle_x, y_foil, -angle_y, energy_offset]
                                is_duplicate = False
                                
                                for prev_idx in range(ray_index):
                                    if np.allclose(self.input_beam[prev_idx], ray):
                                        duplicates += 1
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    self.input_beam[ray_index] = ray
                                    ray_index += 1
                                
                                progress_bar.next()
        
        progress_bar.finish()
        print(f'Generated {ray_index} unique rays')
        print(f'Found {duplicates} duplicate rays')
    
    def generate_monte_carlo_rays(
        self,
        neutron_energies: np.ndarray,
        energy_distribution: np.ndarray,
        num_hydrons: int,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        z_sampling: Literal['exp', 'uni'] = 'exp'
    ) -> None:
        """
        Generate hydron rays from neutron energy distribution using Monte Carlo.
        
        Args:
            neutron_energies: Array of neutron energies in MeV
            energy_distribution: Relative probability distribution (normalized automatically)
            num_hydrons: Number of hydrons to simulate
            include_kinematics: Include kinematic energy transfer
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            z_sampling: Depth sampling method ('exp' or 'uni')
        """
        progress_bar = Bar('Generating Monte Carlo hydron trajectories...', max=num_hydrons)
        
        self.input_beam = np.zeros((num_hydrons, 5))
        
        # Weight energy distribution by n-h scattering cross section
        # This is not fully correct, but is sufficient for these calculations.
        # The fully correct solution would be to sample from the energy distribution, and then
        # perform a monte carlo progression where the cross section is turned into a normalized distribution,
        # and if a random number is less than the value of the distribution at E, generate a proton.
        # This is very inefficienct, so we go with this method instead.
        weighted_distribution = (energy_distribution * 
                               self.conversion_foil.get_nh_cross_section(neutron_energies))
        weighted_distribution /= np.sum(weighted_distribution)
        
        generated_count = 0
        total_attempts = 0
        
        while generated_count < num_hydrons:
            # Sample neutron energy
            neutron_energy = np.random.choice(neutron_energies, p=weighted_distribution)
            
            # Generate scattered hydron
            try:
                x0, y0, theta_s, phi_s, hydron_energy = self.conversion_foil.generate_scattered_hydron(
                    neutron_energy, include_kinematics, include_stopping_power_loss, z_sampling=z_sampling
                )
                
                if self.conversion_foil._check_aperture_acceptance(x0, y0, theta_s, phi_s):
                    # Convert to spectrometer coordinates
                    x_aperture = x0 + self.conversion_foil.aperture_distance * np.tan(theta_s) * np.cos(phi_s)
                    y_aperture = y0 + self.conversion_foil.aperture_distance * np.tan(theta_s) * np.sin(phi_s)
                    
                    angle_x = np.arctan((x_aperture - x0) / self.conversion_foil.aperture_distance)
                    angle_y = np.arctan((y_aperture - y0) / self.conversion_foil.aperture_distance)
                    
                    # Store relative energy
                    energy_relative = (hydron_energy - self.reference_energy) / self.reference_energy
                    
                    self.input_beam[generated_count] = [x0, angle_x, y0, angle_y, energy_relative]
                    generated_count += 1
                    progress_bar.next()
                
            except Exception as e:
                # Handle generation failures gracefully
                continue
            
            total_attempts += 1
            
            # Prevent infinite loops
            if total_attempts > num_hydrons * 10:
                print(f"Warning: High rejection rate. Generated {generated_count}/{num_hydrons} hydrons")
                break
        
        progress_bar.finish()
        print(f'Generated {generated_count} hydrons from {total_attempts} attempts')
    
    def apply_transfer_map(self, map_order: int = 5) -> None:
        """
        Apply ion optical transfer map to transport hydrons through spectrometer.
        
        Args:
            map_order: Order of transfer map to apply (1-5 typically)
        """        
        num_hydrons = len(self.input_beam)
        progress_bar = Bar(f'Applying order {map_order} transfer map...', max=num_hydrons)
        
        self.output_beam = np.zeros_like(self.input_beam)
        
        for i, input_ray in enumerate(self.input_beam):
            # Initialize output ray with input energy
            output_ray = np.array([0.0, 0.0, 0.0, 0.0, input_ray[4]])
            
            # Apply each map term
            for j, term_index in enumerate(self.transfer_map[-1]):
                term_powers = self._extract_digits(term_index)
                
                # Only include terms up to specified order
                if np.sum(term_powers) <= map_order:
                    # Calculate monomial term
                    # TODO: Why is the last term multiplied by term_powers[5], but shouldn't it be term_powers[4]?
                    monomial = np.prod([input_ray[k]**term_powers[k] for k in range(4)]) * input_ray[4]**term_powers[5]
                    
                    # Add contributions to each coordinate
                    for coord in range(4):  # x, angle_x, y, angle_y
                        output_ray[coord] += self.transfer_map[coord, j] * monomial
            
            self.output_beam[i] = output_ray
            progress_bar.next()
        
        progress_bar.finish()
        print('Transfer map applied successfully!')
        
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
    
    def analyze_monoenergetic_performance(
        self,
        neutron_energy: float,
        num_hydrons: int = 10000,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        generate_figure: bool = False,
        figure_name: str = 'default',
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
            include_stopping_power_loss
        )
        self.apply_transfer_map(map_order=5)
        
        # Analyze focal plane distribution
        x_positions = self.output_beam[:, 0]
        #TODO - consider more sophisticated method for evaluating resolution
        mean_position, std_deviation = norm.fit(x_positions)
        fwhm = 2.355 * std_deviation
        
        # Calculate energy resolution
        dispersion = self.transfer_map[0, 5]  # Assuming this is the dispersion term
        energy_resolution = 1 / (dispersion / fwhm) if fwhm > 0 else 0
        
        # Generate figure if requested
        if generate_figure:
            if figure_name == 'default':
                figure_name = (
                    f'''
                    {self.figure_directory}Monoenergetic_En{neutron_energy:.2f}MeV_
                    T{self.conversion_foil.thickness_um:.0f}um_
                    f'E0{self.reference_energy:.2f}MeV.png
                    ''')
            
            self._plot_monoenergetic_analysis(figure_name, neutron_energy, mean_position, std_deviation)
        
        if verbose:
            print('Ion Optical Image Parameters:')
            print(f'  Mean position [cm]: {mean_position * 100:.3f}')
            print(f'  Standard deviation [cm]: {std_deviation * 100:.3f}')
            print(f'  FWHM [cm]: {fwhm * 100:.3f}')
            print(f'  Energy resolution [%]: {energy_resolution * 100:.2f}')
        
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
        x_positions = self.output_beam[:, 0]
        counts, bins = np.histogram(x_positions, bins=30)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        axes[0].hist(x_positions, bins=30, alpha=0.7, density=True, label='Simulation')
        
        # Gaussian fit overlay
        x_fit = np.linspace(x_positions.min(), x_positions.max(), 100)
        gaussian_fit = norm.pdf(x_fit, mean_pos, std_dev)
        axes[0].plot(x_fit, gaussian_fit, 'r-', label='Gaussian Fit', linewidth=2)
        
        axes[0].set_xlabel('X Position [m]')
        axes[0].set_ylabel('Probability Density')
        axes[0].set_title(f'X-Position Distribution\n{neutron_energy:.1f} MeV Neutrons')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Scatter plot
        hydron_energies = self.input_beam[:, 4] * self.reference_energy + self.reference_energy
        scatter = axes[1].scatter(
            self.output_beam[:, 0], 
            self.output_beam[:, 2], 
            c=hydron_energies, 
            s=1.0, 
            cmap='viridis',
            alpha=0.6
        )
        
        plt.colorbar(scatter, ax=axes[1], label=f'{self.conversion_foil.particle.capitalize()} Energy [MeV]')
        axes[1].set_xlabel('X Position [m]')
        axes[1].set_ylabel('Y Position [m]')
        axes[1].set_title(f'Focal Plane Distribution\n{neutron_energy:.1f} MeV Neutrons')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_dispersion_curve(
        self,
        num_energies: int = 15,
        num_hydrons_per_energy: int = 100,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        output_filename: str = 'default',
        generate_figure: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate energy dispersion curve for the spectrometer.
        
        Args:
            num_energies: Number of energy points to simulate
            num_hydrons_per_energy: Number of hydrons per energy point
            include_kinematics: Include kinematic effects
            include_stopping_power_loss: Include stopping power energy loss via SRIM
            output_filename: Name for output data file
            generate_figure: Whether to generate dispersion plot
            
        Returns:
            Tuple of (energies, mean_positions, std_deviations)
        """
        print('Generating energy dispersion curve...')
        
        # Energy range
        energies = np.linspace(self.min_energy, self.max_energy, num_energies)
        
        mean_positions = np.zeros_like(energies)
        std_deviations = np.zeros_like(energies)
        
        progress_bar = Bar('Calculating dispersion...', max=num_energies)
        
        for i, energy in enumerate(energies):
            mean_pos, std_dev, _, _ = self.analyze_monoenergetic_performance(
                energy, 
                num_hydrons_per_energy, 
                include_kinematics, 
                include_stopping_power_loss,
                generate_figure=False,
                verbose=False
            )
            mean_positions[i] = mean_pos
            std_deviations[i] = std_dev
            progress_bar.next()
        
        progress_bar.finish()
        
        # Save dispersion data
        if output_filename == 'default':
            output_filename = f'{self.figure_directory}dispersion_curve'
        
        dispersion_data = np.column_stack([energies, mean_positions, std_deviations])
        np.savetxt(f'{output_filename}.txt', dispersion_data, 
                  header='Energy[MeV] MeanPosition[m] StdDeviation[m]')
        
        # Generate figure if requested
        if generate_figure:
            self._plot_dispersion_curve(f'{output_filename}.png', energies, 
                                      mean_positions, std_deviations)
        
        print(f'Dispersion curve saved to {output_filename}.txt')
        return energies, mean_positions, std_deviations
    
    def _plot_dispersion_curve(
        self, 
        filename: str, 
        energies: np.ndarray, 
        positions: np.ndarray, 
        uncertainties: np.ndarray
    ) -> None:
        """Generate dispersion curve plot."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.plot(energies, positions, 'b-', linewidth=2, label='Mean Position')
        ax.fill_between(energies, positions - uncertainties, positions + uncertainties,
                       alpha=0.3, color='blue', label='±1σ')
        
        ax.set_xlabel('Neutron Energy [MeV]')
        ax.set_ylabel(f'{self.conversion_foil.particle.capitalize()} Position [m]')
        ax.set_title('Energy Dispersion Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_focal_plane_distribution(
        self, 
        filename: str = 'default',
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
        if filename == 'default':
            filename = f'{self.figure_directory}focal_plane_distribution.png'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw hodoscope if requested
        if include_hodoscope:
            detector_width = self.hodoscope.detector_width
            detector_height = self.hodoscope.detector_height
            centers = self.hodoscope.get_detector_centers()
            
            # Draw detector boundaries
            for i, center in enumerate(centers):
                left_edge = center - detector_width / 2
                right_edge = center + detector_width / 2
                
                # Vertical lines for detector edges
                line_style = '-' if i == 0 else '--'
                line_width = 1.0 if i == 0 else 0.5
                ax.axvline(left_edge, -detector_height/2, detector_height/2, 
                          color='black', linestyle=line_style, linewidth=line_width)
                if i == len(centers) - 1:  # Last detector
                    ax.axvline(right_edge, -detector_height/2, detector_height/2,
                              color='black', linestyle='-', linewidth=1.0)
            
            # Horizontal lines for detector top/bottom
            ax.axhline(detector_height/2, centers[0] - detector_width/2, 
                      centers[-1] + detector_width/2, color='black', linewidth=1.0)
            ax.axhline(-detector_height/2, centers[0] - detector_width/2,
                      centers[-1] + detector_width/2, color='black', linewidth=1.0)
        
        # Scatter plot of hydron positions
        hydron_energies = self.input_beam[:, 4] * self.reference_energy + self.reference_energy
        scatter = ax.scatter(
            self.output_beam[:, 0], 
            self.output_beam[:, 2],
            c=hydron_energies,
            s=point_size,
            cmap='viridis',
            alpha=0.7
        )
        
        plt.colorbar(scatter, label=f'{self.conversion_foil.particle.capitalize()} Energy [MeV]')
        ax.set_xlabel('Horizontal Position [m]')
        ax.set_ylabel('Vertical Position [m]')
        ax.set_title(f'{self.conversion_foil.particle.capitalize()} Distribution in Focal Plane')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Focal plane plot saved to {filename}')
    
    def plot_phase_space_portraits(self, filename: str = 'default') -> None:
        """
        Generate phase space portrait plots.
        
        Args:
            filename: Output filename for the plot
        """
        if filename == 'default':
            filename = f'{self.figure_directory}phase_space_portraits.png'
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Phase Space Portraits', fontsize=16)
        
        # Color by hydron energy
        hydron_energies = self.input_beam[:, 4] * self.reference_energy + self.reference_energy
        
        # X-Y position plot
        scatter1 = axes[0, 0].scatter(
            self.output_beam[:, 0] * 100, self.output_beam[:, 2] * 100,
            c=hydron_energies, s=2.0, cmap='viridis', alpha=0.7
        )
        axes[0, 0].set_xlabel('X Position [cm]')
        axes[0, 0].set_ylabel('Y Position [cm]')
        axes[0, 0].set_title('X-Y Position')
        axes[0, 0].grid(True, alpha=0.3)
        
        # X position vs X angle
        scatter2 = axes[0, 1].scatter(
            self.output_beam[:, 0] * 100, self.output_beam[:, 1] * 1000,
            c=hydron_energies, s=2.0, cmap='viridis', alpha=0.7
        )
        axes[0, 1].set_xlabel('X Position [cm]')
        axes[0, 1].set_ylabel('X Angle [mrad]')
        axes[0, 1].set_title('X Position-Angle')
        axes[0, 1].grid(True, alpha=0.3)
        
        # X position vs energy
        scatter3 = axes[1, 0].scatter(
            self.output_beam[:, 0] * 100, self.input_beam[:, 4] * 100,
            c=hydron_energies, s=2.0, cmap='viridis', alpha=0.7
        )
        axes[1, 0].set_xlabel('X Position [cm]')
        axes[1, 0].set_ylabel('ΔE/E [%]')
        axes[1, 0].set_title('X Position-Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Y position vs Y angle
        scatter4 = axes[1, 1].scatter(
            self.output_beam[:, 2] * 100, self.output_beam[:, 3] * 1000,
            c=hydron_energies, s=2.0, cmap='viridis', alpha=0.7
        )
        axes[1, 1].set_xlabel('Y Position [cm]')
        axes[1, 1].set_ylabel('Y Angle [mrad]')
        axes[1, 1].set_title('Y Position-Angle')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add colorbar
        fig.colorbar(scatter1, ax=axes, label=f'{self.conversion_foil.particle.capitalize()} Energy [MeV]', shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
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