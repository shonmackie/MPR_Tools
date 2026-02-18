from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legval
from scipy.interpolate import CubicSpline, make_interp_spline

from ..config.constants import ELECTRON_REST_ENERGY, NEUTRON_MASS, CLASSICAL_ELECTRON_RADIUS


class GenericInteraction:
    def __init__(
            self,
            name: str,
            target_density: float,
            cross_section_path: Path,
    ):
        """
        Load a cross section from an ENDF data file
        
        Args:
            name: a uniquely identifying string, for plotting purposes
            target_density: Number density of the targets being interacted with in m^-3
            cross_section_path: Name of the data file with the total cross section data
        """
        self.name = name
        self.target_density = target_density
        self.cross_section_data = np.genfromtxt(cross_section_path, skip_header=6, usecols=(0, 1), unpack=True)
        self.recoil_probability = 0
        print(f'Loaded cross sections from {cross_section_path}')
    
    def get_cross_section(self, energy_MeV: float) -> float:
        """ get the total macroscopic cross section, for calculating attenuation, in m^-1 """
        energy_eV = energy_MeV * 1e6
        cross_section_barns = np.interp(
            energy_eV,
            self.cross_section_data[0],
            self.cross_section_data[1])
        cross_section_m2 = cross_section_barns * 1e-28  # barns to m^2
        return self.target_density * cross_section_m2
        
    def calculate_angular_distribution(self, input_energy: float):
        """ prepare to generate recoil particles from the given incident input particle energy """
        pass

    def get_recoil_probability(self, max_angle=np.pi) -> float:
        """ calculate the fraction of interactions that produce a valid recoil particle """
        return 0
    
    def generate_recoil_particle(self, rng: np.random.Generator, include_kinematics: bool, max_angle: float) -> tuple[float, float]:
        """ draw a recoil particle and return its energy and scattering angle """
        raise ValueError("This interaction does not produce ")


class ElasticScattering(GenericInteraction):
    def __init__(
            self,
            name: str,
            target_density: float,
            cross_section_path: Path,
            differential_xs_path: Path,
            particle_mass: float,
    ):
        """
        Initialize an elastic scattering process and load its cross sections from an ENDF data file
        
        Args:
            name: a uniquely identifying string, for plotting purposes
            target_density: Number density of the targets being scattered off of in m^-3
            cross_section_path: Name of the data file with the total cross section data
            differential_xs_path: Name of the data file with the angular distribution data
            particle_mass: Mass of the particle getting scattered in amu
        """
        super().__init__(name, target_density, cross_section_path)
        
        # Need to read as a pandas df and convert to numpy because some
        # differential cross sections have different number of Legendre
        # coefficients for each energy
        self.differential_xs_data = pd.read_csv(differential_xs_path, sep=r'\s+', comment='#').to_numpy(dtype=np.float64).T
        print(f'Loaded differential scattering data from {differential_xs_path}')
        
        self.particle_mass = particle_mass
        
        # Extract energy grid
        energies = self.differential_xs_data[0]
        # Cosine grid
        cos_theta_cm = np.linspace(-1, 1, 100)
        
        # Initialize lookup table
        sigma_cm = np.zeros((len(energies), len(cos_theta_cm)))
        
        # Fill lookup table
        for i, energy in enumerate(energies):
            # Extract coefficients for this energy
            # a0 term is always 1
            coefficients = np.append(np.array([1.0]), self.differential_xs_data[1:, i])
            
            # Remove nan values
            coefficients = coefficients[(~np.isnan(coefficients)) & (coefficients != 0)]
            
            # Get energy cross section
            sigma_E = self.get_cross_section(energy / 1e6)
            
            # Evaluate for all cosine theta values
            for j, cos_theta in enumerate(cos_theta_cm):
                legendre_coefficient = ElasticScattering._evaluate_legendre(coefficients, cos_theta)
                sigma_cm[i, j] = sigma_E / (2 * np.pi) * legendre_coefficient
        
        '''
        Convert center of mass frame to lab frame
        From Dan Casey MIT Thesis (2012), Appendix A
        '''
        # This is the angle and differential cross section for the recoil particle, NOT the scattered input particle
        cos_theta_lab_recoil = np.sqrt((1 - cos_theta_cm) / 2)
        sigma_lab = 4 * cos_theta_lab_recoil * sigma_cm
        
        # Create interpolation object for fast lookups
        self.theta_lab = np.acos(cos_theta_lab_recoil)
        self.diff_xs_recoil_interpolator = make_interp_spline(
            energies, sigma_lab, k=1, axis=0)  # Linear interpolation, no smoothing
        
        # leave these as None until we evaluate them at a specific energy
        self.input_energy = None
        self.scatter_angles = None
        self.angle_distribution = None
        self.recoil_probability = 1
    
    @staticmethod
    def _evaluate_legendre(coefficients, cos_theta):
        """ Helper function to evaluate Legendre expansion """
        # Build coefficient array for legval
        legendre_coeffs = np.zeros(len(coefficients))
        
        for l in range(len(legendre_coeffs)):
            # Weight factor is (2l+1)/2 for the expansion
            weight = (2*l + 1) / 2
            legendre_coeffs[l] = weight * coefficients[l]
        
        # Evaluate at cos_theta
        return legval(cos_theta, legendre_coeffs)
    
    def _calculate_differential_xs_lab(self, energy_MeV: float) -> np.ndarray:
        """
        Calculate lab-frame differential scattering cross section
        
        Args:
            energy_MeV: Incident input particle energy in MeV
            
        Returns:
            Lab-frame differential cross section
        """
        # Convert to energy in eV and use interpolator
        energy_eV = energy_MeV * 1e6
        return self.diff_xs_recoil_interpolator(energy_eV)*np.sin(self.theta_lab)
    
    def calculate_angular_distribution(self, input_energy: float):
        """ prepare to generate recoil particles from the given incident input particle energy """
        self.input_energy = input_energy
        
        # Calculate differential cross section weights
        self.angle_distribution = ProbabilityDistribution(
            self.theta_lab,
            self._calculate_differential_xs_lab(input_energy))
        
    def get_recoil_probability(self, max_angle=np.pi) -> float:
        """ calculate the fraction of interactions that produce a valid recoil particle """
        if max_angle >= np.pi/2:
            return 1
        elif self.angle_distribution is None:
            raise ValueError("You need to call calculate_angular_distribution() before I can start generating recoil rays.")
        else:
            return self.angle_distribution.integral(0, max_angle)
    
    def generate_recoil_particle(self, rng: np.random.Generator, include_kinematics: bool, max_angle: float) -> tuple[float, float]:
        """ draw a recoil particle and return its scattering angle (radians) and initial energy (MeV) """
        if self.angle_distribution is None:
            raise ValueError("You need to call calculate_angular_distribution() before I can start generating recoil rays.")
        
        theta_scatter = self.angle_distribution.draw(rng, upper=max_angle)
        
        # Initialize recoil energy
        recoil_energy = self.input_energy
        
        # Apply kinematic energy loss
        if include_kinematics:
            # Calculate energy loss from (382) of https://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node52.html
            gamma = NEUTRON_MASS / self.particle_mass
            recoil_energy *= 4 * gamma / (1 + gamma)**2 * np.cos(theta_scatter)**2
        
        return theta_scatter, recoil_energy


class ComptonScattering:
    def __init__(
            self,
            target_density: float,
    ):
        """
        Initialize a Compton scattering process
        
        Args:
            target_density: Electron number density in m^-3
        """
        self.name = "Compton"
        self.target_density = target_density
        self.recoil_probability = 1
        
        input_energies = np.linspace(1, 25, 25)
        self.photon_angles = np.linspace(0, np.pi, 37)
        photon_energy_grid, photon_angle_grid = np.meshgrid(input_energies, self.photon_angles, indexing='ij')
        differential_cross_section = self._get_differential_cross_section(photon_energy_grid, photon_angle_grid)
        photon_angle_distributions = differential_cross_section*np.sin(photon_angle_grid)
        
        # Create interpolation object for fast lookups
        self.photon_angle_distribution_interpolator = make_interp_spline(
            input_energies, photon_angle_distributions, k=1, axis=0)
        
        # leave these as None until we evaluate them at a specific energy
        self.input_energy = None
        self.photon_angle_distribution = None

    def get_macroscopic_cross_section(self, input_energy: float) -> float:
        """ get the total macroscopic cross section, for calculating attenuation, in m^-1 """
        if input_energy <= ELECTRON_REST_ENERGY*2:
            raise ValueError("This formula only works for photon energies much greater than the electron rest energy")
        s0 = 8/3*np.pi*CLASSICAL_ELECTRON_RADIUS**2  # m2
        a0 = input_energy / ELECTRON_REST_ENERGY
        return s0 * 3 / 8 / a0 * (np.log(2 * a0) + 0.5) * self.target_density
    
    def _get_differential_cross_section(self, input_energy: np.ndarray, photon_angle: np.ndarray):
        alpha_0 = input_energy / ELECTRON_REST_ENERGY
        cos_theta = np.cos(photon_angle)
        return (
                CLASSICAL_ELECTRON_RADIUS**2
                * (1 + cos_theta**2)
                / (1 + alpha_0 * (1 - cos_theta)) ** 2
                * (
                        1
                        + (alpha_0**2 * (1 - cos_theta) ** 2)
                        / ((1 + cos_theta**2) * (1 + alpha_0 * (1 - cos_theta)))
                )
        ) * self.target_density
    
    @staticmethod
    def _convert_to_electron_angle(input_energy: float, photon_angle: float) -> float:
        a_0 = input_energy / ELECTRON_REST_ENERGY
        electron_angle = np.arctan(
            1 / ((a_0 + 1) * np.tan(photon_angle / 2))
        )
        return electron_angle
    
    @staticmethod
    def _convert_to_photon_angle(input_energy: float, electron_angle: float) -> float:
        a_0 = input_energy / ELECTRON_REST_ENERGY
        return 2*np.arctan(
            1 / ((a_0 + 1) * np.tan(electron_angle))
        )
        
    def calculate_angular_distribution(self, input_energy: float):
        """ prepare to generate recoil particles from the given incident input particle energy """
        self.input_energy = input_energy
        self.photon_angle_distribution = ProbabilityDistribution(
            self.photon_angles, self.photon_angle_distribution_interpolator(input_energy))
    
    def get_recoil_probability(self, max_angle=np.pi):
        """ calculate the fraction of interactions that produce a valid recoil particle """
        if max_angle >= np.pi/2:
            return 1
        elif self.photon_angle_distribution is None:
            raise ValueError("You need to call calculate_angular_distribution() before I can start generating recoil rays.")
        else:
            min_photon_angle = ComptonScattering._convert_to_photon_angle(self.input_energy, max_angle)
            return self.photon_angle_distribution.integral(min_photon_angle, np.pi)

    def generate_recoil_particle(self, rng: np.random.Generator, include_kinematics: bool, max_angle: float) -> tuple[float, float]:
        """ draw a recoil particle and return its scattering angle (radians) and initial energy (MeV) """
        if self.photon_angle_distribution is None:
            raise ValueError("You need to call calculate_angular_distribution() before I can start generating recoil rays.")
        min_photon_angle = ComptonScattering._convert_to_photon_angle(self.input_energy, max_angle)
        photon_angle = self.photon_angle_distribution.draw(rng, lower=min_photon_angle)
        electron_angle = ComptonScattering._convert_to_electron_angle(self.input_energy, photon_angle)
        if include_kinematics:
            a_0 = self.input_energy / ELECTRON_REST_ENERGY
            a = a_0 / (1 + a_0 * (1 - np.cos(photon_angle)))
            electron_energy = (a_0 - a) * ELECTRON_REST_ENERGY
        else:
            electron_energy = self.input_energy - ELECTRON_REST_ENERGY/2
        return electron_angle, electron_energy


Interaction = GenericInteraction | ComptonScattering
""" an interaction between radiation and matter """


class ProbabilityDistribution:
    def __init__(self, values: np.ndarray, probability_densities: np.ndarray):
        """
        A one-dimensional probability distribution based on interpolating between some known probability densities
        
        Args:
            values: The values of the random variable at which the probability density is defined
            probability_densities: The probability density at each given value (it does not need to be normalized)
        """
        if np.any(np.diff(values) < 0):
            raise ValueError("the x-axis for a probability distribution must always be monotonically increasing.")
        self.min_value = values[0]
        self.max_value = values[-1]
        # make a piecewise polynomial out of the given PDF
        self.pdf = CubicSpline(values, probability_densities, extrapolate=False)
        # this object can then be integrated efficiently and exactly
        self.cdf = self.pdf.antiderivative()
    
    def integral(self, lower: float, upper: float):
        """
        calculate the fraction of the distribution that lies between two points
        """
        absolute_min = self.cdf(self.min_value)
        absolute_max = self.cdf(self.max_value)
        interval_min = self.cdf(max(lower, self.min_value))
        interval_max = self.cdf(min(upper, self.max_value))
        return (interval_max - interval_min)/(absolute_max - absolute_min)
    
    def draw(self, rng: np.random.Generator, lower: float = -np.inf, upper: float = np.inf):
        """
        randomly draw a number from the probability distribution
        
        Args:
            rng: the random number generator to use
            lower: pass a number to truncate the distribution to above this point
            upper: pass a number to truncate the distribution to below this point
        """
        u_min = self.cdf(max(lower, self.min_value))
        u_max = self.cdf(min(upper, self.max_value))
        u = rng.uniform(u_min, u_max)
        return self.cdf.solve(u)[0]
