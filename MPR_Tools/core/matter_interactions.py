from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legval
from scipy import integrate
from scipy.interpolate import CubicSpline, make_interp_spline

from ..config.constants import ELECTRON_REST_ENERGY, NEUTRON_MASS, CLASSICAL_ELECTRON_RADIUS, FINE_STRUCTURE_CONSTANT


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
        print(f'Loaded cross sections from {cross_section_path}')
    
    def get_cross_section(self, energy_MeV: float | np.ndarray) -> float | np.ndarray:
        """ get the total macroscopic cross section, for calculating attenuation, in m^-1 """
        energy_eV = energy_MeV * 1e6
        if np.any(energy_eV < self.cross_section_data[0][0]) or np.any(energy_eV > self.cross_section_data[0][-1]):
            raise ValueError(f"I don't have {self.name} cross section data for {energy_MeV} MeV")
        cross_section_barns = np.interp(
            energy_eV,
            self.cross_section_data[0],
            self.cross_section_data[1])
        cross_section_m2 = cross_section_barns * 1e-28  # barns to m^2
        return self.target_density * cross_section_m2
        
    def calculate_angular_distribution(self, incident_energy: float):
        """ prepare to generate recoil particles from the given incident particle energy """
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
        # This is the angle and differential cross section for the recoil particle, NOT the scattered incident particle
        cos_theta_lab_recoil = np.sqrt((1 - cos_theta_cm) / 2)
        sigma_lab = 4 * cos_theta_lab_recoil * sigma_cm
        
        # Create interpolation object for fast lookups
        self.theta_lab = np.arccos(cos_theta_lab_recoil)
        self.diff_xs_recoil_interpolator = make_interp_spline(
            energies, sigma_lab, k=1, axis=0)  # Linear interpolation, no smoothing
        
        # leave these as None until we evaluate them at a specific energy
        self.incident_energy = None
        self.scatter_angles = None
        self.angle_distribution = None
    
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
            energy_MeV: Incident particle energy in MeV
            
        Returns:
            Lab-frame differential cross section
        """
        # Convert to energy in eV and use interpolator
        energy_eV = energy_MeV * 1e6
        return self.diff_xs_recoil_interpolator(energy_eV)*np.sin(self.theta_lab)
    
    def calculate_angular_distribution(self, incident_energy: float):
        """ prepare to generate recoil particles from the given incident particle energy """
        self.incident_energy = incident_energy
        
        # Calculate differential cross section weights
        self.angle_distribution = ProbabilityDistribution(
            self.theta_lab,
            self._calculate_differential_xs_lab(incident_energy))
        
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
        recoil_energy = self.incident_energy
        
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
        
        incident_energies = np.linspace(1, 25, 25)
        self.photon_angles = np.linspace(0, np.pi, 37)
        photon_energy_grid, photon_angle_grid = np.meshgrid(incident_energies, self.photon_angles, indexing='ij')
        differential_cross_section = self._get_differential_cross_section(photon_energy_grid, photon_angle_grid)
        photon_angle_distributions = differential_cross_section*np.sin(photon_angle_grid)
        
        # Create interpolation object for fast lookups
        self.photon_angle_distribution_interpolator = make_interp_spline(
            incident_energies, photon_angle_distributions, k=1, axis=0)
        
        # leave these as None until we evaluate them at a specific energy
        self.incident_energy = None
        self.photon_angle_distribution = None

    def get_macroscopic_cross_section(self, incident_energy: float | np.ndarray) -> float | np.ndarray:
        """ get the total macroscopic cross section, for calculating attenuation, in m^-1 """
        if np.any(incident_energy <= ELECTRON_REST_ENERGY*2):
            raise ValueError("This formula only works for photon energies much greater than the electron rest energy")
        s0 = 8/3*np.pi*CLASSICAL_ELECTRON_RADIUS**2  # m2
        a0 = incident_energy / ELECTRON_REST_ENERGY
        return s0 * 3 / 8 / a0 * (np.log(2 * a0) + 0.5) * self.target_density
    
    def _get_differential_cross_section(self, incident_energy: np.ndarray, photon_angle: np.ndarray):
        alpha_0 = incident_energy / ELECTRON_REST_ENERGY
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
    def _convert_to_electron_angle(incident_energy: float, photon_angle: float) -> float:
        a_0 = incident_energy / ELECTRON_REST_ENERGY
        electron_angle = np.arctan(
            1 / ((a_0 + 1) * np.tan(photon_angle / 2))
        )
        return electron_angle
    
    @staticmethod
    def _convert_to_photon_angle(incident_energy: float, electron_angle: float) -> float:
        a_0 = incident_energy / ELECTRON_REST_ENERGY
        return 2*np.arctan(
            1 / ((a_0 + 1) * np.tan(electron_angle))
        )
        
    def calculate_angular_distribution(self, incident_energy: float):
        """ prepare to generate recoil particles from the given incident particle energy """
        self.incident_energy = incident_energy
        self.photon_angle_distribution = ProbabilityDistribution(
            self.photon_angles, self.photon_angle_distribution_interpolator(incident_energy))
    
    def get_recoil_probability(self, max_angle=np.pi):
        """ calculate the fraction of interactions that produce a valid recoil particle """
        if max_angle >= np.pi/2:
            return 1
        elif self.photon_angle_distribution is None:
            raise ValueError("You need to call calculate_angular_distribution() before I can start generating recoil rays.")
        else:
            min_photon_angle = ComptonScattering._convert_to_photon_angle(self.incident_energy, max_angle)
            return self.photon_angle_distribution.integral(min_photon_angle, np.pi)

    def generate_recoil_particle(self, rng: np.random.Generator, include_kinematics: bool, max_angle: float) -> tuple[float, float]:
        """ draw a recoil particle and return its scattering angle (radians) and initial energy (MeV) """
        if self.photon_angle_distribution is None:
            raise ValueError("You need to call calculate_angular_distribution() before I can start generating recoil rays.")
        min_photon_angle = ComptonScattering._convert_to_photon_angle(self.incident_energy, max_angle)
        photon_angle = self.photon_angle_distribution.draw(rng, lower=min_photon_angle)
        electron_angle = ComptonScattering._convert_to_electron_angle(self.incident_energy, photon_angle)
        if include_kinematics:
            a_0 = self.incident_energy / ELECTRON_REST_ENERGY
            a = a_0 / (1 + a_0 * (1 - np.cos(photon_angle)))
            electron_energy = (a_0 - a) * ELECTRON_REST_ENERGY
        else:
            electron_energy = self.incident_energy - ELECTRON_REST_ENERGY/2
        return electron_angle, electron_energy
    
    
class PairProduction:
    def __init__(self, target_density: float, charge: int):
        """
        Initialize a pair production process for a given matter particle,
        and calculate the double-differential cross section
        
        Args:
            target_density: Number density of the interacting particle in m^-3
            charge: Charge number of the interacting particle
        """
        self.name = "pair production"
        self.target_density = target_density
        
        # photon energies at which to calculate the cross section
        self.incident_energy_table = np.geomspace(2*ELECTRON_REST_ENERGY, 25, 11)  # MeV
        # electron parameters at which to calculate the cross section
        self.energy_fraction_table = np.linspace(0, 1, 51)
        self.angle_CM_table = np.linspace(0, np.pi/2, 51)
        # positron parameters to integrate over when calculating the cross section
        positron_angle_CM_table = np.linspace(0, np.pi/2, 51)
        dihedral_angle_table = np.linspace(0, np.pi, 21)
        # the double-differential cross section ma pat each input energy
        self.total_xs_table, differential_xs_table, double_differential_xs_table = PairProduction._calculate_cross_section(
            charge, self.incident_energy_table, self.energy_fraction_table,
            self.angle_CM_table, positron_angle_CM_table, dihedral_angle_table)
        
        print(f'Calculated pair production tables for {len(self.incident_energy_table)} photon energies')
    
        # Create interpolation objects for fast lookups
        self.differential_xs_interpolator = make_interp_spline(
            self.incident_energy_table, differential_xs_table, k=1, axis=0)
        self.double_differential_xs_interpolator = make_interp_spline(
            self.incident_energy_table, double_differential_xs_table, k=1, axis=0)
        
        # leave these as None until we evaluate them at a specific energy
        self.double_differential_xs = None
        self.electron_energy_table = None
        self.angle_distribution = None
        self.energy_distribution_interpolator = None

    @staticmethod
    def _calculate_cross_section(
            charge: int, incident_energy: np.ndarray, energy_fraction: np.ndarray,
            electron_angle_CM: np.ndarray, positron_angle_CM: np.ndarray, dihedral_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        calculate the double-differential cross section that will be used to sample electron angles and energies
        
        unlike elastic scattering, which is fully defined by two numbers (theta and phi),
        pair-production requires four numbers to be fully defined, and is energy-dependent.
        this function assumes that the mass of the interacting particle is much greater than the photon energy
        and ignores electron screening.
        
        the angles passed should all be in the quasi-CM frame (see _convert_CM_to_lab) even tho the integration
        and sampling is all done in the lab frame,
        
        Args:
            charge: the charge number of the particle with which the photon is interacting
            incident_energy: the incident photon energies at which to calculate the cross section (MeV)
            energy_fraction: the fractions of kinetic energy given to the electron at which to calculate the cross section
            electron_angle_CM: the angles between the photon and electron trajectories at which to calculate the cross section (rad)
            positron_angle_CM: the angles between the photon and positron trajectories to sample (rad)
            dihedral_angle: the angles between the electron and positron trajectories in the plane normal to the photon trajectory to sample (rad)
        
        Returns:
            0: a 1D array of the total microscopic cross section at each given energy, in m^2
            1: a 2D array of the differential microscopic cross section at each combination of energy and electron angle, in m^2/sr
            2: a 3D array of the double-differential microscopic cross section at each combination of energy, electron angle, and electron energy fraction, in m^2/sr/MeV
        """
        incident_energy, energy_fraction, electron_angle_CM, positron_angle_CM, dihedral_angle = np.meshgrid(
            incident_energy, energy_fraction, electron_angle_CM, positron_angle_CM, dihedral_angle,
            indexing="ij", sparse=True)
        
        # convert from these normalized coordinates to the coordinates over which we'll actual integrate
        electron_kinetic_energy = energy_fraction*(incident_energy - 2*ELECTRON_REST_ENERGY)  # MeV
        electron_angle = PairProduction._convert_CM_to_lab(electron_angle_CM, incident_energy)
        positron_angle = PairProduction._convert_CM_to_lab(positron_angle_CM, incident_energy)
        
        # this section of the code uses abbreviated notation: γ for photons, n for electrons, and p for positrons
        # E is total relativistic energy in MeV, p is momentum magnitude in MeV/c,
        # and q is recoiling nucleus momentum in MeV/c
        Eγ = incident_energy
        En = electron_kinetic_energy + ELECTRON_REST_ENERGY
        Ep = Eγ - En
        pn2 = np.maximum(0, En**2 - ELECTRON_REST_ENERGY**2)  # these should both always >= 0 in theory but because of roundoff we have to enforce that
        pp2 = np.maximum(0, Ep**2 - ELECTRON_REST_ENERGY**2)
        pn = np.sqrt(pn2)
        pp = np.sqrt(pp2)
        
        pny = pn * np.sin(electron_angle)
        pnR = pny
        pnz = pn * np.cos(electron_angle)
        ppR = pp * np.sin(positron_angle)
        ppx = pp * np.sin(positron_angle) * np.sin(dihedral_angle)
        ppy = pp * np.sin(positron_angle) * np.cos(dihedral_angle)
        ppz = pp * np.cos(positron_angle)
        qx = -ppx
        qy = -pny - ppy
        qz = Eγ - pnz - ppz
        q2 = qx**2 + qy**2 + qz**2
        q4 = q2**2
        
        quadruple_differential_xs = (
            FINE_STRUCTURE_CONSTANT * CLASSICAL_ELECTRON_RADIUS**2 * ELECTRON_REST_ENERGY**2
            / (2*np.pi)**2
            * charge**2
            * pp * pn / (q4 * Eγ**3)
            * (
                - (pnR / (En - pnz))**2 * (4*Ep**2 - q2)
                - (ppR / (Ep - ppz))**2 * (4*En**2 - q2)
                + 2/((Ep - ppz) * (En - pnz)) * ((ppR**2 + pnR**2) * Eγ**2 + ppy*pny * (2*Ep**2 + 2*En**2 - q2))
            )
        )  # m^2/sr^2/MeV
        
        # integrate out the positron parameters
        double_differential_xs = 2*integrate.simpson(  # factor of 2 because we only integrated over half the dihedral angles
            integrate.simpson(
                quadruple_differential_xs,
                dihedral_angle[:, :, :, :, :], axis=4),
            -np.cos(positron_angle[:, :, :, :, 0]), axis=3)
        
        differential_xs = integrate.simpson(
            double_differential_xs,
            electron_kinetic_energy[:, :, :, 0, 0], axis=1)  # m^2/sr
        
        total_xs = 2*np.pi*integrate.simpson(  # factor of 2pi to represent the integral over azimuthal angle
            differential_xs,
            -np.cos(electron_angle[:, 0, :, 0, 0]), axis=1)
        
        # the final distribution should have axes of 0: input energy, 1: electron angle, 2: electron energy
        double_differential_xs = np.transpose(double_differential_xs, (0, 2, 1))
        
        return total_xs, differential_xs, double_differential_xs
    
    @staticmethod
    def _convert_CM_to_lab(angle: np.ndarray, incident_energy: float | np.ndarray) -> np.ndarray:
        """
        convert an electron or positron trajectory angle from the quasi-center-of-mass frame to the lab frame.
        this isn't really the center of mass frame, since I think that would depend on the mass of the interacting
        nucleus, but it's the frame in which the photon energy is exactly 1.022 MeV.  this conversion assumes everything
        is fully relativistic.  I think.
        
        truthfully I (Justin) haven't done all of the math out, because it doesn't really matter, because this is just
        a convenient way to concentrate integration samples on the more probable parts of the distribution.
        none of the actual math is done in the quasi-CM frame.
        """
        return np.arctan(2*ELECTRON_REST_ENERGY/incident_energy*np.tan(angle))
    
    def get_cross_section(self, incident_energy: float | np.ndarray) -> float | np.ndarray:
        """ get the total macroscopic cross section, for calculating attenuation, in m^-1 """
        if np.any(incident_energy > self.incident_energy_table[-1]):
            raise ValueError(f"I didn't calculate the PP xs for {incident_energy} MeV photons")
        else:
            return np.where(
                incident_energy > 2*ELECTRON_REST_ENERGY,
                self.target_density * np.interp(
                    incident_energy,
                    self.incident_energy_table,
                    self.total_xs_table),
                0,
            )
    
    def calculate_angular_distribution(self, incident_energy: float):
        """ prepare to generate recoil particles from the given incident input particle energy """
        # calculate the angular distribution
        differential_xs = self.differential_xs_interpolator(incident_energy)
        angle_table = PairProduction._convert_CM_to_lab(self.angle_CM_table, incident_energy)
        self.angle_distribution = ProbabilityDistribution(
            angle_table,
            differential_xs * np.sin(angle_table))
        # prepare to calculate the energy distribution
        double_differential_xs = self.double_differential_xs_interpolator(incident_energy)
        self.electron_energy_table = self.energy_fraction_table * (incident_energy - 2*ELECTRON_REST_ENERGY)
        self.energy_distribution_interpolator = make_interp_spline(
            angle_table, double_differential_xs, k=1, axis=0)
    
    def get_recoil_probability(self, max_angle=np.pi) -> float:
        """
        calculate the fraction of interactions that produce a valid recoil particle
        (I apologize for the somewhat confusing wording; "recoil particle" means electron, not the recoiling nucleus)
        """
        if max_angle >= np.pi/2:
            return 1
        elif self.angle_distribution is None:
            raise ValueError("You need to call calculate_angular_distribution() before I can start generating recoil rays.")
        else:
            return self.angle_distribution.integral(0, max_angle)
    
    def generate_recoil_particle(self, rng: np.random.Generator, include_kinematics: bool, max_angle: float) -> tuple[float, float]:
        """ draw an electron and return its angle (radians) and initial energy (MeV) """
        if self.angle_distribution is None:
            raise ValueError("You need to call calculate_angular_distribution() before I can start generating recoil rays.")
        
        angle = self.angle_distribution.draw(rng, upper=max_angle)
        
        electron_energy_distribution = ProbabilityDistribution(
            self.electron_energy_table,
            self.energy_distribution_interpolator(angle),
        )
        electron_energy = electron_energy_distribution.draw(rng)
        
        return angle, electron_energy


Interaction = GenericInteraction | ComptonScattering | PairProduction
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
