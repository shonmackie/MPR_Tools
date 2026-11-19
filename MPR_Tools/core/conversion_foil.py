"""Conversion foil implementation for neutral-charged particle scattering."""

from typing import Tuple, Optional, Literal
import numpy as np
from pathlib import Path

from scipy import integrate
from concurrent.futures import Executor
import multiprocessing as mp
from tqdm import tqdm

from .matter_interactions import Interaction, GenericInteraction, ElasticScattering, ComptonScattering, PairProduction, \
    ContinuousProbabilityDistribution, DiscreteProbabilityDistribution
from .parallelization import run_concurrently
from ..config.constants import AVOGADRO, FOIL_MATERIALS, LIGHT_SPEED, NEUTRON_MASS, MASS_TO_MEV


class ConversionFoil:
    """
    Represents a conversion foil and aperture system for generating recoil particles.
    
    The foil is where incident particles scatter and generate recoil particles, while the aperture
    defines the ion optical acceptance.
    """    
    def __init__(
        self,
        foil_radius: float,
        thickness: float,
        aperture_distance: float,
        aperture_radius: Optional[float] = None,
        aperture_width: Optional[float] = None,
        aperture_height: Optional[float] = None,
        foil_material: Literal['CH2', 'CD2', 'LiH', 'Be', 'B'] = 'CH2',
        particle: Optional[str] = None,
        aperture_type: Literal['circ', 'rect'] = 'circ',
    ):
        """
        Initialize conversion foil system.


        Args:
            foil_radius: Foil radius in cm
            thickness: Foil thickness in μm
            aperture_distance: Distance from foil to aperture in cm
            aperture_radius: Aperture radius in cm (required for circular apertures)
            aperture_width: Aperture width (in dispersion direction [x]) in cm (required for rectangular apertures)
            aperture_height: Aperture height (in non-dispersion direction [y]) in cm (required for rectangular apertures)
            foil_material: Foil material to use ('CH2' or 'CD2')
            particle: Recoil particle species to use (or None to infer from the foil material)
            aperture_type: Type of aperture ('circ' or 'rect')
        """
        print('Initializing conversion foil...')

        # Convert units and store geometry
        self.foil_radius = foil_radius * 1e-2  # cm to m
        self.thickness = thickness * 1e-6      # μm to m
        self.aperture_distance = aperture_distance * 1e-2  # cm to m
        if aperture_type == 'circ':
            if aperture_radius is None:
                raise ValueError("aperture_radius must be provided for circular apertures.")
            self.aperture_radius = aperture_radius * 1e-2      # cm to m
            # Limit scattering angles for computational efficiency
            self.max_angle = np.arctan((foil_radius + aperture_radius) / aperture_distance)  # rad
        elif aperture_type == 'rect':
            if aperture_width is None or aperture_height is None:
                raise ValueError("aperture_width and aperture_height must be provided for rectangular apertures.")
            self.aperture_width = aperture_width * 1e-2        # cm to m
            self.aperture_height = aperture_height * 1e-2      # cm to m
            # Limit scattering angles for computational efficiency
            half_diag = np.sqrt((aperture_width / 2)**2 + (aperture_height / 2)**2)
            self.max_angle = np.arctan((foil_radius + half_diag) / aperture_distance)  # rad
        else:
            raise ValueError(f"Unsupported aperture_type: {aperture_type!r}. Use 'circ' or 'rect'.")

        self.aperture_type = aperture_type
        
        # Load particle densities
        self.foil_material = foil_material
        self.foil_density = FOIL_MATERIALS[foil_material]['density'] # g/cm^3
        self.molecular_weight = FOIL_MATERIALS[foil_material]['molecular_weight'] # g/mol
        self.particle_mass = FOIL_MATERIALS[foil_material]['particle_mass'] # amu
        self.interaction_info = FOIL_MATERIALS[self.foil_material]['interactions']
        
        # Set or infer the incident and recoil particle species
        self.incident_particle, self.particle = None, None
        for interaction in self.interaction_info:
            if 'recoil_particle' in interaction and (particle is None or interaction['recoil_particle'] == particle):
                self.incident_particle = interaction['incident_particle']
                self.particle = interaction['recoil_particle']
                break
        if self.particle is None or self.incident_particle is None:
            raise ValueError(f"The material '{foil_material}' has no interactions that produce {particle}s.")
        
        # Load cross section and stopping power data
        if self.particle == 'electron':
            self.stopping_data_path = FOIL_MATERIALS[foil_material]['electron_stopping_power']
        else:
            self.stopping_data_path = FOIL_MATERIALS[foil_material]['ion_stopping_power']
        self._load_data_files()
        
        print('Conversion foil initialization complete.\n')
    
    def _load_data_files(self) -> None:
        """Load all required data files."""
        data_dir = Path(__file__).parent.parent / "data"
    
        # load cross section data
        if self.particle == "electron":
            energy, mass_stopping_power = np.genfromtxt(data_dir / self.stopping_data_path, skip_header=8, unpack=True)
            stopping_power = mass_stopping_power * self.foil_density / 1e-2  # MeV/m
            self.integrated_stopping_data = ConversionFoil._preintegrate_stopping_data(energy, stopping_power)
            print(f'Loaded ESTAR data from {self.stopping_data_path}')
        else:
            energy, electronic_stopping_power, nuclear_stopping_power = np.genfromtxt(data_dir / self.stopping_data_path, skip_header=2, unpack=True)
            stopping_power = (electronic_stopping_power + nuclear_stopping_power) / 1e-3  # MeV/m
            self.integrated_stopping_data = ConversionFoil._preintegrate_stopping_data(energy, stopping_power)
            print(f'Loaded SRIM data from {self.stopping_data_path}')

        # Load cross section data
        self.interactions: list[Interaction] = []
        for interaction_info in self.interaction_info:
            if interaction_info['incident_particle'] != self.incident_particle:
                continue
            molecular_density = self.foil_density / self.molecular_weight * AVOGADRO * 1e6  # molecules/m^3
            target_density = interaction_info['target_abundance'] * molecular_density  # targets/m^3
            if interaction_info['type'] == 'generic':
                self.interactions.append(GenericInteraction(
                    interaction_info['name'],
                    target_density, data_dir / interaction_info['total_cross_section']))
            elif interaction_info['type'] == 'elastic_scattering':
                assert interaction_info['incident_particle'] == 'neutron'
                self.interactions.append(ElasticScattering(
                    f'(n,{self.particle[0]}) elastic',
                    interaction_info['recoil_particle'],
                    target_density,
                    data_dir / interaction_info['total_cross_section'],
                    data_dir / interaction_info['diff_cross_section'],
                    self.particle_mass))
            elif interaction_info['type'] == 'compton_scattering':
                assert interaction_info['incident_particle'] == 'photon'
                assert interaction_info['recoil_particle'] == 'electron'
                self.interactions.append(ComptonScattering(target_density))
            elif interaction_info['type'] == 'pair_production':
                assert interaction_info['incident_particle'] == 'photon'
                assert interaction_info['recoil_particle'] == 'electron'
                self.interactions.append(PairProduction(target_density, interaction_info['target_charge']))
            else:
                raise ValueError(f"I don't know the interaction type {interaction_info['type']!r}.")
    
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
    def aperture_radius_cm(self) -> Optional[float]:
        """Get aperture radius in cm (circular apertures only)."""
        if self.aperture_type != 'circ':
            return None
        return self.aperture_radius * 1e2

    @property
    def aperture_width_cm(self) -> Optional[float]:
        """Get aperture width in cm (rectangular apertures only)."""
        if self.aperture_type != 'rect':
            return None
        return self.aperture_width * 1e2

    @property
    def aperture_height_cm(self) -> Optional[float]:
        """Get aperture height in cm (rectangular apertures only)."""
        if self.aperture_type != 'rect':
            return None
        return self.aperture_height * 1e2
    
    def set_foil_radius(self, radius_cm: float) -> None:
        """Set foil radius in cm."""
        self.foil_radius = radius_cm * 1e-2
    
    def set_thickness(self, thickness_um: float) -> None:
        """Set foil thickness in μm."""
        self.thickness = thickness_um * 1e-6
    
    def set_aperture_distance(self, distance_cm: float) -> None:
        """
        Set foil-aperture separation in cm.
        
        Warning: This impacts ion optical transfer maps. Ensure consistency 
        with COSY inputs for accurate results.
        """
        self.aperture_distance = distance_cm * 1e-2
    
    def set_aperture_radius(self, radius_cm: float) -> None:
        """Set aperture radius in cm (circular apertures only)."""
        if self.aperture_type != 'circ':
            raise ValueError("aperture_radius can only be set for circular apertures.")
        self.aperture_radius = radius_cm * 1e-2
        
    def set_aperture_width(self, width_cm: float) -> None:
        """Set aperture width in cm (for rectangular apertures)."""
        if self.aperture_type != 'rect':
            raise ValueError("Aperture width can only be set for rectangular apertures.")
        self.aperture_width = width_cm * 1e-2
        
    def set_aperture_height(self, height_cm: float) -> None:
        """Set aperture height in cm (for rectangular apertures)."""
        if self.aperture_type != 'rect':
            raise ValueError("Aperture height can only be set for rectangular apertures.")
        self.aperture_height = height_cm * 1e-2
        
    def get_incident_energy(self, recoil_energy: float) -> float:
        """
        Estimate the incident particle energy that best corresponds to the given recoil particle spectrum peak.
        """
        recoil_birth_energy = self.calculate_initial_energy(recoil_energy, self.thickness/2)
        incident_energy = None
        for interaction in self.interactions:
            try:
                incident_energy = interaction.get_incident_energy(recoil_birth_energy)
            except ValueError:
                pass
        return incident_energy
    
    def calculate_stopping_power_loss(
        self, 
        initial_energy: float, 
        path_length: float, 
    ) -> float:
        """
        Calculate energy after slowing down in foil material.
        
        Neglects straggling (valid for thin foils, less accurate at low energies).
        
        Args:
            initial_energy: Initial recoil particle energy in MeV
            path_length: Distance traveled through material in m
            
        Returns:
            Final recoil particle energy in MeV
        """
        # check to make sure we're within the data bounds
        if initial_energy > self.integrated_stopping_data[0][-1]:
            raise ValueError(
                f"We can't slow a {initial_energy} MeV particle because the SRIM data "
                f"doesn't go up that high.")
        initial_x = np.interp(
            initial_energy, self.integrated_stopping_data[0], self.integrated_stopping_data[1])
        final_x = initial_x - path_length
        final_energy = np.interp(
            final_x, self.integrated_stopping_data[1], self.integrated_stopping_data[0])
        return final_energy
    
    def calculate_initial_energy(
        self, 
        final_energy: float, 
        path_length: float, 
    ) -> float:
        """
        Calculate initial energy by reversing energy loss calculation.
        
        Args:
            final_energy: Final recoil particle energy in MeV
            path_length: Distance traveled through material in m
            
        Returns:
            Initial recoil particle energy in MeV
        """
        final_x = np.interp(
            final_energy, self.integrated_stopping_data[0], self.integrated_stopping_data[1])
        initial_x = final_x + path_length
        initial_energy = np.interp(
            initial_x, self.integrated_stopping_data[1], self.integrated_stopping_data[0])
        # check to make sure we didn't hit the data bound
        if initial_energy == self.integrated_stopping_data[0][-1]:
            raise ValueError(
                f"Calculating the initial energy of this particle failed because we hit "
                f"the upper limit of the SRIM data, {initial_energy} MeV.")
        return initial_energy

    @staticmethod
    def _preintegrate_stopping_data(energy, stopping_power) -> Tuple[np.ndarray, np.ndarray]:
        """
        convert dE/dx vs. E table into a range vs. E table, so that we can do slowing calculations with just lookups

        Args:
            energy: Array of charged particle energies at which stopping has been calculated, in MeV
            stopping_power: the total stopping rate, in MeV/m

        Returns:
            Array of charged particle energies at which range has been calculated, in MeV
            Distance a charged at the given energy can travel in solid matter before stopping, in m
        """
        # add an infinity to the bottom of the stopping table so that behavior is defined down to E=0
        E = np.concatenate([[0], energy])  # MeV
        dE_dx = np.concatenate([[np.inf], stopping_power])  # MeV/m

        # do the integral
        dx_dE = 1 / dE_dx  # m/MeV
        x = integrate.cumulative_trapezoid(x=E, y=dx_dE, initial=0)  # m

        return E, x
    
    def generate_monte_carlo_rays(
        self,
        incident_energies: np.ndarray,
        probability_distribution: np.ndarray,
        num_recoil_particles: int,
        reference_energy: float,
        stop_condition: Literal['attempts', 'successes'] = 'successes',
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        z_sampling: Literal['exp', 'uni'] = 'exp',
        target_to_foil_distance: Optional[float] = None,
        burn_duration: Optional[float] = None,
        min_energy: float = 0,
        max_energy: float = np.inf,
        executor: Optional[Executor] = None,
        max_workers: Optional[int] = None,
        y_restriction: Optional[Literal['positive', 'negative']] = None,
        continuous_energy_sampling: bool = True,
        importance_sampling: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Generate recoil rays from incident particle energy distribution using Monte Carlo with multiprocessing.

        Args:
            incident_energies:   Array of incident particle energies in MeV.
            probability_distribution: Relative probability for each energy bin. Normalized automatically.
            num_recoil_particles: Target number of recoil particle trajectories to return.
            stop_condition: How to interpret `num_recoil_particles`;
                            should we go for a certain number of particle generation *attempts*, or
                            for a certain number of successful (i.e. through the aperture) particle generations
            reference_energy: Reference energy in MeV (this is needed because the outputs are relative to it)
            include_kinematics: Include kinematic energy transfer.
            include_stopping_power_loss: Include stopping power energy loss via SRIM.
            z_sampling: Depth sampling method ('exp' or 'uni').
            target_to_foil_distance: Distance from neutron source to foil in meters, for calculating time spread.
                                     If None, arrival time is set to 0.
            burn_duration: FWHM duration of the neutron source in seconds (assumed Gaussian).
            min_energy: Don't generate recoil particles with less than this energy (MeV).
            max_energy: Don't generate recoil particles with more than this energy (MeV).
            executor: Pool of workers to use (if None, a new pool is created).
            max_workers: Maximum number of worker processes (None -> CPU count).
            y_restriction: Restrict sampled foil y position to 'positive' or 'negative' half.
            continuous_energy_sampling: If True, sample energy continuously via inverse CDF. If False,
                                        sample from discrete energy values.
            importance_sampling: If True, sample all energies with equal (uniform) probability and
                                 attach a per-particle weight w = p_weighted(E) / p_uniform(E) so
                                 that low-probability energies are sampled more uniformly while
                                 preserving the correct weighted response.
        
        Return:
            8xn array of [x0, p_x_relative, y0, p_y_relative, foil_time, energy_relative, incident_energy, importance_weight]
            Scattering efficiency as a fraction of incident particles
            Geometric efficiency as a fraction of scattered particles
        """
        if max_workers is None:
            max_workers = mp.cpu_count()
        
        print(f'Generating {num_recoil_particles} Monte Carlo {self.particle} trajectories using {max_workers} processes...')
        
        # Calculate recoil events per process
        particles_per_process = num_recoil_particles // max_workers
        remaining_particles = num_recoil_particles % max_workers
        
        # Calculate scattering probability in foil at each energy (non-parallelizable part)
        xs_total = np.zeros_like(probability_distribution) # the macroscopic cross section for the incident particle undergoing an interaction
        xs_scatter = np.zeros_like(probability_distribution)  # the macroscopic cross section for the incident particle producing a recoil particle
        xs_forward_scatter = np.zeros_like(probability_distribution)  # the macroscopic cross section for the incident particle producing a recoil particle within max_angle
        for i, incident_energy in enumerate(incident_energies):
            for interaction in self.interactions:
                interaction_xs = interaction.get_cross_section(incident_energy)
                xs_total[i] += interaction_xs
                if interaction.recoil_particle == self.particle:
                    xs_scatter[i] += interaction_xs
                    angle_distribution = interaction.get_angle_distribution(incident_energies[i])
                    angle_fraction = angle_distribution.integral(0, self.max_angle)
                    xs_forward_scatter[i] += interaction_xs * angle_fraction
        
        # Integrate the cross sections to calculate efficiency
        p_anything = 1 - np.exp(-xs_total * self.thickness)
        p_scatter = p_anything * xs_scatter/xs_total
        p_forward_scatter = p_anything * xs_forward_scatter/xs_total
        if continuous_energy_sampling and len(probability_distribution) > 1:
            probability_distribution = probability_distribution / integrate.trapezoid(probability_distribution, incident_energies)  # MeV^-1
            scattering_efficiency = integrate.trapezoid(probability_distribution * p_scatter, incident_energies)  # the probability than an incident particle generates a recoil particle
            forward_scattering_efficiency = integrate.trapezoid(probability_distribution * p_forward_scatter, incident_energies)  # the probability that an incident particle generates a recoil particle within max_angle
        else:
            probability_distribution = probability_distribution / np.sum(probability_distribution)  # (dimensionless)
            scattering_efficiency = np.sum(probability_distribution * p_scatter)  # the probability than an incident particle generates a recoil particle
            forward_scattering_efficiency = np.sum(probability_distribution * p_forward_scatter)  # the probability that an incident particle generates a recoil particle within max_angle
        angular_factor = scattering_efficiency / forward_scattering_efficiency  # the factor by which our total sampling density is increased because we're using max_angle
        
        weighted_distribution = probability_distribution*xs_forward_scatter / np.sum(probability_distribution*xs_forward_scatter)
        
        # With importance sampling, draw energies from a flat distribution and weight each
        # particle by weighted_distribution / uniform_distribution (source biasing).
        if importance_sampling:
            sampling_distribution = np.ones_like(weighted_distribution) / len(weighted_distribution)
        else:
            sampling_distribution = weighted_distribution
        
        # Execute in parallel
        worker_args = []
        for i in range(max_workers):
            batch_size = particles_per_process + (1 if i < remaining_particles else 0)
            if batch_size > 0:  # Only submit if there's work to do
                # Package all parameters for the worker
                worker_args.append((
                    batch_size,
                    stop_condition,
                    12345 + i * 1000,  # seed_offset
                    incident_energies,
                    sampling_distribution,
                    include_kinematics,
                    include_stopping_power_loss,
                    z_sampling,
                    reference_energy,
                    target_to_foil_distance,
                    burn_duration,
                    min_energy,
                    max_energy,
                    y_restriction,
                    continuous_energy_sampling,
                    importance_sampling,
                    weighted_distribution,
                ))
        
        outputs = run_concurrently(
            self._generate_batch_worker, worker_args, executor,
            progress_counter_total=num_recoil_particles,
            task_title=f'Generating Monte Carlo {self.particle} trajectories',
        )
        output_batches, generated_weights = zip(*outputs)
        
        # Collect results
        beam = np.concatenate(output_batches)
        accepted_weight = np.sum(beam[:, 7])
        generated_weight = sum(generated_weights)
        # Sanity check
        if stop_condition == 'successes':
            assert len(beam) == num_recoil_particles
         
        geometric_efficiency = accepted_weight / generated_weight / angular_factor
    
        return beam, scattering_efficiency, geometric_efficiency
        
    # Class method worker function (used with multiprocessing)
    def _generate_batch_worker(
        self,
        batch_size: int,
        stop_condition: Literal['attempts', 'successes'],
        seed_offset: int,
        incident_energies: np.ndarray,
        sampling_distribution: np.ndarray,
        include_kinematics: bool,
        include_stopping_power_loss: bool,
        z_sampling: Literal['exp', 'uni'],
        reference_energy: float,
        target_to_foil_distance: Optional[float],
        burn_duration: Optional[float],
        min_energy: float,
        max_energy: float,
        y_restriction: Optional[Literal['positive', 'negative']],
        continuous_energy_sampling: bool,
        importance_sampling: bool,
        importance_weights: np.ndarray,
        progress_counter,
        progress_lock,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate a batch of recoil particles in a separate process.

        Each row of the returned array is:
            [x0, p_x_relative, y0, p_y_relative, foil_time, energy_relative, incident_energy, importance_weight]
            
        Also return the total amount of particle weight that was generated (before checking aperture acceptance)
        """
        # Create an independent RNG for this worker to ensure reproducibility
        rng = np.random.default_rng(seed_offset)
        
        # Precompute inverse CDF for continuous energy sampling
        if continuous_energy_sampling:
            incident_energy_distribution = ContinuousProbabilityDistribution(incident_energies, sampling_distribution)
        else:
            incident_energy_distribution = DiscreteProbabilityDistribution(incident_energies, sampling_distribution)
        
        particle_rest_energy = self.particle_mass * MASS_TO_MEV  # MeV
        reference_gamma = 1 + reference_energy / particle_rest_energy   # Lorentz factor of the central ray
        
        # Limit scattering to the interactions that actually scatter
        recoil_interactions = []
        for interaction in self.interactions:
            if interaction.recoil_particle == self.particle:
                recoil_interactions.append(interaction)
        
        # Cache energy-dependent quantities to avoid recomputing them when the same energy is
        # sampled twice in a row (i.e. for monoenergetic inputs).
        previous_incident_energy = None
        angle_distributions = None
        interaction_weights = None
        attenuation = 0.0
        
        # count both total and accepted particles
        num_generated = 0
        weight_generated = 0.
        num_accepted = 0
        batch_results = []
        
        while True:
            if stop_condition == 'attempts':
                if num_generated == batch_size:
                    break
            elif stop_condition == 'successes':
                if num_accepted == batch_size:
                    break
            else:
                raise ValueError(f"Unrecognized stop condition: {stop_condition!r}.")
            if num_accepted == 0 and num_generated > 1000:
                raise ValueError(f"Unable to generate a recoil particle that passes through the aperture and is in the energy range [{min_energy}, {max_energy}].")
            
            # Sample incident particle energy from weighted distribution
            incident_energy = incident_energy_distribution.draw(rng)
            
            # Only recompute energy-dependent cross sections and angle distributions when the
            # incident energy changes
            if incident_energy != previous_incident_energy:
                # Do the cross section calculations
                angle_distributions = {}
                interaction_weights = []
                for interaction in recoil_interactions:
                    angle_distributions[interaction] = interaction.get_angle_distribution(incident_energy)
                    weight = (interaction.get_cross_section(incident_energy) *
                              angle_distributions[interaction].integral(0, self.max_angle))
                    interaction_weights.append(weight)
                interaction_weights = np.array(interaction_weights) / sum(interaction_weights)
                
                # Attenuation coefficient for exponential depth sampling; 0 gives uniform sampling
                if z_sampling == 'exp':
                    attenuation = float(sum(
                        interaction.get_cross_section(incident_energy)
                        for interaction in self.interactions
                    ))
                else:
                    attenuation = 0.0
                
                previous_incident_energy = incident_energy
            
            interaction = rng.choice(recoil_interactions, p=interaction_weights)
            x0, y0, z0, theta_scatter, phi_scatter = self._sample_scattered_ray(
                rng, angle_distributions[interaction], attenuation, y_restriction,
            )
            
            # Compute importance sampling weight: w = p_weighted(E) / p_uniform(E).
            if importance_sampling:
                if continuous_energy_sampling and len(incident_energies) > 1:
                    e_range = incident_energies[-1] - incident_energies[0]
                    particle_weight = float(
                        np.interp(incident_energy, incident_energies, importance_weights)
                    ) * e_range
                else:
                    idx = int(np.clip(
                        np.searchsorted(incident_energies, incident_energy),
                        0, len(incident_energies) - 1,
                           ))
                    particle_weight = importance_weights[idx] * len(incident_energies)
            else:
                particle_weight = 1.0
            
            num_generated += 1
            weight_generated += particle_weight
            
            # Check if recoil particle passes through aperture
            if self._check_aperture_acceptance(x0, y0, theta_scatter, phi_scatter):
                # Initialize recoil energy
                recoil_energy = interaction.get_recoil_energy(
                    incident_energy,
                    theta_scatter if include_kinematics else 0,
                    rng)
                
                # Apply stopping power energy loss
                if include_stopping_power_loss:
                    path_length = (-z0) / np.cos(theta_scatter)
                    recoil_energy = self.calculate_stopping_power_loss(recoil_energy, path_length)
                
                if recoil_energy <= 0:
                    continue
            else:
                continue
                
            # Reject recoil particles outside the spectrometer acceptance window.
            # This is the correct place to filter: for heavier recoils (e.g. deuterons),
            # high-energy neutrons can still produce accepted recoils at large lab angles,
            # so filtering on incident energy instead would wrongly exclude them.
            if not (min_energy <= recoil_energy <= max_energy):
                continue
            
            # Convert foil position and scattering angles to COSY phase-space coordinates
            x_aperture = x0 + self.aperture_distance * np.tan(theta_scatter) * np.cos(phi_scatter)
            y_aperture = y0 + self.aperture_distance * np.tan(theta_scatter) * np.sin(phi_scatter)
            
            total_distance = np.sqrt(
                (x_aperture - x0)**2 + (y_aperture - y0)**2 +
                self.aperture_distance**2)
            sin_angle_x = (x_aperture - x0) / total_distance
            sin_angle_y = (y_aperture - y0) / total_distance
            
            gamma = 1 + recoil_energy/particle_rest_energy  # Lorentz factor of the ray
            p_relative = np.sqrt((gamma**2 - 1)/(reference_gamma**2 - 1))
            p_x_relative = p_relative * sin_angle_x
            p_y_relative = p_relative * sin_angle_y
            
            energy_relative = (recoil_energy - reference_energy) / reference_energy
            
            # Calculate foil arrival time
            if self.incident_particle == 'photon':
                velocity = LIGHT_SPEED  # m/s
            elif self.incident_particle == 'neutron':
                # Calculate velocity of the incident particle with its own relativistic correction
                incident_rest_energy = NEUTRON_MASS * MASS_TO_MEV # MeV
                incident_gamma = 1 + incident_energy / incident_rest_energy
                velocity = LIGHT_SPEED * np.sqrt(1.0 - 1.0 / incident_gamma**2)  # m/s
            else:
                raise ValueError(f"Unsupported incident particle type: {self.incident_particle}")
            foil_time = target_to_foil_distance / velocity if target_to_foil_distance else 0.0
            
            # Add Gaussian timing noise if burn duration is provided
            if burn_duration:
                # Convert FWHM burn duration to standard deviation for Gaussian noise
                timing_noise = rng.normal(0, burn_duration / (2 * np.sqrt(2 * np.log(2))))
                foil_time += timing_noise
            
            num_accepted += 1
            batch_results.append(
                (x0, p_x_relative, y0, p_y_relative, foil_time, energy_relative, incident_energy, particle_weight)
            )
            
            # Update progress counter thread-safely
            with progress_lock:
                progress_counter.value += 1
                
        # Consolidate in a numpy array, making sure it's the right shape even if we generated 0 rays
        if len(batch_results) == 0:
            return np.empty((0, 8)), weight_generated
        else:
            return np.array(batch_results), weight_generated
    
    def _sample_scattered_ray(
        self,
        rng: np.random.Generator,
        theta_distribution: ContinuousProbabilityDistribution,
        attenuation: float,
        y_restriction: Optional[Literal['positive', 'negative']] = None
    ) -> Tuple[float, float, float, float, float]:
        """
        Sample a scattered ray from the foil.
        
        Args:
            rng: Random number generator
            theta_distribution: The scattering angle probability distribution, in radians
            attenuation: incident fluence falloff rate for z-sampling, in 1/m
                         (0 for uniform sampling; -inf for front-surface-only sampling)
            y_restriction: Restrict y to positive or negative half (None for full foil)
        """
        # Sample initial position on foil surface
        radius_sample = self.foil_radius * np.sqrt(rng.random())
        angle_sample = 2 * np.pi * rng.random()
        x0 = radius_sample * np.cos(angle_sample)
        
        # Apply y-restriction if specified
        if y_restriction == 'positive':
            y0 = abs(radius_sample * np.sin(angle_sample))
        elif y_restriction == 'negative':
            y0 = -abs(radius_sample * np.sin(angle_sample))
        else:
            y0 = radius_sample * np.sin(angle_sample)
        
        # Sample depth
        if attenuation == -np.inf:
            z0 = 0.0  # Sample at exit surface
        elif attenuation == np.inf:
            z0 = -self.thickness  # Sample at entrance surface
        elif abs(self.thickness*attenuation) == 0:
            z0 = rng.uniform(-self.thickness, 0)  # Uniform distribution
        else:
            front_bound = 0
            back_bound = np.expm1(self.thickness*attenuation)
            z0 = -np.log1p(rng.uniform(front_bound, back_bound))/attenuation  # Truncated exponential distribution
        
        # Sample scattering angles
        phi_scatter = 2 * np.pi * rng.random()
        theta_scatter = theta_distribution.draw(rng, upper=self.max_angle)
        
        # Adjust initial coordinates for transport through foil
        x0 += z0 * np.tan(theta_scatter) * np.cos(phi_scatter)
        y0 += z0 * np.tan(theta_scatter) * np.sin(phi_scatter)
        
        return x0, y0, z0, theta_scatter, phi_scatter
    
    def _check_aperture_acceptance(
            self,
            x0: float,
            y0: float,
            theta_scatter: float,
            phi_scatter: float
    ) -> bool:
        """
        Check if a scattered recoil particle passes through the aperture.
        
        Args:
            x0, y0: Initial position in foil
            theta_scatter, phi_scatter: Scattering angles
            
        Returns:
            True if recoil particle passes through aperture
        """
        # Calculate position at aperture
        x_aperture = x0 + self.aperture_distance * np.tan(theta_scatter) * np.cos(phi_scatter)
        y_aperture = y0 + self.aperture_distance * np.tan(theta_scatter) * np.sin(phi_scatter)
        
        if self.aperture_type == 'circ':
            return (x_aperture**2 + y_aperture**2) <= self.aperture_radius**2
        elif self.aperture_type == 'rect':
            return (np.abs(x_aperture) <= self.aperture_width/2 and
                    np.abs(y_aperture) <= self.aperture_height/2)
        else:
            raise ValueError(f"Unsupported aperture type: {self.aperture_type}")
    
    def generate_recoil_particle(
        self,
        incident_energies: np.ndarray,
        probability_distribution: np.ndarray,
        reference_energy: float,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        z_sampling: Literal['exp', 'uni'] = 'exp',
        y_restriction: Optional[Literal['positive', 'negative']] = None,
        continuous_energy_sampling: bool = True,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Generate a scattered charged particle from incident particle interaction.

        Args:
            incident_energies:   Array of incident particle energies in MeV.
            probability_distribution: Relative probability for each energy bin. Normalized automatically.
            reference_energy: Reference energy in MeV (this is needed because the outputs are relative to it)
            include_kinematics: Include energy loss from non-perpendicular scattering.
            include_stopping_power_loss: Include SRIM energy loss calculation.
            z_sampling: Depth sampling method ('exp' for exponential, 'uni' for uniform).
            y_restriction: Restrict sampled y position to 'positive' or 'negative' half of foil.
            continuous_energy_sampling: If True, sample energy continuously via inverse CDF. If False,
                                        sample from discrete bin centres.
        Returns:
            Tuple of (x0, p_x_relative, y0, p_y_relative, foil_time, recoil_energy_relative, incident_energy)
        """
        x0, p_x_relative, y0, p_y_relative, foil_time, recoil_energy_relative, incident_energy, importance_weight = self.generate_monte_carlo_rays(
            incident_energies, probability_distribution,
            num_recoil_particles=1,
            reference_energy=reference_energy,
            include_kinematics=include_kinematics,
            include_stopping_power_loss=include_stopping_power_loss,
            z_sampling=z_sampling,
            y_restriction=y_restriction,
            continuous_energy_sampling=continuous_energy_sampling,
        )[0][0, :]
        return x0, p_x_relative, y0, p_y_relative, foil_time, recoil_energy_relative, incident_energy
    
    def calculate_efficiency(
        self, 
        incident_energy: float,
        num_samples: int = 10000,
        executor: Optional[Executor] = None,
        max_workers: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """
        Estimate intrinsic efficiency (recoils/incident) of the spectrometer using parallel processing.
        
        Args:
            incident_energy: Incident particle energy in MeV
            num_samples: Number of particles to simulate
            executor: Pool of workers to use (if None, we will make our own)
            max_workers: Maximum number of worker processes (None for CPU count)
            
        Returns:
            Tuple of scattering, geometric, and total efficiency as fraction of incident particles
        """
        print(f'\nEstimating intrinsic efficiency for {incident_energy:.3f} MeV particles using {max_workers} processes...')
        
        _, scattering_efficiency, geometric_efficiency = self.generate_monte_carlo_rays(
            np.array([incident_energy]),
            np.array([1.0]),
            num_recoil_particles=num_samples,
            stop_condition='attempts',
            executor=executor,
            max_workers=max_workers,
            reference_energy=1,  # it doesn't matter what this reference energy is because we're only looking at the number of acceptances
        )
        
        # Calculate total efficiencie
        total_efficiency = scattering_efficiency * geometric_efficiency
        
        print(f'Processed {num_samples} samples using {max_workers} processes')
        print(f'Scattering efficiency: {scattering_efficiency:.2e}')
        print(f'Geometric efficiency: {geometric_efficiency:.2e}')
        print(f'Total efficiency: {total_efficiency:.2e}')
        
        return scattering_efficiency, geometric_efficiency, total_efficiency
    
    def get_recoil_energy_distribution(
        self, 
        incident_energies: np.ndarray,
        energy_distribution: np.ndarray, 
        num_recoil_particles: int = int(1e2)
    ) -> np.ndarray:
        """
        Calculate the recoil particle energy distribution at foil exit for a given incident particle energy distribution.
        
        Args:
            incident_energies: Array of incident particle energies in MeV
            energy_distribution: Distribution of incident particle energies (will be normalized and interpolated)
            num_recoil_particles: Number of recoil events to simulate
            
        Returns:
            Array of proton energies at foil exit
        """
        recoil_energies = np.zeros(num_recoil_particles)
        
        # Weight distribution by scattering cross section and normalize
        interaction_probability = np.zeros_like(energy_distribution)
        for interaction in self.interactions:
            if interaction.recoil_particle == self.particle:
                interaction_probability += interaction.get_cross_section(incident_energies)
        weighted_distribution = energy_distribution * interaction_probability
        weighted_distribution = weighted_distribution / np.sum(weighted_distribution)
        
        for i in tqdm(range(num_recoil_particles), desc='Calculating proton energy distribution...'):
            # Generate scattered recoil particle and extract final energy
            _, _, _, _, _, recoil_energy_relative, _ = self.generate_recoil_particle(
                incident_energies,
                weighted_distribution,
                reference_energy=1,
                include_kinematics=True,
                include_stopping_power_loss=True
            )
            
            recoil_energies[i] = 1 + recoil_energy_relative
            
        return recoil_energies
