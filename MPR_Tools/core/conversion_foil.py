"""Conversion foil implementation for neutral-charged particle scattering."""

from typing import Tuple, Optional, Literal
import numpy as np
from pathlib import Path

from scipy import integrate
from concurrent.futures import Executor
import multiprocessing as mp
from tqdm import tqdm

from .matter_interactions import Interaction, GenericInteraction, ElasticScattering, ComptonScattering, PairProduction, \
    ProbabilityDistribution
from .parallelization import run_concurrently
from ..config.constants import AVOGADRO, FOIL_MATERIALS, NEUTRON_MASS


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
        aperture_radius: float,
        aperture_width: Optional[float] = None,
        aperture_height: Optional[float] = None,
        foil_material: Literal['CH2', 'CD2', 'LiH', 'Be', 'B'] = 'CH2',
        aperture_type: Literal['circ', 'rect'] = 'circ',
        target_to_foil_distance: Optional[float] = None,
        burn_duration: Optional[float] = None,
    ):
        """
        Initialize conversion foil system.

        Args:
            foil_radius: Foil radius in cm
            thickness: Foil thickness in μm
            aperture_distance: Distance from foil to aperture in cm
            aperture_radius: Aperture radius in cm
            aperture_width: Aperture width (in dispersion direction [x]) in cm (for rectangular apertures)
            aperture_height: Aperture height (in non-dispersion direction [y]) in cm (for rectangular apertures)
            foil_material: Foil material to use ('CH2' or 'CD2')
            aperture_type: Type of aperture ('circ' or 'rect')
            target_to_foil_distance: Distance from neutron source to foil in meters. When provided,
                                     the arrival time at the foil is calculated from the incident
                                     particle's energy and mass. If None, arrival time is set to 0.
            burn_duration: FWHM duration of the neutron source in seconds. When provided (along with
                           target_to_foil_distance), Gaussian timing noise is added to each particle's foil arrival time.
        """
        print('Initializing conversion foil...')
        
        # Convert units and store geometry
        self.target_to_foil_distance = target_to_foil_distance  # cm to m
        self.burn_duration = burn_duration  # s
        self.foil_radius = foil_radius * 1e-2  # cm to m
        self.thickness = thickness * 1e-6      # μm to m
        self.aperture_distance = aperture_distance * 1e-2  # cm to m
        self.aperture_radius = aperture_radius * 1e-2      # cm to m
        if aperture_type == 'rect':
            if aperture_width is None or aperture_height is None:
                raise ValueError("Aperture width and height must be provided for rectangular apertures.")
            self.aperture_width = aperture_width * 1e-2        # cm to m
            self.aperture_height = aperture_height * 1e-2      # cm to m
            
        self.aperture_type = aperture_type
        
        # Calculate particle densities in CH2
        self.foil_material = foil_material
        self.incident_particle = FOIL_MATERIALS[foil_material]['incident_particle']
        self.particle = FOIL_MATERIALS[foil_material]['particle']
        self.foil_density = FOIL_MATERIALS[foil_material]['density'] # g/cm^3
        self.molecular_weight = FOIL_MATERIALS[foil_material]['molecular_weight'] # g/mol
        self.particle_mass = FOIL_MATERIALS[foil_material]['particle_mass'] # amu
        self.stopping_data_path = FOIL_MATERIALS[foil_material]['stopping_power']
        self.interaction_info = FOIL_MATERIALS[self.foil_material]['interactions']
    
        # Calculate relative mass, either 0 for protons or ~1 for deuterons
        self.relative_mass = (self.particle_mass - FOIL_MATERIALS['CH2']['particle_mass']) / FOIL_MATERIALS['CH2']['particle_mass']
        
        # Load cross section and stopping power data
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
            molecular_density = self.foil_density / self.molecular_weight * AVOGADRO * 1e6  # molecules/m^3
            target_density = interaction_info['target_abundance'] * molecular_density  # targets/m^3
            if interaction_info['type'] == 'generic':
                self.interactions.append(GenericInteraction(
                    interaction_info['name'],
                    target_density, data_dir / interaction_info['total_cross_section']))
            elif interaction_info['type'] == 'elastic_scattering':
                self.interactions.append(ElasticScattering(
                    f'(n,{self.particle[0]}) elastic',
                    target_density,
                    data_dir / interaction_info['total_cross_section'],
                    data_dir / interaction_info['diff_cross_section'],
                    self.particle_mass))
            elif interaction_info['type'] == 'compton_scattering':
                self.interactions.append(ComptonScattering(target_density))
            elif interaction_info['type'] == 'pair_production':
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
    def aperture_radius_cm(self) -> float:
        """Get aperture radius in cm."""
        return self.aperture_radius * 1e2
    
    @property
    def aperture_width_cm(self) -> float:
        """Get aperture width in cm."""
        return self.aperture_width * 1e2
    
    @property
    def aperture_height_cm(self) -> float:
        """Get aperture height in cm."""
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
        """Set aperture radius in cm."""
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
    
    def _sample_scattered_ray(
        self,
        rng: np.random.Generator,
        theta_distribution: ProbabilityDistribution,
        attenuation: float,
        max_angle: float,
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
        theta_scatter = theta_distribution.draw(rng, upper=max_angle)
        
        # Adjust initial coordinates for transport through foil
        x0 += z0 * np.tan(theta_scatter) * np.cos(phi_scatter)
        y0 += z0 * np.tan(theta_scatter) * np.sin(phi_scatter)
        
        return x0, y0, z0, theta_scatter, phi_scatter

    
    def generate_recoil_particle(
        self,
        incident_energies: np.ndarray,
        probability_distribution: np.ndarray,
        include_kinematics: bool = True,
        include_stopping_power_loss: bool = True,
        z_sampling: Literal['exp', 'uni'] = 'exp',
        rng: Optional[np.random.Generator] = None,
        y_restriction: Optional[Literal['positive', 'negative']] = None,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Generate a scattered charged particle from incident particle interaction.

        Args:
            incident_energies: Array of incident particle energies in MeV. One energy value per bin;
                               the particle energy is sampled from this 1-D distribution.
            probability_distribution: Probability weight for each bin in incident_energies (normalised
                                 internally). Must include cross-section weighting before calling.
            include_kinematics: Include energy loss from non-perpendicular scattering.
            include_stopping_power_loss: Include SRIM energy loss calculation.
            z_sampling: Depth sampling method ('exp' for exponential, 'uni' for uniform).
            rng: Random number generator (pass the worker's RNG for thread safety).
            y_restriction: Restrict sampled y position to 'positive' or 'negative' half of foil.

        Returns:
            Tuple of (x0, y0, theta_scatter, phi_scatter, incident_energy, recoil_energy, arrival_time_at_foil)
        """
        # Use provided RNG or create a default one
        if rng is None:
            rng = np.random.default_rng()

        # Limit scattering angles for computational efficiency
        max_angle = np.arctan((self.foil_radius + self.aperture_radius) / self.aperture_distance)

        # Collect only the interactions that produce recoil particles
        recoil_interactions = []
        for interaction in self.interactions:
            if interaction.generates_recoil_particles:
                recoil_interactions.append(interaction)

        # Cache energy-dependent quantities to avoid recomputing them when the same energy is
        # sampled twice in a row (i.e. for monoenergetic inputs).
        previous_incident_energy = None
        angle_distributions = None
        interaction_weights = None
        attenuation = 0.0

        # Generate rays until one passes through aperture
        accepted = False
        # Limit number of rejections to avoid infinite loops
        rejected = 0
        while not accepted and rejected < 100:
            # Sample incident particle energy from weighted distribution
            incident_energy = rng.choice(incident_energies, p=probability_distribution)
            
            # Only recompute energy-dependent cross sections and angle distributions when the
            # incident energy changes
            if incident_energy != previous_incident_energy:
                # Do the cross section calculations
                angle_distributions = {}
                interaction_weights = []
                for interaction in recoil_interactions:
                    angle_distributions[interaction] = interaction.get_angle_distribution(incident_energy)
                    weight = (interaction.get_cross_section(incident_energy) *
                              angle_distributions[interaction].integral(0, max_angle))
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
                rng, angle_distributions[interaction], attenuation, max_angle, y_restriction,
            )

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
                    
                # Calculate foil arrival time from incident particle kinematics
                if self.target_to_foil_distance is not None:
                    if self.incident_particle == 'photon':
                        velocity = 2.998e8  # m/s
                    else:
                        # Non-relativistic: neutron (or other massive incident particle)
                        incident_mass_kg = NEUTRON_MASS * 1.6605e-27
                        velocity = np.sqrt(2 * incident_energy * 1e6 * 1.602e-19 / incident_mass_kg)
                    arrival_time_at_foil = self.target_to_foil_distance / velocity
                    if self.burn_duration is not None:
                        # burn_duration is FWHM; convert to sigma for Gaussian sampling
                        sigma = self.burn_duration / (2 * np.sqrt(2 * np.log(2)))
                        arrival_time_at_foil += rng.normal(0, sigma)
                else:
                    arrival_time_at_foil = 0.0

                accepted = True
            else:
                rejected += 1

        if not accepted:
            raise ValueError("Unable to generate a recoil particle that passes through the aperture.")

        return x0, y0, theta_scatter, phi_scatter, incident_energy, recoil_energy, arrival_time_at_foil
    
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
        if max_workers is None:
            max_workers = mp.cpu_count()
        
        print(f'\nEstimating intrinsic efficiency for {incident_energy:.3f} MeV particles using {max_workers} processes...')
        
        # Limit scattering angles for computational efficiency
        max_angle = np.arctan((self.foil_radius + self.aperture_radius) / self.aperture_distance)
        
        # Calculate scattering probability in foil (non-parallelizable part)
        total_xs = 0
        effective_xs = 0
        angle_distributions = {}
        interaction_weights = []
        recoil_interactions = []
        for interaction in self.interactions:
            total_xs += interaction.get_cross_section(incident_energy)
            if interaction.generates_recoil_particles:
                effective_xs += interaction.get_cross_section(incident_energy)
                angle_distributions[interaction] = interaction.get_angle_distribution(incident_energy)
                interaction_weights.append(
                    interaction.get_cross_section(incident_energy) *
                    angle_distributions[interaction].integral(0, max_angle))
                recoil_interactions.append(interaction)
        angular_factor = effective_xs/sum(interaction_weights)  # the factor by which our sampling density is increased because we're using max_angle
        interaction_weights = np.array(interaction_weights)/sum(interaction_weights)
        
        scattering_efficiency = effective_xs * (1 - np.exp(-total_xs * self.thickness)) / total_xs
        
        # Calculate samples per process
        samples_per_process = num_samples // max_workers
        remaining_samples = num_samples % max_workers
        
        # Execute in parallel
        worker_args = []
        for i in range(max_workers):
            batch_size = samples_per_process + (1 if i < remaining_samples else 0)
            if batch_size > 0:  # Only submit if there's work to do
                # Package all parameters for the worker
                worker_args.append((
                    batch_size,
                    12345 + i * 1000,  # seed_offset
                    recoil_interactions,
                    interaction_weights,
                    angle_distributions,
                    max_angle,
                ))
                
        results = run_concurrently(
            self._calculate_efficiency_batch_worker, worker_args, executor,
            progress_counter_total=num_samples,
            task_title='Calculating geometric acceptance',
        )
        
        # Collect results
        total_accepted = 0
        total_processed = 0
        
        for accepted_count, processed_count in results:
            total_accepted += accepted_count
            total_processed += processed_count
        
        # Calculate final efficiencies
        geometric_efficiency = total_accepted / total_processed / angular_factor if total_processed > 0 else 0.0
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
        interactions: list[Interaction],
        interaction_weights: np.ndarray,
        angle_distributions: dict[Interaction, ProbabilityDistribution],
        max_angle: float,
        progress_counter,
        progress_lock
    ) -> Tuple[int, int]:
        """
        Generate a batch of efficiency samples in a separate process.
        
        Args:
            batch_size: Number of samples to process in this batch
            seed_offset: Random seed offset for this worker
            interactions: Processes by which recoil particles are generated
            interaction_weights: Relative probability of each interaction process
            max_angle: The maximum scattering angle to sample
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
                
                # Sample scattered ray using helper method (no z-sampling for efficiency calculation)
                interaction = rng.choice(interactions, p=interaction_weights)
                x0, y0, _, theta_scatter, phi_scatter = self._sample_scattered_ray(
                    rng, angle_distributions[interaction], max_angle=max_angle, attenuation=-np.inf
                )
                
                # N.B. We do not consider the very small displacement due to foil thickness here

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
            energy_distribution: Distribution of incident particle energies (will be normalized)
            num_recoil_particles: Number of recoil events to simulate
            
        Returns:
            Array of proton energies at foil exit
        """
        recoil_energies = np.zeros(num_recoil_particles)
        
        # Weight distribution by scattering cross section and normalize
        interaction_probability = np.zeros_like(energy_distribution)
        for interaction in self.interactions:
            if interaction.generates_recoil_particles:
                interaction_probability += interaction.get_cross_section(incident_energies)
        weighted_distribution = energy_distribution * interaction_probability
        weighted_distribution = weighted_distribution / np.sum(weighted_distribution)
        
        for i in tqdm(range(num_recoil_particles), desc='Calculating proton energy distribution...'):
            # Generate scattered recoil particle and extract final energy
            _, _, _, _, _, recoil_energy, _ = self.generate_recoil_particle(
                incident_energies,
                weighted_distribution,
                include_kinematics=True, 
                include_stopping_power_loss=True
            )
            
            recoil_energies[i] = recoil_energy
            
        return recoil_energies
