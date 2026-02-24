from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from numpy import random, pi, inf, isclose, empty, degrees, array, mean, sqrt, std, linspace, log, exp, hypot, median, \
    tan, cos, sin

from MPR_Tools import ConversionFoil
from MPR_Tools.core.matter_interactions import Interaction, GenericInteraction, ElasticScattering, ComptonScattering, \
    PairProduction, ProbabilityDistribution


def test_nC12_scatter():
    carbon_scatter = GenericInteraction(
        name='(n,C12)',
        target_density=1,
        cross_section_path=Path('MPR_Tools/data/nC12_crosssection.txt'),
    )
    
    assert isclose(carbon_scatter.get_cross_section(14.0), 0.819214e-28, rtol=1e-9, atol=0)
    
    assert carbon_scatter.get_recoil_probability() == 0


def test_np_scatter():
    proton_scatter = ElasticScattering(
        name='(n,p)',
        target_density=1,
        particle_mass=1.007,
        cross_section_path=Path('MPR_Tools/data/np_crosssection.txt'),
        differential_xs_path=Path('MPR_Tools/data/np_diffxs.txt'),
    )
    
    assert isclose(proton_scatter.get_cross_section(14.0), 0.687562e-28, rtol=1e-9, atol=0)
    
    angles, energies = generate_recoil_particles(proton_scatter, 14.0)
    assert all((angles >= 0) & (angles <= pi/2))
    assert all((energies >= 0) & (energies <= 14))
    
    assert proton_scatter.get_recoil_probability() == 1
    
    plt.figure()
    plt.hist(energies, density=True, bins=50)
    plt.xlabel("Proton energy (MeV)")
    plt.ylabel("Proton spectrum (MeV^-1)")
    plt.title(f"Knock-on proton spectrum from 14 MeV neutrons")
    plt.tight_layout()
    plt.savefig("tests/output/test_np_cross_section.png")
    plt.close()


def test_nd_scatter():
    deuteron_scatter = ElasticScattering(
        name='(n,d)',
        target_density=1,
        particle_mass=2.014,
        cross_section_path=Path('MPR_Tools/data/nd_crosssection.txt'),
        differential_xs_path=Path('MPR_Tools/data/nd_diffxs.txt'),
    )
    
    assert isclose(deuteron_scatter.get_cross_section(14.0), 0.6435662e-28, rtol=1e-9, atol=0)
    
    angles, energies = generate_recoil_particles(deuteron_scatter, 14.0)
    assert all((angles >= 0) & (angles <= pi/2))
    assert all((energies >= 0) & (energies <= 12.5))
    
    assert deuteron_scatter.get_recoil_probability() == 1
    
    plt.figure()
    plt.hist(energies, density=True, bins=50)
    plt.xlabel("Deuteron energy (MeV)")
    plt.ylabel("Deuteron spectrum (MeV^-1)")
    plt.title(f"Knock-on deuteron spectrum from 14 MeV neutrons")
    plt.tight_layout()
    plt.savefig("tests/output/test_nd_cross_section.png")
    plt.close()


def test_compton_scatter():
    compton_scatter = ComptonScattering(
        target_density=1,
    )
    
    assert isclose(compton_scatter.get_cross_section(16.7), 0.0348e-28, rtol=0.05, atol=0)  # 3.5 mb is from the exact Klein-Nishina formula
    
    angles, energies = generate_recoil_particles(compton_scatter, 16.7)
    assert all((angles >= 0) & (angles <= pi/2))
    assert all((energies >= 0) & (energies <= 16.45))
    
    assert compton_scatter.get_recoil_probability() == 1
    
    plt.figure()
    plt.hist(degrees(angles), density=True, bins=100)
    plt.xlabel("Electron scattering angle (degrees)")
    plt.ylabel("Electron distribution (degrees^-1)")
    plt.title(f"{compton_scatter.name} electron distribution from 16.7 MeV photons")
    plt.tight_layout()
    plt.savefig("tests/output/test_compton_cross_section.png")
    plt.close()


def test_pair_production():
    pair_production = PairProduction(
        target_density=1,
        charge=5,
    )
    
    assert isclose(pair_production.get_cross_section(16.7), 0.07438e-28, rtol=0.05, atol=0)  # 74 mb is from NIST's XCOM database
    
    angles, energies = generate_recoil_particles(pair_production, 16.7)
    assert all((angles >= 0) & (angles <= pi))
    assert all((energies >= 0) & (energies <= 15.7))
    
    assert pair_production.get_recoil_probability() == 1
    
    plt.figure()
    plt.hist(energies, density=True, bins=50)
    plt.xlabel("Electron energy (MeV)")
    plt.ylabel("Electron spectrum (MeV^-1)")
    plt.title(f"{pair_production.name} electron spectrum from 16.7 MeV photons")
    plt.tight_layout()
    plt.savefig("tests/output/test_pair_production_cross_section.png")
    plt.close()


def test_z_sampling():
    foil = ConversionFoil(
        foil_radius=1.5,
        thickness=50,
        aperture_distance=50,
        aperture_radius=1.5,
        foil_material='B',
    )
    
    scattering_process = ComptonScattering(1)
    scattering_process.calculate_angular_distribution(16.7)
    
    rng = random.default_rng(0)
    N = 10000
    x, y, z0, theta, phi = empty(N), empty(N), empty(N), empty(N), empty(N)
    for i in range(N):
        x[i], y[i], z0[i], theta[i], phi[i], _ = foil._sample_scattered_ray(
            rng,
            scattering_process,
            attenuation=1/20e-6,
            include_kinematics=False,
            max_angle=pi/2,
        )
        
    # correct x and y so that it's birth location, not pass-thru-the-z-plane location
    x0 = x - z0*tan(theta)*cos(phi)
    y0 = y - z0*tan(theta)*sin(phi)
    
    theoretical_median = -50e-6 - 20e-6*log(0.5 + 0.5*exp(-50/20))
    assert all(hypot(x0, y0) <= 1.5e-2)
    assert all((z0 >= -50e-6) & (z0 <= 0e-6))
    assert isclose(median(z0), theoretical_median, atol=3*0.2e-6, rtol=0)  # the std is about 20 um, so random error is about 0.2 um
    
    plt.figure()
    plt.hist2d(x/1e-2, y/1e-2, bins=20)
    plt.axis("square")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title("Recoil distribution (1.5 cm radius foil)")
    plt.tight_layout()
    plt.savefig("tests/output/test_foil_xy_distribution.png")
    plt.close()
    
    plt.figure()
    plt.hist(z0/1e-6, density=True, bins=50)
    plt.xlabel("z (um)")
    plt.ylabel("Distribution (um^-1)")
    plt.title("Recoil distribution (20 um attenuation depth)")
    plt.tight_layout()
    plt.savefig("tests/output/test_foil_z_distribution.png")
    plt.close()


def test_probability_distribution():
    # with spline interpolation, this set of points will make a parabolic probability distribution
    x_table = array([10, 20, 30])
    p_table = array([20, 10, 20])
    distribution = ProbabilityDistribution(x_table, p_table)
    
    assert distribution.integral(20, inf) == 1/2
    
    rng = random.default_rng(0)
    samples = empty(10000)
    for i in range(samples.size):
        samples[i] = distribution.draw(rng, lower=20, upper=inf)
    assert all((samples >= 20) & (samples <= 30))
    assert isclose(mean(samples), 25.625, atol=3*0.003)  # the standard deviation is about 0.3, so random error is about 0.003
    assert isclose(std(samples), sqrt(8.359375), atol=3*0.003)
    
    plt.figure()
    plt.hist(samples, density=True, bins=50)
    x = linspace(20, 30)
    plt.plot(x, .075 + .00075*(x - 20)**2, '--')
    plt.title("Parabolic probability distribution")
    plt.tight_layout()
    plt.savefig("tests/output/test_probability_distribution.png")
    plt.close()


def generate_recoil_particles(interaction: Interaction, input_energy: float, num_particles=10000):
    rng = random.default_rng(0)
    interaction.calculate_angular_distribution(input_energy)
    angles = empty(num_particles)
    energies = empty(num_particles)
    for i in range(num_particles):
        angles[i], energies[i] = interaction.generate_recoil_particle(
            include_kinematics=True, rng=rng, max_angle=pi)
    return angles, energies
