from concurrent.futures.process import ProcessPoolExecutor

from matplotlib import pyplot as plt
from numpy import array, interp, histogram, average, isclose, linspace, exp, where, arange, diff, stack, full
from scipy import integrate

from MPR_Tools import MPRSpectrometer, ConversionFoil, Hodoscope, SpectrometerPlotter, PerformanceAnalyzer


def test_energy_sampling():
    """ test that the monte carlo rays incident energy sampling is proportional to the calculated efficiency """
    # instantiate a gamma spectrometer
    spectrometer = MPRSpectrometer(
        ConversionFoil(
            foil_radius=0.5,
            thickness=10,
            aperture_distance=100,
            aperture_radius=0.5,
            foil_material="CH2",
            aperture_type="circ",
        ),
        "tests/map_ideal.txt",  # dispersion = 1 cm/MeV
        10.0,
        10.0, 18.0,
        Hodoscope(
            channels=stack([
                linspace(0.0, 8.0, 41),  # 2 mm (200 keV) channels
                full(41, 1000.0),
            ], axis=1),
        ),
        run_directory="tests/output"
    )
    analyzer = PerformanceAnalyzer(spectrometer)
    
    # use a trapezoidal incident energy distribution
    spectrum_energies = array([10., 11., 13., 18.])
    spectrum_values = array([0., 0., 1., 1.])
    with ProcessPoolExecutor(max_workers=4) as executor:
        spectrometer.generate_monte_carlo_rays(
            incident_energies=spectrum_energies,
            probability_distribution=spectrum_values,
            num_recoil_particles=10_000,
            continuous_energy_sampling=True,
            executor=executor,
        )
    sampled_energies = spectrometer.input_beam[:, 6]
    sampled_weights = spectrometer.input_beam[:, 7]
    
    # calculate the hodoscope response
    spectrometer.apply_transfer_map()
    hodoscope_response = analyzer.get_channel_response(particle_yield=1.).signal
    energy_bins = linspace(10, 18, 41)  # this lines up with the hodoscope bins
    hodoscope_normalization = (hodoscope_response*diff(energy_bins)).sum()
    
    # calculate the foil efficiency curve
    with ProcessPoolExecutor(max_workers=4) as executor:
        performance_curve = analyzer.generate_performance_curve(
            num_energies=25,
            num_recoils_per_energy=1000,
            executor=executor,
        )
    measured_energies = performance_curve["energy [MeV]"]
    measured_efficiency = performance_curve["total efficiency"]
    
    actual_spectrum = histogram(sampled_energies, weights=sampled_weights, bins=energy_bins, density=True)[0]
    theoretical_spectrum = (interp(energy_bins, measured_energies, measured_efficiency)*
                            interp(energy_bins, spectrum_energies, spectrum_values))
    spectrum_normalization = integrate.trapezoid(theoretical_spectrum, energy_bins)
    
    plt.figure()
    plt.stairs(actual_spectrum, energy_bins, color="C0", fill=True, label="Actual")
    plt.plot(energy_bins, theoretical_spectrum/spectrum_normalization, "C1", label="Theoretical")
    plt.plot(measured_energies, measured_efficiency/spectrum_normalization, "C2--", label="Efficiency")
    plt.stairs(hodoscope_response/hodoscope_normalization, energy_bins, color="C3", linestyle=":", label="At hodoscope")
    plt.legend()
    plt.xlabel("Energy (MeV)")
    plt.tight_layout()
    plt.savefig("tests/output/figures/test_energy_sampling.png")
    plt.close()
    
    assert len(sampled_energies) == 10_000
    assert isclose(
        integrate.trapezoid(energy_bins*theoretical_spectrum, energy_bins)/
            integrate.trapezoid(theoretical_spectrum, energy_bins),
        average(sampled_energies, weights=sampled_weights),
        atol=0.1, rtol=0,
    )


def test_importance_sampling():
    """ make sure we get the same answer with importance sampling turned on as we do with it off """
    # instantiate a gamma spectrometer
    spectrometer = MPRSpectrometer(
        ConversionFoil(
            foil_radius=1.00,
            thickness=500,
            aperture_distance=50,
            aperture_radius=1.25,
            foil_material="Be",
            aperture_type="circ",
        ),
        "tests/map_identity.txt",
        16.2,
        9.5, 17.5,
        Hodoscope(
            channels_left=-10,
            channels_right=10,
            detector_width=200,
            detector_height=200,
        ),
        run_directory="tests/output"
    )
    
    energies = linspace(10, 18, 81)
    # use a two-peaked distribution
    spectrum = exp(-2*(energies - 16.7)**2) + 1/3*exp(-2*(energies - 14)**2)
    # and throw in a uniform background for good measure
    spectrum = spectrum.max() - (spectrum.max() - spectrum)*where(energies < 16.7, 2/3, 1)
    
    # generate rays without importance sampling
    with ProcessPoolExecutor(max_workers=4) as executor:
        spectrometer.generate_monte_carlo_rays(
            incident_energies=energies,
            probability_distribution=spectrum,
            num_recoil_particles=10_000,
            importance_sampling=False,
            executor=executor,
        )
    raw_energies = spectrometer.input_beam[:, 6]
    raw_weights = spectrometer.input_beam[:, 7]
    
    # generate rays with importance sampling
    with ProcessPoolExecutor(max_workers=4) as executor:
        spectrometer.generate_monte_carlo_rays(
            incident_energies=energies,
            probability_distribution=spectrum,
            num_recoil_particles=10_000,
            importance_sampling=True,
            executor=executor,
        )
    fancy_energies = spectrometer.input_beam[:, 6]
    fancy_weights = spectrometer.input_beam[:, 7]
    
    energy_bins = linspace(10, 18, 41)
    
    plt.figure()
    plt.hist(raw_energies, weights=raw_weights, bins=energy_bins, density=True, alpha=0.5, label="Without IS")
    plt.hist(fancy_energies, weights=fancy_weights, bins=energy_bins, density=True, alpha=0.5, label="With IS")
    plt.legend()
    plt.xlabel("Energy (MeV)")
    plt.tight_layout()
    plt.savefig("tests/output/figures/test_importance_sampling.png")
    plt.close()
    
    assert len(raw_energies) == 10_000
    assert len(fancy_energies) == 10_000
    assert all(raw_weights == 1)
    assert not all(fancy_weights == 1)
    assert isclose(
        average(raw_energies, weights=raw_weights),
        average(fancy_energies, weights=fancy_weights),
        atol=0.1, rtol=0,
    )
    
    
def test_response_matrix():
    """ test to make sure the response matrix accurately represents the spectrometer's response """
    # instantiate the spectrometer
    spectrometer = MPRSpectrometer(
        ConversionFoil(
            foil_radius=1.00,
            thickness=5,
            aperture_distance=50,
            aperture_radius=1.25,
            foil_material="CH2",
            aperture_type="circ",
        ),
        "tests/map_first_order.txt",
        14.0,
        10.0, 18.0,
        Hodoscope(
            channels_left=50,
            channels_right=50,
            detector_width=36,
            detector_height=10,
        ),
        run_directory="tests/output"
    )
    
    energies = linspace(10, 18, 41)
    # use a two-peaked distribution
    spectrum = exp(-2*(energies - 16.7)**2) + 1/3*exp(-2*(energies - 14)**2)
    # and throw in a uniform background for good measure
    spectrum = spectrum.max() - (spectrum.max() - spectrum)*where(energies < 16.7, 2/3, 1)
    # normalize it
    spectrum /= spectrum.sum()
    
    # generate a response matrix and linearly estimate the signal
    analyzer = PerformanceAnalyzer(spectrometer)
    with ProcessPoolExecutor(max_workers=4) as executor:
        matrix = analyzer.build_response_matrix(
            energies, executor=executor, num_recoils_per_energy=1000)["CH2"].T
    data_matrix_mult = matrix @ spectrum

    # calculate the signal using Monte Carlo
    with ProcessPoolExecutor(max_workers=4) as executor:
        spectrometer.generate_monte_carlo_rays(
            energies, spectrum, num_recoil_particles=80_000, executor=executor)
        spectrometer.apply_transfer_map(executor=executor)
    data_monte_carlo = analyzer.get_channel_response(None).signal

    plt.figure()
    plt.scatter(arange(1/2, 101), data_matrix_mult, color="C0", label="Matrix multiplication")
    plt.stairs(data_monte_carlo, color="C1", label="Monte Carlo")
    plt.legend()
    plt.xlabel("Channel")
    plt.tight_layout()
    plt.savefig("tests/output/figures/test_response_matrix_multiplication.png")
    
    assert isclose(
        data_matrix_mult.sum(), data_monte_carlo.sum(),
        atol=0, rtol=0.1,
    )
    assert isclose(
        (data_matrix_mult*arange(101)).sum()/data_matrix_mult.sum(),
        (data_monte_carlo*arange(101)).sum()/data_monte_carlo.sum(),
        atol=1.0, rtol=0,
    )
    

def test_plotter():
    """ call all the plot functions and ensures they don't have any errors. """
    # instantiate the spectrometer
    spectrometer = MPRSpectrometer(
        ConversionFoil(
            foil_radius=1.00,
            thickness=5,
            aperture_distance=50,
            aperture_radius=1.25,
            foil_material="CD2",
            aperture_type="circ",
        ),
        "tests/map_first_order.txt",
        14.0,
        9.4, 15.6,
        Hodoscope(
            channels_left=70,
            channels_right=30,
            detector_width=100,
            detector_height=10,
        ),
        run_directory="tests/output"
    )
    
    spectrometer.generate_monte_carlo_rays(array([14.0]), array([1.]), num_recoil_particles=1000)
    spectrometer.apply_transfer_map()

    plotter = SpectrometerPlotter(spectrometer)
    plotter.plot_focal_plane_distribution(include_hodoscope=True)
    plotter.plot_phase_space()
    plotter.plot_position_histogram()
    plotter.plot_monoenergetic_analysis(14)
    plotter.plot_data(14)
    plotter.plot_characteristic_rays(radial_points=1, angular_points=8, aperture_radial_points=1, aperture_angular_points=8)
