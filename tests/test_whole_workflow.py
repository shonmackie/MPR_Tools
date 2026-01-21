import os

import numpy as np

from MPR_Tools import MPRSpectrometer, ConversionFoil, Hodoscope, SpectrometerPlotter

def test_whole_workflow():
    # instantiate the output directory
    os.makedirs("tests/output", exist_ok=True)

    # instantiate the spectrometer
    nspc = MPRSpectrometer(
        ConversionFoil(
            foil_radius=1.00,
            thickness=5,
            aperture_distance=50,
            aperture_radius=1.25,
            foil_material="CD2",
            aperture_type="circ",
        ),
        "tests/test_map.txt",
        16,
        12, 20,
        Hodoscope(
            channels_left=70,
            channels_right=30,
            detector_width=100,
            detector_height=10,
        ),
        figure_directory="tests/output"
    )

    # use a Dirac delta function input spectrum
    energies = np.linspace(0.0, 20.0, 201)
    birth_spectrum = np.zeros_like(energies)
    birth_spectrum[140] = 1

    # do the math
    nspc.generate_monte_carlo_rays(
        energies, birth_spectrum,
        num_hydrons=100,
    )
    nspc.apply_transfer_map()
    nspc.read_beams()
    nspc.bin_hodoscope_response()

    plotter = SpectrometerPlotter(nspc)
    plotter.plot_focal_plane_distribution(include_hodoscope=True)
    plotter.plot_phase_space()
    plotter.plot_characteristic_rays()
    plotter.plot_simple_position_histogram()
    plotter.plot_monoenergetic_analysis(2.4, 2.4, 0.2)
    plotter.plot_data(2.4)
