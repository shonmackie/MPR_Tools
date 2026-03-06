import os
from concurrent.futures.process import ProcessPoolExecutor

from MPR_Tools import MPRSpectrometer, ConversionFoil, Hodoscope, SpectrometerPlotter, PerformanceAnalyzer

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

    analyzer = PerformanceAnalyzer(nspc)
    with ProcessPoolExecutor(max_workers=4) as executor:
        analyzer.generate_performance_curve(
            num_energies=3,
            num_recoils_per_energy=100,
            num_efficiency_samples=100,
            executor=executor,
        )

    plotter = SpectrometerPlotter(nspc)
    plotter.plot_focal_plane_distribution(include_hodoscope=True)
    plotter.plot_phase_space()
    plotter.plot_characteristic_rays(radial_points=1, angular_points=8, aperture_radial_points=1, aperture_angular_points=8)
    plotter.plot_simple_position_histogram()
    plotter.plot_monoenergetic_analysis(14, 0, 0.1)
    plotter.plot_data(14)
