import numpy.testing as npt
from MPR_Tools import ConversionFoil

basic_foil = ConversionFoil(
    foil_radius=1.5,
    thickness=50,
    aperture_distance=50,
    aperture_radius=1.5,
    foil_material='CH2',
    aperture_type='circ',
)

def test_stopping_power():
    # through a typical foil thickness
    initial_energy = 14
    path_length = 50e-6
    final_energy = basic_foil.calculate_stopping_power_loss(initial_energy, path_length)
    npt.assert_allclose(final_energy, 13.813, rtol=1e-3)  # calculated with scipy.integrate.odeint using the data in this repo

    # ranging all the way out
    path_length = 1000
    final_energy = basic_foil.calculate_stopping_power_loss(initial_energy, path_length)
    npt.assert_allclose(final_energy, 0, rtol=0)

def test_reverse_stopping_power():
    initial_energy = 14
    path_length = 50e-6
    roundtrip_energy = basic_foil.calculate_initial_energy(
        basic_foil.calculate_stopping_power_loss(
            initial_energy,
            path_length,
        ),
        path_length,
    )
    npt.assert_allclose(roundtrip_energy, initial_energy)
