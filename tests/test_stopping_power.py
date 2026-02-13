from numpy import isclose

from MPR_Tools import ConversionFoil

def test_proton_stopping_power():
    CH2_foil = ConversionFoil(
        foil_radius=1.5,
        thickness=50,
        aperture_distance=50,
        aperture_radius=1.5,
        foil_material='CH2',
    )

    # through a moderate foil thickness
    initial_energy = 15
    path_length = 0.162e-2
    final_energy = CH2_foil.calculate_stopping_power_loss(initial_energy, path_length)
    assert isclose(final_energy, 8.0, rtol=1e-3)  # taken from PSTAR's range table for polyethylene

    # ranging all the way out
    path_length = 1000
    final_energy = CH2_foil.calculate_stopping_power_loss(initial_energy, path_length)
    assert isclose(final_energy, 0, rtol=0)


def test_deuteron_stopping_power():
    CD2_foil = ConversionFoil(
        foil_radius=1.5,
        thickness=50,
        aperture_distance=50,
        aperture_radius=1.5,
        foil_material='CH2',
    )
    
    initial_energy = 15
    path_length = 0.162e-2/2  # a deuteron should stop twice as fast as a proton
    final_energy = CD2_foil.calculate_stopping_power_loss(initial_energy, path_length)
    assert isclose(final_energy, 8.0, rtol=1e-3)


def test_electron_stopping_power():
    boron_foil = ConversionFoil(
        foil_radius=1.5,
        thickness=50,
        aperture_distance=50,
        aperture_radius=1.5,
        foil_material='B',
    )
    
    initial_energy = 15
    path_length = 1.6898e-2
    final_energy = boron_foil.calculate_stopping_power_loss(initial_energy, path_length)
    assert isclose(final_energy, 8.0, rtol=1e-3)  # taken from ESTAR's range table


def test_reverse_stopping_power():
    CH2_foil = ConversionFoil(
        foil_radius=1.5,
        thickness=50,
        aperture_distance=50,
        aperture_radius=1.5,
        foil_material='CH2',
    )

    initial_energy = 14
    path_length = 0.162e-2
    roundtrip_energy = CH2_foil.calculate_initial_energy(
        CH2_foil.calculate_stopping_power_loss(
            initial_energy,
            path_length,
        ),
        path_length,
    )
    assert isclose(roundtrip_energy, initial_energy)
