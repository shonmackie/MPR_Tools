from numpy import all, random, isclose, ndarray, array, radians, arcsin

from MPR_Tools import MPRSpectrometer, ConversionFoil, Hodoscope
from MPR_Tools.core.spectrometer import ray_cylinder_intersection


def test_perfect_map():
    rng = random.default_rng(seed=0)
    input_rays = rng.normal(size=(100, 6))
    output_rays = apply_transfer_map(input_rays, "tests/map_identity.txt")
    assert all(isclose(output_rays, input_rays[:, 0:5]))
    

def test_first_order_map():
    input_rays = array([
        [0, 0, 0, 0, 0],  # central ray
        [0.1, 0, 0, 0, 0],  # x-perturbation
        [0, 0.1, 0, 0, 0],  # a-perturbation
        [0, 0, 0.1, 0, 0],  # y-perturbation
        [0, 0, 0, 0.1, 0],  # b-perturbation
        [0, 0, 0, 0, 0.1],  # dE-perturbation
        [0.1, 0.1, 0, 0, 0],  # combined x- and a-perturbation
    ])
    output_rays = apply_transfer_map(input_rays, "tests/map_first_order.txt")
    expected_output_rays = array([
        [0, 0, 0, 0, 0],  # central ray
        [-0.1, -0.2, 0, 0, 0],  # x-perturbation
        [0, -0.1, 0, 0, 0],      # a-perturbation
        [0, 0, 0.1, 0, 0],       # y-perturbation
        [0, 0, 0.02, 0.1, 0],   # b-perturbation
        [0.05, 0, 0, 0, 0.1],  # dE-perturbation
        [-0.1, -0.3, 0, 0, 0],  # combined x- and a-perturbation
    ])
    assert all(isclose(output_rays, expected_output_rays))


def test_second_order_map():
    input_rays = array([
        [0.1, 0, 0, 0, 0],
    ])
    output_rays = apply_transfer_map(input_rays, "tests/map_second_order.txt")
    expected_output_rays = array([
        [-0.2, -0.4, 0, 0, 0],  # n.b. the second order coefficients are not second derivatives.  so it goes like (x|xx)*x^2, not like 1/2! (x|xx)*x^2.
    ])
    assert all(isclose(output_rays, expected_output_rays))


def test_shifted_detector():
    input_rays = array([
        [0, 0, 0, 0, 0, 0],  # central ray
        [.10, 0, 0, 0, 0, 0],  # x-perturbation
        [0, 9/41, 0, 0, 0, 0],  # a-perturbation (I'm using 9/41 because 9-40-41 is a pythagorean triple)
        [0, 0, .10, 0, 0, 0],  # y-perturbation
        [0, 0, 0, 9/41, 0, 0],  # b-perturbation
        [0, 0, 0, 0, 0, 0.1],  # dE-perturbation
    ])
    output_rays = ray_cylinder_intersection(
        input_rays, +.40, 0, 0,  # making the path length 40 cm so that the diagonal path length is 41 cm
        reference_energy=10,
        particle_mass=1e-7,
    )
    expected_output_rays = array([
        [0, 0, 0, 0, .40, 0],  # central ray experiences longer time-of-flight
        [.10, 0, 0, 0, .40, 0],  # x-perturbation is transferred to the new plane
        [.09, 9/41, 0, 0, .41, 0],  # a-perturbation causes displacement in x and an even longer time-of-flight
        [0, 0, 0.1, 0, .40, 0],  # y-perturbation is transferred to the new plane
        [0, 0, 0.09, 9/41, .41, 0],  # b-perturbation causes displacement in y and a longer time-of-flight
        [0, 0, 0, 0, .40 - 2e-18, 0.1],  # dE doesn't really change time-of-flight because it's fully relativistic
    ])
    assert all(isclose(output_rays, expected_output_rays))
    
    
def test_tilted_detector():
    input_rays = array([
        [0, 0, 0, 0, 0, 0],  # central ray
        [.04, 5/13, 0, 0, 0, 0],  # positive x-displacement, and angled slightly along the detector (using 5/13 because 5-12-13 is a pythagorean triple)
        [-.04, 5/13, 0, 0, 0, 0],  # negative x-displacement, and angled slightly along the detector
        [0, 0, .10, 0, 0, 0],  # y-perturbation has no effect since the tilt is about the y-axis
    ])
    output_rays = ray_cylinder_intersection(
        input_rays, 0, radians(53.130102), 0,  # this is the big 3-4-5 angle
        reference_energy=10,
        particle_mass=1e-7,
    )
    expected_output_rays = array([
        [0, 4/5, 0, 0, 0, 0],  # central ray hits with a different angle of incidence
        [.15, 63/65, 0, 0, .13, 0],  # the ray has to go farther, and has a much steeper angle of incidence
        [-.15, 63/65, 0, 0, -.13, 0],  # the ray has to backtrack, and has a much steeper angle of incidence
        [0, 4/5, .10, 0, 0, 0],  # y-perturbation has no effect since the tilt is about the y-axis
    ])
    assert all(isclose(output_rays, expected_output_rays))


def test_curved_detector():
    input_rays = array([
        [0, 0, 0, 0, 0, 0],  # central ray
        [.09, 0, 0, 0, 0, 0],  # x-perturbation
        [0, 9/41, 0, 0, 0, 0],  # a-perturbation
        [0, 0, .10, 0, 0, 0],  # y-perturbation
    ])
    output_rays = ray_cylinder_intersection(
        input_rays, .41, 0, -1/.41,  # an arc of radius 41 cm centered on the origin
        reference_energy=10,
        particle_mass=1e-7,
    )
    expected_output_rays = array([
        [0, 0, 0, 0, .41, 0],  # central ray experiences a longer time-of-flight
        [.41*arcsin(9/41), -9/41, 0, 0, .40, 0],  # offset ray hits detector slightly sooner and has slightly skewed incidence
        [.41*arcsin(9/41), 0, 0, 0, .41, 0],  # this is a radius so it has the same time-of-flight and incidence angle
        [0, 0, .10, 0, .41, 0],  # y-perturbation has no effect since the bend is about the y-axis
    ])
    assert all(isclose(output_rays, expected_output_rays))


def test_slightly_curved_detector():
    input_rays = array([
        [0, 0, 0, 0, 0, 0],  # central ray
        [.10, 0, 0, 0, 0, 0],  # x-perturbation
    ])
    output_rays = ray_cylinder_intersection(
        input_rays, 0, 0, -1e-20,  # an arc with such a slight curvature that it could conceivably cause roundoff problems
        reference_energy=10,
        particle_mass=1e-7,
    )
    expected_output_rays = array([
        [0, 0, 0, 0, 0, 0],
        [.10, -1e-22, 0, 0, -5e-23, 0],  # don't worry about the fact that the tolerance is too coarse to catch this, because I don't care if it actually has roundoff problems or not; I just care that it doesn't freak out.
    ])
    assert all(isclose(output_rays, expected_output_rays))


def apply_transfer_map(input_beam: ndarray, transfer_map_path: str):
    spectrometer = MPRSpectrometer(
        conversion_foil=ConversionFoil(1, 1, 10, 1),
        reference_energy=14,
        min_energy=10, max_energy=16,
        transfer_map_path=transfer_map_path,
        hodoscope=Hodoscope(array([[-50, 1], [50, 1]])),
    )
    spectrometer.input_beam = input_beam
    spectrometer.apply_transfer_map(executor=None, max_workers=1)
    return spectrometer.output_beam
