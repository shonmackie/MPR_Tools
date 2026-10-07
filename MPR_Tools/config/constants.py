"""Physical constants and default parameters."""

# Physical constants
AVOGADRO = 6.022e23
NEUTRON_MASS = 1.00867  # amu
ELECTRON_REST_ENERGY = 0.510999  # MeV
CLASSICAL_ELECTRON_RADIUS = 2.81794032e-15  # m
FINE_STRUCTURE_CONSTANT = 1/137.035999
LIGHT_SPEED = 2.99792458e8  # m/s
MASS_TO_MEV = 931.494  # MeV/amu

# Material properties
FOIL_MATERIALS = {
    'CH2': {
        'density': 0.98,  # g/cm^3
        'molecular_weight': 14.0266,  # g/mol
        'particle_mass': 1.00728,  # amu
        'ion_stopping_power': 'CH2srimdata.txt',
        'electron_stopping_power': 'CH2_estar.txt',
        'interactions': [
            {
                'type': 'elastic_scattering',
                'incident_particle': 'neutron',
                'recoil_particle': 'proton',
                'target_abundance': 2,
                'total_cross_section': 'np_crosssection.txt',
                'diff_cross_section': 'np_diffxs.txt'
            },
            {
                'type': 'generic',
                'incident_particle': 'neutron',
                'name': '(n,C12) elastic',
                'target_abundance': 1,
                'total_cross_section': 'nC12_crosssection.txt',
            },
            {
                'type': 'compton_scattering',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 8,
            },
            {
                'type': 'pair_production',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 1,
                'target_charge': 6,
            },
            {
                'type': 'pair_production',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 2,
                'target_charge': 1,
            },
        ]
    },
    'CD2': {
        'density': 1.131,  # g/cm^3
        'molecular_weight': 16.0395,  # g/mol
        'particle_mass': 2.0136,  # amu
        'ion_stopping_power': 'CD2srimdata.txt',
        'electron_stopping_power': 'CD2_estar.txt',
        'interactions': [
            {
                'type': 'elastic_scattering',
                'incident_particle': 'neutron',
                'recoil_particle': 'deuteron',
                'target_abundance': 2,
                'total_cross_section': 'nd_crosssection.txt',
                'diff_cross_section': 'nd_diffxs.txt'
            },
            {
                'type': 'generic',
                'incident_particle': 'neutron',
                'recoil_particle': 'deuteron',
                'name': '(n,C12)',
                'target_abundance': 1,
                'total_cross_section': 'nC12_crosssection.txt',
            },
            {
                'type': 'compton_scattering',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 8,
            },
            {
                'type': 'pair_production',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 1,
                'target_charge': 6,
            },
            {
                'type': 'pair_production',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 2,
                'target_charge': 1,
            },
        ]
    },
    'LiH': {
        'density': 0.78,  # g/cm^3
        'molecular_weight': 7.95,  # g/mol
        'particle_mass': 5.4858e-4,  # amu
        'electron_stopping_power': 'LiH_estar.txt',
        'interactions': [
            {
                'type': 'compton_scattering',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 4,
            },
            {
                'type': 'pair_production',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 1,
                'target_charge': 1,
            },
            {
                'type': 'pair_production',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 1,
                'target_charge': 3,
            },
        ]
    },
    'Be': {
        'density': 1.845,  # g/cm^3
        'molecular_weight': 9.0122,  # g/mol
        'particle_mass': 5.4858e-4,  # amu
        'electron_stopping_power': 'Be_estar.txt',
        'interactions': [
            {
                'type': 'compton_scattering',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 4,
            },
            {
                'type': 'pair_production',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 1,
                'target_charge': 4,
            },
        ]
    },
    'B': {
        'density': 2.35,  # g/cm^3
        'molecular_weight': 10.81,  # g/mol
        'particle_mass': 5.4858e-4,  # amu
        'electron_stopping_power': 'B_estar.txt',
        'interactions': [
            {
                'type': 'compton_scattering',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 5,
            },
            {
                'type': 'pair_production',
                'incident_particle': 'photon',
                'recoil_particle': 'electron',
                'target_abundance': 1,
                'target_charge': 5,
            },
        ]
    },
}
