"""Physical constants and default parameters."""

# Physical constants
AVOGADRO = 6.022e23
NEUTRON_MASS = 1.00867  # amu
ELECTRON_REST_ENERGY = 0.510999  # MeV
CLASSICAL_ELECTRON_RADIUS = 2.81794032e-15  # m

# Material properties
FOIL_MATERIALS = {
    'CH2': {
        'input_particle': 'neutron',
        'particle': 'proton',
        'density': 0.98,  # g/cm^3
        'molecular_weight': 14.0266,  # g/mol
        'particle_mass': 1.00728,  # amu
        'interactions': [
            {
                'type': 'elastic_scattering',
                'target_abundance': 2,
                'total_cross_section': 'np_crosssection.txt',
                'diff_cross_section': 'np_diffxs.txt'
            },
            {
                'type': 'generic',
                'name': '(n,C12) elastic',
                'target_abundance': 1,
                'total_cross_section': 'nC12_crosssection.txt',
            },
        ]
    },
    'CD2': {
        'input_particle': 'neutron',
        'particle': 'deuteron',
        'density': 1.131,  # g/cm^3
        'molecular_weight': 16.0395,  # g/mol
        'particle_mass': 2.0136,  # amu
        'interactions': [
            {
                'type': 'elastic_scattering',
                'target_abundance': 2,
                'total_cross_section': 'nd_crosssection.txt',
                'diff_cross_section': 'nd_diffxs.txt'
            },
            {
                'type': 'generic',
                'name': '(n,C12)',
                'target_abundance': 1,
                'total_cross_section': 'nC12_crosssection.txt',
            },
        ]
    },
    'LiH': {
        'input_particle': 'photon',
        'particle': 'electron',
        'density': 0.78,  # g/cm^3
        'molecular_weight': 7.95,  # g/mol
        'particle_mass': 5.4858e-4,  # amu
        'interactions': [
            {
                'type': 'compton_scattering',
                'target_abundance': 4,
            },
        ]
    },
    'Be': {
        'input_particle': 'photon',
        'particle': 'electron',
        'density': 1.845,  # g/cm^3
        'molecular_weight': 9.0122,  # g/mol
        'particle_mass': 5.4858e-4,  # amu
        'interactions': [
            {
                'type': 'compton_scattering',
                'target_abundance': 4,
            },
        ]
    },
    'B': {
        'input_particle': 'photon',
        'particle': 'electron',
        'density': 2.35,  # g/cm^3
        'molecular_weight': 10.81,  # g/mol
        'particle_mass': 5.4858e-4,  # amu
        'interactions': [
            {
                'type': 'compton_scattering',
                'target_abundance': 5,
            },
        ]
    },
}

# Data file paths (relative to data/ folder)
DATA_PATHS = {
    'CH2_proton_stopping': 'CH2srimdata.txt',
    'np_cross_section': 'np_crosssection.txt',
    'np_diff_cross_section': 'np_diffxs.txt',
    'CD2_proton_stopping': 'CD2srimdata.txt',
    'nd_cross_section': 'nd_crosssection.txt',
    'nd_diff_cross_section': 'nd_diffxs.txt',
    'nc12_cross_section': 'nC12_crosssection.txt',
    'LiH_electron_stopping': 'Li_estar.txt',
    'Be_electron_stopping': 'Be_estar.txt',
    'B_electron_stopping': 'B_estar.txt',
}
