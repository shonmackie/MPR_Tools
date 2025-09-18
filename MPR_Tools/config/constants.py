"""Physical constants and default parameters."""

# Physical constants
AVOGADRO = 6.022e23
NEUTRON_MASS = 1.00867  # amu

# Material properties
FOIL_MATERIALS = {
    'CH2': {
        'particle': 'proton',
        'density': 0.98,  # g/cm^3
        'molecular_weight': 14.0266,  # g/mol
        'hydron_mass': 1.00728  # amu
    },
    'CD2': {
        'particle': 'deuteron', 
        'density': 1.131,  # g/cm^3
        'molecular_weight': 16.0395,  # g/mol
        'hydron_mass': 2.0136  # amu
    }
}

# Default file paths (relative to package)
DEFAULT_DATA_PATHS = {
    'CH2': {
        'srim': '../data/CH2srimdata.txt',
        'cross_section': '../data/np_crosssection.txt',
        'differential_xs': '../data/np_diffxs.txt'
    },
    'CD2': {
        'srim': '../data/CD2srimdata.txt', 
        'cross_section': '../data/nd_crosssection.txt',
        'differential_xs': '../data/nd_diffxs.txt'
    },
    'nc12_cross_section': '../data/nC12_crosssection.txt'
}