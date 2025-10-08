"""
MPR Spectrometer Analysis Toolkit

A comprehensive toolkit for analyzing the performance of Magnetic Proton Recoil (MPR) spectrometers.
"""

from .core.conversion_foil import ConversionFoil
from .core.hodoscope import Hodoscope
from .core.spectrometer import MPRSpectrometer
from .analysis.parameter_sweep import FoilSweeper

__version__ = "1.0.0"
__all__ = ['ConversionFoil', 'Hodoscope', 'MPRSpectrometer', 'FoilSweeper']