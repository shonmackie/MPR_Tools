# MPR_Tools

This is a python package designed for the deisgn and analysis of Magnetic Proton Recoil based spectrometers. 

The code is built around objects that represent the various components of an MPR spectrometer, including the foil and aperture (termed acceptance), the ion optical magnet system, and the hodoscope in the focal plane.
Each object has functionality that numerically simulates the action of that subsystem on the protons which constitute the signal under study.

This package is setup so that you can easily add MPR_Tools to your python path and import it in other scripts elsewhere on your machine. To do this, navigate to the directory this README is in, and run in the command prompt "pip install -e ."
This will automatically add MPR_Tools as an importable package.

There are also tests meant to be run using Pytest.
Run them with `python -m pytest` from the root project directory.
