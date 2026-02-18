# MPR_Tools

This is a python package designed for the design and analysis of Magnetic Particle Recoil (MPR) based spectrometers.
An MPR spectrometer measures a neutral input particle (neutrons or photons)
by converting it to a charged recoil particle (protons, deuterons, or electrons) via elastic scattering,
and then dispersing the recoil particles via magnetic fields onto a detector.

The code is built around objects that represent the various components of an MPR spectrometer, including the foil and aperture (termed acceptance), the ion/electron optical magnet system, and the hodoscope in the focal plane.
Each object has functionality that numerically simulates the action of that subsystem on the protons which constitute the signal under study.

This package is setup so that you can easily add MPR_Tools to your python path and import it in other scripts elsewhere on your machine. To do this, navigate to the directory this README is in, and run in the command prompt "pip install -e ."
This will automatically add MPR_Tools as an importable package.

There are also tests meant to be run using Pytest.
After pip installing MPR_Tools to your python path, navigate to the the directory this README is in and run `python -m pytest`.
If something is wrong with the code, they'll raise an error.
It also saves a bunch of plots to `tests/output/`.  Check those out to make sure they look okay and normal.
Except `characteristic_rays.png`; that one never looks normal and I don't know why.
