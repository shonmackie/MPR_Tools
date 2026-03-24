"""Hodoscope detector array implementation."""

from pathlib import Path
from typing import Optional, Union, Literal
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class Hodoscope:
    """
    Detector array at the focal plane.
    
    Detectors are assumed to be centered on the final position of the reference ray.
    """
    
    def __init__(
        self,
        channels: Optional[Union[str, np.ndarray]] = None,
        channels_left: Optional[int] = None,
        channels_right: Optional[int] = None,
        detector_width: Optional[float] = None,
        detector_height: Optional[float] = None,
        detector_material: Optional[str] = None,
        detector_thickness: Optional[float] = None
    ):
        """
        Initialize hodoscope detector array in one of three ways:
            as an array of identical detectors on either side of x=0 (the final position of the reference ray),
            from an array specifying detector positions and dimensions, or
            from a file containing such an array.

        If the first method is used, the arguments `channels_left`, `channels_right`, `detector_width`, and
        `detector_height` are used to generate the array of edge positions and channel heights.

        If one of the other two methods is used, the array or filename is passed to `channels`.
        The array or the contents of the file should have two columns representing
        the left edge and height of each channel (in cm); and a number of rows equal to one plus the number of channels.
        The right edge of each channel is assumed to be the left edge of the next one,
        and the right edge of the last channel is stored in the last row (the last element of the last row is ignored).
        
        Args:
            channels: Either an (n+1)×2 array containing information about the detector array,
                      or a filename pointing to a CSV file containing such an array. If `channels` is passed, then none
                      of `channels_left`, `channels_right`, `detector_width`, or `detector_height` should be passed.
            channels_left: Number of channels to the left of the central channel (low energy).
                           If this is passed, then `channels` should not be passed.
            channels_right: Number of channels to the right of the central channel (high energy).
                            If this is passed, then `channels` should not be passed.
            detector_width: Total detector width in cm.
                            If this is passed, then `channels` should not be passed.
            detector_height: Total detector height in cm.
                             If this is passed, then `channels` should not be passed.
            detector_material: The material of the detector. Only used for detector sensitivity calculation.
        """
        # Calculate detector centers
        if channels is not None:
            if channels_left is not None or channels_right is not None or detector_width is not None or detector_height is not None:
                raise ValueError('If channels is an array or filename, no other channel dimension arguments should be passed.')
            if type(channels) is str:
                channels = np.loadtxt(channels, delimiter=',', encoding='utf-8')
            self._calculate_channel_edges_from_array(channels)
        elif channels_left is not None and channels_right is not None and detector_width is not None and detector_height is not None:
            self._calculate_channel_edges_from_parameters(channels_left, channels_right, detector_width, detector_height)
        else:
            raise ValueError("There isn't enough information to constrain the channel dimensions.  Please pass either `channels` or all four of the channel dimension parameters.")

        if detector_material and detector_thickness:
            self.detector_material = detector_material
            self.detector_thickness = detector_thickness
            self._build_detector_sensitivity()
            self.detector_used = True
        else:
            self.detector_used = False

    def _calculate_channel_edges_from_array(self, data: np.ndarray) -> None:
        """Calculate the dimensions and center positions of all channels"""
        # Extract channel x-coordinates from the left column
        self.channel_edges = data[:, 0] * 1e-2  # cm to m
        self.channel_centers = (self.channel_edges[:-1] + self.channel_edges[1:]) / 2
        self.detector_width = self.channel_edges[-1] - self.channel_edges[0]

        # Create array of channel heights from the right column (ignoring the last row)
        self.channel_heights = data[:-1, 1] * 1e-2

        self.total_channels = data.shape[0] - 1

    def _calculate_channel_edges_from_parameters(
            self,
            channels_left: int,
            channels_right: int,
            detector_width: float,
            detector_height: float
    ) -> None:
        """Calculate the center positions of all channels."""
        self.detector_width = detector_width * 1e-2  # cm to m
        detector_height = detector_height * 1e-2  # cm to m

        self.total_channels = channels_left + channels_right + 1  # +1 for central channel

        # Calculate individual channel width
        channel_width = self.detector_width / self.total_channels
        
        # The central channel (index = channels_left) should be centered at x=0
        # So its left edge is at -channel_width/2 and right edge is at +channel_width/2
        central_left_edge = -channel_width / 2
        
        # Calculate all channel edges starting from the leftmost
        leftmost_edge = central_left_edge - channels_left * channel_width
        
        # Create array of all channel edges (N+1 edges for N channels)
        self.channel_edges = np.linspace(leftmost_edge, leftmost_edge + self.detector_width, self.total_channels + 1)
        
        # Calculate channel centers for convenience
        self.channel_centers = (self.channel_edges[:-1] + self.channel_edges[1:]) / 2

        # Channel heights are all the same
        self.channel_heights = np.full(self.total_channels, detector_height)

    @property
    def detector_width_cm(self) -> float:
        """Get detector length in cm."""
        return self.detector_width * 1e2

    def get_channel_for_position(self, x_position: float) -> Optional[int]:
        """
        Get the detector channel number for a given x position.
        
        Args:
            x_position: X position in meters
            
        Returns:
            Channel number (0-indexed) or None if outside detector array
        """
        # Check if position is within hodoscope bounds
        if x_position < self.channel_edges[0] or x_position >= self.channel_edges[-1]:
            return None
        
        # Find which channel the position falls into
        channel = np.searchsorted(self.channel_edges[1:], x_position)
        return int(channel)
    
    def _build_detector_sensitivity(self) -> None:
        """
        Build detector sensitivity from a CSV in data/detector_sensitivity/.
        The sensitivity matrices are pre-generated in a Geant4 simulation.
        The file is named '{detector_material}_sensitivity.csv' and must have columns:
        particle, energy_MeV, detector_material, detector_thickness_mm, mean_eDep_MeV.
        Sensitivity is defined as mean_eDep_MeV / energy_MeV (fraction of incident energy deposited).
        Each particle is stored as a scipy interp1d object over energy, with thickness interpolated
        from the available values in the CSV.
        """
        data_dir = Path(__file__).parent.parent / "data"
        csv_path = data_dir / "detector_sensitivity" / f"{self.detector_material}_sensitivity.csv"
        df = pd.read_csv(csv_path)
        df['sensitivity'] = df['mean_eDep_MeV'] / df['energy_MeV']

        self.sensitivity = {}
        for particle, particle_df in df.groupby('particle'):
            energies = np.sort(particle_df['energy_MeV'].unique())
            sensitivities = np.empty(len(energies))
            for i, energy in enumerate(energies):
                energy_df = particle_df[particle_df['energy_MeV'] == energy].sort_values('detector_thickness_mm')
                thicknesses = energy_df['detector_thickness_mm'].values
                sens_values = energy_df['sensitivity'].values
                if self.detector_thickness is None or len(thicknesses) == 1:
                    sensitivities[i] = sens_values[0]
                else:
                    sensitivities[i] = np.interp(self.detector_thickness, thicknesses, sens_values)
            self.sensitivity[particle] = interp1d(energies, sensitivities, bounds_error=False, fill_value=(sensitivities[0], sensitivities[-1]))
              
    def get_detector_response(
        self,
        energies: np.ndarray,
        particle: Literal['proton', 'deuteron', 'neutron', 'gamma']
    ) -> np.ndarray:
        """
        Get the detector response for a given particle and energy.
        """
        if self.detector_used:
            sensitivity_efficiencies = self.sensitivity[particle](energies)
        else:
            sensitivity_efficiencies = np.ones(len(energies))

        return sensitivity_efficiencies
    
    def get_total_background(
        self,
        neutron_energy: Optional[float] = None,
        photon_energy: Optional[float] = None,
        neutron_flux: Optional[float] = None,
        photon_flux: Optional[float] = None,
        neutron_background_file: Optional[str] = None,
        photon_background_file: Optional[str] = None
    ) -> float:
        """
        Calculate the total background signal deposited in the detector.

        Each particle type (neutron, photon) can be specified either as a scalar flux at a single
        energy, or as a background file containing an energy spectrum. If neither is provided for a
        particle type, its contribution is zero.

        Args:
            neutron_energy: Neutron energy in MeV (used with neutron_flux for scalar input).
            photon_energy: Photon energy in MeV (used with photon_flux for scalar input).
            neutron_flux: Neutron flux in particles/cm^2-source (scalar).
            photon_flux: Photon flux in particles/cm^2-source (scalar).
            neutron_background_file: Path to CSV with columns 'energy' (MeV) and 'energy_mean'
                (particles/cm^2-source) describing the neutron flux spectrum.
            photon_background_file: Path to CSV with columns 'energy' (MeV) and 'energy_mean'
                (particles/cm^2-source) describing the photon flux spectrum.

        Returns:
            float: Total background signal in weighted particles/cm^2-source, where each particle
                is weighted by its detection sensitivity (mean_eDep / incident_energy).
        """
        if not self.detector_used:
            raise ValueError("Detector not used; cannot calculate background.")

        def _contribution_from_file(particle: str, filepath: str) -> float:
            df = pd.read_csv(filepath)
            energies = df['energy'].to_numpy()
            flux = df['energy_mean'].to_numpy()  # [particles/cm^2-source] per energy bin
            sens = self.sensitivity[particle](energies)
            return float(np.dot(flux, sens))  # [weighted_particles/cm^2-source]

        def _contribution_from_scalar(particle: str, energy: float, flux: float) -> float:
            return flux * float(self.sensitivity[particle](energy))

        total = 0.0

        if neutron_background_file is not None:
            total += _contribution_from_file('neutron', neutron_background_file)
        elif neutron_energy is not None and neutron_flux is not None:
            total += _contribution_from_scalar('neutron', neutron_energy, neutron_flux)

        if photon_background_file is not None:
            total += _contribution_from_file('gamma', photon_background_file)
        elif photon_energy is not None and photon_flux is not None:
            total += _contribution_from_scalar('gamma', photon_energy, photon_flux)

        return total
