"""Hodoscope detector array implementation."""

from typing import Optional
import numpy as np
import os

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
        detector_sensitivity_dir: Optional[str] = None
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
            detector_sensitivity_dir: Name of a directory containing data files for the sensitivity
                                      of the individual detectors to radiation
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

        if detector_sensitivity_dir is not None:
            self.detector_sensitivity_dir = detector_sensitivity_dir
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
        Build detector sensitivity from files in the specified directory. Sensitivity files should
        include the detector type and the sensitivity for protons, deuterons, neutrons, and photons at various energies.
        """
        # Parse directory name
        self.detector_type, self.thickness = self.detector_sensitivity_dir.split("/")[-1].split('_')[:2]
        # Thickness will be in mm, get it as a float
        self.thickness = float(self.thickness.replace("mm", ""))
        
        self.sensitivity = {}
        
        # Parse the detector sensitivity files
        for file in os.listdir(self.detector_sensitivity_dir):
            if file.endswith(".txt"):
                # Parse the file name to get the detector type and energy
                particle = file.split("_")[0]
                
                # Load the sensitivity data
                sensitivity_data = np.loadtxt(f'{self.detector_sensitivity_dir}/{file}', delimiter=',')
                energy = sensitivity_data[:, 0] # in MeV
                yields = sensitivity_data[:, 1] # in # photons/particle
                
                self.sensitivity[particle] = {
                    'energy': energy,
                    'yields': yields
                }
    
    def get_total_background(self) -> float:
        """
        Calculate the density of background photons generated in the detector.
            
        Returns:
            float: Total background density in photons/cm^2-source
        """
        # Assume background neutrons and gammas are uniformly distributed across the detector
        if not self.detector_used:
            raise ValueError("Detector not used; cannot calculate background density map.")
        
        # TODO: implement actual background with energy distribution
        total_photons = 1.4e-14 # Photons/cm^2-source
        total_neutrons = 1.1e-14 # Neutrons/cm^2-source
        
        # Randomly sample background neutrons and gammas across the detector area with uniform distribution across energy range
        neutron_sensitivity = self.sensitivity['neutron']
        gamma_sensitivity = self.sensitivity['gamma']
        
        average_neutron_yield = np.trapezoid(neutron_sensitivity['yields'], neutron_sensitivity['energy']) / (neutron_sensitivity['energy'][-1] - neutron_sensitivity['energy'][0])
        average_gamma_yield = np.trapezoid(gamma_sensitivity['yields'], gamma_sensitivity['energy']) / (gamma_sensitivity['energy'][-1] - gamma_sensitivity['energy'][0])
        
        # Calculate total background in units of photons/cm^2-source
        total_background = total_neutrons * average_neutron_yield + total_photons * average_gamma_yield
        
        return total_background