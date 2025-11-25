"""Hodoscope detector array implementation."""

from typing import Optional
import numpy as np
import os

class Hodoscope:
    """
    Detector array at the focal plane.
    
    Detectors are assumed to be centered on the final position of the reference ray.
    TODO - add functionality to load hodo config from file
    """
    
    def __init__(
        self, 
        channels_left: int, 
        channels_right: int, 
        detector_width: float, 
        detector_height: float,
        detector_sensitivity_dir: Optional[str] = None
    ):
        """
        Initialize hodoscope detector array.
        
        Args:
            channels_left: Number of channels to the left (low energy)
            channels_right: Number of channels to the right (high energy)  
            detector_width: Total detector width in cm
            detector_height: Total detector height in cm
        """
        self.channels_left = channels_left
        self.channels_right = channels_right
        self.total_channels = channels_left + channels_right + 1  # +1 for central channel
        
        self.detector_width = detector_width * 1e-2   # cm to m
        self.detector_height = detector_height * 1e-2  # cm to m
        
        if detector_sensitivity_dir is not None:
            self.detector_sensitivity_dir = detector_sensitivity_dir
            self._build_detector_sensitivity()
            self.detector_used = True
        else:
            self.detector_used = False
        
        # Calculate detector centers
        self._calculate_channel_edges()
    
    def _calculate_channel_edges(self) -> None:
        """Calculate the center positions of all channels."""        
        # Calculate individual channel width
        self.channel_width = self.detector_width / self.total_channels
        
        # The central channel (index = channels_left) should be centered at x=0
        # So its left edge is at -channel_width/2 and right edge is at +channel_width/2
        central_left_edge = -self.channel_width / 2
        
        # Calculate all channel edges starting from the leftmost
        leftmost_edge = central_left_edge - self.channels_left * self.channel_width
        
        # Create array of all channel edges (N+1 edges for N channels)
        self.channel_edges = np.linspace(leftmost_edge, leftmost_edge + self.detector_width, self.total_channels + 1)
        
        # Calculate channel centers for convenience
        self.channel_centers = (self.channel_edges[:-1] + self.channel_edges[1:]) / 2
    
    @property
    def detector_width_cm(self) -> float:
        """Get detector width in cm."""
        return self.detector_width * 1e2
    
    @property
    def detector_height_cm(self) -> float:
        """Get detector height in cm."""
        return self.detector_height * 1e2
    
    def set_detector_width(self, width_cm: float) -> None:
        """Set detector width in cm."""
        self.detector_width = width_cm * 1e-2
        self._calculate_channel_edges()
    
    def set_detector_height(self, height_cm: float) -> None:
        """Set detector height in cm."""
        self.detector_height = height_cm * 1e-2
    
    def get_channel_centers(self) -> np.ndarray:
        """Get array of detector center positions in meters."""
        return self.channel_centers
    
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