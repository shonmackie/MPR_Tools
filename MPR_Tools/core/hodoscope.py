"""Hodoscope detector array implementation."""

from typing import Optional
import numpy as np

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
        detector_height: float
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