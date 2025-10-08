"""Utility functions for MPR analysis."""

import numpy as np

def calculate_fwhm(data: np.ndarray, domain: np.ndarray) -> float:
    """
    Calculate the full width at half maximum (FWHM) of data over a given domain.
    
    Args:
        data: Array of data values
        domain: Array of domain values corresponding to data
        
    Returns:
        Full width at half maximum
    """
    half_max = np.max(data) / 2.0
    
    # Find where function crosses half_max line (sign changes)
    diff = np.sign(half_max - data[:-1]) - np.sign(half_max - data[1:])
    
    # Find leftmost and rightmost crossings
    left_indices = np.where(diff > 0)[0]
    right_indices = np.where(diff < 0)[0]
    
    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0.0
        
    left_idx = left_indices[0]
    right_idx = right_indices[-1]
    
    return domain[right_idx] - domain[left_idx]