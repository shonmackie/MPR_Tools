"""Hodoscope detector array implementation."""

from pathlib import Path
from typing import Optional, Tuple, Union, Literal
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
        detector_thickness: Optional[float] = None,
        use_time_gating: bool = False,
        y_center: float = 0.0,
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
            use_time_gating: When True, background CSV files are expected to contain a 'time' column,
                             and get_background() will get the energy deposited as a function of time.
                             When False (default), all background calculations integrate over the full time axis.
            y_center: Vertical center of the detector in cm (default 0). The y-acceptance of each channel is
                      [y_center - channel_height/2, y_center + channel_height/2].
        """
        # Calculate detector centers
        if channels is not None:
            if channels_left or channels_right or detector_width or detector_height:
                raise ValueError('If channels is an array or filename, no other channel dimension arguments should be passed.')
            if type(channels) is str:
                channels = np.loadtxt(channels, delimiter=',', encoding='utf-8')
            self._calculate_channel_edges_from_array(channels)
        elif channels_left and channels_right and detector_width and detector_height:
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

        # Whether to use time-resolved background data for per-channel time gating.
        self.use_time_gating = use_time_gating

        # Vertical center of the detector (meters). Channel y-acceptance is [y_center ± height/2].
        self.y_center = y_center * 1e-2  # cm to m

    def _calculate_channel_edges_from_array(self, data: np.ndarray) -> None:
        """Calculate the dimensions and center positions of all channels"""
        # Extract channel x-coordinates from the left column
        self.channel_edges = data[:, 0] * 1e-2  # cm to m
        self.channel_centers = (self.channel_edges[:-1] + self.channel_edges[1:]) / 2
        self.channel_widths = np.diff(self.channel_edges)
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
        
        # Calculate channel centers and widths for convenience
        self.channel_centers = (self.channel_edges[:-1] + self.channel_edges[1:]) / 2
        self.channel_widths = np.diff(self.channel_edges)
        
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
        Get the detector response for each particle.

        Args:
            energies: Absolute kinetic energies in MeV.
            particle: Particle type.

        Returns:
            When detector_used is True: mean energy deposited per particle [MeV].
            When detector_used is False: ones (particle count weight).
        """
        if self.detector_used:
            return self.sensitivity[particle](energies) * energies  # mean_eDep [MeV]
        else:
            return np.ones(len(energies))
    
    def _load_background_2d(
        self,
        filepath: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a time- and energy-resolved background spectrum from a CSV.

        The CSV must have an 'energy' and 'mean' column, and optionally a 'time' column:
            time: time bin centre [s]  (optional)
            energy: energy bin centre [MeV]
            mean: background fluence [particles / cm² / source particle] for that (time, energy) bin

        The data are pivoted into a 2-D array so that downstream code can index it as
        bg_2d[time_index, energy_index].

        Args:
            filepath: Path to the background CSV file.

        Returns:
            time_bins: 1-D array of unique sorted time values [s]
            energy_bins: 1-D array of unique sorted energy values [MeV]
            bg_2d: 2-D array of shape (n_time_bins, n_energy_bins) [particles / cm² / source]
        """
        df = pd.read_csv(filepath)
        energy_bins = np.sort(df['energy'].unique())

        if 'time' in df.columns:
            time_bins = np.sort(df['time'].unique())
            bg_2d = np.zeros((len(time_bins), len(energy_bins)))
            time_index = {t: i for i, t in enumerate(time_bins)}
            energy_index = {e: i for i, e in enumerate(energy_bins)}
            for _, row in df.iterrows():
                bg_2d[time_index[row['time']], energy_index[row['energy']]] = row['mean']
        else:
            time_bins = np.array([0.0])
            bg_2d = np.zeros((1, len(energy_bins)))
            energy_index = {e: i for i, e in enumerate(energy_bins)}
            for _, row in df.iterrows():
                bg_2d[0, energy_index[row['energy']]] += row['mean']

        return time_bins, energy_bins, bg_2d

    def get_background(
        self,
        neutron_background_file: str,
        photon_background_file: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the background energy deposited in the detector from CSV files.

        Background spectra are loaded from energy-resolved CSV files. They can be time-resolved if use_time_gating is True. For each
        time bin the energy axis is contracted against the detector sensitivity:
            deposited(t) = sum_E  bg(t, E) * sensitivity(E) * E   [MeV / cm^2 / source]

        Always returns a 3-tuple of arrays so callers are uniform regardless of use_time_gating.

        Args:
            neutron_background_file: Path to a time-resolved CSV with columns
                                     'time' [s], 'energy' [MeV], 'mean' [particles/cm²/source].
            photon_background_file:  Same format as neutron_background_file.

        Returns:
            time_bins: 1-D array; length 1 (value 0.0) when use_time_gating=False
            neutron_background: 1-D array same length as time_bins [MeV / cm² / source]
            photon_background: 1-D array same length as time_bins [MeV / cm² / source]

        Raises:
            ValueError: If the detector has not been configured, if use_time_gating=True but a
                        background file lacks a 'time' column, or if the two files have different
                        time axes.
        """
        if not self.detector_used:
            raise ValueError("Detector not used; cannot calculate background.")

        if self.use_time_gating:
            for filepath in (neutron_background_file, photon_background_file):
                header = pd.read_csv(filepath, nrows=0)
                if 'time' not in header.columns:
                    raise ValueError(
                        f"use_time_gating=True but background file has no 'time' column: {filepath}"
                    )

        neutron_time_bins, neutron_energy_bins, neutron_bg_2d = self._load_background_2d(neutron_background_file)
        photon_time_bins, photon_energy_bins, photon_bg_2d = self._load_background_2d(photon_background_file)

        if not np.array_equal(neutron_time_bins, photon_time_bins):
            raise ValueError(
                "Neutron and photon background files must share the same time axis. "
                f"Got neutron: {len(neutron_time_bins)} bins, photon: {len(photon_time_bins)} bins."
            )

        # Contract the energy axis against the detector sensitivity to get energy deposited per
        # unit area per source particle as a function of time [MeV / cm² / source].
        neutron_sensitivity = self.sensitivity['neutron'](neutron_energy_bins)
        neutron_background = neutron_bg_2d @ (neutron_sensitivity * neutron_energy_bins)

        photon_sensitivity = self.sensitivity['gamma'](photon_energy_bins)
        photon_background = photon_bg_2d @ (photon_sensitivity * photon_energy_bins)

        if self.use_time_gating:
            # Return full time-resolved arrays so the caller can apply per-channel time windows.
            return neutron_time_bins, neutron_background, photon_background
        else:
            # Collapse to a single time bin at t=0 so the return type is always a 3-tuple of
            # arrays.  The caller can treat both cases uniformly: time_bins has length 1 and
            # each background array contains the total energy deposited over all time.
            return (
                np.array([0.0]),
                np.array([float(np.sum(neutron_background))]),
                np.array([float(np.sum(photon_background))]),
            )
