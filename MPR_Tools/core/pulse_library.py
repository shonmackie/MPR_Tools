"""Allpix pulse library: per-particle induced-charge kernels for Si pad channels."""

from typing import Optional, Tuple
import numpy as np


class PulseLibrary:
    """
    Mean pulse, second moment E[h^2] and integrated-charge second moment E[Q^2] per incident particle,
    as built by detector_response/pad_response/build_library.py.

    Pulses are in electrons per time bin on pads [-1, 0, +1] relative to the hit pad (+1 = the neighbor on the
    side the hit is offset towards). Raw Allpix sign (collected electrons -> negative).
    Charged species depend on (energy, distance from the gap center); neutral species on energy only
    (incident uniformly over the pad).
    """

    def __init__(self, path: str):
        d = np.load(path)
        self._d = {k: d[k] for k in d.files}
        self.dt = float(d['dt_ns']) * 1e-9  # s
        self.window = float(d['window_ns']) * 1e-9  # s
        self.thickness = float(d['thickness_um']) * 1e-6  # m
        self.bias_voltage = float(d['bias_v'])  # V
        self.pitches = {}
        for key in self._d:
            if key.endswith('_energy_mev'):
                species, pitch = key[:-len('_energy_mev')].rsplit('_p', 1)
                self.pitches.setdefault(species, []).append(int(pitch))
        for species in self.pitches:
            self.pitches[species].sort()
        self.n_t = next(v.shape[-1] for k, v in self._d.items() if k.endswith('_mean'))

    def nearest_pitch(self, width_m: float) -> int:
        """Library pitch [mm] closest to a channel width [m], among pitches simulated for every species."""
        pitches = np.array(sorted(set.intersection(*(set(p) for p in self.pitches.values()))))
        return int(pitches[np.argmin(np.abs(pitches - width_m * 1e3))])

    def kernels(
        self,
        species: str,
        pitch: int,
        energies: np.ndarray,
        offsets_um: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolated kernels for a set of particles.

        Args:
            species: 'proton', 'deuteron', 'neutron' or 'photon'.
            pitch: Library pitch [mm].
            energies: Incident kinetic energies [MeV], shape (n,).
            offsets_um: Charged species only: distance from the gap center [um], shape (n,).

        Returns:
            mean, m2: shape (n, 3, n_t) [electrons, electrons^2 per bin]; q2: shape (n, 3) [electrons^2].
            Linear interpolation in energy (and offset); charged energies are clamped to the grid,
            neutral kernels are zero below the lowest simulated energy.
        """
        key = f'{species}_p{pitch}'
        grid_e = self._d[f'{key}_energy_mev']
        mean, m2, q2 = (self._d[f'{key}_{q}'] for q in ('mean', 'm2', 'q2'))
        ie, fe = self._weights(grid_e, energies)

        if species in ('neutron', 'photon'):
            below = energies < grid_e[0]
            out = [(1 - fe)[:, None, None] * a[ie] + fe[:, None, None] * a[ie + 1] for a in (mean, m2)]
            q = (1 - fe)[:, None] * q2[ie] + fe[:, None] * q2[ie + 1]
            for a in out:
                a[below] = 0.0
            q[below] = 0.0
            return out[0], out[1], q

        ix, fx = self._weights(self._d[f'{key}_offset_um'], offsets_um)
        w = [((1 - fe) * (1 - fx), ie, ix), ((1 - fe) * fx, ie, ix + 1), (fe * (1 - fx), ie + 1, ix), (fe * fx, ie + 1, ix + 1)]
        out_mean = sum(c[:, None, None] * mean[i, j] for c, i, j in w)
        out_m2 = sum(c[:, None, None] * m2[i, j] for c, i, j in w)
        out_q2 = sum(c[:, None] * q2[i, j] for c, i, j in w)
        return out_mean, out_m2, out_q2

    @staticmethod
    def _weights(grid: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Lower grid index and fractional weight, clamped to the grid."""
        if len(grid) == 1:
            return np.zeros(len(values), int), np.zeros(len(values))
        v = np.clip(values, grid[0], grid[-1])
        i = np.clip(np.searchsorted(grid, v, side='right') - 1, 0, len(grid) - 2)
        return i, (v - grid[i]) / (grid[i + 1] - grid[i])
