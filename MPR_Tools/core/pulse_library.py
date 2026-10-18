"""Allpix pulse library: per-particle induced-charge kernels for Si pad channels."""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

# Cumulative-charge eigenvectors, loaded on first use and shared by every PulseLibrary on the same file
_QV_CACHE: Dict[Tuple[str, str], np.ndarray] = {}


class PulseLibrary:
    """
    Mean pulse, second moment E[h^2] and integrated-charge second moment E[Q^2] per incident particle,
    as built by detector_response/pad_response/build_library.py. Libraries with qv also give the exact second moment
    of the charge inside any window of the pulse (gated_q2()).

    Pulses are in electrons per time bin on pads [-1, 0, +1] relative to the hit pad (+1 = the neighbor on the
    side the hit is offset towards). Raw Allpix sign (collected electrons -> negative).
    Charged species depend on (energy, distance from the gap center); neutral species on energy only
    (incident uniformly over the pad).
    """

    def __init__(self, path: str):
        d = np.load(path)
        self._path = path
        self._d = {k: d[k] for k in d.files if not k.endswith('_qv')}  # qv is large: loaded on first use
        self.has_gated_moments = any(k.endswith('_qv') for k in d.files)
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
        mean, m2, q2 = (self._d[f'{key}_{q}'] for q in ('mean', 'm2', 'q2'))
        corners = self._corners(species, pitch, energies, offsets_um)
        out_mean = sum(c[:, None, None] * mean[idx] for c, idx in corners)
        out_m2 = sum(c[:, None, None] * m2[idx] for c, idx in corners)
        out_q2 = sum(c[:, None] * q2[idx] for c, idx in corners)
        return out_mean, out_m2, out_q2

    def gated_q2(
        self,
        species: str,
        pitch: int,
        energies: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        offsets_um: Optional[np.ndarray] = None,
        pads: Sequence[int] = (0, 1, 2),
    ) -> np.ndarray:
        """
        Second moment E[Q_gate^2] of each particle's charge inside pulse bins [lo, hi), per pad.

        Exact up to the library's eigenvector truncation (qq_max_rel_err): sum over r of (qv_r(hi) - qv_r(lo))^2,
        mixed over the same interpolation corners as kernels().

        Args:
            species, pitch, energies, offsets_um: As in kernels().
            lo, hi: Window bounds in pulse bins from the particle's arrival, shape (n, len(pads)); clipped to [0, n_t].
            pads: Library pad indices (0, 1, 2 for pads -1, 0, +1) of the columns of lo and hi.

        Returns:
            shape (n, len(pads)) [electrons^2].
        """
        if not self.has_gated_moments:
            raise ValueError(f"Pulse library {self._path} has no qv; rebuild it with build_library.py.")
        key = f'{species}_p{pitch}_qv'
        if (self._path, key) not in _QV_CACHE:
            _QV_CACHE[self._path, key] = np.load(self._path)[key]
        qv = _QV_CACHE[self._path, key]  # (3, rows, n_t + 1): every grid point's block at its own rank
        first, n_rows = self._d[f'{key}_start'], self._d[f'{key}_rows']
        lo, hi = (np.clip(b, 0, self.n_t)[:, :, None] for b in (lo, hi))
        pad = np.asarray(pads)[None, :, None]

        def window(idx: tuple) -> np.ndarray:
            # Grouped by grid point, since each point's block has its own number of rows
            out = np.zeros(lo.shape[:2])
            starts, group = np.unique(first[idx], return_inverse=True)
            for g, s0 in enumerate(starts):
                m = group.ravel() == g
                rows = s0 + np.arange(n_rows[tuple(i[m][0] for i in idx)])[None, None, :]
                out[m] = ((qv[pad, rows, hi[m]].astype(float) - qv[pad, rows, lo[m]]) ** 2).sum(axis=-1)
            return out

        return sum(c[:, None] * window(idx) for c, idx in self._corners(species, pitch, energies, offsets_um))

    def _corners(
        self,
        species: str,
        pitch: int,
        energies: np.ndarray,
        offsets_um: Optional[np.ndarray],
    ) -> List[Tuple[np.ndarray, tuple]]:
        """Interpolation weights and grid indices: energy (and offset for charged species)."""
        key = f'{species}_p{pitch}'
        grid_e = self._d[f'{key}_energy_mev']
        ie, fe = self._weights(grid_e, energies)
        if species in ('neutron', 'photon'):
            inside = (energies >= grid_e[0]).astype(float)  # zero below the lowest simulated energy
            return [((1 - fe) * inside, (ie,)), (fe * inside, (ie + 1,))]
        ix, fx = self._weights(self._d[f'{key}_offset_um'], offsets_um)
        return [((1 - fe) * (1 - fx), (ie, ix)), ((1 - fe) * fx, (ie, ix + 1)),
                (fe * (1 - fx), (ie + 1, ix)), (fe * fx, (ie + 1, ix + 1))]

    @staticmethod
    def _weights(grid: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Lower grid index and fractional weight, clamped to the grid."""
        if len(grid) == 1:
            return np.zeros(len(values), int), np.zeros(len(values))
        v = np.clip(values, grid[0], grid[-1])
        i = np.clip(np.searchsorted(grid, v, side='right') - 1, 0, len(grid) - 2)
        return i, (v - grid[i]) / (grid[i + 1] - grid[i])
