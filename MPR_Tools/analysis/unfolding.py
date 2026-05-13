"""Spectrum unfolding (inverse problem) for MPR spectrometer hodoscope data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np
from scipy.optimize import nnls, minimize


@dataclass
class UnfoldingResult:
    """Container for the output of a spectrum unfolding calculation.

    Attributes:
        spectrum: Recovered incident spectrum f_i [arb. units], shape (n_energies,).
        uncertainties: 1-sigma uncertainties σ_i on spectrum, shape (n_energies,).
        energy_grid: Incident energies E_i [MeV], shape (n_energies,).
        chi_square: Goodness-of-fit Σ_k ε_k² / σ_k².
        method: Name of the unfolding method used.
        converged: Whether the iterative/optimisation algorithm converged.
        n_iterations: Number of iterations taken (0 for non-iterative methods).
        lambdas: Lagrange multipliers λ_k (maximum entropy only; None otherwise).
    """
    spectrum: np.ndarray
    uncertainties: np.ndarray
    energy_grid: np.ndarray
    chi_square: float
    method: str
    converged: bool = True
    n_iterations: int = 0
    lambdas: Optional[np.ndarray] = None


class SpectrumUnfolder:
    """Inverse-problem solver: recovers the incident neutron spectrum from hodoscope counts.

    The forward model is  N_k = Σ_i R[i,k] * f_i  (+ background),  where:
      - N_k   : measured signal in channel k
      - R[i,k]: instrument response matrix — expected signal in channel k per source
                neutron of energy E_i.  Built by PerformanceAnalyzer.build_response_matrix().
      - f_i   : incident spectrum at energy E_i (what we want to recover)

    Single-foil usage
    -----------------
    unfolder = SpectrumUnfolder(R, energy_grid)
    result   = unfolder.unfold_least_squares(counts, sigma)

    Dual-foil usage
    ---------------
    Use the ``for_dual_foil`` class method.  All unfold_* methods then automatically
    unfold each hodoscope independently and return an inverse-variance weighted average.
    Time-gating ensures complete signal separation; no cross-contamination correction
    is needed.

    response_matrices = analyzer.build_response_matrix(energy_grid, ...)
    unfolder = SpectrumUnfolder.for_dual_foil(response_matrices, energy_grid, foil_a='CH2')
    result   = unfolder.unfold_least_squares(joint_counts, joint_sigma)

    Parameters
    ----------
    response_matrix : np.ndarray, shape (n_energies, n_channels)
        R[i, k] = expected signal in channel k per source neutron at energy E_i.
    energy_grid : np.ndarray, shape (n_energies,)
        Incident neutron energies [MeV] corresponding to rows of response_matrix.
    """

    def __init__(
        self,
        response_matrix: np.ndarray,
        energy_grid: np.ndarray,
    ) -> None:
        self.R = np.asarray(response_matrix, dtype=float)          # (n_E, n_ch)
        self.energy_grid = np.asarray(energy_grid, dtype=float)    # (n_E,)

        if self.R.shape[0] != len(self.energy_grid):
            raise ValueError(
                f'response_matrix has {self.R.shape[0]} rows but '
                f'energy_grid has {len(self.energy_grid)} entries.'
            )

        # Set by for_dual_foil(); None means single-foil mode.
        self._n_ch2: Optional[int] = None
        self._R_aa: Optional[np.ndarray] = None  # primary foil direct response
        self._R_bb: Optional[np.ndarray] = None  # secondary foil direct response

    # ------------------------------------------------------------------
    # Class method constructor for dual-foil independent unfolding
    # ------------------------------------------------------------------

    @classmethod
    def for_dual_foil(
        cls,
        response_matrices: Dict[str, np.ndarray],
        energy_grid: np.ndarray,
        foil_a: str,
    ) -> 'SpectrumUnfolder':
        """Build an unfolder configured for dual-foil independent unfolding.

        Constructs the joint response matrix by horizontally stacking the two foil
        response matrices and stores each foil's direct response for independent
        per-foil unfolding followed by inverse-variance weighted averaging.

        Because the dual-foil spectrometer uses time-gating, proton and deuteron
        signals are fully separated in time and cross-contamination between the two
        hodoscopes is physically impossible.  No cross-contamination correction
        is needed.

        Parameters
        ----------
        response_matrices : dict returned by
            ``PerformanceAnalyzer.build_response_matrix()`` for a dual-foil setup.
            Expected keys: foil_a material name and one other foil material name.
        energy_grid : shape (n_energies,)
        foil_a : material name of the primary foil (placed first in the joint
            response matrix), e.g. ``'CH2'``.
        """
        remaining = set(response_matrices.keys()) - {foil_a}
        if len(remaining) != 1:
            raise ValueError(
                f"Expected exactly two foil names in response_matrices; "
                f"found {sorted(response_matrices)}. "
                f"Could not identify a unique secondary foil after removing '{foil_a}'."
            )
        foil_b = remaining.pop()

        if foil_a not in response_matrices:
            raise KeyError(
                f"Expected response_matrices key '{foil_a}' not found. "
                f"Available keys: {sorted(response_matrices)}."
            )

        R_aa = np.asarray(response_matrices[foil_a], dtype=float)   # (n_E, n_ch_a)
        R_bb = np.asarray(response_matrices[foil_b], dtype=float)   # (n_E, n_ch_b)
        R_joint = np.hstack([R_aa, R_bb])

        instance = cls(R_joint, energy_grid)
        instance._n_ch2 = R_aa.shape[1]
        instance._R_aa = R_aa
        instance._R_bb = R_bb
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _subtract_background(
        self,
        measured_counts: np.ndarray,
        background: Optional[np.ndarray],
    ) -> np.ndarray:
        if background is not None:
            return measured_counts - np.asarray(background, dtype=float)
        return np.asarray(measured_counts, dtype=float)

    def _chi_square(
        self,
        spectrum: np.ndarray,
        measured_counts: np.ndarray,
        uncertainties: np.ndarray,
    ) -> float:
        predicted = self.R.T @ spectrum
        residuals = measured_counts - predicted
        return float(np.sum((residuals / uncertainties) ** 2))

    def _unfold_dual(
        self,
        N: np.ndarray,
        sigma: np.ndarray,
        ch2_fn: Callable,
        cd2_fn: Callable,
    ) -> UnfoldingResult:
        """Independent unfolding for dual-foil mode with inverse-variance averaging.

        Splits the joint count vector into the two hodoscope segments, unfolds each
        independently using its own direct response matrix, then returns an
        inverse-variance weighted average spectrum.

        Parameters
        ----------
        N : background-subtracted joint counts, shape (n_ch_a + n_ch_b,).
        sigma : per-channel uncertainties, same shape as N.
        ch2_fn : callable (unfolder, counts, sigma) → UnfoldingResult for primary channels.
        cd2_fn : callable (unfolder, counts, sigma) → UnfoldingResult for secondary channels.
        """
        n = self._n_ch2
        N_a, sigma_a = N[:n], sigma[:n]
        N_b, sigma_b = N[n:], sigma[n:]

        assert self._R_aa is not None and self._R_bb is not None

        unfolder_a = SpectrumUnfolder(self._R_aa, self.energy_grid)
        res_a = ch2_fn(unfolder_a, N_a, sigma_a)

        unfolder_b = SpectrumUnfolder(self._R_bb, self.energy_grid)
        res_b = cd2_fn(unfolder_b, N_b, sigma_b)

        # Inverse-variance weighted average
        w_a = np.where(res_a.uncertainties > 0, 1.0 / res_a.uncertainties ** 2, 0.0)
        w_b = np.where(res_b.uncertainties > 0, 1.0 / res_b.uncertainties ** 2, 0.0)
        w_total = w_a + w_b
        f_combined = np.where(w_total > 0,
                              (w_a * res_a.spectrum + w_b * res_b.spectrum) / w_total,
                              0.0)
        sigma_combined = np.where(w_total > 0, 1.0 / np.sqrt(w_total), np.inf)
        chi2 = (res_a.chi_square + res_b.chi_square) / 2.0

        return UnfoldingResult(
            spectrum=f_combined,
            uncertainties=sigma_combined,
            energy_grid=self.energy_grid,
            chi_square=chi2,
            method=res_a.method + '_dual',
            converged=res_a.converged and res_b.converged,
            n_iterations=res_a.n_iterations + res_b.n_iterations,
        )

    # ------------------------------------------------------------------
    # Public unfolding methods
    # ------------------------------------------------------------------

    def unfold_maximum_entropy(
        self,
        measured_counts: np.ndarray,
        uncertainties: np.ndarray,
        default_spectrum: np.ndarray,
        omega: Optional[float] = None,
        background: Optional[np.ndarray] = None,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> UnfoldingResult:
        """Recover the spectrum using the maximum entropy (MAXED) algorithm.

        Maximises the entropy  S = -Σ_i [ f_i ln(f_i / f_DEF_i) + f_DEF_i - f_i ]
        subject to the chi-square constraint  Σ_k ε_k² / σ_k² = Ω.

        The dual potential D(λ) = Σ_i f_i + Σ_k λ_k(N_k + ε_k) is convex and is
        minimised directly (equivalent to maximising the concave −D from the MAXED paper).
        The solution spectrum is  f_i = f_DEF_i exp(−Σ_k λ_k R_{ki}),  with
        ε_k = (λ_k σ_k²/2)(4Ω/Σ_j(λ_j σ_j)²)^{1/2}  [Eq. 4, Reginatto et al. 2002].

        In dual-foil mode (constructed via ``for_dual_foil``) each hodoscope is unfolded
        independently and the spectra are combined by inverse-variance weighting.

        Parameters
        ----------
        measured_counts : shape (n_channels,)
        uncertainties : shape (n_channels,)  — σ_k per channel (√N_k for Poisson)
        default_spectrum : shape (n_energies,)  — f_DEF, a priori spectrum (must be > 0)
        omega : chi-square target Ω; defaults to the number of channels.
        background : optional background per channel subtracted before unfolding.
        max_iter : maximum optimiser iterations.
        tol : convergence tolerance for the optimiser.
        """
        N = self._subtract_background(measured_counts, background)
        sigma = np.asarray(uncertainties, dtype=float)
        f_def = np.asarray(default_spectrum, dtype=float)

        if self._n_ch2 is not None:
            return self._unfold_dual(
                N, sigma,
                lambda u, Na, sa: u.unfold_maximum_entropy(
                    Na, sa, f_def, omega=omega, max_iter=max_iter, tol=tol),
                lambda u, Nb, sb: u.unfold_maximum_entropy(
                    Nb, sb, f_def, omega=omega, max_iter=max_iter, tol=tol),
            )

        R = self.R  # (n_E, n_ch)
        if omega is None:
            omega = float(len(N))

        def D_and_grad(lam):
            exponent = R @ lam
            f = f_def * np.exp(-exponent)

            lam_sigma = lam * sigma
            denom = np.sqrt(np.sum(lam_sigma ** 2))
            if denom == 0:
                eps = np.zeros_like(lam)
            else:
                eps = (lam * sigma ** 2) / (2.0 * denom) * np.sqrt(4.0 * omega)

            # D(λ) is convex — minimise it directly
            predicted_N = R.T @ f
            D = np.sum(f) + np.dot(lam, N + eps)
            grad = -predicted_N + (N + eps)
            return D, grad

        lam0 = np.zeros(len(N))
        result = minimize(
            D_and_grad,
            lam0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol},
        )

        lam_opt = result.x
        f = f_def * np.exp(-R @ lam_opt)

        sigma_f = np.sqrt(
            np.sum((f[:, np.newaxis] * R * sigma[np.newaxis, :]) ** 2, axis=1)
        )
        chi2 = self._chi_square(f, N, sigma)
        return UnfoldingResult(
            spectrum=f,
            uncertainties=sigma_f,
            energy_grid=self.energy_grid,
            chi_square=chi2,
            method='maximum_entropy',
            converged=result.success,
            n_iterations=result.nit,
            lambdas=lam_opt,
        )

    def unfold_least_squares(
        self,
        measured_counts: np.ndarray,
        uncertainties: np.ndarray,
        background: Optional[np.ndarray] = None,
    ) -> UnfoldingResult:
        """Recover the spectrum via non-negative least squares (no regularisation).

        Minimises  ||W^{1/2}(R^T f − N)||²  subject to  f_i ≥ 0.

        In dual-foil mode each hodoscope is unfolded independently and the spectra
        are combined by inverse-variance weighting.

        Parameters
        ----------
        measured_counts : shape (n_channels,)
        uncertainties : shape (n_channels,)
        background : optional per-channel background.
        """
        N = self._subtract_background(measured_counts, background)
        sigma = np.asarray(uncertainties, dtype=float)

        if self._n_ch2 is not None:
            return self._unfold_dual(
                N, sigma,
                lambda u, Na, sa: u.unfold_least_squares(Na, sa),
                lambda u, Nb, sb: u.unfold_least_squares(Nb, sb),
            )

        w = 1.0 / sigma
        A = self.R.T * w[:, np.newaxis]
        b = N * w
        f, _ = nnls(A, b)

        try:
            H_inv = np.linalg.pinv(A.T @ A)
            sigma_f = np.sqrt(np.maximum(np.diag(H_inv), 0.0))
        except np.linalg.LinAlgError:
            sigma_f = np.full(len(self.energy_grid), np.nan)

        chi2 = self._chi_square(f, N, sigma)
        return UnfoldingResult(
            spectrum=f,
            uncertainties=sigma_f,
            energy_grid=self.energy_grid,
            chi_square=chi2,
            method='least_squares',
            converged=True,
            n_iterations=0,
        )

    def unfold_maximum_likelihood(
        self,
        measured_counts: np.ndarray,
        default_spectrum: np.ndarray,
        n_iterations: int = 100,
        background: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        response_threshold: float = 1e-3,
    ) -> UnfoldingResult:
        """Recover the spectrum via expectation-maximisation (maximum likelihood).

        Multiplicative EM update maximising the Poisson log-likelihood:
            f_i^{(n+1)} = f_i^{(n)} * Σ_k R[i,k] N_k / Σ_k R[i,k] (R^T f^{(n)})_k

        Only energies where the instrument has meaningful sensitivity are updated;
        rows of R whose sum is below ``response_threshold * max(R_rowsum)`` are
        outside the instrument range and are set to zero in the result.  Without
        this masking the EM is underdetermined and concentrates the spectrum at
        arbitrary spike energies (classic ML-EM artefact).

        In dual-foil mode each hodoscope is unfolded independently and the spectra
        are combined by inverse-variance weighting.

        Parameters
        ----------
        measured_counts : shape (n_channels,)
        default_spectrum : shape (n_energies,)  — starting point f^{(0)} (must be > 0).
        n_iterations : maximum EM iterations.
        background : optional per-channel background.
        tol : relative convergence tolerance on f.
        response_threshold : energies with R_rowsum < response_threshold * max(R_rowsum)
            are treated as outside the instrument range and zeroed in the result.
        """
        N = self._subtract_background(measured_counts, background)
        N = np.maximum(N, 0.0)
        f_def = np.asarray(default_spectrum, dtype=float)

        if self._n_ch2 is not None:
            sigma = np.sqrt(np.maximum(N, 1.0))
            return self._unfold_dual(
                N, sigma,
                lambda u, Na, _s: u.unfold_maximum_likelihood(
                    Na, f_def, n_iterations=n_iterations, tol=tol,
                    response_threshold=response_threshold),
                lambda u, Nb, _s: u.unfold_maximum_likelihood(
                    Nb, f_def, n_iterations=n_iterations, tol=tol,
                    response_threshold=response_threshold),
            )

        R = self.R
        R_rowsum = R.sum(axis=1)

        # Restrict EM to energies where the instrument has meaningful sensitivity.
        # Energies outside this support are underdetermined; allowing them to be
        # free causes the EM to produce spurious spikes at arbitrary energies.
        support = R_rowsum >= response_threshold * R_rowsum.max()
        R_s = R[support, :]
        R_rowsum_s = R_rowsum[support]
        f_def_s = f_def[support].copy()
        # EM is multiplicative — zero entries are permanently stuck.  Add a
        # small floor so every support bin can be updated by the data.
        f_floor = 1e-6 * f_def_s.max() if f_def_s.max() > 0 else 1e-30
        f = np.maximum(f_def_s, f_floor)

        converged = False
        actual_iterations = 0
        for it in range(n_iterations):
            predicted = R_s.T @ f
            safe_pred = np.where(predicted > 0, predicted, 1.0)
            ratio = N / safe_pred
            f_new = f * (R_s @ ratio) / R_rowsum_s
            f_new = np.maximum(f_new, 0.0)
            max_rel_change = np.max(np.abs(f_new - f) / np.where(f > 0, f, 1.0))
            f = f_new
            actual_iterations = it + 1
            if max_rel_change < tol:
                converged = True
                break

        predicted = R_s.T @ f
        safe_pred = np.where(predicted > 0, predicted, 1.0)
        Fisher_s = (R_s / safe_pred[np.newaxis, :]) @ R_s.T
        cov_diag_s = np.diag(np.linalg.pinv(Fisher_s))
        sigma_f_s = np.sqrt(np.maximum(cov_diag_s, 0.0))

        f_full = np.zeros(len(self.energy_grid))
        f_full[support] = f
        sigma_f_full = np.full(len(self.energy_grid), np.inf)
        sigma_f_full[support] = sigma_f_s

        sigma_counts = np.sqrt(np.maximum(N, 1.0))
        chi2 = self._chi_square(f_full, N, sigma_counts)
        return UnfoldingResult(
            spectrum=f_full,
            uncertainties=sigma_f_full,
            energy_grid=self.energy_grid,
            chi_square=chi2,
            method='maximum_likelihood',
            converged=converged,
            n_iterations=actual_iterations,
        )
