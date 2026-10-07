"""Forward-fitting of parametric spectral models to hodoscope count data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .performance import HodoscopeResponse
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import differential_evolution, least_squares
from tqdm import tqdm


def ballabio_model(
    energy_grid: np.ndarray,
    foil_geometric_factor: float = 1.0,
    n_t: float = 0.5,
    n_d: float = 0.5,
    n_h: float = 0.0,
) -> Tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], Dict[str, np.ndarray]],
    List[str],
    Tuple[List[float], List[float]],
]:
    """Return a Ballabio spectral model for DT primary fusion neutrons.

    Models the incident spectrum as from Ballabio's 1997 and 1998 papers

    Parameters
    ----------
    energy_grid : shape (n_energies,)
        Incident particle energies [MeV].
    foil_geometric_factor : float
        Defaults to 1.0.
    n_t, n_d, n_h : float
        Fuel number fractions. Default is 50/50 DT.

    Returns
    -------
    model : callable ``params -> spectrum``
        Fitting model passed to ``SpectrumFitter.fit()``.
    components_model : callable ``params -> dict[str, ndarray]``
        Keys: 'Primary', 'Downscatter', 'AKN'. 
        Call with ``result.params`` after fitting to decompose the fitted spectrum.
    param_names : list[str]
    param_bounds : (lower_bounds, upper_bounds)
        Suggested physical bounds.
    """
    _BALLABIO_DATA_DIR = Path(__file__).parent.parent / 'data' / 'ballabio'
    _EC_AKN = 15.5    # MeV - reference energy in AKN exponential (Ballabio 1997 Eq. 24)
    _E_AKN_START = 14.0  # MeV - lower bound for AKN in components_model display

    fuel_t = np.loadtxt(_BALLABIO_DATA_DIR / 'fuel_T_100keVbins.txt')
    fuel_d = np.loadtxt(_BALLABIO_DATA_DIR / 'fuel_D_100keVbins.txt')
    fuel_h = np.loadtxt(_BALLABIO_DATA_DIR / 'fuel_H_100keVbins.txt')
    fuel_bin_width = fuel_t[1, 0] - fuel_t[0, 0]  # 0.1 MeV - files store b/bin
    avg_mol_weight = n_d * 2.02 + n_t * 3.03 + n_h * 1.01
    # Weighted downscatter spectrum
    fuel_tot = (
        (fuel_d[:, 1] * n_d + fuel_t[:, 1] * n_t + fuel_h[:, 1] * n_h)
        * 6.022e23 / avg_mol_weight * 1e-24 / 1000 / fuel_bin_width
    )
    # Interpolate fuel_tot to energy_grid
    fs = np.interp(energy_grid, fuel_t[:, 0], fuel_tot)

    bw = np.empty(len(energy_grid))
    bw[:-1] = np.diff(energy_grid)
    bw[-1] = bw[-2]

    above_ec_fit  = energy_grid > _EC_AKN
    above_ec_eval = energy_grid > _E_AKN_START

    def _primary(I: float, Tion: float, dE: float) -> np.ndarray:
        Emean = (
            14.021
            + (5.30509 / (1 + 2.4736e-3 * Tion**1.84) * Tion ** (2 / 3) + 1.3818 * Tion) * 1e-3
            + dE
        )
        sigma = 75.2749665 * np.sqrt(Tion) / 1000.0
        return I * foil_geometric_factor * np.exp(-0.5 * ((energy_grid - Emean) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma) * bw

    def _downscatter(I: float, rhoR: float) -> np.ndarray:
        return I * foil_geometric_factor * rhoR * fs * bw

    def _akn(I: float, Y_akn_rel: float, lam: float, mask: np.ndarray) -> np.ndarray:
        return np.where(
            mask,
            (I * foil_geometric_factor * Y_akn_rel / lam) * np.exp(-(energy_grid - _EC_AKN) / lam) * bw,
            0.0,
        )

    param_names = ['I', 'Tion [keV]', 'dE [MeV]', 'rhoR [mg/cm2]', 'Y_akn_rel', 'lambda_akn [MeV]']
    param_bounds = ([0., 1., -1., 0., 0., 0.], [np.inf, 20., 1., np.inf, 1., 2.0])

    def model(params: np.ndarray) -> np.ndarray:
        I, Tion, dE, rhoR, Y_akn_rel, lam = params
        return _primary(I, Tion, dE) + _downscatter(I, rhoR) + _akn(I, Y_akn_rel, lam, above_ec_fit)
    model.model_name = 'Ballabio'  # type: ignore[attr-defined]

    def components_model(params: np.ndarray) -> Dict[str, np.ndarray]:
        I, Tion, dE, rhoR, Y_akn_rel, lam = params
        return {'Primary': _primary(I, Tion, dE), 'Downscatter': _downscatter(I, rhoR), 'AKN': _akn(I, Y_akn_rel, lam, above_ec_eval)}

    return model, components_model, param_names, param_bounds


def nonstop_model(
    excel_path: Union[str, Path],
    energy_grid: np.ndarray,
    param_columns: Optional[List[str]] = None,
    foil_geometric_factor: float = 1.0,
) -> Tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], Dict[str, np.ndarray]],
    List[str],
    Tuple[List[float], List[float]],
]:
    """Return a NONSTOP spectral model that interpolates over a pre-computed parameter grid.

    Fitting parameters are ``[I, *physical_params]`` where:

    * ``I``                - overall source yield: sum(model) == I * foil_geometric_factor
    * ``physical_params``  - the columns named by ``param_columns`` in the summary
                             sheet (e.g. T_ion_keV, DT_rhoR_gcm2, Liner_rhoR_gcm2,
                             BR_Tcm). ``param_bounds`` for these axes is derived
                             directly from the grid extremes.

    Interpolation is performed in log10 space to handle the
    orders-of-magnitude variation across spectral components. After interpolation,
    values are converted back to linear space and rebinned to ``energy_grid``.

    Parameters
    ----------
    excel_path : str or Path
    energy_grid : ndarray, shape (n_energies,)
        Instrument energy grid (same as rows of response matrix).
    param_columns : list[str], optional
        Summary-sheet columns to use as physical parameters. Defaults to
        ``['T_ion_keV', 'DT_rhoR_gcm2', 'Liner_rhoR_gcm2', 'BR_Tcm']``.
    foil_geometric_factor : float
        Multiplied with I for absolute scaling.

    Returns
    -------
    model, components_model, param_names, param_bounds
        See ``ballabio_model`` for the unified interface description.
        ``components_model`` returns keys:
        'Primary', 'DT downscatter', 'Liner downscatter', 'Tertiary'.
    """
    if param_columns is None:
        param_columns = ['T_ion_keV', 'DT_rhoR_gcm2', 'Liner_rhoR_gcm2', 'BR_Tcm']

    sheets = pd.read_excel(excel_path, sheet_name=None)
    en = sheets['En_MeV']
    energy_edges = np.concatenate([en['E_low_MeV'].to_numpy(), en['E_high_MeV'].to_numpy()[-1:]])
    raw_components: Dict[str, np.ndarray] = {}
    for name in ['primary', 'downscatter_DT', 'downscatter_Liner', 'tertiary']:
        raw_components[name] = sheets[name].set_index('grid_id').to_numpy(dtype=float)
    summary_df = sheets['summary'].set_index('grid_id')

    e_lo = energy_edges[:-1]
    e_hi = energy_edges[1:]
    e_centers = (e_lo + e_hi) / 2.0
    excel_bw = e_hi - e_lo

    # Interpolate NONSTOP bin widths onto the instrument grid so model yields are per-NONSTOP-bin,
    # matching how probability = per_bin / sum(per_bin) is used for beam generation.
    rebin_bw = np.interp(energy_grid, e_centers, excel_bw)

    param_points = summary_df[param_columns].to_numpy(dtype=float)

    comp_display_names = {
        'primary': 'Primary',
        'downscatter_DT': 'DT downscatter',
        'downscatter_Liner': 'Liner downscatter',
        'tertiary': 'AKN',
    }
    interpolators: Dict[str, LinearNDInterpolator] = {}
    _LOG_EPS = 1e-30  # floor before log10 to avoid -inf on near-zero spectral bins
    for sheet_key in raw_components:
        per_bin = raw_components[sheet_key]
        density = per_bin / excel_bw[np.newaxis, :]
        log_density = np.log10(density + _LOG_EPS)
        interpolators[sheet_key] = LinearNDInterpolator(param_points, log_density)

    param_lo = param_points.min(axis=0).tolist()
    param_hi = param_points.max(axis=0).tolist()

    def _eval_components(phys_params: np.ndarray) -> Dict[str, np.ndarray]:
        # Interpolate all components at phys_params, rebin to instrument grid.
        query = phys_params.reshape(1, -1)
        result: Dict[str, np.ndarray] = {}
        for sheet_key, display_name in comp_display_names.items():
            log_dens_excel = interpolators[sheet_key](query)[0]
            if np.any(np.isnan(log_dens_excel)):
                raise ValueError(
                    f'Parameters {phys_params} are outside the NONSTOP grid for '
                    f'component "{display_name}". Ensure bounds stay within the grid.'
                )
            # log-space interp avoids large errors near the primary peak
            log_dens_inst = np.interp(energy_grid, e_centers, log_dens_excel)
            result[display_name] = 10.0 ** log_dens_inst * rebin_bw
        return result

    def model(params: np.ndarray) -> np.ndarray:
        I = params[0]
        comps = _eval_components(params[1:])
        stacked = np.sum(list(comps.values()), axis=0)
        return I * foil_geometric_factor * stacked / stacked.sum()
    model.model_name = 'NONSTOP'  # type: ignore[attr-defined]

    def components_model(params: np.ndarray) -> Dict[str, np.ndarray]:
        I = params[0]
        comps = _eval_components(params[1:])
        total = np.sum(list(comps.values()))
        return {k: I * foil_geometric_factor * v / total for k, v in comps.items()}

    param_names = ['I'] + [col.replace('_', ' ') for col in param_columns]
    param_bounds = (
        [0.0] + param_lo,
        [np.inf] + param_hi,
    )

    return model, components_model, param_names, param_bounds

@dataclass
class ForwardFittingResult:
    """Container for the output of a forward-fitting calculation.

    Attributes
    ----------
    spectrum : model spectrum at optimal params, shape (n_energies,).
    uncertainties : 1-sigma on spectrum via linear error propagation, shape (n_energies,).
    energy_grid : incident particle energies [MeV], shape (n_energies,).
    chi_square : reduced chi-square sum_k(residuals_k^2) / n_dof.
    converged : whether the optimizer reported success.
    n_iterations : number of model evaluations.
    params : optimal parameter values, shape (n_params,).
    param_names : parameter labels for display.
    param_uncertainties : 1-sigma from Jacobian covariance, shape (n_params,).
    covariance_matrix : full (n_params, n_params) parameter covariance.
    message : optimizer termination message.
    bin_widths : energy bin widths [MeV] for converting spectrum to spectral density.
    signal : per-channel background-subtracted signal (valid channels only),
        shape (n_valid_channels,). Used by compute_spectrum_data_points for visualization.
    sigma_counts : per-channel 1-sigma weights used in the chi-square (valid channels only),
        shape (n_valid_channels,).
    response_matrix : response matrix restricted to valid channels, shape (n_energies, n_valid_channels).
    background : per-channel background that was subtracted before fitting (valid channels only),
        shape (n_valid_channels,). None if no background was supplied.

    """
    spectrum: np.ndarray
    uncertainties: np.ndarray
    energy_grid: np.ndarray
    chi_square: float
    converged: bool
    n_iterations: int
    params: np.ndarray
    param_names: List[str]
    param_uncertainties: np.ndarray
    covariance_matrix: np.ndarray
    message: str
    model_name: str = ''
    bin_widths: Optional[np.ndarray] = None
    signal: Optional[np.ndarray] = None
    sigma_counts: Optional[np.ndarray] = None
    response_matrix: Optional[np.ndarray] = None
    background: Optional[np.ndarray] = None
    def summary(self, true_params: Optional[np.ndarray] = None) -> None:
        width = 22 if true_params is not None else 18
        header = f'=== {self.model_name} ===' if self.model_name else '==='
        print(f'\n{header}')
        print(f'  Converged: {self.converged}  |  chi_square: {self.chi_square:.3f}  |  nfev: {self.n_iterations}')
        if true_params is not None:
            print(f'  {"Parameter":<{width}} {"True":>14} {"Fitted":>14} {'±1σ':>14}')
            for name, tv, fv, unc in zip(self.param_names, true_params, self.params, self.param_uncertainties):
                print(f'  {name:<{width}} {tv:>14.4g}  {fv:>14.4g}  {unc:>14.4g}')
        else:
            print(f'  {"Parameter":<{width}} {"Value":>12}  {'±1σ':>12}')
            for name, val, unc in zip(self.param_names, self.params, self.param_uncertainties):
                print(f'  {name:<{width}} {val:>12.4g}  {unc:>12.4g}')


class SpectrumFitter:
    """Forward-fitting solver: fits a parametric spectral model to hodoscope counts.

    The forward model is  N_k = sum_i R[i,k] * f(params)_i  (+ background),  where:
      - N_k         : measured signal in channel k
      - R[i,k]      : instrument response matrix
      - f(params)_i : parametric model spectrum - a user-supplied callable

    The user provides any callable ``model(params) -> spectrum`` that maps a parameter
    vector to an incident spectrum on the instrument's energy grid. The fitter adjusts
    params to minimize the weighted residuals via scipy's trust-region least squares.

    Parameter uncertainties are estimated from the Jacobian at the solution:
    ``cov = pinv(J^T J) * reduced_chi_square``. Spectrum uncertainties are propagated
    from parameter covariance via finite-difference derivatives of the model.

    After fitting, call ``components_model(result.params)`` (returned by the model
    factory) to decompose the fitted spectrum into its constituent parts for plotting.

    Parameters
    ----------
    response : HodoscopeResponse or dict[str, HodoscopeResponse]
        Single response object for single-foil setups, or a dict mapping foil
        material name to HodoscopeResponse for dual-foil setups (matrices are
        stacked column-wise).  Each response must have ``response_matrix`` and
        ``energy_grid`` set (use ``PerformanceAnalyzer.get_channel_response()``).
    """

    def __init__(
        self,
        response: Union[HodoscopeResponse, Dict[str, HodoscopeResponse]],
    ) -> None:

        if isinstance(response, dict):
            responses = list(response.values())
            R_list = [resp.response_matrix for resp in responses]
            if any(R is None for R in R_list):
                raise ValueError(
                    'All HodoscopeResponse objects in a dual-foil dict must have '
                    'response_matrix set.'
                )
            self.R = np.hstack(R_list)
            self.energy_grid = responses[0].energy_grid
            self._signal = np.concatenate([resp.signal for resp in responses])
            self._signal_std = np.concatenate([resp.signal_std for resp in responses])
            self._count = np.concatenate([resp.count for resp in responses])
            self._background = np.concatenate([resp.background for resp in responses])
            self._background_std = np.concatenate([resp.background_std for resp in responses])
        else:
            if response.response_matrix is None:
                raise ValueError(
                    'HodoscopeResponse must have response_matrix set to use SpectrumFitter.'
                )
            self.R = response.response_matrix
            self.energy_grid = response.energy_grid
            self._signal = response.signal
            self._signal_std = response.signal_std
            self._count = response.count
            self._background = response.background
            self._background_std = response.background_std

        if self.energy_grid is None:
            raise ValueError('energy_grid must be set in HodoscopeResponse.')
        if self.R.shape[0] != len(self.energy_grid):
            raise ValueError(
                f'response_matrix has {self.R.shape[0]} rows but '
                f'energy_grid has {len(self.energy_grid)} entries.'
            )

    def fit(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        initial_params: Union[np.ndarray, List[float]],
        signal: Optional[np.ndarray] = None,
        inject_noise: bool = False,
        param_names: Optional[List[str]] = None,
        bounds: Optional[Tuple] = None,
        true_params: Optional[np.ndarray] = None,
        use_global_optimizer: bool = False,
    ) -> ForwardFittingResult:
        """Fit a parametric model to measured hodoscope counts.

        Minimises the standard weighted chi-squared:
            chi^2 = sum_k [(R^T f(params))_k - d_k]^2 / sigma_k^2

        Channels with sigma_k == 0 are excluded from the fit.
        Parameter uncertainties are the Gauss-Newton approximation:
            cov = pinv(J^T J) * chi^2_nu
        which is exact in the linear limit and approximate otherwise.

        Signal, background, and per-channel sigma are taken from the
        HodoscopeResponse passed to the constructor.  Pass ``signal`` to
        override (e.g. when fitting a synthetic count vector).

        Parameters
        ----------
        model : callable ``params -> spectrum``
            Returns the incident spectrum on ``self.energy_grid``.
            Use the first element of any model factory's return tuple.
        initial_params : shape (n_params,)
        signal : shape (n_channels,), optional
            Override the signal stored in the HodoscopeResponse.  Useful for
            testing with synthetic count vectors.
        param_names : display labels (defaults to p0, p1, ...).
        bounds : ``(lower_bounds, upper_bounds)`` for scipy. Use the ``param_bounds``
            returned by the model factory as a starting point.

        Returns
        -------
        ForwardFittingResult
            Call ``components_model(result.params)`` on the factory's second return
            value to decompose the fitted spectrum for plotting.
        """
        if signal is not None and inject_noise:
            raise ValueError('Cannot pass both signal= and inject_noise=True.')
        if signal is not None:
            # Explicit signal override: treat as net signal, no background subtraction.
            # Useful for synthetic count vectors (e.g. R.T @ f_true). Background shot noise
            # is still folded into the weights so the fit sees realistic per-channel uncertainty.
            signal_std = np.sqrt(np.maximum(signal, 1.0) + self._background_std ** 2)
        else:
            signal = self._signal
            signal_std = np.sqrt(self._signal_std ** 2 + self._background_std ** 2)
            if inject_noise:
                # TODO: Gaussian noise is not strictly correct for low counts, but it's a reasonable approximation for testing.
                signal = np.random.default_rng().normal(signal, self._signal_std)

        if param_names is None:
            param_names = [f'p{i}' for i in range(len(initial_params))]

        bounds = (-np.inf, np.inf) if bounds is None else bounds
        lo, hi = bounds[0], bounds[1]
        valid = signal_std > 0
        if not np.any(valid):
            raise ValueError('All channels have signal_std == 0; cannot fit.')

        signal_valid = signal[valid]
        signal_std_valid = signal_std[valid]
        # Use a local copy of R restricted to valid channels so self.R is never mutated.
        R_valid = self.R[:, valid]

        def _residuals(params: np.ndarray) -> np.ndarray:
            try:
                predicted = R_valid.T @ model(params)
            except ValueError:
                return np.full(len(signal_valid), 1e10)
            return (predicted - signal_valid) / signal_std_valid

        if use_global_optimizer and bounds != (-np.inf, np.inf):
            hi_de = np.where(np.isinf(hi), np.abs(initial_params) * 1e3, hi)
            lo_de = np.where(np.isinf(lo), 0.0, lo)
            de_bounds = list(zip(lo_de, hi_de))
            def _de_objective(p):
                try:
                    return float(np.sum(_residuals(p) ** 2))
                except ValueError:
                    return 1e30

            de_result = differential_evolution(
                _de_objective,
                de_bounds,
                seed=42,
                maxiter=500,
                tol=1e-6,
                polish=False,
            )
            p0 = de_result.x
        else:
            p0 = initial_params

        opt = least_squares(_residuals, p0, bounds=bounds)

        p_opt = opt.x
        # Degrees of freedom: valid channels minus fitted parameters
        n_dof = max(int(valid.sum()) - len(p_opt), 1)
        chi_sq_total = float(np.sum(opt.fun ** 2))
        chi_sq_reduced = chi_sq_total / n_dof

        # Find the parameter uncertainties from the Jacobian at the solution
        J = opt.jac  # (n_ch, n_params)
        # Scale the covariance by the reduced chi-square
        cov = np.linalg.pinv(J.T @ J) * chi_sq_reduced
        param_uncertainties = np.sqrt(np.maximum(np.diag(cov), 0.0))

        # Find the spectrum uncertainties
        f_opt = model(p_opt)
        # Arbitrarily small step for finite difference, scaled to parameter magnitude
        eps = np.maximum(np.abs(p_opt) * 1e-5, 1e-10)
        n_E = len(self.energy_grid)
        J_model = np.zeros((n_E, len(p_opt)))
        for i in range(len(p_opt)):
            dp = np.zeros(len(p_opt))
            dp[i] = eps[i]
            # Clamp to bounds so finite-difference steps don't leave the model's valid domain
            p_plus = np.clip(p_opt + dp, lo, hi)
            p_minus = np.clip(p_opt - dp, lo, hi)
            step = p_plus[i] - p_minus[i]
            if step == 0.0:
                continue  # parameter pinned at a bound; leave column as zero
            try:
                J_model[:, i] = (model(p_plus) - model(p_minus)) / step
            except ValueError:
                pass  # outside grid convex hull; leave column as zero
        # Mathematically, to first order: var(f_j) = sum_{k,l} (df_j/dp_k) * cov[k,l] * (df_j/dp_l)
        spec_cov_diag = np.einsum('ij,jk,ik->i', J_model, cov, J_model)
        spectrum_uncertainties = np.sqrt(np.maximum(spec_cov_diag, 0.0))

        bw = np.empty(len(self.energy_grid))
        bw[:-1] = np.diff(self.energy_grid)
        bw[-1] = bw[-2]

        result = ForwardFittingResult(
            spectrum=f_opt,
            uncertainties=spectrum_uncertainties,
            energy_grid=self.energy_grid,
            chi_square=chi_sq_reduced,
            converged=opt.success,
            n_iterations=opt.nfev,
            params=p_opt,
            param_names=list(param_names),
            param_uncertainties=param_uncertainties,
            covariance_matrix=cov,
            message=opt.message,
            model_name=getattr(model, 'model_name', ''),
            bin_widths=bw,
            signal=signal[valid],
            sigma_counts=signal_std_valid,
            response_matrix=R_valid,
            background=self._background[valid],
        )
        result.summary(true_params=true_params)
        return result

def compute_spectrum_data_points(
    result: ForwardFittingResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-channel hodoscope counts to approximate spectrum units for visualization.

    For each hodoscope channel k this computes:
      - A nominal incident energy E_k as the response-matrix weighted mean over the energy grid.
      - An energy spread sigma_E_k as the response-matrix weighted rms (x error bar).
      - A background-subtracted spectrum estimate in the same units as the fitted spectrum.
      - A propagated 1-sigma uncertainty on that estimate (y error bar).

    This function is a **visualization utility only**.  It uses a single-channel approximation
    (dividing by the total column response) and does not replace the parametric forward fit.

    Parameters
    ----------
    result : ForwardFittingResult
        Must contain result.signal, result.sigma_counts, result.response_matrix,
        and result.energy_grid. These are populated automatically by SpectrumFitter.fit().

    Returns
    -------
    spectrum_data : shape (n_valid_channels,)
        Signal divided by total column response.  Same units as
        result.spectrum (particles or MeV, depending on the response matrix).
    spectrum_data_sigma : shape (n_valid_channels,)
        1-sigma uncertainty on spectrum_data.
    nominal_energy : shape (n_valid_channels,)
        Response-matrix weighted mean incident energy per channel [MeV].
    energy_spread : shape (n_valid_channels,)
        Response-matrix weighted rms energy width per channel [MeV] (x error bar).
    """
    if result.signal is None or result.response_matrix is None or result.sigma_counts is None:
        raise ValueError(
            'ForwardFittingResult is missing signal, sigma_counts, or response_matrix. '
            'Re-run SpectrumFitter.fit() to populate these fields.'
        )

    R = result.response_matrix  # (n_energies, n_valid_channels)
    energy_grid = result.energy_grid  # (n_energies,)
    signal = result.signal             # (n_valid_channels,)
    sigma = result.sigma_counts        # (n_valid_channels,)

    # Per-channel weighted mean and rms energy from the response matrix columns.
    col_sums = R.sum(axis=0)  # (n_valid_channels,) - total response per channel
    col_sums_safe = np.where(col_sums > 0, col_sums, 1.0)

    nominal_energy = (energy_grid[:, np.newaxis] * R).sum(axis=0) / col_sums_safe
    energy_variance = ((energy_grid[:, np.newaxis] - nominal_energy[np.newaxis, :]) ** 2 * R).sum(axis=0) / col_sums_safe
    energy_spread = np.sqrt(np.maximum(energy_variance, 0.0))

    spectrum_data = signal / col_sums_safe
    spectrum_data_sigma = sigma / col_sums_safe

    # Edge-of-acceptance channels have noisy MC response; flag them like empty ones (nominal_energy == 0)
    has_response = col_sums > 0
    left_edge = np.r_[True, ~has_response[:-1]]
    right_edge = np.r_[~has_response[1:], True]
    boundary_artifact = has_response & (left_edge | right_edge)
    nominal_energy[boundary_artifact] = 0.0

    return spectrum_data, spectrum_data_sigma, nominal_energy, energy_spread
