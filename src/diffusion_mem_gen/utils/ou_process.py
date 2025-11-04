"""Utilities for the anisotropic Ornstein--Uhlenbeck diffusion setting.

This module implements the matrix-valued quantities that appear in the
non-isotropic forward process specification described in the accompanying
research notes.  The construction mirrors
:math:`\S\ref{sec:forward}`--:math:`\S\ref{sec:capacity}` and exposes helpers
that other modules can use to build the anisotropic process end-to-end:

* The drift operator is ``A = (I + Q) U`` with a positive definite diagonal
  ``U`` and a small skew-symmetric ``Q``.
* ``Sigma_sto(t)`` follows Eq. (\ref{eq:sigsto}) and is used to whiten
  geometries and compute Tweedie-style estimators.
* ``Gamma(t)`` follows Eq. (\ref{eq:gamma}) and provides the spectral SNR
  operator that replaces the scalar :math:`\psi_t` schedule.
* Mixture-specific helpers implement Eq. (\ref{eq:denoiser}) and the posterior
  responsibilities required for Section :math:`\S\ref{sec:capacity}`.
* ``capacity_crossover_fraction`` realises Eq. (\ref{eq:xo}) for discrete time
  quadrature rules.

All routines are implemented with ``jax.numpy`` so that they can participate in
JIT compilation or automatic differentiation when required.
"""

from __future__ import annotations

from typing import Tuple

import jax
from jax import Array, numpy as jnp
from jax.scipy.linalg import expm


def _drift_operator(U: Array, Q: Array) -> Array:
    """Compute the drift operator ``A = (I + Q) @ U``."""

    identity = jnp.eye(U.shape[0], dtype=U.dtype)
    return (identity + Q) @ U


def build_anisotropic_ou_generators(
    dim: int,
    q_strength: float = 0.01,
) -> Tuple[Array, Array, Array]:
    """Construct ``U``, ``Q`` and ``A`` for the anisotropic OU process.

    Args:
        dim: Dimensionality ``d`` of the ambient space.
        q_strength: Magnitude of the skew-symmetric perturbation ``Q``.

    Returns:
        A tuple ``(U, Q, A)`` where ``U`` is positive definite diagonal with
        diagonal entries ``linspace(1, 9, dim)``, ``Q`` is skew-symmetric with a
        constant magnitude ``q_strength`` off the diagonal (positive in the
        strictly upper-triangular portion and negative in the lower-triangular
        portion), and ``A = (I + Q) @ U``.
    """

    diag_entries = jnp.linspace(1.0, 9.0, dim)
    U = jnp.diag(diag_entries)

    upper = jnp.triu(jnp.ones((dim, dim), dtype=U.dtype), k=1)
    Q = q_strength * (upper - upper.T)

    A = _drift_operator(U, Q)
    return U, Q, A


def stationary_covariance(U: Array) -> Array:
    """Return the stationary covariance ``Sigma_s = U^{-1}``."""

    return jnp.linalg.inv(U)


def sigma_sto(t: Array, U: Array, Q: Array) -> Array:
    """Compute the stochastic covariance ``Sigma_sto(t)``.

    ``Sigma_sto(t) = U^{-1} - e^{-A t} U^{-1} e^{-A^T t}`` with
    ``A = (I + Q) U``.
    """

    A = _drift_operator(U, Q)
    U_inv = jnp.linalg.inv(U)

    def _sigma_single(t_scalar: Array) -> Array:
        exp_neg_At = expm(-A * t_scalar)
        sigma = U_inv - exp_neg_At @ U_inv @ exp_neg_At.T
        return _symmetrize(sigma)

    if jnp.ndim(t) == 0:
        return _sigma_single(t)
    return jax.vmap(_sigma_single)(t)


def _symmetrize(mat: Array) -> Array:
    """Return the symmetric part of ``mat``."""

    return 0.5 * (mat + mat.T)


def _symmetric_matrix_power(mat: Array, power: float) -> Array:
    """Raise a symmetric matrix to a (possibly fractional) power."""

    eigvals, eigvecs = jnp.linalg.eigh(_symmetrize(mat))
    clipped = jnp.clip(eigvals, a_min=1e-12)
    powered = clipped**power
    return _symmetrize((eigvecs * powered[None, :]) @ eigvecs.T)


def gamma_operator(t: Array, sigma_star: Array, U: Array, Q: Array) -> Array:
    """Compute the spectral SNR operator ``Gamma(t)``.

    Args:
        t: Time(s) at which to evaluate the operator.
        sigma_star: Typical intra-cluster covariance ``Sigma_*``.
        U: Symmetric positive definite matrix from the OU drift.
        Q: Skew-symmetric perturbation matrix.

    Returns:
        ``Gamma(t)`` as defined in Eq. (\ref{eq:gamma}).
    """

    sigma_t = sigma_sto(t, U, Q)
    identity = jnp.eye(U.shape[0], dtype=U.dtype)
    A = _drift_operator(U, Q)

    def _gamma_single(sigma_t_single: Array, t_scalar: Array) -> Array:
        sigma_inv_sqrt = _symmetric_matrix_power(sigma_t_single, -0.5)
        exp_neg_At = expm(-A * t_scalar)
        gamma = sigma_inv_sqrt @ exp_neg_At @ sigma_star @ exp_neg_At.T @ sigma_inv_sqrt
        return _symmetrize(gamma)

    if jnp.ndim(t) == 0:
        return _gamma_single(sigma_t, t)

    return jax.vmap(_gamma_single)(sigma_t, t)


def forward_mean(t: Array, means: Array, U: Array, Q: Array) -> Array:
    """Propagate mixture component means under the forward OU flow."""

    A = _drift_operator(U, Q)

    def _single_time(t_scalar: Array) -> Array:
        exp_neg_At = expm(-A * t_scalar)
        return means @ exp_neg_At.T

    if jnp.ndim(t) == 0:
        return _single_time(t)
    return jax.vmap(_single_time)(t)


def effective_observation_covariance(t: Array, covariances: Array, U: Array, Q: Array) -> Array:
    """Return ``C_i(t)`` for each mixture component at time ``t``."""

    sigma_t = sigma_sto(t, U, Q)
    A = _drift_operator(U, Q)

    def _single_time(t_scalar: Array, sigma_t_single: Array) -> Array:
        exp_neg_At = expm(-A * t_scalar)

        def _per_component(cov: Array) -> Array:
            pushed = exp_neg_At @ cov @ exp_neg_At.T
            return _symmetrize(sigma_t_single + pushed)

        return jax.vmap(_per_component)(covariances)

    if jnp.ndim(t) == 0:
        return _single_time(t, sigma_t)

    return jax.vmap(_single_time)(t, sigma_t)


def posterior_weights(
    x: Array,
    t: Array,
    means: Array,
    covariances: Array,
    priors: Array,
    U: Array,
    Q: Array,
    stabilisation: float = 1e-6,
) -> Array:
    """Compute posterior responsibilities ``pi_i(t \mid x)`` for a GMM.

    ``t`` must be a scalar (``Array[]``) or a length-one vector.  The return
    value has shape ``(num_components,)``.
    """

    priors = priors / priors.sum()
    if jnp.size(t) != 1:
        raise ValueError("posterior_weights expects a scalar time input")
    t_scalar = jnp.reshape(t, (1,))[0]

    component_covs = effective_observation_covariance(t_scalar, covariances, U, Q)
    forward_means = forward_mean(t_scalar, means, U, Q)

    identity = jnp.eye(component_covs.shape[-1], dtype=component_covs.dtype)

    def _per_component(mean_t: Array, cov_t: Array, prior: Array) -> Array:
        cov_t = _symmetrize(cov_t) + stabilisation * identity
        cov_inv = jnp.linalg.solve(cov_t, identity)
        diff = x - mean_t
        mahal = diff @ cov_inv @ diff
        log_det = jnp.linalg.slogdet(cov_t)[1]
        return jnp.log(prior) - 0.5 * (log_det + mahal)

    log_weights = jax.vmap(_per_component)(forward_means, component_covs, priors)
    return jax.nn.softmax(log_weights)


def posterior_denoiser(
    x: Array,
    t: Array,
    means: Array,
    covariances: Array,
    priors: Array,
    U: Array,
    Q: Array,
    stabilisation: float = 1e-6,
) -> Array:
    """Matrix-form MMSE denoiser :math:`\bar x_\theta(t, x)` from Eq. (\ref{eq:denoiser})."""

    if jnp.size(t) != 1:
        raise ValueError("posterior_denoiser expects a scalar time input")
    t_scalar = jnp.reshape(t, (1,))[0]
    responsibilities = posterior_weights(
        x, t_scalar, means, covariances, priors, U, Q, stabilisation
    )
    component_covs = effective_observation_covariance(t_scalar, covariances, U, Q)
    forward_means = forward_mean(t_scalar, means, U, Q)

    identity = jnp.eye(component_covs.shape[-1], dtype=component_covs.dtype)
    A = _drift_operator(U, Q)
    exp_neg_At = expm(-A * t_scalar)

    def _posterior(mean_t: Array, cov_t: Array, mean_0: Array, cov_0: Array) -> Array:
        cov_t = _symmetrize(cov_t) + stabilisation * identity
        cov_inv = jnp.linalg.solve(cov_t, identity)
        gain = cov_0 @ exp_neg_At.T @ cov_inv
        diff = x - mean_t
        return mean_0 + gain @ diff

    components = jax.vmap(_posterior)(forward_means, component_covs, means, covariances)
    return jnp.tensordot(responsibilities, components, axes=1)


def capacity_crossover_fraction(
    t: Array,
    lambda_weights: Array,
    sigma_star: Array,
    U: Array,
    Q: Array,
    constant: float = 1.5,
) -> Array:
    """Evaluate the capacity crossover fraction :math:`M^*/N` (Eq. (\ref{eq:xo}))."""

    gamma_ts = gamma_operator(t, sigma_star, U, Q)
    if jnp.ndim(t) == 0:
        gamma_ts = gamma_ts[None, ...]
        lambda_weights = lambda_weights[None]

    gamma_eigs = jax.vmap(jnp.linalg.eigvalsh)(gamma_ts)
    sigma_star_eigs = jnp.linalg.eigvalsh(_symmetrize(sigma_star))
    sigma_star_eigs = jnp.clip(sigma_star_eigs, a_min=1e-12)

    frac = jnp.mean(1.0 / (1.0 + sigma_star_eigs[None, :] * gamma_eigs), axis=1)
    weighted = (lambda_weights * frac).sum()
    denom = constant * lambda_weights.sum()
    return jnp.clip(1.0 - weighted / denom, 0.0, 1.0)


__all__ = [
    "build_anisotropic_ou_generators",
    "stationary_covariance",
    "sigma_sto",
    "gamma_operator",
    "forward_mean",
    "effective_observation_covariance",
    "posterior_weights",
    "posterior_denoiser",
    "capacity_crossover_fraction",
]
