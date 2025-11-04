from jax import numpy as jnp

from diffusion_mem_gen.utils.ou_process import (
    build_anisotropic_ou_generators,
    capacity_crossover_fraction,
    effective_observation_covariance,
    forward_mean,
    gamma_operator,
    posterior_denoiser,
    posterior_weights,
    sigma_sto,
    stationary_covariance,
)


def test_build_anisotropic_ou_generators_properties():
    dim = 5
    U, Q, A = build_anisotropic_ou_generators(dim)

    expected_diag = jnp.linspace(1.0, 9.0, dim)
    assert jnp.allclose(jnp.diag(U), expected_diag)
    assert jnp.allclose(U, jnp.diag(jnp.diag(U)))

    assert jnp.allclose(Q + Q.T, jnp.zeros_like(Q))
    off_diag = Q[jnp.triu_indices(dim, k=1)]
    assert jnp.allclose(off_diag, jnp.full(off_diag.shape, 0.01))
    assert jnp.allclose(Q[jnp.tril_indices(dim, k=-1)], -0.01)

    identity = jnp.eye(dim)
    assert jnp.allclose(A, (identity + Q) @ U)


def test_sigma_sto_matches_stationary_limit():
    dim = 3
    U, Q, _ = build_anisotropic_ou_generators(dim)
    t = jnp.array([0.0, 0.5, 1.5])
    sigma_t = sigma_sto(t, U, Q)

    zero_cov = sigma_t[0]
    assert jnp.allclose(zero_cov, jnp.zeros_like(zero_cov))

    sigma_stationary = stationary_covariance(U)
    large_t = sigma_sto(jnp.array(10.0), U, Q)
    assert jnp.allclose(large_t, sigma_stationary, atol=1e-4)


def test_gamma_operator_is_psd():
    dim = 4
    U, Q, _ = build_anisotropic_ou_generators(dim)
    sigma_star = jnp.eye(dim) * 2.0
    ts = jnp.linspace(0.1, 1.0, 5)

    gamma_ts = gamma_operator(ts, sigma_star, U, Q)

    for gamma in gamma_ts:
        # Symmetry
        assert jnp.allclose(gamma, gamma.T, atol=1e-6)
        eigvals = jnp.linalg.eigvalsh(gamma)
        assert jnp.all(eigvals >= -1e-6)

        # Eigenvalues decrease smoothly with time due to mixing
        # Ensure they are finite and real
        assert jnp.all(jnp.isfinite(eigvals))


def test_forward_mean_and_effective_covariance_at_zero_time():
    dim = 3
    U, Q, _ = build_anisotropic_ou_generators(dim)
    means = jnp.stack([jnp.arange(dim), -jnp.arange(dim)])
    covariances = jnp.stack([jnp.eye(dim), 2.0 * jnp.eye(dim)])

    propagated = forward_mean(jnp.array(0.0), means, U, Q)
    assert jnp.allclose(propagated, means)

    C_t = effective_observation_covariance(jnp.array(0.0), covariances, U, Q)
    assert jnp.allclose(C_t, covariances)


def test_posterior_weights_and_denoiser_prefer_closest_component():
    dim = 2
    U, Q, _ = build_anisotropic_ou_generators(dim)
    means = jnp.array([[2.0, 0.0], [-2.0, 0.0]])
    cov = jnp.stack([jnp.eye(dim) * 0.5, jnp.eye(dim) * 0.5])
    priors = jnp.array([0.5, 0.5])
    x = jnp.array([2.1, -0.1])
    t = jnp.array(0.0)

    weights = posterior_weights(x, t, means, cov, priors, U, Q)
    assert jnp.allclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]

    denoised = posterior_denoiser(x, t, means, cov, priors, U, Q)
    assert jnp.allclose(denoised, means[0], atol=1e-2)


def test_capacity_crossover_fraction_is_bounded():
    dim = 3
    U, Q, _ = build_anisotropic_ou_generators(dim)
    sigma_star = jnp.eye(dim)
    ts = jnp.linspace(0.1, 1.0, 4)
    lambda_weights = jnp.ones_like(ts)

    frac = capacity_crossover_fraction(ts, lambda_weights, sigma_star, U, Q)
    assert 0.0 <= float(frac) <= 1.0

