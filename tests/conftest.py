"""Shared pytest setup.

Expose 4 host CPU devices to JAX so numpyro runs MCMC chains in PARALLEL
instead of sequentially. This must happen before JAX initializes its
backend, i.e. before any test module imports jax -- conftest.py is
imported first, which is why it lives here. Skipped silently when the
`[model]` extra (numpyro) is not installed.
"""

try:
    import numpyro

    numpyro.set_host_device_count(4)
except ImportError:  # pragma: no cover - model extra not installed
    pass
