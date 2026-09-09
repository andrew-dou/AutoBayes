"""C1-C2: device resolution is explicit and honest."""
import pytest

import autobayes as ab


def test_gpu_on_pymc_backend_is_error(model):
    with pytest.raises(ab.DeviceError, match="CPU only"):
        model.fit(backend="pymc", device="gpu", chains=1, draws=5, tune=5)


def test_gpu_unavailable_is_error_not_fallback(model):
    jax = pytest.importorskip("jax")
    if any(d.platform in {"gpu", "cuda", "rocm"} for d in jax.devices()):
        pytest.skip("a GPU is visible; this test covers the no-GPU case")
    with pytest.raises(ab.DeviceError, match="does not fall back to CPU silently"):
        model.fit(backend="numpyro", device="gpu", chains=1, draws=5, tune=5)


def test_bad_device_and_backend(model):
    with pytest.raises(ab.DeviceError):
        model.fit(backend="pymc", device="tpu")
    with pytest.raises(ValueError):
        model.fit(backend="stan")


def test_numpyro_cpu_report(model):
    pytest.importorskip("numpyro")
    r = model.fit(backend="numpyro", device="cpu", chains=2, draws=100, tune=100, seed=2)
    assert r.backend == "numpyro" and r.platform == "cpu" and r.dtype == "float64"
    assert r.chain_method == "parallel"
    assert r.idata.posterior.sizes == {"chain": 2, "draw": 100, "g__factor_dim": model.design.J}
    assert r.idata.posterior.attrs.get("inference_library") == "numpyro"
    text = r.report()
    assert "device actual    : cpu" in text and "float64" in text
