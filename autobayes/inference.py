"""Backend and device resolution, sampling, and an honest device report.

Two backends:

- ``"pymc"``: PyMC's own NUTS, CPU only. ``device="gpu"`` is an error here rather than
  a silent fallback.
- ``"numpyro"``: the PyTensor graph is compiled to JAX and sampled by NumPyro's NUTS
  through ``pm.sample(nuts_sampler="numpyro")``. ``device`` selects the JAX device the
  sampler runs on. A requested device that JAX cannot see raises ``DeviceError``.

The report on the returned ``FitResult`` states what actually ran: backend, requested
and actual device, dtype, chain method, and wall time. Nothing here upgrades "the code
supports GPU" into "a GPU was used".
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

_GPU_PLATFORMS = {"gpu", "cuda", "rocm"}


class DeviceError(RuntimeError):
    """The requested device is not available for the requested backend."""


@dataclass
class FitResult:
    idata: object
    backend: str
    device_requested: str
    device_actual: str
    platform: str
    dtype: str
    chain_method: str | None
    chains: int
    draws: int
    tune: int
    target_accept: float
    wall_seconds: float
    notes: list[str] = field(default_factory=list)
    seed: int | None = None
    _target_identity_json: str | None = field(default=None, repr=False)

    @property
    def target_identity(self):
        """A defensive copy of the target captured before sampling, if available."""
        if self._target_identity_json is None:
            return None
        try:
            identity = json.loads(self._target_identity_json)
            if identity is None:
                raise ValueError("null identity")
            return identity
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid fit-result target identity") from exc

    def report(self) -> str:
        lines = [
            "AutoBayes fit report",
            f"  backend          : {self.backend}",
            f"  device requested : {self.device_requested}",
            f"  device actual    : {self.device_actual} (platform {self.platform})",
            f"  dtype            : {self.dtype}",
            f"  chain method     : {self.chain_method or 'n/a'}",
            f"  chains/tune/draws: {self.chains}/{self.tune}/{self.draws}, target_accept {self.target_accept}",
            f"  seed             : {self.seed}",
            f"  wall time        : {self.wall_seconds:.1f} s (includes compilation)",
        ]
        lines += [f"  note             : {n}" for n in self.notes]
        return "\n".join(lines)


def _jax_devices_summary(jax) -> str:
    return ", ".join(f"{d.platform}:{getattr(d, 'device_kind', '?')}(id={d.id})" for d in jax.devices())


def resolve_jax_device(device: str, chains: int, notes: list[str]):
    """Return (jax_device, chain_method). Raises DeviceError if ``device`` is absent."""
    try:
        import jax  # noqa: F401  (import only; backends initialise on first device query)
        import numpyro
    except ImportError as e:
        raise DeviceError(
            "backend='numpyro' needs jax, jaxlib and numpyro, which are not importable here "
            f"({e}). Install the [jax] extra (uv sync --extra jax) or use backend='pymc'."
        ) from e

    if device == "cpu":
        # Must run before the JAX backend initialises, else it is a no-op.
        numpyro.set_host_device_count(int(chains))
    jax.config.update("jax_enable_x64", True)
    devs = jax.devices()
    if device == "gpu":
        gpus = [d for d in devs if d.platform in _GPU_PLATFORMS]
        if not gpus:
            raise DeviceError(
                "device='gpu' requested for backend='numpyro', but JAX sees no GPU. "
                f"Visible devices: {_jax_devices_summary(jax)}. Install a CUDA-enabled "
                "jaxlib on a machine with an NVIDIA GPU, or pass device='cpu'. "
                "AutoBayes does not fall back to CPU silently."
            )
        return gpus[0], "vectorized"
    if device == "cpu":
        cpus = jax.devices("cpu")
        if chains > 1 and len(cpus) < chains:
            notes.append(
                f"only {len(cpus)} CPU device(s) visible to JAX for {chains} chains: the JAX "
                "backend was initialised before autobayes could set the host device count "
                "(import autobayes before anything that touches jax.devices()). Chains run "
                "sequentially; the posterior is the same, the wall time is not."
            )
            return cpus[0], "sequential"
        return cpus[0], "parallel"
    raise DeviceError(f"unknown device {device!r}; use 'cpu' or 'gpu'")


def fit(model, *, backend: str = "pymc", device: str = "cpu", chains: int = 4,
        draws: int = 1000, tune: int = 1000, seed: int | None = None,
        target_accept: float = 0.9, cores: int | None = None,
        chain_method: str | None = None, progressbar: bool = False,
        **sample_kwargs) -> FitResult:
    """Sample the model's posterior on an explicitly chosen backend and device."""
    if backend not in {"pymc", "numpyro"}:
        raise ValueError(f"backend must be 'pymc' or 'numpyro', got {backend!r}")
    if device not in {"cpu", "gpu"}:
        raise DeviceError(f"unknown device {device!r}; use 'cpu' or 'gpu'")
    from autobayes.identity import FIT_IDENTITY_ATTR
    target_json = json.dumps(model.target_identity, sort_keys=True)
    notes: list[str] = []
    common = dict(draws=draws, tune=tune, chains=chains, random_seed=seed,
                  target_accept=target_accept, progressbar=progressbar, **sample_kwargs)

    if backend == "pymc":
        if device != "cpu":
            raise DeviceError(
                "backend='pymc' runs PyMC's own NUTS on the CPU only; for a GPU use "
                "backend='numpyro' with device='gpu'."
            )
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = model.bambi_model.fit(inference_method="pymc", cores=cores, **common)
        wall = time.perf_counter() - t0
        import pytensor

        idata.attrs[FIT_IDENTITY_ATTR] = target_json
        return FitResult(idata=idata, backend="pymc", device_requested="cpu",
                         device_actual="cpu", platform="cpu",
                         dtype=str(pytensor.config.floatX), chain_method=None,
                         chains=chains, draws=draws, tune=tune, target_accept=target_accept,
                         wall_seconds=wall, notes=notes, seed=seed, _target_identity_json=target_json)

    jax_device, default_method = resolve_jax_device(device, chains, notes)
    method = chain_method or default_method
    import jax
    import jax.numpy as jnp

    t0 = time.perf_counter()
    with jax.default_device(jax_device):
        probe = jnp.ones(1)
        actual = next(iter(probe.devices()))
        dtype = str(probe.dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = model.bambi_model.fit(
                inference_method="numpyro",
                nuts_sampler_kwargs={"chain_method": method},
                **common,
            )
    wall = time.perf_counter() - t0
    if actual.platform in _GPU_PLATFORMS and device != "gpu":
        notes.append("arrays landed on a GPU although device='cpu' was requested; check jax.default_device")
    if dtype != "float64":
        notes.append(f"JAX default dtype is {dtype}, not float64; see reference/pymc-to-jax.md")
    idata.attrs[FIT_IDENTITY_ATTR] = target_json
    return FitResult(idata=idata, backend="numpyro", device_requested=device,
                     device_actual=f"{actual.platform}:{getattr(actual, 'device_kind', '?')}(id={actual.id})",
                     platform=actual.platform, dtype=dtype, chain_method=method,
                     chains=chains, draws=draws, tune=tune, target_accept=target_accept,
                     wall_seconds=wall, notes=notes, seed=seed, _target_identity_json=target_json)
