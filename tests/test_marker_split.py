"""The automatic `sampling` mark (tests/conftest.py): detection by source text, directly, through fixtures and through
same-module helpers; the opt-out; and that the fast suite's known sampling sites carry the mark."""
import pytest

import tests.test_marker_split as me
from tests import conftest


class _Model:
    def fit(self, **kw):
        return kw


def _direct():
    return _Model().fit(chains=1)


def _through_one_helper():
    return _direct()


def _through_two_helpers():
    return _through_one_helper()


def _clean():
    return _Model()


def test_detection_direct_through_helpers_and_clean():
    assert conftest.calls_sampler(_direct) and conftest.calls_sampler(_direct, me)
    assert conftest.calls_sampler(_through_one_helper, me) and conftest.calls_sampler(_through_two_helpers, me)
    assert not conftest.calls_sampler(_through_two_helpers)          # helpers are only followed inside a module
    assert not conftest.calls_sampler(_clean, me)
    assert not conftest.calls_sampler(len, me)                        # no retrievable source: not marked
    assert not conftest.calls_sampler(conftest.calls_sampler, conftest) and not conftest.calls_sampler(conftest.simulate, conftest)


@pytest.mark.nosampling
def test_opt_out_is_respected(request):
    assert _through_two_helpers() == {"chains": 1}                    # a helper chain ending in a fit call: marked without the opt-out
    assert request.node.get_closest_marker("sampling") is None and request.node.nodeid not in request.config.sampling_nodeids


def test_known_sampling_sites_are_marked_and_deterministic_modules_are_not(request):
    sampling_files = {"tests/test_predict.py", "tests/test_session.py", "tests/test_device.py", "tests/test_fresh_process.py", "tests/test_sampling.py"}
    if not sampling_files <= request.config.collected_files:
        pytest.skip("needs the whole suite collected")
    marked = request.config.sampling_nodeids
    files = {nodeid.split("::")[0] for nodeid in marked}
    assert sampling_files <= files
    assert not files & {"tests/test_formula.py", "tests/test_logp.py", "tests/test_population_only.py", "tests/test_marker_split.py"}
    assert "tests/test_predict.py::test_in_sample_predictive_meaning" in marked          # through the small_fit fixture
    assert request.node.get_closest_marker("sampling") is None
