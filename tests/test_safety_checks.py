import pytest

from badil.linking.build import MEDICINES_FILE, SUBSTITUTES_FILE
from badil.safety.checks import run_checks


@pytest.mark.skipif(not (MEDICINES_FILE.exists() and SUBSTITUTES_FILE.exists()), reason="graph not built")
def test_built_graph_respects_safety_rules():
    failed = {name: items[:3] for name, items in run_checks().items() if items}
    assert not failed, failed
