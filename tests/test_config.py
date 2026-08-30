from pathlib import Path

import pytest

from datacoding.config import DATA_ROOT, external_path


def test_default_data_root_is_outside_vault():
    vault = Path(__file__).resolve().parents[1]
    assert DATA_ROOT == Path.home() / "Desktop" / "geek" / "DataCoding-data"
    assert vault not in DATA_ROOT.parents


def test_external_path_uses_known_areas():
    assert external_path("processed", "demo.csv") == DATA_ROOT / "processed" / "demo.csv"


@pytest.mark.parametrize("unsafe_part", ["../models/oops.pt", "/tmp/oops.pt"])
def test_external_path_rejects_escape_attempts(unsafe_part):
    with pytest.raises(ValueError, match="must stay inside"):
        external_path("raw", unsafe_part)
