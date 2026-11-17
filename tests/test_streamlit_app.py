"""Smoke test for the Streamlit fallback app (skipped when Streamlit or the model is not installed)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytest.importorskip("streamlit")
pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "sitesafe_base.onnx").exists(), reason="model not present")


def test_app_starts_and_analyses_a_photo(monkeypatch):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "streamlit_app" / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    assert "SiteSafe" in at.title[0].value
    assert len(at.tabs) == 3


def test_analysis_function_on_a_sample():
    import sys
    import cv2
    sys.path.insert(0, str(ROOT))
    img = ROOT / "web" / "public" / "samples" / "s01.jpg"
    if not img.exists():
        pytest.skip("sample images not built")
    from sitesafe.classes import SITE_PROFILES
    from sitesafe.detector import Detector
    from sitesafe.matching import match_people
    rgb = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
    people = match_people(Detector(str(ROOT / "models" / "sitesafe_base.onnx"))(rgb), SITE_PROFILES["general"])
    assert len(people) >= 1
