from pathlib import Path

from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py"


def test_streamlit_ui_renders(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_file(str(UI), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "AI Legal Contract Analyzer"
    assert at.metric[0].value.endswith("/100")
