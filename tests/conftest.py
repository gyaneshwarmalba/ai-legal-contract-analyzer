from __future__ import annotations

import pytest

from app.config import Settings
from app.pipeline import ContractAnalyzer


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="local",
        embeddings_provider="local",
        vector_store="memory",
        result_sink="local",
        data_dir=tmp_path,
        seed_sample_data=True,
    )


@pytest.fixture()
def analyzer(settings) -> ContractAnalyzer:
    return ContractAnalyzer(settings)
