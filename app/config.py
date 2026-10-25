"""Runtime configuration loaded from environment variables.

Every external dependency (Bedrock, FAISS, Salesforce) is selected here, so the
same pipeline runs fully offline by default and switches to cloud services by
changing configuration only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at runtime
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "local"))
    embeddings_provider: str = field(default_factory=lambda: _env("EMBEDDINGS_PROVIDER", "local"))
    vector_store: str = field(default_factory=lambda: _env("VECTOR_STORE", "memory"))
    result_sink: str = field(default_factory=lambda: _env("RESULT_SINK", "local"))

    aws_region: str = field(default_factory=lambda: _env("AWS_DEFAULT_REGION", "us-west-2"))
    bedrock_model_id: str = field(default_factory=lambda: _env("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"))
    bedrock_embed_model_id: str = field(default_factory=lambda: _env("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"))

    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))))
    risk_rules_path: Path = field(
        default_factory=lambda: Path(_env("RISK_RULES_PATH", str(PROJECT_ROOT / "policies" / "risk_rules.json")))
    )
    seed_sample_data: bool = field(default_factory=lambda: _env_bool("SEED_SAMPLE_DATA", True))

    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.08")))

    @property
    def sample_contracts_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "sample_contracts"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"
