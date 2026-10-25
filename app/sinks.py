"""Destinations for reviewer-approved results.

`SalesforceSink` writes an approved risk report to a custom object so results
land where legal teams already work. The object and field API names are
configurable; see README for the expected schema.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ResultSink(Protocol):
    name: str

    def publish(self, review: dict[str, Any]) -> str: ...


class LocalJsonSink:
    name = "local-json"

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

    def publish(self, review: dict[str, Any]) -> str:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{review['contract_id']}-{review['id']}.json"
        record = {**review, "published_at": datetime.now(UTC).isoformat()}
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info("result published", extra={"sink": self.name, "path": str(path)})
        return str(path)


class SalesforceSink:
    name = "salesforce"

    def __init__(self) -> None:
        try:
            from simple_salesforce import Salesforce
        except ImportError as exc:
            raise RuntimeError("RESULT_SINK=salesforce requires simple-salesforce (requirements-cloud.txt)") from exc
        self._sf = Salesforce(
            username=os.environ["SF_USERNAME"],
            password=os.environ["SF_PASSWORD"],
            security_token=os.environ["SF_SECURITY_TOKEN"],
            domain=os.getenv("SF_DOMAIN", "login"),
        )
        self.sobject = os.getenv("SF_RESULT_OBJECT", "Contract_Analysis__c")

    def publish(self, review: dict[str, Any]) -> str:
        payload = review["payload"]
        record = {
            "Contract_Key__c": review["contract_id"],
            "Review_Kind__c": review["kind"],
            "Risk_Score__c": payload.get("score"),
            "Risk_Level__c": payload.get("level"),
            "Reviewer__c": review.get("reviewer"),
            "Reviewer_Note__c": review.get("note"),
            "Details_JSON__c": json.dumps(payload)[:131000],
        }
        try:
            result = getattr(self._sf, self.sobject).create(record)
        except Exception:
            logger.exception("salesforce publish failed", extra={"sobject": self.sobject, "review_id": review["id"]})
            raise
        logger.info("result published", extra={"sink": self.name, "record_id": result.get("id")})
        return str(result.get("id"))
