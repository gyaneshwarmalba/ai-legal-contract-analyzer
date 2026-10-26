"""Human-in-the-loop review queue backed by a JSON file.

Nothing is published to downstream systems until a reviewer approves it. The
file is rewritten atomically so a crash cannot leave a half-written queue.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATUSES = ("pending", "approved", "rejected")


@dataclass
class ReviewItem:
    kind: str
    contract_id: str
    reason: str
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    reviewer: str | None = None
    note: str | None = None
    decided_at: str | None = None


class ReviewQueue:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._items: dict[str, ReviewItem] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._items = {i["id"]: ReviewItem(**i) for i in raw}
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.exception("review queue file is unreadable; starting empty", extra={"path": str(self.path)})
            self._items = {}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump([asdict(i) for i in self._items.values()], fh, indent=2)
        os.replace(tmp, self.path)

    def submit(self, kind: str, contract_id: str, reason: str, payload: dict[str, Any]) -> ReviewItem:
        with self._lock:
            for existing in self._items.values():
                if existing.status == "pending" and existing.kind == kind and existing.payload == payload:
                    return existing
            item = ReviewItem(kind=kind, contract_id=contract_id, reason=reason, payload=payload)
            self._items[item.id] = item
            self._persist()
        logger.info("review item submitted", extra={"review_id": item.id, "kind": kind, "contract_id": contract_id})
        return item

    def list(self, status: str | None = None) -> list[ReviewItem]:
        with self._lock:
            items = list(self._items.values())
        if status:
            items = [i for i in items if i.status == status]
        return sorted(items, key=lambda i: i.created_at, reverse=True)

    def get(self, review_id: str) -> ReviewItem | None:
        return self._items.get(review_id)

    def decide(self, review_id: str, approve: bool, reviewer: str, note: str | None = None) -> ReviewItem:
        with self._lock:
            item = self._items.get(review_id)
            if item is None:
                raise KeyError(review_id)
            if item.status != "pending":
                raise ValueError(f"review {review_id} is already {item.status}")
            item.status = "approved" if approve else "rejected"
            item.reviewer = reviewer
            item.note = note
            item.decided_at = datetime.now(UTC).isoformat()
            self._persist()
        logger.info("review decided", extra={"review_id": review_id, "status": item.status, "reviewer": reviewer})
        return item
