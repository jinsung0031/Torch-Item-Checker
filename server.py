"""FastAPI server for sharing Torch item value observations.

This module exposes a minimal REST API that allows players to submit the
estimated value of their drops and query aggregated statistics that combine the
community's knowledge.  All state is kept in-memory which keeps the
implementation straightforward; for production usage you can swap the
``ItemValueAggregator`` with a database-backed implementation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Internal storage models


@dataclass
class StoredItemEntry:
    """Single submission of an item's value."""

    item_name: str
    value: float
    quantity: int
    currency: str
    map_name: Optional[str]
    source: Optional[str]
    note: Optional[str]
    recorded_at: datetime = field(default_factory=datetime.utcnow)

    def as_dict(self) -> Dict[str, object]:
        return {
            "item_name": self.item_name,
            "value": self.value,
            "quantity": self.quantity,
            "currency": self.currency,
            "map_name": self.map_name,
            "source": self.source,
            "note": self.note,
            "recorded_at": self.recorded_at,
        }


@dataclass
class ItemStats:
    """Aggregated statistics for an item/currency pair."""

    item_name: str
    currency: str
    submissions: int = 0
    total_value: float = 0.0
    total_quantity: int = 0
    min_value: float = field(default=float("inf"))
    max_value: float = field(default=float("-inf"))
    last_updated: datetime = field(default_factory=datetime.utcnow)
    history: Deque[StoredItemEntry] = field(default_factory=deque)

    def add_entry(self, entry: StoredItemEntry, history_limit: int) -> None:
        self.submissions += 1
        self.total_value += entry.value
        self.total_quantity += entry.quantity
        self.min_value = min(self.min_value, entry.value)
        self.max_value = max(self.max_value, entry.value)
        self.last_updated = entry.recorded_at
        self.history.appendleft(entry)
        while len(self.history) > history_limit:
            self.history.pop()

    def snapshot(self) -> Dict[str, object]:
        average_per_submission = (
            self.total_value / self.submissions if self.submissions else 0.0
        )
        average_per_unit = (
            self.total_value / self.total_quantity if self.total_quantity else 0.0
        )
        return {
            "item_name": self.item_name,
            "currency": self.currency,
            "submissions": self.submissions,
            "total_value": self.total_value,
            "total_quantity": self.total_quantity,
            "average_per_submission": average_per_submission,
            "average_per_unit": average_per_unit,
            "min_value": self.min_value if self.submissions else 0.0,
            "max_value": self.max_value if self.submissions else 0.0,
            "last_updated": self.last_updated,
        }


class ItemValueAggregator:
    """Thread-safe in-memory aggregator for value submissions."""

    def __init__(self, history_limit: int = 25) -> None:
        self._history_limit = history_limit
        self._stats: Dict[str, Dict[str, ItemStats]] = {}
        self._lock = Lock()

    def _normalize_name(self, item_name: str) -> str:
        return item_name.strip().lower()

    def _normalize_currency(self, currency: str) -> str:
        return currency.strip().lower()

    def add_entry(self, entry: StoredItemEntry) -> Tuple[StoredItemEntry, ItemStats]:
        normalized_name = self._normalize_name(entry.item_name)
        normalized_currency = self._normalize_currency(entry.currency)
        with self._lock:
            by_currency = self._stats.setdefault(normalized_name, {})
            stats = by_currency.get(normalized_currency)
            if stats is None:
                stats = ItemStats(item_name=entry.item_name, currency=entry.currency)
                by_currency[normalized_currency] = stats
            else:
                # Update with the latest casing provided by the user.
                stats.item_name = entry.item_name
                stats.currency = entry.currency
            stats.add_entry(entry, self._history_limit)
            return entry, stats

    def get_stats(self, item_name: str, currency: Optional[str] = None) -> List[ItemStats]:
        normalized_name = self._normalize_name(item_name)
        with self._lock:
            by_currency = self._stats.get(normalized_name)
            if not by_currency:
                return []
            if currency is None:
                return list(by_currency.values())
            normalized_currency = self._normalize_currency(currency)
            stats = by_currency.get(normalized_currency)
            return [stats] if stats else []

    def iter_stats(self, query: Optional[str] = None) -> Iterable[ItemStats]:
        normalized_query = query.strip().lower() if query else None
        with self._lock:
            for by_currency in self._stats.values():
                for stats in by_currency.values():
                    if normalized_query and normalized_query not in stats.item_name.lower():
                        continue
                    yield stats

    def get_entries(
        self,
        item_name: str,
        currency: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[StoredItemEntry]:
        normalized_name = self._normalize_name(item_name)
        with self._lock:
            by_currency = self._stats.get(normalized_name)
            if not by_currency:
                return []
            if currency is None:
                # Return the newest entries across currencies sorted by timestamp.
                entries: List[StoredItemEntry] = []
                for stats in by_currency.values():
                    entries.extend(list(stats.history))
                entries.sort(key=lambda entry: entry.recorded_at, reverse=True)
            else:
                normalized_currency = self._normalize_currency(currency)
                stats = by_currency.get(normalized_currency)
                if not stats:
                    return []
                entries = list(stats.history)
            if limit is not None:
                entries = entries[:limit]
            return entries


# ---------------------------------------------------------------------------
# Pydantic schemas


class ItemValuePayload(BaseModel):
    """Client submission payload."""

    item_name: str = Field(..., min_length=1, description="드랍된 아이템 이름")
    value: float = Field(..., gt=0, description="아이템의 총 가치 (거래 기준)")
    quantity: int = Field(1, gt=0, description="획득한 수량")
    currency: str = Field("chaos", min_length=1, description="가치에 사용한 화폐")
    map_name: Optional[str] = Field(None, description="아이템을 먹은 맵")
    source: Optional[str] = Field(None, description="제출자 식별용 정보")
    note: Optional[str] = Field(None, description="추가 메모")

    @validator("item_name", "currency", "map_name", "source", "note", pre=True)
    def _strip_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            return None
        return stripped


class ItemEntryResponse(BaseModel):
    item_name: str
    value: float
    quantity: int
    currency: str
    map_name: Optional[str]
    source: Optional[str]
    note: Optional[str]
    recorded_at: datetime


class ItemStatsResponse(BaseModel):
    item_name: str
    currency: str
    submissions: int
    total_value: float
    total_quantity: int
    average_per_submission: float
    average_per_unit: float
    min_value: float
    max_value: float
    last_updated: datetime


class ItemSummaryResponse(BaseModel):
    item_name: str
    aggregates: List[ItemStatsResponse]


class ItemSubmissionResponse(BaseModel):
    entry: ItemEntryResponse
    aggregate: ItemStatsResponse


# ---------------------------------------------------------------------------
# FastAPI wiring


aggregator = ItemValueAggregator()

app = FastAPI(
    title="Torch Item Value Exchange",
    description=(
        "여러 사용자의 드랍 가치를 모아 평균을 계산하는 간단한 API입니다.\n\n"
        "POST /items 로 가치를 제출하고 GET /items/{item_name} 으로 합산 정보를"
        "조회할 수 있습니다."
    ),
    version="0.1.0",
)


def _entry_to_response(entry: StoredItemEntry) -> ItemEntryResponse:
    return ItemEntryResponse(**entry.as_dict())


def _stats_to_response(stats: ItemStats) -> ItemStatsResponse:
    return ItemStatsResponse(**stats.snapshot())


@app.post("/items", response_model=ItemSubmissionResponse, status_code=201)
def submit_item(payload: ItemValuePayload) -> ItemSubmissionResponse:
    entry = StoredItemEntry(
        item_name=payload.item_name,
        value=payload.value,
        quantity=payload.quantity,
        currency=payload.currency,
        map_name=payload.map_name,
        source=payload.source,
        note=payload.note,
    )
    entry, stats = aggregator.add_entry(entry)
    return ItemSubmissionResponse(
        entry=_entry_to_response(entry),
        aggregate=_stats_to_response(stats),
    )


@app.get("/items", response_model=List[ItemStatsResponse])
def list_items(
    query: Optional[str] = Query(
        None,
        description="부분 일치 검색어 (예: 'divine')",
        min_length=1,
    ),
    currency: Optional[str] = Query(
        None,
        description="특정 화폐의 통계만 반환",
        min_length=1,
    ),
    limit: int = Query(
        100,
        description="최대 반환 개수",
        ge=1,
        le=500,
    ),
) -> List[ItemStatsResponse]:
    results: List[ItemStatsResponse] = []
    for stats in aggregator.iter_stats(query=query):
        if currency and stats.currency.lower() != currency.strip().lower():
            continue
        results.append(_stats_to_response(stats))
        if len(results) >= limit:
            break
    return results


@app.get("/items/{item_name}", response_model=ItemSummaryResponse)
def get_item(item_name: str, currency: Optional[str] = Query(None, min_length=1)) -> ItemSummaryResponse:
    stats_list = aggregator.get_stats(item_name, currency=currency)
    if not stats_list:
        raise HTTPException(status_code=404, detail="등록된 정보가 없습니다.")
    return ItemSummaryResponse(
        item_name=item_name,
        aggregates=[_stats_to_response(stats) for stats in stats_list],
    )


@app.get("/items/{item_name}/entries", response_model=List[ItemEntryResponse])
def list_entries(
    item_name: str,
    currency: Optional[str] = Query(None, min_length=1),
    limit: Optional[int] = Query(
        20,
        description="최근 n개의 기록만 반환 (None은 전체)",
        ge=1,
        le=200,
    ),
) -> List[ItemEntryResponse]:
    entries = aggregator.get_entries(item_name, currency=currency, limit=limit)
    if not entries:
        raise HTTPException(status_code=404, detail="등록된 기록이 없습니다.")
    return [_entry_to_response(entry) for entry in entries]


@app.delete("/items/{item_name}", status_code=204)
def delete_item(item_name: str, currency: Optional[str] = Query(None, min_length=1)) -> None:
    """Administrative helper to wipe data for a specific item.

    FastAPI's automatic docs make it clear that this endpoint exists mostly for
    testing/maintenance purposes.  It is deliberately simple and operates purely
    on in-memory data.
    """

    normalized_name = aggregator._normalize_name(item_name)
    normalized_currency = (
        aggregator._normalize_currency(currency)
        if currency is not None
        else None
    )
    with aggregator._lock:  # type: ignore[attr-defined]
        if normalized_name not in aggregator._stats:  # type: ignore[attr-defined]
            raise HTTPException(status_code=404, detail="삭제할 데이터가 없습니다.")
        if normalized_currency is None:
            del aggregator._stats[normalized_name]  # type: ignore[attr-defined]
        else:
            by_currency = aggregator._stats[normalized_name]  # type: ignore[attr-defined]
            if normalized_currency not in by_currency:
                raise HTTPException(status_code=404, detail="삭제할 데이터가 없습니다.")
            del by_currency[normalized_currency]
            if not by_currency:
                del aggregator._stats[normalized_name]  # type: ignore[attr-defined]


# Convenience for ``uvicorn server:app``
__all__ = ["app", "aggregator", "ItemValueAggregator"]

