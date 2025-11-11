"""FastAPI server for sharing Torch item value observations.

This module exposes a minimal REST API that allows players to submit the
estimated value of their drops and query aggregated statistics that combine the
community's knowledge.  All state is kept in-memory which keeps the
implementation straightforward; for production usage you can swap the
``ItemValueAggregator`` with a database-backed implementation.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Deque, Dict, Iterable, List, Optional, Tuple
from html import escape

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
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

    def snapshot_all(self) -> List[Dict[str, object]]:
        """Return immutable snapshots for all tracked items."""

        with self._lock:
            return [
                stats.snapshot()
                for by_currency in self._stats.values()
                for stats in by_currency.values()
            ]

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


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Render a lightweight HTML dashboard for quick manual inspection."""

    snapshots = aggregator.snapshot_all()
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for snapshot in snapshots:
        grouped[snapshot["item_name"]].append(snapshot)

    # Sort items by total accumulated value (descending) for relevance.
    sorted_items = sorted(
        grouped.items(),
        key=lambda item: sum(stats["total_value"] for stats in item[1]),
        reverse=True,
    )

    rows: List[str] = []
    for item_name, stats_list in sorted_items:
        stats_list.sort(key=lambda stats: stats["currency"].lower())
        for index, stats in enumerate(stats_list):
            last_updated = stats["last_updated"].strftime("%Y-%m-%d %H:%M:%S UTC")
            row_cells = [
                f"<td>{escape(stats['currency'])}</td>",
                f"<td class=\"num\">{stats['submissions']}</td>",
                f"<td class=\"num\">{stats['total_value']:.2f}</td>",
                f"<td class=\"num\">{stats['total_quantity']}</td>",
                f"<td class=\"num\">{stats['average_per_submission']:.2f}</td>",
                f"<td class=\"num\">{stats['average_per_unit']:.2f}</td>",
                f"<td class=\"num\">{stats['min_value']:.2f}</td>",
                f"<td class=\"num\">{stats['max_value']:.2f}</td>",
                f"<td>{escape(last_updated)}</td>",
            ]

            if index == 0:
                row_item = (
                    f"<td rowspan=\"{len(stats_list)}\" class=\"item\">"
                    f"{escape(item_name)}</td>"
                )
            else:
                row_item = ""

            rows.append("<tr>" + row_item + "".join(row_cells) + "</tr>")

    if not rows:
        rows.append(
            "<tr><td colspan=\"10\" class=\"empty\">아직 제출된 아이템이 없습니다. "
            "POST /items 로 데이터를 추가해 주세요.</td></tr>"
        )

    html = f"""
    <!DOCTYPE html>
    <html lang=\"ko\">
      <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <meta http-equiv=\"refresh\" content=\"30\" />
        <title>Torch Item Value Dashboard</title>
        <style>
          body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f1419;
            color: #f2f4f8;
            margin: 0;
            padding: 2rem;
          }}
          h1 {{
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
          }}
          p {{
            margin-top: 0;
            color: #9ba1a6;
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1.5rem;
            background: #141c24;
            border-radius: 12px;
            overflow: hidden;
          }}
          th, td {{
            padding: 0.6rem 0.8rem;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
          }}
          th {{
            background: rgba(255, 255, 255, 0.05);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            color: #aab4be;
          }}
          tr:hover {{
            background: rgba(255, 255, 255, 0.04);
          }}
          td.num {{
            font-variant-numeric: tabular-nums;
            text-align: right;
          }}
          td.item {{
            font-weight: 600;
          }}
          td.empty {{
            text-align: center;
            padding: 1.5rem;
            color: #76808a;
          }}
          footer {{
            margin-top: 2rem;
            font-size: 0.8rem;
            color: #5f6b78;
          }}
          a {{
            color: #64b5f6;
          }}
        </style>
      </head>
      <body>
        <h1>커뮤니티 드랍 가치 현황</h1>
        <p>이 화면은 30초마다 자동으로 새로고침됩니다. REST API는 <code>/docs</code>에서 확인할 수 있습니다.</p>
        <table>
          <thead>
            <tr>
              <th>아이템</th>
              <th>화폐</th>
              <th>제출</th>
              <th>총 가치</th>
              <th>총 수량</th>
              <th>평균(제출)</th>
              <th>평균(개당)</th>
              <th>최소</th>
              <th>최대</th>
              <th>최근 업데이트</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
        <footer>
          Torch Item Value Exchange &mdash; 공유 데이터는 서버가 재시작되면 초기화됩니다.
        </footer>
      </body>
    </html>
    """

    return HTMLResponse(content=html)


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

