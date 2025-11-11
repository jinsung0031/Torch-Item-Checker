"""Tkinter-based desktop interface for Torch Item Checker.

This module adapts the existing CLI timers and aggregation logic into a single
window experience that mirrors the workflow of the community-made
`TLI-Tracker`.  Players can keep the overall session timer running, start and
end map sessions, log loot, and review their history without touching the
terminal.  The window also includes an optional community value search panel
that queries the FastAPI server defined in :mod:`server` so that personal
decisions can reference the shared market data.

Run the application with ``python gui_app.py``.
"""

from __future__ import annotations

import json
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from torch_item_checker import (
    ItemRecord,
    MapSession,
    MapSummary,
    Timer,
    _format_timedelta,
    _per_minute,
)


class TrackerGUI:
    """Main window controller for the desktop tracker."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Torch Item Checker")
        self.root.geometry("1024x720")

        self.overall_timer = Timer()
        self.overall_timer.start()
        self.overall_total = 0.0

        self.current_map: Optional[MapSession] = None
        self.last_map: Optional[MapSummary] = None
        self.map_history: List[MapSummary] = []
        self.item_log: List[ItemRecord] = []

        self.community_base_url = tk.StringVar(value="http://localhost:8000")
        self.community_query = tk.StringVar(value="")

        self._build_layout()
        self._schedule_updates()

    # ------------------------------------------------------------------
    # UI construction helpers

    def _build_layout(self) -> None:
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        header = ttk.Frame(self.root, padding=16)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=1)

        # Overall session summary ------------------------------------------------
        overall_frame = ttk.LabelFrame(header, text="전체 세션")
        overall_frame.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        overall_frame.columnconfigure(1, weight=1)

        ttk.Label(overall_frame, text="전체 시간:").grid(row=0, column=0, sticky="w", pady=2)
        self.lbl_overall_time = ttk.Label(overall_frame, text="00:00:00", font=("Segoe UI", 12, "bold"))
        self.lbl_overall_time.grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(overall_frame, text="전체 파밍 결정:").grid(row=1, column=0, sticky="w", pady=2)
        self.lbl_overall_total = ttk.Label(overall_frame, text="0.00")
        self.lbl_overall_total.grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(overall_frame, text="전체 분당 결정:").grid(row=2, column=0, sticky="w", pady=2)
        self.lbl_overall_per_minute = ttk.Label(overall_frame, text="0.00")
        self.lbl_overall_per_minute.grid(row=2, column=1, sticky="w", pady=2)

        # Current map summary ----------------------------------------------------
        map_frame = ttk.LabelFrame(header, text="현재 맵")
        map_frame.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
        map_frame.columnconfigure(1, weight=1)

        ttk.Label(map_frame, text="맵 이름:").grid(row=0, column=0, sticky="w", pady=2)
        self.lbl_map_name = ttk.Label(map_frame, text="-", font=("Segoe UI", 11, "bold"))
        self.lbl_map_name.grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(map_frame, text="맵 시간:").grid(row=1, column=0, sticky="w", pady=2)
        self.lbl_map_time = ttk.Label(map_frame, text="00:00:00")
        self.lbl_map_time.grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(map_frame, text="맵 파밍 결정:").grid(row=2, column=0, sticky="w", pady=2)
        self.lbl_map_total = ttk.Label(map_frame, text="0.00")
        self.lbl_map_total.grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(map_frame, text="맵 분당 결정:").grid(row=3, column=0, sticky="w", pady=2)
        self.lbl_map_per_minute = ttk.Label(map_frame, text="0.00")
        self.lbl_map_per_minute.grid(row=3, column=1, sticky="w", pady=2)

        ttk.Label(map_frame, text="상태:").grid(row=4, column=0, sticky="w", pady=2)
        self.lbl_map_status = ttk.Label(map_frame, text="대기")
        self.lbl_map_status.grid(row=4, column=1, sticky="w", pady=2)

        # Controls ---------------------------------------------------------------
        controls = ttk.Frame(self.root, padding=(16, 0))
        controls.grid(row=1, column=0, sticky="nsew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(controls)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_panel.rowconfigure(2, weight=1)
        left_panel.columnconfigure(0, weight=1)

        right_panel = ttk.Frame(controls)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        # Map control box --------------------------------------------------------
        map_control = ttk.LabelFrame(left_panel, text="맵 제어", padding=12)
        map_control.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        for col in range(4):
            map_control.columnconfigure(col, weight=1)

        ttk.Label(map_control, text="맵 이름").grid(row=0, column=0, sticky="w")
        self.entry_map_name = ttk.Entry(map_control)
        self.entry_map_name.grid(row=0, column=1, sticky="ew", padx=4)
        self.entry_map_name.insert(0, "Map")

        ttk.Label(map_control, text="시작 비용").grid(row=0, column=2, sticky="w")
        self.entry_map_cost = ttk.Entry(map_control)
        self.entry_map_cost.grid(row=0, column=3, sticky="ew", padx=4)

        self.btn_start_map = ttk.Button(map_control, text="맵 시작", command=self.start_map)
        self.btn_start_map.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.btn_end_map = ttk.Button(map_control, text="맵 종료", command=self.end_map)
        self.btn_end_map.grid(row=1, column=2, columnspan=2, sticky="ew", pady=(8, 0))

        self.btn_reset = ttk.Button(map_control, text="전체 리셋", command=self.reset_session)
        self.btn_reset.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        # Item logging -----------------------------------------------------------
        item_frame = ttk.LabelFrame(left_panel, text="아이템 기록", padding=12)
        item_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        item_frame.columnconfigure(1, weight=1)

        ttk.Label(item_frame, text="설명").grid(row=0, column=0, sticky="w")
        self.entry_item_desc = ttk.Entry(item_frame)
        self.entry_item_desc.grid(row=0, column=1, sticky="ew", padx=4)

        ttk.Label(item_frame, text="가치").grid(row=1, column=0, sticky="w")
        self.entry_item_value = ttk.Entry(item_frame)
        self.entry_item_value.grid(row=1, column=1, sticky="ew", padx=4)

        ttk.Label(item_frame, text="수량").grid(row=2, column=0, sticky="w")
        self.entry_item_quantity = ttk.Entry(item_frame)
        self.entry_item_quantity.grid(row=2, column=1, sticky="ew", padx=4)
        self.entry_item_quantity.insert(0, "1")

        self.btn_add_item = ttk.Button(item_frame, text="기록 추가", command=self.add_item)
        self.btn_add_item.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        # Item log ---------------------------------------------------------------
        log_frame = ttk.LabelFrame(left_panel, text="드랍 로그", padding=12)
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        columns = ("time", "map", "description", "quantity", "value")
        self.log_tree = ttk.Treeview(
            log_frame,
            columns=columns,
            show="headings",
            height=10,
        )
        self.log_tree.heading("time", text="시간")
        self.log_tree.heading("map", text="맵")
        self.log_tree.heading("description", text="설명")
        self.log_tree.heading("quantity", text="수량")
        self.log_tree.heading("value", text="총 가치")
        self.log_tree.column("time", width=90, anchor="center")
        self.log_tree.column("map", width=80, anchor="center")
        self.log_tree.column("description", width=200)
        self.log_tree.column("quantity", width=60, anchor="center")
        self.log_tree.column("value", width=90, anchor="e")
        self.log_tree.grid(row=0, column=0, sticky="nsew")

        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_tree.yview)
        self.log_tree.configure(yscrollcommand=log_scroll.set)
        log_scroll.grid(row=0, column=1, sticky="ns")

        # Map history ------------------------------------------------------------
        history_frame = ttk.LabelFrame(right_panel, text="맵 히스토리", padding=12)
        history_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        history_frame.rowconfigure(0, weight=1)
        history_frame.columnconfigure(0, weight=1)

        history_columns = ("name", "duration", "total", "per_minute", "start")
        self.history_tree = ttk.Treeview(
            history_frame,
            columns=history_columns,
            show="headings",
            height=8,
        )
        self.history_tree.heading("name", text="맵")
        self.history_tree.heading("duration", text="시간")
        self.history_tree.heading("total", text="총 가치")
        self.history_tree.heading("per_minute", text="분당")
        self.history_tree.heading("start", text="시작")
        self.history_tree.column("name", width=120)
        self.history_tree.column("duration", width=90, anchor="center")
        self.history_tree.column("total", width=90, anchor="e")
        self.history_tree.column("per_minute", width=90, anchor="e")
        self.history_tree.column("start", width=130, anchor="center")
        self.history_tree.grid(row=0, column=0, sticky="nsew")

        history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        history_scroll.grid(row=0, column=1, sticky="ns")

        # Community values -------------------------------------------------------
        community_frame = ttk.LabelFrame(right_panel, text="커뮤니티 시세 검색", padding=12)
        community_frame.grid(row=1, column=0, sticky="nsew")
        community_frame.columnconfigure(1, weight=1)
        community_frame.rowconfigure(2, weight=1)

        ttk.Label(community_frame, text="서버 주소").grid(row=0, column=0, sticky="w")
        ttk.Entry(community_frame, textvariable=self.community_base_url).grid(
            row=0, column=1, sticky="ew", padx=4
        )

        ttk.Label(community_frame, text="아이템 검색").grid(row=1, column=0, sticky="w", pady=(8, 0))
        search_row = ttk.Frame(community_frame)
        search_row.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        search_row.columnconfigure(0, weight=1)

        ttk.Entry(search_row, textvariable=self.community_query).grid(row=0, column=0, sticky="ew")
        ttk.Button(search_row, text="검색", command=self.search_community_values).grid(
            row=0, column=1, padx=(6, 0)
        )

        results_columns = ("currency", "average", "per_unit", "min", "max", "count")
        self.community_tree = ttk.Treeview(
            community_frame,
            columns=results_columns,
            show="headings",
            height=6,
        )
        self.community_tree.heading("currency", text="화폐")
        self.community_tree.heading("average", text="평균(제출)")
        self.community_tree.heading("per_unit", text="평균(개당)")
        self.community_tree.heading("min", text="최소")
        self.community_tree.heading("max", text="최대")
        self.community_tree.heading("count", text="제출")
        self.community_tree.column("currency", width=80)
        self.community_tree.column("average", width=90, anchor="e")
        self.community_tree.column("per_unit", width=90, anchor="e")
        self.community_tree.column("min", width=80, anchor="e")
        self.community_tree.column("max", width=80, anchor="e")
        self.community_tree.column("count", width=70, anchor="center")
        self.community_tree.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

        community_scroll = ttk.Scrollbar(community_frame, orient="vertical", command=self.community_tree.yview)
        self.community_tree.configure(yscrollcommand=community_scroll.set)
        community_scroll.grid(row=2, column=2, sticky="ns")

    # ------------------------------------------------------------------
    # Timer updates

    def _schedule_updates(self) -> None:
        self.update_stats()
        self.root.after(500, self._schedule_updates)

    def update_stats(self) -> None:
        overall_duration = self.overall_timer.elapsed()
        self.lbl_overall_time.configure(text=_format_timedelta(overall_duration))
        self.lbl_overall_total.configure(text=f"{self.overall_total:.2f}")
        overall_per_minute = _per_minute(self.overall_total, overall_duration)
        self.lbl_overall_per_minute.configure(text=f"{overall_per_minute:.2f}")

        if self.current_map is not None:
            duration = self.current_map.timer.elapsed()
            per_minute = _per_minute(self.current_map.total_value, duration)
            self.lbl_map_name.configure(text=self.current_map.name)
            self.lbl_map_time.configure(text=_format_timedelta(duration))
            self.lbl_map_total.configure(text=f"{self.current_map.total_value:.2f}")
            self.lbl_map_per_minute.configure(text=f"{per_minute:.2f}")
            status = "진행 중" if self.current_map.timer.is_running() else "일시 정지"
        elif self.last_map is not None:
            duration = self.last_map.duration
            per_minute = _per_minute(self.last_map.total_value, duration)
            self.lbl_map_name.configure(text=self.last_map.name + " (종료)")
            self.lbl_map_time.configure(text=_format_timedelta(duration))
            self.lbl_map_total.configure(text=f"{self.last_map.total_value:.2f}")
            self.lbl_map_per_minute.configure(text=f"{per_minute:.2f}")
            status = "마지막 결과"
        else:
            self.lbl_map_name.configure(text="-")
            self.lbl_map_time.configure(text="00:00:00")
            self.lbl_map_total.configure(text="0.00")
            self.lbl_map_per_minute.configure(text="0.00")
            status = "대기"

        self.lbl_map_status.configure(text=status)

    # ------------------------------------------------------------------
    # Core actions

    def start_map(self) -> None:
        if self.current_map is not None:
            messagebox.showwarning("경고", "이미 맵이 진행 중입니다.")
            return

        name = self.entry_map_name.get().strip() or "Map"
        cost_text = self.entry_map_cost.get().strip()
        cost_value = 0.0
        if cost_text:
            try:
                cost_value = float(cost_text)
            except ValueError:
                messagebox.showerror("오류", "시작 비용은 숫자로 입력하세요.")
                return

        self.current_map = MapSession(name=name)
        self.current_map.start()
        self.last_map = None

        if cost_value:
            cost_record = self.current_map.record_item("시작 비용", -abs(cost_value), 1)
            self.item_log.append(cost_record)
            self.overall_total += cost_record.total_value
            self.refresh_item_log()

        self.lbl_map_status.configure(text="진행 중")
        self.entry_map_name.delete(0, tk.END)
        self.entry_map_name.insert(0, name)
        self.entry_map_cost.delete(0, tk.END)

    def end_map(self) -> None:
        if self.current_map is None:
            messagebox.showinfo("정보", "진행 중인 맵이 없습니다.")
            return

        self.current_map.timer.stop()
        summary = MapSummary(
            name=self.current_map.name,
            duration=self.current_map.timer.elapsed(),
            total_value=self.current_map.total_value,
            items=list(self.current_map.items),
            started_at=self.current_map.started_at,
        )
        self.map_history.append(summary)
        self.last_map = summary

        self.current_map = None
        self.refresh_history()
        self.lbl_map_status.configure(text="대기")

    def add_item(self) -> None:
        if self.current_map is None:
            messagebox.showwarning("경고", "맵이 진행 중이 아닙니다.")
            return

        description = self.entry_item_desc.get().strip()
        value_text = self.entry_item_value.get().strip()
        quantity_text = self.entry_item_quantity.get().strip() or "1"

        try:
            value = float(value_text)
        except ValueError:
            messagebox.showerror("오류", "가치는 숫자로 입력하세요.")
            return

        try:
            quantity = int(quantity_text)
        except ValueError:
            messagebox.showerror("오류", "수량은 정수로 입력하세요.")
            return

        record = self.current_map.record_item(description, value, quantity)
        self.item_log.append(record)
        self.overall_total += record.total_value

        self.entry_item_desc.delete(0, tk.END)
        self.entry_item_value.delete(0, tk.END)
        self.entry_item_quantity.delete(0, tk.END)
        self.entry_item_quantity.insert(0, "1")

        self.refresh_item_log()

    def reset_session(self) -> None:
        if not messagebox.askyesno("확인", "전체 세션을 초기화할까요?"):
            return

        self.overall_timer.reset()
        self.overall_timer.start()
        self.overall_total = 0.0
        self.current_map = None
        self.last_map = None
        self.map_history.clear()
        self.item_log.clear()

        for tree in (self.log_tree, self.history_tree):
            for item in tree.get_children():
                tree.delete(item)

        self.lbl_map_status.configure(text="대기")

    # ------------------------------------------------------------------
    # Table refresh helpers

    def refresh_item_log(self) -> None:
        self.log_tree.delete(*self.log_tree.get_children())
        for record in self.item_log[-200:]:  # Keep the table manageable.
            time_str = record.timestamp.strftime("%H:%M:%S")
            total_value = f"{record.total_value:.2f}"
            self.log_tree.insert(
                "",
                tk.END,
                values=(time_str, record.map_name, record.description or "드랍", record.quantity, total_value),
            )

    def refresh_history(self) -> None:
        self.history_tree.delete(*self.history_tree.get_children())
        for summary in self.map_history[-100:]:
            duration = _format_timedelta(summary.duration)
            per_minute = _per_minute(summary.total_value, summary.duration)
            self.history_tree.insert(
                "",
                tk.END,
                values=(
                    summary.name,
                    duration,
                    f"{summary.total_value:.2f}",
                    f"{per_minute:.2f}",
                    summary.started_at.strftime("%m-%d %H:%M"),
                ),
            )

    # ------------------------------------------------------------------
    # Community search

    def search_community_values(self) -> None:
        query = self.community_query.get().strip()
        if not query:
            messagebox.showwarning("경고", "검색어를 입력하세요.")
            return

        base_url = self.community_base_url.get().rstrip("/")
        params = urlencode({"query": query, "limit": 25})
        url = f"{base_url}/items?{params}"

        def worker() -> None:
            try:
                with urlopen(url, timeout=5) as response:
                    data = json.load(response)
            except URLError as exc:  # pragma: no cover - network errors
                self.root.after(0, lambda: messagebox.showerror("오류", f"서버에 접속할 수 없습니다: {exc}"))
                return
            except Exception as exc:  # pragma: no cover - unexpected errors
                self.root.after(0, lambda: messagebox.showerror("오류", f"응답 파싱 실패: {exc}"))
                return

            self.root.after(0, lambda: self._update_community_table(data))

        threading.Thread(target=worker, daemon=True).start()

    def _update_community_table(self, data: List[dict]) -> None:
        self.community_tree.delete(*self.community_tree.get_children())
        for entry in data:
            self.community_tree.insert(
                "",
                tk.END,
                values=(
                    entry.get("currency", "-"),
                    f"{entry.get('average_per_submission', 0.0):.2f}",
                    f"{entry.get('average_per_unit', 0.0):.2f}",
                    f"{entry.get('min_value', 0.0):.2f}",
                    f"{entry.get('max_value', 0.0):.2f}",
                    entry.get("submissions", 0),
                ),
            )

    # ------------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    gui = TrackerGUI()
    gui.run()


if __name__ == "__main__":
    main()

