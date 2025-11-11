"""Interactive tracker for Torch farming sessions.

This script keeps track of the overall farming time/value as well as the
per-map metrics that reset whenever you enter a new map.  It is designed to
support a simple manual workflow:

* Launch the program before you begin farming.  The overall timer starts
  automatically.
* Use the ``enter`` command when you load into a map.  You can supply the
  initial cost (for beacons/compasses) so that the map decision starts as a
  negative value.
* Record loot with ``add`` commands while the map timer is running.
* Use ``exit`` once you leave the map to freeze the timer and log the result.
* ``status`` shows the real-time totals and per-minute value calculations for
  the overall session and the current (or most recent) map.

All state lives in memory and is cleared when you run the ``reset`` command or
exit the tool.  The implementation relies only on the Python standard library
so it can be launched with ``python torch_item_checker.py``.
"""

from __future__ import annotations

import cmd
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional


def _format_timedelta(delta: timedelta) -> str:
    """Render ``delta`` as ``HH:MM:SS`` for display."""

    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _per_minute(value: float, duration: timedelta) -> float:
    """Return the per-minute value earned over ``duration``."""

    minutes = duration.total_seconds() / 60
    return value / minutes if minutes > 0 else 0.0


@dataclass
class Timer:
    """Simple start/stop timer that accumulates elapsed seconds."""

    _start_time: Optional[datetime] = None
    _elapsed: timedelta = field(default_factory=timedelta)

    def start(self) -> None:
        if self._start_time is None:
            self._start_time = datetime.now()

    def stop(self) -> None:
        if self._start_time is not None:
            self._elapsed += datetime.now() - self._start_time
            self._start_time = None

    def reset(self) -> None:
        self._start_time = None
        self._elapsed = timedelta()

    def elapsed(self) -> timedelta:
        if self._start_time is None:
            return self._elapsed
        return self._elapsed + (datetime.now() - self._start_time)

    def is_running(self) -> bool:
        return self._start_time is not None


@dataclass
class ItemRecord:
    """단일 드랍 기록."""

    map_name: str
    description: str
    value: float
    quantity: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_value(self) -> float:
        return self.value * self.quantity

    def to_log_line(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        name = self.description or "드랍"
        quantity = f" x{self.quantity}" if self.quantity != 1 else ""
        return f"[{time_str}] {name}{quantity}: {self.total_value:.2f}"


@dataclass
class MapSession:
    """State for the currently active map."""

    name: str
    timer: Timer = field(default_factory=Timer)
    total_value: float = 0.0
    items: List[ItemRecord] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)

    def start(self) -> None:
        self.timer.reset()
        self.timer.start()
        self.started_at = datetime.now()

    def record_item(self, description: str, value: float, quantity: int = 1) -> ItemRecord:
        record = ItemRecord(
            map_name=self.name,
            description=description,
            value=value,
            quantity=quantity,
        )
        self.items.append(record)
        self.total_value += record.total_value
        return record


@dataclass
class MapSummary:
    name: str
    duration: timedelta
    total_value: float
    items: List[ItemRecord]
    started_at: datetime


class FarmingTracker(cmd.Cmd):
    intro = (
        "Torch Item Checker\n"
        "Type 'help' for available commands, 'status' to view totals, and 'quit' to exit."
    )
    prompt = "torch> "

    def __init__(self) -> None:
        super().__init__()
        self.overall_timer = Timer()
        self.overall_timer.start()
        self.overall_total = 0.0
        self.current_map: Optional[MapSession] = None
        self.last_map: Optional[MapSummary] = None
        self.map_history: List[MapSummary] = []
        self.item_log: List[ItemRecord] = []
        self.export_dir = Path.cwd()

    # ------------------------------------------------------------------
    # Core commands

    def do_status(self, arg: str) -> None:  # noqa: D401 - cmd.Cmd signature
        """Show the timers, totals, and per-minute values."""

        overall_duration = self.overall_timer.elapsed()
        overall_per_minute = _per_minute(self.overall_total, overall_duration)

        print("\n===== Overall Session =====")
        print(f"전체_시간           : {_format_timedelta(overall_duration)}")
        print(f"전체_파밍_결정      : {self.overall_total:.2f}")
        print(f"전체_분당_파밍_결정 : {overall_per_minute:.2f}")

        print("\n===== Current Map =====")
        if self.current_map is not None:
            map_duration = self.current_map.timer.elapsed()
            map_total = self.current_map.total_value
            per_minute = _per_minute(map_total, map_duration)
            running = "(진행 중)" if self.current_map.timer.is_running() else ""
            print(f"맵 이름             : {self.current_map.name} {running}")
            print(f"맵_시간             : {_format_timedelta(map_duration)}")
            print(f"맵_파밍_결정        : {map_total:.2f}")
            print(f"맵_분당_파밍_결정   : {per_minute:.2f}")
            if self.current_map.items:
                print("- 최근 드랍:")
                for record in self.current_map.items[-5:]:
                    print(f"    {record.to_log_line()}")
        elif self.last_map is not None:
            per_minute = _per_minute(self.last_map.total_value, self.last_map.duration)
            print(f"마지막 맵           : {self.last_map.name}")
            print(f"맵_시간             : {_format_timedelta(self.last_map.duration)}")
            print(f"맵_파밍_결정        : {self.last_map.total_value:.2f}")
            print(f"맵_분당_파밍_결정   : {per_minute:.2f}")
            if self.last_map.items:
                print("- 기록된 아이템:")
                for record in self.last_map.items:
                    print(f"    {record.to_log_line()}")
        else:
            print("맵 정보가 없습니다. 'enter'로 맵을 시작하세요.")

        print()

    def do_enter(self, arg: str) -> None:  # noqa: D401 - cmd.Cmd signature
        """Enter a new map.

        Usage: enter [맵이름] [소모비용]

        ``맵이름``은 공백이 없는 문자열이며, 생략하면 ``Map``으로 표시됩니다.
        ``소모비용``은 숫자입니다.  제공하면 맵_파밍_결정과 전체_파밍_결정에
        비용이 음수로 즉시 반영됩니다.
        """

        if self.current_map is not None:
            print("이미 맵이 진행 중입니다. 'exit' 후 다시 시도하세요.")
            return

        parts = arg.split()
        name = parts[0] if parts else "Map"
        cost = 0.0
        if len(parts) >= 2:
            try:
                cost = float(parts[1])
            except ValueError:
                print("소모비용은 숫자여야 합니다.")
                return

        self.current_map = MapSession(name=name)
        self.current_map.start()
        if cost:
            cost_value = -abs(cost)
            cost_record = self.current_map.record_item("시작 비용", cost_value, 1)
            self.overall_total += cost_record.total_value
            self.item_log.append(cost_record)

        print(
            f"'{name}' 맵을 시작했습니다. 초기 비용 {cost:.2f}이 반영되었습니다."
            if cost
            else f"'{name}' 맵을 시작했습니다."
        )

    def do_add(self, arg: str) -> None:  # noqa: D401 - cmd.Cmd signature
        """Add loot that dropped in the current map.

        Usage: add <가치> [아이템_설명] [수량]

        ``가치``는 숫자이며, ``수량``을 지정하면 ``가치 * 수량``이 추가됩니다.
        설명은 로그에 기록되어 ``status``에서 확인할 수 있습니다.
        """

        if self.current_map is None:
            print("맵이 진행 중이 아닙니다. 'enter' 명령으로 맵을 시작하세요.")
            return

        parts = arg.split()
        if not parts:
            print("가치를 입력하세요. 예: add 12.5 희귀아이템 2")
            return

        try:
            value = float(parts[0])
        except ValueError:
            print("가치는 숫자여야 합니다.")
            return

        quantity = 1
        description_parts = parts[1:]
        if description_parts and description_parts[-1].isdigit():
            quantity = int(description_parts.pop())

        description = " ".join(description_parts).strip()
        record = self.current_map.record_item(description, value, quantity)
        self.item_log.append(record)
        self.overall_total += record.total_value

        print(
            f"아이템 기록 완료: {description or '드랍'} x{quantity} (총 {record.total_value:.2f})."
        )

    def do_exit(self, arg: str) -> None:  # noqa: D401 - cmd.Cmd signature
        """Exit the current map and store its summary."""

        if self.current_map is None:
            print("진행 중인 맵이 없습니다.")
            return

        self.current_map.timer.stop()
        summary = MapSummary(
            name=self.current_map.name,
            duration=self.current_map.timer.elapsed(),
            total_value=self.current_map.total_value,
            items=list(self.current_map.items),
            started_at=self.current_map.started_at,
        )
        self.last_map = summary
        self.map_history.append(summary)
        print(
            "\n맵 종료" 
            f"\n- 이름   : {summary.name}"
            f"\n- 시간   : {_format_timedelta(summary.duration)}"
            f"\n- 가치   : {summary.total_value:.2f}"
            f"\n- 분당   : {_per_minute(summary.total_value, summary.duration):.2f}"
        )
        if summary.items:
            print("- 로그   :")
            for record in summary.items:
                print(f"    {record.to_log_line()}")

        self.current_map = None

    def do_reset(self, arg: str) -> None:  # noqa: D401 - cmd.Cmd signature
        """Reset the entire session including timers and totals."""

        self.overall_timer.reset()
        self.overall_timer.start()
        self.overall_total = 0.0
        self.current_map = None
        self.last_map = None
        self.map_history.clear()
        self.item_log.clear()
        print("전체 세션이 초기화되었습니다.")

    # ------------------------------------------------------------------
    # 추가 기능

    def do_history(self, arg: str) -> None:  # noqa: D401 - cmd.Cmd signature
        """이전 맵 기록을 요약해 보여줍니다.

        Usage: history [개수]

        ``개수``를 지정하면 최근 N개의 맵만 출력합니다.
        """

        if not self.map_history:
            print("저장된 맵 기록이 없습니다.")
            return

        limit = None
        if arg.strip():
            try:
                limit = int(arg.strip())
            except ValueError:
                print("개수는 정수여야 합니다.")
                return

        records = self.map_history[-limit:] if limit is not None else self.map_history
        start_index = len(self.map_history) - len(records) + 1
        print("\n===== 맵 기록 =====")
        header = "번호 | 시작 시각 | 맵 이름 | 진행 시간 | 총 가치 | 분당 가치"
        print(header)
        print("-" * len(header))
        for offset, summary in enumerate(records):
            per_minute = _per_minute(summary.total_value, summary.duration)
            print(
                f"{start_index + offset:>3} | {summary.started_at.strftime('%m-%d %H:%M')} | "
                f"{summary.name:<10} | {_format_timedelta(summary.duration)} | "
                f"{summary.total_value:>8.2f} | {per_minute:>8.2f}"
            )
        print()

    def do_filter(self, arg: str) -> None:  # noqa: D401 - cmd.Cmd signature
        """최근 N분 동안의 드랍 합계를 확인합니다.

        Usage: filter <분> [맵이름|전체]

        ``맵이름``을 입력하면 해당 맵만 필터링합니다. ``전체``는 전체 세션입니다.
        """

        parts = arg.split()
        if not parts:
            print("필터링할 시간을 분 단위로 입력하세요. 예: filter 15 전체")
            return

        try:
            minutes = float(parts[0])
        except ValueError:
            print("시간은 숫자여야 합니다.")
            return

        scope = parts[1] if len(parts) >= 2 else "전체"
        cutoff = datetime.now() - timedelta(minutes=minutes)

        if scope == "전체":
            records: Iterable[ItemRecord] = self.item_log
        else:
            records = [r for r in self.item_log if r.map_name == scope]

        filtered = [r for r in records if r.timestamp >= cutoff]

        if not filtered:
            print("해당 조건에 맞는 드랍 기록이 없습니다.")
            return

        total_value = sum(r.total_value for r in filtered)
        print(
            f"\n최근 {minutes:.1f}분 동안의 '{scope}' 기록 요약"
            f"\n- 드랍 수 : {len(filtered)}"
            f"\n- 총 가치 : {total_value:.2f}"
        )
        print("- 상세 로그:")
        for record in filtered:
            print(f"    [{record.map_name}] {record.to_log_line()}")
        print()

    def do_export(self, arg: str) -> None:  # noqa: D401 - cmd.Cmd signature
        """현재까지의 맵 요약을 CSV 파일로 저장합니다.

        Usage: export [파일이름]

        기본 파일명은 ``tli_tracker_export.csv``이며, 실행 디렉터리에 생성됩니다.
        """

        if not self.map_history:
            print("내보낼 맵 기록이 없습니다.")
            return

        filename = arg.strip() or "tli_tracker_export.csv"
        target = self.export_dir / filename

        with target.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "순번",
                    "맵 이름",
                    "시작 시각",
                    "진행 시간(초)",
                    "총 가치",
                    "분당 가치",
                    "기록된 아이템 수",
                ]
            )

            for idx, summary in enumerate(self.map_history, start=1):
                duration_seconds = int(summary.duration.total_seconds())
                per_minute = _per_minute(summary.total_value, summary.duration)
                writer.writerow(
                    [
                        idx,
                        summary.name,
                        summary.started_at.isoformat(),
                        duration_seconds,
                        f"{summary.total_value:.2f}",
                        f"{per_minute:.2f}",
                        len(summary.items),
                    ]
                )

        print(f"CSV 파일이 저장되었습니다: {target}")

    def do_quit(self, arg: str) -> bool:  # noqa: D401 - cmd.Cmd signature
        """Exit the tracker."""

        print("Torch Item Checker를 종료합니다.")
        return True

    def do_EOF(self, arg: str) -> bool:  # noqa: N802 - method name required by cmd.Cmd
        print()
        return self.do_quit(arg)

    # ------------------------------------------------------------------
    # Helpers & niceties

    def emptyline(self) -> None:
        """Avoid repeating the last command when the user presses Enter."""

        # By default, ``cmd.Cmd`` repeats the previous command when the user
        # hits Enter on an empty line.  That behaviour is confusing in this
        # context, so we simply override the method with a no-op.
        return


def main() -> None:
    FarmingTracker().cmdloop()


if __name__ == "__main__":
    main()
