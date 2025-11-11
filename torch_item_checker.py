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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional


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
class MapSession:
    """State for the currently active map."""

    name: str
    timer: Timer = field(default_factory=Timer)
    total_value: float = 0.0
    items: List[str] = field(default_factory=list)

    def start(self) -> None:
        self.timer.reset()
        self.timer.start()

    def record_item(self, description: str, value: float) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.items.append(f"[{timestamp}] {description}: {value:.2f}")
        self.total_value += value


@dataclass
class MapSummary:
    name: str
    duration: timedelta
    total_value: float
    items: List[str]


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
        elif self.last_map is not None:
            per_minute = _per_minute(self.last_map.total_value, self.last_map.duration)
            print(f"마지막 맵           : {self.last_map.name}")
            print(f"맵_시간             : {_format_timedelta(self.last_map.duration)}")
            print(f"맵_파밍_결정        : {self.last_map.total_value:.2f}")
            print(f"맵_분당_파밍_결정   : {per_minute:.2f}")
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
            self.current_map.total_value += cost_value
            self.overall_total += cost_value
            self.current_map.items.append(f"시작 비용: {cost_value:.2f}")

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

        description = " ".join(description_parts) if description_parts else "드랍".strip()
        total_value = value * quantity

        self.current_map.record_item(description, total_value)
        self.overall_total += total_value

        print(
            f"아이템 기록 완료: {description or '드랍'} x{quantity} (총 {total_value:.2f})."
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
        )
        self.last_map = summary
        print(
            "\n맵 종료" 
            f"\n- 이름   : {summary.name}"
            f"\n- 시간   : {_format_timedelta(summary.duration)}"
            f"\n- 가치   : {summary.total_value:.2f}"
            f"\n- 분당   : {_per_minute(summary.total_value, summary.duration):.2f}"
        )
        if summary.items:
            print("- 로그   :")
            for line in summary.items:
                print(f"    {line}")

        self.current_map = None

    def do_reset(self, arg: str) -> None:  # noqa: D401 - cmd.Cmd signature
        """Reset the entire session including timers and totals."""

        self.overall_timer.reset()
        self.overall_timer.start()
        self.overall_total = 0.0
        self.current_map = None
        self.last_map = None
        print("전체 세션이 초기화되었습니다.")

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
