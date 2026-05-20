from __future__ import annotations

import asyncio
import os
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from daily_email import daily_email


WORK_DIR = Path(__file__).parent
LOGS_DIR = WORK_DIR / "logs" / "daily_logs"
load_dotenv(WORK_DIR / ".env")


def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if value < min_value or value > max_value:
        raise ValueError(f"{name} 必须在 {min_value}-{max_value} 之间")
    return value


def run_hour() -> int:
    return _env_int("DAILY_EMAIL_HOUR", 8, 0, 23)


def run_minute() -> int:
    return _env_int("DAILY_EMAIL_MINUTE", 0, 0, 59)


def check_interval_seconds() -> int:
    return _env_int("DAILY_EMAIL_CHECK_INTERVAL_SECONDS", 30, 5, 3600)


def log_path(now: datetime | None = None) -> Path:
    current = now or datetime.now()
    return LOGS_DIR / f"{current:%Y-%m-%d}.log"


def write_log(message: str, now: datetime | None = None, *, print_to_console: bool = True) -> None:
    current = now or datetime.now()
    line = f"[{current.isoformat(timespec='seconds')}] {message}"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with log_path(current).open("a", encoding="utf-8") as file:
        file.write(line + "\n")

    if print_to_console:
        print(line, flush=True)


def write_exception_log(exc: Exception) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with log_path().open("a", encoding="utf-8") as file:
        file.write("".join(traceback.format_exception(exc)))
        file.write("\n")


def summarize_daily_result(result: dict) -> str:
    parts = [
        f"run_id={result.get('run_id')}",
        f"status={result.get('status')}",
        f"saved={result.get('saved')}",
        f"emailed={result.get('emailed')}",
        f"elapsed_seconds={result.get('elapsed_seconds')}",
    ]
    if result.get("output_file"):
        parts.append(f"output_file={result['output_file']}")
    return " ".join(parts)


async def schedular() -> None:
    last_run_date = None
    hour = run_hour()
    minute = run_minute()
    interval = check_interval_seconds()

    write_log("schedular started")
    write_log(f"daily email time: {hour:02d}:{minute:02d}")
    write_log(f"check interval: {interval}s")

    while True:
        now = datetime.now()

        if now.hour == hour and now.minute == minute and last_run_date != now.date():
            write_log("running daily_email", now)
            try:
                result = await daily_email()
                write_log(f"daily_email success: {summarize_daily_result(result)}")
                last_run_date = now.date()
            except Exception as exc:
                write_log(f"daily_email error: {type(exc).__name__}: {exc}")
                write_exception_log(exc)
                write_log("daily_email failed; schedular will keep running")

        await asyncio.sleep(interval)


if __name__ == "__main__":
    asyncio.run(schedular())
