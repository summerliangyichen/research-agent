from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from daily_email import daily_email


WORK_DIR = Path(__file__).parent
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


async def schedular() -> None:
    last_run_date = None
    hour = run_hour()
    minute = run_minute()
    interval = check_interval_seconds()

    print("schedular started", flush=True)
    print(f"daily email time: {hour:02d}:{minute:02d}", flush=True)
    print(f"check interval: {interval}s", flush=True)

    while True:
        now = datetime.now()

        if now.hour == hour and now.minute == minute and last_run_date != now.date():
            print(f"[{now.isoformat(timespec='seconds')}] running daily_email", flush=True)
            try:
                result = await daily_email()
                print(f"[{datetime.now().isoformat(timespec='seconds')}] daily_email success: {result}", flush=True)
                last_run_date = now.date()
            except Exception as exc:
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] daily_email error: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        await asyncio.sleep(interval)


if __name__ == "__main__":
    asyncio.run(schedular())
