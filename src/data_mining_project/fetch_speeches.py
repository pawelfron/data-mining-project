"""Download spoken statements by MPs for one Sejm term."""

from __future__ import annotations

import argparse
import csv
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from data_mining_project.client import SejmClient, configure_logging
from data_mining_project.html_text import html_to_text

logger = logging.getLogger(__name__)

SPEECH_COLUMNS = [
    "term",
    "proceeding",
    "date",
    "statement_num",
    "member_id",
    "speaker_name",
    "text",
]


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    directory = args.output_dir / f"term{args.term}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "speeches.csv"
    done = _load_done(path)
    writer = _SpeechWriter(path)

    try:
        with SejmClient() as client:
            proceedings = client.get_json(f"/term{args.term}/proceedings")
            if not isinstance(proceedings, list):
                raise SystemExit(f"No proceedings found for term {args.term}")
            days = _sitting_days(proceedings)
            if args.max_days is not None:
                days = days[: args.max_days]
            logger.info("Term %s: %s sitting days", args.term, len(days))
            written = _download_days(
                client,
                writer,
                done,
                term=args.term,
                days=days,
                workers=args.workers,
            )
    finally:
        writer.close()

    logger.info("Wrote %s new speeches to %s", written, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download speeches from one Sejm term.",
    )
    parser.add_argument("--term", type=int, required=True, help="Parliament term number")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory that will contain term{N}/speeches.csv",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Download only the first N sitting days",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent statement downloads",
    )
    return parser


def _sitting_days(proceedings: list[dict[str, Any]]) -> list[tuple[int, str]]:
    seen: set[tuple[int, str]] = set()
    days: list[tuple[int, str]] = []
    for proceeding in proceedings:
        number = int(proceeding["number"])
        for date in proceeding.get("dates") or []:
            key = (number, str(date))
            if key in seen:
                continue
            seen.add(key)
            days.append(key)
    return days


def _keep_statement(statement: dict[str, Any]) -> bool:
    member_id = statement.get("memberID") or 0
    number = statement.get("num")
    if not isinstance(member_id, int) or member_id <= 0:
        return False
    if not isinstance(number, int) or number == 0:
        return False
    if statement.get("unspoken"):
        return False
    return True


def _download_days(
    client: SejmClient,
    writer: _SpeechWriter,
    done: set[tuple[str, str, str]],
    *,
    term: int,
    days: list[tuple[int, str]],
    workers: int,
) -> int:
    written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for proceeding, date in days:
            payload = client.get_json(
                f"/term{term}/proceedings/{proceeding}/{date}/transcripts"
            )
            if not isinstance(payload, dict):
                continue
            statements = [
                statement
                for statement in payload.get("statements") or []
                if _keep_statement(statement)
                and (str(proceeding), date, str(statement["num"])) not in done
            ]
            logger.info(
                "Proceeding %s %s: %s statements to download",
                proceeding,
                date,
                len(statements),
            )
            futures = [
                pool.submit(_fetch_statement, client, term, proceeding, date, statement)
                for statement in statements
            ]
            for future in as_completed(futures):
                row = future.result()
                if row is None:
                    continue
                writer.write(row)
                done.add((row["proceeding"], row["date"], row["statement_num"]))
                written += 1
    return written


def _fetch_statement(
    client: SejmClient,
    term: int,
    proceeding: int,
    date: str,
    statement: dict[str, Any],
) -> dict[str, str] | None:
    number = statement["num"]
    path = f"/term{term}/proceedings/{proceeding}/{date}/transcripts/{number}"
    html = client.get_text(path)
    if html is None:
        return None
    text = html_to_text(html)
    if not text:
        logger.info("Empty statement %s %s #%s", proceeding, date, number)
        return None
    return {
        "term": str(term),
        "proceeding": str(proceeding),
        "date": date,
        "statement_num": str(number),
        "member_id": str(statement["memberID"]),
        "speaker_name": statement.get("name") or "",
        "text": text,
    }


def _load_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            done.add((row["proceeding"], row["date"], row["statement_num"]))
    return done


class _SpeechWriter:
    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        has_content = path.exists() and path.stat().st_size > 0
        self._handle = path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=SPEECH_COLUMNS,
            lineterminator="\n",
        )
        if not has_content:
            self._writer.writeheader()
            self._handle.flush()

    def write(self, row: dict[str, str]) -> None:
        with self._lock:
            self._writer.writerow(row)
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


if __name__ == "__main__":
    main()
