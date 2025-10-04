"""Utility helpers for loading and filtering EM-DAT disaster records."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_DATA_GLOB = "public_emdat_custom_request_*.csv"
SAMPLE_FALLBACK = DATA_DIR / "emdat_sample.csv"

TARGET_COUNTRIES = {"Bangladesh", "India", "Nepal"}
TARGET_TYPES = {"Flood", "Storm"}

COLUMN_OVERRIDES = {
    "disno": "disaster_id",
    "iso": "iso3",
    "start_year": "start_year",
    "start_month": "start_month",
    "start_day": "start_day",
    "end_year": "end_year",
    "end_month": "end_month",
    "end_day": "end_day",
    "total_damage_000_us": "total_damage_kusd",
    "total_damage_adjusted_000_us": "total_damage_adjusted_kusd",
    "insured_damage_000_us": "insured_damage_kusd",
    "insured_damage_adjusted_000_us": "insured_damage_adjusted_kusd",
    "reconstruction_costs_000_us": "reconstruction_costs_kusd",
    "reconstruction_costs_adjusted_000_us": "reconstruction_costs_adjusted_kusd",
    "no_injured": "injured",
    "no_affected": "affected",
    "no_homeless": "homeless",
}

NUMERIC_COLUMNS = {
    "total_deaths",
    "injured",
    "affected",
    "homeless",
    "total_affected",
    "total_damage_kusd",
    "total_damage_adjusted_kusd",
    "insured_damage_kusd",
    "insured_damage_adjusted_kusd",
    "reconstruction_costs_kusd",
    "reconstruction_costs_adjusted_kusd",
    "latitude",
    "longitude",
}

DATE_PARTS = ("start_year", "start_month", "start_day", "end_year", "end_month", "end_day")


@dataclass
class DisasterImpactRecord:
    disaster_id: str
    iso3: str
    country: str
    disaster_type: str
    disaster_subtype: Optional[str]
    start_date: Optional[date]
    end_date: Optional[date]
    location: Optional[str]
    origin: Optional[str]
    magnitude: Optional[float]
    magnitude_scale: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    total_deaths: Optional[float]
    injured: Optional[float]
    affected: Optional[float]
    homeless: Optional[float]
    total_affected: Optional[float]
    total_damage_kusd: Optional[float]
    total_damage_adjusted_kusd: Optional[float]
    source_entry_date: Optional[str]
    source_last_update: Optional[str]
    admin_units: Optional[List[Dict[str, Any]]]
    raw_row: Dict[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Convert date objects to ISO strings for JSON serialization.
        for field in ("start_date", "end_date"):
            if payload[field] is not None:
                payload[field] = payload[field].isoformat()
        return payload


def _normalize_column(name: str) -> str:
    """Best-effort conversion of EM-DAT headers to snake_case keys."""
    import re

    cleaned = name.strip()
    cleaned = cleaned.replace("(\"", "(").replace("\")", ")")
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_").lower()


def resolve_data_path(candidate: Optional[Path] = None) -> Path:
    if candidate and candidate.exists():
        return candidate
    matches = sorted(DATA_DIR.glob(DEFAULT_DATA_GLOB))
    if matches:
        return matches[-1]
    return SAMPLE_FALLBACK


def load_raw_records(path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = resolve_data_path(path)
    df = pd.read_csv(csv_path)
    normalized: Dict[str, str] = {}
    for column in df.columns:
        norm = _normalize_column(column)
        normalized[column] = COLUMN_OVERRIDES.get(norm, norm)
    df = df.rename(columns=normalized)
    return df


def filter_target_events(
    df: pd.DataFrame,
    countries: Optional[Sequence[str]] = None,
    disaster_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    countries = set(countries) if countries else TARGET_COUNTRIES
    disaster_types = set(disaster_types) if disaster_types else TARGET_TYPES
    mask = df["country"].isin(countries) & df["disaster_type"].isin(disaster_types)
    filtered = df.loc[mask].copy()
    return filtered


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _compose_date(year: Any, month: Any, day: Any) -> Optional[date]:
    try:
        if pd.isna(year):
            return None
        year = int(year)
        month = int(month) if not pd.isna(month) else 1
        day = int(day) if not pd.isna(day) else 1
        return date(year, month, day)
    except Exception:
        return None


def _parse_admin_units(value: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def build_records(df: pd.DataFrame) -> List[DisasterImpactRecord]:
    records: List[DisasterImpactRecord] = []
    for row in df.to_dict(orient="records"):
        start_date = _compose_date(row.get("start_year"), row.get("start_month"), row.get("start_day"))
        end_date = _compose_date(row.get("end_year"), row.get("end_month"), row.get("end_day"))
        admin_units = _parse_admin_units(row.get("admin_units"))
        record = DisasterImpactRecord(
            disaster_id=row.get("disaster_id") or row.get("disno"),
            iso3=row.get("iso3"),
            country=row.get("country"),
            disaster_type=row.get("disaster_type"),
            disaster_subtype=row.get("disaster_subtype"),
            start_date=start_date,
            end_date=end_date,
            location=row.get("location"),
            origin=row.get("origin"),
            magnitude=row.get("magnitude"),
            magnitude_scale=row.get("magnitude_scale"),
            latitude=row.get("latitude"),
            longitude=row.get("longitude"),
            total_deaths=row.get("total_deaths"),
            injured=row.get("injured"),
            affected=row.get("affected"),
            homeless=row.get("homeless"),
            total_affected=row.get("total_affected"),
            total_damage_kusd=row.get("total_damage_kusd"),
            total_damage_adjusted_kusd=row.get("total_damage_adjusted_kusd"),
            source_entry_date=row.get("entry_date"),
            source_last_update=row.get("last_update"),
            admin_units=admin_units,
            raw_row=row,
        )
        records.append(record)
    return records


def iter_event_batches(
    records: Sequence[DisasterImpactRecord],
    batch_size: int = 10,
) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for record in records:
        batch.append(record.to_payload())
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_filtered_records(
    path: Optional[Path] = None,
    countries: Optional[Sequence[str]] = None,
    disaster_types: Optional[Sequence[str]] = None,
) -> List[DisasterImpactRecord]:
    df = load_raw_records(path)
    df = filter_target_events(df, countries=countries, disaster_types=disaster_types)
    df = coerce_numeric_columns(df)
    return build_records(df)


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Preview EM-DAT disaster records.")
    parser.add_argument("--path", type=Path, default=None, help="Optional explicit CSV path")
    parser.add_argument("--limit", type=int, default=5, help="Number of records to print")
    parser.add_argument("--countries", nargs="*", help="Override target countries")
    parser.add_argument("--types", nargs="*", help="Override disaster types")
    parser.add_argument("--batch-size", type=int, default=5, help="Group records for agent consumption")
    args = parser.parse_args(argv)

    records = load_filtered_records(
        path=args.path,
        countries=args.countries,
        disaster_types=args.types,
    )
    batches = list(iter_event_batches(records, batch_size=args.batch_size))
    limited = batches[0][: args.limit] if batches else []
    print(json.dumps(limited, indent=2))
    print(f"Loaded {len(records)} records across {len(batches)} batches", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
