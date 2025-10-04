"""Disaster-impact graph builder."""
from __future__ import annotations

import hashlib
import json
import math
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.config import get_settings

from app.graph_schema import EdgeType, Evidence, GraphEdge, GraphNode, NodeType
from scripts.data_loader import load_filtered_records, resolve_data_path

LOGGER = logging.getLogger("graph_builder")


SETTINGS = get_settings()
CACHE_ENABLED = SETTINGS.cache_enabled

CACHE_DIR = Path("graphs")
CACHE_MANIFEST = CACHE_DIR / "cache_manifest.json"

def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> Dict[str, Any]:
    if not CACHE_MANIFEST.exists():
        return {"entries": {}}
    try:
        return json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"entries": {}}


def _save_manifest(manifest: Dict[str, Any]) -> None:
    CACHE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    CACHE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
SEVERITY_WEIGHTS: Dict[str, float] = {
    "total_deaths": 3.0,
    "injured": 1.5,
    "affected": 0.8,
    "homeless": 1.2,
    "total_damage_kusd": 0.002,
    "total_damage_adjusted_kusd": 0.002,
}

IMPACT_FIELDS: Dict[str, Dict[str, str]] = {
    "total_deaths": {"label": "Fatalities", "unit": "people"},
    "injured": {"label": "Injured", "unit": "people"},
    "affected": {"label": "Affected", "unit": "people"},
    "homeless": {"label": "Homeless", "unit": "people"},
    "total_damage_kusd": {"label": "Economic Damage", "unit": "thousand_usd"},
    "total_damage_adjusted_kusd": {
        "label": "Economic Damage (adjusted)",
        "unit": "thousand_usd",
    },
}

TEMPORAL_WINDOW_DAYS = 75
CROSS_COUNTRY_WINDOW_DAYS = 3


@dataclass
class DisasterSummary:
    disaster_id: str
    country: Optional[str]
    disaster_type: Optional[str]
    start: Optional[datetime]
    end: Optional[datetime]
    severity: float


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        result = float(value)
        if not math.isfinite(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _severity_level(score: float) -> str:
    if score >= 75:
        return "extreme"
    if score >= 40:
        return "high"
    if score >= 20:
        return "moderate"
    return "low"


def _severity_score(payload: Dict[str, Any]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for metric, weight in SEVERITY_WEIGHTS.items():
        value = _safe_float(payload.get(metric))
        totals[metric] = (value or 0.0) * weight
    aggregate = sum(totals.values())
    if not math.isfinite(aggregate):
        aggregate = 0.0
    totals["aggregate"] = aggregate
    return totals


def _confidence_from_severity(score: float, base: float = 0.55, scale: float = 120.0) -> float:
    if score <= 0:
        return max(0.45, base - 0.1)
    return min(base + score / scale, 0.95)


def _impact_confidence(value: Optional[float]) -> float:
    if value is None:
        return 0.5
    if not math.isfinite(value):
        return 0.5
    return min(0.5 + min(abs(value), 1_000_000) / 5_000_000, 0.92)


class GraphBuilder:
    def __init__(self, records: Iterable[Dict[str, Any]]) -> None:
        self.records = list(records)
        self.graph = nx.MultiDiGraph()
        self.index: List[DisasterSummary] = []

    def build(self) -> nx.MultiDiGraph:
        LOGGER.info("Building graph for %s disasters", len(self.records))
        for payload in self.records:
            self._add_disaster(payload)
        self._link_related_disasters()
        return self.graph

    def _add_disaster(self, payload: Dict[str, Any]) -> None:
        disaster_id = payload.get("disaster_id") or str(uuid4())
        country = payload.get("country")
        disaster_type = payload.get("disaster_type")
        disaster_subtype = payload.get("disaster_subtype")
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")

        severity_totals = _severity_score(payload)
        severity_score = severity_totals["aggregate"]
        severity_level = _severity_level(severity_score)
        confidence = _confidence_from_severity(severity_score)

        attributes = {
            "country": country,
            "disaster_type": disaster_type,
            "disaster_subtype": disaster_subtype,
            "start_date": start_date,
            "end_date": end_date,
            "magnitude": _safe_float(payload.get("magnitude")),
            "magnitude_scale": payload.get("magnitude_scale"),
            "location": payload.get("location"),
            "severity_score": round(severity_score, 2),
            "severity_level": severity_level,
            "people_affected": payload.get("affected"),
            "people_homeless": payload.get("homeless"),
            "damage_kusd": payload.get("total_damage_kusd"),
            "damage_adjusted_kusd": payload.get("total_damage_adjusted_kusd"),
        }

        summary = " ".join(
            part
            for part in [
                disaster_type or "Disaster",
                f"in {country}" if country else None,
                f"starting {start_date}" if start_date else None,
                f"severity {severity_level} ({severity_score:.1f})",
            ]
            if part
        )
        evidence = Evidence(
            source="EM-DAT",
            source_ref=disaster_id,
            summary=summary,
            confidence=round(confidence, 2),
        )

        node = GraphNode(
            id=disaster_id,
            type=NodeType.disaster,
            label=f"{disaster_type or 'Event'} – {country}".strip(),
            attributes=attributes,
            evidence=evidence,
        )
        self.graph.add_node(
            node.id,
            type=node.type.value,
            label=node.label,
            attributes=node.attributes,
            evidence=node.evidence.dict(),
        )

        self._add_impacts(disaster_id, payload, severity_totals)
        self.index.append(
            DisasterSummary(
                disaster_id=disaster_id,
                country=country,
                disaster_type=disaster_type,
                start=_parse_date(start_date),
                end=_parse_date(end_date),
                severity=severity_score,
            )
        )

    def _add_impacts(self, disaster_id: str, payload: Dict[str, Any], totals: Dict[str, float]) -> None:
        for metric, meta in IMPACT_FIELDS.items():
            raw_value = _safe_float(payload.get(metric))
            if raw_value is None or raw_value == 0:
                continue
            severity_component = totals.get(metric, 0.0)
            label = meta["label"]
            unit = meta["unit"]

            evidence = Evidence(
                source="EM-DAT",
                source_ref=disaster_id,
                summary=f"{label} recorded as {raw_value} {unit}.",
                confidence=round(_impact_confidence(raw_value), 2),
            )

            node = GraphNode(
                id=f"{disaster_id}:{metric}",
                type=NodeType.impact,
                label=label,
                attributes={
                    "metric": metric,
                    "value": raw_value,
                    "unit": unit,
                    "severity_score": round(severity_component, 2),
                },
                evidence=evidence,
            )
            self.graph.add_node(
                node.id,
                type=node.type.value,
                label=node.label,
                attributes=node.attributes,
                evidence=node.evidence.dict(),
            )

            edge = GraphEdge(
                type=EdgeType.causes,
                source=disaster_id,
                target=node.id,
                attributes={
                    "metric": metric,
                    "value": raw_value,
                    "unit": unit,
                    "severity_score": round(severity_component, 2),
                },
                evidence=evidence,
            )
            self.graph.add_edge(
                edge.source,
                edge.target,
                key=f"{edge.source}->{edge.target}:{metric}",
                type=edge.type.value,
                attributes=edge.attributes,
                evidence=edge.evidence.dict(),
            )

    def _link_related_disasters(self) -> None:
        if len(self.index) < 2:
            return
        for left, right in combinations(self.index, 2):
            if not left.start or not right.start:
                continue
            days_apart = abs((left.start - right.start).days)
            reason = None
            if (
                left.country
                and right.country
                and left.country == right.country
                and days_apart <= TEMPORAL_WINDOW_DAYS
            ):
                reason = f"Temporal cluster in {left.country} within {days_apart} days."
            elif (
                left.disaster_type
                and right.disaster_type
                and left.disaster_type == right.disaster_type
                and left.country
                and right.country
                and left.country != right.country
                and days_apart <= CROSS_COUNTRY_WINDOW_DAYS
            ):
                reason = (
                    f"Concurrent {left.disaster_type} across {left.country} and {right.country}"
                )
            if not reason:
                continue

            source, target = (left, right)
            if left.start and right.start and left.start > right.start:
                source, target = right, left

            combined_severity = round(source.severity + target.severity, 2)
            confidence = min(0.45 + combined_severity / 200, 0.9)
            evidence = Evidence(
                source="EM-DAT",
                source_ref=source.disaster_id,
                summary=reason,
                confidence=round(confidence, 2),
            )
            edge = GraphEdge(
                type=EdgeType.related,
                source=source.disaster_id,
                target=target.disaster_id,
                attributes={
                    "reason": reason,
                    "days_apart": days_apart,
                    "combined_severity": combined_severity,
                },
                evidence=evidence,
            )
            key = f"{edge.source}->{edge.target}:related:{uuid4()}"
            self.graph.add_edge(
                edge.source,
                edge.target,
                key=key,
                type=edge.type.value,
                attributes=edge.attributes,
                evidence=edge.evidence.dict(),
            )


class GraphSerializer:
    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    def export(self, output_dir: Path) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        graphml_path = output_dir / "disaster_impacts.graphml"
        json_path = output_dir / "disaster_impacts.json"
        nx.write_graphml(self._graphml_ready(), graphml_path)
        json_path.write_text(self._to_json(), encoding="utf-8")
        return {"graphml": graphml_path, "json": json_path}

    def _graphml_ready(self) -> nx.MultiDiGraph:
        safe = nx.MultiDiGraph()
        for node_id, data in self.graph.nodes(data=True):
            safe.add_node(
                node_id,
                **{key: self._safe_value(value) for key, value in data.items()},
            )
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            safe.add_edge(
                source,
                target,
                key=key,
                **{key_: self._safe_value(value) for key_, value in data.items()},
            )
        return safe

    def _to_json(self) -> str:
        return json.dumps(
            {
                "nodes": [
                    {"id": node, **data} for node, data in self.graph.nodes(data=True)
                ],
                "edges": [
                    {"source": u, "target": v, "key": k, **data}
                    for u, v, k, data in self.graph.edges(data=True, keys=True)
                ],
            },
            indent=2,
        )

    @staticmethod
    def _safe_value(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return value


def build_and_export(path: Optional[Path] = None) -> Dict[str, Path]:
    data_file = resolve_data_path(path)

    data_hash: Optional[str] = None
    manifest: Dict[str, Any] = {}
    if CACHE_ENABLED:
        data_hash = _hash_file(data_file)
        manifest = _load_manifest()
        cached_entry = manifest.get("entries", {}).get(data_hash)
        if cached_entry:
            graphml_path = Path(cached_entry.get("graphml", ""))
            json_path = Path(cached_entry.get("json", ""))
            if graphml_path.exists() and json_path.exists():
                LOGGER.info("Using cached graph built from %s", data_file)
                return {"graphml": graphml_path, "json": json_path}

    records = [record.to_payload() for record in load_filtered_records(path=data_file)]

    if not CACHE_ENABLED and SETTINGS.max_batches > 0:
        limit = SETTINGS.max_batches * max(1, SETTINGS.batch_size)
        records = records[:limit]
        LOGGER.info("Caching disabled; limiting to %s records", len(records))

    builder = GraphBuilder(records)
    graph = builder.build()
    LOGGER.info(
        "Constructed graph with %s nodes and %s edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    serializer = GraphSerializer(graph)
    outputs = serializer.export(CACHE_DIR)

    if CACHE_ENABLED and data_hash is not None:
        manifest.setdefault("entries", {})[data_hash] = {
            "graphml": str(outputs["graphml"]),
            "json": str(outputs["json"]),
            "data_file": str(data_file),
        }
        _save_manifest(manifest)

    return outputs
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    data_path_env = os.getenv("EMDAT_DATA_PATH")
    data_path = Path(data_path_env) if data_path_env else None
    outputs = build_and_export(data_path)
    for artifact, path in outputs.items():
        LOGGER.info("Exported %s to %s", artifact, path)


if __name__ == "__main__":
    main()
