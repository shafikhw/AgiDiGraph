"""Graph access utilities for the API."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from app.graph_schema import EdgeType
from scripts.build_graph import build_and_export

GRAPH_JSON_PATH = Path("graphs/disaster_impacts.json")


def _ensure_graph_exists(target: Path) -> None:
    if target.exists():
        return
    build_and_export()


@lru_cache()
def load_graph(path: Optional[Path] = None) -> nx.MultiDiGraph:
    target = path or GRAPH_JSON_PATH
    if path is None:
        _ensure_graph_exists(target)
    if not target.exists():
        raise FileNotFoundError(f"Graph artifact not found at {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    graph = nx.MultiDiGraph()
    for node in data.get("nodes", []):
        node_id = node.get("id")
        if node_id is None:
            continue
        stored = {k: v for k, v in node.items() if k != "id"}
        graph.add_node(node_id, **stored)
    for edge in data.get("edges", []):
        source = edge.get("source")
        target_id = edge.get("target")
        if source is None or target_id is None:
            continue
        key = edge.get("key") or f"{source}->{target_id}:{len(graph.edges)}"
        stored = {k: v for k, v in edge.items() if k not in {"source", "target", "key"}}
        graph.add_edge(source, target_id, key=key, **stored)
    return graph


def iter_disaster_nodes(graph: nx.MultiDiGraph) -> Iterable[Tuple[str, Dict]]:
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "disaster":
            yield node_id, data


def get_disaster_detail(graph: nx.MultiDiGraph, disaster_id: str) -> Optional[Dict]:
    if disaster_id not in graph:
        return None
    node_data = graph.nodes[disaster_id]
    if node_data.get("type") != "disaster":
        return None
    impacts: List[Dict] = []
    edges: List[Dict] = []
    for _, target, edge_key, data in graph.out_edges(disaster_id, keys=True, data=True):
        if graph.nodes[target].get("type") == "impact":
            impacts.append(
                {
                    "id": target,
                    "label": graph.nodes[target].get("label"),
                    "value": graph.nodes[target].get("attributes", {}).get("value"),
                    "unit": graph.nodes[target].get("attributes", {}).get("unit"),
                    "metric": data.get("attributes", {}).get("metric"),
                    "metadata": graph.nodes[target],
                }
            )
        edges.append(
            {
                "source": disaster_id,
                "target": target,
                "type": data.get("type"),
                "evidence": data.get("evidence", {}),
                "attributes": data.get("attributes", {}),
            }
        )
    return {
        "disaster": {
            "id": disaster_id,
            "label": node_data.get("label"),
            "country": node_data.get("attributes", {}).get("country"),
            "disaster_type": node_data.get("attributes", {}).get("disaster_type"),
            "disaster_subtype": node_data.get("attributes", {}).get("disaster_subtype"),
            "start_date": node_data.get("attributes", {}).get("start_date"),
            "end_date": node_data.get("attributes", {}).get("end_date"),
            "metadata": node_data,
        },
        "impacts": impacts,
        "edges": edges,
    }


def get_disasters_filtered(
    graph: nx.MultiDiGraph,
    year: Optional[int] = None,
    country: Optional[str] = None,
) -> List[Dict]:
    results: List[Dict] = []
    for node_id, data in iter_disaster_nodes(graph):
        attrs = data.get("attributes", {})
        start_date = attrs.get("start_date")
        disaster_year = int(start_date.split("-")[0]) if start_date else None
        if year and disaster_year != year:
            continue
        if country and (attrs.get("country") or "").lower() != country.lower():
            continue
        total_impacts = sum(
            1
            for _, _, _, edge_data in graph.out_edges(node_id, keys=True, data=True)
            if edge_data.get("type") == EdgeType.causes.value
        )
        results.append(
            {
                "id": node_id,
                "label": data.get("label"),
                "country": attrs.get("country"),
                "disaster_type": attrs.get("disaster_type"),
                "start_date": attrs.get("start_date"),
                "severity_score": attrs.get("severity_score"),
                "severity_level": attrs.get("severity_level"),
                "total_impacts": total_impacts,
            }
        )
    return results


def get_related_disasters(graph: nx.MultiDiGraph, disaster_id: str) -> List[Dict]:
    related: List[Dict] = []
    if disaster_id not in graph:
        return related
    for source, target, key, data in graph.edges(disaster_id, keys=True, data=True):
        if data.get("type") != EdgeType.related.value:
            continue
        other_id = target if source == disaster_id else source
        if other_id not in graph:
            continue
        related.append(
            {
                "id": other_id,
                "label": graph.nodes[other_id].get("label"),
                "relation": data.get("type"),
            }
        )
    return related


def graph_status(graph: nx.MultiDiGraph) -> Dict[str, int]:
    return {"node_count": graph.number_of_nodes(), "edge_count": graph.number_of_edges()}
