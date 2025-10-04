import json
from pathlib import Path

import networkx as nx

from app import services


def test_load_graph(tmp_path):
    services.load_graph.cache_clear()
    sample = {
        "nodes": [
            {
                "id": "D1",
                "type": "disaster",
                "label": "Event",
                "attributes": {
                    "country": "India",
                    "start_date": "2020-01-01",
                    "disaster_type": "Storm",
                },
            },
            {
                "id": "D1:total_deaths",
                "type": "impact",
                "label": "Total Deaths",
                "attributes": {"value": 10, "unit": "people"},
            },
        ],
        "edges": [
            {
                "source": "D1",
                "target": "D1:total_deaths",
                "type": "causes",
                "evidence": {"source": "EM-DAT"},
                "attributes": {"metric": "total_deaths"},
            }
        ],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(sample))
    graph = services.load_graph(path)
    assert isinstance(graph, nx.MultiDiGraph)
    detail = services.get_disaster_detail(graph, "D1")
    assert detail is not None
    assert detail["disaster"]["country"] == "India"
    assert detail["impacts"][0]["value"] == 10


def test_geospatial_filters(tmp_path):
    services.load_graph.cache_clear()
    sample = {
        "nodes": [
            {
                "id": "D1",
                "type": "disaster",
                "label": "Event",
                "attributes": {
                    "country": "India",
                    "start_date": "2020-01-01",
                    "disaster_type": "Storm",
                },
            },
            {
                "id": "D2",
                "type": "disaster",
                "label": "Second",
                "attributes": {
                    "country": "Bangladesh",
                    "start_date": "2019-06-01",
                    "disaster_type": "Flood",
                },
            },
        ],
        "edges": [],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(sample))
    graph = services.load_graph(path)
    filtered = services.get_disasters_filtered(graph, year=2020, country="india")
    assert len(filtered) == 1
    assert filtered[0]["id"] == "D1"
