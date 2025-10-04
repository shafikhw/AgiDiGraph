import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import services
from app.main import app


def _seed_graph(tmp_path: Path):
    payload = {
        "nodes": [
            {
                "id": "D1",
                "type": "disaster",
                "label": "Cyclone Demo",
                "attributes": {
                    "country": "Bangladesh",
                    "start_date": "2020-05-01",
                    "disaster_type": "Storm",
                },
            },
            {
                "id": "D1:total_deaths",
                "type": "impact",
                "label": "Total Deaths",
                "attributes": {"value": 20, "unit": "people"},
            },
        ],
        "edges": [
            {
                "source": "D1",
                "target": "D1:total_deaths",
                "type": "causes",
                "evidence": {"source": "EM-DAT", "source_ref": "D1"},
                "attributes": {"metric": "total_deaths"},
            }
        ],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload))
    services.load_graph.cache_clear()
    services.GRAPH_JSON_PATH = path


def test_disaster_listing(tmp_path):
    _seed_graph(tmp_path)
    client = TestClient(app)
    response = client.get("/disasters", params={"year": 2020, "country": "Bangladesh"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == "D1"


def test_disaster_detail(tmp_path):
    _seed_graph(tmp_path)
    client = TestClient(app)
    response = client.get("/disasters/D1")
    assert response.status_code == 200
    body = response.json()
    assert body["disaster"]["country"] == "Bangladesh"
    assert body["impacts"][0]["value"] == 20
