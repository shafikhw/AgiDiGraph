"""FastAPI entrypoint for AgiDiGraph."""
from __future__ import annotations

from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query

from app.config import get_settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DisasterDetail,
    DisasterNode,
    DisasterSummary,
    EdgeModel,
    GraphStatus,
    ImpactNode,
    RelatedDisaster,
    DisasterDetailBatchRequest,
    DisasterDetailBatchResponse,
)
from app.services import (
    get_disaster_detail,
    get_disasters_filtered,
    get_related_disasters,
    graph_status,
    load_graph,
)
from app.agent import answer_question

app = FastAPI(title="AgiDiGraph", version="0.1.0")

def _serialize_detail(detail: dict) -> DisasterDetail:
    impacts = [
        ImpactNode(
            id=impact["id"],
            label=impact["label"],
            value=impact.get("value"),
            unit=impact.get("unit"),
            metric=impact.get("metric"),
            metadata=impact.get("metadata", {}),
        )
        for impact in detail.get("impacts", [])
    ]
    edge_models = [EdgeModel(**edge) for edge in detail.get("edges", [])]
    return DisasterDetail(
        disaster=DisasterNode(**detail["disaster"]),
        impacts=impacts,
        edges=edge_models,
    )


def graph_dependency():
    return load_graph()


@app.get("/healthz", response_model=dict)
def healthcheck(settings=Depends(get_settings)):
    return {"status": "ok", "model": settings.openai_model, "offline": settings.use_offline_reasoner}


@app.get("/graph/status", response_model=GraphStatus)
def get_graph_status(graph=Depends(graph_dependency)):
    return GraphStatus(**graph_status(graph))


@app.get("/disasters", response_model=List[DisasterSummary])
def list_disasters(
    year: Optional[int] = Query(None),
    country: Optional[str] = Query(None),
    graph=Depends(graph_dependency),
):
    disasters = get_disasters_filtered(graph, year=year, country=country)
    return [DisasterSummary(**item) for item in disasters]


@app.get("/disasters/{disaster_id}", response_model=DisasterDetail)
def read_disaster(disaster_id: str, graph=Depends(graph_dependency)):
    detail = get_disaster_detail(graph, disaster_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Disaster not found")
    return _serialize_detail(detail)


@app.get("/disasters/{disaster_id}/related", response_model=List[RelatedDisaster])
def read_related(disaster_id: str, graph=Depends(graph_dependency)):
    related = get_related_disasters(graph, disaster_id)
    return [RelatedDisaster(**item) for item in related]


@app.post("/disasters/batch", response_model=DisasterDetailBatchResponse)
def batch_disaster_details(request: DisasterDetailBatchRequest, graph=Depends(graph_dependency)):
    details = {}
    for disaster_id in request.disaster_ids:
        detail = get_disaster_detail(graph, disaster_id)
        if not detail:
            continue
        details[disaster_id] = _serialize_detail(detail)
    return DisasterDetailBatchResponse(details=details)

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, graph=Depends(graph_dependency)):
    answer, sources = answer_question(
        graph,
        request.message,
        year=request.year,
        country=request.country,
    )
    if not sources:
        sources = ['graphs/disaster_impacts.json']
    return ChatResponse(answer=answer, sources=sources)
