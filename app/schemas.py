"""API schemas for AgiDiGraph service."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.graph_schema import EdgeType, NodeType


class DisasterNode(BaseModel):
    id: str
    label: str
    country: Optional[str]
    disaster_type: Optional[str]
    disaster_subtype: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImpactNode(BaseModel):
    id: str
    label: str
    value: Optional[float]
    unit: Optional[str]
    metric: Optional[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EdgeModel(BaseModel):
    source: str
    target: str
    type: EdgeType
    evidence: Dict[str, Any]
    attributes: Dict[str, Any] = Field(default_factory=dict)


class DisasterDetail(BaseModel):
    disaster: DisasterNode
    impacts: List[ImpactNode]
    edges: List[EdgeModel]


class DisasterDetailBatchRequest(BaseModel):
    disaster_ids: List[str]


class DisasterDetailBatchResponse(BaseModel):
    details: Dict[str, DisasterDetail]


class DisasterSummary(BaseModel):
    id: str
    label: str
    country: Optional[str]
    disaster_type: Optional[str]
    start_date: Optional[str]
    severity_score: Optional[float]
    severity_level: Optional[str]
    total_impacts: int


class RelatedDisaster(BaseModel):
    id: str
    label: str
    relation: EdgeType


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    year: Optional[int] = None
    country: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)


class GraphStatus(BaseModel):
    node_count: int
    edge_count: int


__all__ = [
    "DisasterNode",
    "ImpactNode",
    "EdgeModel",
    "DisasterDetail",
    "DisasterDetailBatchRequest",
    "DisasterDetailBatchResponse",
    "DisasterSummary",
    "RelatedDisaster",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "GraphStatus",
]
