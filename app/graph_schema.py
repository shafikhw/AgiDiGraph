"""Shared graph data contracts for nodes, edges, and agent payloads."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, validator


class NodeType(str, Enum):
    disaster = "disaster"
    impact = "impact"
    climate_event = "climate_event"


class EdgeType(str, Enum):
    causes = "causes"
    related = "related"
    triggered_by = "triggered_by"


class Evidence(BaseModel):
    source: str = Field(..., description="Data source such as EM-DAT")
    source_ref: str = Field(..., description="Row identifier or citation")
    summary: str = Field(..., description="LLM-generated justification")
    confidence: float = Field(..., ge=0.0, le=1.0)
    url: Optional[HttpUrl] = Field(None, description="Optional external reference link")


class GraphNode(BaseModel):
    id: str = Field(..., description="Stable node identifier")
    type: NodeType
    label: str = Field(..., description="Human-readable node label")
    attributes: Dict[str, Any] = Field(default_factory=dict)
    evidence: Evidence

    @validator("attributes", pre=True, always=True)
    def ensure_attributes_dict(cls, value: Dict[str, Any]) -> Dict[str, Any]:  # noqa: N805
        return value or {}


class GraphEdge(BaseModel):
    id: Optional[str] = Field(None, description="Optional edge identifier")
    type: EdgeType
    source: str
    target: str
    evidence: Evidence
    attributes: Dict[str, Any] = Field(default_factory=dict)

    @validator("source", "target")
    def ensure_non_empty(cls, value: str) -> str:  # noqa: N805
        if not value:
            raise ValueError("source and target must be non-empty")
        return value


class AgentGraphPayload(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

    def summarize(self) -> Dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}


class DisasterContext(BaseModel):
    batch_id: str
    records: List[Dict[str, Any]]
    instructions: str


class GraphBuildResult(BaseModel):
    payload: AgentGraphPayload
    warnings: List[str] = Field(default_factory=list)


__all__ = [
    "NodeType",
    "EdgeType",
    "Evidence",
    "GraphNode",
    "GraphEdge",
    "AgentGraphPayload",
    "DisasterContext",
    "GraphBuildResult",
]
