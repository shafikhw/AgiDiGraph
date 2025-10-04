"""LLM-powered agent for answering graph questions."""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

from app.config import get_settings
from app.services import (
    get_disaster_detail,
    get_disasters_filtered,
    get_related_disasters,
)

LOGGER = logging.getLogger("graph_agent")
MAX_DISASTERS = 0
MAX_IMPACTS_PER_DISASTER = 3
SYSTEM_PROMPT = """
You are an expert climate-disaster analyst. You are given structured context extracted from a
knowledge graph of South Asian floods and storms. Answer user questions by citing disaster IDs
(e.g., 2020-0211-BGD) and summarising impacts, causes, and relationships. When you speculate,
state the uncertainty. Prefer concise paragraphs followed by bullet highlights when helpful.
"""


def _parse_year_from_text(text: str) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    return None



def _graph_overview(disasters: List[Dict[str, Any]]) -> List[str]:
    if not disasters:
        return []
    by_year: Dict[str, Dict[str, float]] = {}
    by_country: Dict[str, Dict[str, float]] = {}
    total_severity = 0.0
    for item in disasters:
        start = item.get("start_date")
        year = start.split("-")[0] if isinstance(start, str) and "-" in start else "unknown"
        by_year.setdefault(year, {"count": 0, "severity": 0.0})
        by_year[year]["count"] += 1
        severity = item.get("severity_score") or 0.0
        by_year[year]["severity"] += severity
        country = (item.get("country") or "unknown").title()
        by_country.setdefault(country, {"count": 0, "severity": 0.0})
        by_country[country]["count"] += 1
        by_country[country]["severity"] += severity
        total_severity += severity
    lines: List[str] = []
    lines.append(f"Total disasters: {len(disasters)} (aggregate severity {total_severity:.1f})")
    for year, stats in sorted(by_year.items()):
        lines.append(
            f"  Year {year}: {int(stats['count'])} events, severity {stats['severity']:.1f}"
        )
    top_countries = sorted(
        by_country.items(),
        key=lambda kv: (kv[1]["severity"], kv[1]["count"]),
        reverse=True,
    )[:5]
    if top_countries:
        lines.append(
            "  Leading countries: "
            + ", ".join(
                f"{name} ({int(data['count'])} events, severity {data['severity']:.1f})"
                for name, data in top_countries
            )
        )
    return lines

def _build_context(
    graph: Any,
    disasters: Iterable[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    disaster_list = list(disasters)
    lines: List[str] = []
    sources: List[str] = []

    overview = _graph_overview(disaster_list)
    if overview:
        lines.append("Graph overview:")
        lines.extend(overview)
        lines.append("")

    limit = len(disaster_list) if MAX_DISASTERS in (None, 0) else min(len(disaster_list), MAX_DISASTERS)
    selected = disaster_list[:limit]
    remainder = len(disaster_list) - len(selected)

    for item in selected:
        disaster_id = item["id"]
        sources.append(disaster_id)
        detail = get_disaster_detail(graph, disaster_id)
        if not detail:
            continue
        attrs = detail["disaster"].get("metadata", {}).get("attributes", {})
        severity_level = attrs.get("severity_level")
        severity_score = attrs.get("severity_score")
        header = (
            f"Disaster {disaster_id} ({detail['disaster']['label']}), "
            f"type={attrs.get('disaster_type')}, severity={severity_level} ({severity_score})."
        )
        lines.append(header)
        impacts: List[str] = []
        impact_records = detail.get("impacts", [])
        impact_records.sort(
            key=lambda imp: imp.get("metadata", {}).get("attributes", {}).get("severity_score") or 0.0,
            reverse=True,
        )
        for impact in impact_records[:MAX_IMPACTS_PER_DISASTER]:
            metric = impact.get("metric") or impact.get("label")
            value = impact.get("value")
            unit = impact.get("unit") or ""
            severity_component = (
                impact.get("metadata", {}).get("attributes", {}).get("severity_score")
                if impact.get("metadata")
                else None
            )
            if severity_component is not None:
                impacts.append(f"  - {metric}: {value} {unit} (severity {severity_component})")
            else:
                impacts.append(f"  - {metric}: {value} {unit}")
        if MAX_IMPACTS_PER_DISASTER > 0 and len(impact_records) > MAX_IMPACTS_PER_DISASTER:
            impacts.append(
                f"  - ... plus {len(impact_records) - MAX_IMPACTS_PER_DISASTER} additional impact metrics"
            )
        if impacts:
            lines.extend(impacts)
        related_edges = get_related_disasters(graph, disaster_id)
        if related_edges:
            rel_text = ", ".join(
                f"{edge['id']} ({edge.get('relation')})" for edge in related_edges
            )
            lines.append(f"  - Related: {rel_text}")

    if remainder > 0:
        lines.append(f"... {remainder} additional disasters truncated for brevity.")

    context = "\n".join(lines)
    return context, sorted(set(sources))


class GraphAgent:
    def __init__(self, model: str) -> None:
        settings = get_settings()
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model

    def answer(
        self,
        graph: Any,
        question: str,
        year: Optional[int] = None,
        country: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        disasters = get_disasters_filtered(graph, year=year, country=country)
        if not disasters and year is None:
            inferred_year = _parse_year_from_text(question)
            if inferred_year:
                disasters = get_disasters_filtered(graph, year=inferred_year, country=country)
        if not disasters:
            disasters = get_disasters_filtered(graph, year=None, country=country)
        disasters.sort(
            key=lambda item: item.get("severity_score") or 0.0,
            reverse=True,
        )
        limit = len(disasters) if MAX_DISASTERS in (None, 0) else min(len(disasters), MAX_DISASTERS)
        selected = disasters[:limit]
        context, sources = _build_context(graph, selected)
        if not context.strip():
            answer = (
                "I could not locate relevant disasters in the current graph for that query. "
                "Try adjusting the year or country filter."
            )
            return answer, sources
        user_prompt = (
            "CONTEXT:\n" + context + "\n\nQUESTION:\n" + question + "\n\n"
            "Respond with factual statements grounded in the context. Refer to disasters by ID."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content if response.choices else ""
        if not content:
            content = (
                "I could not produce an answer right now. Please try again with a narrower query."
            )
        return content.strip(), sources


def _fallback_answer(graph: Any, year: Optional[int], country: Optional[str]) -> Tuple[str, List[str]]:
    disasters = get_disasters_filtered(graph, year=year, country=country)
    disasters.sort(key=lambda item: item.get("severity_score") or 0.0, reverse=True)
    top = disasters[:5]
    if not top:
        return (
            "No disasters matched your filters. Try a different year or country.",
            [],
        )
    lines = [
        "Here are notable disasters from the graph:",
    ]
    for item in top:
        lines.append(
            f"- {item['id']} ({item['label']}), severity {item.get('severity_level')} "
            f"({item.get('severity_score')})."
        )
    return "\n".join(lines), [item["id"] for item in top]


@lru_cache()
def get_agent() -> Optional[GraphAgent]:
    settings = get_settings()
    if settings.use_offline_reasoner:
        LOGGER.info("Agent disabled because offline reasoner is enabled")
        return None
    if OpenAI is None:
        LOGGER.warning("openai package is missing; agent disabled")
        return None
    try:
        return GraphAgent(model=settings.openai_model)
    except RuntimeError as exc:
        LOGGER.warning("Agent initialisation failed: %s", exc)
        return None


def answer_question(
    graph: Any,
    question: str,
    year: Optional[int] = None,
    country: Optional[str] = None,
) -> Tuple[str, List[str]]:
    agent = get_agent()
    if agent is None:
        return _fallback_answer(graph, year, country)
    try:
        return agent.answer(graph, question, year=year, country=country)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Agent failed: %s", exc)
        return _fallback_answer(graph, year, country)
