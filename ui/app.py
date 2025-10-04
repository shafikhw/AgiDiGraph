"""Streamlit UI for AgiDiGraph."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
import streamlit as st
from pyvis.network import Network

API_BASE = os.getenv("AGIDIGRAPH_API_BASE", "http://localhost:8000")
HTTP_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
RETRY_ATTEMPTS = 2


def _request(method: str, path: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    url = f"{API_BASE}{path}"
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = httpx.request(method, url, timeout=HTTP_TIMEOUT, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            if attempt + 1 < RETRY_ATTEMPTS:
                st.warning(f"API timed out; retrying ({attempt + 1}/{RETRY_ATTEMPTS})...")
                continue
            st.error(f"API request failed: {exc}")
            return None
        except httpx.HTTPError as exc:
            st.error(f"API request failed: {exc}")
            return None
    return None


def _api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    return _request('GET', path, params=params)


def _api_post(path: str, json_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _request('POST', path, json=json_payload)

@st.cache_data(show_spinner=False)
def fetch_health() -> Optional[Dict[str, Any]]:
    return _api_get("/healthz")



@st.cache_data(show_spinner=False)
def fetch_status() -> Optional[Dict[str, int]]:
    return _api_get("/graph/status")


@st.cache_data(show_spinner=False)
def fetch_disasters(year: Optional[int], country: Optional[str]) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {}
    if year is not None:
        params["year"] = year
    if country:
        params["country"] = country
    data = _api_get("/disasters", params=params)
    if not data:
        return []
    return data  # type: ignore[return-value]


@st.cache_data(show_spinner=False)
def fetch_disaster_detail(disaster_id: str) -> Optional[Dict[str, Any]]:
    return _api_get(f"/disasters/{disaster_id}")


@st.cache_data(show_spinner=False)
def fetch_disaster_details_batch(disaster_ids: List[str]) -> Dict[str, Any]:
    if not disaster_ids:
        return {}
    data = _api_post("/disasters/batch", {"disaster_ids": disaster_ids})
    if not data:
        return {}
    return data.get("details", {})


@st.cache_data(show_spinner=False)
def fetch_related(disaster_id: str) -> List[Dict[str, Any]]:
    data = _api_get(f"/disasters/{disaster_id}/related")
    if not data:
        return []
    return data  # type: ignore[return-value]


def build_network(
    disasters: List[Dict[str, Any]],
    related_cache: Dict[str, List[Dict[str, Any]]],
    detail_cache: Dict[str, Dict[str, Any]],
) -> str:
    net = Network(height="600px", width="100%", bgcolor="#FFFFFF", font_color="#1f2933")
    net.barnes_hut()

    for item in disasters:
        disaster_id = item["id"]
        disaster_type = item.get("disaster_type")
        severity_raw = item.get("severity_score")
        try:
            severity = float(severity_raw) if severity_raw is not None else 1.0
        except (TypeError, ValueError):
            severity = 1.0
        tooltip = "<br>".join(
            filter(
                None,
                [
                    item.get("label"),
                    disaster_type,
                    item.get("country"),
                    item.get("start_date"),
                    f"severity {item.get('severity_level')} ({severity:.1f})",
                ],
            )
        )
        color = "#2563eb" if disaster_type == "Storm" else "#059669"
        net.add_node(
            disaster_id,
            label=item.get("label", disaster_id),
            title=tooltip,
            color=color,
            value=max(severity, 1.0),
        )

        detail = detail_cache.get(disaster_id)
        if detail:
            for impact in detail.get("impacts", []):
                impact_id = impact["id"]
                attributes = impact.get("metadata", {}).get("attributes", {})
                impact_severity = attributes.get("severity_score") or 0.5
                impact_label = impact.get("label") or impact.get("metric")
                impact_value = impact.get("value")
                impact_unit = impact.get("unit") or ""
                net.add_node(
                    impact_id,
                    label=impact_label,
                    title=f"{impact_label}: {impact_value} {impact_unit}",
                    color="#f97316",
                    value=max(float(impact_severity), 0.4),
                    shape="dot",
                )
                net.add_edge(
                    disaster_id,
                    impact_id,
                    title=f"{impact_label}: {impact_value} {impact_unit}",
                    color="#f97316",
                )

    for node in disasters:
        edges = related_cache.get(node["id"], [])
        for edge in edges:
            target = edge.get("id")
            if not target:
                continue
            title = edge.get("relation") or edge.get("reason") or "related"
            net.add_edge(
                node["id"],
                target,
                title=title,
                color="#6366f1",
            )

    return net.generate_html(notebook=False)


def ensure_chat_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def append_chat(role: str, content: str) -> None:
    st.session_state["chat_history"].append({"role": role, "content": content})


def render_sidebar(
    health: Optional[Dict[str, Any]],
    status: Optional[Dict[str, int]],
) -> Tuple[Optional[int], Optional[str]]:
    st.sidebar.header("Filters")
    year: Optional[int] = None
    if st.sidebar.checkbox("Filter by year", value=True):
        year = int(
            st.sidebar.number_input(
                "Year", min_value=2020, max_value=2025, value=2020, step=1
            )
        )

    countries = ["All", "Bangladesh", "India", "Nepal"]
    country_option = st.sidebar.selectbox("Country", countries)
    country = None if country_option == "All" else country_option

    st.sidebar.markdown("---")
    if health:
        st.sidebar.success(
            f"Model: {health.get('model')} • Offline: {health.get('offline')}"
        )
    if status:
        st.sidebar.info(
            f"Graph nodes: {status.get('node_count', 0)} • edges: {status.get('edge_count', 0)}"
        )
    return year, country


def render_chat_tab(year: Optional[int], country: Optional[str]) -> None:
    ensure_chat_state()
    st.subheader("Chat with the graph")
    if not st.session_state.get("chat_style_injected"):
        st.session_state["chat_style_injected"] = True
        st.markdown(
            """
            <style>
            div[data-testid="stChatMessageContainer"] {
                max-height: calc(100vh - 9rem);
                overflow-y: auto;
                padding-bottom: 8rem;
            }
            div[data-testid="stChatInputContainer"] {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                padding: 0.75rem 1rem;
                background: var(--background-color);
                box-shadow: 0 -2px 6px rgba(15, 23, 42, 0.08);
                z-index: 999;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    for message in st.session_state["chat_history"]:
        alignment = "user" if message["role"] == "user" else "assistant"
        st.chat_message(alignment).write(message["content"])

    user_input = st.chat_input("Ask a question about disasters and impacts...")
    if user_input:
        # render the user message immediately
        append_chat("user", user_input)
        st.chat_message("user").write(user_input)

        payload: Dict[str, Any] = {"message": user_input}
        history_payload = st.session_state.get("chat_history", [])[:-1]
        if history_payload:
            payload["history"] = history_payload[-10:]
        if year is not None:
            payload["year"] = year
        if country:
            payload["country"] = country
        response = _api_post("/chat", payload)
        if response:
            answer = response.get("answer", "No answer")
            sources = response.get("sources", [])
            if sources:
                answer = f"{answer}\n\nSources: {', '.join(sources)}"
            append_chat("assistant", answer)
            st.chat_message("assistant").write(answer)


def render_graph_tab(
    disasters: List[Dict[str, Any]],
    related_cache: Dict[str, List[Dict[str, Any]]],
    detail_cache: Dict[str, Dict[str, Any]],
) -> None:
    st.subheader("Disaster graph")
    if not disasters:
        st.warning("No disasters found for the selected filters.")
        return

    with st.spinner("Building visualization..."):
        html = build_network(disasters, related_cache, detail_cache)
        st.components.v1.html(html, height=620, scrolling=True)

    disaster_ids = [item["id"] for item in disasters]
    selection = st.selectbox("Select a disaster for details", disaster_ids)
    if selection:
        detail = detail_cache.get(selection) or fetch_disaster_detail(selection)
        if not detail:
            st.error("Failed to load disaster details.")
            return
        st.markdown(f"### {detail['disaster']['label']}")
        attrs = detail["disaster"].get("metadata", {}).get("attributes", {})
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Country:**", attrs.get("country"))
            st.write("**Type:**", attrs.get("disaster_type"))
            severity_score = attrs.get("severity_score")
            severity_level = attrs.get("severity_level")
            if severity_score is not None and severity_level:
                st.write("**Severity:**", f"{severity_level} ({severity_score})")
            else:
                st.write("**Severity:**", "n/a")
            st.write("**Start:**", attrs.get("start_date"))
            st.write("**End:**", attrs.get("end_date"))
        with col2:
            st.write("**Impacts:**")
            for impact in detail.get("impacts", []):
                metric = impact.get("metric") or impact.get("label")
                value = impact.get("value")
                unit = impact.get("unit") or ""
                st.write(f"- {metric}: {value} {unit}")
        st.write("**Evidence:**")
        for edge in detail.get("edges", []):
            evidence = edge.get("evidence", {})
            st.caption(
                f"{edge.get('type')} -> {edge.get('target')}: {evidence.get('summary')} "
                f"(confidence={evidence.get('confidence')})"
            )


def load_related_cache(disasters: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    cache: Dict[str, List[Dict[str, Any]]] = {}
    for item in disasters:
        cache[item["id"]] = fetch_related(item["id"])
    return cache


def load_detail_cache(disasters: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    ids = [item["id"] for item in disasters]
    details = fetch_disaster_details_batch(ids)
    return {key: value for key, value in details.items() if value}


def main() -> None:
    st.set_page_config(page_title="AgiDiGraph", layout="wide")
    st.title("AgiDiGraph Exploration Console")

    health = fetch_health()
    status = fetch_status()

    year, country = render_sidebar(health, status)

    disasters = fetch_disasters(year, country)
    related_cache = load_related_cache(disasters) if disasters else {}
    detail_cache = load_detail_cache(disasters) if disasters else {}

    tabs = st.tabs(["Chat", "Graph"])
    with tabs[0]:
        render_chat_tab(year, country)
    with tabs[1]:
        render_graph_tab(disasters, related_cache, detail_cache)


if __name__ == "__main__":
    main()
