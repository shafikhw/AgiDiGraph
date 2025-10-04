# AgiDiGraph

AgiDiGraph builds an evidence-rich disaster impact knowledge graph for South Asia floods and storms. An LLM agent interprets EM-DAT rows, generates nodes/edges with provenance, and publishes them through a FastAPI service (Streamlit UI coming in a later phase).

## Environment

- Activate the provided Anaconda environment: `source C:/Users/shafi/anaconda3/Scripts/activate ai-agentic`
- Install project dependencies once: `pip install -e .[dev]`
- Populate `.env` with `OPENAI_API_KEY`, `USE_OFFLINE_REASONER`, `GRAPH_MAX_BATCHES`, `GRAPH_CACHE_ENABLED`, etc.

Key toggles:
- `USE_OFFLINE_REASONER=true` forces deterministic extraction (no OpenAI spend).
- `USE_OFFLINE_REASONER=false` keeps the OpenAI agent active.
- `GRAPH_CACHE_ENABLED=true` (default) reuses graph artifacts when the EM-DAT file is unchanged; set to `false` to force rebuilds.
- `GRAPH_MAX_BATCHES` only applies when cache is disabled and limits how many records are processed (for quick smoke runs).

## Build the Graph

```bash
# from repository root with the environment activated
python scripts/build_graph.py
```

The builder loads EM-DAT records, processes them in batches, calls OpenAI (`gpt-4o-mini`) when enabled, validates the JSON payload, and merges into a NetworkX graph. Artifacts are written to `graphs/disaster_impacts.json` and `graphs/disaster_impacts.graphml`.

The results are cached in `graphs/cache_manifest.json`; if the EM-DAT CSV is unchanged, subsequent runs reuse the cached artifacts for faster setup.

If the graph artifacts are missing when the API starts, `scripts.build_graph.build_and_export` is invoked automatically to regenerate them.

## Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /healthz` – shows active model and offline/online mode.
- `GET /graph/status` – counts nodes/edges in the current graph.
- `GET /disasters?year=2020&country=Bangladesh` – filters disaster summaries.
- `GET /disasters/{id}` – returns impacts and evidence for a specific event.
- `GET /disasters/{id}/related` – lists temporally related disasters.
- `POST /chat` – placeholder; will be wired to conversational graph reasoning in a later phase.

## Testing

```bash
pytest
```

Service and graph helper tests inject synthetic graphs to keep runs fast and offline-friendly.

## LLM Budget Tips

- Set `GRAPH_MAX_BATCHES` (default 5) to limit how many batches the builder will send to OpenAI. With the South Asia dataset, a value of `2` yields roughly two requests (~1K tokens per batch).
- Delete `logs/cache/*.json` when you need fresh generations; otherwise cached payloads are reused without additional cost.
- You can rebuild offline first (to ensure pipeline health) and then re-run online with the cap tightened.

## Roadmap

- Streamlit chat + pyvis visualization wired to the FastAPI endpoints.
- NOAA anomaly integration to extend the graph with climate-event nodes.
- Docker/Makefile polish and demo storyline for the hackathon judging flow.

## Run the Streamlit UI

```bash
python -m streamlit run ui/app.py --server.port 8501
```

The UI expects the FastAPI server to be available (default `http://localhost:8000`). Override with `AGIDIGRAPH_API_BASE` if you host the API elsewhere. The sidebar lets you filter by year/country, while chat requests reuse the FastAPI `/chat` endpoint. Cached API responses live in Streamlit session state; use the "Rerun" button if you want a fresh pull.

## Docker Image

```bash
docker build -t agidigraph:latest .
docker run --rm -p 8000:8000 agidigraph:latest
```
or to load an existing docker image
```
docker run -p 8000:8000 ghcr.io/shafikhw/agidigraph:latest
```

Mount local data or provide `OPENAI_API_KEY` via `-e` when you need online graph construction inside the container.

## Demo Scenario Script

```bash
python scripts/demo_scenario.py --base-url http://localhost:8000 --year 2020 --country Bangladesh
```

The script walks through the hackathon user stories: it checks health, lists disasters for the chosen filters, prints impact breakdowns, and shows related events.


## LLM Agent

The `/chat` endpoint and Streamlit chat pane now call an OpenAI-powered agent. Set `OPENAI_API_KEY` in `.env` and keep `USE_OFFLINE_REASONER=false` to enable it. When the key is missing or offline mode is active, the chat falls back to a deterministic summary.

## Contributors:
- Lama Mawlawi 
- Suha abo Dahish 
- Shafik Houeidi
