"""CLI demo scenario for AgiDiGraph API."""
from __future__ import annotations

import argparse
from typing import Any, Dict

import httpx

DEFAULT_BASE = "http://localhost:8000"
TIMEOUT = 10.0


def call_endpoint(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    response = client.request(method, path, timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a scripted API walkthrough.")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="FastAPI base URL")
    parser.add_argument("--year", type=int, default=2020, help="Year filter for disaster queries")
    parser.add_argument(
        "--country",
        default="Bangladesh",
        choices=["Bangladesh", "India", "Nepal"],
        help="Country filter",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    with httpx.Client(base_url=base) as client:
        print("-> Health check")
        health = call_endpoint(client, "GET", "/healthz")
        print(health)

        print("\n-> Graph status")
        status = call_endpoint(client, "GET", "/graph/status")
        print(status)

        print(f"\n-> Disasters in {args.year} ({args.country})")
        disasters = call_endpoint(
            client,
            "GET",
            "/disasters",
            params={"year": args.year, "country": args.country},
        )
        if not disasters:
            print("No disasters found for filters.")
            return
        for item in disasters:
            print(f"- {item['id']}: {item['label']} (impacts={item['total_impacts']})")

        first_disaster = disasters[0]["id"]
        print(f"\n-> Impacts for {first_disaster}")
        detail = call_endpoint(client, "GET", f"/disasters/{first_disaster}")
        for impact in detail.get("impacts", []):
            metric = impact.get("metric") or impact.get("label")
            value = impact.get("value")
            unit = impact.get("unit") or ""
            print(f"  - {metric}: {value} {unit}".strip())

        print(f"\n-> Related disasters for {first_disaster}")
        related = call_endpoint(client, "GET", f"/disasters/{first_disaster}/related")
        if related:
            for edge in related:
                print(f"  - {edge['id']}: {edge.get('relation')}")
        else:
            print("  (none)")

        print("\n-> Demo complete")


if __name__ == "__main__":
    main()
