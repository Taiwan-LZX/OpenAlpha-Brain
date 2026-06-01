#!/usr/bin/env python3
"""
Fetch operators and data-fields from WorldQuant BRAIN API.
Saves to brain_operators.json and brain_datafields.json.

Usage:
    python fetch_brain_schema.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

BRAIN_BASE = "https://api.worldquantbrain.com"
AUTH_URL = f"{BRAIN_BASE}/authentication"
OPERATORS_URL = f"{BRAIN_BASE}/operators"
DATA_FIELDS_URL = f"{BRAIN_BASE}/data-fields"

EMAIL = os.getenv("BRAIN_EMAIL", "")
PASSWORD = os.getenv("BRAIN_PASSWORD", "")

OUTPUT_DIR = Path(__file__).resolve().parent


def authenticate(client: httpx.Client) -> None:
    print(f"[auth] Logging in as {EMAIL} ...")
    resp = client.post(AUTH_URL, auth=(EMAIL, PASSWORD))
    if resp.status_code not in (200, 201):
        print(f"[auth] FAILED (status={resp.status_code}): {resp.text[:300]}")
        sys.exit(1)
    print("[auth] Success.")


def fetch_operators(client: httpx.Client) -> list[dict]:
    print("[operators] Fetching ...")
    resp = client.get(OPERATORS_URL, params={"limit": 200})
    if resp.status_code != 200:
        print(f"[operators] FAILED (status={resp.status_code}): {resp.text[:300]}")
        sys.exit(1)
    data = resp.json()
    operators = data if isinstance(data, list) else data.get("results", [])
    print(f"[operators] Got {len(operators)} operators.")
    return operators


def fetch_data_fields(client: httpx.Client) -> list[dict]:
    print("[data-fields] Fetching (paginated, rate-limit aware) ...")
    all_fields: list[dict] = []
    offset = 0
    limit = 50
    max_retries = 5
    base_params = {
        "instrumentType": "EQUITY",
        "region": "USA",
        "delay": 1,
        "universe": "TOP3000",
    }
    while True:
        params = {**base_params, "limit": limit, "offset": offset}
        str_params = {k: str(v) for k, v in params.items()}
        for attempt in range(max_retries):
            resp = client.get(DATA_FIELDS_URL, params=str_params)
            if resp.status_code == 200:
                break
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "5")) + 1
                print(f"  ... rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            print(f"[data-fields] FAILED at offset={offset} (status={resp.status_code}): {resp.text[:300]}")
            break
        else:
            print(f"[data-fields] Max retries exceeded at offset={offset}")
            break
        if resp.status_code != 200:
            break
        data = resp.json()
        results = data.get("results", []) if isinstance(data, dict) else data
        if not results:
            break
        all_fields.extend(results)
        count = data.get("count", 0) if isinstance(data, dict) else len(results)
        offset += len(results)
        print(f"  ... fetched {len(all_fields)}/{count} fields")
        if len(all_fields) >= count:
            break
        time.sleep(1.0)
    print(f"[data-fields] Got {len(all_fields)} data fields total.")
    return all_fields


def save_json(data: list[dict], filename: str) -> None:
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[save] {filename} → {path} ({len(data)} entries)")


def main() -> None:
    if not EMAIL or not PASSWORD:
        print("[error] BRAIN_EMAIL and BRAIN_PASSWORD must be set in .env")
        sys.exit(1)

    with httpx.Client(timeout=30.0) as client:
        authenticate(client)
        operators = fetch_operators(client)
        data_fields = fetch_data_fields(client)

    save_json(operators, "brain_operators.json")
    save_json(data_fields, "brain_datafields.json")
    print("\n[done] Schema fetch complete.")


if __name__ == "__main__":
    main()
