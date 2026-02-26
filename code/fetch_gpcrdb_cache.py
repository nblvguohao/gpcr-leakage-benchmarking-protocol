#!/usr/bin/env python3
"""
Fetch BW residue annotations from GPCRdb REST API.
===================================================
Populates data/gpcrdb_residues_cache.json, which is required by:
  - run_paper_bw_enhanced.py
  - run_interpretability.py
  - run_esm2_bwsite.py
  - run_multi_gprotein.py
  - run_reviewer_analyses.py

Usage:
  python code/fetch_gpcrdb_cache.py            # fetch all 230 receptors
  python code/fetch_gpcrdb_cache.py --limit 50  # fetch first 50 only (quick test)

Note: Fetching all 230 receptors takes ~5-10 minutes due to API rate limits.
"""
import os
import sys
import json
import time
import argparse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CACHE_PATH = os.path.join(DATA_DIR, "gpcrdb_residues_cache.json")
CSV_PATH = os.path.join(DATA_DIR, "gpcrdb_coupling_dataset.csv")

GPCRDB_API = "https://gpcrdb.org/services/residues/extended"


def fetch_residues_requests(entry_name, timeout=60):
    """Fetch via requests library."""
    url = f"{GPCRDB_API}/{entry_name}/"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_residues_urllib(entry_name, timeout=60):
    """Fallback: fetch via urllib (no extra dependencies)."""
    import urllib.request
    url = f"{GPCRDB_API}/{entry_name}/"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_one(entry_name, timeout=60):
    """Try requests first, fall back to urllib."""
    if HAS_REQUESTS:
        return fetch_residues_requests(entry_name, timeout)
    return fetch_residues_urllib(entry_name, timeout)


def main():
    parser = argparse.ArgumentParser(description="Fetch GPCRdb BW residue cache")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max receptors to fetch (0 = all)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="HTTP timeout per request (seconds)")
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run fetch_gpcrdb_data.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    entries = df["entry_name"].tolist()
    if args.limit > 0:
        entries = entries[:args.limit]

    # Load existing cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"Loaded existing cache: {len(cache)} entries")
        except Exception:
            pass

    to_fetch = [e for e in entries if e not in cache]
    print(f"Total entries: {len(entries)}, already cached: {len(entries) - len(to_fetch)}, "
          f"to fetch: {len(to_fetch)}")

    if not to_fetch:
        print("Cache is complete. Nothing to do.")
        return

    success, fail = 0, 0
    for i, entry in enumerate(to_fetch):
        print(f"[{i+1}/{len(to_fetch)}] {entry} ... ", end="", flush=True)
        try:
            data = fetch_one(entry, timeout=args.timeout)
            cache[entry] = data
            success += 1
            print("OK")
        except Exception as e:
            fail += 1
            print(f"FAIL ({e})")

        # Rate-limit: ~1 request/second
        time.sleep(1.0)

        # Checkpoint every 20 entries
        if (i + 1) % 20 == 0:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            print(f"  [checkpoint] saved {len(cache)} entries")

    # Final save
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    print(f"\nDone. Success: {success}, Failed: {fail}, Total cached: {len(cache)}")
    print(f"Saved: {CACHE_PATH}")


if __name__ == "__main__":
    main()
