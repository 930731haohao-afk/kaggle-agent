"""Whitelisted external-data sources (v5 Plan C).

Only sources listed in knowledge/task_priors.md may be fetched. Every fetch is
cached with a snapshot date so joins are replayable; the World Bank `lastupdated`
field is recorded as the source snapshot.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"

# Whitelist — mirrors knowledge/task_priors.md. Do not add sources here without
# adding them to the whitelist table with a leakage rule first.
WB_INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "gdp": "NY.GDP.MKTP.CD",
    "population": "SP.POP.TOTL",
    "cpi_inflation": "FP.CPI.TOTL.ZG",
}

# Common competition-name -> World Bank-name variants.
WB_NAME_ALIASES = {
    "South Korea": "Korea, Rep.",
    "Russia": "Russian Federation",
    "Turkey": "Turkiye",
    "Vietnam": "Viet Nam",
    "Egypt": "Egypt, Arab Rep.",
    "Iran": "Iran, Islamic Rep.",
    "Venezuela": "Venezuela, RB",
    "Slovakia": "Slovak Republic",
    "Czech Republic": "Czechia",
    "United States of America": "United States",
}

_WB_BASE = "https://api.worldbank.org/v2"


def _http_json(url: str, timeout: int = 120, retries: int = 2) -> object:
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001 — retry once, then surface
            last = e
            logger.warning("fetch attempt %d failed: %s", attempt + 1, e)
    raise last


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / name


def fetch_country_map(refresh: bool = False) -> pd.DataFrame:
    """World Bank country table: columns [name, iso2, iso3]. Cached."""
    cp = _cache_path("wb_countries.json")
    if cp.exists() and not refresh:
        raw = json.loads(cp.read_text())
    else:
        raw = _http_json(f"{_WB_BASE}/country/all?format=json&per_page=400")
        cp.write_text(json.dumps(raw))
        logger.info("fetched WB country table (%d entries)", len(raw[1]))
    rows = [
        {"name": c["name"], "iso2": c["id"] if len(c["id"]) == 2 else c["iso2Code"], "iso3": c["id"] if len(c["id"]) == 3 else ""}
        for c in raw[1]
        if (c.get("region") or {}).get("id") not in (None, "NA")  # drop aggregates & regionless
    ]
    df = pd.DataFrame(rows)
    return df[df["iso3"] != ""].reset_index(drop=True)


def resolve_iso3(names: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map competition country names -> ISO3. Returns (mapping, unmatched).

    Unmatched names are returned, never guessed — the caller must either extend
    WB_NAME_ALIASES or record the gap in dossier.json `not_recorded`.
    """
    table = fetch_country_map()
    by_name = {r["name"]: r["iso3"] for _, r in table.iterrows()}
    mapping, unmatched = {}, []
    for n in names:
        wb_name = WB_NAME_ALIASES.get(n, n)
        if wb_name in by_name:
            mapping[n] = by_name[wb_name]
        else:
            unmatched.append(n)
    return mapping, unmatched


def fetch_worldbank(indicator_key: str, year_from: int, year_to: int,
                    refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Fetch one whitelisted WB indicator for all countries.

    Returns (df[iso3, country, year, value], meta{indicator, snapshot, fetched}).
    """
    if indicator_key not in WB_INDICATORS:
        raise ValueError(f"indicator '{indicator_key}' is not whitelisted: {list(WB_INDICATORS)}")
    code = WB_INDICATORS[indicator_key]
    cp = _cache_path(f"wb_{indicator_key}_{year_from}_{year_to}.json")
    if cp.exists() and not refresh:
        raw = json.loads(cp.read_text())
    else:
        url = (f"{_WB_BASE}/country/all/indicator/{code}"
               f"?format=json&per_page=20000&date={year_from}:{year_to}")
        raw = _http_json(url)
        pages = raw[0].get("pages", 1)
        for p in range(2, pages + 1):
            more = _http_json(url + f"&page={p}")
            raw[1].extend(more[1])
        cp.write_text(json.dumps(raw))
    meta = {
        "indicator": code,
        "snapshot": raw[0].get("lastupdated", "unknown"),
        "fetched": date.today().isoformat(),
    }
    rows = [
        {"iso3": e["countryiso3code"], "country": e["country"]["value"],
         "year": int(e["date"]), "value": e["value"]}
        for e in raw[1]
        if e["value"] is not None and e["countryiso3code"] and str(e["date"]).isdigit()
    ]
    return pd.DataFrame(rows), meta


def fetch_holidays(country_names: list[str], years: list[int]) -> tuple[pd.DataFrame, dict]:
    """Holiday calendar via the `holidays` package (offline, deterministic).

    Returns (df[country, date, holiday], meta). Unmapped countries are reported
    in meta['unmatched'], never guessed.
    """
    import holidays as _hol

    mapping, unmatched = resolve_iso3(country_names)
    iso2_by_name = {}
    table = fetch_country_map()
    iso2_of_iso3 = {r["iso3"]: r["iso2"] for _, r in table.iterrows()}
    for name, iso3 in mapping.items():
        iso2_by_name[name] = iso2_of_iso3.get(iso3, "")

    rows = []
    for name, iso2 in iso2_by_name.items():
        try:
            cal = _hol.country_holidays(iso2, years=years)
        except (KeyError, NotImplementedError):
            unmatched.append(name)
            continue
        for d, label in sorted(cal.items()):
            rows.append({"country": name, "date": pd.Timestamp(d), "holiday": label})
    meta = {
        "source": f"holidays=={_hol.__version__}",
        "years": years,
        "unmatched": unmatched,
        "fetched": date.today().isoformat(),
    }
    return pd.DataFrame(rows), meta
