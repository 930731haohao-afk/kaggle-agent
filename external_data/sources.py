"""Whitelisted external-data sources (v5 Plan C).

Only sources listed in knowledge/task_priors.md may be fetched. Every network
fetch is cached under cache/ with a recorded upstream snapshot (WB lastupdated,
ECB latest period, OWID Last-Modified) so joins are replayable; holiday
calendars are offline-deterministic and pinned by package version instead.
Cache writes are atomic (tmp + rename) and responses are validated BEFORE
caching — a bad 200 must never poison the cache.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"

# Whitelist — mirrors knowledge/task_priors.md. Do not add sources here without
# adding them to the whitelist table with a leakage rule first.
WB_INDICATORS = {
    # Level covariates. PREFER the constant-price / PPP series for ratio_target and
    # log_offset: current-USD series carry exchange-rate and inflation shocks straight into
    # a multiplicative prediction. Measured 2021->2022 cross-country spread on the s3e19
    # countries: current USD 14.6pp (Japan -14.5%, Argentina +30%) vs constant price 3.0pp.
    "gdp_per_capita": "NY.GDP.PCAP.CD",             # current USD — features only
    "gdp_per_capita_const": "NY.GDP.PCAP.KD",       # constant 2015 USD — level covariate
    "gdp_per_capita_ppp": "NY.GDP.PCAP.PP.KD",      # PPP, constant intl$ — level covariate
    "gdp": "NY.GDP.MKTP.CD",
    "gdp_const": "NY.GDP.MKTP.KD",
    "gdp_growth_pct": "NY.GDP.MKTP.KD.ZG",
    "population": "SP.POP.TOTL",
    "cpi_inflation": "FP.CPI.TOTL.ZG",
    "unemployment_pct": "SL.UEM.TOTL.ZS",
    "urban_pop_pct": "SP.URB.TOTL.IN.ZS",
    "internet_users_pct": "IT.NET.USER.ZS",
}

# ECB reference rates: currency vs EUR, monthly averages — currency whitelist
ECB_FX_CURRENCIES = {"USD", "GBP", "JPY", "SEK", "NOK", "DKK", "CHF", "CAD", "AUD", "PLN", "CZK", "HUF"}

# OWID COVID compact dataset — metric whitelist (columns of compact.csv)
OWID_COVID_METRICS = {
    "new_cases_per_million",
    "new_deaths_per_million",
    "new_cases_smoothed_per_million",
    "new_deaths_smoothed_per_million",
    "total_cases_per_million",
    "total_deaths_per_million",
}
OWID_COVID_URL = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"

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


def _retrying(fn, retries: int = 2):
    last = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — retry, then surface
            last = e
            logger.warning("fetch attempt %d failed: %s", attempt + 1, e)
    raise last


def _http_json(url: str, timeout: int = 120, retries: int = 2) -> object:
    def go():
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    return _retrying(go, retries)


def _http_text(url: str, timeout: int = 120, retries: int = 2) -> str:
    def go():
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    return _retrying(go, retries)


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / name


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# vintage pinning (2026-07-30, readiness-audit finding)
#
# The cache is keyed by indicator code and year range only -- not by vintage -- so a cold
# cache or `refresh=True` could silently hand a rerun a different revision of the same
# series than the one an experiment's reported numbers came from. This matters specifically
# for World Bank GDP: the Bank periodically rebases its constant-price series to a new base
# year, which rewrites EVERY year of history, not just recent ones -- so "the same indicator,
# refetched" can change what a ratio_target operator multiplies into a 2017 training row as
# well as a 2022 test row.
#
# SOURCES_LOCK_PATH is a tracked file, not a cache artifact: it is committed to git so the
# expected vintage travels with the code, the same way a dependency lockfile does. The first
# fetch of a given (source, year range) pins it; every fetch after that must match or the
# call fails loudly instead of proceeding on a silently different series.
# ---------------------------------------------------------------------------
SOURCES_LOCK_PATH = Path(__file__).parent / "sources.lock.json"


class VintageMismatchError(RuntimeError):
    """A whitelisted source returned a different vintage than the pinned one."""


def _load_lock() -> dict:
    if SOURCES_LOCK_PATH.exists():
        return json.loads(SOURCES_LOCK_PATH.read_text())
    return {}


def pin_or_verify_snapshot(lock_key: str, snapshot: str, *, pin_new: bool = True) -> None:
    """Pin `snapshot` for `lock_key` on first sight; raise if a later fetch disagrees.

    Called once per fetch, right after a source's `meta["snapshot"]` is known. Writing the
    pin is atomic and the file is meant to be committed, so an unexpected vintage change
    shows up as a git diff on `sources.lock.json` the moment it is (re)pinned deliberately,
    and as a loud `VintageMismatchError` if it is not.
    """
    lock = _load_lock()
    pinned = lock.get(lock_key)
    if pinned is None:
        if not pin_new:
            raise VintageMismatchError(
                f"{lock_key!r} has no pinned vintage in {SOURCES_LOCK_PATH.name} and "
                f"pin_new=False refuses to create one silently")
        lock[lock_key] = snapshot
        _atomic_write_text(SOURCES_LOCK_PATH, json.dumps(lock, indent=2, sort_keys=True))
        return
    if pinned != snapshot:
        raise VintageMismatchError(
            f"{lock_key!r} vintage changed: pinned {pinned!r} in {SOURCES_LOCK_PATH.name}, "
            f"fetch returned {snapshot!r}. If this is a deliberate refresh (e.g. World Bank "
            f"republished the series), delete this key from sources.lock.json, re-fetch, and "
            f"commit the new pin along with a note of why every downstream number using it "
            f"needs re-checking; do not let this pass silently.")


def _download_file(url: str, dest: Path, timeout: int = 300, retries: int = 2) -> str | None:
    """Stream url to dest atomically; verify Content-Length when the server
    sends one. Returns the Last-Modified header if present."""
    def go():
        tmp = dest.with_suffix(dest.suffix + f".part{os.getpid()}")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                last_modified = r.headers.get("Last-Modified")
                expected = r.headers.get("Content-Length")
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                got = tmp.stat().st_size
                if expected is not None and got != int(expected):
                    raise IOError(f"partial download: got {got} of {expected} bytes")
            os.replace(tmp, dest)
            return last_modified
        finally:
            tmp.unlink(missing_ok=True)
    return _retrying(go, retries)


def fetch_country_map(refresh: bool = False) -> pd.DataFrame:
    """World Bank country table: columns [name, iso2, iso3, region]. Cached."""
    cp = _cache_path("wb_countries.json")
    if cp.exists() and not refresh:
        raw = json.loads(cp.read_text())
    else:
        raw = _http_json(f"{_WB_BASE}/country/all?format=json&per_page=400")
        if not (isinstance(raw, list) and len(raw) > 1 and raw[1]):
            raise IOError("WB country table response malformed; not caching")
        _atomic_write_text(cp, json.dumps(raw))
        logger.info("fetched WB country table (%d entries)", len(raw[1]))
    rows = [
        {"name": c["name"],
         "iso2": c["id"] if len(c["id"]) == 2 else c["iso2Code"],
         "iso3": c["id"] if len(c["id"]) == 3 else "",
         "region": (c.get("region") or {}).get("value", "")}
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
                    refresh: bool = False, verify_vintage: bool = True) -> tuple[pd.DataFrame, dict]:
    """Fetch one whitelisted WB indicator for all countries (all pages).

    Returns (df[iso3, country, year, value], meta{indicator, snapshot, fetched}).
    Cache is keyed by the WB indicator CODE, so remapping a whitelist key can
    never serve stale data for the old code.

    `verify_vintage` (default True) pins/checks the fetched snapshot against
    `sources.lock.json` (see `pin_or_verify_snapshot`) — the guard against a rebase or a
    republished series silently changing what a `ratio_target`/`log_offset` operator
    multiplies into every training and test row alike.
    """
    if indicator_key not in WB_INDICATORS:
        raise ValueError(f"indicator '{indicator_key}' is not whitelisted: {list(WB_INDICATORS)}")
    code = WB_INDICATORS[indicator_key]
    cp = _cache_path(f"wb_{code}_{year_from}_{year_to}.json")
    if cp.exists() and not refresh:
        raw = json.loads(cp.read_text())
    else:
        url = (f"{_WB_BASE}/country/all/indicator/{code}"
               f"?format=json&per_page=20000&date={year_from}:{year_to}")
        raw = _http_json(url)
        if not (isinstance(raw, list) and len(raw) > 1 and isinstance(raw[1], list)):
            raise IOError(f"WB indicator response malformed for {code}; not caching")
        pages = raw[0].get("pages", 1)
        for p in range(2, pages + 1):
            more = _http_json(url + f"&page={p}")
            raw[1].extend(more[1])
        _atomic_write_text(cp, json.dumps(raw))
    meta = {
        "indicator": code,
        "snapshot": raw[0].get("lastupdated", "unknown"),
        "fetched": date.today().isoformat(),
    }
    if verify_vintage:
        pin_or_verify_snapshot(f"worldbank:{code}:{year_from}:{year_to}", meta["snapshot"])
    rows = [
        {"iso3": e["countryiso3code"], "country": e["country"]["value"],
         "year": int(e["date"]), "value": e["value"]}
        for e in raw[1]
        if e["value"] is not None and e["countryiso3code"] and str(e["date"]).isdigit()
    ]
    return pd.DataFrame(rows), meta


def fetch_holidays(country_names: list[str], years: list[int]) -> tuple[pd.DataFrame, dict]:
    """Holiday calendar via the `holidays` package (offline, deterministic —
    no cache needed; the package version pins the snapshot).

    Returns (df[country, date, holiday], meta). Unmapped countries are reported
    in meta['unmatched'], never guessed.
    """
    import holidays as _hol

    mapping, unmatched = resolve_iso3(country_names)
    table = fetch_country_map()
    iso2_of_iso3 = {r["iso3"]: r["iso2"] for _, r in table.iterrows()}
    iso2_by_name = {name: iso2_of_iso3.get(iso3, "") for name, iso3 in mapping.items()}

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
        "snapshot": f"holidays=={_hol.__version__}",
        "years": years,
        "unmatched": unmatched,
        "fetched": date.today().isoformat(),
    }
    return pd.DataFrame(rows), meta


def fetch_ecb_fx(currencies: list[str], start: str, end: str,
                 refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Monthly ECB reference rates (currency per EUR).

    start/end: "YYYY-MM". Returns (df[currency, month, value], meta) where
    `month` is an integer year*12+month ordinal — join it with
    merge_period_safe(freq="M", ext_time_unit="month_ordinal").
    Responses are validated before caching; empty result sets raise.
    """
    import csv as _csv
    import io

    frames, latest = [], None
    for cur in currencies:
        if cur not in ECB_FX_CURRENCIES:
            raise ValueError(f"currency '{cur}' is not whitelisted: {sorted(ECB_FX_CURRENCIES)}")
        cp = _cache_path(f"ecb_fx_{cur}_{start}_{end}.csv")
        if cp.exists() and not refresh:
            text = cp.read_text()
        else:
            url = (f"https://data-api.ecb.europa.eu/service/data/EXR/M.{cur}.EUR.SP00.A"
                   f"?format=csvdata&startPeriod={start}&endPeriod={end}")
            text = _http_text(url, timeout=60)
            # validate BEFORE caching — ECB returns 200 with empty body for empty sets
            probe = list(_csv.DictReader(io.StringIO(text)))
            if not probe or "OBS_VALUE" not in probe[0] or "TIME_PERIOD" not in probe[0]:
                raise IOError(f"ECB response for {cur} {start}..{end} empty/malformed; not caching")
            _atomic_write_text(cp, text)
        for row in _csv.DictReader(io.StringIO(text)):
            y, m = row["TIME_PERIOD"].split("-")
            frames.append({"currency": cur, "month": int(y) * 12 + int(m),
                           "value": float(row["OBS_VALUE"])})
            latest = max(latest or row["TIME_PERIOD"], row["TIME_PERIOD"])
    meta = {"source": "ECB SDMX EXR (reference rate vs EUR, monthly)",
            "snapshot": f"data through {latest}",
            "range": [start, end], "fetched": date.today().isoformat()}
    return pd.DataFrame(frames), meta


def fetch_owid_covid(metrics: list[str], allow_download: bool = False,
                     refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """OWID COVID compact dataset (country x date).

    The upstream file is ~178 MB and is cached once under cache/. To avoid
    surprise downloads (tests, CI), the first fetch requires allow_download=True.
    Download is atomic and size-verified — a partial file can never be cached.
    Returns (df[iso3, country, date, <metrics...>], meta).
    """
    for m in metrics:
        if m not in OWID_COVID_METRICS:
            raise ValueError(f"metric '{m}' is not whitelisted: {sorted(OWID_COVID_METRICS)}")
    cp = _cache_path("owid_covid_compact.csv")
    meta_p = _cache_path("owid_covid_compact.meta.json")
    if not cp.exists() or refresh:
        if not allow_download:
            raise FileNotFoundError(
                "OWID COVID cache missing; call with allow_download=True to fetch (~178 MB, one-time)")
        logger.warning("downloading OWID COVID compact (~178 MB, one-time)…")
        last_modified = _download_file(OWID_COVID_URL, cp)
        _atomic_write_text(meta_p, json.dumps({"last_modified": last_modified}))
    upstream = json.loads(meta_p.read_text()).get("last_modified") if meta_p.exists() else None
    usecols = ["country", "date", "code"] + list(metrics)
    df = pd.read_csv(cp, usecols=lambda c: c in set(usecols))
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise IOError(f"OWID schema drift: requested metrics missing from file: {missing} "
                      f"(found: {sorted(df.columns)})")
    if "code" in df.columns:
        df = df.rename(columns={"code": "iso3"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["iso3"].notna() & (df["iso3"].str.len() == 3)]  # drop aggregates (OWID_*)
    meta = {"source": OWID_COVID_URL, "metrics": metrics,
            "snapshot": upstream or f"data through {df['date'].max().date().isoformat()}",
            "cache_bytes": cp.stat().st_size, "fetched": date.today().isoformat()}
    return df.reset_index(drop=True), meta
