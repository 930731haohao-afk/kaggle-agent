"""Domain-scoped source registry — the vocabulary that may GROW, and the rules for growing it.

Why this exists. The injection layer's trustworthiness rests on the dossier being unable to
reach arbitrary data: it may propose only whitelisted sources, so every joined column has a
stated origin and a stated leakage rule. That closed list is also what has kept the layer
narrow — every whitelisted source is keyed by country and time, which is why the dossier
fires on exactly the country-panel competitions and nothing else. The list is the bottleneck,
not the judgment.

This module replaces the dataset whitelist with a DOMAIN whitelist plus an admission
procedure. A new source may enter the vocabulary only by passing mechanical gates
(external_data/admit_source.py); the domain list is what makes the dangerous class
structurally unreachable rather than merely discouraged.

THE DANGEROUS CLASS, stated plainly. On Kaggle the most valuable "external data" for a given
competition is frequently a dataset another competitor assembled — sometimes containing the
test labels themselves. A temporal leakage test cannot catch that: nothing is dated wrong,
the answer is simply present. Two structural rules exclude it, and they are the reason this
file is a domain list and not a URL allow-list:

  1. The host must be a general-purpose reference publisher (a statistics agency, a standards
     body, a central bank), never a data-sharing platform where competitors publish.
  2. The source must predate the competition and must not be authored by a participant.

Neither rule is a matter of judgment at admission time: (1) is a set membership test, and (2)
is an assertion the proposer must evidence and the admission record keeps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# the domain whitelist
# ---------------------------------------------------------------------------
# Each entry: host suffix -> why this publisher is a general-purpose reference source.
# Adding a host is a deliberate, reviewed act (it widens what the agent can reach); adding a
# DATASET from an already-listed host is what the admission procedure automates.
ALLOWED_DOMAINS: dict[str, str] = {
    "api.worldbank.org":      "World Bank Open Data — official macro statistics",
    "data.worldbank.org":     "World Bank Open Data — landing/metadata",
    "data-api.ecb.europa.eu": "European Central Bank SDMX — official reference rates",
    "sdw-wsrest.ecb.europa.eu": "European Central Bank SDMX (legacy host)",
    "catalog.ourworldindata.org": "Our World in Data — curated public health/economics",
    "ourworldindata.org":     "Our World in Data",
    "www.epo.org":            "European Patent Office — official CPC classification",
    "epo.org":                "European Patent Office",
    "www.uspto.gov":          "US Patent and Trademark Office — official classification data",
    "uspto.gov":              "US Patent and Trademark Office",
    "www.cooperativepatentclassification.org": "Official CPC scheme publisher",
    "cooperativepatentclassification.org": "Official CPC scheme publisher",
    "unstats.un.org":         "UN Statistics Division",
    "ec.europa.eu":           "Eurostat / European Commission open data",
    "www.census.gov":         "US Census Bureau",
    "census.gov":             "US Census Bureau",
    "www.ncei.noaa.gov":      "NOAA climate archives",
    "ncei.noaa.gov":          "NOAA climate archives",
}

# Hosts that are structurally excluded even if someone lists them by mistake. These are
# data-sharing platforms: a file there may have been uploaded by a competitor, so no amount
# of leakage testing makes them safe for this purpose.
FORBIDDEN_DOMAINS: tuple[str, ...] = (
    "kaggle.com", "www.kaggle.com", "storage.googleapis.com",
    "github.com", "raw.githubusercontent.com", "gist.github.com",
    "huggingface.co", "drive.google.com", "dropbox.com", "mega.nz",
    "zenodo.org",          # legitimate archive, but anyone may publish a derived dataset
    "figshare.com", "data.mendeley.com",
)

# ---------------------------------------------------------------------------
# join key classes
# ---------------------------------------------------------------------------
# The class decides which merge function may be used and which leakage rule applies. Making
# it explicit is half the point: until now the key class lived in a prose column of the
# whitelist table and in the dispatcher's hardcoded call to merge_year_safe.
JOIN_KEY_CLASSES: dict[str, str] = {
    "country_year":  "one value per (country, year); as-of merge with a declared lag",
    "country_date":  "one value per (country, date); as-of merge with a declared lag",
    "currency_month": "one value per (currency, month); as-of merge with a declared lag",
    "date":          "one value per date, no entity key (calendars, market sessions)",
    "lookup":        "static attribute table keyed by a code/id; NO time axis, so no "
                     "temporal leakage is possible — the leakage rule is 'static'",
}
# Classes with a time axis must declare a lag; classes without one must not.
TEMPORAL_CLASSES = {"country_year", "country_date", "currency_month", "date"}


@dataclass
class SourceSpec:
    """A proposed (or admitted) source. Every field is required evidence, not documentation.

    `expected_value` is a pre-registration, in the same spirit as
    docs/preregistrations/: a source admitted without a stated expectation can be declared
    useful after the fact whatever it does.
    """
    key: str                        # vocabulary name the dossier will emit, e.g. "cpc:titles"
    url: str
    publisher: str
    join_key_class: str
    join_columns: list[str]         # columns in the competition data used as the key
    value_columns: list[str]        # columns this source contributes
    leakage_rule: str               # "lag=N" for temporal classes, "static" for lookup
    licence: str
    predates_competition: str       # evidence: publication date / version, and the comp date
    author_is_not_participant: str  # evidence for the second structural rule
    expected_value: str             # pre-registered prediction, incl. how it will be judged
    # Column name of the key IN THE FETCHED FRAME, when it differs from the competition's.
    # apply.py reads this from the registry; it was missing from the spec, so a proposal that
    # declared it died with an uncaught TypeError and re-admitting an existing source silently
    # deleted it (2026-08-04 adversarial verification).
    ext_key: str = "code"
    notes: str = ""
    admitted: bool = False
    admission_evidence: dict = field(default_factory=dict)

    def host(self) -> str:
        return (urlparse(self.url).hostname or "").lower()


class SourceRejected(ValueError):
    """A proposed source failed a structural gate. The reason is always stated."""


def check_domain(spec: SourceSpec) -> str:
    """Structural gate 1: the host must be a listed general-purpose reference publisher."""
    host = spec.host()
    if not host:
        raise SourceRejected(f"{spec.key!r}: url {spec.url!r} has no host")
    for bad in FORBIDDEN_DOMAINS:
        if host == bad or host.endswith("." + bad):
            raise SourceRejected(
                f"{spec.key!r}: host {host!r} is a data-sharing platform. A file there may "
                f"have been published by a competition participant and may contain the "
                f"answer; no leakage test can detect that, so the class is excluded "
                f"structurally rather than checked.")
    for good in ALLOWED_DOMAINS:
        if host == good or host.endswith("." + good):
            return ALLOWED_DOMAINS[good]
    raise SourceRejected(
        f"{spec.key!r}: host {host!r} is not in ALLOWED_DOMAINS. Widening the domain list is "
        f"a deliberate review step -- it changes what the agent can reach -- and is not part "
        f"of automatic admission. Known hosts: {sorted(ALLOWED_DOMAINS)}")


def check_key_class(spec: SourceSpec) -> str:
    """Structural gate 2: the join key class must be known, and its leakage rule must match."""
    if spec.join_key_class not in JOIN_KEY_CLASSES:
        raise SourceRejected(
            f"{spec.key!r}: unknown join_key_class {spec.join_key_class!r} "
            f"(known: {sorted(JOIN_KEY_CLASSES)})")
    if not spec.join_columns:
        raise SourceRejected(f"{spec.key!r}: join_columns must name the competition columns "
                             f"used as the key")
    if not spec.value_columns:
        raise SourceRejected(f"{spec.key!r}: value_columns must name what this source adds")
    rule = spec.leakage_rule.strip().lower()
    if spec.join_key_class in TEMPORAL_CLASSES:
        m = re.fullmatch(r"lag=(\d+)", rule)
        if not m:
            raise SourceRejected(
                f"{spec.key!r}: join_key_class {spec.join_key_class!r} has a time axis, so "
                f"leakage_rule must be 'lag=N' (0 allows same-period values, 1 restricts to "
                f"strictly earlier); got {spec.leakage_rule!r}")
        return f"temporal, {rule}"
    if rule != "static":
        raise SourceRejected(
            f"{spec.key!r}: join_key_class {spec.join_key_class!r} has no time axis, so its "
            f"leakage_rule must be exactly 'static'; got {spec.leakage_rule!r}. A lag here "
            f"would imply a temporal guarantee the data cannot carry.")
    return "static, no time axis"


def check_provenance(spec: SourceSpec) -> str:
    """Structural gate 3: the source predates the competition and is not participant-authored.

    Evidence is required as free text because it is a claim about the world, not about the
    data — but it is required, recorded in the admission trace, and empty strings fail. The
    check is that someone stated the evidence, not that a regex approved of it.
    """
    for fname in ("predates_competition", "author_is_not_participant", "licence"):
        val = (getattr(spec, fname) or "").strip()
        if len(val) < 10:
            raise SourceRejected(
                f"{spec.key!r}: {fname} needs stated evidence (got {val!r}). This field is "
                f"the record that the structural rules were considered; an unsupported "
                f"source cannot be admitted on the assumption that someone checked.")

    # expected_value is a PRE-REGISTRATION, so a sentence of intent does not satisfy it:
    # "should help" passes any length check and predicts nothing, which is exactly how a
    # source gets declared useful after the fact whatever it did. Require a magnitude (a
    # number) and a stated way to judge it. This mirrors docs/preregistrations/ discipline.
    exp = (spec.expected_value or "").strip()
    if len(exp) < 40 or not any(ch.isdigit() for ch in exp):
        raise SourceRejected(
            f"{spec.key!r}: expected_value must be a falsifiable pre-registration -- a "
            f"predicted magnitude (a number, even a range) AND how it will be judged. Got "
            f"{exp!r}. Without it the source can be called useful whatever it does.")
    if not any(w in exp.lower() for w in
               ("judg", "test", "compar", "against", "baseline", "arm", "falsif")):
        raise SourceRejected(
            f"{spec.key!r}: expected_value states a magnitude but not how it will be judged "
            f"(name the comparison: a matched no-external arm, the paired test, ...). "
            f"A prediction with no adjudication rule cannot be wrong.")
    return "provenance evidence recorded; expectation is falsifiable"


def structural_gates(spec: SourceSpec) -> dict:
    """Run every gate that needs no network and no data. Raises SourceRejected on the first
    failure, so the caller always learns exactly which rule stopped it."""
    return {
        "domain": check_domain(spec),
        "key_class": check_key_class(spec),
        "provenance": check_provenance(spec),
    }
