"""Self-test for the external-data module: fetch, resolve, join, leakage guard."""
import pandas as pd
from sources import fetch_worldbank, fetch_holidays, resolve_iso3
from join import merge_year_safe, merge_holiday_flags, LeakageError

def main():
    toy = pd.DataFrame({
        "country": ["Sweden", "Norway", "Finland", "Atlantis", "Sweden"],
        "year":    [2019, 2020, 2021, 2020, 2021],
        "date":    ["2019-06-06", "2020-12-25", "2021-01-01", "2020-01-01", "2021-11-26"],
    })
    mapping, unmatched = resolve_iso3(toy["country"].unique().tolist())
    print("mapping:", mapping, "| unmatched:", unmatched)
    assert unmatched == ["Atlantis"], "Atlantis must be reported, not guessed"

    gdp, meta = fetch_worldbank("gdp_per_capita", 2015, 2021)
    print("WB rows:", len(gdp), "| snapshot:", meta["snapshot"])
    assert len(gdp) > 500

    out, rep = merge_year_safe(toy, gdp, df_country="country", df_year="year",
                               out_col="gdp_pc", mapping=mapping, lag=0, meta=meta)
    print("lag=0 matched:", rep["rows_matched"], "/", rep["rows"], "| unmatched:", rep["unmatched_countries"])
    assert rep["rows_matched"] == 4 and rep["unmatched_countries"] == ["Atlantis"]
    swe21 = out[(out.country == "Sweden") & (out.year == 2021)]["gdp_pc"].iloc[0]
    ref = gdp[(gdp.iso3 == "SWE") & (gdp.year == 2021)]["value"].iloc[0]
    assert abs(swe21 - ref) < 1e-6, "lag=0 must use same-year value"

    out1, rep1 = merge_year_safe(toy, gdp, df_country="country", df_year="year",
                                 out_col="gdp_pc", mapping=mapping, lag=1, meta=meta)
    swe21_l1 = out1[(out1.country == "Sweden") & (out1.year == 2021)]["gdp_pc"].iloc[0]
    ref20 = gdp[(gdp.iso3 == "SWE") & (gdp.year == 2020)]["value"].iloc[0]
    assert abs(swe21_l1 - ref20) < 1e-6, "lag=1 must use previous-year value"
    print("lag semantics: OK (2021 row got 2021 value at lag=0, 2020 value at lag=1)")

    hol, hmeta = fetch_holidays(["Sweden", "Norway", "Finland"], [2019, 2020, 2021])
    print("holiday rows:", len(hol), "| unmatched:", hmeta["unmatched"])
    out2, rep2 = merge_holiday_flags(toy, hol, df_country="country", df_date="date")
    flags = out2.set_index(["country", "date"])["is_holiday"].to_dict()
    assert flags[("Sweden", "2019-06-06")] == 1, "Swedish National Day"
    assert flags[("Norway", "2020-12-25")] == 1, "Christmas"
    print("holiday flags: OK")

    # order preservation: shuffled input must come back in identical row order
    shuf = toy.sample(frac=1, random_state=7).reset_index(drop=True)
    out3, _ = merge_year_safe(shuf, gdp, df_country="country", df_year="year",
                              out_col="gdp_pc", mapping=mapping, lag=0, meta=meta)
    assert list(out3["country"]) == list(shuf["country"]), "row order must be preserved"
    assert list(out3["year"]) == list(shuf["year"]), "row order must be preserved"
    print("order preservation: OK")

    # ECB FX: monthly join semantics
    from sources import fetch_ecb_fx
    from join import merge_period_safe
    fx, fmeta = fetch_ecb_fx(["SEK", "NOK"], "2020-11", "2021-02")
    assert len(fx) == 8, f"expected 2 currencies x 4 months, got {len(fx)}"
    dfm = pd.DataFrame({"cur": ["SEK", "SEK", "Atlantis$"],
                        "when": ["2021-01-15", "2021-02-10", "2021-01-01"]})
    outm, repm = merge_period_safe(dfm, fx, freq="M", df_key="cur", df_time="when",
                                   ext_key="currency", ext_time="month",
                                   ext_time_unit="month_ordinal",
                                   value_cols=["value"], lag=0)
    jan = fx[(fx.currency == "SEK") & (fx.month == 2021 * 12 + 1)]["value"].iloc[0]
    assert abs(outm["value"].iloc[0] - jan) < 1e-9, "lag=0 month join must use same-month rate"
    outm1, _ = merge_period_safe(dfm, fx, freq="M", df_key="cur", df_time="when",
                                 ext_key="currency", ext_time="month",
                                 ext_time_unit="month_ordinal",
                                 value_cols=["value"], lag=1)
    dec = fx[(fx.currency == "SEK") & (fx.month == 2020 * 12 + 12)]["value"].iloc[0]
    assert abs(outm1["value"].iloc[0] - dec) < 1e-9, "lag=1 month join must use previous month"
    assert repm["keys_not_in_ext"] == ["Atlantis$"], repm
    print("ECB FX + monthly lag semantics: OK")

    # daily-period join on synthetic data (no network)
    ext_d = pd.DataFrame({"iso3": ["SWE"] * 3,
                          "date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                          "cases": [1.0, 2.0, 3.0]})
    dfd = pd.DataFrame({"country": ["Sweden"], "date": ["2021-01-02"]})
    outd, _ = merge_period_safe(dfd, ext_d, freq="D", df_key="country", df_time="date",
                                ext_key="iso3", ext_time="date", value_cols=["cases"],
                                lag=1, mapping={"Sweden": "SWE"})
    assert outd["cases"].iloc[0] == 1.0, "daily lag=1 must use previous day"
    print("daily lag semantics: OK")

    # OWID COVID: only if cache present (no surprise 178MB download in tests)
    from sources import fetch_owid_covid, CACHE_DIR
    if (CACHE_DIR / "owid_covid_compact.csv").exists():
        cov, cmeta = fetch_owid_covid(["new_cases_smoothed_per_million"])
        assert {"iso3", "country", "date"}.issubset(cov.columns)
        assert (cov["iso3"].str.len() == 3).all()
        print(f"OWID COVID cache: OK ({len(cov)} rows)")
    else:
        try:
            fetch_owid_covid(["new_cases_smoothed_per_million"])
            raise AssertionError("must refuse download without allow_download")
        except FileNotFoundError:
            print("OWID COVID download guard: OK (cache absent, refused)")

    # ---- adversarial-review regressions ----
    from join import LeakageError, _assert_no_future

    # 1) undeclared numeric time must raise, never guess (epoch/ordinal misparse)
    yr_ext = pd.DataFrame({"iso3": ["SWE"], "year": [2021], "v": [1.0]})
    for freq in ("M", "D"):
        try:
            merge_period_safe(dfd, yr_ext, freq=freq, df_key="country", df_time="date",
                              ext_key="iso3", ext_time="year", value_cols=["v"],
                              mapping={"Sweden": "SWE"})
            raise AssertionError(f"numeric ext_time with freq={freq} must raise TypeError")
        except TypeError:
            pass
    # plain years declared as month ordinals must fail the range check
    try:
        merge_period_safe(dfd, yr_ext, freq="M", df_key="country", df_time="date",
                          ext_key="iso3", ext_time="year", ext_time_unit="month_ordinal",
                          value_cols=["v"], mapping={"Sweden": "SWE"})
        raise AssertionError("year ints as month_ordinal must fail range validation")
    except ValueError:
        pass
    print("numeric-time input contract: OK")

    # 2) finer-grained ext without explicit coarsen must raise; explicit coarsen works
    daily = pd.DataFrame({"iso3": ["SWE"] * 3,
                          "date": ["2021-01-01", "2021-01-15", "2021-01-31"],
                          "cases": [10.0, 50.0, 999.0]})
    row = pd.DataFrame({"country": ["Sweden"], "date": ["2021-01-05"]})
    try:
        merge_period_safe(row, daily, freq="M", df_key="country", df_time="date",
                          ext_key="iso3", ext_time="date", value_cols=["cases"],
                          mapping={"Sweden": "SWE"})
        raise AssertionError("coarsening without explicit choice must raise")
    except ValueError:
        pass
    outc, repc = merge_period_safe(row, daily, freq="M", df_key="country", df_time="date",
                                   ext_key="iso3", ext_time="date", value_cols=["cases"],
                                   mapping={"Sweden": "SWE"}, coarsen="first", lag=0)
    assert outc["cases"].iloc[0] == 10.0 and repc["ext_rows_collapsed"] == 2
    outl, repl = merge_period_safe(row, daily, freq="M", df_key="country", df_time="date",
                                   ext_key="iso3", ext_time="date", value_cols=["cases"],
                                   mapping={"Sweden": "SWE"}, coarsen="last", lag=0)
    assert "warning" in repl, "coarsen=last + lag=0 must carry a within-period warning"
    print("coarsen contract: OK")

    # 3) NaN value rows must not shadow earlier valid values
    nan_ext = pd.DataFrame({"iso3": ["SWE", "SWE"], "date": ["2021-01-01", "2021-01-02"],
                            "v": [7.0, float("nan")]})
    outn, _ = merge_period_safe(pd.DataFrame({"country": ["Sweden"], "date": ["2021-01-02"]}),
                                nan_ext, freq="D", df_key="country", df_time="date",
                                ext_key="iso3", ext_time="date", value_cols=["v"],
                                mapping={"Sweden": "SWE"}, lag=0)
    assert outn["v"].iloc[0] == 7.0, "NaN row must fall back to earlier valid value"
    print("NaN fallback: OK")

    # 4) the leakage invariant itself (direct unit test — unreachable via backward asof)
    fake = pd.DataFrame({"_used": [5], "_row": [3]})
    try:
        _assert_no_future(fake, "_used", "_row", 0, "unit-test")
        raise AssertionError("violating frame must raise LeakageError")
    except LeakageError:
        pass
    print("leakage invariant unit test: OK")

    # 5) alias branch of resolve_iso3
    amap, aun = resolve_iso3(["South Korea", "Czech Republic"])
    assert amap == {"South Korea": "KOR", "Czech Republic": "CZE"} and aun == []
    print("alias resolution: OK")

    # 6) mapping-collision diagnostics
    coll_df = pd.DataFrame({"country": ["A-land", "B-land"], "year": [2021, 2021]})
    coll_ext = pd.DataFrame({"iso3": ["SWE"], "year": [2021], "value": [42.0]})
    _, repk = merge_year_safe(coll_df, coll_ext, df_country="country", df_year="year",
                              out_col="x", mapping={"A-land": "SWE", "B-land": "SWE"}, lag=0)
    assert repk["mapping_collisions"] == {"SWE": ["A-land", "B-land"]}
    print("mapping-collision report: OK")

    print("ALL SELFTESTS PASSED")

if __name__ == "__main__":
    main()
