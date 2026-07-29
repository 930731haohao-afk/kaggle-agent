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
    print("ALL SELFTESTS PASSED")

if __name__ == "__main__":
    main()
