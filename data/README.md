# Data

```
data/
├── ohlcv_data.csv                 # coursework — daily OHLCV
├── primary_signals.csv            # coursework — daily primary signals
└── external/
    ├── macro_xasset.csv           # external — cleaned macro / cross-asset panel
    └── macro_xasset_raw.xlsx      # external — raw source for macro_xasset.csv
```

The two coursework files are downloaded from Insendi under *Coursework*
without modification. The `external/` files are added for feature engineering
and are committed to the repository so the pipeline is fully reproducible.

The released window covers all dates up to **30 June 2022**. The final six
months (July–December 2022) are the hidden test set used by the markers when
they re-run the code.

## Coursework files

### `ohlcv_data.csv`

Daily OHLCV history for all eleven instruments. One row per
`(instrument, date)`.

| Column          | Description                                     |
| --------------- | ----------------------------------------------- |
| `date`          | Trading date (YYYY-MM-DD).                      |
| `instrument`    | Lowercase ticker (e.g. `cl1s`, `es1s`, `gc1s`). |
| `open`          | Continuous-contract open.                       |
| `high`          | Continuous-contract high.                       |
| `low`           | Continuous-contract low.                        |
| `close`         | Continuous-contract close.                      |
| `volume`        | Daily volume.                                   |
| `open_interest` | Daily open interest.                            |

History starts in 1990 for most instruments. Equity index futures start
later: ES1S in 1997, FESX1S in 1998, NQ1S in 1999.

### `primary_signals.csv`

Daily primary-model signals from January 2020 onwards. One row per `date`,
one column per instrument.

| Column                       | Description                      |
| ---------------------------- | -------------------------------- |
| `date`                       | Trading date (YYYY-MM-DD).       |
| `es1s`, `nq1s`, …, `pl1s`    | Primary signal in `{-1, 0, +1}`. |

Convention: `+1` long, `-1` short, `0` no position.

## External files

### `external/macro_xasset.csv`

Cleaned single-axis daily panel of 21 macro and cross-asset series,
spanning **1990-01-02 to 2022-06-30**. One row per `Date`.

| Group                | Columns                                                                            |
| -------------------- | ---------------------------------------------------------------------------------- |
| Rates                | `2Y_UST`, `10Y_UST`, `10Y_BUND`, `TIPS10Y`, `BE10Y`                                |
| Equity vol           | `VIX`, `VIX3M`, `CBOE_SKEW`                                                        |
| Rates vol            | `MOVE`                                                                             |
| FX                   | `DXY`, `EURUSD`                                                                    |
| Credit spreads       | `HY_OAS`, `IG_OAS`                                                                 |
| Shipping             | `BAL_DRY_INDEX`                                                                    |
| Inventories / stocks | `LME_COPPER_STOCK`, `EIA_CRUDE_STOCK`, `EIA_DIST_STOCK`, `EIA_GASOLINE_STOCK`, `EIA_NG_STOCK` |
| Activity (PMI)       | `US_ISM_MFG_PMI`, `CHINA_PMI_MFG`                                                  |

Date alignment is by calendar day; missing values are left as blanks (no
forward-fill in this file — fill rules are handled inside the feature pipeline).

### `external/macro_xasset_raw.xlsx`

The original multi-axis spreadsheet from which `macro_xasset.csv` was derived.
A single sheet (`Sheet1`) with 22 series; each series occupies a `Date`
column followed by a value column, so columns do not share a date axis.
`GERMANY_PMI_MFG` is present in the raw file but is not carried through to
the cleaned CSV. Kept in the repository for provenance.
