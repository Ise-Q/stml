# Data

Place the two coursework CSVs in this directory before running the pipeline.
Both are available on Insendi under *Coursework*.

## `ohlcv_data.csv`

Daily OHLCV history for all eleven instruments. One row per
`(instrument, date)`.

| Column          | Description                                             |
| --------------- | ------------------------------------------------------- |
| `date`          | Trading date (YYYY-MM-DD).                              |
| `instrument`    | Lowercase ticker (e.g. `cl1s`, `es1s`, `gc1s`).         |
| `open`          | Continuous-contract open.                               |
| `high`          | Continuous-contract high.                               |
| `low`           | Continuous-contract low.                                |
| `close`         | Continuous-contract close.                              |
| `volume`        | Daily volume.                                           |
| `open_interest` | Daily open interest.                                    |

History starts in 1990 for most instruments. Equity index futures start
later: ES1S in 1997, FESX1S in 1998, NQ1S in 1999.

## `primary_signals.csv`

Daily primary-model signals from January 2020 onwards. One row per `date`,
one column per instrument.

| Column                       | Description                                |
| ---------------------------- | ------------------------------------------ |
| `date`                       | Trading date (YYYY-MM-DD).                 |
| `es1s`, `nq1s`, …, `pl1s`    | Primary signal in `{-1, 0, +1}`.           |

Convention: `+1` long, `-1` short, `0` no position.

## Coverage

The released window covers all dates up to **30 June 2022**. The final six
months (July–December 2022) are a hidden test set used by the markers when
they re-run the code.
