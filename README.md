# dendrodataprocessing

Processing pipeline for high-frequency dendrometer (stem diameter) records
from zero-friction magnetic dendrometers developed at the Openly Published
Environmental Sensing (OPEnS) Lab, Oregon State University
(https://github.com/OPEnSLab-OSU/Dendrometer).

The code takes raw logger CSVs through outlier cleaning, resampling,
detrending and z-score standardization, producing series that can be
compared across sensors and plants.

---

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `interactive_data_cleaning.py` | Interactive review of each raw file: flags outliers, corrects shifts/jumps/dips, bridges gaps, resamples to 30 min |
| 2 | `resample_30min.py` | Batch resampling of full-resolution cleaned files to 30-min means |
| 3 | `detrend_standardize.py` | Removes growth with a 24-h rolling mean, then z-score standardizes the detrended signal |

Supporting modules used by the above:

| Module | Contents |
|---|---|
| `data_loading.py` | File loading, time parsing, date-window subsetting |
| `data_processing.py` | Outlier detection/correction, resampling, gap handling |
| `data_plotting.py` | Diagnostic and comparison plots |
| `main_example.py` | Minimal worked example: load → clean → resample → plot |

---

## Configuring paths

No absolute paths are hard-coded. Point the scripts at your own data in
either of two ways:

**Environment variable**

```bash
export DENDRO_DATA_DIR=/path/to/your/data
python detrend_standardize.py
```

**Command line** (interactive cleaning only)

```bash
python interactive_data_cleaning.py /path/to/raw_data cleaned_data
python interactive_data_cleaning.py /path/to/raw_data cleaned_data \
       --start-date 2024-08-01 --end-date 2024-10-31
```

If neither is set, the scripts default to `raw_data/` and `cleaned_data/`
in the working directory.

`detrend_standardize.py` and `resample_30min.py` also expose a small
CONFIG block at the top (`TIME_COL`, `SIGNAL_COL`, `FILE_PATTERN`,
`WINDOW_HOURS`) — set these to match your column names.

---

## Expected input

CSVs with, at minimum:

- a timestamp column (default name `timestamp`)
- a displacement column (default name `Cleaned Displacement` after step 1)

Sensor and trial names are parsed from the filename, so a consistent
naming scheme (e.g. `Hazelnut_Trial_2025_CC1_high.csv`) is recommended.

---

## Note on interactive cleaning

`interactive_data_cleaning.py` requires operator judgement at the prompts
(which flagged points to accept, where to bridge gaps). It documents and
reproduces the *procedure* used, but rerunning it will not necessarily
reproduce a previously cleaned file byte-for-byte, since decisions are
made by the user at run time.

---

## Requirements

Python 3.10 or later, with:

```
numpy
pandas
scipy
matplotlib
```

---

## License

See `LICENSE.txt`.
