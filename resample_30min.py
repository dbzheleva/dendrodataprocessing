# Batch resample CSVs to 30-min using linear interpolation (NumPy)
# - Looks for *_fullres.csv in FOLDER
# - Detects the time column automatically (or set TIME_COL)
# - Resamples ALL numeric columns onto a strict 30-min timeline
# - Writes <original> with "_30min_linear.csv" suffix next to the source file

import os
import re
import numpy as np
import pandas as pd

# -------------------- CONFIG --------------------
FOLDER   = os.environ.get("DENDRO_DATA_DIR", "cleaned_data")
TIME_COL = 'time_local'   # e.g. "timestamp"; leave None to auto-detect by name
FILE_RE  = r"_fullres\.csv$"  # which files to process; change to r"\.csv$" to process all
MAX_GAP  = "2H"     # do not interpolate across gaps larger than this; None = allow all gaps
OVERWRITE = False    # if False, skip files that already have an output
# ------------------------------------------------

def detect_time_col(df):
    """Pick a column that looks like time/date/timestamp; fallback to first column."""
    for c in df.columns:
        if re.search(r"(time|date|stamp)", str(c), re.I):
            return c
    return df.columns[0]

def to_datetime_sorted(df, tcol):
    """Return a copy with a parsed 'dt' column (datetime), sorted, and rows with bad time removed."""
    out = df.copy()
    out["dt"] = pd.to_datetime(out[tcol], errors="coerce", infer_datetime_format=True)
    out = out.dropna(subset=["dt"]).sort_values("dt")
    return out

def resample_30min_linear_numpy(s, start=None, end=None, max_gap="2H"):
    """
    Linear interpolation using numpy.interp on a 30-min grid.
    Does not extrapolate; optionally prevents bridging long gaps.
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a DatetimeIndex")

    s = s.sort_index()
    x = s.index.view("int64") / 1e9            # seconds
    y = s.values.astype(float)

    # define window
    start = (pd.to_datetime(start) if start is not None else s.index.min()).floor("30min")
    end   = (pd.to_datetime(end)   if end   is not None else s.index.max()).ceil("30min")
    grid = pd.date_range(start, end, freq="30min")
    gx = grid.view("int64") / 1e9

    m = np.isfinite(y)
    if m.sum() < 2:
        return pd.Series(np.nan, index=grid, name=s.name)

    yi = np.interp(gx, x[m], y[m], left=np.nan, right=np.nan)

    if max_gap is not None:
        max_gap_sec = pd.Timedelta(max_gap).total_seconds()
        xm = x[m]
        pos = np.searchsorted(xm, gx)
        left_pos  = np.clip(pos - 1, 0, xm.size - 1)
        right_pos = np.clip(pos,     0, xm.size - 1)
        left_dist  = np.where(pos > 0,        np.abs(gx - xm[left_pos]),  np.inf)
        right_dist = np.where(pos < xm.size,  np.abs(xm[right_pos] - gx), np.inf)
        too_far = (left_dist > max_gap_sec) | (right_dist > max_gap_sec)
        yi[too_far] = np.nan

    return pd.Series(yi, index=grid, name=s.name)

def resample_dataframe_30min_linear(df, time_col=None, max_gap="2H", start=None, end=None):
    """
    Resample ALL numeric columns in df to a strict 30-min grid via linear interpolation.
    Returns a DataFrame indexed by the 30-min timeline.
    """
    tcol = time_col or detect_time_col(df)
    dfx = to_datetime_sorted(df, tcol)

    # Use only numeric columns (drop 'dt' and time col)
    numeric_cols = [c for c in dfx.columns if c not in (tcol, "dt") and pd.api.types.is_numeric_dtype(dfx[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns to resample.")

    # Global window for this file (so all columns share the same grid)
    smin = dfx["dt"].min()
    smax = dfx["dt"].max()
    start = (pd.to_datetime(start) if start is not None else smin)
    end   = (pd.to_datetime(end)   if end   is not None else smax)

    # Build each resampled series and combine
    out = None
    for col in numeric_cols:
        s = pd.Series(pd.to_numeric(dfx[col], errors="coerce").values, index=dfx["dt"].values, name=col)
        s = s.groupby(level=0).mean()  # collapse duplicate timestamps if any
        r = resample_30min_linear_numpy(s, start=start, end=end, max_gap=max_gap)
        if out is None:
            out = pd.DataFrame(index=r.index)
        out[col] = r

    out.index.name = "timestamp"
    # Optionally drop columns that remained all-NaN after interpolation
    out = out.dropna(axis=1, how="all")
    return out

def make_out_name(in_name):
    """Turn '..._fullres.csv' into '..._30min_linear.csv', else append suffix."""
    if re.search(r"_fullres\.csv$", in_name, re.I):
        return re.sub(r"_fullres\.csv$", "_30min_linear.csv", in_name, flags=re.I)
    base, ext = os.path.splitext(in_name)
    return f"{base}_30min_linear{ext or '.csv'}"

def main():
    files = [f for f in os.listdir(FOLDER) if f.lower().endswith(".csv")]
    if FILE_RE:
        files = [f for f in files if re.search(FILE_RE, f, re.I)]

    if not files:
        print("No CSVs matched in:", FOLDER)
        return

    print(f"Found {len(files)} file(s). Processing…\n")
    for fname in files:
        in_path = os.path.join(FOLDER, fname)
        out_name = make_out_name(fname)
        out_path = os.path.join(FOLDER, out_name)

        if not OVERWRITE and os.path.exists(out_path):
            print(f"SKIP (exists): {out_name}")
            continue

        try:
            df = pd.read_csv(in_path)
            res = resample_dataframe_30min_linear(df, time_col=TIME_COL, max_gap=MAX_GAP)
            if res.empty:
                print(f"EMPTY after resample (no numeric cols or all NaN): {fname}")
                continue
            res.to_csv(out_path)
            print(f"OK  → {out_name}   ({len(res)} rows)")
        except Exception as e:
            print(f"FAIL: {fname}  → {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
    
