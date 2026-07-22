# Batch detrend + z-score standardize 30-min CSVs (simple version)
# - You provide TIME_COL and SIGNAL_COL.
# - Looks for files with "_30min" in the name (change FILE_PATTERN if needed).
# - Writes <original>_detrend_z.csv in the same folder.

import os
import re
import pandas as pd
import numpy as np

# ---------------- CONFIG: set these ----------------
FOLDER     = os.environ.get("DENDRO_DATA_DIR", "cleaned_data")
TIME_COL   = "timestamp"              # you said you have this
SIGNAL_COL = "Cleaned Displacement"    # set your signal column here
FILE_PATTERN = r"_30min.*\.csv$"       # matches "..._30min.csv" or "..._30min_linear.csv"
WINDOW_HOURS = 24                      # rolling mean window for detrending
EDGE_FILL    = False                    # fill edges via time-interp + ffill/bfill
OVERWRITE    = False                    # overwrite outputs if they exist
# ---------------------------------------------------

# -------- your functions (unchanged) ---------------
def detrend_by_rolling_mean(
    df: pd.DataFrame,
    signal_col: str = "value",        # <- we work on the numeric value column
    time_col: str = "time_local",
    window_hours: int = 24,
    out_col: str = "Displacement_detr",
    edge_fill: bool = True,
) -> pd.DataFrame:
    """
    Centered rolling-mean baseline over `window_hours` using a DatetimeIndex.
    Returns df with `baseline_ma` and `out_col`.
    """
    d = df.copy()
    # operate with a DatetimeIndex so time-based window strings work
    x = (d.set_index(time_col)[signal_col]).astype(float)

    ma = x.rolling(f"{window_hours}H", center=True, min_periods=1).mean()
    if edge_fill:
        ma = ma.interpolate(method="time").ffill().bfill()

    out = (x - ma).rename(out_col)

    d["baseline_ma"] = ma.reindex(d.set_index(time_col).index).values
    d[out_col]       = out.reindex(d.set_index(time_col).index).values
    return d

def standardize_inplace(df, detr_col="Displacement_detr"):
    """
    Adds two columns based on `detr_col`:
      - Displacement_01 : min–max to [0,1]
      - Displacement_z  : z-score (population std)
    """
    x = df[detr_col].dropna()
    if x.empty:
        df["Displacement_01"] = 0.0
        df["Displacement_z"]  = 0.0
        return df
    xmin, xmax = x.min(), x.max()
    rng = xmax - xmin
    df["Displacement_01"] = 0.0 if (pd.isna(rng) or rng == 0) else (df[detr_col] - xmin) / rng
    mu, sd = x.mean(), x.std(ddof=0)
    df["Displacement_z"] = 0.0 if (pd.isna(sd) or sd == 0) else (df[detr_col] - mu) / sd
    return df
# ---------------------------------------------------

def make_out_name(name):
    base, ext = os.path.splitext(name)
    return f"{base}_detrend_z{ext or '.csv'}"

def process_one(csv_path):
    df = pd.read_csv(csv_path)

    # Minimal checks
    if TIME_COL not in df.columns:
        raise ValueError(f"{os.path.basename(csv_path)} missing TIME_COL='{TIME_COL}'.")
    if SIGNAL_COL not in df.columns:
        raise ValueError(f"{os.path.basename(csv_path)} missing SIGNAL_COL='{SIGNAL_COL}'.")

    # Clean time/value
    df[TIME_COL]   = pd.to_datetime(df[TIME_COL], errors="coerce", infer_datetime_format=True)
    df[SIGNAL_COL] = pd.to_numeric(df[SIGNAL_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, SIGNAL_COL]).sort_values(TIME_COL)

    # Keep just what we need
    work = df[[TIME_COL, SIGNAL_COL]].rename(columns={SIGNAL_COL: "value"})

    # Detrend (rolling 24h mean by default)
    detr = detrend_by_rolling_mean(
        work, signal_col="value", time_col=TIME_COL,
        window_hours=WINDOW_HOURS, out_col="Displacement_detr", edge_fill=EDGE_FILL
    )

    # Standardize (z + [0,1])
    out = standardize_inplace(detr, detr_col="Displacement_detr")

    # Reorder + restore signal name for clarity
    out = out[[TIME_COL, "baseline_ma", "value", "Displacement_detr", "Displacement_01", "Displacement_z"]].copy()
    out = out.rename(columns={"value": SIGNAL_COL})

    return out

def main():
    # Pick matching files
    patt = re.compile(FILE_PATTERN, re.I)
    files = [f for f in os.listdir(FOLDER) if patt.search(f)]
    if not files:
        print("No files matched FILE_PATTERN in:", FOLDER)
        return

    print(f"Processing {len(files)} file(s):\n")
    for fname in sorted(files):
        in_path = os.path.join(FOLDER, fname)
        out_path = os.path.join(FOLDER, make_out_name(fname))
        if (not OVERWRITE) and os.path.exists(out_path):
            print(f"SKIP (exists): {fname}")
            continue
        try:
            out = process_one(in_path)
            out.to_csv(out_path, index=False)
            print(f"OK → {os.path.basename(out_path)}  ({len(out)} rows)")
        except Exception as e:
            print(f"FAIL {fname}: {e}")
            
    

    # ---------- pick one example output and plot ----------
    example = None
    for fname in sorted(files):
        out_candidate = os.path.join(FOLDER, make_out_name(fname))
        if os.path.exists(out_candidate):
            example = out_candidate
            break

    if example is not None:
        print(f"\nPlotting example from: {os.path.basename(example)}")
        plot_example(
            csv_path=example,
            weather_xlsx_path=(
                "<path to raw data>"
                "Blueberry Trial 2 2024/2024 Blue Lys Weather.xlsx"
            ),font_family="Arial", base_font_size=15
        )
    else:
        print("\nNo output files found to use as an example.")

    print("\nDone.")
    
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.patches as patches

def plot_example(
    csv_path,
    weather_xlsx_path,
    time_col=TIME_COL,
    signal_col=SIGNAL_COL,
    font_family="Arial",     # <--- new
    base_font_size=13,       # <--- new
):
    """
    Two-panel example:

      TOP:
        - Original cleaned signal + 24h rolling baseline + detrended (µm)
        - Daily precipitation from Excel weather file (blocks, mm)

      BOTTOM:
        - Z-score (Displacement_z)
    """

    # ---- unified font setup ----
    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 2,
        "axes.labelsize": base_font_size + 1,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
    })

    # ---------- dendrometer processed file ----------
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"{time_col} not found in example file.")
    for col in ("baseline_ma", "Displacement_detr", "Displacement_z"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in example file.")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    t    = df[time_col]
    raw  = df[signal_col]
    base = df["baseline_ma"]
    detr = df["Displacement_detr"]
    zsig = df["Displacement_z"]

    # ---------- WEATHER EXCEL FILE (.xlsx) ----------
    w = pd.read_excel(
        weather_xlsx_path,
        skiprows=[0, 2, 3]   # metadata, units, qualifiers
    )

    if "TIMESTAMP" not in w.columns or "Precipitation" not in w.columns:
        raise ValueError(
            "Weather Excel file must contain columns 'TIMESTAMP' and 'Precipitation'."
        )

    w["TIMESTAMP"] = pd.to_datetime(w["TIMESTAMP"], errors="coerce")
    w["Precipitation"] = pd.to_numeric(w["Precipitation"], errors="coerce")
    w = w.dropna(subset=["TIMESTAMP", "Precipitation"]).sort_values("TIMESTAMP")

    # restrict to overlapping window with dendro data
    tmin, tmax = t.min(), t.max()
    w = w[(w["TIMESTAMP"] >= tmin) & (w["TIMESTAMP"] <= tmax)]

    # Make daily precipitation by summing mm per calendar day
    w["date"] = w["TIMESTAMP"].dt.date
    daily = w.groupby("date")["Precipitation"].sum().reset_index()
    
    # Convert back to timestamps for plotting (start of day)
    daily["day_start"] = pd.to_datetime(daily["date"])
    daily["day_end"] = daily["day_start"] + pd.Timedelta(days=1)

    # ---------- build figure ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # TOP: raw + baseline + detrended + precip
    ax1.plot(t, raw,  label="Original Cleaned",      color="tab:blue",  alpha=0.7)
    ax1.plot(t, base, label="24h Rolling Baseline",  color="tab:orange", linewidth=1.8)
    ax1.plot(t, detr, label="Detrended Signal",      color="tab:green",  linewidth=1.2)

    ax1.set_ylabel("Displacement (µm)")
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    
    for _, row in daily.iterrows():
        rect = patches.Rectangle(
            (row["day_start"], 0),               # x0, y0
            row["day_end"] - row["day_start"],   # width (1 day)
            row["Precipitation"],                # height = daily total
            color="#1abc9c",                     # teal rainfall color
            alpha=0.35,                          # soft background shading
            linewidth=0
        )
        ax1b.add_patch(rect)
    ax1b.set_ylabel("Daily Precip (mm)")

    # Collect existing line handles (raw, baseline, detrended)
    lines = ax1.get_lines()
    
    # Create a dummy patch for precipitation legend
    precip_patch = patches.Patch(
        color="#1abc9c", alpha=0.35, label="Daily Precipitation (mm)"
    )
    
    ax1b.set_ylim(0, daily["Precipitation"].max() * 1.2)
    
    # Build legend: lines + precip rectangle
    ax1.legend(
        list(lines) + [precip_patch],
        [ln.get_label() for ln in lines] + ["Daily Precipitation (mm)"],
        loc="upper left"
    )

    # BOTTOM: z-score only
    ax2.plot(
        t,
        zsig,
        label="Z-score (Displacement_z)",
        color="tab:purple",
        linestyle="--",
        linewidth=1.2,
    )
    ax2.set_ylabel("Z-score")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
if __name__ == "__main__":
    main()
    
#%%
'''
# Batch detrend + z-score standardize 30-min CSVs (simple version)
import os
import re
import pandas as pd
import numpy as np

# ---------------- CONFIG: set these ----------------
FOLDER     = os.environ.get("DENDRO_DATA_DIR", "cleaned_data")
TIME_COL   = "timestamp"               # present in file
SIGNAL_COL = "Cleaned Displacement"    # your signal column
FILE_PATTERN = r"_30min.*\.csv$"       # matches "..._30min.csv" or "..._30min_linear.csv"
WINDOW_HOURS = 24                      # rolling mean window for detrending
EDGE_FILL    = False                   # fill edges via time-interp + ffill/bfill
OVERWRITE    = True                    # overwrite outputs if they exist
# ---------------------------------------------------

def detrend_by_rolling_mean(
    df: pd.DataFrame,
    signal_col: str = "value",
    time_col: str = "time_local",
    window_hours: int = 24,
    out_col: str = "Displacement_detr",
    edge_fill: bool = True,
) -> pd.DataFrame:
    """
    Centered rolling-mean baseline over `window_hours` using a DatetimeIndex.
    Returns df with `baseline_ma` and `out_col`.
    """
    d = df.copy()
    x = (d.set_index(time_col)[signal_col]).astype(float)

    ma = x.rolling(f"{window_hours}H", center=True, min_periods=1).mean()
    if edge_fill:
        ma = ma.interpolate(method="time").ffill().bfill()

    out = (x - ma).rename(out_col)

    # put results back on the original row order of d
    idx = d.set_index(time_col).index
    d["baseline_ma"] = ma.reindex(idx).values
    d[out_col]       = out.reindex(idx).values
    return d

def standardize_inplace(df, detr_col="Displacement_detr"):
    """
    Adds two columns based on `detr_col`:
      - Displacement_01 : min–max to [0,1]
      - Displacement_z  : z-score (population std)
    """
    x = df[detr_col].dropna()
    if x.empty:
        df["Displacement_01"] = 0.0
        df["Displacement_z"]  = 0.0
        return df
    xmin, xmax = x.min(), x.max()
    rng = xmax - xmin
    df["Displacement_01"] = 0.0 if (pd.isna(rng) or rng == 0) else (df[detr_col] - xmin) / rng
    mu, sd = x.mean(), x.std(ddof=0)
    df["Displacement_z"] = 0.0 if (pd.isna(sd) or sd == 0) else (df[detr_col] - mu) / sd
    return df

def make_out_name(name):
    base, ext = os.path.splitext(name)
    return f"{base}_detrend_z{ext or '.csv'}"

def process_one(csv_path):
    # 1) Load full 30-min file (we'll preserve all non-signal cols from here)
    df0 = pd.read_csv(csv_path)

    # Minimal checks
    if TIME_COL not in df0.columns:
        raise ValueError(f"{os.path.basename(csv_path)} missing TIME_COL='{TIME_COL}'.")
    if SIGNAL_COL not in df0.columns:
        raise ValueError(f"{os.path.basename(csv_path)} missing SIGNAL_COL='{SIGNAL_COL}'.")

    # 2) Clean time/signal only; leave all other columns untouched
    df0[TIME_COL] = pd.to_datetime(df0[TIME_COL], errors="coerce")
    df0[SIGNAL_COL] = pd.to_numeric(df0[SIGNAL_COL], errors="coerce")

    # We only drop rows where time/signal are NA (keeps every other column intact)
    df_clean = df0.dropna(subset=[TIME_COL, SIGNAL_COL]).sort_values(TIME_COL)

    # 3) Compute detrend/z on a tiny working frame, then merge results back
    work = df_clean[[TIME_COL, SIGNAL_COL]].rename(columns={SIGNAL_COL: "value"})
    detr = detrend_by_rolling_mean(
        work, signal_col="value", time_col=TIME_COL,
        window_hours=WINDOW_HOURS, out_col="Displacement_detr", edge_fill=EDGE_FILL
    )
    out = standardize_inplace(detr, detr_col="Displacement_detr")

    # Columns to append
    add_cols = out[[TIME_COL, "baseline_ma", "Displacement_detr", "Displacement_01", "Displacement_z"]]

    # 4) Merge new columns onto the cleaned original rows so ALL other columns are preserved
    result = df_clean.merge(add_cols, on=TIME_COL, how="left")

    # (Optional) reorder: keep original columns first, then the new ones
    new_cols = ["baseline_ma", "Displacement_detr", "Displacement_01", "Displacement_z"]
    base_cols = [c for c in df_clean.columns if c not in new_cols]
    result = result[base_cols + new_cols]

    return result

def main():
    patt = re.compile(FILE_PATTERN, re.I)
    files = [f for f in os.listdir(FOLDER) if patt.search(f)]
    if not files:
        print("No files matched FILE_PATTERN in:", FOLDER)
        return

    print(f"Processing {len(files)} file(s):\n")
    for fname in sorted(files):
        in_path  = os.path.join(FOLDER, fname)
        out_path = os.path.join(FOLDER, make_out_name(fname))
        if (not OVERWRITE) and os.path.exists(out_path):
            print(f"SKIP (exists): {fname}")
            continue
        try:
            res = process_one(in_path)
            res.to_csv(out_path, index=False)
            print(f"OK → {os.path.basename(out_path)}  ({len(res)} rows)")
        except Exception as e:
            print(f"FAIL {fname}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()'''
    
#%%
'''
# Overlay original 30-min data with new detrended/standardized data
# - No saving. Just a plot.
# - Assumes both files have time column 'time_local'.
# - Plots original 'Cleaned Displacement' vs the chosen series from the new file.

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------- EDIT THESE ----------
FOLDER     = os.environ.get("DENDRO_DATA_DIR", "cleaned_data")
ORIG_FILE   = "Blueberry_Trial_1_2023_BB1_complete_cleaned_30min_linear.csv"        # e.g. "Blueberry_..._30min_linear.csv"
NEW_FILE    = "Blueberry_Trial_1_2023_BB1_complete_cleaned_30min_linear_detrend_z.csv"  # the new file you produced
TIME_COL    = "timestamp"
SIGNAL_COL  = "Cleaned Displacement"                # column in the original file
NEW_SERIES  = "Displacement_z"                   # or "Displacement_z" / "Displacement_01"
# -------------------------------

def read_with_dt(path, time_col):
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise SystemExit(f"'{time_col}' not in {os.path.basename(path)}.\nAvailable: {list(df.columns)}")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.rename(columns={time_col: "dt"})
    return df

def main():
    p_orig = os.path.join(FOLDER, ORIG_FILE)
    p_new  = os.path.join(FOLDER, NEW_FILE)

    df_o = read_with_dt(p_orig, TIME_COL)
    df_n = read_with_dt(p_new,  TIME_COL)

    if SIGNAL_COL not in df_o.columns:
        raise SystemExit(f"'{SIGNAL_COL}' not in original file.\nAvailable: {list(df_o.columns)}")
    if NEW_SERIES not in df_n.columns:
        raise SystemExit(f"'{NEW_SERIES}' not in new file.\nAvailable: {list(df_n.columns)}")

    # make numeric and drop NaNs
    df_o[SIGNAL_COL] = pd.to_numeric(df_o[SIGNAL_COL], errors="coerce")
    df_n[NEW_SERIES] = pd.to_numeric(df_n[NEW_SERIES], errors="coerce")
    df_o = df_o.dropna(subset=[SIGNAL_COL])
    df_n = df_n.dropna(subset=[NEW_SERIES])

    # overlap window
    start = max(df_o["dt"].min(), df_n["dt"].min())
    end   = min(df_o["dt"].max(), df_n["dt"].max())
    if not (start < end):
        raise SystemExit("No overlapping timestamps between files.")

    o = df_o[(df_o["dt"] >= start) & (df_o["dt"] <= end)]
    n = df_n[(df_n["dt"] >= start) & (df_n["dt"] <= end)]

    # Decide if we need a second y-axis (if comparing z-scores / 0-1 vs raw)
    dual_axis = NEW_SERIES.lower().endswith("_z") or NEW_SERIES.lower().endswith("_01")

    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.plot(o["dt"], o[SIGNAL_COL], label=f"Original ({SIGNAL_COL})", linewidth=0.9, alpha=0.75)

    if dual_axis:
        ax2 = ax.twinx()
        ax2.plot(n["dt"], n[NEW_SERIES], label=f"New ({NEW_SERIES})", color="tab:orange", linewidth=1.2)
        ax2.set_ylabel(NEW_SERIES)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.plot(n["dt"], n[NEW_SERIES], label=f"New ({NEW_SERIES})", linewidth=1.2)
        ax.legend(loc="best")

    ax.set_title(f"{SIGNAL_COL} vs {NEW_SERIES} (overlap {start} to {end})")
    ax.set_xlabel("Time")
    ax.set_ylabel(SIGNAL_COL if not dual_axis else SIGNAL_COL)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()'''