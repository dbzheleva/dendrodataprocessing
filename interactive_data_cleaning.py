#!/usr/bin/env python3

"""
interactive_cleaning.py

Script to interactively clean each raw data file:
  - Load raw data
  - Extract device name from the file name (anything before first underscore)
  - Plot raw vs. cleaned
  - Ask user to accept or adjust the cleaning parameter
  - Once accepted, save cleaned data to disk
  - If user says "no", they can still choose to change threshold or skip saving
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Import your existing functions
from data_loading import load_and_prepare_data_time_limit

from data_plotting import plot_original_vs_cleaned2

def extract_trial_name(file_path):
    """
    Returns the name of the parent folder for the given file path.
    e.g. ".../Blueberry Trial 1 2023/BB1_complete.csv" -> "Blueberry Trial 1 2023"
    """
    return os.path.basename(os.path.dirname(file_path))

def extract_device_name_from_filename(file_path):
    """
    Extracts device name by splitting the base filename on the FIRST underscore.
    e.g. "BB2_complete.csv" -> "BB2"
         "BB2_something_else.csv" -> "BB2"
    If no underscore found, returns the entire name (minus extension).
    """
    base_name = os.path.basename(file_path)         # "BB2_complete.csv"
    without_ext = os.path.splitext(base_name)[0]    # "BB2_complete"
    parts = without_ext.split('_', 1)               # split once on underscore
    if len(parts) > 1:
        # e.g. ["BB2", "complete"]
        return parts[0]
    else:
        # e.g. no underscore
        return without_ext
    
import numpy as np

def _short_run_mask_from_bool(series_bool, run_threshold=2):
    """Return a boolean Series that is True for outlier positions that belong to runs < run_threshold."""
    arr = series_bool.to_numpy()
    short = np.zeros_like(arr, dtype=bool)
    run_start = None
    for i in range(len(arr)+1):
        if i < len(arr) and arr[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_end = i - 1
                length = run_end - run_start + 1
                if length < run_threshold:
                    short[run_start:run_end+1] = True
                run_start = None
    return pd.Series(short, index=series_bool.index)

def preview_flags_and_clean(
    df_raw,
    displacement_col='um',
    multiplier=3,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2,
    slope_threshold=None,
    slope_method='q3',
    max_gap_minutes=30,
    device_name=None
):
    """
    Mirrors the cleaning pipeline to:
      1) compute & return df_cleaned (same columns as in your flow),
      2) produce a preview plot of RAW data with flags for each removal/flagging stage:
         - alignment/pressed/pos
         - single-step jumps (> daily range) after gap-collapse
         - IQR mask (flagged; not necessarily removed)
         - short runs removed
         - final daily-range removals (on cleaned displacement)
    """
    prefix = f"[{device_name}] " if device_name else ""

    # ---- Start from sorted RAW; keep a stable key to map back to raw rows
    df0 = df_raw.copy().sort_values('time_local').reset_index(drop=True)
    df0['time_local'] = pd.to_datetime(df0['time_local'])
    df0['__idx0__'] = df0.index

    # ---- Step A: alignment/pressed/pos checks (REMOVED)
    align_mask = pd.Series(False, index=df0.index)
    if alignment_col in df0.columns:
        align_mask = df0[alignment_col].isin(['Red'])

    pressed_mask = pd.Series(False, index=df0.index)
    if pressed_col in df0.columns:
        pressed_mask = df0[pressed_col] == True

    if pos_raw_col in df0.columns and pos_avg_col in df0.columns:
        df0[pos_raw_col] = pd.to_numeric(df0[pos_raw_col], errors='coerce')
        df0[pos_avg_col] = pd.to_numeric(df0[pos_avg_col], errors='coerce')
        pos_diff = (df0[pos_raw_col] - df0[pos_avg_col]).abs()
    else:
        pos_diff = pd.Series(0.0, index=df0.index)

    ap_mask = align_mask | pressed_mask | (pos_diff > pos_diff_threshold)
    idx_ap = set(df0.loc[ap_mask, '__idx0__'])

    df1 = df0.loc[~ap_mask].copy().reset_index(drop=True)

    # ---- Step B: daily range (capped 50 µm)
    tmp = df1.copy()
    tmp['date'] = tmp['time_local'].dt.date
    daily_ranges = tmp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min()).clip(upper=50)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"{prefix}[DEBUG] daily_range => {avg_daily_range:.3f} µm")

    # ---- Step 0: collapse gaps (no deletion)
    def collapse_large_gaps_no_delete(df_local, disp_col, daily_range_cap, gap_minutes=30):
        df_local = df_local.sort_values('time_local').copy()
        df_local.reset_index(drop=True, inplace=True)
        n = len(df_local)
        for i in range(n-1):
            t_i = df_local.loc[i, 'time_local']
            t_next = df_local.loc[i+1, 'time_local']
            dt_min = (t_next - t_i).total_seconds()/60.0
            if dt_min > gap_minutes:
                diff_val = df_local.loc[i+1, disp_col] - df_local.loc[i, disp_col]
                if abs(diff_val) > daily_range_cap:
                    df_local.loc[i+1:, disp_col] -= diff_val
                    print(f"{prefix}[DEBUG] Collapsed gap> {gap_minutes} min at index={i}, "
                          f"time gap={dt_min:.1f}, shift={diff_val:.3f} µm.")
        return df_local

    df1c = collapse_large_gaps_no_delete(df1, displacement_col, avg_daily_range, gap_minutes=max_gap_minutes)

    # ---- Step 1: single-step jump removals (> daily range)  (REMOVED)
    df1c['Difference'] = df1c[displacement_col].diff()
    big_jump_mask = df1c['Difference'].abs() > avg_daily_range
    idx_big = set(df1c.loc[big_jump_mask, '__idx0__'])

    df2 = df1c.loc[~big_jump_mask].copy().reset_index(drop=True)

    # ---- Step 2: slope threshold
    if slope_col is None:
        slope_col = displacement_col

    # windowed slope stats (08:00–08:30, 19:00–19:30)
    df2['time_of_day'] = df2['time_local'].dt.time
    mask1 = (df2['time_of_day'] >= pd.to_datetime("08:00").time()) & (df2['time_of_day'] < pd.to_datetime("08:30").time())
    mask2 = (df2['time_of_day'] >= pd.to_datetime("19:00").time()) & (df2['time_of_day'] < pd.to_datetime("19:30").time())
    df_sl = df2[mask1 | mask2].copy().sort_values('time_local')
    small_thr = None
    if slope_threshold is None:
        if df_sl.empty:
            small_thr = 2.0
            print(f"{prefix}[WARN] fallback=2.0 (no slope window data)")
        else:
            diffs = df_sl[slope_col].diff().abs().dropna()
            if diffs.empty:
                small_thr = 2.0
                print(f"{prefix}[WARN] fallback=2.0 (no slope diffs)")
            else:
                med, mean, q3 = diffs.median(), diffs.mean(), diffs.quantile(0.75)
                if slope_method.lower() == 'median':
                    small_thr = med
                elif slope_method.lower() == 'mean':
                    small_thr = mean
                else:
                    small_thr = q3
    else:
        small_thr = float(slope_threshold)
    print(f"{prefix}[DEBUG] slope threshold => {('user=' if slope_threshold is not None else '')}{small_thr:.3f}")

    # ---- Step 3: IQR mask on large diffs (FLAGGED; not necessarily removed)
    df2['Difference'] = df2[displacement_col].diff()
    df2['Diff_Abs'] = df2['Difference'].abs()
    large_mask = df2['Diff_Abs'] > small_thr
    large_vals = df2.loc[large_mask, 'Diff_Abs']
    idx_iqr = set()
    idx_short_removed = set()

    if large_vals.empty:
        # nothing flagged => no IQR/short-run
        df3 = df2.copy()
        outlier_keep_mask_df3 = pd.Series(False, index=df3.index)
    else:
        Q1, Q3 = large_vals.quantile(0.25), large_vals.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - multiplier * IQR)
        upper_bound = Q3 + multiplier * IQR
        print(f"{prefix}[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")

        outlier_mask = df2['Diff_Abs'] > upper_bound
        idx_iqr = set(df2.loc[outlier_mask, '__idx0__'])  # flagged by IQR (for plotting)

        # ---- Step 4: remove short outlier runs (REMOVED)
        short_mask = _short_run_mask_from_bool(outlier_mask, run_threshold=run_threshold)
        idx_short_removed = set(df2.loc[outlier_mask & short_mask, '__idx0__'])

        # materialize the drop for the remainder of the pipeline
        df3 = df2.loc[~(outlier_mask & short_mask)].copy().reset_index(drop=True)

        # Outliers that REMAIN (after pruning) — for baseline shift
        keep_keys = set(df2.loc[outlier_mask & ~short_mask, '__idx0__'])
        outlier_keep_mask_df3 = df3['__idx0__'].isin(keep_keys)

    # ---- Step 5: baseline shift => Cleaned Displacement
    df3['Difference'] = df3[displacement_col].diff()
    df3['Adjustment'] = 0.0
    total_shift = 0.0
    keep_mask_vals = outlier_keep_mask_df3.to_numpy()
    for i in range(len(df3)):
        if keep_mask_vals[i]:
            if i == 0 and remove_first_outlier:
                df3.loc[i, 'Difference'] = np.nan
            else:
                jump = df3.loc[i, 'Difference']
                total_shift += jump
        df3.loc[i, 'Adjustment'] = total_shift
    df3['Cleaned Displacement'] = df3[displacement_col] - df3['Adjustment']

    # ---- Step 6: final daily-range pass (REMOVED on cleaned)
    df3['Diff_Final'] = df3['Cleaned Displacement'].diff().abs()
    final_mask = df3['Diff_Final'] > avg_daily_range
    idx_final = set(df3.loc[final_mask, '__idx0__'])

    # ---- Make the preview plot: RAW with all flags
    df_plot = df0  # raw, sorted, idx0 aligned
    t = df_plot['time_local']; y = df_plot[displacement_col]

    plt.figure(figsize=(12, 4))
    plt.plot(t, y, linewidth=1)
    # category markers (no explicit colors)
    cat_specs = [
        ("Alignment/Pressed/Pos", idx_ap, 'x'),
        ("Single-step > daily range", idx_big, 'o'),
        ("IQR mask (flagged)", idx_iqr, 's'),
        (f"Short runs < {run_threshold} (removed)", idx_short_removed, '^'),
        ("Final daily-range (removed)", idx_final, 'D'),
    ]
    for label, idxset, marker in cat_specs:
        if idxset:
            m = df_plot['__idx0__'].isin(idxset)
            plt.scatter(t[m], y[m], s=22, marker=marker, label=label)
    ttl = f"{device_name} — Raw with removal flags" if device_name else "Raw with removal flags"
    plt.title(ttl)
    plt.xlabel("Time"); plt.ylabel(displacement_col)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    # ---- Build df_cleaned comparable to your main function's output
    # Note: df3 mirrors your post-pruning, post-shift state.
    df_cleaned = df3.copy()
    return df_cleaned

def plot_raw_with_flags(
    df_raw,
    df_cleaned,
    displacement_col='um',
    multiplier=3,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    slope_col=None,
    run_threshold=2,
    slope_threshold=None,
    slope_method='q3',
    max_gap_minutes=30,
    device_name=None,
):
    """Plot RAW with markers for each category of removal/flagging, without mutating data."""
    prefix = f"[{device_name}] " if device_name else ""

    # Base copy with stable id
    df0 = df_raw.copy().sort_values('time_local').reset_index(drop=True)
    df0['time_local'] = pd.to_datetime(df0['time_local'])
    df0['__idx0__'] = df0.index

    # --- Alignment / Pressed / Pos mismatches (REMOVED in cleaner)
    align_mask = df0[alignment_col].isin(['Red']) if alignment_col in df0.columns else False
    pressed_mask = (df0[pressed_col] == True) if pressed_col in df0.columns else False
    if pos_raw_col in df0.columns and pos_avg_col in df0.columns:
        r = pd.to_numeric(df0[pos_raw_col], errors='coerce')
        a = pd.to_numeric(df0[pos_avg_col], errors='coerce')
        pos_diff = (r - a).abs()
    else:
        pos_diff = pd.Series(0.0, index=df0.index)
    ap_mask = (align_mask.astype(bool)) | (pressed_mask.astype(bool)) | (pos_diff > pos_diff_threshold)
    idx_ap = set(df0.loc[ap_mask, '__idx0__'])
    if len(idx_ap):
        print(f"{prefix}[DEBUG] preview: alignment/pressed/pos => {len(idx_ap)} rows")

    # --- Compute avg daily range (capped) on data *after* integrity skip (for context)
    df1 = df0.loc[~ap_mask].copy()
    tmp = df1.copy()
    tmp['date'] = tmp['time_local'].dt.date
    daily_ranges = tmp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min()).clip(upper=50)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"{prefix}[DEBUG] daily_range => {avg_daily_range:.3f} µm")

    # --- Gap collapse (no deletion) – we only apply to a working copy for masks downstream
    def _collapse(df_local, disp_col, cap, gap_minutes):
        df_local = df_local.sort_values('time_local').copy()
        df_local.reset_index(drop=True, inplace=True)
        for i in range(len(df_local)-1):
            dt = (df_local.loc[i+1, 'time_local'] - df_local.loc[i, 'time_local']).total_seconds()/60.0
            if dt > gap_minutes:
                diff_val = df_local.loc[i+1, disp_col] - df_local.loc[i, disp_col]
                if abs(diff_val) > cap:
                    df_local.loc[i+1:, disp_col] -= diff_val
        return df_local

    df1c = _collapse(df1, displacement_col, avg_daily_range, max_gap_minutes)

    # --- Big single-step jumps (ANCHORS in cleaner; NOT dropped here)
    df1c['Difference'] = df1c[displacement_col].diff()
    big_jump_mask = df1c['Difference'].abs() > avg_daily_range
    idx_big = set(df1c.loc[big_jump_mask, '__idx0__'])
    print(f"{prefix}[DEBUG] preview: big single-step anchors => {len(idx_big)}")

    # --- Slope threshold (same rule as cleaner)
    if slope_col is None:
        slope_col = displacement_col
    df1c['time_of_day'] = df1c['time_local'].dt.time
    m1 = (df1c['time_of_day'] >= pd.to_datetime("08:00").time()) & (df1c['time_of_day'] < pd.to_datetime("08:30").time())
    m2 = (df1c['time_of_day'] >= pd.to_datetime("19:00").time()) & (df1c['time_of_day'] < pd.to_datetime("19:30").time())
    df_sl = df1c[m1 | m2].sort_values('time_local')
    if slope_threshold is None:
        if df_sl.empty or df_sl[slope_col].diff().abs().dropna().empty:
            small_thr = 2.0
            print(f"{prefix}[WARN] preview: fallback slope=2.0")
        else:
            diffs = df_sl[slope_col].diff().abs().dropna()
            if slope_method.lower() == 'median': small_thr = diffs.median()
            elif slope_method.lower() == 'mean': small_thr = diffs.mean()
            else: small_thr = diffs.quantile(0.75)
    else:
        small_thr = float(slope_threshold)
    print(f"{prefix}[DEBUG] preview: slope threshold => {small_thr:.3f}")

    # --- IQR mask on large diffs (FLAGGED)
    df1c['Diff_Abs'] = df1c['Difference'].abs()
    large_mask = df1c['Diff_Abs'] > small_thr
    large_vals = df1c.loc[large_mask, 'Diff_Abs']
    idx_iqr, idx_short_removed = set(), set()
    if not large_vals.empty:
        Q1, Q3 = large_vals.quantile(0.25), large_vals.quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + multiplier * IQR
        iqr_mask = df1c['Diff_Abs'] > upper
        idx_iqr = set(df1c.loc[iqr_mask, '__idx0__'])
        # short-run on those flags (no deletion; just mark)
        short_mask = _short_run_mask_from_bool(iqr_mask, run_threshold=run_threshold)
        idx_short_removed = set(df1c.loc[iqr_mask & short_mask, '__idx0__'])
        print(f"{prefix}[DEBUG] preview: IQR flags => {len(idx_iqr)}, short runs removed => {len(idx_short_removed)}")
    else:
        print(f"{prefix}[DEBUG] preview: no large diffs above slope => 0 IQR flags")

    # --- Final daily-range removals computed from df_cleaned
    df_fin = df_cleaned.copy().sort_values('time_local').reset_index(drop=True)
    df_fin['Diff_Final'] = df_fin['Cleaned Displacement'].diff().abs()
    final_mask = df_fin['Diff_Final'] > avg_daily_range
    # map final times back to RAW by timestamp
    final_times = set(df_fin.loc[final_mask, 'time_local'])
    idx_final = set(df0.loc[df0['time_local'].isin(final_times), '__idx0__'])
    if len(idx_final):
        print(f"{prefix}[DEBUG] preview: final daily-range removals => {len(idx_final)}")

    # --- Plot RAW with category markers
    t, y = df0['time_local'], df0[displacement_col]
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, linewidth=1, label="Raw")
    cat_specs = [
        ("Alignment/Pressed/Pos", idx_ap, 'x'),
        ("Single-step > daily range (anchors)", idx_big, 'o'),
        ("IQR mask (flagged)", idx_iqr, 's'),
        (f"Short runs < {run_threshold} (removed)", idx_short_removed, '^'),
        ("Final daily-range (removed)", idx_final, 'D'),
    ]
    for label, idxset, marker in cat_specs:
        if idxset:
            m = df0['__idx0__'].isin(idxset)
            plt.scatter(t[m], y[m], s=22, marker=marker, label=label)
    ttl = f"{device_name} — Raw with removal flags" if device_name else "Raw with removal flags"
    plt.title(ttl)
    plt.xlabel("Time"); plt.ylabel(displacement_col)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    
def resample_to_30min(
    df,
    time_col="time_local",
    rule="30T",
    agg="median",
    device_name=None
):
    """
    Resample to fixed 30-min bins *before* cleaning.
    - Numeric cols (um, pos_raw, pos_avg): median (or mean if agg='mean')
    - Pressed?: True if any True in the bin
    - Alignment: 'Red' if any Red in the bin; else last non-null value
    - All other columns: last

    Returns a df with a regularized timeline (bins with all NaNs dropped).
    """
    prefix = f"[{device_name}] " if device_name else ""

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)

    # choose numeric aggregator
    num_agg = np.median if agg == "median" else np.mean

    num_cols = [c for c in ["um", "pos_raw", "pos_avg"] if c in df.columns]
    agg_map = {c: num_agg for c in num_cols}

    if "Pressed?" in df.columns:
        # any True in the bin -> True
        def _pressed_any(s):
            try:
                return bool(np.nanmax(s.astype(float)))
            except Exception:
                return bool(s.dropna().astype(bool).any())
        agg_map["Pressed?"] = _pressed_any

    if "Alignment" in df.columns:
        def _align_bin(s):
            s = s.dropna().astype(str)
            if (s == "Red").any():
                return "Red"
            return s.iloc[-1] if len(s) else np.nan
        agg_map["Alignment"] = _align_bin

    # everything else: keep last value seen in the bin
    for c in df.columns:
        if c not in agg_map:
            agg_map[c] = "last"

    res = df.resample(rule).agg(agg_map)

    # drop rows where *all* numeric targets are NaN (keeps real gaps)
    if num_cols:
        res = res.dropna(subset=num_cols, how="all")

    res = res.reset_index().rename(columns={time_col: "time_local"})
    print(f"{prefix}[DEBUG] resample -> 30 min: {len(df)} → {len(res)} rows")
    return res

def fill_gaps_30min(
    df_local,
    disp_col="um",
    time_col="time_local",
    strategy="tod",          # 'tod' (time-of-day) or 'linear'
    freq="30T",
    max_bins_to_fill=None,   # e.g. 48 (= 1 day) to skip very long outages; None = no cap
    device_name=None
):
    """
    Fill missing 30-min bins.
    - 'tod': use typical increment for that HH:MM (median of diffs), then
             scale so the run ends exactly at the next observed value.
    - 'linear': pandas time interpolation.

    Returns a frame on the full 30-min grid with disp_col filled; other columns
    are reindexed (left as NaN for bins we synthesize).
    """
    import numpy as np
    prefix = f"[{device_name}] " if device_name else ""

    df_local = df_local.sort_values(time_col).copy()
    df_local[time_col] = pd.to_datetime(df_local[time_col])

    # Build the full 30-min grid over the observed span
    full_idx = pd.date_range(df_local[time_col].min(), df_local[time_col].max(), freq=freq)

    # Reindex whole frame; we'll only *fill* disp_col
    df_full = df_local.set_index(time_col).reindex(full_idx)
    s = df_full[disp_col]

    # Locate NaN runs
    isn = s.isna()
    runs = []
    i = 0
    while i < len(s):
        if isn.iloc[i]:
            j = i
            while j < len(s) and isn.iloc[j]:
                j += 1
            runs.append((i, j))  # [i, j) missing
            i = j
        else:
            i += 1

    filled_bins = 0

    if strategy == "linear":
        s = s.interpolate(method="time", limit_direction="both")
        filled_bins = sum(j - i for i, j in runs)
    else:
        # --- time-of-day increments (median of diffs per HH:MM) ---
        tod_labels = pd.Index(full_idx.strftime("%H:%M"))
        diffs = s.diff()
        typical = diffs.groupby(tod_labels).median()

        s_filled = s.copy()
        for (i, j) in runs:
            run_len = j - i
            if max_bins_to_fill is not None and run_len > max_bins_to_fill:
                continue  # skip very long outages

            prev_val = s_filled.iloc[i - 1] if i > 0 else np.nan
            next_val = s_filled.iloc[j] if j < len(s_filled) else np.nan

            # typical increments for each slot in the run
            incs = [float(typical.get(tod_labels[k], 0.0)) for k in range(i, j)]
            total_pred = float(np.nansum(incs)) if len(incs) else 0.0

            if not np.isnan(prev_val):
                if not np.isnan(next_val) and run_len > 0:
                    # scale increments so we land exactly on next_val
                    needed = next_val - prev_val
                    if abs(total_pred) < 1e-12:
                        incs = [needed / run_len] * run_len
                    else:
                        scale = needed / total_pred
                        incs = [d * scale for d in incs]

                # forward fill
                v = prev_val
                for k, d in enumerate(incs):
                    v = v + d
                    s_filled.iloc[i + k] = v

            elif not np.isnan(next_val):
                # backfill if we only have the right boundary
                v = next_val
                for k in range(run_len - 1, -1, -1):
                    v = v - incs[k]
                    s_filled.iloc[i + k] = v
            # else: no boundaries—leave NaN

            filled_bins += run_len - s_filled.iloc[i:j].isna().sum()

        s = s_filled

    df_full[disp_col] = s
    df_full = df_full.reset_index().rename(columns={"index": time_col})
    print(f"{prefix}[DEBUG] gap-fill ({strategy}) => runs={len(runs)}, filled_bins={int(filled_bins)}")
    return df_full
    
'''def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=3,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2,
    slope_threshold=None,
    slope_method='q3',  # 'median', 'mean', or 'q3' if slope_threshold is not given
    max_gap_minutes=30,  # If gap>30, we consider it a "break in data"
    device_name=None     # New: device/sensor name for debug statements
):
    """
    Identifies and adjusts outliers in the displacement data. Gaps are collapsed by
    shifting subsequent data (no row deletion). Large single-step jumps are now kept
    as *anchors* to force baseline shifts rather than being dropped up front.

    Steps:
      0) Collapse large time gaps without deleting rows.
      1) Remove alignment/pressed/pos-based outlier rows.
      2) Compute capped average daily range (<=50 µm).
      3) Mark big single-step jumps (> daily range) as *shift anchors* (no deletion).
      4) Compute or accept slope threshold (median/mean/Q3).
      5) IQR-based outlier detection on diffs; remove short outlier runs.
      6) Baseline shift using (IQR outliers ∪ anchors) => 'Cleaned Displacement'.
      7) Final pass vs daily range on cleaned series.
    """
    prefix = f"[{device_name}] " if device_name else ""

    # -------------------------------------------------------
    # Helper: collapse large time gaps without dropping rows
    # -------------------------------------------------------
    def collapse_large_gaps_no_delete(df_local, disp_col, daily_range_cap, gap_minutes=30):
        df_local = df_local.sort_values('time_local').copy()
        df_local.reset_index(drop=True, inplace=True)

        n = len(df_local)
        for i in range(n-1):
            t_i = df_local.loc[i, 'time_local']
            t_next = df_local.loc[i+1, 'time_local']
            dt_min = (t_next - t_i).total_seconds()/60.0
            if dt_min > gap_minutes:
                # big time gap => check displacement difference
                diff_val = df_local.loc[i+1, disp_col] - df_local.loc[i, disp_col]
                if abs(diff_val) > daily_range_cap:
                    # SHIFT subsequent data so next merges at the same displacement
                    shift_amount = diff_val
                    df_local.loc[i+1:, disp_col] -= shift_amount
                    print(f"{prefix}[DEBUG] Collapsed gap> {gap_minutes} min at index={i}, "
                          f"time gap={dt_min:.1f}, shift={shift_amount:.3f} µm.")
        return df_local

    # -------------------------------------------------------
    # Helper: slope stats
    # -------------------------------------------------------
    def get_slope_stats(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time

        # Time windows 8:00–8:30, 19:00–19:30
        mask1_start = pd.to_datetime("08:00").time()
        mask1_end   = pd.to_datetime("08:30").time()
        mask1 = (df_local['time_of_day'] >= mask1_start) & (df_local['time_of_day'] < mask1_end)

        mask2_start = pd.to_datetime("19:00").time()
        mask2_end   = pd.to_datetime("19:30").time()
        mask2 = (df_local['time_of_day'] >= mask2_start) & (df_local['time_of_day'] < mask2_end)

        df_filtered = df_local[mask1 | mask2].sort_values('time_local').copy()
        if df_filtered.empty:
            print(f"{prefix}[DEBUG] No data in the time windows => cannot compute slope stats.")
            return None, None, None

        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()
        if slope_series.empty:
            print(f"{prefix}[DEBUG] slope_series empty => returning None.")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)
        print(f"{prefix}[DEBUG] Slope stats => median={median_val:.3f}, mean={mean_val:.3f}, Q3={q3_val:.3f}")
        return median_val, mean_val, q3_val

    # -------------------------------------------------------
    # Helper: remove short outlier runs
    # -------------------------------------------------------
    def remove_short_outlier_runs(df_local, outlier_mask, disp_col, upper_bound, run_threshold):
        outlier_mask = outlier_mask.reset_index(drop=True)
        df_local = df_local.reset_index(drop=True)

        short_run_mask = pd.Series(False, index=outlier_mask.index)
        is_outlier = outlier_mask.values
        n = len(is_outlier)

        run_start = None
        for i in range(n):
            if is_outlier[i] and run_start is None:
                run_start = i
            elif not is_outlier[i] and run_start is not None:
                run_end = i - 1
                length = run_end - run_start + 1
                if length < run_threshold:
                    short_run_mask.iloc[run_start:run_end+1] = True
                run_start = None

        # tail run
        if run_start is not None:
            run_end = n - 1
            length = run_end - run_start + 1
            if length < run_threshold:
                short_run_mask.iloc[run_start:run_end+1] = True

        # Drop short-run outliers
        df_local.loc[short_run_mask, disp_col] = np.nan
        pre_drop = len(df_local)
        df_local.dropna(subset=[disp_col], inplace=True)
        post_drop = len(df_local)
        removed_count = pre_drop - post_drop
        df_local.reset_index(drop=True, inplace=True)

        df_local['Difference'] = df_local[disp_col].diff()
        df_local['Diff_Abs']   = df_local['Difference'].abs()
        new_outlier_mask = (df_local['Diff_Abs'] > upper_bound)

        print(f"{prefix}[DEBUG] Removed {removed_count} row(s) from short runs (< {run_threshold}).")
        print(f"{prefix}[DEBUG] After short-run removal, {new_outlier_mask.sum()} row(s) remain flagged.")
        return df_local, new_outlier_mask

    # -------------------------------------------------------
    # Main logic
    # -------------------------------------------------------
    df = df.copy().sort_values('time_local').reset_index(drop=True)

    # Step 1: Remove alignment/pressed/pos-based rows
    alignment_mask = pd.Series(False, index=df.index)
    if alignment_col in df.columns:
        alignment_mask = df[alignment_col].isin(['Red'])

    pressed_mask = pd.Series(False, index=df.index)
    if pressed_col in df.columns:
        pressed_mask = df[pressed_col] == True

    if pos_raw_col in df.columns and pos_avg_col in df.columns:
        df[pos_raw_col] = pd.to_numeric(df[pos_raw_col], errors='coerce')
        df[pos_avg_col] = pd.to_numeric(df[pos_avg_col], errors='coerce')
        df['pos_diff'] = (df[pos_raw_col] - df[pos_avg_col]).abs()
    else:
        df['pos_diff'] = 0.0

    combined_skip = alignment_mask | pressed_mask | (df['pos_diff'] > pos_diff_threshold)
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"{prefix}[DEBUG] Removing {skip_count} rows due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Keep a stable key for anchor tracking through later pruning
    df['__key'] = df.index

    # Step 2: Compute average daily range (capped at 50 µm)
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())
    daily_ranges = daily_ranges.clip(upper=50)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"{prefix}[DEBUG] daily_range => {avg_daily_range:.3f} µm")

    # Step 0: Collapse large gaps (no deletion)
    df = collapse_large_gaps_no_delete(df, displacement_col, avg_daily_range, gap_minutes=max_gap_minutes)

    # Step 3 (modified): Mark big single-step jumps as *anchors* (DO NOT DROP)
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    big_jump_keys = set(df.loc[big_jump_mask, '__key'])
    if big_jump_mask.any():
        print(f"{prefix}[DEBUG] Marking {big_jump_mask.sum()} big single-step jump(s) as shift anchors (> {avg_daily_range:.3f}).")

    # Step 4: Determine slope threshold
    if slope_col is None:
        slope_col = displacement_col

    if slope_threshold is None:
        median_slope, mean_slope, q3_slope = get_slope_stats(df, slope_col)
        if any(x is None for x in [median_slope, mean_slope, q3_slope]):
            small_slope_threshold = 2.0
            slope_source = "[WARN] fallback=2.0"
            print(f"{prefix}{slope_source}")
        else:
            if slope_method.lower() == 'median':
                small_slope_threshold = median_slope
                slope_source = f"median={median_slope:.3f}"
            elif slope_method.lower() == 'mean':
                small_slope_threshold = mean_slope
                slope_source = f"mean={mean_slope:.3f}"
            else:
                small_slope_threshold = q3_slope
                slope_source = f"q3={q3_slope:.3f}"
    else:
        small_slope_threshold = slope_threshold
        slope_source = f"user={slope_threshold:.3f}"
    print(f"{prefix}[DEBUG] slope threshold => {slope_source}")

    # Step 5: IQR-based outlier detection on large diffs
    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()

    outlier_mask = pd.Series(False, index=df.index)
    large_diff_mask = df['Diff_Abs'] > small_slope_threshold
    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']

    if large_diff_values.empty:
        print(f"{prefix}[DEBUG] No diffs exceed slope => skip IQR outlier detection.")
        # keep df as-is; outlier_mask remains all False for now
    else:
        Q1 = large_diff_values.quantile(0.25)
        Q3 = large_diff_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - multiplier*IQR)
        upper_bound = Q3 + multiplier*IQR
        print(f"{prefix}[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")

        outlier_mask = df['Diff_Abs'] > upper_bound
        outlier_count = outlier_mask.sum()
        print(f"{prefix}[DEBUG] Found {outlier_count} outliers by IQR mask.")

        # Remove short outlier runs (prunes df and returns new mask aligned to pruned df)
        df, outlier_mask = remove_short_outlier_runs(
            df, outlier_mask, displacement_col, upper_bound, run_threshold
        )

    # Force anchors to be shift points (even if IQR didn’t flag them)
    anchor_mask = df['__key'].isin(big_jump_keys)
    outlier_mask = outlier_mask | anchor_mask

    # Step 6: Baseline shift using (IQR outliers ∪ anchors)
    df.reset_index(drop=True, inplace=True)
    outlier_mask = outlier_mask.reset_index(drop=True)

    df['Adjustment'] = 0.0
    total_shift = 0.0
    for i in range(len(df)):
        if outlier_mask[i]:
            if i == 0 and remove_first_outlier:
                df.loc[i, 'Difference'] = np.nan
                print(f"{prefix}[DEBUG] first row => skip shift.")
                continue
            jump = df.loc[i, 'Difference']
            total_shift += jump
        df.loc[i, 'Adjustment'] = total_shift
    df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    # Step 7: Final check for big jumps vs avg_daily_range on cleaned
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Difference'] = df['Cleaned Displacement'].diff()
    final_mask = df['Difference'].abs() > avg_daily_range
    if final_mask.any():
        c = final_mask.sum()
        print(f"{prefix}[DEBUG] final pass => removing {c} rows > daily range.")
        df.loc[final_mask, 'Cleaned Displacement'] = np.nan
        pre = len(df)
        df.dropna(subset=['Cleaned Displacement'], inplace=True)
        post = len(df)
        print(f"{prefix}[DEBUG] dropped {pre-post} final rows.")

    # Tidy up helper column
    if '__key' in df.columns:
        df.drop(columns='__key', inplace=True)

    print(f"{prefix}[DEBUG] done. slope={slope_source}, no-deletion collapse of gaps > {max_gap_minutes} min, anchors used, outliers removed, shift done.")
    return df'''

'''def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2,
    slope_threshold=None,
    slope_method='q3',   # 'median', 'mean', or 'q3' if slope_threshold is not given
    max_gap_minutes=30,  # kept for compatibility; no longer used for gap "collapse"
    device_name=None,    # sensor name for debug
    # --- NEW ---
    gap_fill_strategy='tod',    # 'tod' (time-of-day mean increments) or 'linear' or None
    gap_fill_max_bins=None,     # cap length of NaN runs to fill (e.g. 48 = 1 day)
    resample_freq='30T'         # 30-minute grid
):
    """
    Identifies and adjusts outliers in the displacement data.

    Changes: Instead of collapsing large gaps by shifting, we now *fill*
    gaps on a fixed 30-minute grid either by:
      - 'tod' (time-of-day): use the mean 30-min increment for each HH:MM slot,
        then scale the filled run so it lands exactly on the next observed point;
      - 'linear': pandas time interpolation.

    Steps:
      1) Remove alignment/pressed/pos-based outlier rows.
      0) Gap fill on 30-min grid (no deletion; fill only `displacement_col`).
      2) Compute capped average daily range (<=50 µm).
      3) Mark big single-step jumps (> daily range) as shift anchors (no deletion).
      4) Compute or accept slope threshold (median/mean/Q3).
      5) IQR-based outlier detection on diffs; remove short outlier runs.
      6) Baseline shift using (IQR outliers ∪ anchors) => 'Cleaned Displacement'.
      7) Final pass vs daily range on cleaned series.
    """
    import pandas as pd
    import numpy as np

    prefix = f"[{device_name}] " if device_name else ""

    # -------------------------------------------------------
    # Helper: fill missing 30-min bins for displacement_col
    # -------------------------------------------------------
    def fill_gaps_30min(
        df_local,
        disp_col="um",
        time_col="time_local",
        strategy="tod",
        freq="30T",
        max_bins_to_fill=None
    ):
        """
        Returns df on the full `freq` grid, filling only disp_col.
        Other columns are reindexed and left NaN for synthetic bins.

        strategy:
          - 'tod': use MEAN increment for each HH:MM slot from historical diffs,
                   then scale the run so it lands exactly on the next observed value.
          - 'linear': pandas time interpolation ('time') across NaNs.
        """
        df_local = df_local.sort_values(time_col).copy()
        df_local[time_col] = pd.to_datetime(df_local[time_col])

        # Build full 30-min grid
        full_idx = pd.date_range(df_local[time_col].min(), df_local[time_col].max(), freq=freq)
        df_full = df_local.set_index(time_col).reindex(full_idx)
        s = df_full[disp_col]

        # Find NaN runs
        isn = s.isna()
        runs = []
        i = 0
        while i < len(s):
            if isn.iloc[i]:
                j = i
                while j < len(s) and isn.iloc[j]:
                    j += 1
                runs.append((i, j))  # [i, j) missing
                i = j
            else:
                i += 1

        filled_bins = 0

        if strategy == "linear":
            s_interp = s.interpolate(method="time", limit_direction="both")
            filled_bins = int((s_interp.notna() & s.isna()).sum())
            s = s_interp
        elif strategy == "tod":
            # Typical time-of-day increment = MEAN of diffs per HH:MM slot
            tod_labels = pd.Index(full_idx.strftime("%H:%M"))
            diffs = s.diff()
            typical = diffs.groupby(tod_labels).mean()

            s_filled = s.copy()
            for (a, b) in runs:
                run_len = b - a
                if run_len == 0:
                    continue
                if max_bins_to_fill is not None and run_len > max_bins_to_fill:
                    continue

                prev_val = s_filled.iloc[a - 1] if a > 0 else np.nan
                next_val = s_filled.iloc[b] if b < len(s_filled) else np.nan

                # time-of-day mean increments for each slot in the run
                incs = [float(typical.get(tod_labels[k], 0.0)) for k in range(a, b)]
                total_pred = float(np.nansum(incs)) if len(incs) else 0.0

                if not np.isnan(prev_val):
                    if not np.isnan(next_val):
                        # scale increments so we end at next_val
                        needed = next_val - prev_val
                        if abs(total_pred) < 1e-12:
                            incs = [needed / run_len] * run_len
                        else:
                            scale = needed / total_pred
                            incs = [d * scale for d in incs]
                    # forward fill from prev_val
                    v = prev_val
                    for k, d in enumerate(incs):
                        v = v + d
                        s_filled.iloc[a + k] = v
                elif not np.isnan(next_val):
                    # only right boundary known => backfill
                    v = next_val
                    for k in range(run_len - 1, -1, -1):
                        v = v - incs[k]
                        s_filled.iloc[a + k] = v
                # else: both boundaries NaN => leave as NaN

                filled_bins += int(s_filled.iloc[a:b].notna().sum())

            s = s_filled
        else:
            # No gap filling
            pass

        df_full[disp_col] = s
        df_full = df_full.reset_index().rename(columns={"index": "time_local"})
        print(f"{prefix}[DEBUG] gap-fill ({strategy}) => runs={len(runs)}, filled_bins={filled_bins}")
        return df_full

    # -------------------------------------------------------
    # Helper: slope stats
    # -------------------------------------------------------
    def get_slope_stats(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time
    
        # Morning-only window: 08:00–11:00
        mask_start = pd.to_datetime("08:00").time()
        mask_end   = pd.to_datetime("10:00").time()
        mask = (df_local['time_of_day'] >= mask_start) & (df_local['time_of_day'] < mask_end)
    
        df_filtered = df_local[mask].sort_values('time_local').copy()
        if df_filtered.empty:
            print(f"{prefix}[DEBUG] No data in 08:00–11:00 => cannot compute slope stats.")
            return None, None, None
    
        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()
        if slope_series.empty:
            print(f"{prefix}[DEBUG] slope_series empty in 08:00–11:00 => returning None.")
            return None, None, None
    
        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)
        print(f"{prefix}[DEBUG] Slope stats (08:00–11:00) => median={median_val:.3f}, mean={mean_val:.3f}, Q3={q3_val:.3f}")
        return median_val, mean_val, q3_val

    # -------------------------------------------------------
    # Helper: remove short outlier runs
    # -------------------------------------------------------
    def remove_short_outlier_runs(df_local, outlier_mask, disp_col, upper_bound, run_threshold):
        outlier_mask = outlier_mask.reset_index(drop=True)
        df_local = df_local.reset_index(drop=True)

        short_run_mask = pd.Series(False, index=outlier_mask.index)
        is_outlier = outlier_mask.values
        n = len(is_outlier)

        run_start = None
        for i in range(n):
            if is_outlier[i] and run_start is None:
                run_start = i
            elif not is_outlier[i] and run_start is not None:
                run_end = i - 1
                length = run_end - run_start + 1
                if length < run_threshold:
                    short_run_mask.iloc[run_start:run_end+1] = True
                run_start = None
        if run_start is not None:
            run_end = n - 1
            length = run_end - run_start + 1
            if length < run_threshold:
                short_run_mask.iloc[run_start:run_end+1] = True

        # Drop short-run outliers
        df_local.loc[short_run_mask, disp_col] = np.nan
        pre_drop = len(df_local)
        df_local.dropna(subset=[disp_col], inplace=True)
        post_drop = len(df_local)
        removed_count = pre_drop - post_drop
        df_local.reset_index(drop=True, inplace=True)

        df_local['Difference'] = df_local[disp_col].diff()
        df_local['Diff_Abs']   = df_local['Difference'].abs()
        new_outlier_mask = (df_local['Diff_Abs'] > upper_bound)

        print(f"{prefix}[DEBUG] Removed {removed_count} row(s) from short runs (< {run_threshold}).")
        print(f"{prefix}[DEBUG] After short-run removal, {new_outlier_mask.sum()} row(s) remain flagged.")
        return df_local, new_outlier_mask

    # -------------------------------------------------------
    # Main logic
    # -------------------------------------------------------
    df = df.copy().sort_values('time_local').reset_index(drop=True)

    # Step 1: Remove alignment/pressed/pos-based rows (integrity filter)
    alignment_mask = pd.Series(False, index=df.index)
    if alignment_col in df.columns:
        alignment_mask = df[alignment_col].isin(['Red'])

    pressed_mask = pd.Series(False, index=df.index)
    if pressed_col in df.columns:
        pressed_mask = df[pressed_col] == True

    if pos_raw_col in df.columns and pos_avg_col in df.columns:
        df[pos_raw_col] = pd.to_numeric(df[pos_raw_col], errors='coerce')
        df[pos_avg_col] = pd.to_numeric(df[pos_avg_col], errors='coerce')
        df['pos_diff'] = (df[pos_raw_col] - df[pos_avg_col]).abs()
    else:
        df['pos_diff'] = 0.0

    combined_skip = alignment_mask | pressed_mask | (df['pos_diff'] > pos_diff_threshold)
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"{prefix}[DEBUG] Removing {skip_count} rows due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 0: Gap fill on the 30-min grid (fill only displacement_col)
    if gap_fill_strategy:
        df = fill_gaps_30min(
            df,
            disp_col=displacement_col,
            time_col='time_local',
            strategy=gap_fill_strategy,
            freq=resample_freq,
            max_bins_to_fill=gap_fill_max_bins
        )

    # Keep a stable key for anchor tracking through later pruning
    df['__key'] = df.index

    # Step 2: Compute average daily range (capped at 50 µm)
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['time_local']).dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())
    daily_ranges = daily_ranges.clip(upper=50)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"{prefix}[DEBUG] daily_range => {avg_daily_range:.3f} µm")

    # Step 3: Mark big single-step jumps as *anchors* (DO NOT DROP)
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    big_jump_keys = set(df.loc[big_jump_mask, '__key'])
    if big_jump_mask.any():
        print(f"{prefix}[DEBUG] Marking {big_jump_mask.sum()} big single-step jump(s) as shift anchors (> {avg_daily_range:.3f}).")

    # Step 4: Determine slope threshold
    if slope_col is None:
        slope_col = displacement_col

    if slope_threshold is None:
        median_slope, mean_slope, q3_slope = get_slope_stats(df, slope_col)
        if any(x is None for x in [median_slope, mean_slope, q3_slope]):
            small_slope_threshold = 2.0
            slope_source = "[WARN] fallback=2.0"
            print(f"{prefix}{slope_source}")
        else:
            if slope_method.lower() == 'median':
                small_slope_threshold = median_slope
                slope_source = f"median={median_slope:.3f}"
            elif slope_method.lower() == 'mean':
                small_slope_threshold = mean_slope
                slope_source = f"mean={mean_slope:.3f}"
            else:
                small_slope_threshold = q3_slope
                slope_source = f"q3={q3_slope:.3f}"
    else:
        small_slope_threshold = slope_threshold
        slope_source = f"user={slope_threshold:.3f}"
    print(f"{prefix}[DEBUG] slope threshold => {slope_source}")

    # Step 5: IQR-based outlier detection on large diffs
    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()

    outlier_mask = pd.Series(False, index=df.index)
    large_diff_mask = df['Diff_Abs'] > small_slope_threshold
    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']

    if large_diff_values.empty:
        print(f"{prefix}[DEBUG] No diffs exceed slope => skip IQR outlier detection.")
    else:
        Q1 = large_diff_values.quantile(0.25)
        Q3 = large_diff_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - multiplier*IQR)
        upper_bound = Q3 + multiplier*IQR
        print(f"{prefix}[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")

        outlier_mask = df['Diff_Abs'] > upper_bound
        outlier_count = outlier_mask.sum()
        print(f"{prefix}[DEBUG] Found {outlier_count} outliers by IQR mask.")

        # Remove short outlier runs (prunes df and returns new mask aligned to pruned df)
        df, outlier_mask = remove_short_outlier_runs(
            df, outlier_mask, displacement_col, upper_bound, run_threshold
        )

    # Force anchors to be shift points (even if IQR didn’t flag them)
    anchor_mask = df['__key'].isin(big_jump_keys)
    outlier_mask = outlier_mask | anchor_mask

    # Step 6: Baseline shift using (IQR outliers ∪ anchors)
    df.reset_index(drop=True, inplace=True)
    outlier_mask = outlier_mask.reset_index(drop=True)

    df['Adjustment'] = 0.0
    total_shift = 0.0
    for i in range(len(df)):
        if outlier_mask[i]:
            if i == 0 and remove_first_outlier:
                df.loc[i, 'Difference'] = np.nan
                print(f"{prefix}[DEBUG] first row => skip shift.")
                continue
            jump = df.loc[i, 'Difference']
            total_shift += jump
        df.loc[i, 'Adjustment'] = total_shift
    df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    # Step 7: Final check for big jumps vs avg_daily_range on cleaned
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Difference'] = df['Cleaned Displacement'].diff()
    final_mask = df['Difference'].abs() > avg_daily_range
    if final_mask.any():
        c = final_mask.sum()
        print(f"{prefix}[DEBUG] final pass => removing {c} rows > daily range.")
        df.loc[final_mask, 'Cleaned Displacement'] = np.nan
        pre = len(df)
        df.dropna(subset=['Cleaned Displacement'], inplace=True)
        post = len(df)
        print(f"{prefix}[DEBUG] dropped {pre-post} final rows.")

    # Tidy up helper column
    if '__key' in df.columns:
        df.drop(columns='__key', inplace=True)

    print(f"{prefix}[DEBUG] done. gap_fill={gap_fill_strategy}, slope={slope_source}, anchors used, outliers removed, shift done.")
    return df'''

def _robust_line(t_series, y_series):
    """Robust-ish simple linear fit with a MAD outlier trim."""
    t = pd.to_datetime(t_series).astype("int64") / 1e9  # seconds
    y = pd.to_numeric(y_series, errors="coerce")
    m = y.notna()
    if m.sum() < 2:
        return np.nan, np.nan
    x = t[m].to_numpy()
    z = y[m].to_numpy()
    # first pass
    m1, b1 = np.polyfit(x, z, 1)
    # residual MAD trim
    r = z - (m1 * x + b1)
    mad = np.median(np.abs(r - np.median(r)))
    if mad > 0 and m.sum() >= 3:
        keep = np.abs(r) <= 3 * 1.4826 * mad
        if keep.sum() >= 2:
            x, z = x[keep], z[keep]
            m1, b1 = np.polyfit(x, z, 1)
    return m1, b1

def slope_bridge_gaps(
    df,
    time_col="time_local",
    value_col="um",
    max_gap_minutes=30,
    pre_window="12H",
    post_window="12H",
    blend=0.5,               # 0 = use only pre-gap fit; 1 = use only post-gap fit
):
    """
    Do NOT fill gaps. Leave NaNs. For each time gap (or NaN run), shift the section
    after the gap so that it begins where the pre/post-window slope predicts.

    Returns a copy with 'Adjustment' and 'Cleaned Displacement'.
    """
    d = df.copy().sort_values(time_col).reset_index(drop=True)
    t = pd.to_datetime(d[time_col])
    y = pd.to_numeric(d[value_col], errors="coerce")

    # 1) find big *time* gaps (index of first sample *after* the gap)
    dt_min = t.diff().dt.total_seconds().div(60)
    gap_after_idx = list(np.where(dt_min.values > max_gap_minutes)[0])

    # 2) also treat *NaN runs* as gaps (value gaps, not time gaps)
    isn = y.isna().to_numpy()
    i = 0
    while i < len(isn):
        if isn[i]:
            j = i
            while j < len(isn) and isn[j]:
                j += 1
            # run [i, j) => if we have data on both sides, treat j as the gap 'after' index
            if i > 0 and j < len(isn):
                gap_after_idx.append(j)
            i = j
        else:
            i += 1

    gap_after_idx = sorted(set(gap_after_idx))
    if not gap_after_idx:
        out = d.copy()
        out["Adjustment"] = 0.0
        out["Cleaned Displacement"] = y
        return out

    # cumulative adjustment
    adj = np.zeros(len(d), dtype=float)

    for j in gap_after_idx:
        # guard: need a point before and a point after
        if j <= 0 or j >= len(d):
            continue
        t_before = t.iloc[j - 1]
        t_after  = t.iloc[j]
        y_after  = y.iloc[j]

        # build windows
        pre_mask  = (t >= t_before - pd.Timedelta(pre_window)) & (t <= t_before)
        post_mask = (t >= t_after) & (t <= t_after + pd.Timedelta(post_window))

        m1, b1 = _robust_line(t[pre_mask],  y[pre_mask])
        m2, b2 = _robust_line(t[post_mask], y[post_mask])

        x_after = t_after.value / 1e9
        preds = []
        if not np.isnan(m1): preds.append(m1 * x_after + b1)
        if not np.isnan(m2): preds.append(m2 * x_after + b2)
        if not preds:
            # can't estimate; skip this gap
            continue

        if len(preds) == 2:
            expected = (1 - blend) * preds[0] + blend * preds[1]
        else:
            expected = preds[0]

        if np.isnan(y_after):
            # first point after gap is NaN: skip (no anchor to shift from)
            continue

        # Amount by which the post-gap section must be shifted down/up
        shift = y_after - expected
        adj[j:] += float(shift)

    out = d.copy()
    out["Adjustment"] = adj
    out["Cleaned Displacement"] = y - adj
    return out

def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=10,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2,       # kept for API compatibility (unused)
    slope_threshold=None,
    slope_method='q3',     # 'median', 'mean', or 'q3' if slope_threshold is not given
    max_gap_minutes=30,    # kept for compatibility; not used for gap "collapse"
    device_name=None,      # sensor name for debug
    # gap handling / resampling
    gap_fill_strategy='tod',   # 'tod' (time-of-day mean increments) or 'linear' or None
    gap_fill_max_bins=None,    # cap length of NaN runs to fill (e.g. 48 = 1 day)
    resample_freq='30T', 
    preserve_expected_increment=False,   # set True to leave a natural 30-min change =partial stitch
    expected_stat='mean',                # 'mean' or 'median' for the expected increment        # 30-minute grid
):
    """
    Pipeline:
      1) Integrity filter (alignment/pressed/pos).
      0) Gap fill on 30-min grid (fill only `displacement_col`).
      2) Compute capped average daily range (<=50 µm).
      3) Mark big single-step jumps (> daily range) as anchors (no deletion).
      4) Compute/accept slope threshold using 08:00–11:00 window.
      5) IQR-based outlier detection on diffs (NO short-run pruning).
         => REMOVE those IQR-flagged samples (set NaN) and
            FILL & SCALE them via mean time-of-day increments.
      6) Baseline shift using anchors => 'Cleaned Displacement'.
      7) Final pass vs daily range on cleaned series.
    """
    import pandas as pd
    import numpy as np

    prefix = f"[{device_name}] " if device_name else ""

    # ---------- gap filling helper (used twice: initial + post-IQR) ----------
    def fill_gaps_30min(
        df_local,
        disp_col="um",
        time_col="time_local",
        strategy="tod",
        freq="30T",
        max_bins_to_fill=None
    ):
        """
        Returns df on the full `freq` grid, filling only disp_col.
        Other columns are reindexed and left NaN for synthetic bins.

        strategy:
          - 'tod': use MEAN increment for each HH:MM slot from observed diffs,
                   then scale each NaN-run so it lands exactly on the next observed value.
          - 'linear': pandas time interpolation across NaNs.
        """
        df_local = df_local.sort_values(time_col).copy()
        df_local[time_col] = pd.to_datetime(df_local[time_col])

        # 30-min grid over current span
        full_idx = pd.date_range(df_local[time_col].min(), df_local[time_col].max(), freq=freq)
        df_full = df_local.set_index(time_col).reindex(full_idx)
        s = df_full[disp_col]

        # find NaN runs [a,b)
        isn = s.isna()
        runs = []
        i = 0
        while i < len(s):
            if isn.iloc[i]:
                j = i
                while j < len(s) and isn.iloc[j]:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1

        filled_bins = 0

        if strategy == "linear":
            s_interp = s.interpolate(method="time", limit_direction="both")
            filled_bins = int((s_interp.notna() & s.isna()).sum())
            s = s_interp

        elif strategy == "tod":
            # mean time-of-day increment from observed diffs
            tod_labels = pd.Index(full_idx.strftime("%H:%M"))
            diffs = s.diff()
            typical = diffs.groupby(tod_labels).mean()

            s_filled = s.copy()
            for (a, b) in runs:
                run_len = b - a
                if run_len == 0:
                    continue
                if max_bins_to_fill is not None and run_len > max_bins_to_fill:
                    continue

                prev_val = s_filled.iloc[a - 1] if a > 0 else np.nan
                next_val = s_filled.iloc[b] if b < len(s_filled) else np.nan

                # mean increment for each 30-min slot in the run
                incs = [float(typical.get(tod_labels[k], 0.0)) for k in range(a, b)]
                total_pred = float(np.nansum(incs)) if len(incs) else 0.0

                if not np.isnan(prev_val):
                    if not np.isnan(next_val):
                        needed = next_val - prev_val
                        if abs(total_pred) < 1e-12:
                            incs = [needed / run_len] * run_len
                        else:
                            scale = needed / total_pred
                            incs = [d * scale for d in incs]
                    # forward fill
                    v = prev_val
                    for k, d in enumerate(incs):
                        v = v + d
                        s_filled.iloc[a + k] = v
                elif not np.isnan(next_val):
                    # only right boundary known => backfill
                    v = next_val
                    for k in range(run_len - 1, -1, -1):
                        v = v - incs[k]
                        s_filled.iloc[a + k] = v
                # else both boundaries NaN => leave NaN

                filled_bins += int(s_filled.iloc[a:b].notna().sum())

            s = s_filled

        # write back
        df_full[disp_col] = s
        df_full = df_full.reset_index().rename(columns={"index": "time_local"})
        print(f"{prefix}[DEBUG] gap-fill ({strategy}) => runs={len(runs)}, filled_bins={filled_bins}")
        return df_full

    # ---------- slope stats (08:00–11:00 only) ----------
    def get_slope_stats(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time
        mask_start = pd.to_datetime("08:00").time()
        mask_end   = pd.to_datetime("11:00").time()
        mask = (df_local['time_of_day'] >= mask_start) & (df_local['time_of_day'] < mask_end)

        df_filtered = df_local[mask].sort_values('time_local').copy()
        if df_filtered.empty:
            print(f"{prefix}[DEBUG] No data in 08:00–11:00 => cannot compute slope stats.")
            return None, None, None

        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()
        if slope_series.empty:
            print(f"{prefix}[DEBUG] slope_series empty in 08:00–11:00 => returning None.")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)
        print(f"{prefix}[DEBUG] Slope stats (08:00–11:00) => median={median_val:.3f}, mean={mean_val:.3f}, Q3={q3_val:.3f}")
        return median_val, mean_val, q3_val

    # ---------- main ----------
    df = df.copy().sort_values('time_local').reset_index(drop=True)

    # (1) integrity filter
    alignment_mask = pd.Series(False, index=df.index)
    if alignment_col in df.columns:
        alignment_mask = df[alignment_col].isin(['Red'])
    pressed_mask = pd.Series(False, index=df.index)
    if pressed_col in df.columns:
        pressed_mask = df[pressed_col] == True
    if pos_raw_col in df.columns and pos_avg_col in df.columns:
        df[pos_raw_col] = pd.to_numeric(df[pos_raw_col], errors='coerce')
        df[pos_avg_col] = pd.to_numeric(df[pos_avg_col], errors='coerce')
        df['pos_diff'] = (df[pos_raw_col] - df[pos_avg_col]).abs()
    else:
        df['pos_diff'] = 0.0

    combined_skip = alignment_mask | pressed_mask | (df['pos_diff'] > pos_diff_threshold)
    if combined_skip.any():
        print(f"{prefix}[DEBUG] Removing {combined_skip.sum()} rows due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # (0) initial gap fill on 30-min grid (only displacement_col)
    if gap_fill_strategy:
        df = fill_gaps_30min(
            df,
            disp_col=displacement_col,
            time_col='time_local',
            strategy=gap_fill_strategy,
            freq=resample_freq,
            max_bins_to_fill=gap_fill_max_bins
        )

    # (2) avg daily range (capped 50 µm)
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['time_local']).dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min()).clip(upper=50)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"{prefix}[DEBUG] daily_range => {avg_daily_range:.3f} µm")

    # (3) big single-step anchors — store by timestamp so we can reapply after fills
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    anchor_times = set(df.loc[big_jump_mask, 'time_local'])
    if big_jump_mask.any():
        print(f"{prefix}[DEBUG] Marking {big_jump_mask.sum()} big single-step jump(s) as shift anchors (> {avg_daily_range:.3f}).")

    # (4) slope threshold (08:00–11:00)
    if slope_col is None:
        slope_col = displacement_col
    if slope_threshold is None:
        median_slope, mean_slope, q3_slope = get_slope_stats(df, slope_col)
        if any(x is None for x in [median_slope, mean_slope, q3_slope]):
            small_slope_threshold = 2.0
            slope_source = "[WARN] fallback=2.0"
        else:
            if slope_method.lower() == 'median':
                small_slope_threshold = median_slope
                slope_source = f"median={median_slope:.3f}"
            elif slope_method.lower() == 'mean':
                small_slope_threshold = mean_slope
                slope_source = f"mean={mean_slope:.3f}"
            else:
                small_slope_threshold = q3_slope
                slope_source = f"q3={q3_slope:.3f}"
    else:
        small_slope_threshold = float(slope_threshold)
        slope_source = f"user={small_slope_threshold:.3f}"
    print(f"{prefix}[DEBUG] slope threshold => {slope_source}")

    # (5) IQR outlier detection
    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()
    large_diff_mask = df['Diff_Abs'] > small_slope_threshold
    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']

    outlier_mask = pd.Series(False, index=df.index)
    if large_diff_values.empty:
        print(f"{prefix}[DEBUG] No diffs exceed slope => skip IQR outlier detection.")
    else:
        Q1 = large_diff_values.quantile(0.25)
        Q3 = large_diff_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0.0, Q1 - multiplier * IQR)
        upper_bound = Q3 + multiplier * IQR
        print(f"{prefix}[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")

        outlier_mask = df['Diff_Abs'] > upper_bound
        print(f"{prefix}[DEBUG] Outliers flagged by IQR: {int(outlier_mask.sum())}")

        # ---- REMOVE IQR-flagged samples (set to NaN) ----
        df.loc[outlier_mask, displacement_col] = np.nan

        # ---- FILL & SCALE those NaNs using mean time-of-day increments ----
        df = fill_gaps_30min(
            df,
            disp_col=displacement_col,
            time_col='time_local',
            strategy='tod',                 # always do mean-TOD fill here
            freq=resample_freq,
            max_bins_to_fill=gap_fill_max_bins
        )

    # ---- Step 6: Baseline shift using (IQR outliers ∪ anchors)
    df.reset_index(drop=True, inplace=True)
    outlier_mask = outlier_mask.reset_index(drop=True)

    # Recompute per-row difference on the current (gap-filled, pruned) series
    df['Difference'] = df[displacement_col].diff()

    # Precompute the expected 30-min increment per HH:MM if we want to preserve it
    if preserve_expected_increment:
        df['__hhmm__'] = pd.to_datetime(df['time_local']).dt.strftime('%H:%M')
        if expected_stat.lower() == 'median':
            typical_inc_map = df.groupby('__hhmm__')['Difference'].median()
        else:
            typical_inc_map = df.groupby('__hhmm__')['Difference'].mean()
    else:
        typical_inc_map = None

    df['Adjustment'] = 0.0
    total_shift = 0.0
    for i in range(len(df)):
        if outlier_mask[i]:
            if i == 0 and remove_first_outlier:
                # keep the first point as-is (no shift applied)
                df.loc[i, 'Difference'] = np.nan
                continue

            jump = df.loc[i, 'Difference']

            # PARTIAL STITCH: only remove the excess over the expected 30-min change
            if preserve_expected_increment and typical_inc_map is not None:
                hhmm = df.loc[i, '__hhmm__']
                expected = float(typical_inc_map.get(hhmm, 0.0))
                jump = jump - expected    # after correction, the anchor leaves ~'expected' gap

            total_shift += jump

        df.loc[i, 'Adjustment'] = total_shift

    df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']
    if '__hhmm__' in df.columns:
        df.drop(columns='__hhmm__', inplace=True)

    # (7) final daily-range pass on cleaned
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Difference'] = df['Cleaned Displacement'].diff()
    final_mask = df['Difference'].abs() > avg_daily_range
    if final_mask.any():
        c = int(final_mask.sum())
        print(f"{prefix}[DEBUG] final pass => removing {c} rows > daily range.")
        df.loc[final_mask, 'Cleaned Displacement'] = np.nan
        pre = len(df)
        df.dropna(subset=['Cleaned Displacement'], inplace=True)
        post = len(df)
        print(f"{prefix}[DEBUG] dropped {pre - post} final rows.")

    # tidy
    if '__key' in df.columns:
        df.drop(columns='__key', inplace=True)

    print(f"{prefix}[DEBUG] done. gap_fill(initial)={gap_fill_strategy}, "
          f"post-IQR fill=mean-TOD, slope={slope_source}, anchors used, baseline shift done.")
    return df

def interactive_clean_file(
    file_path,
    start_date="2023-07-01",
    end_date="2024-11-15",
    initial_threshold=None,
    slope_method="mean",
    output_folder="cleaned_data"
):
    """
    Interactively clean one file:
      1) Determine trial name from the folder, device name from file name.
      2) Load raw data (via `load_and_prepare_data_time_limit`).
      3) Apply cleaning with slope_threshold.
      4) Plot raw vs cleaned for inspection.
      5) Prompt user if it looks good:
         - If "y" => save to `output_folder`.
         - If "n" => user can skip saving or change threshold.
         - If "change" => prompt for new threshold and re-clean.
    """
    # 1) Get trial name & device name from the file path
    trial_name = extract_trial_name(file_path)
    device_name = extract_device_name_from_filename(file_path)
    
    # 2) Load raw data. 
    #    (If your load function returns a device_name, you can ignore or override it here.)
    df_raw, _ = load_and_prepare_data_time_limit(
        file_path,
        start_date=start_date,
        end_date=end_date
    )
    df_raw = resample_to_30min(df_raw, device_name=device_name)
    # 3) Start slope_threshold
    slope_threshold = initial_threshold
    os.makedirs(output_folder, exist_ok=True)

    while True:
        # 1) Clean with the real function (anchors + baseline shift)
        df_cleaned = identify_and_adjust_outliers(
            df_raw,
            slope_threshold=slope_threshold,
            slope_method=slope_method,
            device_name=device_name,
            preserve_expected_increment=True,   # turn on partial stitch
            expected_stat='mean'                # or 'median'
        )
    
        # 2) (Optional) Plot RAW with flags only—no deletions here
        plot_raw_with_flags(
            df_raw,
            df_cleaned,
            slope_threshold=slope_threshold,
            slope_method=slope_method,
            device_name=device_name
        )
        plt.show()
    
        # 3) Original vs cleaned overlay (uses df_cleaned from the real cleaner)
        plot_original_vs_cleaned2(df_raw, df_cleaned, device_name=device_name)
        plt.show()
    
        # 4) Prompt (unchanged)
        threshold_str = f"{slope_threshold} (method={slope_method})" if slope_threshold is not None else f"Auto/None (method={slope_method})"
        print(f"\nTRIAL: {trial_name}\nFile: {file_path}\nDevice: {device_name}")
        print(f"Current slope threshold: {threshold_str}")
    
        user_input = input("Does this look okay? [y/n/change]: ").strip().lower()

        if user_input == "y":
            # C) Save & break
            safe_trial = trial_name.replace(" ", "_")
            clean_filename = f"{safe_trial}_{device_name}_cleaned.csv"
            out_path = os.path.join(output_folder, clean_filename)
            df_cleaned.to_csv(out_path, index=False)
            print(f"✔ Accepted. Cleaned data saved to: {out_path}\n")
            break

        elif user_input == "n":
            # D) skip or change threshold
            skip_or_change = input("You selected NO. Skip saving (s) or change threshold (c)? [s/c]: ").strip().lower()
            if skip_or_change == "s":
                print("Skipping saving. Moving to next file.\n")
                break
            elif skip_or_change == "c":
                new_thr = input("Enter new slope threshold (or 'None' for auto): ").strip()
                if new_thr.lower() == "none":
                    slope_threshold = None
                else:
                    try:
                        slope_threshold = float(new_thr)
                    except ValueError:
                        print("Invalid input. Retaining old threshold.")
            else:
                print("Invalid choice. Type 's' or 'c'.")

        elif user_input == "change":
            new_thr = input("Enter new slope threshold (or 'None' for auto): ").strip()
            if new_thr.lower() == "none":
                slope_threshold = None
            else:
                try:
                    slope_threshold = float(new_thr)
                except ValueError:
                    print("Invalid input. Retaining old threshold.")

        else:
            print("Invalid option. Please enter 'y', 'n', or 'change'.")

def main():
    """Interactively clean every CSV in a folder.

    Usage:
        python interactive_data_cleaning.py <data_dir> [output_folder]

    Or set the data directory with the DENDRO_DATA_DIR environment
    variable.  Each CSV in <data_dir> is opened in turn; cleaning
    decisions are made interactively at the prompts.
    """
    import argparse

    ap = argparse.ArgumentParser(
        description="Interactive dendrometer cleaning.")
    ap.add_argument("data_dir", nargs="?",
                    default=os.environ.get("DENDRO_DATA_DIR", "raw_data"),
                    help="folder containing raw dendrometer CSVs")
    ap.add_argument("output_folder", nargs="?", default="cleaned_data",
                    help="folder for cleaned output (default: cleaned_data)")
    ap.add_argument("--start-date", default=None,
                    help="optional start date, YYYY-MM-DD")
    ap.add_argument("--end-date", default=None,
                    help="optional end date, YYYY-MM-DD")
    ap.add_argument("--slope-method", default="median",
                    help="slope estimator for gap bridging (default: median)")
    args = ap.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not all_files:
        print(f"[WARN] No CSV files found in: {args.data_dir}")
        return

    print(f"Found {len(all_files)} file(s) in {args.data_dir}")
    for raw_f in all_files:
        interactive_clean_file(
            file_path=raw_f,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_threshold=None,
            slope_method=args.slope_method,
            output_folder=args.output_folder,
        )


if __name__ == "__main__":
    main()
