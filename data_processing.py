"""
data_processing.py

Contains data cleaning, outlier detection, and transformations (e.g., 'Cleaned Displacement').
"""

import pandas as pd
import numpy as np


'''def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    small_slope_threshold=2,
    remove_first_outlier=True
):
    """
    Identifies and adjusts outliers in the displacement data using:
      1) alignment/pressed/pos checks to remove certain rows entirely.
      2) big-jump removal if a single diff > average daily range.
      3) IQR-based outlier detection, BUT:
         - The IQR bounds are computed only from diffs > small_slope_threshold.
         - The resulting IQR lower/upper bounds are then applied to *all* diffs.
      4) A cumulative shift logic for flagged outliers, resulting in 'Cleaned Displacement'.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have 'time_local' and a displacement_col (default 'um').
    displacement_col : str
        Column name for displacement data.
    multiplier : float
        Multiplier for IQR (default 1.5).
    alignment_col, pressed_col, pos_raw_col, pos_avg_col : str (optional)
        Columns to remove if alignment='Red', pressed_col=True,
        or pos_raw-pos_avg>pos_diff_threshold, etc.
    pos_diff_threshold : float
        If (pos_raw_col - pos_avg_col).abs() > this => remove the row entirely.
    small_slope_threshold : float
        Only diffs > this are used to *compute* IQR. But then we apply those
        IQR bounds to all diffs.
    remove_first_outlier : bool
        If True, skip applying the baseline shift to the first row if it's an outlier.

    Returns
    -------
    df : pandas.DataFrame
        The updated DataFrame with new columns:
          - 'Difference': consecutive difference
          - 'Adjustment': cumulative shift
          - 'Cleaned Displacement': final corrected displacement
    """
    df = df.copy()
    df.sort_values(by='time_local', inplace=True)

    #
    # Step A: average daily range => big-jump cutoff
    #
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())
    avg_daily_range = daily_ranges.mean()
    print(f"[identify_and_adjust_outliers] Avg daily range for '{displacement_col}'= {avg_daily_range:.2f} µm")

    #
    # Step B: remove alignment/pressed/pos outliers (row removal)
    #
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

    pos_mask = df['pos_diff'] > pos_diff_threshold

    combined_skip = alignment_mask | pressed_mask | pos_mask
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"Removing {skip_count} row(s) due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()

    #
    # Step C: remove single-step jumps bigger than avg_daily_range
    #
    df.sort_values('time_local', inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    if big_jump_mask.any():
        count_big_jumps = big_jump_mask.sum()
        print(f"Removing {count_big_jumps} row(s) with single-step jump > {avg_daily_range:.2f} µm.")
        df.loc[big_jump_mask, displacement_col] = np.nan

    # Drop NaNs & recalc difference
    df.dropna(subset=[displacement_col], inplace=True)
    df.sort_values('time_local', inplace=True)
    df['Difference'] = df[displacement_col].diff()

    #
    # Step D: IQR-based outlier detection
    #         BUT we compute Q1/Q3 only from diffs > small_slope_threshold,
    #         then apply those bounds to *all* diffs (including < threshold).
    #
    # 1) compute the IQR only from large diffs
    large_diff_mask = df['Difference'].abs() > small_slope_threshold
    large_diff_values = df.loc[large_diff_mask, 'Difference']

    if large_diff_values.empty:
        # If no "large" diffs exist, fallback
        print("No diffs exceed small_slope_threshold => skipping IQR logic.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]
        return df

    Q1 = large_diff_values.quantile(0.25)
    Q3 = large_diff_values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # 2) Outlier mask applies to *all* diffs, not just large ones
    outlier_mask = (df['Difference'] < lower_bound) | (df['Difference'] > upper_bound)

    #
    # Step E: baseline shift logic
    #
    df['Adjustment'] = 0.0
    total_shift = 0.0

    df.reset_index(drop=True, inplace=True)
    outlier_mask = outlier_mask.reset_index(drop=True)

    for i in range(len(df)):
        if outlier_mask[i]:
            if i == 0 and remove_first_outlier:
                df.loc[i, 'Difference'] = np.nan
                continue
            jump = df.loc[i, 'Difference']
            total_shift += jump

        if i == 0:
            df.loc[i, 'Adjustment'] = 0.0
        else:
            df.loc[i, 'Adjustment'] = total_shift

    df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    # Clean up
    drop_cols = ['Difference', 'Adjustment', 'pos_diff', 'date']
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    print("[identify_and_adjust_outliers] Done. IQR thresholds derived from diffs >"
          f" {small_slope_threshold}, but applied to entire dataset.")
    return df




def identify_and_adjust_outliers2(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None
):
    """
    Identifies and adjusts outliers in the displacement data using:
      1) alignment/pressed/pos checks to remove certain rows entirely.
      2) Automatic slope calculation (median, mean, Q3) from time windows 8–15h, 19–23h.
         We pick the largest of {median, mean, Q3} as our threshold.
      3) Single-step big-jump removal if a single diff > average daily range.
      4) IQR-based outlier detection:
         - The IQR bounds are computed only from diffs > the chosen slope threshold.
         - Those IQR bounds are then applied to all diffs.
      5) Baseline shift logic for flagged outliers, resulting in 'Cleaned Displacement'.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have 'time_local' (datetime) and a displacement_col (default 'um').
    displacement_col : str
        Column name for displacement data.
    multiplier : float
        Multiplier for IQR (default 1.5).
    alignment_col, pressed_col, pos_raw_col, pos_avg_col : str (optional)
        Column names to remove rows if alignment='Red', pressed_col=True,
        or |pos_raw_col - pos_avg_col| > pos_diff_threshold.
    pos_diff_threshold : float
        If (pos_raw_col - pos_avg_col).abs() > this => remove the row entirely.
    remove_first_outlier : bool
        If True, skip applying the baseline shift to the first row if it's an outlier.
    slope_col : str or None
        Which column to use when computing slope stats. If None, uses displacement_col.

    Returns
    -------
    df : pandas.DataFrame
        The updated DataFrame with outlier-cleaned displacement in
        'Cleaned Displacement'. All row-removal steps have already happened.
    """

    # -------------------------------------------------------------------------
    # A helper function (nested) to compute median, mean, and Q3 of row-by-row
    # absolute slopes in [08:00–15:00) & [19:00–23:00).
    # -------------------------------------------------------------------------
    def get_slope_stats_per_index(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time

        # Build time masks
        mask1_start = pd.to_datetime("10:00").time()
        mask1_end   = pd.to_datetime("14:00").time()
        mask1 = (df_local['time_of_day'] >= mask1_start) & (df_local['time_of_day'] < mask1_end)

        mask2_start = pd.to_datetime("21:00").time()
        mask2_end   = pd.to_datetime("23:00").time()
        mask2 = (df_local['time_of_day'] >= mask2_start) & (df_local['time_of_day'] < mask2_end)

        df_filtered = df_local[mask1 | mask2].sort_values('time_local').copy()
        if df_filtered.empty:
            print("[INFO] No data in the specified time windows (8am–3pm or 7pm–11pm).")
            return None, None, None

        # Compute row-by-row absolute slope
        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()

        if slope_series.empty:
            print("[INFO] No valid slope data (all NaN).")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)
        return median_val, mean_val, q3_val
    # -------------------------------------------------------------------------

    # Copy & sort our input DataFrame
    df = df.copy()
    df.sort_values(by='time_local', inplace=True)

    # Decide which column to use for slope calculation
    if slope_col is None:
        slope_col = displacement_col  # fallback

    #
    # Step A: average daily range => big-jump cutoff
    #
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())
    avg_daily_range = daily_ranges.mean()
    print(f"[identify_and_adjust_outliers] Avg daily range for '{displacement_col}'= {avg_daily_range:.2f} µm")

    #
    # Step B: remove alignment/pressed/pos outliers (row removal)
    #
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

    pos_mask = df['pos_diff'] > pos_diff_threshold

    combined_skip = alignment_mask | pressed_mask | pos_mask
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"Removing {skip_count} row(s) due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()

    #
    # Step B2: automatically compute the slope threshold from what's left
    #    - we'll get (median, mean, Q3) and pick the largest
    #
    median_slope, mean_slope, q3_slope = get_slope_stats_per_index(df, slope_col)
    if any(x is None for x in [median_slope, mean_slope, q3_slope]):
        # Fallback if we can't compute slope
        small_slope_threshold = 2.0
        print("[WARN] Could not compute slope in time windows; falling back to small_slope_threshold=2.0")
    else:
        small_slope_threshold = max(median_slope, mean_slope, q3_slope)
        print(
            f"[INFO] Slopes: median={median_slope:.2f}, mean={mean_slope:.2f}, Q3={q3_slope:.2f} "
            f"=> using largest={small_slope_threshold:.2f} µm/index"
        )

    #
    # Step C: remove single-step jumps bigger than avg_daily_range
    #
    df.sort_values('time_local', inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    if big_jump_mask.any():
        count_big_jumps = big_jump_mask.sum()
        print(f"Removing {count_big_jumps} row(s) with single-step jump > {avg_daily_range:.2f} µm.")
        df.loc[big_jump_mask, displacement_col] = np.nan

    # Drop NaNs & recalc difference
    df.dropna(subset=[displacement_col], inplace=True)
    df.sort_values('time_local', inplace=True)
    df['Difference'] = df[displacement_col].diff()

    #
    # Step D: IQR-based outlier detection
    #         (Compute Q1/Q3 only from diffs > small_slope_threshold,
    #          then apply to *all* diffs).
    #
    large_diff_mask = df['Difference'].abs() > small_slope_threshold
    large_diff_values = df.loc[large_diff_mask, 'Difference']

    if large_diff_values.empty:
        print("No diffs exceed small_slope_threshold => skipping IQR logic.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]

        # Clean up
        drop_cols = ['Difference', 'Adjustment', 'pos_diff', 'date']
        for c in drop_cols:
            if c in df.columns:
                df.drop(columns=c, inplace=True)
        return df

    Q1 = large_diff_values.quantile(0.25)
    Q3 = large_diff_values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outlier_mask = (df['Difference'] < lower_bound) | (df['Difference'] > upper_bound)

    #
    # Step E: baseline shift logic
    #
    df['Adjustment'] = 0.0
    total_shift = 0.0

    df.reset_index(drop=True, inplace=True)
    outlier_mask = outlier_mask.reset_index(drop=True)

    for i in range(len(df)):
        if outlier_mask[i]:
            # If it's the first row and we're told to remove it entirely:
            if i == 0 and remove_first_outlier:
                df.loc[i, 'Difference'] = np.nan
                continue
            jump = df.loc[i, 'Difference']
            total_shift += jump

        # Set the per-row 'Adjustment'
        if i == 0:
            df.loc[i, 'Adjustment'] = 0.0
        else:
            df.loc[i, 'Adjustment'] = total_shift

    df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    # Clean up
    drop_cols = ['Difference', 'Adjustment', 'pos_diff', 'date']
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    print(
        "[identify_and_adjust_outliers] Done. IQR thresholds derived from diffs >"
        f" {small_slope_threshold:.2f}, but applied to entire dataset."
    )
    return df

'''


def identify_and_adjust_outliers(
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
    Identifies and adjusts outliers in the displacement data, except we
    do NOT remove big gaps but rather 'collapse' them by shifting subsequent data
    so that the next record merges at the same displacement if the gap is large
    AND the difference is bigger than the daily range.

    Steps:
      0) Collapse large time gaps without deleting rows.
      1) Remove alignment/pressed/pos-based outlier rows.
      2) Compute 'capped' average daily range on remaining data & remove big jumps once.
      3) Compute or accept slope threshold (median, mean, or Q3).
      4) IQR-based outlier detection on absolute diffs; remove short outlier runs.
      5) Baseline shift => 'Cleaned Displacement'.
      6) Re-check big jumps & remove them if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least ['time_local', displacement_col].
    displacement_col : str
        Column with the displacement measurements (default 'um').
    multiplier : float
        Multiplicative factor for IQR-based outlier detection. Default=1.5.
    alignment_col : str
        Column indicating alignment errors to remove (default 'Alignment').
    pressed_col : str
        Column indicating pressed events to remove (default 'Pressed?').
    pos_raw_col : str
        Raw position column name (default 'pos_raw').
    pos_avg_col : str
        Smoothed/average position column name (default 'pos_avg').
    pos_diff_threshold : float
        Threshold for difference between pos_raw & pos_avg to remove row (default=5).
    remove_first_outlier : bool
        If True, skip baseline shift on the very first outlier row.
    slope_col : str
        Column on which to calculate slope if different from displacement_col.
    run_threshold : int
        Min outlier-run length to keep as outliers (shorter runs removed). Default=2.
    slope_threshold : float or None
        If None, automatically compute from slope stats (median, mean, or Q3).
        If a numeric value, use that directly.
    slope_method : str
        One of ['median','mean','q3'] for auto slope threshold if slope_threshold is None.
    max_gap_minutes : int
        If time gap > this many minutes, consider it a "break" to collapse. Default=30.
    device_name : str or None
        If provided, all debug statements will prefix with [device_name] for clarity.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with outlier-cleaned displacement in 'Cleaned Displacement'.
    """

    # A small helper to prepend device name in debug messages
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

        # Check if a run continues to the end
        if run_start is not None:
            run_end = n - 1
            length = run_end - run_start + 1
            if length < run_threshold:
                short_run_mask.iloc[run_start:run_end+1] = True

        # Mark outliers in short runs as NaN
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

    # Step A: Remove alignment/pressed/pos-based outlier rows
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

    combined_skip = alignment_mask | pressed_mask | (df['pos_diff']>pos_diff_threshold)
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"{prefix}[DEBUG] Removing {skip_count} rows due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step B: Compute average daily range (capped at 50 µm) for collapse logic
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())
    daily_ranges = daily_ranges.clip(upper=50)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"{prefix}[DEBUG] daily_range => {avg_daily_range:.3f} µm")

    # Step 0: Collapse large gaps if single-step jump > avg_daily_range
    df = collapse_large_gaps_no_delete(df, displacement_col, avg_daily_range, gap_minutes=max_gap_minutes)

    # Step 1: Remove big jumps again if needed
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    if big_jump_mask.any():
        count_big_jumps = big_jump_mask.sum()
        print(f"{prefix}[DEBUG] Removing {count_big_jumps} row(s) with single-step jump > {avg_daily_range:.3f} after gap collapse.")
        df.loc[big_jump_mask, displacement_col] = np.nan
        df.dropna(subset=[displacement_col], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Step 2: Determine slope threshold
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
                # Default to 'q3'
                small_slope_threshold = q3_slope
                slope_source = f"q3={q3_slope:.3f}"
    else:
        small_slope_threshold = slope_threshold
        slope_source = f"user={slope_threshold:.3f}"

    print(f"{prefix}[DEBUG] slope threshold => {slope_source}")

    # Step 3: IQR-based outlier detection
    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()
    large_diff_mask = df['Diff_Abs'] > small_slope_threshold
    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']
    if large_diff_values.empty:
        print(f"{prefix}[DEBUG] No diffs exceed slope => skip IQR outlier detection.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]
    else:
        # Compute IQR over the large diffs
        Q1 = large_diff_values.quantile(0.25)
        Q3 = large_diff_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - multiplier*IQR)
        upper_bound = Q3 + multiplier*IQR
        print(f"{prefix}[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")

        outlier_mask = df['Diff_Abs'] > upper_bound
        outlier_count = outlier_mask.sum()
        print(f"{prefix}[DEBUG] Found {outlier_count} outliers by IQR mask.")

        # Step 4: Remove short outlier runs
        df, outlier_mask = remove_short_outlier_runs(
            df, outlier_mask, displacement_col, upper_bound, run_threshold
        )

        # Step 5: Baseline shift
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

    # Step 6: Final check for big jumps vs avg_daily_range
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)

    if 'Cleaned Displacement' not in df.columns:
        df['Cleaned Displacement'] = df[displacement_col]

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

    print(f"{prefix}[DEBUG] done. slope={slope_source}, no-deletion collapse of gaps > {max_gap_minutes} min, outliers removed, shift done.")
    return df
'''


def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None
):
    """
    Identifies and adjusts outliers in the displacement data using:
      1) alignment/pressed/pos checks to remove certain rows entirely.
      2) Automatic slope calculation (median, mean, Q3) from time windows 8–15h, 19–23h.
         => The largest of {median, mean, Q3} is used as the slope threshold.
      3) Single-step big-jump removal if a single diff > average daily range.
      4) IQR-based outlier detection on the *absolute* differences:
         - We compute Q1 and Q3 only from rows whose |Difference| > chosen slope threshold.
         - Then the IQR bounds are applied to the entire dataset (in absolute value).
      5) Baseline shift logic for flagged outliers, resulting in 'Cleaned Displacement'.

    Returns
    -------
    df : pandas.DataFrame
        The updated DataFrame with outlier-cleaned displacement in
        'Cleaned Displacement'.
    """

    # -------------------------------------------------------------------------
    # A helper function (nested) to compute median, mean, and Q3 of row-by-row
    # absolute slopes in [08:00–15:00) & [19:00–23:00).
    # -------------------------------------------------------------------------
    def get_slope_stats_per_index(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time

        # Build time masks
        mask1_start = pd.to_datetime("07:30").time()
        mask1_end   = pd.to_datetime("08:30").time()
        mask1 = (df_local['time_of_day'] >= mask1_start) & (df_local['time_of_day'] < mask1_end)

        mask2_start = pd.to_datetime("18:30").time()
        mask2_end   = pd.to_datetime("19:30").time()
        mask2 = (df_local['time_of_day'] >= mask2_start) & (df_local['time_of_day'] < mask2_end)

        df_filtered = df_local[mask1 | mask2].sort_values('time_local').copy()
        if df_filtered.empty:
            print("[DEBUG] No data in the specified time windows (8am–3pm or 7pm–11pm).")
            return None, None, None

        # Compute row-by-row absolute slope
        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()

        if slope_series.empty:
            print("[DEBUG] No valid slope data (all NaN) in the time windows.")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)

        print(f"[DEBUG] Slope stats in allowed times => Median={median_val:.3f}, Mean={mean_val:.3f}, Q3={q3_val:.3f}")
        return median_val, mean_val, q3_val
    # -------------------------------------------------------------------------

    # Copy & sort our input DataFrame
    df = df.copy()
    df.sort_values(by='time_local', inplace=True)

    # Decide which column to use for slope calculation
    if slope_col is None:
        slope_col = displacement_col  # fallback

    #
    # Step A: average daily range => big-jump cutoff
    #
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"[DEBUG] Avg daily range (displacement_col='{displacement_col}') = {avg_daily_range:.3f} µm")

    #
    # Step B: remove alignment/pressed/pos outliers (row removal)
    #
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

    pos_mask = df['pos_diff'] > pos_diff_threshold
    combined_skip = alignment_mask | pressed_mask | pos_mask

    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"[DEBUG] Removing {skip_count} row(s) due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()
    else:
        print("[DEBUG] No alignment/pressed/pos rows to remove.")

    #
    # Step B2: automatically compute the slope threshold from what's left
    #
    median_slope, mean_slope, q3_slope = get_slope_stats_per_index(df, slope_col)
    if any(x is None for x in [median_slope, mean_slope, q3_slope]):
        # Fallback if we can't compute slope
        small_slope_threshold = 2.0
        print("[WARN] Could not compute slope (Median/Mean/Q3) => fallback to 2.0")
    else:
        small_slope_threshold = max(median_slope, mean_slope, q3_slope)
        print(
            f"[DEBUG] Will use largest slope among median={median_slope:.3f}, "
            f"mean={mean_slope:.3f}, Q3={q3_slope:.3f} => {small_slope_threshold:.3f}"
        )

    #
    # Step C: remove single-step jumps bigger than avg_daily_range
    #
    df.sort_values('time_local', inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range

    if big_jump_mask.any():
        count_big_jumps = big_jump_mask.sum()
        print(f"[DEBUG] Removing {count_big_jumps} row(s) with single-step jump > {avg_daily_range:.3f} µm.")
        df.loc[big_jump_mask, displacement_col] = np.nan
    else:
        print("[DEBUG] No single-step jumps exceed the average daily range.")

    # Drop NaNs (if any) & recalc difference
    pre_drop_count = len(df)
    df.dropna(subset=[displacement_col], inplace=True)
    post_drop_count = len(df)
    if post_drop_count < pre_drop_count:
        print(f"[DEBUG] Dropped {pre_drop_count - post_drop_count} rows due to big jumps (NaN).")

    df.sort_values('time_local', inplace=True)
    df['Difference'] = df[displacement_col].diff()

    #
    # Step D: IQR-based outlier detection on absolute diffs
    #
    # 1) Create abs difference column
    df['Diff_Abs'] = df['Difference'].abs()

    # 2) Identify which rows exceed our slope threshold
    large_diff_mask = df['Diff_Abs'] > small_slope_threshold
    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']

    if large_diff_values.empty:
        print("[DEBUG] No absolute diffs exceed small_slope_threshold => skipping IQR logic.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]
        return df

    # 3) Compute IQR from those large diffs
    Q1 = large_diff_values.quantile(0.25)
    Q3 = large_diff_values.quantile(0.75)
    IQR = Q3 - Q1

    # For absolute distribution, negative bounds don't make sense; clamp lower bound to 0
    lower_bound = max(0, Q1 - multiplier * IQR)
    upper_bound = Q3 + multiplier * IQR

    print(f"[DEBUG] IQR outlier bounds on abs diffs => lower={lower_bound:.3f}, upper={upper_bound:.3f}")

    # 4) For the entire dataset, any Diff_Abs > upper_bound is flagged as outlier
    outlier_mask = df['Diff_Abs'] > upper_bound
    print(f"[DEBUG] Found {outlier_mask.sum()} outlier row(s) by IQR mask on abs diffs.")

    #
    # Step E: baseline shift logic
    #
    # We still shift using the *signed* 'Difference' but only if |Difference| is out-of-bounds.
    #
    df['Adjustment'] = 0.0
    total_shift = 0.0

    # Reset index so we can iterate in order
    df.reset_index(drop=True, inplace=True)
    outlier_mask = outlier_mask.reset_index(drop=True)

    for i in range(len(df)):
        if outlier_mask[i]:
            # If it's the first row and we want to skip it, set that difference to NaN
            if i == 0 and remove_first_outlier:
                df.loc[i, 'Difference'] = np.nan
                print("[DEBUG] First row is outlier => removing difference for that row.")
                continue
            # Add the *signed* difference to total_shift
            jump = df.loc[i, 'Difference']
            total_shift += jump

        # Each row accumulates the total shift so far
        df.loc[i, 'Adjustment'] = total_shift

    df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    # (Optionally, remove debug columns at the end)
    # drop_cols = ['Difference', 'Adjustment', 'Diff_Abs', 'pos_diff', 'date']
    # for c in drop_cols:
    #     if c in df.columns:
    #         df.drop(columns=c, inplace=True)

    print("[DEBUG] Done. IQR thresholds derived from abs diffs > "
          f"{small_slope_threshold:.3f}, applied to entire dataset.")
    return df
'''

'''

import numpy as np
import pandas as pd

def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2   # number of consecutive outliers required
):
    """
    Identifies and adjusts outliers in the displacement data using:
      1) alignment/pressed/pos checks to remove certain rows entirely.
      2) Automatic slope calculation (median, mean, Q3) within 07:30–08:30 or 18:30–19:30,
         picking the largest of {median, mean, Q3} as 'small_slope_threshold'.
      3) Single-step big-jump removal if a single diff > *capped* average daily range (capped at 100 um).
      4) IQR-based outlier detection on *absolute* differences:
         - We compute Q1 and Q3 only from rows whose |Difference| > chosen slope threshold,
         - Then the IQR bounds are applied to the entire dataset (in absolute value).
      5) Remove short runs (< run_threshold) of flagged outliers (instead of only single points).
      6) Baseline shift logic for multi-point flagged outliers, resulting in 'Cleaned Displacement'.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have a datetime column 'time_local' and a displacement_col (default 'um').
    displacement_col : str
        The column with displacement data.
    multiplier : float
        IQR multiplier (1.5 is typical, but you can increase if needed).
    alignment_col, pressed_col, pos_raw_col, pos_avg_col : str
        Columns to remove if alignment='Red', pressed_col=True, or abs(pos_raw-col - pos_avg_col) > pos_diff_threshold.
    pos_diff_threshold : float
        If (pos_raw - pos_avg).abs() > this => remove the row entirely.
    remove_first_outlier : bool
        If True, the first row flagged as outlier won't cause a baseline shift.
    slope_col : str or None
        Column to compute slope stats from. If None, uses 'displacement_col'.
    run_threshold : int
        Minimum length of a consecutive outlier run needed to shift the baseline.
        Shorter runs are dropped entirely.

    Returns
    -------
    df : pandas.DataFrame
        Updated DataFrame with outlier-cleaned displacement in 'Cleaned Displacement'.
    """

    # -------------------------------------------------------------------------
    # Helper: compute median, mean, Q3 of row-by-row abs slopes in [07:30–08:30) & [18:30–19:30).
    # -------------------------------------------------------------------------
    def get_slope_stats_per_index(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time

        # Build time masks
        mask1_start = pd.to_datetime("07:30").time()
        mask1_end   = pd.to_datetime("08:30").time()
        mask1 = (df_local['time_of_day'] >= mask1_start) & (df_local['time_of_day'] < mask1_end)

        mask2_start = pd.to_datetime("18:30").time()
        mask2_end   = pd.to_datetime("19:30").time()
        mask2 = (df_local['time_of_day'] >= mask2_start) & (df_local['time_of_day'] < mask2_end)

        df_filtered = df_local[mask1 | mask2].sort_values('time_local').copy()
        if df_filtered.empty:
            print("[DEBUG] No data in the 07:30–08:30 or 18:30–19:30 windows.")
            return None, None, None

        # Compute row-by-row absolute slope
        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()

        if slope_series.empty:
            print("[DEBUG] No valid slope data (all NaN) in the time windows.")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)

        print(f"[DEBUG] Slope stats => Median={median_val:.3f}, Mean={mean_val:.3f}, Q3={q3_val:.3f}")
        return median_val, mean_val, q3_val

    # -------------------------------------------------------------------------
    # Helper: remove short runs of outliers (< run_threshold)
    #         Then re-calc 'Difference' and 'Diff_Abs'.
    # -------------------------------------------------------------------------
    def remove_short_outlier_runs(df, outlier_mask, displacement_col, upper_bound, run_threshold):
        outlier_mask = outlier_mask.reset_index(drop=True)
        df = df.reset_index(drop=True)

        short_run_mask = pd.Series(False, index=outlier_mask.index)
        is_outlier = outlier_mask.values
        n = len(is_outlier)

        run_start = None
        for i in range(n):
            if is_outlier[i] and run_start is None:
                # begin a new run
                run_start = i
            elif not is_outlier[i] and run_start is not None:
                # end of a run
                run_end = i - 1
                length = run_end - run_start + 1
                if length < run_threshold:
                    short_run_mask.iloc[run_start:run_end+1] = True
                run_start = None

        # If the last row ended in an outlier run
        if run_start is not None:
            run_end = n - 1
            length = run_end - run_start + 1
            if length < run_threshold:
                short_run_mask.iloc[run_start:run_end+1] = True

        # Remove (NaN) these short-run outliers
        df.loc[short_run_mask, displacement_col] = np.nan
        pre_drop = len(df)
        df.dropna(subset=[displacement_col], inplace=True)
        post_drop = len(df)
        removed_count = pre_drop - post_drop

        df.reset_index(drop=True, inplace=True)
        df['Difference'] = df[displacement_col].diff()
        df['Diff_Abs']   = df['Difference'].abs()

        # Rebuild the outlier_mask for the updated df
        new_outlier_mask = (df['Diff_Abs'] > upper_bound)

        print(f"[DEBUG] Removed {removed_count} row(s) in short outlier runs (< {run_threshold}).")
        print(f"[DEBUG] After short-run removal, {new_outlier_mask.sum()} row(s) remain flagged.")
        return df, new_outlier_mask

    # -------------------------------------------------------------------------
    # Main logic
    # -------------------------------------------------------------------------
    df = df.copy()
    df.sort_values(by='time_local', inplace=True)

    if slope_col is None:
        slope_col = displacement_col

    # Step A: average daily range, with a cap of 100 µm per day
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date

    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())
    # Clip each day's range to max value
    daily_ranges = daily_ranges.clip(upper=200)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0

    print(f"[DEBUG] Capping daily range at 100 => average daily range = {avg_daily_range:.3f} µm")

    # Step B: remove alignment/pressed/pos outliers
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

    pos_mask = df['pos_diff'] > pos_diff_threshold
    combined_skip = alignment_mask | pressed_mask | pos_mask

    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"[DEBUG] Removing {skip_count} row(s) due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()
    else:
        print("[DEBUG] No alignment/pressed/pos rows to remove.")

    # Step B2: slope threshold (largest of median, mean, Q3)
    median_slope, mean_slope, q3_slope = get_slope_stats_per_index(df, slope_col)
    if any(x is None for x in [median_slope, mean_slope, q3_slope]):
        small_slope_threshold = 2.0
        print("[WARN] Could not compute slope => fallback to 2.0")
    else:
        small_slope_threshold = max(median_slope, mean_slope, q3_slope)
        print(f"[DEBUG] Using largest slope: {small_slope_threshold:.3f}")

    # Step C: remove single-step jumps bigger than our newly capped avg_daily_range
    df.sort_values('time_local', inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    if big_jump_mask.any():
        count_big_jumps = big_jump_mask.sum()
        print(f"[DEBUG] Removing {count_big_jumps} row(s) with single-step jump > {avg_daily_range:.3f} µm.")
        df.loc[big_jump_mask, displacement_col] = np.nan
    else:
        print("[DEBUG] No single-step jumps exceed the daily range.")

    pre_drop_count = len(df)
    df.dropna(subset=[displacement_col], inplace=True)
    post_drop_count = len(df)
    if post_drop_count < pre_drop_count:
        print(f"[DEBUG] Dropped {pre_drop_count - post_drop_count} row(s) from big jumps.")
    df.reset_index(drop=True, inplace=True)

    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()

    # Step D: IQR-based detection on abs diffs
    large_diff_mask = (df['Diff_Abs'] > small_slope_threshold)
    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']
    if large_diff_values.empty:
        print("[DEBUG] No diffs exceed threshold => skipping IQR logic.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]
        return df

    Q1 = large_diff_values.quantile(0.25)
    Q3 = large_diff_values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - multiplier * IQR)
    upper_bound = Q3 + multiplier * IQR
    print(f"[DEBUG] IQR bounds => lower={lower_bound:.3f}, upper={upper_bound:.3f}")

    outlier_mask = df['Diff_Abs'] > upper_bound
    print(f"[DEBUG] Found {outlier_mask.sum()} outlier row(s) by IQR mask.")

    # Step E: remove short outlier runs (< run_threshold)
    df, outlier_mask = remove_short_outlier_runs(
        df, outlier_mask,
        displacement_col=displacement_col,
        upper_bound=upper_bound,
        run_threshold=run_threshold
    )

    # Step F: baseline shift logic for remaining runs
    df.reset_index(drop=True, inplace=True)
    outlier_mask = outlier_mask.reset_index(drop=True)
    df['Adjustment'] = 0.0
    total_shift = 0.0

    for i in range(len(df)):
        if outlier_mask[i]:
            if i == 0 and remove_first_outlier:
                df.loc[i, 'Difference'] = np.nan
                print("[DEBUG] First row outlier => removing difference.")
                continue
            jump = df.loc[i, 'Difference']
            total_shift += jump
        df.loc[i, 'Adjustment'] = total_shift

    df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    print(
        f"[DEBUG] Done. Using daily range capped at 100 => {avg_daily_range:.3f}, "
        f"IQR from diffs > {small_slope_threshold:.3f}, short runs (<{run_threshold}) removed, baseline shift applied."
    )
    return df
'''
import numpy as np
import pandas as pd

'''def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2   # number of consecutive outliers required
):
    """
    Identifies and adjusts outliers in the displacement data using:
      1) Remove alignment/pressed/pos-based outliers (entire rows) first.
      2) Compute 'capped' average daily range on the *remaining* data, remove big jumps.
      3) Automatic slope calculation (median, mean, Q3) for small_slope_threshold.
      4) IQR-based outlier detection on *absolute* differences, removing short outlier runs.
      5) Baseline shift logic for any remaining multi-point flagged outliers => 'Cleaned Displacement'.

    Returns
    -------
    df : pandas.DataFrame
        Updated DataFrame with outlier-cleaned displacement in 'Cleaned Displacement'.
    """

    # -------------------------------------------------------------------------
    # Helper: compute median, mean, Q3 of row-by-row abs slopes in time windows
    # -------------------------------------------------------------------------
    def get_slope_stats_per_index(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time

        # Build time masks
        mask1_start = pd.to_datetime("07:30").time()
        mask1_end   = pd.to_datetime("08:30").time()
        mask1 = (df_local['time_of_day'] >= mask1_start) & (df_local['time_of_day'] < mask1_end)

        mask2_start = pd.to_datetime("18:30").time()
        mask2_end   = pd.to_datetime("19:30").time()
        mask2 = (df_local['time_of_day'] >= mask2_start) & (df_local['time_of_day'] < mask2_end)

        df_filtered = df_local[mask1 | mask2].sort_values('time_local').copy()
        if df_filtered.empty:
            print("[DEBUG] No data in the 07:30–08:30 or 18:30–19:30 windows.")
            return None, None, None

        # Compute row-by-row absolute slope
        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()

        if slope_series.empty:
            print("[DEBUG] No valid slope data (all NaN) in the time windows.")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)
        print(f"[DEBUG] Slope stats => Median={median_val:.3f}, Mean={mean_val:.3f}, Q3={q3_val:.3f}")
        return median_val, mean_val, q3_val

    # -------------------------------------------------------------------------
    # Helper: remove short runs of outliers (< run_threshold)
    #         Then re-calc 'Difference' and 'Diff_Abs'.
    # -------------------------------------------------------------------------
    def remove_short_outlier_runs(df, outlier_mask, displacement_col, upper_bound, run_threshold):
        outlier_mask = outlier_mask.reset_index(drop=True)
        df = df.reset_index(drop=True)

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

        # If the last row ended in an outlier run
        if run_start is not None:
            run_end = n - 1
            length = run_end - run_start + 1
            if length < run_threshold:
                short_run_mask.iloc[run_start:run_end+1] = True

        # Remove (NaN) these short-run outliers
        df.loc[short_run_mask, displacement_col] = np.nan
        pre_drop = len(df)
        df.dropna(subset=[displacement_col], inplace=True)
        post_drop = len(df)
        removed_count = pre_drop - post_drop

        df.reset_index(drop=True, inplace=True)
        df['Difference'] = df[displacement_col].diff()
        df['Diff_Abs']   = df['Difference'].abs()

        new_outlier_mask = (df['Diff_Abs'] > upper_bound)

        print(f"[DEBUG] Removed {removed_count} row(s) in short outlier runs (< {run_threshold}).")
        print(f"[DEBUG] After short-run removal, {new_outlier_mask.sum()} row(s) remain flagged.")
        return df, new_outlier_mask

    # -------------------------------------------------------------------------
    # Main logic
    # -------------------------------------------------------------------------
    df = df.copy()
    df.sort_values(by='time_local', inplace=True)

    if slope_col is None:
        slope_col = displacement_col

    #
    # Step A: Remove alignment/pressed/pos rows FIRST
    #
    alignment_mask = pd.Series(False, index=df.index)
    if alignment_col in df.columns:
        alignment_mask = df[alignment_col].isin(['Red'])

    pressed_mask = pd.Series(False, index=df.index)
    if pressed_col in df.columns:
        pressed_mask = df[pressed_col] == True

    # pos-diff check
    if pos_raw_col in df.columns and pos_avg_col in df.columns:
        df[pos_raw_col] = pd.to_numeric(df[pos_raw_col], errors='coerce')
        df[pos_avg_col] = pd.to_numeric(df[pos_avg_col], errors='coerce')
        df['pos_diff'] = (df[pos_raw_col] - df[pos_avg_col]).abs()
    else:
        df['pos_diff'] = 0.0

    pos_mask = df['pos_diff'] > pos_diff_threshold

    # Combine them
    combined_skip = alignment_mask | pressed_mask | pos_mask
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"[DEBUG] Removing {skip_count} row(s) due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()
    else:
        print("[DEBUG] No alignment/pressed/pos rows to remove.")

    # after removing those rows, reset index
    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    #
    # Step B: Now that pos outliers are removed, compute daily range on the *remaining* data
    #
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())

    # Clip each day's range to max 100 µm (or whatever)
    daily_ranges = daily_ranges.clip(upper=100)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"[DEBUG] After pos/alignment removal => daily range capped => {avg_daily_range:.3f} µm")

    #
    # Step C: Remove single-step jumps bigger than avg_daily_range
    #
    df.sort_values('time_local', inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    if big_jump_mask.any():
        count_big_jumps = big_jump_mask.sum()
        print(f"[DEBUG] Removing {count_big_jumps} row(s) with single-step jump > {avg_daily_range:.3f} µm.")
        df.loc[big_jump_mask, displacement_col] = np.nan
    else:
        print("[DEBUG] No single-step jumps exceed the daily range.")

    pre_drop_count = len(df)
    df.dropna(subset=[displacement_col], inplace=True)
    post_drop_count = len(df)
    if post_drop_count < pre_drop_count:
        print(f"[DEBUG] Dropped {pre_drop_count - post_drop_count} row(s) from big jumps.")
    df.reset_index(drop=True, inplace=True)

    #
    # Step D: Slope threshold (median, mean, Q3) => IQR-based outlier detection
    #
    median_slope, mean_slope, q3_slope = get_slope_stats_per_index(df, slope_col)
    if any(x is None for x in [median_slope, mean_slope, q3_slope]):
        small_slope_threshold = 2.0
        print("[WARN] Could not compute slope => fallback to 2.0")
    else:
        small_slope_threshold = q3_slope
        print(f"[DEBUG] Using largest slope => {small_slope_threshold:.3f}")

    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()
    large_diff_mask  = (df['Diff_Abs'] > small_slope_threshold)
    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']

    if large_diff_values.empty:
        print("[DEBUG] No diffs exceed threshold => skipping IQR logic.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]
        return df

    Q1 = large_diff_values.quantile(0.25)
    Q3 = large_diff_values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - multiplier * IQR)
    upper_bound = Q3 + multiplier * IQR
    print(f"[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")

    outlier_mask = df['Diff_Abs'] > upper_bound
    print(f"[DEBUG] Found {outlier_mask.sum()} outlier row(s) by IQR mask.")

    #
    # Step E: remove short outlier runs (< run_threshold)
    #
    df, outlier_mask = remove_short_outlier_runs(
        df, outlier_mask,
        displacement_col=displacement_col,
        upper_bound=upper_bound,
        run_threshold=run_threshold
    )

    #
    # Step F: baseline shift logic
    #
    df.reset_index(drop=True, inplace=True)
    outlier_mask = outlier_mask.reset_index(drop=True)
    df['Adjustment'] = 0.0
    total_shift = 0.0

    for i in range(len(df)):
        if outlier_mask[i]:
            if i == 0 and remove_first_outlier:
                df.loc[i, 'Difference'] = np.nan
                print("[DEBUG] First row outlier => removing difference.")
                continue
            jump = df.loc[i, 'Difference']
            total_shift += jump
        df.loc[i, 'Adjustment'] = total_shift

    df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    print(f"[DEBUG] Done. daily range = {avg_daily_range:.3f} after pos removal => big jumps => IQR => shift complete.")
    return df'''


'''def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2,
    slope_threshold=None
):
    """
    Identifies and adjusts outliers in the displacement data via:
      1) Remove alignment/pressed/pos-based outliers (entire rows).
      2) Compute 'capped' average daily range on the *remaining* data, remove big jumps.
      3) Compute or accept a slope threshold:
         - If slope_threshold is None, auto-compute Q3 of row-by-row abs slopes 
           from time windows (8–8:30, 19–19:30).
      4) IQR-based outlier detection on *absolute* differences, removing short outlier runs.
      5) Baseline shift logic for multi-point flagged outliers => 'Cleaned Displacement'.

    Parameters
    ----------
    df : pandas.DataFrame
        Main data. Must have 'time_local' (datetime) and 'um' columns.
    displacement_col : str
        The column with displacement data (default 'um').
    multiplier : float
        IQR multiplier (default 2.5).
    alignment_col, pressed_col, pos_raw_col, pos_avg_col : str
        For removing row-level outliers if alignment='Red', pressed=TRUE, 
        or |pos_raw-col - pos_avg_col|>pos_diff_threshold.
    pos_diff_threshold : float
        If (pos_raw - pos_avg).abs() > this => remove entire row.
    remove_first_outlier : bool
        If True, skip applying baseline shift if the first row is outlier.
    slope_col : str or None
        Column to use when computing slopes for Q3. If None, uses displacement_col.
    run_threshold : int
        Consecutive outlier run length needed to keep them. 
        Shorter runs are removed entirely.
    slope_threshold : float or None
        If None, auto-compute Q3 from row-by-row absolute slopes. 
        If set to a float, use that slope threshold instead.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with outlier-cleaned displacement in 'Cleaned Displacement'.
    """

    # -------------------------------------------------------------------------
    # Helper: compute median, mean, Q3 of row-by-row abs slopes
    #         in time windows [8:00–8:30, 19:00–19:30]
    # -------------------------------------------------------------------------
    def get_slope_stats(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time

        # Windows: 8:00–8:30, 19:00–19:30
        mask1_start = pd.to_datetime("08:00").time()
        mask1_end   = pd.to_datetime("08:30").time()
        mask1 = (df_local['time_of_day'] >= mask1_start) & (df_local['time_of_day'] < mask1_end)

        mask2_start = pd.to_datetime("19:00").time()
        mask2_end   = pd.to_datetime("19:30").time()
        mask2 = (df_local['time_of_day'] >= mask2_start) & (df_local['time_of_day'] < mask2_end)

        df_filtered = df_local[mask1 | mask2].sort_values('time_local').copy()
        if df_filtered.empty:
            print("[DEBUG] No data in the 8:00–8:30 or 19:00–19:30 windows => cannot compute slope stats.")
            return None, None, None

        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()

        if slope_series.empty:
            print("[DEBUG] slope_series is empty => returning None.")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)
        print(f"[DEBUG] Slope stats => Median={median_val:.3f}, Mean={mean_val:.3f}, Q3={q3_val:.3f}")
        return median_val, mean_val, q3_val

    # -------------------------------------------------------------------------
    # Helper: remove short runs of outliers (< run_threshold)
    # -------------------------------------------------------------------------
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

        # If ended in an outlier run
        if run_start is not None:
            run_end = n - 1
            length = run_end - run_start + 1
            if length < run_threshold:
                short_run_mask.iloc[run_start:run_end+1] = True

        df_local.loc[short_run_mask, disp_col] = np.nan
        pre_drop = len(df_local)
        df_local.dropna(subset=[disp_col], inplace=True)
        post_drop = len(df_local)
        removed_count = pre_drop - post_drop

        df_local.reset_index(drop=True, inplace=True)
        df_local['Difference'] = df_local[disp_col].diff()
        df_local['Diff_Abs']   = df_local['Difference'].abs()

        new_outlier_mask = (df_local['Diff_Abs'] > upper_bound)
        print(f"[DEBUG] Removed {removed_count} row(s) from short runs (< {run_threshold}).")
        print(f"[DEBUG] After short-run removal, {new_outlier_mask.sum()} row(s) remain flagged.")
        return df_local, new_outlier_mask

    # -------------------------------------------------------------------------
    # Main logic
    # -------------------------------------------------------------------------
    df = df.copy()
    df.sort_values(by='time_local', inplace=True)

    # If slope_col not provided, use displacement_col
    if slope_col is None:
        slope_col = displacement_col

    # Step 1: Remove alignment/pressed/pos outliers
    alignment_mask = pd.Series(False, index=df.index)
    if alignment_col in df.columns:
        alignment_mask = df[alignment_col].isin(['Red'])

    pressed_mask = pd.Series(False, index=df.index)
    if pressed_col in df.columns:
        pressed_mask = df[pressed_col] == True

    # pos-diff check
    if pos_raw_col in df.columns and pos_avg_col in df.columns:
        df[pos_raw_col] = pd.to_numeric(df[pos_raw_col], errors='coerce')
        df[pos_avg_col] = pd.to_numeric(df[pos_avg_col], errors='coerce')
        df['pos_diff'] = (df[pos_raw_col] - df[pos_avg_col]).abs()
    else:
        df['pos_diff'] = 0.0

    pos_mask = df['pos_diff'] > pos_diff_threshold
    combined_skip = alignment_mask | pressed_mask | pos_mask
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"[DEBUG] Removing {skip_count} row(s) due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()
    else:
        print("[DEBUG] No alignment/pressed/pos rows to remove.")

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 2: Compute daily range, remove big jumps
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())

    # Clip daily range
    daily_ranges = daily_ranges.clip(upper=200)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"[DEBUG] Capped daily range => {avg_daily_range:.3f} µm")

    df.sort_values('time_local', inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    if big_jump_mask.any():
        c_jumps = big_jump_mask.sum()
        print(f"[DEBUG] Removing {c_jumps} row(s) with single-step jump > {avg_daily_range:.3f} µm.")
        df.loc[big_jump_mask, displacement_col] = np.nan

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 3: slope threshold => user-provided or auto-compute Q3
    if slope_threshold is None:
        median_slope, mean_slope, q3_slope = get_slope_stats(df, slope_col)
        if any(x is None for x in [median_slope, mean_slope, q3_slope]):
            small_slope_threshold = 2.0
            print("[WARN] Could not compute slope => fallback=2.0")
            slope_source = "fallback=2.0"
        else:
            small_slope_threshold = median_slope
            slope_source = f"Q3={q3_slope:.3f}"
    else:
        small_slope_threshold = slope_threshold
        slope_source = f"user={slope_threshold:.3f}"

    print(f"[DEBUG] Using slope_threshold => {slope_source}")

    # Step 4: IQR-based outlier detection for diffs above slope_threshold
    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()
    large_diff_mask  = (df['Diff_Abs'] > small_slope_threshold)

    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']
    if large_diff_values.empty:
        print("[DEBUG] No diffs exceed slope_threshold => skipping IQR logic.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]
        return df

    Q1 = large_diff_values.quantile(0.25)
    Q3 = large_diff_values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - multiplier * IQR)
    upper_bound = Q3 + multiplier * IQR

    print(f"[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")
    outlier_mask = df['Diff_Abs'] > upper_bound
    outlier_count = outlier_mask.sum()
    print(f"[DEBUG] Found {outlier_count} outlier row(s) by IQR mask.")

    # remove short outlier runs
    df, outlier_mask = remove_short_outlier_runs(df, outlier_mask, displacement_col, upper_bound, run_threshold)

    # Step 5: baseline shift for remaining outliers
    df.reset_index(drop=True, inplace=True)
    outlier_mask = outlier_mask.reset_index(drop=True)

    df['Adjustment'] = 0.0
    total_shift = 0.0

    for i in range(len(df)):
        if outlier_mask[i]:
            # if first row outlier => skip shift
            if i == 0 and remove_first_outlier:
                df.loc[i, 'Difference'] = np.nan
                print("[DEBUG] First row is outlier => removing difference for that row.")
                continue
            jump = df.loc[i, 'Difference']
            total_shift += jump

        df.loc[i, 'Adjustment'] = total_shift

    df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    # Final debug print
    print(f"[DEBUG] Done. slope_threshold used => {slope_source}, IQR => short-run => baseline shift complete.")
    return df

'''

'''def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2,
    slope_threshold=None
):
    """
    Identifies and adjusts outliers in the displacement data via:
      1) Remove alignment/pressed/pos-based outliers (entire rows).
      2) Compute 'capped' average daily range on the *remaining* data, remove big jumps (Pass #1).
      3) Compute or accept a slope threshold (Q3 if none provided).
      4) IQR-based outlier detection on *absolute* differences, removing short outlier runs.
      5) Baseline shift logic for multi-point flagged outliers => 'Cleaned Displacement'.
      6) (NEW) Optionally re-check big jumps after baseline shift (Pass #2) and remove them.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with outlier-cleaned displacement in 'Cleaned Displacement'.
    """

    # -------------------------------------------------------
    # Helper: compute median, mean, Q3 of row-by-row abs slopes
    # -------------------------------------------------------
    def get_slope_stats(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time

        # Windows: 8:00–8:30, 19:00–19:30
        mask1_start = pd.to_datetime("08:00").time()
        mask1_end   = pd.to_datetime("08:30").time()
        mask1 = (df_local['time_of_day'] >= mask1_start) & (df_local['time_of_day'] < mask1_end)

        mask2_start = pd.to_datetime("19:00").time()
        mask2_end   = pd.to_datetime("19:30").time()
        mask2 = (df_local['time_of_day'] >= mask2_start) & (df_local['time_of_day'] < mask2_end)

        df_filtered = df_local[mask1 | mask2].sort_values('time_local').copy()
        if df_filtered.empty:
            print("[DEBUG] No data in the 8:00–8:30 or 19:00–19:30 windows => cannot compute slope stats.")
            return None, None, None

        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()

        if slope_series.empty:
            print("[DEBUG] slope_series is empty => returning None.")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)
        print(f"[DEBUG] Slope stats => Median={median_val:.3f}, Mean={mean_val:.3f}, Q3={q3_val:.3f}")
        return median_val, mean_val, q3_val

    # -------------------------------------------------------
    # Helper: remove short runs of outliers (< run_threshold)
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

        # If ended in an outlier run
        if run_start is not None:
            run_end = n - 1
            length = run_end - run_start + 1
            if length < run_threshold:
                short_run_mask.iloc[run_start:run_end+1] = True

        df_local.loc[short_run_mask, disp_col] = np.nan
        pre_drop = len(df_local)
        df_local.dropna(subset=[disp_col], inplace=True)
        post_drop = len(df_local)
        removed_count = pre_drop - post_drop

        df_local.reset_index(drop=True, inplace=True)
        df_local['Difference'] = df_local[disp_col].diff()
        df_local['Diff_Abs']   = df_local['Difference'].abs()

        new_outlier_mask = (df_local['Diff_Abs'] > upper_bound)
        print(f"[DEBUG] Removed {removed_count} row(s) from short runs (< {run_threshold}).")
        print(f"[DEBUG] After short-run removal, {new_outlier_mask.sum()} row(s) remain flagged.")
        return df_local, new_outlier_mask

    # -------------------------------------------------------
    # Main logic
    # -------------------------------------------------------
    df = df.copy()
    df.sort_values(by='time_local', inplace=True)

    # If slope_col not provided, use displacement_col
    if slope_col is None:
        slope_col = displacement_col

    # Step 1: remove alignment/pressed/pos outliers
    alignment_mask = pd.Series(False, index=df.index)
    if alignment_col in df.columns:
        alignment_mask = df[alignment_col].isin(['Red'])

    pressed_mask = pd.Series(False, index=df.index)
    if pressed_col in df.columns:
        pressed_mask = df[pressed_col] == True

    # pos-diff check
    if pos_raw_col in df.columns and pos_avg_col in df.columns:
        df[pos_raw_col] = pd.to_numeric(df[pos_raw_col], errors='coerce')
        df[pos_avg_col] = pd.to_numeric(df[pos_avg_col], errors='coerce')
        df['pos_diff'] = (df[pos_raw_col] - df[pos_avg_col]).abs()
    else:
        df['pos_diff'] = 0.0

    pos_mask = df['pos_diff'] > pos_diff_threshold
    combined_skip = alignment_mask | pressed_mask | pos_mask
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"[DEBUG] Removing {skip_count} row(s) due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()
    else:
        print("[DEBUG] No alignment/pressed/pos rows to remove.")

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 2: Compute daily range, remove big jumps (Pass #1)
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())

    # Clip daily range
    daily_ranges = daily_ranges.clip(upper=200)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"[DEBUG] First pass big-jump => daily range capped => {avg_daily_range:.3f} µm")

    df.sort_values('time_local', inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    if big_jump_mask.any():
        c_jumps = big_jump_mask.sum()
        print(f"[DEBUG] Removing {c_jumps} row(s) with single-step jump > {avg_daily_range:.3f} µm.")
        df.loc[big_jump_mask, displacement_col] = np.nan

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 3: slope threshold => user-provided or auto-compute Q3
    if slope_threshold is None:
        median_slope, mean_slope, q3_slope = get_slope_stats(df, slope_col)
        if any(x is None for x in [median_slope, mean_slope, q3_slope]):
            small_slope_threshold = 2.0
            print("[WARN] Could not compute slope => fallback=2.0")
            slope_source = "fallback=2.0"
        else:
            small_slope_threshold = q3_slope
            slope_source = f"Q3={q3_slope:.3f}"
    else:
        small_slope_threshold = slope_threshold
        slope_source = f"user={slope_threshold:.3f}"

    print(f"[DEBUG] Using slope_threshold => {slope_source}")

    # Step 4: IQR-based outlier detection for diffs above slope_threshold
    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()
    large_diff_mask  = (df['Diff_Abs'] > small_slope_threshold)

    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']
    if large_diff_values.empty:
        print("[DEBUG] No diffs exceed slope_threshold => skipping IQR logic.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]
    else:
        Q1 = large_diff_values.quantile(0.25)
        Q3 = large_diff_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - multiplier * IQR)
        upper_bound = Q3 + multiplier * IQR

        print(f"[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")
        outlier_mask = df['Diff_Abs'] > upper_bound
        outlier_count = outlier_mask.sum()
        print(f"[DEBUG] Found {outlier_count} outlier row(s) by IQR mask.")

        # remove short outlier runs
        df, outlier_mask = remove_short_outlier_runs(
            df, outlier_mask, displacement_col, upper_bound, run_threshold
        )

        # Step 5: baseline shift for remaining outliers
        df.reset_index(drop=True, inplace=True)
        outlier_mask = outlier_mask.reset_index(drop=True)

        df['Adjustment'] = 0.0
        total_shift = 0.0

        for i in range(len(df)):
            if outlier_mask[i]:
                # if first row outlier => skip shift
                if i == 0 and remove_first_outlier:
                    df.loc[i, 'Difference'] = np.nan
                    print("[DEBUG] First row is outlier => removing difference for that row.")
                    continue
                jump = df.loc[i, 'Difference']
                total_shift += jump

            df.loc[i, 'Adjustment'] = total_shift

        df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    # Step 6 (NEW): Final check => if any single-step jump STILL bigger than avg_daily_range
    # after baseline shift
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Difference'] = df['Cleaned Displacement'].diff()  # check the final, cleaned displacement
    final_big_jump_mask = df['Difference'].abs() > avg_daily_range*0.5
    if final_big_jump_mask.any():
        final_count = final_big_jump_mask.sum()
        print(f"[DEBUG] FINAL pass => removing {final_count} row(s) with single-step jump > {avg_daily_range:.3f} µm.")
        df.loc[final_big_jump_mask, 'Cleaned Displacement'] = np.nan

        # Drop them? Or just set to NaN? 
        # We'll drop them entirely so they won't confuse future analysis:
        pre_len = len(df)
        df.dropna(subset=['Cleaned Displacement'], inplace=True)
        post_len = len(df)
        print(f"[DEBUG] Dropped {pre_len - post_len} rows after final big-jump check.")

    # Final debug
    print(f"[DEBUG] Done. slope_threshold => {slope_source}, IQR => short-run => baseline shift => final big-jump check complete.")
    return df

'''
'''
def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2,
    slope_threshold=None
):
    """
    Identifies and adjusts outliers in the displacement data:
      0) (NEW) If there's a gap > 30 min (2x the normal 15-min interval),
         shift subsequent data so that the next record begins at the same
         displacement as the last known point.
      1) Remove alignment/pressed/pos-based outliers.
      2) Compute 'capped' average daily range on the *remaining* data, remove big jumps (Pass #1).
      3) Compute or accept slope threshold (Q3 if none).
      4) IQR-based outlier detection on absolute diffs, removing short outlier runs.
      5) Baseline shift logic for multi-point flagged outliers => 'Cleaned Displacement'.
      6) Optional re-check big jumps after baseline shift (Pass #2) and remove them.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with outlier-cleaned displacement in 'Cleaned Displacement'.
    """

    # -------------------------------------------------------
    # 0) Pre-step: collapse big time gaps by shifting subsequent data
    # -------------------------------------------------------
    def collapse_large_time_gaps(df_local, displacement_col, max_gap_minutes=30):
        """
        If the time gap between row i and i+1 is > max_gap_minutes,
        shift the entire subsequent block so that df.loc[i+1, displacement_col]
        becomes equal to df.loc[i, displacement_col].
        
        That means we effectively remove any jump that occurred in that gap.
        """
        df_local = df_local.sort_values('time_local').copy()
        df_local.reset_index(drop=True, inplace=True)

        for i in range(len(df_local) - 1):
            t_i = df_local.loc[i, 'time_local']
            t_next = df_local.loc[i+1, 'time_local']
            dt_minutes = (t_next - t_i).total_seconds() / 60.0

            if dt_minutes > max_gap_minutes:
                # We interpret "the next value (and all subsequent consecutive ones)
                # should begin at the same displacement as the last available one."
                # So we SHIFT all future rows down/up by the difference at i+1.
                old_val = df_local.loc[i+1, displacement_col]
                new_val = df_local.loc[i, displacement_col]
                shift_amount = old_val - new_val

                # Shift everything from i+1 onward
                df_local.loc[i+1:, displacement_col] -= shift_amount

                print(f"[DEBUG] Collapsed gap at index={i}, gap={dt_minutes:.1f} min. "
                      f"Shifted subsequent data by {shift_amount:.3f} µm.")
        return df_local

    # -------------------------------------------------------
    # Helper: compute slope stats
    # -------------------------------------------------------
    def get_slope_stats(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time

        # Windows: 8:00–8:30, 19:00–19:30
        mask1_start = pd.to_datetime("08:00").time()
        mask1_end   = pd.to_datetime("08:30").time()
        mask1 = (df_local['time_of_day'] >= mask1_start) & (df_local['time_of_day'] < mask1_end)

        mask2_start = pd.to_datetime("19:00").time()
        mask2_end   = pd.to_datetime("19:30").time()
        mask2 = (df_local['time_of_day'] >= mask2_start) & (df_local['time_of_day'] < mask2_end)

        df_filtered = df_local[mask1 | mask2].sort_values('time_local').copy()
        if df_filtered.empty:
            print("[DEBUG] No data in the 8:00–8:30 or 19:00–19:30 windows => cannot compute slope stats.")
            return None, None, None

        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()

        if slope_series.empty:
            print("[DEBUG] slope_series is empty => returning None.")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)
        print(f"[DEBUG] Slope stats => Median={median_val:.3f}, Mean={mean_val:.3f}, Q3={q3_val:.3f}")
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

        df_local.loc[short_run_mask, disp_col] = np.nan
        pre_drop = len(df_local)
        df_local.dropna(subset=[disp_col], inplace=True)
        post_drop = len(df_local)
        removed_count = pre_drop - post_drop

        df_local.reset_index(drop=True, inplace=True)
        df_local['Difference'] = df_local[disp_col].diff()
        df_local['Diff_Abs']   = df_local['Difference'].abs()

        new_outlier_mask = (df_local['Diff_Abs'] > upper_bound)
        print(f"[DEBUG] Removed {removed_count} row(s) from short runs (< {run_threshold}).")
        print(f"[DEBUG] After short-run removal, {new_outlier_mask.sum()} row(s) remain flagged.")
        return df_local, new_outlier_mask

    # -------------------------------------------------------
    # Main logic
    # -------------------------------------------------------
    df = df.copy()
    df.sort_values(by='time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # STEP 0: collapse large time gaps (>30 min => more than 1 missing 15-min record)
    # do this first
    df = collapse_large_time_gaps(df, displacement_col, max_gap_minutes=60)

    # If slope_col not provided, use displacement_col
    if slope_col is None:
        slope_col = displacement_col

    # Step 1: remove alignment/pressed/pos outliers
    alignment_mask = pd.Series(False, index=df.index)
    if alignment_col in df.columns:
        alignment_mask = df[alignment_col].isin(['Red'])

    pressed_mask = pd.Series(False, index=df.index)
    if pressed_col in df.columns:
        pressed_mask = df[pressed_col] == True

    # pos-diff check
    if pos_raw_col in df.columns and pos_avg_col in df.columns:
        df[pos_raw_col] = pd.to_numeric(df[pos_raw_col], errors='coerce')
        df[pos_avg_col] = pd.to_numeric(df[pos_avg_col], errors='coerce')
        df['pos_diff'] = (df[pos_raw_col] - df[pos_avg_col]).abs()
    else:
        df['pos_diff'] = 0.0

    pos_mask = df['pos_diff'] > pos_diff_threshold
    combined_skip = alignment_mask | pressed_mask | pos_mask
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"[DEBUG] Removing {skip_count} row(s) due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()
    else:
        print("[DEBUG] No alignment/pressed/pos rows to remove.")

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 2: Compute daily range, remove big jumps (Pass #1)
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())

    # Clip daily range
    daily_ranges = daily_ranges.clip(upper=200)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"[DEBUG] First pass big-jump => daily range capped => {avg_daily_range:.3f} µm")

    df.sort_values('time_local', inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    if big_jump_mask.any():
        c_jumps = big_jump_mask.sum()
        print(f"[DEBUG] Removing {c_jumps} row(s) with single-step jump > {avg_daily_range:.3f} µm.")
        df.loc[big_jump_mask, displacement_col] = np.nan

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 3: slope threshold => user-provided or auto-compute Q3
    if slope_threshold is None:
        median_slope, mean_slope, q3_slope = get_slope_stats(df, slope_col)
        if any(x is None for x in [median_slope, mean_slope, q3_slope]):
            small_slope_threshold = 2.0
            print("[WARN] Could not compute slope => fallback=2.0")
            slope_source = "fallback=2.0"
        else:
            small_slope_threshold = q3_slope
            slope_source = f"Q3={q3_slope:.3f}"
    else:
        small_slope_threshold = slope_threshold
        slope_source = f"user={slope_threshold:.3f}"

    print(f"[DEBUG] Using slope_threshold => {slope_source}")

    # Step 4: IQR-based outlier detection for diffs above slope_threshold
    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()
    large_diff_mask  = (df['Diff_Abs'] > small_slope_threshold)

    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']
    if large_diff_values.empty:
        print("[DEBUG] No diffs exceed slope_threshold => skipping IQR logic.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]
    else:
        Q1 = large_diff_values.quantile(0.25)
        Q3 = large_diff_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - multiplier * IQR)
        upper_bound = Q3 + multiplier * IQR

        print(f"[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")
        outlier_mask = df['Diff_Abs'] > upper_bound
        outlier_count = outlier_mask.sum()
        print(f"[DEBUG] Found {outlier_count} outlier row(s) by IQR mask.")

        # remove short outlier runs
        df, outlier_mask = remove_short_outlier_runs(
            df, outlier_mask, displacement_col, upper_bound, run_threshold
        )

        # Step 5: baseline shift for remaining outliers
        df.reset_index(drop=True, inplace=True)
        outlier_mask = outlier_mask.reset_index(drop=True)

        df['Adjustment'] = 0.0
        total_shift = 0.0

        for i in range(len(df)):
            if outlier_mask[i]:
                if i == 0 and remove_first_outlier:
                    df.loc[i, 'Difference'] = np.nan
                    print("[DEBUG] First row is outlier => removing difference for that row.")
                    continue
                jump = df.loc[i, 'Difference']
                total_shift += jump

            df.loc[i, 'Adjustment'] = total_shift

        df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    # Step 6 (NEW): Final check => if any single-step jump STILL bigger than avg_daily_range
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    if 'Cleaned Displacement' not in df.columns:
        # means we did no IQR, so let's define it for consistency
        df['Cleaned Displacement'] = df[displacement_col]

    df['Difference'] = df['Cleaned Displacement'].diff()
    final_big_jump_mask = df['Difference'].abs() > avg_daily_range
    if final_big_jump_mask.any():
        final_count = final_big_jump_mask.sum()
        print(f"[DEBUG] FINAL pass => removing {final_count} row(s) with single-step jump > {avg_daily_range:.3f} µm.")
        df.loc[final_big_jump_mask, 'Cleaned Displacement'] = np.nan

        # Drop them entirely
        pre_len = len(df)
        df.dropna(subset=['Cleaned Displacement'], inplace=True)
        post_len = len(df)
        print(f"[DEBUG] Dropped {pre_len - post_len} rows after final big-jump check.")

    print(f"[DEBUG] Done. slope_threshold => {slope_source}, IQR => short-run => baseline shift => final big-jump + gap collapse.")
    return df
'''



'''def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2,
    slope_threshold=None,
    max_gap_minutes=30  # If gap>30, we consider it "break in data"
):
    """
    Identifies and adjusts outliers in the displacement data, except we
    do NOT remove big gaps but rather 'collapse' them by shifting subsequent data
    so that the next record merges at the same displacement if the gap is large
    AND the difference is bigger than daily range.

    Steps:
      0) Collapse large time gaps without deleting (instead, shift subsequent data).
      1) Remove alignment/pressed/pos-based row outliers.
      2) Compute 'capped' average daily range on remaining data, remove big jumps (pass #1).
      3) Compute or accept slope threshold (Q3 if none).
      4) IQR-based outlier detection on *absolute* diffs, removing short outlier runs.
      5) Baseline shift logic => 'Cleaned Displacement'.
      6) Re-check big jumps & remove them if needed (optional).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with outlier-cleaned displacement in 'Cleaned Displacement'.
    """

    # -------------------------------------------------------
    # Helper: collapse large time gaps without dropping rows
    # -------------------------------------------------------
    def collapse_large_gaps_no_delete(df_local, disp_col, daily_range_cap, gap_minutes=30):
        """
        For each pair (i, i+1):
          - If (time gap>gap_minutes) AND the difference in displacement
            is > daily_range_cap, SHIFT subsequent data so that
            df.loc[i+1, disp_col]==df.loc[i, disp_col].
          - This ensures we do not interpret that jump as real.
        """
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
                    print(f"[DEBUG] Collapsed gap> {gap_minutes} min at index={i}, "
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
            print("[DEBUG] No data in the time windows => cannot compute slope stats.")
            return None, None, None

        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()
        if slope_series.empty:
            print("[DEBUG] slope_series empty => returning None.")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)
        print(f"[DEBUG] Slope stats => median={median_val:.3f}, mean={mean_val:.3f}, Q3={q3_val:.3f}")
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

        # last run
        if run_start is not None:
            run_end = n - 1
            length = run_end - run_start + 1
            if length < run_threshold:
                short_run_mask.iloc[run_start:run_end+1] = True

        df_local.loc[short_run_mask, disp_col] = np.nan
        pre_drop = len(df_local)
        df_local.dropna(subset=[disp_col], inplace=True)
        post_drop = len(df_local)
        removed_count = pre_drop - post_drop
        df_local.reset_index(drop=True, inplace=True)

        df_local['Difference'] = df_local[disp_col].diff()
        df_local['Diff_Abs']   = df_local['Difference'].abs()
        new_outlier_mask = (df_local['Diff_Abs'] > upper_bound)

        print(f"[DEBUG] Removed {removed_count} row(s) from short runs (< {run_threshold}).")
        print(f"[DEBUG] After short-run removal, {new_outlier_mask.sum()} row(s) remain flagged.")
        return df_local, new_outlier_mask

    # -------------------------------------------------------
    # Main function
    # -------------------------------------------------------
    df = df.copy().sort_values('time_local').reset_index(drop=True)

    # Step A: We'll compute the daily range (capped) *before* we collapse big gaps
    #         so we know how large the jump must be to trigger the collapse.

    # First, remove alignment/pressed/pos outliers
    alignment_mask = pd.Series(False, index=df.index)
    if alignment_col in df.columns:
        alignment_mask = df[alignment_col].isin(['Red'])

    pressed_mask = pd.Series(False, index=df.index)
    if pressed_col in df.columns:
        pressed_mask = df[pressed_col] == True

    # pos_diff check
    if pos_raw_col in df.columns and pos_avg_col in df.columns:
        df[pos_raw_col] = pd.to_numeric(df[pos_raw_col], errors='coerce')
        df[pos_avg_col] = pd.to_numeric(df[pos_avg_col], errors='coerce')
        df['pos_diff'] = (df[pos_raw_col] - df[pos_avg_col]).abs()
    else:
        df['pos_diff'] = 0.0

    combined_skip = alignment_mask | pressed_mask | (df['pos_diff']>pos_diff_threshold)
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"[DEBUG] Removing {skip_count} rows due to alignment/pressed/pos checks.")
        df = df.loc[~combined_skip].copy()
    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Compute a rough average daily range => to see how big a jump triggers "collapse"
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())
    daily_ranges = daily_ranges.clip(upper=50)
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"[DEBUG] daily_range => {avg_daily_range:.3f} µm")

    # Step 0: Now we collapse big time gaps if the displacement jump is > avg_daily_range
    df = collapse_large_gaps_no_delete(df, displacement_col, avg_daily_range, gap_minutes=max_gap_minutes)

    # Step 1: remove big jumps again if needed
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    if big_jump_mask.any():
        count_big_jumps = big_jump_mask.sum()
        print(f"[DEBUG] Removing {count_big_jumps} row(s) with single-step jump > {avg_daily_range:.3f} after gap collapse.")
        df.loc[big_jump_mask, displacement_col] = np.nan

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 2: slope threshold => user-provided or auto Q3
    if slope_col is None:
        slope_col = displacement_col

    if slope_threshold is None:
        median_slope, mean_slope, q3_slope = get_slope_stats(df, slope_col)
        if any(x is None for x in [median_slope, mean_slope, q3_slope]):
            small_slope_threshold = 2.0
            print("[WARN] fallback slope => 2.0")
            slope_source = "2.0"
        else:
            small_slope_threshold = q3_slope
            slope_source = f"Q3={q3_slope:.3f}"
    else:
        small_slope_threshold = slope_threshold
        slope_source = f"user={slope_threshold:.3f}"

    print(f"[DEBUG] slope threshold => {slope_source}")
    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()

    large_diff_mask = df['Diff_Abs'] > small_slope_threshold
    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']
    if large_diff_values.empty:
        print("[DEBUG] No diffs exceed slope => skip IQR.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]
    else:
        Q1 = large_diff_values.quantile(0.25)
        Q3 = large_diff_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - multiplier*IQR)
        upper_bound = Q3 + multiplier*IQR
        print(f"[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")

        outlier_mask = df['Diff_Abs']>upper_bound
        outlier_count = outlier_mask.sum()
        print(f"[DEBUG] Found {outlier_count} outliers by IQR mask.")
        df, outlier_mask = remove_short_outlier_runs(df, outlier_mask, displacement_col, upper_bound, run_threshold)

        # baseline shift
        df.reset_index(drop=True, inplace=True)
        outlier_mask = outlier_mask.reset_index(drop=True)
        df['Adjustment'] = 0.0
        total_shift=0.0
        for i in range(len(df)):
            if outlier_mask[i]:
                if i==0 and remove_first_outlier:
                    df.loc[i,'Difference']=np.nan
                    print("[DEBUG] first row => skip shift.")
                    continue
                jump = df.loc[i,'Difference']
                total_shift+= jump
            df.loc[i,'Adjustment'] = total_shift
        df['Cleaned Displacement'] = df[displacement_col]- df['Adjustment']

    # Step 6: final check
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)

    if 'Cleaned Displacement' not in df.columns:
        df['Cleaned Displacement']= df[displacement_col]

    df['Difference'] = df['Cleaned Displacement'].diff()
    final_mask = df['Difference'].abs() > avg_daily_range
    if final_mask.any():
        c = final_mask.sum()
        print(f"[DEBUG] final pass => removing {c} rows > daily range.")
        df.loc[final_mask, 'Cleaned Displacement']= np.nan
        pre= len(df)
        df.dropna(subset=['Cleaned Displacement'], inplace=True)
        post= len(df)
        print(f"[DEBUG] dropped {pre-post} final rows.")

    print(f"[DEBUG] done. slope={slope_source}, no-deletion collapse of gaps > {max_gap_minutes} min, outliers removed, shift done.")
    return df'''

'''def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2   # number of consecutive outliers required
):
    """
    Identifies and adjusts outliers in the displacement data, but includes
    logic to 'fix' rows where |pos_raw - pos_avg| exceeds pos_diff_threshold
    by recalculating displacement from pos_raw, using a local scale factor.

    Steps:
      1) Fix mismatch rows: if (pos_raw - pos_avg).abs() > pos_diff_threshold,
         compute a local ratio (µm per raw-count) from the previous few valid rows,
         recalc displacement from pos_raw * ratio (instead of removing).
      2) alignment/pressed checks remove certain rows entirely.
      3) Automatic slope calculation (median, mean, Q3) for small_slope_threshold.
      4) Single-step big-jump removal if a single diff > *capped* average daily range.
      5) IQR-based outlier detection on *absolute* differences, removing short outlier runs.
      6) Baseline shift logic for multi-point flagged outliers => 'Cleaned Displacement'.
    """

    # -------------------------------------------------------------------------
    # Helper: compute median, mean, Q3 of row-by-row abs slopes in time windows
    # -------------------------------------------------------------------------
    def get_slope_stats_per_index(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time

        mask1_start = pd.to_datetime("07:30").time()
        mask1_end   = pd.to_datetime("08:30").time()
        mask1 = (df_local['time_of_day'] >= mask1_start) & (df_local['time_of_day'] < mask1_end)

        mask2_start = pd.to_datetime("18:30").time()
        mask2_end   = pd.to_datetime("19:30").time()
        mask2 = (df_local['time_of_day'] >= mask2_start) & (df_local['time_of_day'] < mask2_end)

        df_filtered = df_local[mask1 | mask2].sort_values('time_local').copy()
        if df_filtered.empty:
            print("[DEBUG] No data in 07:30–08:30 or 18:30–19:30 windows.")
            return None, None, None

        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()

        if slope_series.empty:
            print("[DEBUG] No valid slope data (all NaN) in the time windows.")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)
        return median_val, mean_val, q3_val

    # -------------------------------------------------------------------------
    # Helper: remove short runs of outliers (< run_threshold)
    # -------------------------------------------------------------------------
    def remove_short_outlier_runs(df, outlier_mask, displacement_col, upper_bound, run_threshold):
        # Make sure indices match
        df.reset_index(drop=True, inplace=True)
        outlier_mask = outlier_mask.reset_index(drop=True)

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

        # If outlier run extends to the final row
        if run_start is not None:
            run_end = n - 1
            length = run_end - run_start + 1
            if length < run_threshold:
                short_run_mask.iloc[run_start:run_end+1] = True

        # Remove (NaN) these short-run outliers
        df.loc[short_run_mask, displacement_col] = np.nan
        pre_drop = len(df)
        df.dropna(subset=[displacement_col], inplace=True)
        post_drop = len(df)
        removed_count = pre_drop - post_drop

        df.reset_index(drop=True, inplace=True)
        df['Difference'] = df[displacement_col].diff()
        df['Diff_Abs']   = df['Difference'].abs()

        new_outlier_mask = (df['Diff_Abs'] > upper_bound)
        print(f"[DEBUG] Removed {removed_count} row(s) in short outlier runs (< {run_threshold}).")
        print(f"[DEBUG] After short-run removal, {new_outlier_mask.sum()} row(s) remain flagged.")
        return df, new_outlier_mask

    # -------------------------------------------------------------------------
    # Main logic
    # -------------------------------------------------------------------------
    df = df.copy()
    df.sort_values(by='time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)  # ensure a clean 0..N-1 index

    if slope_col is None:
        slope_col = displacement_col

    #
    # STEP A: compute daily range (capped) for big-jump removal
    #
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())
    daily_ranges = daily_ranges.clip(upper=100)  # or whatever cap
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"[DEBUG] Using daily range capped => {avg_daily_range:.3f} µm")

    #
    # STEP B0: "Fix" large pos_diff instead of removing them
    #
    if pos_raw_col in df.columns and pos_avg_col in df.columns:
        df[pos_raw_col] = pd.to_numeric(df[pos_raw_col], errors='coerce')
        df[pos_avg_col] = pd.to_numeric(df[pos_avg_col], errors='coerce')
    else:
        print("[DEBUG] Missing pos_raw or pos_avg columns => skipping mismatch logic.")
        df['pos_diff'] = 0.0

    df['pos_diff'] = (df[pos_raw_col] - df[pos_avg_col]).abs()
    mismatch_mask = df['pos_diff'] > pos_diff_threshold
    mismatch_mask = mismatch_mask.reset_index(drop=True)

    # We'll fix those mismatch rows by recomputing displacement from raw
    window_size = 5

    for i in range(len(df)):
        if mismatch_mask[i]:  
            start_idx = max(0, i - window_size)
            prev_df = df.loc[start_idx:i-1].copy()
            # exclude any row where pos_raw=NaN or displacement=NaN
            prev_df = prev_df.dropna(subset=[pos_raw_col, displacement_col])

            if len(prev_df) < 2:
                print(f"[WARN] Row {i}: pos_diff>threshold but insufficient prior data => NaN.")
                df.loc[i, displacement_col] = np.nan
                continue

            valid = prev_df[prev_df[pos_raw_col] != 0]
            if valid.empty:
                print(f"[WARN] Row {i}: pos_diff>threshold but pos_raw=0 => NaN.")
                df.loc[i, displacement_col] = np.nan
                continue

            ratio = (valid[displacement_col] / valid[pos_raw_col]).mean()
            new_disp = df.loc[i, pos_raw_col] * ratio
            print(f"[DEBUG] Row {i}: mismatch => recalc {displacement_col} from raw: {new_disp:.3f} µm")
            df.loc[i, displacement_col] = new_disp

    #
    # STEP B: remove alignment/pressed rows
    #
    alignment_mask = pd.Series(False, index=df.index)
    if alignment_col in df.columns:
        alignment_mask = df[alignment_col].isin(['Red'])

    pressed_mask = pd.Series(False, index=df.index)
    if pressed_col in df.columns:
        pressed_mask = df[pressed_col] == True

    combined_skip = alignment_mask | pressed_mask
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"[DEBUG] Removing {skip_count} row(s) due to alignment/pressed checks.")
        df = df.loc[~combined_skip].copy()
        df.reset_index(drop=True, inplace=True)

    #
    # STEP B2: slope threshold from median, mean, q3
    #
    median_slope, mean_slope, q3_slope = get_slope_stats_per_index(df, slope_col)
    if any(x is None for x in [median_slope, mean_slope, q3_slope]):
        small_slope_threshold = 2.0
        print("[WARN] Could not compute slope => fallback=2.0")
    else:
        small_slope_threshold = max(median_slope, mean_slope, q3_slope)
        print(f"[DEBUG] Using largest slope => {small_slope_threshold:.3f}")

    #
    # STEP C: remove single-step jumps > avg_daily_range
    #
    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    if big_jump_mask.any():
        c = big_jump_mask.sum()
        print(f"[DEBUG] Removing {c} row(s) with single-step jump > {avg_daily_range:.3f} µm.")
        df.loc[big_jump_mask, displacement_col] = np.nan

    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    #
    # STEP D: IQR-based detection on abs diffs
    #
    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()
    large_diff_mask  = (df['Diff_Abs'] > small_slope_threshold)
    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']

    if large_diff_values.empty:
        print("[DEBUG] No diffs exceed threshold => skipping IQR logic.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]
        return df

    Q1 = large_diff_values.quantile(0.25)
    Q3 = large_diff_values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - multiplier * IQR)
    upper_bound = Q3 + multiplier * IQR
    print(f"[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")

    outlier_mask = df['Diff_Abs'] > upper_bound
    print(f"[DEBUG] Found {outlier_mask.sum()} outlier row(s).")

    #
    # STEP E: remove short runs (< run_threshold)
    #
    df, outlier_mask = remove_short_outlier_runs(
        df, outlier_mask, displacement_col, upper_bound, run_threshold
    )

    #
    # STEP F: baseline shift logic
    #
    df.reset_index(drop=True, inplace=True)
    outlier_mask = outlier_mask.reset_index(drop=True)
    df['Adjustment'] = 0.0
    total_shift = 0.0

    for i in range(len(df)):
        if outlier_mask[i]:
            if i == 0 and remove_first_outlier:
                df.loc[i, 'Difference'] = np.nan
                print("[DEBUG] First row outlier => removing difference.")
                continue
            jump = df.loc[i, 'Difference']
            total_shift += jump
        df.loc[i, 'Adjustment'] = total_shift

    df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    print("[DEBUG] Done with mismatch fix, outlier removal, baseline shift.")
    return df'''



'''def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    remove_first_outlier=True,
    slope_col=None,
    run_threshold=2   # number of consecutive outliers required
):
    """
    1) Drop the first 10 rows to ignore 'installation' or initial data.
    2) If ANY row has |pos_raw - pos_avg| > pos_diff_threshold, recalc entire displacement
       from pos_raw using 0.4883 µm/count. The first valid row's displacement
       is kept from the original; subsequent rows increment by delta*0.4883,
       with wrap-around (4095→0 => +1, 0→4095 => -1).
    3) Remove alignment/pressed rows.
    4) Remove single-step jumps exceeding a 'capped' average daily range.
    5) Perform IQR-based outlier detection on large diffs, remove short runs (< run_threshold).
    6) Apply baseline-shift logic for multi-point outliers => 'Cleaned Displacement'.
    """

    # -------------------------------------------------------------------------
    # Helper: compute median, mean, Q3 of row-by-row abs slopes in time windows
    # -------------------------------------------------------------------------
    def get_slope_stats_per_index(df_local, disp_col):
        df_local = df_local.copy()
        df_local['time_of_day'] = df_local['time_local'].dt.time

        mask1_start = pd.to_datetime("07:30").time()
        mask1_end   = pd.to_datetime("08:30").time()
        mask1 = (df_local['time_of_day'] >= mask1_start) & (df_local['time_of_day'] < mask1_end)

        mask2_start = pd.to_datetime("18:30").time()
        mask2_end   = pd.to_datetime("19:30").time()
        mask2 = (df_local['time_of_day'] >= mask2_start) & (df_local['time_of_day'] < mask2_end)

        df_filtered = df_local[mask1 | mask2].sort_values('time_local').copy()
        if df_filtered.empty:
            print("[DEBUG] No data in 07:30–08:30 or 18:30–19:30 windows.")
            return None, None, None

        df_filtered['disp_diff'] = df_filtered[disp_col].diff()
        df_filtered['abs_slope'] = df_filtered['disp_diff'].abs()
        slope_series = df_filtered['abs_slope'].dropna()

        if slope_series.empty:
            print("[DEBUG] No valid slope data (all NaN) in the time windows.")
            return None, None, None

        median_val = slope_series.median()
        mean_val   = slope_series.mean()
        q3_val     = slope_series.quantile(0.75)
        return median_val, mean_val, q3_val

    # -------------------------------------------------------------------------
    # Helper: remove short runs of outliers (< run_threshold)
    # -------------------------------------------------------------------------
    def remove_short_outlier_runs(df, outlier_mask, displacement_col, upper_bound, run_threshold):
        df.reset_index(drop=True, inplace=True)
        outlier_mask = outlier_mask.reset_index(drop=True)

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

        # If outlier run extends to final row
        if run_start is not None:
            run_end = n - 1
            length = run_end - run_start + 1
            if length < run_threshold:
                short_run_mask.iloc[run_start:run_end+1] = True

        # Remove (NaN) these short-run outliers
        df.loc[short_run_mask, displacement_col] = np.nan
        pre_drop = len(df)
        df.dropna(subset=[displacement_col], inplace=True)
        post_drop = len(df)
        removed_count = pre_drop - post_drop

        df.reset_index(drop=True, inplace=True)
        df['Difference'] = df[displacement_col].diff()
        df['Diff_Abs']   = df['Difference'].abs()

        new_outlier_mask = (df['Diff_Abs'] > upper_bound)
        print(f"[DEBUG] Removed {removed_count} row(s) in short outlier runs (< {run_threshold}).")
        print(f"[DEBUG] After short-run removal, {new_outlier_mask.sum()} row(s) remain flagged.")
        return df, new_outlier_mask

    # -------------------------------------------------------------------------
    # Helper: recalc entire displacement from raw, preserving the first row's
    # existing displacement. All subsequent rows = prior disp + delta*0.4883
    # with wrap-around logic.
    # -------------------------------------------------------------------------
    def recalc_displacement_from_raw(df_local, displacement_col, pos_raw_col):
        """
        1) new_disp[0] = the old displacement_col[0]
        2) for i=1..N-1:
            if old_val=4095 & new_val=0 => delta=+1
            elif old_val=0 & new_val=4095 => delta=-1
            else => delta=(new_val - old_val)
            new_disp[i] = new_disp[i-1] + delta*0.4883
        """
        scale = 0.4883
        n = len(df_local)
        if n == 0:
            return df_local

        # Keep the original first displacement
        first_val = df_local.loc[0, displacement_col]
        df_local[displacement_col] = np.nan
        df_local.loc[0, displacement_col] = first_val

        for i in range(1, n):
            old_val = df_local.loc[i-1, pos_raw_col]
            new_val = df_local.loc[i, pos_raw_col]
            prev_disp = df_local.loc[i-1, displacement_col]

            if pd.isna(old_val) or pd.isna(new_val) or pd.isna(prev_disp):
                df_local.loc[i, displacement_col] = np.nan
                continue

            # handle wrap-around
            if old_val == 4095 and new_val == 0:
                delta = +1
            elif old_val == 0 and new_val == 4095:
                delta = -1
            else:
                delta = new_val - old_val

            df_local.loc[i, displacement_col] = prev_disp + (delta * scale)

        return df_local

    # -------------------------------------------------------------------------
    # Main logic
    # -------------------------------------------------------------------------
    df = df.copy()
    df.sort_values(by='time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 1) Drop the first 10 rows (ignore 'installation')
    if len(df) <= 10:
        print("[DEBUG] Data has fewer than 10 rows => returning empty after drop.")
        return df.iloc[0:0].copy()  # empty
    else:
        df = df.iloc[10:].copy()
        df.reset_index(drop=True, inplace=True)
        print("[DEBUG] Dropped the first 10 rows to ignore installation phase.")

    if slope_col is None:
        slope_col = displacement_col

    # Convert pos_raw & pos_avg to numeric
    if pos_raw_col in df.columns and pos_avg_col in df.columns:
        df[pos_raw_col] = pd.to_numeric(df[pos_raw_col], errors='coerce')
        df[pos_avg_col] = pd.to_numeric(df[pos_avg_col], errors='coerce')
        df['pos_diff'] = (df[pos_raw_col] - df[pos_avg_col]).abs()
    else:
        print("[WARN] Missing pos_raw or pos_avg => cannot check mismatch.")
        df['pos_diff'] = 0.0

    # STEP A: If ANY mismatch => recalc entire displacement from raw
    mismatch_exists = (df['pos_diff'] > pos_diff_threshold).any()
    if mismatch_exists:
        print(f"[DEBUG] Found row(s) with |pos_raw - pos_avg| > {pos_diff_threshold} => Recalc entire displacement from raw.")
        df = recalc_displacement_from_raw(df, displacement_col, pos_raw_col)

    # STEP B: remove alignment/pressed
    alignment_mask = pd.Series(False, index=df.index)
    if alignment_col in df.columns:
        alignment_mask = df[alignment_col].isin(['Red'])

    pressed_mask = pd.Series(False, index=df.index)
    if pressed_col in df.columns:
        pressed_mask = df[pressed_col] == True

    combined_skip = alignment_mask | pressed_mask
    if combined_skip.any():
        skip_count = combined_skip.sum()
        print(f"[DEBUG] Removing {skip_count} row(s) due to alignment/pressed checks.")
        df = df.loc[~combined_skip].copy()
        df.reset_index(drop=True, inplace=True)

    # STEP C: compute daily range (capped) & remove single-step big jumps
    df_temp = df.copy()
    df_temp['date'] = df_temp['time_local'].dt.date
    daily_ranges = df_temp.groupby('date')[displacement_col].agg(lambda x: x.max() - x.min())
    daily_ranges = daily_ranges.clip(upper=100)  # or any cap
    avg_daily_range = daily_ranges.mean() if not daily_ranges.empty else 0.0
    print(f"[DEBUG] avg_daily_range (capped) => {avg_daily_range:.3f} µm")

    df.sort_values('time_local', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Difference'] = df[displacement_col].diff()
    big_jump_mask = df['Difference'].abs() > avg_daily_range
    if big_jump_mask.any():
        c = big_jump_mask.sum()
        print(f"[DEBUG] Removing {c} row(s) with single-step jump > {avg_daily_range:.3f} µm.")
        df.loc[big_jump_mask, displacement_col] = np.nan
    df.dropna(subset=[displacement_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # STEP D: IQR-based detection
    df['Difference'] = df[displacement_col].diff()
    df['Diff_Abs']   = df['Difference'].abs()

    median_slope, mean_slope, q3_slope = get_slope_stats_per_index(df, slope_col)
    if any(x is None for x in [median_slope, mean_slope, q3_slope]):
        small_slope_threshold = 2.0
        print("[WARN] Could not compute slope => fallback=2.0")
    else:
        small_slope_threshold = max(median_slope, mean_slope, q3_slope)
        print(f"[DEBUG] Using largest slope => {small_slope_threshold:.3f}")

    large_diff_mask  = (df['Diff_Abs'] > small_slope_threshold)
    large_diff_values = df.loc[large_diff_mask, 'Diff_Abs']
    if large_diff_values.empty:
        print("[DEBUG] No diffs exceed threshold => skipping IQR logic.")
        df['Adjustment'] = 0.0
        df['Cleaned Displacement'] = df[displacement_col]
        return df

    Q1 = large_diff_values.quantile(0.25)
    Q3 = large_diff_values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - multiplier * IQR)
    upper_bound = Q3 + multiplier * IQR
    print(f"[DEBUG] IQR => lower={lower_bound:.3f}, upper={upper_bound:.3f}")

    outlier_mask = df['Diff_Abs'] > upper_bound
    print(f"[DEBUG] Found {outlier_mask.sum()} outlier row(s).")

    # STEP E: remove short outlier runs
    df, outlier_mask = remove_short_outlier_runs(df, outlier_mask, displacement_col, upper_bound, run_threshold)

    # STEP F: baseline shift
    df.reset_index(drop=True, inplace=True)
    outlier_mask = outlier_mask.reset_index(drop=True)
    df['Adjustment'] = 0.0
    total_shift = 0.0

    for i in range(len(df)):
        if outlier_mask[i]:
            if i == 0 and remove_first_outlier:
                df.loc[i, 'Difference'] = np.nan
                print("[DEBUG] First row is outlier => removing difference for that row.")
                continue
            jump = df.loc[i, 'Difference']
            total_shift += jump
        df.loc[i, 'Adjustment'] = total_shift

    df['Cleaned Displacement'] = df[displacement_col] - df['Adjustment']

    print("[DEBUG] Done. Dropped first 10 rows, possibly recalculated displacement from raw, removed big jumps, done IQR + baseline shift.")
    return df'''

def detrend_data(df, column='cleaned_um', window_size=96):
    df_out = df.copy()
    if column not in df_out.columns:
        print(f"WARNING: {column} not in DataFrame. Skipping detrend.")
        df_out['detrended'] = np.nan
        return df_out
    
    rolling_mean = df_out[column].rolling(window=window_size, min_periods=1).mean()
    df_out['detrended'] = df_out[column] - rolling_mean
    
    return df_out
# this is a smoothing function. I used it befgore fitting a curve so don't worry about it
def resample_30T_keep_structure(df):
    """
    Resamples df to 30-minute intervals, 
    calculating the mean for numeric columns (including Cleaned Displacement).
    """
    # Ensure df is time-indexed
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample_30T_keep_structure requires a DatetimeIndex.")

    # Mean of numeric columns every 30 minutes
    df_30min = df.resample('30T').mean(numeric_only=True)
    return df_30min

def calculate_daily_range_cleaned(df):
    """
    Aggregates the cleaned displacement daily, computing max-min for each day.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'time_local' (datetime) and 'Cleaned Displacement'.

    Returns
    -------
    daily_df : pandas.DataFrame
        Daily aggregates with columns: ['time_local', 'max', 'min', 'daily_range'].
    """
    daily_df = df.set_index('time_local').resample('D')['Cleaned Displacement'].agg(['max', 'min'])
    daily_df['daily_range'] = daily_df['max'] - daily_df['min']
    return daily_df.reset_index()



def compute_daily_range_and_max_vpd(
    df,
    dev_name,
    crop_label,    # e.g. "Blueberry" or "Hazelnut"
    year_label,    # e.g. 2023 or 2024
    time_col='time_local',
    disp_col='Cleaned Displacement',
    vpd_col='VPD',
    zero_each_sensor=True
):
    """
    Given a single sensor's DataFrame with time, displacement, and VPD,
    DETREND by assuming the difference in consecutive daily maxima is
    the daily growth or shrinkage.

    Then compute daily_range (max-min) of that detrended displacement
    and also daily_max_vpd.

    Returns a DataFrame with columns:
    [Date, daily_range, daily_max_vpd, Crop, Year, Device]

    Steps:
      1) Convert time_col to datetime if needed.
      2) (Optionally) Zero the displacement so it starts at 0.
      3) First daily aggregation => get each day's (un-detrended) max displacement.
      4) daily_growth = difference between consecutive daily maxima
         => cumulative_growth = cumsum(daily_growth).
      5) Shift the displacement of day D by cumulative_growth[D],
         so that day D's maximum lines up with day (D-1)'s maximum.
      6) Second daily aggregation => compute daily_range = max-min,
         plus daily_max_vpd = max(VPD).
      7) Return final DF with meta columns for crop, year, device.
    """

    import pandas as pd
    import numpy as np

    df = df.copy()

    # -----------------------------
    # A) Ensure datetime
    # -----------------------------
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DF.")
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # Drop rows with no time or disp
    df.dropna(subset=[time_col, disp_col], inplace=True)

    # Sort by time
    df.sort_values(time_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # -----------------------------
    # B) Optionally zero each sensor
    # -----------------------------
    if zero_each_sensor:
        first_val = df[disp_col].iloc[0]
        df[disp_col] = df[disp_col] - first_val

    # -----------------------------
    # C) First daily aggregator => original daily max
    # -----------------------------
    df['DayDate'] = df[time_col].dt.floor('D')  # each day's date
    daily_agg1 = df.groupby('DayDate', as_index=False).agg({
        disp_col: 'max'
    })
    daily_agg1.rename(columns={disp_col: 'orig_daily_max'}, inplace=True)

    # Sort by day
    daily_agg1.sort_values('DayDate', inplace=True, ignore_index=True)

    # Compute day-to-day growth: difference in consecutive daily max
    daily_agg1['daily_growth'] = daily_agg1['orig_daily_max'].diff().fillna(0)

    # Cumulative sum => how much we need to shift day N
    # so that day N's max lines up with day (N-1)'s max
    daily_agg1['cumulative_shift'] = daily_agg1['daily_growth'].cumsum()

    # Build a small lookup from DayDate -> cumulative_shift
    shift_map = dict(
        zip(daily_agg1['DayDate'], daily_agg1['cumulative_shift'])
    )

    # -----------------------------
    # D) Shift each day's data by that day's cumulative_shift
    # -----------------------------
    # So if day N is bigger by +2 µm vs. prior day, we subtract 2 µm
    # from all data on day N to remove that "growth" trend.
    df['DailyShift'] = df['DayDate'].map(shift_map)  # map each row to day shift
    df['DetrendedDisp'] = df[disp_col] - df['DailyShift']

    # -----------------------------
    # E) Second daily aggregator => compute daily_range, daily_max_vpd
    #     from the *DETRENDED* displacement
    # -----------------------------
    daily_agg2 = df.groupby('DayDate', as_index=False).agg({
        'DetrendedDisp': ['min','max'],
        vpd_col: 'max'
    })
    daily_agg2.columns = ['DayDate','detrended_min','detrended_max','daily_max_vpd']

    daily_agg2['daily_range'] = daily_agg2['detrended_max'] - daily_agg2['detrended_min']

    # Rename to "Date"
    daily_agg2.rename(columns={'DayDate': 'Date'}, inplace=True)

    # -----------------------------
    # F) Attach meta columns => Crop, Year, Device
    # -----------------------------
    daily_agg2['Crop'] = crop_label
    daily_agg2['Year'] = year_label
    daily_agg2['Device'] = dev_name

    # Keep final columns => done
    return daily_agg2[['Date','daily_range','daily_max_vpd','Crop','Year','Device']]