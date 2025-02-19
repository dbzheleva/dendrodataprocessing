"""
data_processing.py

Contains data cleaning, outlier detection, and transformations (e.g., 'Cleaned Displacement').
"""

import pandas as pd
import numpy as np

def identify_and_adjust_outliers(
    df,
    displacement_col='um',
    multiplier=1.5,
    alignment_col='Alignment',
    pressed_col='Pressed?',
    pos_raw_col='pos_raw',
    pos_avg_col='pos_avg',
    pos_diff_threshold=5,
    small_slope_threshold=6,
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