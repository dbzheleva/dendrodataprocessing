"""
data_plotting.py

Contains various plotting functions for dendrometer data:
 - Original vs. cleaned displacement
 - Alignment overlays
 - VPD overlays
 - Multiple sensor comparisons
 - Zoom-in plots with peaks/valleys
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
import numpy as np

def plot_cleaned_displacement(df, device_name):
    """
    Plots the original 'um' displacement vs. the cleaned displacement side-by-side.
    """
    instance_name = df['instance'].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Original
    ax[0].plot(df['time_local'], df['um'], label='Original')
    ax[0].set_title(f'{device_name} | {instance_name} | Original Displacement')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Displacement (um)')

    # Cleaned
    ax[1].plot(df['time_local'], df['Cleaned Displacement'], color='red', label='Cleaned')
    ax[1].set_title(f'{device_name} | {instance_name} | Cleaned Displacement')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Displacement (um)')

    for axis in ax:
        labels = axis.get_xticklabels()
        plt.setp(labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def plot_original_vs_cleaned(df, device_name="Unknown"):
    """
    Overlays original displacement vs. cleaned displacement on one plot,
    labeling the plot with the provided device_name.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(df['time_local'], df['um'], label='Original', color='blue')
    plt.plot(df['time_local'], df['Cleaned Displacement'], label='Cleaned', color='red', linestyle='--')

    plt.xlabel('Time')
    plt.ylabel('Displacement (um)')
    plt.title(f'{device_name}: Original vs. Cleaned Displacement')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_original_vs_cleaned2(df_raw, df_cleaned, device_name="Unknown"):
    """
    Overlays original displacement vs cleaned displacement on one plot.
    Styled entirely via global rcParams for consistency.
    """
    plt.figure()   # no figsize → inherits global style

    # Use default color cycle and line widths from rcParams
    plt.plot(df_raw['time_local'], df_raw['um'], label='Original')
    plt.plot(df_cleaned['time_local'], df_cleaned['Cleaned Displacement'],
             label='Cleaned', linestyle='--')

    plt.xlabel("Time")
    plt.ylabel("Displacement (µm)")
    plt.title(f"{device_name}: Original vs. Cleaned Displacement")

    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_displacement_alignment(df, device_name):
    """
    Plots displacement over time, coloring background by alignment_mapped (Green, Yellow, Red, Error).
    """
    min_date = df['time_local'].min().strftime('%Y-%m-%d')
    max_date = df['time_local'].max().strftime('%Y-%m-%d')
    instance_name = df['instance'].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df['time_local'], df['um'], label='Displacement')

    # Fill background based on alignment
    ax.fill_between(df['time_local'], df['um'].max(), df['um'].min(),
                    where=(df['alignment_mapped'] == 0),
                    color='green', alpha=0.3, label='Green Alignment')

    ax.fill_between(df['time_local'], df['um'].max(), df['um'].min(),
                    where=(df['alignment_mapped'] == 1),
                    color='yellow', alpha=0.3, label='Yellow Alignment')

    ax.fill_between(df['time_local'], df['um'].max(), df['um'].min(),
                    where=(df['alignment_mapped'] == 2),
                    color='red', alpha=0.3, label='Red Alignment')

    ax.fill_between(df['time_local'], df['um'].max(), df['um'].min(),
                    where=(df['alignment_mapped'] == 3),
                    color='purple', alpha=0.3, label='Error')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.set_title(f'{device_name} | {instance_name}\n{min_date} to {max_date}')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_displacement_and_vpd(
    df,
    device_name=None,
    start_time=None,
    end_time=None,
    time_col='time_local',
    displacement_col='Cleaned Displacement',
    vpd_col='VPD'
):
    """
    Plots cleaned displacement on the primary y-axis and VPD on the secondary y-axis
    over an optional time window [start_time, end_time].
    
    Parameters
    ----------
    df : pd.DataFrame
        Data must include at least 'time_local', 'Cleaned Displacement', and 'VPD'.
    device_name : str
        String to show in the plot title (e.g. "H4").
    start_time : str or pd.Timestamp or None
        If not None, the lower bound for the time axis (e.g. '2024-07-08 08:30').
    end_time : str or pd.Timestamp or None
        If not None, the upper bound for the time axis (e.g. '2024-07-10 08:30').
    time_col : str
        Name of the datetime column in `df`. Default 'time_local'.
    displacement_col : str
        Name of the displacement column. Default 'Cleaned Displacement'.
    vpd_col : str
        Name of the VPD column. Default 'VPD'.
    """

    # Ensure time_col is actually datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # Filter by start_time/end_time if provided
    if start_time is not None:
        df = df[df[time_col] >= pd.to_datetime(start_time)]
    if end_time is not None:
        df = df[df[time_col] <= pd.to_datetime(end_time)]

    df.sort_values(by=time_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Check if there's any data left
    if df.empty:
        print("No data in the specified time range.")
        return

    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot the displacement on the primary y-axis
    ax1.plot(df[time_col], df[displacement_col], 
             color='blue', label='Displacement', 
             linestyle='-', marker='', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel(f'{displacement_col} (µm)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Title
    if device_name:
        ax1.set_title(f'{device_name}: {displacement_col} and {vpd_col}')
    else:
        ax1.set_title(f'{displacement_col} and {vpd_col}')

    # Plot the VPD on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df[time_col], df[vpd_col], 
             color='orange', label='VPD', 
             linestyle='--', marker='', linewidth=1.8, alpha=0.9)
    ax2.set_ylabel(f'{vpd_col} (kPa)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Format the x-axis with datetime ticks
    ax1.set_xlabel("Time")
    # Optionally set a date formatter & locator (e.g. tick every 6h):
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

    # Combine legends from both lines
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.show()



def plot_multiple_sensors_displacement_raw(sensors_data, sensor_names):
    """
    Plots cleaned displacement for multiple sensors on the same figure.

    Parameters
    ----------
    sensors_data : list of pandas.DataFrame
        Each DataFrame must have 'time_local' and 'Cleaned Displacement'.
    sensor_names : list of str
        Names for labeling each sensor in the legend.
    """
    plt.figure(figsize=(15, 6))

    for df, name in zip(sensors_data, sensor_names):
        plt.plot(df['time_local'], df['um'], label=f'{name}')

    plt.xlabel('Time')
    plt.ylabel('Raw Displacement (um)')
    plt.title('Raw Displacement for Multiple Sensors')
    plt.legend(
    loc='upper left',               # anchor the legend's corner
    bbox_to_anchor=(1.02, 1.0),     # x=1.02 moves it just outside the plot
    borderaxespad=0.0,              # no extra padding
    fontsize='small') 
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_multiple_sensors_displacement_cleaned(sensors_data, sensor_names):
    """
    Plots cleaned displacement for multiple sensors on the same figure.

    Parameters
    ----------
    sensors_data : list of pandas.DataFrame
        Each DataFrame must have 'time_local' and 'Cleaned Displacement'.
    sensor_names : list of str
        Names for labeling each sensor in the legend.
    """
    plt.figure(figsize=(15, 6))

    for df, name in zip(sensors_data, sensor_names):
        plt.plot(df['time_local'], df['Cleaned Displacement'], label=f'{name}')

    plt.xlabel('Time')
    plt.ylabel('Cleaned Displacement (um)')
    plt.title('Cleaned Displacement for Multiple Sensors')
    plt.legend(
    loc='upper left',               # anchor the legend's corner
    bbox_to_anchor=(1.02, 1.0),     # x=1.02 moves it just outside the plot
    borderaxespad=0.0,              # no extra padding
    fontsize='small') 
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multiple_sensors_displacement_aligned(sensors_data, sensor_names):
    """
    Plots cleaned displacement for multiple sensors on the same figure,
    aligning the initial displacement of each sensor to zero.

    Parameters
    ----------
    sensors_data : list of pandas.DataFrame
        Each DataFrame must have 'time_local' and 'Cleaned Displacement'.
    sensor_names : list of str
        Names for labeling each sensor in the legend.
    """
    plt.figure(figsize=(15, 6))

    for df, name in zip(sensors_data, sensor_names):
        if df.empty:
            print(f"[WARNING] DataFrame for {name} is empty; skipping.")
            continue
        
        # Subtract this sensor's initial displacement from its entire series
        initial_value = df['Cleaned Displacement'].iloc[0]
        df['Aligned Displacement'] = df['Cleaned Displacement'] - initial_value

        plt.plot(
            df['time_local'],
            df['Aligned Displacement'],
            label=f"{name}"
        )

    plt.xlabel('Time')
    plt.ylabel('Aligned Displacement (µm)')
    plt.title('Cleaned Displacement (Aligned at Initial=0) for Multiple Sensors')
    plt.legend(
    loc='upper left',               # anchor the legend's corner
    bbox_to_anchor=(1.02, 1.0),     # x=1.02 moves it just outside the plot
    borderaxespad=0.0,              # no extra padding
    fontsize='small')                # make the text smaller

    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_displacement_for_period(df, device_name, start_date, end_date):
    """
    Zooms in on a specific date range for Cleaned Displacement.
    """
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    filtered_df = df[(df['time_local'] >= start_ts) & (df['time_local'] <= end_ts)]

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['time_local'], filtered_df['Cleaned Displacement Zeroed'], label='Displacement', color='blue')

    plt.xlabel('Time')
    plt.ylabel('Cleaned Displacement (um)')
    plt.title(f'Displacement from {start_date} to {end_date} - {device_name}')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_daily_range_vs_temp_vpd_separate(df_agg, device_name):
    """
    Plots daily range vs temperature in one figure, and daily range vs VPD in another figure.
    Each includes a best-fit regression line and annotated R².
    """

    # -- 1) Daily Range vs Max Temperature --
    plt.figure(figsize=(7, 6))
    sns.regplot(
        x='daily_temp_max',
        y='daily_range',
        data=df_agg,
        scatter_kws={'alpha': 0.6},
        line_kws={'color': 'red'}
    )
    slope, intercept, r_value, p_value, std_err = linregress(
        df_agg['daily_temp_max'].dropna(),
        df_agg['daily_range'].dropna()
    )
    plt.text(
        0.05, 0.9,
        f'R² = {r_value**2:.2f}',
        transform=plt.gca().transAxes,
        fontsize=12
    )
    plt.title(f'Maximum Daily Shrinkage vs Max Temp\n({device_name})')
    plt.xlabel('Max Temperature (°C)')
    plt.ylabel('Maximum Daily Shrinkage (µm)')
    plt.tight_layout()
    plt.show()

    # -- 2) Daily Range vs Max VPD --
    plt.figure(figsize=(7, 6))
    sns.regplot(
        x='daily_vpd_max',
        y='daily_range',
        data=df_agg,
        scatter_kws={'alpha': 0.6},
        line_kws={'color': 'red'}
    )
    slope, intercept, r_value, p_value, std_err = linregress(
        df_agg['daily_vpd_max'].dropna(),
        df_agg['daily_range'].dropna()
    )
    plt.text(
        0.05, 0.9,
        f'R² = {r_value**2:.2f}',
        transform=plt.gca().transAxes,
        fontsize=12
    )
    plt.title(f'Maximum Daily Shrinkage vs Max VPD\n({device_name})')
    plt.xlabel('Max VPD (kPa)')
    plt.ylabel('Maximum Daily Shrinkage (µm)')
    plt.tight_layout()
    plt.show()
    
def plot_group_daily_range_and_env(group_dfs, group_names):
    """
    For each dataframe (sensor) in the group:
      1) Aggregate daily metrics using `aggregate_environmental_data()`.
      2) Plot the daily range, daily_temp_max, daily_vpd_max 
         on 3 subplots, one figure, all sensors together.

    Parameters
    ----------
    group_dfs : list of pd.DataFrame
        DataFrames for each sensor in the group (already time-indexed or time_local column).
    group_names : list of str
        Corresponding sensor names (e.g. "BB1", "LB2", etc.).
    """

    # 1) Aggregate daily data for each sensor
    daily_data_list = []
    for df, name in zip(group_dfs, group_names):
        # Calculate daily metrics
        daily_env_df = aggregate_environmental_data(df)
        # Store as tuple: (sensor_name, daily dataframe)
        daily_data_list.append((name, daily_env_df))

    # 2) Create a single figure with 3 subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

    # --- Subplot 1: Daily Range (Cleaned Displacement) ---
    for sensor_name, daily_df in daily_data_list:
        axes[0].plot(daily_df['time_local'], daily_df['daily_range'], label=sensor_name)
    axes[0].set_title('Maximum Daily Shrinkage (Cleaned Displacement)')
    axes[0].set_ylabel('Maximum Daily Shrinkage (µm)')
    axes[0].legend()
    axes[0].grid(True)

    # --- Subplot 2: Daily Max Temperature ---
    for sensor_name, daily_df in daily_data_list:
        axes[1].plot(daily_df['time_local'], daily_df['daily_temp_max'], label=sensor_name)
    axes[1].set_title('Daily Max Temperature')
    axes[1].set_ylabel('Temperature (°C)')  # Or any relevant unit
    axes[1].legend()
    axes[1].grid(True)

    # --- Subplot 3: Daily Max VPD ---
    for sensor_name, daily_df in daily_data_list:
        axes[2].plot(daily_df['time_local'], daily_df['daily_vpd_max'], label=sensor_name)
    axes[2].set_title('Daily Max VPD')
    axes[2].set_ylabel('VPD')
    axes[2].legend()
    axes[2].grid(True)

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()
    
def plot_zoom(
    df, 
    start_time, 
    end_time, 
    raw_col='um', 
    cleaned_col='Cleaned Displacement',
    time_col='time_local'
):
    """
    Plots a zoomed-in view of `raw_col` and `cleaned_col` between start_time and end_time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must include columns for `raw_col`, `cleaned_col`, and a datetime column `time_col`.
    start_time : str or pd.Timestamp
        Lower bound for the time axis (e.g. '2023-09-14 00:00').
    end_time : str or pd.Timestamp
        Upper bound for the time axis (e.g. '2023-09-17 00:00').
    raw_col : str
        Name of the original displacement column.
    cleaned_col : str
        Name of the cleaned displacement column.
    time_col : str
        Name of the DataFrame column containing datetimes.
    """

    # Make sure time_col is in datetime format
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # Filter the data to the specified time window
    mask = (df[time_col] >= pd.to_datetime(start_time)) & (df[time_col] <= pd.to_datetime(end_time))
    df_zoom = df.loc[mask].sort_values(time_col)

    if df_zoom.empty:
        print("No data in the specified time range.")
        return

    # Plot
    plt.figure(figsize=(10,5))
    
    # Plot the raw displacement
    plt.plot(
        df_zoom[time_col],
        df_zoom[raw_col],
        label=raw_col,
        marker='o',
        linestyle='-',
        alpha=0.6
    )

    # Optionally plot the cleaned displacement if it exists
    if cleaned_col in df_zoom.columns:
        plt.plot(
            df_zoom[time_col],
            df_zoom[cleaned_col],
            label=cleaned_col,
            marker='s',
            linestyle='--',
            alpha=0.9
        )

    plt.title(f"Zoomed Plot from {start_time} to {end_time}")
    plt.xlabel("Time")
    plt.ylabel("Displacement (µm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_subgroup_cleaned_data(
    dataframes, 
    dev_names, 
    group_title,
    time_col='time_local', 
    cleaned_col='Cleaned Displacement'
):
    """
    Plots all sensors' cleaned displacement in a single figure for a subgroup.
    
    Parameters
    ----------
    dataframes : list of pd.DataFrame
        List of dataframes, each with a cleaned displacement column.
    dev_names : list of str
        Names/IDs for each sensor, for the legend.
    group_title : str
        Title for the plot.
    time_col : str
        Name of the datetime column in the DataFrames.
    cleaned_col : str
        Name of the cleaned displacement column in the DataFrames.
    """
    plt.figure(figsize=(12, 6))
    for df, name in zip(dataframes, dev_names):
        # Ensure the column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df_sorted = df.sort_values(time_col)
        plt.plot(df_sorted[time_col], df_sorted[cleaned_col], label=name, alpha=0.8)
    
    plt.title(f"Oscillating Trunk Diameter Fluctuations for {group_title}")
    plt.xlabel("Time")
    plt.ylabel("Displacement (µm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_avg_displacement_by_treatment(
    dataframes,
    dev_names,
    group_title,
    time_col='time_local',
    disp_col='Cleaned Displacement',
    irrigation_col='Irrigation',
    resample_freq='D',    # 'D'=daily, 'H'=hourly, None=no resampling
    error_type='std',     # 'std' or 'sem'
    show_fill=True,       # If True, fill_between for error band
    show_error_bars=True, # If True, show discrete error bars
    zero_each_sensor=True, # If True, subtract each sensor's first valid displacement
    debug=True,            # If True, print debug statements
    fill_single_point_err_with_zero=True # If True, set error=0 if count==1
):
    """
    Plots the average displacement (with error bands) for each irrigation treatment,
    in one figure, for a given trial. Optionally "zeros" each sensor's displacement
    so that the first valid point is 0.

    This version includes debug checks:
      1) Print device->irrigation assignments.
      2) Print group sizes after combining/resampling.
      3) Print head of aggregator DF (mean, count, std, err).
      4) Optionally set error=0 if count==1 (otherwise 'NaN').
    """

    if debug:
        print(f"\n=== Plotting for {group_title} ===")
        print(f"[DEBUG] We have {len(dataframes)} sensors in this trial.")

    # 1) Combine all DataFrames
    combined = []
    for df, name in zip(dataframes, dev_names):
        if debug:
            # Print the irrigation assignment(s) found in this DataFrame
            # (Should be 1 unique assignment if each DF belongs to only one irrigation)
            if irrigation_col in df.columns:
                irr_vals = df[irrigation_col].dropna().unique().tolist()
                print(f"[DEBUG] {name} => irrigation(s): {irr_vals}")
            else:
                print(f"[DEBUG] {name} => NO '{irrigation_col}' column. Skipping.")
                continue

        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        # Skip if critical columns are missing
        needed_cols = [time_col, disp_col, irrigation_col]
        if not all(col in df.columns for col in needed_cols):
            print(f"[WARN] Missing required columns in {name}. Skipping.")
            continue

        temp = df[[time_col, disp_col, irrigation_col]].copy()

        # Sort by time (important for zeroing)
        temp.sort_values(time_col, inplace=True)

        # Zero each sensor at its first valid displacement if desired
        if zero_each_sensor:
            temp_nonan = temp.dropna(subset=[disp_col])
            if not temp_nonan.empty:
                first_val = temp_nonan[disp_col].iloc[0]
                temp[disp_col] = temp[disp_col] - first_val

        combined.append(temp)

    if not combined:
        print(f"[{group_title}] No valid data. Nothing to plot.")
        return

    # 2) Concatenate
    combined_df = pd.concat(combined, ignore_index=True)

    # 3) Optional resampling
    if resample_freq is not None:
        # Set the time column as index if it's not already
        if combined_df.index.name != time_col:
            combined_df.set_index(time_col, inplace=True)

        # Group by irrigation, then resample each subset
        irrigation_groups = combined_df.groupby(irrigation_col)
        resampled_list = []
        for irr_name, subdf in irrigation_groups:
            sub_resampled = subdf.resample(resample_freq).agg({disp_col: 'mean'})
            sub_resampled[irrigation_col] = irr_name
            resampled_list.append(sub_resampled)
        combined_df = pd.concat(resampled_list).reset_index()
    else:
        # If no resampling, ensure time_col is a column
        if time_col not in combined_df.columns:
            combined_df.reset_index(inplace=True)
            combined_df.rename(columns={'index': time_col}, inplace=True)

    # 4) Check group sizes
    grouped_size = combined_df.groupby([irrigation_col, time_col]).size()
    if debug:
        print("\n[DEBUG] Group sizes (rows per (irrigation, time)):\n", grouped_size.head(20))

    # 5) Group by [Irrigation, time_col] to compute mean + error
    grouped = combined_df.groupby([irrigation_col, time_col])[disp_col]
    if error_type.lower() == 'std':
        agg_df = grouped.agg(mean='mean', count='count', err='std').reset_index()
        y_label = f"{disp_col} (±1 SD)"
    else:
        # standard error of the mean
        agg_df = grouped.agg(mean='mean', count='count', std='std').reset_index()
        agg_df['err'] = agg_df['std'] / np.sqrt(agg_df['count'])
        agg_df.drop(columns=['std'], inplace=True)
        y_label = f"{disp_col} (±1 SEM)"

    # (Optional) fill single data point bins with err=0 instead of NaN
    if fill_single_point_err_with_zero:
        single_mask = (agg_df['count'] == 1) & (agg_df['err'].isna())
        agg_df.loc[single_mask, 'err'] = 0.0

    if debug:
        print("\n[DEBUG] Aggregated DF (first 20 rows):")
        print(agg_df.head(20))

    # 6) Plot
    fig, ax = plt.subplots(figsize=(10,6))
    unique_irrigations = agg_df[irrigation_col].unique()

    for irr in sorted(unique_irrigations):
        if irr.lower() == 'unknown':
            # skip unknown if you want
            continue

        # Extract subset for this irrigation group
        subset = agg_df[agg_df[irrigation_col] == irr].sort_values(time_col)
        if subset.empty:
            # No data => skip
            continue

        x = subset[time_col]
        y = subset['mean']
        err = subset['err']

        # Plot the mean line
        ax.plot(x, y, label=f"{irr} irrigation")

        if show_fill and not err.dropna().empty:
            ax.fill_between(x, y - err, y + err, alpha=0.2)

        if show_error_bars and not err.dropna().empty:
            ax.errorbar(x, y, yerr=err, fmt='none', ecolor='gray', alpha=0.8, capsize=3)

    ax.set_title(f"Average {disp_col} by Irrigation\n{group_title}")
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def plot_avg_displacement_30min(
    dataframes,
    dev_names,
    group_title,
    irrigation_map=None,
    time_col='time_local',
    disp_col='Cleaned Displacement',
    zero_each_sensor=True,
    error_type='std',       # 'std' or 'sem'
    show_fill=True,
    show_error_bars=True
):
    """
    1) For each sensor's DataFrame:
       - Convert time to datetime.
       - Optionally zero the data so the first valid displacement is 0.
       - Set index=time_col, resample at 30T => create uniform half-hourly data.
       - Tag each row with irrigation=irrigation_map[ dev_name ] (or 'unknown').
       - Store in a list.

    2) Concatenate all sensors into 'combined_df'.

    3) Group by (Irrigation, time) => compute mean, std or sem => produce aggregator DF.

    4) Plot lines for each irrigation treatment, plus a
       semi-transparent "shadow" for ± err, and optional error bars.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        One DataFrame per sensor.
    dev_names : list of str
        Names of the sensors corresponding to the DataFrames, same length as 'dataframes'.
    group_title : str
        Title for the plot (e.g. "Blueberry Trial 1").
    irrigation_map : dict or None
        Dict mapping device_name -> irrigation_label (e.g. "BB1"->"100%", "BB7"->"30%").
        If None, we look for 'Irrigation' in each DF. If provided, we override.
    time_col : str
        Name of time column (default 'time_local').
    disp_col : str
        Name of displacement column (default 'Cleaned Displacement').
    zero_each_sensor : bool
        If True, subtract the sensor's first valid displacement from all data so it starts at 0.
    error_type : str
        'std' => standard deviation, 'sem' => standard error of the mean.
    show_fill : bool
        If True, fill_between for error band (the "shadow").
    show_error_bars : bool
        If True, draw an error bar on each half-hour point.

    Returns
    -------
    None
    """

    # 1) For each sensor, resample to 30-min grid
    sensor_list = []
    for df, dev_name in zip(dataframes, dev_names):
        df = df.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Optionally zero each sensor
        if zero_each_sensor and not df[disp_col].dropna().empty:
            first_val = df[disp_col].dropna().iloc[0]
            df[disp_col] = df[disp_col] - first_val
        
        # Assign irrigation, if needed
        if irrigation_map is not None:
            irrig_label = irrigation_map.get(dev_name, "unknown")
            df["Irrigation"] = irrig_label
        if "Irrigation" not in df.columns:
            df["Irrigation"] = "unknown"

        # Set index = time_col
        df.set_index(time_col, inplace=True)
        
        # Resample to 30-minute intervals
        df_30min = df.resample("30T").mean(numeric_only=True)
        
        # Re-inject irrigation label/device
        # (Assumes the entire sensor is the same irrigation)
        if not df["Irrigation"].dropna().empty:
            df_30min["Irrigation"] = df["Irrigation"].dropna().unique()[0]
        else:
            df_30min["Irrigation"] = "unknown"

        df_30min["Device"] = dev_name
        
        sensor_list.append(df_30min)

    if not sensor_list:
        print(f"No valid data for {group_title}.")
        return

    # 2) Combine
    combined_df = pd.concat(sensor_list)
    combined_df.reset_index(inplace=True)  # time_col is now a normal column again

    # 3) Group by (Irrigation, time_col) => aggregator
    grouped = combined_df.groupby(["Irrigation", time_col])[disp_col]
    agg_df = grouped.agg(mean='mean', count='count', std='std').reset_index()

    if error_type.lower() == 'std':
        agg_df['err'] = agg_df['std']  # standard deviation
        y_label = f"{disp_col} (±1 SD)"
    else:
        # standard error of the mean
        agg_df['err'] = agg_df['std'] / np.sqrt(agg_df['count'])
        y_label = f"{disp_col} (±1 SEM)"

    # (Optional) If some bins have count=1 => std=NaN => set err=0
    single_mask = (agg_df['count'] == 1) & (agg_df['err'].isna())
    agg_df.loc[single_mask, 'err'] = 0.0

    # 4) Plot
    fig, ax = plt.subplots(figsize=(10,6))

    # Custom colors for each irrigation level
    irrigation_colors = {
        "100%": "#1f77b4",  # matplotlib blue
        "30%":  "#2ca02c",  # matplotlib green
        "50%":  "#9467bd",  # purple
        "unknown": "#aaaaaa"
    }

    unique_irrig = agg_df["Irrigation"].unique()
    for irr in sorted(unique_irrig):
        if irr.lower() == 'unknown':
            # skip unknown if you want
            continue

        subset = agg_df[agg_df["Irrigation"] == irr].sort_values(time_col)
        if subset.empty:
            continue

        x = subset[time_col]
        y = subset["mean"]
        err = subset["err"].fillna(0)  # in case some remain NaN

        # Pick a color for this irrigation
        color = irrigation_colors.get(irr, 'gray')

        # --- Plot the "shadow" (± err) behind the line ---
        if show_fill and not err.dropna().empty:
            ax.fill_between(
                x,
                y - err,
                y + err,
                color=color,
                alpha=0.3,    # lighter shade for the shadow
                zorder=1       # behind the line
            )

        # --- Plot the main line ---
        ax.plot(
            x,
            y,
            label=f"{irr} irrigation",
            color=color,
            lw=2,
            zorder=2  # on top of the fill
        )

        # --- Optionally show discrete error bars ---
        if show_error_bars and not all(err == 0):
            ax.errorbar(
                x,
                y,
                yerr=err,
                fmt='none',
                ecolor=color,
                alpha=0.6,
                capsize=3,
                zorder=3  # above the line
            )

    ax.set_title(f"Avg {disp_col} (30-min) by Irrigation\n{group_title}")
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def plot_avg_displacement_30min_with_count(
    dataframes,
    dev_names,
    group_title,
    irrigation_map=None,
    time_col='time_local',
    disp_col='Cleaned Displacement',
    zero_each_sensor=True,
    error_type='std',       # 'std' or 'sem'
    show_fill=True,
    show_error_bars=True
):
    """
    1) For each sensor's DataFrame:
       - Convert time to datetime.
       - Optionally zero the data so the first valid displacement is 0.
       - Set index=time_col, resample at 30T => create uniform half-hourly data.
       - Tag each row with irrigation=irrigation_map[ dev_name ] (or 'unknown').
       - Store in a list.

    2) Concatenate all sensors into 'combined_df'.

    3) Group by (Irrigation, time) => compute mean, std or sem => produce aggregator DF.

    4) Plot in two subplots:
       - Top: mean lines per irrigation + optional shaded error
       - Bottom: "count" lines showing how many sensors contributed at each time.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        One DataFrame per sensor.
    dev_names : list of str
        Names of the sensors corresponding to the DataFrames, same length as 'dataframes'.
    group_title : str
        Title for the plot (e.g. "Blueberry Trial 1").
    irrigation_map : dict or None
        Dict mapping device_name -> irrigation_label (e.g. "BB1"->"100%", "BB7"->"30%").
        If None, we look for 'Irrigation' in the DF itself. If provided, we override.
    time_col : str
        Name of time column (default 'time_local').
    disp_col : str
        Name of displacement column (default 'Cleaned Displacement').
    zero_each_sensor : bool
        If True, subtract the sensor's first valid displacement from all data so it starts at 0.
    error_type : str
        'std' => standard deviation, 'sem' => standard error of the mean.
    show_fill : bool
        If True, fill_between for error band (the "shadow").
    show_error_bars : bool
        If True, draw an error bar on each half-hour point.

    Returns
    -------
    None
    """

    # 1) For each sensor, resample to 30-min grid
    sensor_list = []
    for df, dev_name in zip(dataframes, dev_names):
        df = df.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df.get(time_col, None)):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Optionally zero each sensor
        if zero_each_sensor and disp_col in df.columns and not df[disp_col].dropna().empty:
            first_val = df[disp_col].dropna().iloc[0]
            df[disp_col] = df[disp_col] - first_val
        
        # Assign irrigation, if needed
        if irrigation_map is not None:
            irrig_label = irrigation_map.get(dev_name, "unknown")
            df["Irrigation"] = irrig_label
        
        if "Irrigation" not in df.columns:
            df["Irrigation"] = "unknown"

        # Set index = time_col
        if time_col in df.columns:
            df.set_index(time_col, inplace=True)

        # Resample to 30-minute intervals
        df_30min = df.resample("30T").mean(numeric_only=True)

        # Re-inject irrigation label/device
        # (Assumes the entire sensor is the same irrigation)
        if not df["Irrigation"].dropna().empty:
            df_30min["Irrigation"] = df["Irrigation"].dropna().unique()[0]
        else:
            df_30min["Irrigation"] = "unknown"

        df_30min["Device"] = dev_name
        
        sensor_list.append(df_30min)

    if not sensor_list:
        print(f"[{group_title}] No valid data after resampling.")
        return

    # 2) Combine
    combined_df = pd.concat(sensor_list)
    combined_df.reset_index(inplace=True)  # time_col is now a normal column again

    # 3) Group by (Irrigation, time) => aggregator
    grouped = combined_df.groupby(["Irrigation", time_col])[disp_col]
    agg_df = grouped.agg(mean='mean', count='count', std='std').reset_index()

    if error_type.lower() == 'std':
        agg_df['err'] = agg_df['std']  # standard deviation
        y_label = f"{disp_col} (±1 SD)"
    else:
        # standard error of the mean
        agg_df['err'] = agg_df['std'] / np.sqrt(agg_df['count'])
        y_label = f"{disp_col} (±1 SEM)"

    # If some bins have count=1 => std=NaN => set err=0
    single_mask = (agg_df['count'] == 1) & (agg_df['err'].isna())
    agg_df.loc[single_mask, 'err'] = 0.0

    # 4) Plot with two subplots:
    #    - ax1 => displacement
    #    - ax2 => sensor count
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,8))
    plt.subplots_adjust(hspace=0.05)  # reduce vertical space between plots

    # Custom colors for each irrigation level
    irrigation_colors = {
        "100%": "#1f77b4",  # matplotlib blue
        "30%":  "#2ca02c",  # matplotlib green
        "50%":  "#9467bd",  # purple
        "unknown": "#aaaaaa"
    }

    # We get unique irrigation labels
    unique_irrig = agg_df["Irrigation"].unique()

    # Plot each irrigation group
    for irr in sorted(unique_irrig):
        if irr.lower() == 'unknown':
            # skip unknown if you want
            continue

        subset = agg_df[agg_df["Irrigation"] == irr].sort_values(time_col)
        if subset.empty:
            continue

        x = subset[time_col]
        y = subset["mean"]
        err = subset["err"].fillna(0)  # in case some remain NaN
        ccount = subset["count"]

        # Pick a color for this irrigation
        color = irrigation_colors.get(irr, 'gray')

        # === Top subplot: displacement + error
        # Fill between ± err
        if show_fill and not err.dropna().empty:
            ax1.fill_between(
                x,
                y - err,
                y + err,
                color=color,
                alpha=0.15,    # lighter shade for the shadow
                zorder=1       # behind the line
            )

        # Mean line
        ax1.plot(
            x,
            y,
            label=f"{irr} irrigation",
            color=color,
            lw=2,
            zorder=2  # on top of the fill
        )

        # Error bars
        if show_error_bars and not all(err == 0):
            ax1.errorbar(
                x,
                y,
                yerr=err,
                fmt='none',
                ecolor=color,
                alpha=0.6,
                capsize=3,
                zorder=3
            )

        # === Bottom subplot: sensor count
        # We'll just plot the count with the same color
        ax2.plot(
            x,
            ccount,
            label=f"{irr} count",
            color=color,
            lw=1.5
        )

    # Beautify axes
    ax1.set_title(f"Average {disp_col} (30-min) by Irrigation\n{group_title}")
    ax1.set_ylabel(y_label)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_ylabel("Sensor Count")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Make sure everything fits
    plt.tight_layout()
    plt.show()
    

def plot_mean_daily_range(
    dataframes,
    dev_names,
    group_title,
    irrigation_map=None,
    time_col='time_local',
    disp_col='Cleaned Displacement',
    zero_each_sensor=True,
    error_type='std',   # 'std' or 'sem'
    show_fill=True
):
    """
    Plots the *mean daily range* (max - min) of displacement for each irrigation group,
    with an optional ± error band in one subplot.

    Steps:
      1) Per sensor:
         - Convert to datetime if needed.
         - Optionally zero at the first valid displacement (if zero_each_sensor=True).
         - Resample daily => compute min & max => daily_range = max - min.
         - Tag with irrigation (from irrigation_map or existing column).
      2) Concatenate all sensors into one DataFrame (daily ranges).
      3) Group by (Irrigation, Date) => compute mean + std or sem => aggregator.
      4) Plot a line for each irrigation group over time, optionally with a fill_between band.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        One DataFrame per sensor.
    dev_names : list of str
        Sensor names corresponding to `dataframes`.
    group_title : str
        Plot title (e.g., "Blueberry Trial 1 (2023)").
    irrigation_map : dict or None
        If provided, a dict mapping device_name -> irrigation_label (e.g. "BB1"->"100%").
        If None, we look for "Irrigation" in the DataFrame or default to "unknown".
    time_col : str
        Column name for timestamps (default 'time_local').
    disp_col : str
        Column with displacement data (default 'Cleaned Displacement').
    zero_each_sensor : bool
        If True, subtract each sensor's first valid displacement => starts at 0.
    error_type : str
        'std' => standard deviation, 'sem' => standard error of the mean (std / sqrt(count)).
    show_fill : bool
        If True, plot a fill_between region for ± error around the mean daily range.

    Returns
    -------
    None
        Displays a single subplot with lines (and optionally a shaded error band).
    """

    # 1) Per-sensor daily range
    daily_list = []
    for df, dev_name in zip(dataframes, dev_names):
        df = df.copy()

        # Make sure we have a datetime
        if time_col not in df.columns:
            continue  # skip if the time column is missing
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        # Optionally zero each sensor
        if zero_each_sensor and disp_col in df.columns and not df[disp_col].dropna().empty:
            first_val = df[disp_col].dropna().iloc[0]
            df[disp_col] = df[disp_col] - first_val

        # Assign irrigation
        if irrigation_map is not None:
            irr_label = irrigation_map.get(dev_name, "unknown")
            df["Irrigation"] = irr_label
        if "Irrigation" not in df.columns:
            df["Irrigation"] = "unknown"

        # Resample daily => min & max => daily_range
        df.set_index(df[time_col], inplace=True)
        daily_agg = df.resample("D").agg({
            disp_col: ['min', 'max']
        })
        daily_agg.columns = ['disp_min', 'disp_max']  # rename
        daily_agg['daily_range'] = daily_agg['disp_max'] - daily_agg['disp_min']

        daily_agg.reset_index(inplace=True)  # index is date
        daily_agg.rename(columns={time_col: 'Date'}, inplace=True)

        # Keep irrigation & sensor info
        if not df["Irrigation"].dropna().empty:
            daily_agg["Irrigation"] = df["Irrigation"].dropna().unique()[0]
        else:
            daily_agg["Irrigation"] = "unknown"

        daily_agg["Device"] = dev_name

        daily_list.append(daily_agg[["Date", "daily_range", "Irrigation", "Device"]])

    if not daily_list:
        print(f"[{group_title}] No valid data for daily range.")
        return

    # 2) Combine daily data from all sensors
    combined_df = pd.concat(daily_list, ignore_index=True)

    # 3) Group by (Irrigation, Date) => aggregator
    grouped = combined_df.groupby(["Irrigation", "Date"])["daily_range"]
    agg_df = grouped.agg(mean='mean', count='count', std='std').reset_index()

    # Compute error
    if error_type.lower() == 'std':
        agg_df['err'] = agg_df['std']
        y_label = "Maximum Daily Shrinkage (±1 SD)"
    else:
        agg_df['err'] = agg_df['std'] / np.sqrt(agg_df['count'])
        y_label = "Maximum Daily Shrinkage (±1 SEM)"

    # If only one sensor => no std => set err=0
    single_mask = (agg_df['count'] == 1) & (agg_df['err'].isna())
    agg_df.loc[single_mask, 'err'] = 0.0

    # 4) Single subplot
    fig, ax = plt.subplots(figsize=(10,6))

    # color map
    irrigation_colors = {
        "100%": "#1f77b4",  # blue
        "30%":  "#2ca02c",  # green
        "50%":  "#9467bd",  # purple
        "unknown": "#aaaaaa"
    }

    unique_irrig = agg_df["Irrigation"].unique()
    for irr in sorted(unique_irrig):
        if irr.lower() == "unknown":
            continue
        sub = agg_df[agg_df["Irrigation"] == irr].sort_values("Date")
        if sub.empty:
            continue

        x = sub["Date"]
        y = sub["mean"]
        err = sub["err"].fillna(0)

        color = irrigation_colors.get(irr, "gray")

        # Fill
        if show_fill and not err.dropna().empty:
            ax.fill_between(
                x, 
                y - err, 
                y + err, 
                color=color, 
                alpha=0.15,
                zorder=1
            )

        # Mean line
        ax.plot(
            x, y,
            color=color,
            lw=2,
            label=f"{irr} irrigation",
            zorder=2
        )

    ax.set_title(f"Average Maximum Daily Shrinkage\n{group_title}")
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    


def plot_all_trials_avg_displacement_30min(
    trials_dataframes,
    trials_dev_names,
    trial_labels,
    overall_title="All Trials Average Displacement (30-min)",
    time_col='time_local',
    disp_col='Cleaned Displacement',
    zero_each_sensor=True,
    error_type='sem',       # 'std' or 'sem'
    show_fill=True
):
    """
    Combines multiple trials' data (each trial has a list of DataFrames, a list of dev_names)
    into a single plot showing average displacement (resampled 30-min).

    Steps:
      1) For each trial, label sensors with "Trial" = trial_labels[i].
      2) Resample each sensor to 30-min bins => mean displacement.
      3) Combine all into one big DataFrame with columns [time, disp, Trial].
      4) Group by (Trial, time) => aggregator => plot lines for each trial.

    Parameters
    ----------
    trials_dataframes : list of list of DataFrames
        E.g. [b1_dfs, b2_dfs, b3_dfs, h1_dfs, h2_dfs].
    trials_dev_names : list of list of str
        Parallel structure to trials_dataframes. E.g. [b1_names, b2_names, b3_names, ...].
    trial_labels : list of str
        Name for each trial, e.g. ["BB1_2023", "BB2_2023", "BB3_2024", "HZ1_2023", "HZ2_2024"].
    overall_title : str
        Plot title.
    time_col : str
        Datetime column in each DataFrame.
    disp_col : str
        Displacement column name.
    zero_each_sensor : bool
        If True, zero each sensor at its first valid displacement.
    error_type : str
        'std' => standard deviation, 'sem' => standard error of the mean across sensors in that bin.
    show_fill : bool
        If True, fill the ± error band behind each line.

    Returns
    -------
    None
    """

    # 1) Resample each sensor from each trial
    master_list = []
    for trial_idx, (df_list, dev_list) in enumerate(zip(trials_dataframes, trials_dev_names)):
        trial_label = trial_labels[trial_idx]
        
        for df, dev_name in zip(df_list, dev_list):
            df = df.copy()
            
            # Ensure datetime
            if not pd.api.types.is_datetime64_any_dtype(df.get(time_col, None)):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Zero
            if zero_each_sensor and disp_col in df.columns and not df[disp_col].dropna().empty:
                first_val = df[disp_col].dropna().iloc[0]
                df[disp_col] -= first_val
            
            # Set index for resampling
            if time_col in df.columns:
                df.set_index(time_col, inplace=True)
            
            # 30-min resample => mean displacement
            df_30min = df.resample("30T").mean(numeric_only=True)
            
            # Create columns for "Trial" and "Device"
            df_30min["Trial"] = trial_label
            df_30min["Device"] = dev_name
            
            master_list.append(df_30min[[disp_col, "Trial", "Device"]])

    if not master_list:
        print("No valid data to plot.")
        return

    # 2) Combine everything
    combined_df = pd.concat(master_list)
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={time_col: "Time"}, inplace=True)

    # 3) Group by (Trial, Time)
    grouped = combined_df.groupby(["Trial", "Time"])[disp_col]
    agg_df = grouped.agg(mean='mean', count='count', std='std').reset_index()

    # Compute error
    if error_type.lower() == 'std':
        agg_df['err'] = agg_df['std']
        y_label = f"{disp_col} (±1 SD)"
    else:
        agg_df['err'] = agg_df['std'] / np.sqrt(agg_df['count'])
        y_label = f"{disp_col} (±1 SEM)"

    # If single sensor => std might be NaN => set to 0
    single_mask = (agg_df['count'] == 1) & (agg_df['err'].isna())
    agg_df.loc[single_mask, 'err'] = 0.0

    # 4) Plot
    fig, ax = plt.subplots(figsize=(12,6))
    
    # Let's define some colors for each trial label if you like
    color_map = {
        "Blueberry1_2023": "blue",
        "Blueberry2_2023": "green",
        "Blueberry3_2024": "purple",
        "Hazelnut1_2023":  "orange",
        "Hazelnut2_2024":  "red"
        # add more if needed
    }

    unique_trials = agg_df["Trial"].unique()
    for trial in sorted(unique_trials):
        sub = agg_df[agg_df["Trial"] == trial].sort_values("Time")
        if sub.empty:
            continue
        
        x = sub["Time"]
        y = sub["mean"]
        err = sub["err"]

        # pick color
        color = color_map.get(trial, None)

        # fill
        if show_fill and not err.dropna().empty:
            ax.fill_between(
                x, y - err, y + err,
                color=color,
                alpha=0.15,
                zorder=1
            )
        
        # line
        ax.plot(
            x, y,
            color=color,
            label=trial,
            lw=2,
            zorder=2
        )

    ax.set_title(overall_title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.legend()
    plt.tight_layout()
    plt.show()
    


def plot_30min_all_trials_by_year_irrig(
    trials_dataframes,
    trials_dev_names,
    trial_labels,
    irrigation_maps,
    time_col='time_local',
    disp_col='Cleaned Displacement',
    zero_each_sensor=True,
    error_type='std',  # 'std' or 'sem'
    show_fill=True
):
    """
    Combine multiple trials into a single aggregator, resample each sensor
    to 30 min, assign an 'Irrigation' from the provided map, and label
    the data by trial + year. Then we produce *two* figures:
      - one for 2023
      - one for 2024
    Each figure shows lines split by (Trial, Irrigation).

    Parameters
    ----------
    trials_dataframes : list of list of DataFrames
        E.g. [b1_dfs, b2_dfs, b3_dfs, h1_dfs, h2_dfs].
    trials_dev_names : list of list of str
        Parallel to trials_dataframes: [b1_names, b2_names, b3_names, ...].
    trial_labels : list of str
        Something like ["Blueberry1_2023", "Blueberry2_2023",
                        "Blueberry3_2024", "Hazelnut1_2023", "Hazelnut2_2024"].
    irrigation_maps : list of dict
        Each dict maps dev_name -> irrigation label. Must match the order of trial_labels.
        E.g. [bb1_map, bb2_map, bb3_map, h1_map, h2_map].
    time_col : str
        Column with datetime info, default 'time_local'.
    disp_col : str
        Displacement column, default 'Cleaned Displacement'.
    zero_each_sensor : bool
        If True, subtract first valid displacement from each sensor's series.
    error_type : str
        'std' => standard deviation across sensors, 'sem' => standard error of mean.
    show_fill : bool
        If True, fill the ± error band behind each line.

    Returns
    -------
    None
        Displays two figures: one for 2023, one for 2024.
    """

    master_list = []

    # 1) Resample each trial's data, label 'Trial', 'Irrigation', 'Year'
    for trial_idx, (df_list, dev_list) in enumerate(zip(trials_dataframes, trials_dev_names)):
        trial_label = trial_labels[trial_idx]
        # Determine the "year" from the label or from the data, if you prefer
        # We'll parse from the trial_label if it ends with e.g. "_2023" or "_2024".
        # Alternatively, you could parse from the actual time col. For simplicity:
        if trial_label.endswith("2023"):
            year_val = 2023
        elif trial_label.endswith("2024"):
            year_val = 2024
        else:
            year_val = None  # fallback if the label doesn't contain a year

        irr_map = irrigation_maps[trial_idx]

        for df, dev_name in zip(df_list, dev_list):
            df = df.copy()

            if not pd.api.types.is_datetime64_any_dtype(df.get(time_col, None)):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Zero
            if zero_each_sensor and disp_col in df.columns and not df[disp_col].dropna().empty:
                first_val = df[disp_col].dropna().iloc[0]
                df[disp_col] = df[disp_col] - first_val

            # Set index for resampling
            if time_col in df.columns:
                df.set_index(time_col, inplace=True)

            # Resample 30 min => mean
            df_30m = df.resample("30T").mean(numeric_only=True)

            # Add columns
            df_30m["Trial"] = trial_label
            df_30m["Year"] = year_val
            # Set irrigation from the map
            irrigation_val = irr_map.get(dev_name, "unknown")
            df_30m["Irrigation"] = irrigation_val
            df_30m["Device"] = dev_name

            master_list.append(df_30m[[disp_col,"Trial","Year","Irrigation","Device"]])

    if not master_list:
        print("No data to plot.")
        return

    # 2) Combine
    combined = pd.concat(master_list)
    combined.reset_index(inplace=True)
    combined.rename(columns={time_col: "Time"}, inplace=True)

    # 3) Group by (Year, Trial, Irrigation, Time)
    grouped = combined.groupby(["Year","Trial","Irrigation","Time"])[disp_col]
    agg_df = grouped.agg(mean='mean', count='count', std='std').reset_index()

    # error
    if error_type.lower() == 'std':
        agg_df['err'] = agg_df['std']
        err_label = f"{disp_col} (±1 SD)"
    else:
        agg_df['err'] = agg_df['std'] / np.sqrt(agg_df['count'])
        err_label = f"{disp_col} (±1 SEM)"

    # fill single-sensor bins
    single_mask = (agg_df['count'] == 1) & (agg_df['err'].isna())
    agg_df.loc[single_mask, 'err'] = 0.0

    # 4) For convenience, we'll produce 2 plots: one for 2023, one for 2024.
    # We'll define a small helper:
    def plot_for_year(year_to_plot):
        subdf = agg_df[agg_df["Year"] == year_to_plot]
        if subdf.empty:
            print(f"No data for year={year_to_plot}. Skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_title(f"{year_to_plot} - 30-min Avg Displacement (split by Trial & Irrigation)")

        # We'll differentiate (Trial, Irrigation) combos with color or line style
        # For simplicity, let's just do one color per combo. 
        # You might want something more sophisticated in real usage.
        combos = subdf.groupby(["Trial","Irrigation"]).size().index.tolist()
        # We'll make a color map of combos
        from matplotlib.cm import get_cmap
        cmap = get_cmap("tab20")  # up to 20 distinct colors
        combo_colors = {}
        for i, combo in enumerate(combos):
            combo_colors[combo] = cmap(i % 20)

        # Now plot each (Trial, Irrigation)
        for combo in combos:
            trial_i, irrig_i = combo
            subcombo = subdf[(subdf["Trial"] == trial_i) & (subdf["Irrigation"] == irrig_i)]
            x = subcombo["Time"].sort_values()
            # ensure we align y,err
            subcombo = subcombo.sort_values("Time")
            y = subcombo["mean"]
            err = subcombo["err"].fillna(0)

            color = combo_colors[combo]
            label_str = f"{trial_i} - {irrig_i}"

            # fill
            if show_fill and not err.dropna().empty:
                ax.fill_between(x, y - err, y + err, color=color, alpha=0.2, zorder=1)

            # line
            ax.plot(x, y, color=color, lw=2, zorder=2, label=label_str)

        ax.set_xlabel("Time")
        ax.set_ylabel(err_label)
        ax.legend()
        plt.tight_layout()
        plt.show()

    plot_for_year(2023)
    plot_for_year(2024)
    
def plot_daily_range_all_trials_by_year_irrig(
    trials_dataframes,
    trials_dev_names,
    trial_labels,
    irrigation_maps,
    time_col='time_local',
    disp_col='Cleaned Displacement',
    zero_each_sensor=True,
    error_type='std',
    show_fill=True
):
    """
    Similar to the 30-min function, but we compute daily range (max-min).
    Then group by (Year, Trial, Irrigation, Date) => mean daily range across sensors,
    produce two figures: one for 2023, one for 2024, lines split by (Trial, Irrigation).
    """

    daily_list = []
    for trial_idx, (df_list, dev_list) in enumerate(zip(trials_dataframes, trials_dev_names)):
        trial_label = trial_labels[trial_idx]
        if trial_label.endswith("2023"):
            year_val = 2023
        elif trial_label.endswith("2024"):
            year_val = 2024
        else:
            year_val = None

        irr_map = irrigation_maps[trial_idx]

        for df, dev_name in zip(df_list, dev_list):
            df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df.get(time_col, None)):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            if zero_each_sensor and disp_col in df.columns and not df[disp_col].dropna().empty:
                first_val = df[disp_col].dropna().iloc[0]
                df[disp_col] -= first_val

            if time_col in df.columns:
                df.set_index(df[time_col], inplace=True)

            # daily min/max => range
            daily_agg = df.resample("D").agg({disp_col: ['min','max']})
            daily_agg.columns = ["disp_min","disp_max"]
            daily_agg['daily_range'] = daily_agg['disp_max'] - daily_agg['disp_min']
            daily_agg.reset_index(inplace=True)
            daily_agg.rename(columns={time_col: "Date"}, inplace=True)

            daily_agg["Trial"] = trial_label
            daily_agg["Year"] = year_val
            daily_agg["Irrigation"] = irr_map.get(dev_name, "unknown")
            daily_agg["Device"] = dev_name

            daily_list.append(daily_agg[["Date","daily_range","Trial","Year","Irrigation","Device"]])

    if not daily_list:
        print("No data for daily range.")
        return

    combined_df = pd.concat(daily_list, ignore_index=True)
    grouped = combined_df.groupby(["Year","Trial","Irrigation","Date"])["daily_range"]
    agg_df = grouped.agg(mean='mean', count='count', std='std').reset_index()

    if error_type.lower() == 'std':
        agg_df['err'] = agg_df['std']
        y_label = "Maximum Daily Shrinkage (±1 SD)"
    else:
        agg_df['err'] = agg_df['std'] / np.sqrt(agg_df['count'])
        y_label = "Maximum Daily Shrinkage (±1 SEM)"

    single_mask = (agg_df['count'] == 1) & (agg_df['err'].isna())
    agg_df.loc[single_mask, 'err'] = 0.0

    def plot_for_year(year_to_plot):
        subdf = agg_df[agg_df["Year"] == year_to_plot]
        if subdf.empty:
            print(f"No daily range data for year={year_to_plot}.")
            return

        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_title(f"{year_to_plot} - Average Maximum Daily Shrinkage (split by Trial & Irrigation)")

        combos = subdf.groupby(["Trial","Irrigation"]).size().index.tolist()
        from matplotlib.cm import get_cmap
        cmap = get_cmap("tab20")
        combo_colors = {}
        for i, combo in enumerate(combos):
            combo_colors[combo] = cmap(i % 20)

        for combo in combos:
            trial_i, irrig_i = combo
            subcombo = subdf[(subdf["Trial"] == trial_i) & (subdf["Irrigation"] == irrig_i)]
            subcombo = subcombo.sort_values("Date")

            x = subcombo["Date"]
            y = subcombo["mean"]
            err = subcombo["err"].fillna(0)

            color = combo_colors[combo]
            label_str = f"{trial_i} - {irrig_i}"
            if show_fill and not err.dropna().empty:
                ax.fill_between(x, y-err, y+err, color=color, alpha=0.2, zorder=1)
            ax.plot(x, y, color=color, lw=2, zorder=2, label=label_str)

        ax.set_xlabel("Date")
        ax.set_ylabel(y_label)
        ax.legend()
        plt.tight_layout()
        plt.show()

    plot_for_year(2023)
    plot_for_year(2024)
    


def scatter_daily_range_vpd_all(all_daily_df):
    """
    Single scatter with regression (all 57 sensors combined).
    X-axis: daily_range
    Y-axis: daily_max_vpd
    One regression line for the entire dataset.
    """
    fig, ax = plt.subplots(figsize=(8,6))
    sns.regplot(
        data=all_daily_df,
        x='daily_range',
        y='daily_max_vpd',
        scatter_kws={'alpha':0.3},  # transparency in scatter
        line_kws={'color':'red'},
        ax=ax
    )
    ax.set_title("All Sensors: Daily Range vs. Daily Max VPD")
    ax.set_xlabel("Maximum Daily Shrinkage (µm)")
    ax.set_ylabel("Daily Max VPD")
    plt.tight_layout()
    plt.show()
    
def scatter_daily_range_vpd_by_crop(all_daily_df):
    """
    Single figure, but separate regression lines for blueberry vs. hazelnut.
    """
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        data=all_daily_df,
        x='daily_range',
        y='daily_max_vpd',
        hue='Crop',
        alpha=0.3,
        ax=ax
    )

    # We'll do two regression lines manually:
    # 1) Filter for Blueberry
    blueberry_df = all_daily_df[all_daily_df['Crop'] == 'Blueberry']
    sns.regplot(
        data=blueberry_df,
        x='daily_range',
        y='daily_max_vpd',
        scatter=False,  # skip re-plotting points
        line_kws={'color':'blue'},
        ax=ax
    )

    # 2) Filter for Hazelnut
    hazelnut_df = all_daily_df[all_daily_df['Crop'] == 'Hazelnut']
    sns.regplot(
        data=hazelnut_df,
        x='daily_range',
        y='daily_max_vpd',
        scatter=False,
        line_kws={'color':'green'},
        ax=ax
    )

    ax.set_title("Maximum Daily Shrinkage vs. Daily Max VPD by Crop")
    ax.set_xlabel("Maximum Daily Shrinkage")
    ax.set_ylabel("Daily Max VPD")
    plt.tight_layout()
    plt.show()

def scatter_daily_range_vpd_4panels(all_daily_df):
    """
    Create 4 panels:
      - row=Crop (Blueberry, Hazelnut)
      - col=Year (2023, 2024)

    Each panel has a scatter + regression line of daily_range vs. daily_max_vpd.
    """
    import seaborn as sns
    
    # Filter out only Blueberry/Hazelnut if needed
    subdf = all_daily_df[all_daily_df['Crop'].isin(['Blueberry','Hazelnut'])]
    # Also only years 2023, 2024
    subdf = subdf[subdf['Year'].isin([2023, 2024])]

    # We'll create a combined factor to make it simpler:
    # Or we can do row='Crop', col='Year' in an lmplot
    g = sns.lmplot(
        data=subdf,
        x='daily_range',
        y='daily_max_vpd',
        col='Year',
        row='Crop',
        hue=None,         # single color if we want. Or set hue='Crop' if needed.
        scatter_kws={'alpha':0.3},
        line_kws={'color':'red'},
        sharex=True,
        sharey=True,
        aspect=1.2,
        height=4
    )
    g.set_axis_labels("Maximum Daily Shrinkage", "Daily Max VPD")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Maximum Daily Shrinkage vs. Daily Max VPD (Split by Crop & Year)")
    plt.show()    