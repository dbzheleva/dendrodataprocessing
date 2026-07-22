"""
daily_analysis.py

Daily aggregations, environment stats, and higher-level daily range visuals.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress



def resample_30T_keep_structure(df, time_col='time_local'):
    """
    Resample the DataFrame to 30-minute intervals, aggregating numeric columns by mean
    and non-numeric columns by first. Returns a new DataFrame with 'time_local' reset as a column.
    
    Assumes 'time_local' is already set as the DateTimeIndex.
    """

    # 1) Build an aggregator dict: numeric => 'mean', object => 'first'
    aggregator = {}
    for col in df.columns:
        # We'll skip if col is the same as the index name, but typically after .set_index(), 
        # 'time_local' isn't in df.columns anymore. If it is, handle carefully:
        if col == time_col:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            aggregator[col] = 'mean'
        else:
            aggregator[col] = 'first'

    # 2) Resample every 30 minutes with the aggregator
    df_30min = df.resample('30T').agg(aggregator)

    # 3) Reset index so 'time_local' becomes a column again
    df_30min = df_30min.reset_index()

    return df_30min

def min_max_normalize(series):
    """Min-max normalization to [0..1]. Returns the normalized series."""
    smin = series.min()
    smax = series.max()
    denom = smax - smin
    if denom == 0:
        # all values identical or NaN
        return series * 0.0  # or series.copy() with 0
    return (series - smin) / denom

def aggregate_environmental_data(df, displacement_col='Cleaned Displacement'):
    """
    Groups by day to calculate daily max/min/mean of relevant columns:
     - `displacement_col` (daily range)
     - Temperature (max)
     - VPD (max)
     - Volumetric_Water_Content (mean)

    Returns a DataFrame with columns:
     - time_local
     - <displacement_col>_max
     - <displacement_col>_min
     - daily_temp_max
     - daily_vpd_max
     - daily_soil_moisture_avg
     - daily_range
    """
    # The dictionary below uses displacement_col as the key for max/min
    daily_env_df = df.set_index('time_local').resample('D').agg({
        displacement_col: ['max', 'min'],
        'Temperature': 'max',
        'VPD': 'max',
        'Volumetric_Water_Content': 'mean'
    })
        # 1) Flatten first
    daily_env_df.columns = [
        '_'.join(col).strip() 
        if isinstance(col, tuple) else col
        for col in daily_env_df.columns
    ]
    
    # 2) Then compute daily_range on the flattened columns
    daily_env_df['daily_range'] = (
        daily_env_df[f'{displacement_col}_max'] 
        - daily_env_df[f'{displacement_col}_min']
    )
  
   

    # Rename for clarity
    daily_env_df.rename(columns={
        'Temperature_max': 'daily_temp_max',
        'VPD_max': 'daily_vpd_max',
        'Volumetric_Water_Content_mean': 'daily_soil_moisture_avg',
    }, inplace=True)

    return daily_env_df.reset_index()


def plot_group_daily_range_and_env(group_dfs, group_names, displacement_col='Cleaned Displacement'):
    """
    For each dataframe (sensor) in the group:
      1) Aggregate daily metrics using `aggregate_environmental_data()`,
         passing displacement_col (default='Cleaned Displacement').
      2) Plot the daily range, daily_temp_max, daily_vpd_max 
         on 3 subplots, one figure, all sensors together.

    Parameters
    ----------
    group_dfs : list of pd.DataFrame
        DataFrames for each sensor in the group.
    group_names : list of str
        Sensor names corresponding to each DataFrame.
    displacement_col : str
        The column name to treat as the displacement measurement
        when computing daily max/min/range.
    """

    import matplotlib.pyplot as plt

    # 1) Aggregate daily data for each sensor
    daily_data_list = []
    for df, name in zip(group_dfs, group_names):
        # Calculate daily metrics (use specified column)
        daily_env_df = aggregate_environmental_data(df, displacement_col=displacement_col)
        daily_data_list.append((name, daily_env_df))

    # 2) Create a single figure with 3 subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

    # --- Subplot 1: Daily Range ---
    for sensor_name, daily_df in daily_data_list:
        axes[0].plot(daily_df['time_local'], daily_df['daily_range'], label=sensor_name)
    axes[0].set_title(f'Daily Range ({displacement_col})')
    axes[0].set_ylabel('Range')
    axes[0].legend()
    axes[0].grid(True)

    # --- Subplot 2: Daily Max Temperature ---
    for sensor_name, daily_df in daily_data_list:
        axes[1].plot(daily_df['time_local'], daily_df['daily_temp_max'], label=sensor_name)
    axes[1].set_title('Daily Max Temperature')
    axes[1].set_ylabel('Temperature (°C)') 
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

def plot_daily_range_vs_temp_vpd(df_agg, device_name):
    """
    Simple scatter plot examples of daily range vs. daily temp or VPD, using aggregated data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Daily Range vs Temperature
    sns.scatterplot(x='daily_temp_max', y='daily_range', data=df_agg, ax=axes[0])
    axes[0].set_title(f'Daily Range vs Max Temp\n({device_name})')

    # Daily Range vs VPD
    sns.scatterplot(x='daily_vpd_max', y='daily_range', data=df_agg, ax=axes[1])
    axes[1].set_title(f'Daily Range vs Max VPD\n({device_name})')

    plt.tight_layout()
    plt.show()