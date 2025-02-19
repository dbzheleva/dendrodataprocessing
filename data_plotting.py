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


def plot_original_vs_cleaned(df):
    """
    Overlays original displacement vs. cleaned displacement on one plot.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(df['time_local'], df['um'], label='Original', color='blue')
    plt.plot(df['time_local'], df['Cleaned Displacement'], label='Cleaned', color='red', linestyle='--')

    plt.xlabel('Time')
    plt.ylabel('Displacement (um)')
    plt.title('Original vs. Cleaned Displacement')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
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


def plot_displacement_and_vpd(df, device_name):
    """
    Plots cleaned displacement on the primary y-axis and VPD on the secondary y-axis.
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df['time_local'], df['Cleaned Displacement'], color='blue', label='Displacement')
    ax1.set_ylabel('Cleaned Displacement (um)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f'{device_name}: Cleaned Displacement and VPD')

    ax2 = ax1.twinx()
    ax2.plot(df['time_local'], df['VPD'], color='orange', linestyle='--', label='VPD')
    ax2.set_ylabel('VPD (kPa)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.xticks(rotation=45)
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.show()


def plot_multiple_sensors_displacement(sensors_data, sensor_names):
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
        plt.plot(df['time_local'], df['Cleaned Displacement'], label=f'{name} Displacement')

    plt.xlabel('Time')
    plt.ylabel('Cleaned Displacement (um)')
    plt.title('Cleaned Displacement for Multiple Sensors')
    plt.legend()
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
    plt.plot(filtered_df['time_local'], filtered_df['Cleaned Displacement'], label='Displacement', color='blue')

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
    plt.title(f'Daily Range vs Max Temp\n({device_name})')
    plt.xlabel('Max Temperature (°C)')
    plt.ylabel('Daily Range (µm)')
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
    plt.title(f'Daily Range vs Max VPD\n({device_name})')
    plt.xlabel('Max VPD (kPa)')
    plt.ylabel('Daily Range (µm)')
    plt.tight_layout()
    plt.show()