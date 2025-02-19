"""
data_loading.py

Contains functions to load dendrometer CSV data and do basic preprocessing steps.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_prepare_data(file_name):
    """
    Loads dendrometer data from a CSV and performs initial cleanup.

    Parameters
    ----------
    file_name : str
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
        The loaded DataFrame with mapped alignment, time parsed, and device name columns.
    device_name : str
        Combined string of 'name_instance' for identifying sensor/device.
    """
    try:
        df = pd.read_csv(file_name, header=3)

        # Map alignment to numeric values
        color_dict = {'Green': 0, 'Yellow': 1, 'Red': 2, 'Error': 3}
        df['alignment_mapped'] = df['Alignment'].map(color_dict)

        # Convert time_local to datetime
        df['time_local'] = pd.to_datetime(df['time_local'])

        # Build a device name
        device_name = f"{df['name'].iloc[0]}_{df['instance'].iloc[0]}"
        return df, device_name
    except KeyError as e:
        print(f"KeyError: {e}. Could not find the column in the DataFrame.")
        return None, None


def load_and_prepare_data_time_limit(file_name, start_date=None, end_date=None):
    """
    Loads dendrometer data from a CSV, filters by date range, and performs initial cleanup.

    Parameters
    ----------
    file_name : str
        Path to the CSV file.
    start_date : str or None
        Start date as 'YYYY-MM-DD' to filter. If None, no lower bound.
    end_date : str or None
        End date as 'YYYY-MM-DD' to filter. If None, no upper bound.

    Returns
    -------
    df : pandas.DataFrame
        The filtered DataFrame with alignment mapped and time parsed.
    device_name : str
        Combined string of 'name_instance' for identifying sensor/device.
    """
    try:
        df = pd.read_csv(file_name, header=3)
        color_dict = {'Green': 0, 'Yellow': 1, 'Red': 2, 'Error': 3}
        df['alignment_mapped'] = df['Alignment'].map(color_dict)
        df['time_local'] = pd.to_datetime(df['time_local'])

        # Filter data if start_date and end_date provided
        if start_date:
            df = df[df['time_local'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['time_local'] <= pd.to_datetime(end_date)]

        device_name = f"{df['name'].iloc[0]}_{df['instance'].iloc[0]}"
        return df, device_name
    except KeyError as e:
        print(f"KeyError: {e}. Could not find the column in the DataFrame.")
        return None, None