"""
data_loading.py

Contains functions to load dendrometer CSV data and do basic preprocessing steps.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def get_device_name_from_filename(file_path):
    """
    Given a path like '/some/folder/BB1_complete.csv',
    return 'BB1' by splitting on the underscore.
    """
    base = os.path.splitext(os.path.basename(file_path))[0]  # e.g. 'BB1_complete'
    parts = base.split('_', 1)                               # => ['BB1','complete']
    return parts[0] 

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
        device_name = get_device_name_from_filename(file_name)
        return df, device_name
    except KeyError as e:
        print(f"KeyError: {e}. Could not find the column in the DataFrame.")
        return None, None


'''def load_and_prepare_data_time_limit(file_name, start_date=None, end_date=None):
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
        df = pd.read_csv(file_name, skiprows=1, header=1)
        color_dict = {'Green': 0, 'Yellow': 1, 'Red': 2, 'Error': 3}
        df['alignment_mapped'] = df['Alignment'].map(color_dict)
        df['time_local'] = pd.to_datetime(df['time_local'])

        # Filter data if start_date and end_date provided
        if start_date:
            df = df[df['time_local'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['time_local'] <= pd.to_datetime(end_date)]

        device_name = get_device_name_from_filename(file_name)
        return df, device_name
    except KeyError as e:
        print(f"KeyError: {e}. Could not find the column in the DataFrame.")
        return None, None'''

def load_and_prepare_data_time_limit(file_name, start_date=None, end_date=None):
    """
    Loads dendrometer data from a CSV, filters by date range, and performs initial cleanup.
    Tries multiple skiprows/header configs to find 'Alignment'. Then extracts 'DeviceID'
    from the CSV's 'name'/'instance' columns, e.g. 'BB3'.
    
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
        The filtered DataFrame, with alignment mapped, time parsed, and 'DeviceID' assigned.
    device_id : str
        A parsed sensor/device ID like "BB3", "LB2", etc.
    """
    color_dict = {'Green': 0, 'Yellow': 1, 'Red': 2, 'Error': 3}

    possible_configs = [
        dict(skiprows=0, header=0, dtype={"instance": "Int64"}),
        dict(skiprows=0, header=1, dtype={"instance": "Int64"}),
        dict(skiprows=1, header=0, dtype={"instance": "Int64"}),
        dict(skiprows=1, header=1, dtype={"instance": "Int64"}),
        dict(skiprows=2, header=0, dtype={"instance": "Int64"}),
        dict(skiprows=2, header=1, dtype={"instance": "Int64"}),
    ]

    for config in possible_configs:
        try:
            print(f"[DEBUG] Trying config: skiprows={config.get('skiprows')}, header={config.get('header')}")
            df = pd.read_csv(file_name, **config)

            # optional: strip leading/trailing spaces in columns
            df.columns = df.columns.str.strip()

            # Check 'Alignment'
            if 'Alignment' not in df.columns:
                raise KeyError("Alignment")

            # Map alignment color
            df['alignment_mapped'] = df['Alignment'].map(color_dict)

            # Convert time_local
            df['time_local'] = pd.to_datetime(df['time_local'])

            # Date filtering
            if start_date:
                df = df[df['time_local'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['time_local'] <= pd.to_datetime(end_date)]

            # ---------------------------
            # PARSE THE DEVICE ID FROM CSV
            # ---------------------------
            if 'name' in df.columns and 'instance' in df.columns:
                # We'll define a helper function:
                def parse_device_id(row):
                    raw_name = str(row["name"])             # e.g. "BlueberryDendrometer_"
                    instance_val = row["instance"]          # Int64 or possibly <NA>
                
                    if pd.isna(instance_val):
                        # If instance is <NA>, fallback to 0, "X", or anything you want
                        instance_str = "X"
                    else:
                        # instance_val is a valid integer => just convert it to string directly
                        # (Don't do int(instance_val) again, because it's already an integer)
                        instance_str = str(instance_val)
                
                    # Basic logic for prefix:
                    if "BlueberryDendrometer_" in raw_name:
                        prefix = "BB"
                    elif "LBDendrometer_" in raw_name:
                        prefix = "LB"
                    elif "SDendrometer_" in raw_name:
                        prefix = "S"
                    elif "HazelnutDendrometer_" in raw_name:
                        prefix = "H"
                    else:
                        prefix = "Unknown"
                
                    return prefix + instance_str

                # Apply to each row (or just the first row).
                df["DeviceID"] = df.apply(parse_device_id, axis=1)

                # We'll pick the first row's ID as the "device_name"
                device_id = df["DeviceID"].iloc[0]

            else:
                # If columns are missing, fallback to something else
                # (or raise an error)
                device_id = "UnknownDevice"

            return df, device_id

        except KeyError as e:
            print(f"KeyError: {e}. Column not found under config {config} => trying next config.")
        except pd.errors.ParserError as pe:
            print(f"[WARN] ParserError with config {config}: {pe}")
        except Exception as ex:
            print(f"[WARN] Another error with config {config}: {ex}")

    print("[ERROR] None of the configurations found 'Alignment' or parsed successfully.")
    return None, None