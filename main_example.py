"""
main_example.py

Demonstrates how to load data, outlier-correct it, resample to 30min,
and plot the results, *without* using data_analysis or advanced_analysis code.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# From your narrower modules
from data_loading import load_and_prepare_data_time_limit
from data_processing import (identify_and_adjust_outliers, 
                            resample_30T_keep_structure)
from data_plotting import (
    plot_cleaned_displacement,
    plot_original_vs_cleaned,
    plot_displacement_and_vpd,
    plot_multiple_sensors_displacement
)



def main():
    file1 = 'path to your file'
    file2 = 'path to your file'
    file3 = 'path to your file'

    # 1) Load data with optional date filtering
    df1, dev1_name = load_and_prepare_data_time_limit(file1, start_date='2023-08-01', end_date='2023-08-15')
    df2, dev2_name = load_and_prepare_data_time_limit(file2, start_date='2023-08-01', end_date='2023-08-15')
    df3, dev3_name = load_and_prepare_data_time_limit(file3, start_date='2023-08-01', end_date='2023-08-15')

    # 2) Identify outliers & adjust
    df1 = identify_and_adjust_outliers(df1)
    df2 = identify_and_adjust_outliers(df2)
    df3 = identify_and_adjust_outliers(df3)

    # 3) Convert each DF's 'time_local' to datetime index, then resample
    for df in [df1, df2, df3]:
        df['time_local'] = pd.to_datetime(df['time_local'], errors='coerce')
        df.set_index('time_local', inplace=True)

    df1_30min = resample_30T_keep_structure(df1)
    df2_30min = resample_30T_keep_structure(df2)
    df3_30min = resample_30T_keep_structure(df3)

    # Drop rows that are all-NaN after resampling
    df1_30min.dropna(how='all', inplace=True)
    df2_30min.dropna(how='all', inplace=True)
    df3_30min.dropna(how='all', inplace=True)

    # 4) Basic plots
    #    (Assumes each df now has 'um' and 'Cleaned Displacement' columns)
    df1_30min= df1_30min.reset_index()
    df2_30min= df2_30min.reset_index()
    df3_30min= df3_30min.reset_index()
    # Single-sensor examples
    plot_cleaned_displacement(df1_30min, dev1_name)
    plot_original_vs_cleaned(df1_30min)
    plot_displacement_and_vpd(df1_30min, dev1_name)

    # Multiple sensor comparison
    plot_multiple_sensors_displacement(
        [df1_30min, df2_30min, df3_30min],
        [dev1_name, dev2_name, dev3_name]
    )

    print("Done with main pipeline: load -> outlier -> resample -> plot")

if __name__ == "__main__":
    main()