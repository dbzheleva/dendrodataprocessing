import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import matplotlib.dates as mdates
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
# If you use advanced stats
import statsmodels.api as sm


###############################################################################
# 1. Data Loading and Preparation
###############################################################################

def load_and_prepare_data(file_name):
    """
    Load a CSV file and prepare the data for analysis by:
      1. Reading the CSV file with a header offset of 3.
      2. Mapping the 'Alignment' text to numeric values in a column 'alignment_mapped'.
      3. Converting 'time_local' column to datetime and setting it as the DataFrame index.
      4. Creating a device_name from 'name' and 'instance'.

    :param file_name: Path to the CSV file.
    :return: (DataFrame, device_name) if successful, otherwise (None, None).
    """
    try:
        # Read CSV, skipping the first 3 header lines
        df = pd.read_csv(file_name, header=3)

        # Map color strings to numeric codes
        color_dict = {'Green': 0, 'Yellow': 1, 'Red': 2, 'Error': 3}
        df['alignment_mapped'] = df['Alignment'].map(color_dict)

        # Convert time column to datetime and set as index
        df['time_local'] = pd.to_datetime(df['time_local'])
        df.set_index('time_local', inplace=True)

        # Construct device name from the first row
        device_name = f"{df['name'].iloc[0]}_{df['instance'].iloc[0]}"
        return df, device_name

    except KeyError as e:
        print(f"KeyError: {e}. Could not find the column in the DataFrame.")
        return None, None


###############################################################################
# 2. Outlier Detection Functions
###############################################################################

def detect_outliers_with_moving_window(data, window_size, threshold):
    """
    Detect outliers using a rolling median + IQR approach.
    Outliers lie beyond (median ± threshold * IQR).

    :param data: Pandas Series to analyze.
    :param window_size: Size of the rolling window.
    :param threshold: IQR multiplier defining outlier cutoff.
    :return: Boolean Series indicating outliers.
    """
    rolling_median = data.rolling(window=window_size, min_periods=1, center=True).median()

    rolling_iqr = (data.rolling(window=window_size, min_periods=1, center=True).quantile(0.75)
                   - data.rolling(window=window_size, min_periods=1, center=True).quantile(0.25))

    lower_bound = rolling_median - threshold * rolling_iqr
    upper_bound = rolling_median + threshold * rolling_iqr
    
    outliers = (data < lower_bound) | (data > upper_bound)
    outliers.index = data.index  # Align indices
    return outliers


def detect_outliers_with_moving_window_new(data, window_size, threshold):
    """
    Detect outliers using a rolling Q1 and Q3 (quartiles) and an IQR-based approach.
    Outliers lie beyond (Q1 - threshold * IQR) or (Q3 + threshold * IQR).

    :param data: Pandas Series to analyze.
    :param window_size: Size of the rolling window.
    :param threshold: IQR multiplier defining outlier cutoff.
    :return: Boolean Series indicating outliers.
    """
    Q1 = data.rolling(window=window_size, min_periods=1, center=True).quantile(0.25)
    Q3 = data.rolling(window=window_size, min_periods=1, center=True).quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (data < lower_bound) | (data > upper_bound)
    outliers.index = data.index  # Align indices
    return outliers


def remove_outliers(data, window_size=20, threshold=1.5):
    """
    Remove (replace with NaN) outliers in 'data' based on the 
    detect_outliers_with_moving_window_new() function.

    :param data: Pandas Series.
    :param window_size: Rolling window size for outlier detection.
    :param threshold: IQR multiplier.
    :return: A copy of the data Series with outliers replaced by NaN.
    """
    outliers = detect_outliers_with_moving_window_new(data, window_size, threshold)
    cleaned_data = data.copy()
    cleaned_data[outliers] = np.nan
    return cleaned_data


###############################################################################
# 3. Example: Testing the Outlier Detection on a Simple Series
###############################################################################
if __name__ == "__main__":
    # Simple test data
    data = pd.Series([10, 15, 12, 14, 100, 16, 18])

    # Detect outliers
    outliers = detect_outliers_with_moving_window_new(data, window_size=5, threshold=1.5)

    # Print verification for the data point at index 4
    print(f"Data Point at index 4: {data.iloc[4]}")
    median_4 = data.rolling(5, min_periods=1, center=True).median().iloc[4]
    q75_4 = data.rolling(5, min_periods=1, center=True).quantile(0.75).iloc[4]
    q25_4 = data.rolling(5, min_periods=1, center=True).quantile(0.25).iloc[4]
    iqr_4 = q75_4 - q25_4
    lower_bound_4 = median_4 - 1.5 * iqr_4
    upper_bound_4 = median_4 + 1.5 * iqr_4

    print(f"Median of the window: {median_4}")
    print(f"IQR of the window: {iqr_4}")
    print(f"Lower Bound: {lower_bound_4}")
    print(f"Upper Bound: {upper_bound_4}")
    print(f"Is outlier? {outliers.iloc[4]}")

###############################################################################
# 4. Loading Data from File and Plotting Outliers
###############################################################################

file_name = '/Users/dragomirazheleva/Desktop/PhD/PhD Projects/Dendrometer/All Dendrometer Data/H1-6 complete-selected/H1_complete.csv'  # Replace with your file path
df, device_name = load_and_prepare_data(file_name)

if df is not None:
    # Choose columns and parameters
    window_size = 48
    threshold = 1.5

    # Detect outliers in the 'um' column
    outliers = detect_outliers_with_moving_window_new(df['um'], window_size, threshold)

    # Plot original data with outliers
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['um'], label='Original Data', color='blue')
    plt.scatter(df.index, df['um'].where(outliers), color='red', label='Outliers')
    plt.title(f"Outlier Detection for {device_name}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print outliers
    outlier_count = outliers.sum()
    if outlier_count > 0:
        print(f"Outliers detected: {outlier_count}")
        print(df[outliers])
    else:
        print("No outliers detected.")
else:
    print("DataFrame not loaded properly.")

###############################################################################
# 5. Histograms and Box Plots of First Differences
###############################################################################

# Example of computing differences and plotting
if df is not None:
    # Remove outliers using our function
    cleaned_data = remove_outliers(df['um'])

    # Calculate first differences
    diffs = cleaned_data.diff().dropna()

    # Example: Histogram with logarithmic Y-axis
    plt.figure(figsize=(10, 6))
    plt.hist(diffs, bins=50, edgecolor='black', alpha=0.7, log=True)
    plt.title('Histogram of First Differences (Logarithmic Y-axis)')
    plt.xlabel('Difference')
    plt.ylabel('Log(Frequency)')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Example: Box plot of the differences
    plt.figure(figsize=(10, 6))
    plt.boxplot(diffs)
    plt.title('Box Plot of First Differences')
    plt.ylabel('Difference')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Repeat histogram excluding zero differences
    non_zero_diffs = diffs[diffs != 0]
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_diffs, bins=50, edgecolor='black', alpha=0.7, log=True)
    plt.title('Histogram of Non-Zero First Differences (Logarithmic Y-axis)')
    plt.xlabel('Difference')
    plt.ylabel('Log(Frequency)')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

###############################################################################
# 6. Local Window Print-Out Around the First Outlier
###############################################################################

if df is not None and outliers.any():
    # Find the index of the first outlier
    first_outlier_index = outliers.idxmax()  # Returns the first True index
    
    if first_outlier_index:
        # Get numeric position of the outlier
        window_data_indices = df.index.get_loc(first_outlier_index)
        
        # Define window (10 points before, outlier, 10 after)
        window_start_index = max(0, window_data_indices - 10)
        window_end_index = min(len(df) - 1, window_data_indices + 9)
        
        # Extract the window of data
        window_data = df['um'].iloc[window_start_index:window_end_index + 1]
        
        print("The 21 values (10 before, the outlier, and 10 after):")
        print(window_data)
        print("\nStatistics:")
        print(f"Median of the window: {window_data.median()}")
        print(f"IQR of the window: {window_data.quantile(0.75) - window_data.quantile(0.25)}")
    else:
        print("No outliers detected.")
        
###############################################################################
# 7. Compare Original and Cleaned Data
###############################################################################

if df is not None:
    cleaned_data = remove_outliers(df['um'])  # Re-assign in case not defined yet

    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['um'], label='Original Data', color='lightgreen')
    plt.plot(cleaned_data.index, cleaned_data, label='Cleaned Data', color='blue', linewidth=2)
    plt.title('Comparison of Original and Cleaned Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

###############################################################################
# 8. Adjusting Dips with Rolling Mean - this function doesn't work 
###############################################################################

def adjust_dips_rolling_mean(data, window_size=5, threshold_multiplier=1.5):
    """
    Adjust sudden dips or spikes by moving them closer to the rolling mean 
    if they deviate significantly.
    """

    # Ensure we have a Series; copy to avoid modifying the original
    if not isinstance(data, pd.Series):
        s = pd.Series(data)
    else:
        s = data.copy()
    
    # Make sure index is sorted if it's time-based
    s = s.sort_index()
    
    # Compute the rolling mean
    rolling_mean = s.rolling(window=window_size, center=True, min_periods=1).mean()
    diff = s - rolling_mean

    # Threshold based on standard deviation of differences
    threshold = threshold_multiplier * diff.std()

    adjusted_data = s.copy()
    
    # Indices of "large dips/spikes" beyond the threshold
    dip_indices = diff[diff.abs() > threshold].index

    # Replace those values with the rolling mean
    for idx in dip_indices:
        adjusted_data.loc[idx] = rolling_mean.loc[idx]

    # Return a Series (with the original time index)
    return adjusted_data

# Example usage if cleaned_data is available
if df is not None:
    # Suppose cleaned_data is your time-indexed Series
    adjusted_data = adjust_dips_rolling_mean(cleaned_data, window_size=5)
    
    plt.figure(figsize=(12, 6))
    plt.plot(cleaned_data, label='Cleaned Data', color='blue')
    plt.plot(adjusted_data, label='Adjusted Data', color='green', linestyle='--')
    plt.title("Comparison of Cleaned and Adjusted Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

###############################################################################
# 9. Detecting Trends: Expansion and Contraction
###############################################################################

def detect_trends(data, consecutive_threshold=3):
    """
    Detect sequences of consecutive increases or decreases in the data.

    :param data: Pandas Series.
    :param consecutive_threshold: Minimum run length (e.g. 3 consecutive ups/downs) 
                                  to classify as an expansion or contraction.
    :return: (expansion_indices, contraction_indices) lists of indices.
    """
    # Differences between consecutive data points
    differences = data.diff().dropna()

    # Classify each difference as increase, decrease, or steady
    trends = differences.apply(lambda x: "increase" if x > 0 
                               else ("decrease" if x < 0 else "steady"))

    sequences = []
    current_sequence = []
    current_trend = trends.iloc[0]

    # Build sequences of identical trend labels
    for i, trend in enumerate(trends):
        if trend == current_trend:
            current_sequence.append(i)
        else:
            sequences.append((current_trend, current_sequence))
            current_sequence = [i]
            current_trend = trend
    sequences.append((current_trend, current_sequence))  # last one

    # Filter sequences by length
    expansion_indices = []
    contraction_indices = []
    for trend, seq in sequences:
        if len(seq) >= consecutive_threshold:
            if trend == "increase":
                expansion_indices.extend(seq)
            elif trend == "decrease":
                contraction_indices.extend(seq)

    return expansion_indices, contraction_indices


# Example usage
if df is not None:
    expansion, contraction = detect_trends(cleaned_data)

###############################################################################
# 10. Daily Segments and Advanced Outlier Checks
###############################################################################

def get_day_data(data, start_time, end_time):
    """
    Return data between start_time and end_time.
    """
    return data[(data.index >= start_time) & (data.index < end_time)]

def get_daily_segments(data, start_time, end_time, segments):
    """
    Extract data for segments that lie fully within [start_time, end_time).
    """
    segment_data_list = []  # Collect day-segments here

    for segment in segments:
        segment_start, segment_end = segment
        if segment_start >= start_time and segment_end < end_time:
            segment_data_list.append(get_day_data(data, segment_start, segment_end))

    if segment_data_list:
        segment_data = pd.concat(segment_data_list)
    else:
        # Return an empty structure matching 'data'
        if isinstance(data, pd.DataFrame):
            segment_data = pd.DataFrame(columns=data.columns)
        else:
            # If 'data' is a Series, create an empty Series of the same dtype
            segment_data = pd.Series(dtype=data.dtype)

    return segment_data

def compute_difference(segment_data):
    """
    Compute the first difference of the data.
    """
    return segment_data.diff().dropna()

def identify_outliers(diff_data):
    """
    Identify outliers in a Series using a simple IQR method (1.5 * IQR).
    """
    Q1 = diff_data.quantile(0.25)
    Q3 = diff_data.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (diff_data < (Q1 - 1.5 * IQR)) | (diff_data > (Q3 + 1.5 * IQR))
    return diff_data[outlier_mask].index

def indices_to_timestamp_tuples(data, indices):
    """
    Convert a list of integer indices to a list of adjacent timestamp tuples.
    This helps to define segments in time.
    """
    timestamp_tuples = []
    for i in range(0, len(indices) - 1):
        start_time = data.index[indices[i]]
        end_time = data.index[indices[i + 1]]
        timestamp_tuples.append((start_time, end_time))
    return timestamp_tuples


# Convert expansions/contractions to timestamp tuples
if df is not None:
    expansion_tuples = indices_to_timestamp_tuples(cleaned_data, expansion)
    contraction_tuples = indices_to_timestamp_tuples(cleaned_data, contraction)

    # Example: Check outliers in expansions/contractions day by day
    start_date = cleaned_data.index.min().normalize() + pd.Timedelta(hours=10)
    end_date = cleaned_data.index.max()

    expansion_outliers = []
    contraction_outliers = []

    while start_date < end_date:
        day_end = start_date + pd.Timedelta(days=1)
        
        daily_expansion = get_daily_segments(cleaned_data, start_date, day_end, expansion_tuples)
        daily_contraction = get_daily_segments(cleaned_data, start_date, day_end, contraction_tuples)
        
        expansion_diff = compute_difference(daily_expansion)
        contraction_diff = compute_difference(daily_contraction)
        
        expansion_outliers.extend(identify_outliers(expansion_diff))
        contraction_outliers.extend(identify_outliers(contraction_diff))
        
        start_date = day_end

    #print("Expansion Outliers:", expansion_outliers)
    #print("Contraction Outliers:", contraction_outliers)

###############################################################################
# 11. Plotting Identified Trends (Expansion/Contraction)
###############################################################################

def plot_trends(data, expansion_indices, contraction_indices):
    """
    Plot data and highlight expansion (green) and contraction (red) points
    for a slice of the data (here, 1000:1500).
    """
    plt.figure(figsize=(15, 8))

    # Narrow the view to [1000:1500] for demonstration
    data_slice = data.iloc[1000:1500]

    # Adjust indices of expansions/contractions to match slice
    expansion_slice = [idx - 1000 for idx in expansion_indices if 1000 <= idx < 1500]
    contraction_slice = [idx - 1000 for idx in contraction_indices if 1000 <= idx < 1500]

    plt.plot(data_slice, color='blue', label='Data')

    legend_labels = []

    # Mark expansions
    for idx in expansion_slice:
        label = 'Expansion' if 'Expansion' not in legend_labels else ""
        plt.scatter(data_slice.index[idx], data_slice.iloc[idx], color='green', label=label)
        if label:
            legend_labels.append(label)

    # Mark contractions
    for idx in contraction_slice:
        label = 'Contraction' if 'Contraction' not in legend_labels else ""
        plt.scatter(data_slice.index[idx], data_slice.iloc[idx], color='red', label=label)
        if label:
            legend_labels.append(label)

    plt.title("Identified Trends in Data (Values 1000 to 1500)")
    plt.legend()
    plt.show()


# Example usage (will only work if your data length > 1500):
if df is not None:
    expansion, contraction = detect_trends(cleaned_data)
    plot_trends(cleaned_data, expansion, contraction)

###############################################################################
# 12. Detecting Outliers Within a Trend
###############################################################################

def detect_outliers_within_trend(data, indices, window_size, threshold):
    """
    Given a set of indices (e.g., expansion or contraction),
    detect outliers in their consecutive differences using a rolling IQR approach.
    """
    if isinstance(indices, int):
        indices = [indices]

    # Create a Series from the subset
    if len(indices) == 1:
        trend_data = data.iloc[indices[0]:indices[0]+1]
    else:
        trend_data = data.iloc[indices]

    # Take absolute differences
    differences = trend_data.diff().dropna().abs()
    
    # Filter out differences of 0
    differences = differences[differences != 0]
    
    # Rolling quartiles and IQR
    Q1 = differences.rolling(window=window_size, min_periods=1, center=True).quantile(0.25)
    Q3 = differences.rolling(window=window_size, min_periods=1, center=True).quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    outlier_diffs = (differences < lower_bound) | (differences > upper_bound)
    
    return outlier_diffs.index[outlier_diffs].tolist()

# Example usage:
#expansion_outliers, contraction_outliers = detect_outliers_within_trend(cleaned_data, expansion, window_size=20, threshold=1.5)

###############################################################################
# 13. Final: Detect Outlier Differences in Entire Series
###############################################################################

def detect_outlier_differences(data, window_size, threshold):
    """
    Detect outliers based on large consecutive differences. 
    Only non-zero differences are considered. 
    Values corresponding to these large differences are flagged in the original Series.

    :param data: Pandas Series
    :param window_size: Rolling window size for quartile calculations
    :param threshold: IQR multiplier
    :return: Boolean Series, True where the data point is flagged as an outlier.
    """
    # Calculate absolute differences between consecutive data points
    differences = data.diff().abs().dropna()

    # Filter out zero differences if not useful for detection
    differences = differences[differences != 0]

    # Rolling Q1 and Q3
    Q1 = differences.rolling(window=window_size, min_periods=1, center=True).quantile(0.25)
    Q3 = differences.rolling(window=window_size, min_periods=1, center=True).quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    outlier_diffs = (differences < lower_bound) | (differences > upper_bound)
    
    # Flag the *later* point in each difference as outlier
    outlier_flags = pd.Series(False, index=data.index)
    for idx in outlier_diffs.index[outlier_diffs]:
        if idx in data.index:
            outlier_flags[idx] = True

    return outlier_flags


###############################################################################
# 14. Plot Cleaned Data vs. Outliers
###############################################################################

def plot_cleaned_outliers(data, outliers):
    """
    Plot the original/cleaned data and highlight outliers with red dots.
    """
    # Create a copy and drop outliers
    cleaned_data = data.copy()
    cleaned_data[outliers] = np.nan

    plt.figure(figsize=(15, 8))

    # Plot outliers
    plt.scatter(data[outliers].index, data[outliers], color='red', label='Outliers', s=50, zorder=5)

    # Plot cleaned data
    plt.plot(cleaned_data, label='Cleaned Data', color='green')

    plt.title("Comparison of Cleaned Data and Outliers")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
outlier_flags = detect_outlier_differences(cleaned_data, window_size=96, threshold=20)
plot_cleaned_outliers(cleaned_data, outlier_flags)

# Calculate the difference between 'pos_raw' and 'pos_avg'
df['pos_difference'] = df['pos_raw'] - df['pos_avg']

# Set up the figure and axes
fig, axarr = plt.subplots(2, 1, figsize=(10, 10))

# Plotting the histogram for the difference with log scale on the y-axis
axarr[0].hist(df['pos_difference'], bins=50, edgecolor='k', alpha=0.7, log=True)
axarr[0].set_title('Distribution of Difference between pos_raw and pos_avg')
axarr[0].set_xlabel('Difference')
axarr[0].set_ylabel('Frequency (Log Scale)')
axarr[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Plotting the box plot for the difference
axarr[1].boxplot(df['pos_difference'].dropna())  # Drop NaN values as boxplot can't handle them
axarr[1].set_title('Box plot of Difference between pos_raw and pos_avg')
axarr[1].set_ylabel('Difference')
axarr[1].grid(True, which='both', linestyle='--', linewidth=0.5)
#axarr[1].set_ylim(-10, 10)  # Setting the y-axis limits for the box plot
axarr[1].set_yscale('symlog', linthresh=1)
# Adjusting the layout
plt.tight_layout()
plt.show()

#Functions from another script which builds upon the functions above


###############################################################################
# 1. Data Loading / Preparation
###############################################################################

def load_and_prepare_data(file_name, start_date=None, end_date=None):
    """
    Loads the CSV, maps alignment, converts to datetime.
    Optionally filters rows by [start_date, end_date].

    :param file_name: Path to the CSV file.
    :param start_date: (Optional) Start date (string or Timestamp).
    :param end_date: (Optional) End date (string or Timestamp).
    :return: (df, device_name)
    """
    try:
        df = pd.read_csv(file_name, header=3)
        color_dict = {'Green': 0, 'Yellow': 1, 'Red': 2, 'Error': 3}
        df['alignment_mapped'] = df['Alignment'].map(color_dict)
        df['time_local'] = pd.to_datetime(df['time_local'])

        # If date limits are provided, filter
        if start_date and end_date:
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)
            df = df[(df['time_local'] >= start_ts) & (df['time_local'] <= end_ts)]

        # Construct a device_name from 'name' + 'instance'
        device_name = f"{df['name'].iloc[0]}_{df['instance'].iloc[0]}"
        return df, device_name

    except KeyError as e:
        print(f"KeyError: {e}. Required column not found.")
        return None, None


###############################################################################
# 2. Outlier Detection / Cleaning
###############################################################################

def identify_and_adjust_outliers(df, diff_multiplier=10.0):
    """
    Identify large jumps in 'um' column and shift subsequent rows to remove them.
    The 'Cleaned Displacement' column is created from 'um' minus the accumulated
    adjustments.

    :param df: DataFrame with columns 'um' and 'time_local'.
    :param diff_multiplier: Factor for the IQR-based detection.
    :return: The modified DataFrame (with 'Cleaned Displacement' column).
    """
    df['Difference'] = df['um'].diff()
    Q1 = df['Difference'].quantile(0.25)
    Q3 = df['Difference'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_indices = df[(df['Difference'] < Q1 - diff_multiplier * IQR) |
                         (df['Difference'] > Q3 + diff_multiplier * IQR)].index

    df['Adjustment'] = 0.0
    for idx in outlier_indices:
        if idx != 0:
            adjustment = df.loc[idx, 'Difference']
            df.loc[idx:, 'Adjustment'] += adjustment

    df['Cleaned Displacement'] = df['um'] - df['Adjustment']
    return df


def calculate_daily_range_cleaned(df):
    """
    Calculates daily max/min of 'Cleaned Displacement' and returns a DataFrame
    with daily_range = (max - min).
    """
    daily_df = df.set_index('time_local').resample('D')['Cleaned Displacement'].agg(['max', 'min'])
    daily_df['daily_range'] = daily_df['max'] - daily_df['min']
    return daily_df.reset_index()

###############################################################################
# 3. Basic Plotting
###############################################################################

def plot_cleaned_displacement(df, device_name, show_original=True):
    """
    Plots original vs. cleaned displacement side-by-side or overlay.
    :param show_original: If True, plots both. If False, only cleaned.
    """
    if show_original:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Original
        ax[0].plot(df['time_local'], df['um'], color='blue')
        ax[0].set_title(f'{device_name}: Original Displacement')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Displacement')

        # Cleaned
        ax[1].plot(df['time_local'], df['Cleaned Displacement'], color='red')
        ax[1].set_title(f'{device_name}: Cleaned Displacement')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Displacement')

        for axis in ax:
            labels = axis.get_xticklabels()
            plt.setp(labels, rotation=45, ha='right')

        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(14, 6))
        plt.plot(df['time_local'], df['Cleaned Displacement'], color='red', label='Cleaned')
        plt.title(f'{device_name}: Cleaned Displacement Only')
        plt.xlabel('Time')
        plt.ylabel('Displacement')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_displacement(df, device_name):
    """
    Plots 'um' plus color-coded alignment zones.
    """
    min_date = df['time_local'].min().strftime('%Y-%m-%d')
    max_date = df['time_local'].max().strftime('%Y-%m-%d')
    instance_name = df['instance'].iloc[0] if 'instance' in df.columns else ''

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['time_local'], df['um'], label='Displacement')

    # alignment fill
    ax.fill_between(
        df['time_local'], df['um'].max(), df['um'].min(),
        where=(df['alignment_mapped'] == 0), color='green', alpha=0.3, label='Green'
    )
    ax.fill_between(
        df['time_local'], df['um'].max(), df['um'].min(),
        where=(df['alignment_mapped'] == 1), color='yellow', alpha=0.3, label='Yellow'
    )
    ax.fill_between(
        df['time_local'], df['um'].max(), df['um'].min(),
        where=(df['alignment_mapped'] == 2), color='red', alpha=0.3, label='Red'
    )
    ax.fill_between(
        df['time_local'], df['um'].max(), df['um'].min(),
        where=(df['alignment_mapped'] == 3), color='purple', alpha=0.3, label='Error'
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    ax.set_title(f'{device_name} [{instance_name}] {min_date} to {max_date}')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_displacement_and_vpd(df, device_name):
    """
    Plots 'Cleaned Displacement' vs. time on one axis and 'VPD' on the other axis.
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    ax1.plot(df['time_local'], df['Cleaned Displacement'], color='blue', label='Displacement')
    ax1.set_ylabel('Displacement (um)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f'{device_name}: Cleaned Displacement and VPD')

    ax2 = ax1.twinx()
    ax2.plot(df['time_local'], df['VPD'], color='orange', label='VPD', linestyle='--')
    ax2.set_ylabel('VPD (kPa)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_multiple_sensors(sensors_data, sensor_names):
    """
    Plots the 'Cleaned Displacement' from multiple sensors on one figure.
    """
    plt.figure(figsize=(15, 6))
    for df, name in zip(sensors_data, sensor_names):
        plt.plot(df['time_local'], df['Cleaned Displacement'], label=f'{name}')

    plt.xlabel('Time')
    plt.ylabel('Cleaned Displacement (um)')
    plt.title('Cleaned Displacement for Multiple Sensors')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_daily_range(df, column_name):
    daily_df = df.set_index('time_local').resample('D')[column_name].agg(['max', 'min'])
    daily_df['daily_range'] = daily_df['max'] - daily_df['min']
    return daily_df['daily_range']

def plot_multiple_sensors_with_temp(sensors_data, sensor_names):
    """
    Plots multiple sensors' Cleaned Displacement in one subplot
    and average max daily Temperature in another (or as a second axis).
    """
    # Example: If you want a dual-subplot approach
    fig, (ax_temp, ax_disp) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # (1) Plot average max daily Temperature
    # We'll define a helper function to get daily max from each DataFrame
    def daily_max_temp(df):
        return df.set_index('time_local').resample('D')['Temperature'].max()

    # Combine daily max temps and average across sensors
    daily_temps = [daily_max_temp(df) for df in sensors_data]
    daily_temps_df = pd.concat(daily_temps, axis=1)
    avg_temp = daily_temps_df.mean(axis=1)
    ax_temp.plot(avg_temp.index, avg_temp, color='red', linestyle='--', label='Avg Max Daily Temp')
    ax_temp.set_ylabel('Average Max Daily Temperature (°C)')
    ax_temp.legend(loc='upper right')
    ax_temp.grid(True)

    # (2) Plot Cleaned Displacement
    for df, name in zip(sensors_data, sensor_names):
        ax_disp.plot(df['time_local'], df['Cleaned Displacement'], label=name)
    ax_disp.set_ylabel('Cleaned Displacement (um)')
    ax_disp.set_ylim(-300, 700)
    ax_disp.legend(loc='upper left')
    ax_disp.grid(True)

    fig.suptitle('Displacement and Avg Max Daily Temperature')
    plt.tight_layout()
    plt.show()

###############################################################################
# 4. Additional Analysis
###############################################################################

def aggregate_environmental_data(df):
    """
    Groups data by day and calculates daily max/min for displacement, temperature, VPD, etc.
    Also calculates daily_range = max-min for Cleaned Displacement.
    """
    daily_env_df = df.set_index('time_local').resample('D').agg({
        'Cleaned Displacement': ['max', 'min'],
        'Temperature': 'max',
        'VPD': 'max',
        'Volumetric_Water_Content': 'mean'
    })
    daily_env_df['daily_range'] = daily_env_df[('Cleaned Displacement', 'max')] - daily_env_df[('Cleaned Displacement', 'min')]
    
    # Flatten columns
    daily_env_df.columns = ['_'.join(col).strip() for col in daily_env_df.columns.values]
    # Rename for clarity
    daily_env_df.rename(columns={
        'Temperature_max': 'daily_temp_max',
        'VPD_max': 'daily_vpd_max',
        'Volumetric_Water_Content_mean': 'daily_soil_moisture_avg'
    }, inplace=True)
    return daily_env_df.reset_index()


def plot_daily_range_vs_vpd(df, device_name):
    """
    Plots daily range of displacement on one axis and VPD on another.
    """
    df = identify_and_adjust_outliers(df)
    daily_df = calculate_daily_range_cleaned(df)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(daily_df['time_local'], daily_df['daily_range'], color='purple', label='Daily Range')
    ax1.set_ylabel('Daily Range (um)', color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.set_title(f'{device_name}: Daily Range vs. VPD')

    ax2 = ax1.twinx()
    ax2.plot(df['time_local'], df['VPD'], color='orange', label='VPD', linestyle='--')
    ax2.set_ylabel('VPD (kPa)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_daily_range_vs_max_vpd_with_regression(df, device_name):
    """
    Calculates daily range of displacement and daily max VPD, merges, then
    performs a linear regression on range vs. VPD and plots the result.
    """
    df = identify_and_adjust_outliers(df)
    daily_range_df = calculate_daily_range_cleaned(df)
    
    # daily max of VPD
    daily_max_VPD = df.set_index('time_local').resample('D')['VPD'].max().reset_index().rename(columns={'VPD':'daily_max_VPD'})
    
    merged_df = pd.merge(daily_range_df, daily_max_VPD, on='time_local', how='inner')
    
    slope, intercept, r_value, p_value, std_err = linregress(merged_df['daily_range'], merged_df['daily_max_VPD'])

    plt.figure(figsize=(8,5))
    plt.scatter(merged_df['daily_range'], merged_df['daily_max_VPD'], color='blue')
    plt.plot(merged_df['daily_range'], intercept + slope*merged_df['daily_range'], color='red',
             label=f'R²={r_value**2:.2f}')
    plt.title(f'{device_name} - Range vs. Max VPD')
    plt.xlabel('Daily Range (um)')
    plt.ylabel('Daily Max VPD (kPa)')
    plt.legend()
    plt.grid(True)
    plt.show()

###############################################################################
# 5. Time Series Modeling / Stats
###############################################################################

def check_stationarity(series):
    """
    Simple wrapper for the Augmented Dickey-Fuller test, prints results.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, val in result[4].items():
        print(f'Crit. Value {key}: {val:.3f}')

def model_arima(series, order=(1,0,0)):
    """
    Fits an ARIMA model with given (p,d,q). 
    Returns the fitted model.
    """
    model = ARIMA(series.dropna(), order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def plot_moving_average(series, window, plot_intervals=False, scale=1.96):
    """
    Plot a rolling mean with optional confidence intervals.
    """
    rolling_mean = series.rolling(window=window).mean()
    plt.figure(figsize=(12,5))
    plt.title(f"Moving average (window={window})")
    plt.plot(rolling_mean.index, rolling_mean, 'g', label='Rolling Mean')

    if plot_intervals:
        mae = np.mean(np.abs(series - rolling_mean))
        std = np.std(series - rolling_mean)
        lower_bound = rolling_mean - (mae + scale*std)
        upper_bound = rolling_mean + (mae + scale*std)
        plt.fill_between(rolling_mean.index, lower_bound, upper_bound, color='r', alpha=0.2)

    # Plot actual after the window
    plt.plot(series.index[window:], series[window:], 'orange', label='Actual')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


###############################################################################
# 6. Example Usage (Optional)
###############################################################################

if __name__ == "__main__":
    # Example usage with some file
    file_path = '/Users/dragomirazheleva/Desktop/PhD/PhD Projects/Dendrometer/All Dendrometer Data/H1-6 complete-selected/H1_complete.csv'
    df, device_name = load_and_prepare_data(file_path, start_date="2023-07-15", end_date="2023-09-19")
    if df is not None:
        # Identify outliers
        df = identify_and_adjust_outliers(df)
        
        # Basic plot
        plot_cleaned_displacement(df, device_name, show_original=True)
        
        # Possibly other steps...
        daily_agg = aggregate_environmental_data(df)
        print(daily_agg.head())
        
        # E.g. check stationarity
        check_stationarity(df['Cleaned Displacement'])
        
        # ARIMA example
        model_fit = model_arima(df['Cleaned Displacement'], order=(1,0,1))