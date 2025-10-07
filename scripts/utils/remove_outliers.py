import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

def remove_outliers_row(row, multiplier=1.5):
    """
    Remove outliers for a single row based on the IQR method.

    Parameters:
        row (pd.Series): A single row of the DataFrame.
        multiplier (float): The multiplier for the IQR to define the outlier thresholds. Default is 1.5.

    Returns:
        pd.Series: The row with outliers removed (NaN for outliers).
    """
    Q1 = row.quantile(0.25)
    Q3 = row.quantile(0.75)
    IQR = Q3 - Q1

    return row[(row > (Q1 - multiplier * IQR)) & (row < (Q3 + multiplier * IQR))]

def remove_outliers_parallel(dataframe, multiplier=1.5, n_jobs=1):
    """
    Remove outliers from a DataFrame in parallel based on the IQR method.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame from which outliers will be removed.
        multiplier (float): The multiplier for the IQR to define the outlier thresholds. Default is 1.5.
        n_jobs (int): Number of jobs to run in parallel. Default is 1 (do not run in parallel).

    Returns:
        pd.DataFrame: A DataFrame with outliers removed, maintaining the same orientation and column order as the input.
    """
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(remove_outliers_row, [row for _, row in dataframe.iterrows()], [multiplier] * len(dataframe)), 
                            total=len(dataframe), desc="Removing outliers"))

    # Reassemble the DataFrame
    return pd.DataFrame(results, index=dataframe.index, columns=dataframe.columns)