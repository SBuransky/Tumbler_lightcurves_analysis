import pandas as pd
import numpy as np
from utils.load_dataset import load_data


def simulate_night_observations(df: pd.DataFrame, output_filename: str) -> None:
    """
    Simulate night-only observations from an astronomical time series DataFrame,
    resample to one frame per 10 minutes, and save the result to a .txt file without column headers.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns ['julian_day', 'noisy_flux', 'deviation_used'].
        output_filename (str): Name of the output .txt file to save the filtered data.
    """
    # Check if required columns exist
    required_cols = {"julian_day", "noisy_flux", "deviation_used"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Calculate the time of day from julian_day
    hours = (df["julian_day"] % 1) * 24

    # Keep only night-time observations (before 6 AM or after 6 PM)
    night_mask = (hours > 16) | (hours < 7)
    night_df = df[night_mask].reset_index(drop=True)

    # Sort by julian_day just in case
    night_df = night_df.sort_values("julian_day").reset_index(drop=True)

    # Resample: keep one frame every 10 minutes
    # 10 minutes = 10 / (24*60) days ≈ 0.006944 days
    min_time_diff = 0 / (24 * 60)  # days

    sampled_rows = []
    last_time = -np.inf  # initialize to very small number

    for idx, row in night_df.iterrows():
        if row["julian_day"] - last_time >= min_time_diff:
            sampled_rows.append(row)
            last_time = row["julian_day"]

    resampled_df = pd.DataFrame(sampled_rows)

    # Save to txt without header and index
    resampled_df.to_csv(
        output_filename, sep="\t", index=False, header=False, float_format="%.6f"
    )
    print(len(resampled_df))

    print(f"Night-time observations (resampled) saved to {output_filename}.")


# Example usage:
df = load_data(
    "ID1913",
    column_names=("julian_day", "noisy_flux", "deviation_used"),
    appendix=".txt",
)
simulate_night_observations(df, "data/ID1915.txt")
