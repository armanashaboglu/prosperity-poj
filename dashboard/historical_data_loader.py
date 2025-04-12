'''
Loads historical price and trade data from Excel files for different days.
'''
import pandas as pd
import os
import glob

def load_historical_data(directory_path):
    """
    Loads historical price and trade data from Excel files found in the specified directory.

    Assumes filenames follow the pattern: prices_round_2_day_*.csv and trades_round_2_day_*_nn.csv

    Args:
        directory_path (str): The path to the directory containing the Excel files.

    Returns:
        dict: A dictionary where keys are the days (-1, 0) and values are
              dictionaries containing 'prices' and 'trades' DataFrames for that day.
              Returns an empty dictionary if the directory doesn't exist or no files are found.
              Example: {-1: {'prices': df_prices, 'trades': df_trades}, ...}
    """
    historical_data = {}
    # Round 2 data only contains days -1 and 0
    days = [-1, 0 , 1]

    if not os.path.isdir(directory_path):
        # Update the default path in the error message if needed, but the function uses the passed path
        print(f"Error: Directory not found: {directory_path}")
        return historical_data

    print(f"Loading historical data for Round 2 from: {directory_path}")

    for day in days:
        day_data = {}
        # Construct expected file paths for Round 2
        day_str = str(day) # Use string representation for matching

        prices_pattern = os.path.join(directory_path, f'prices_round_2_day_{day_str}.csv')
        trades_pattern = os.path.join(directory_path, f'trades_round_2_day_{day_str}_nn.csv')

        # Find files matching the pattern
        price_files = glob.glob(prices_pattern)
        trade_files = glob.glob(trades_pattern)

        # Load prices file
        if price_files:
            prices_file_path = price_files[0] # Take the first match
            try:
                df_prices = pd.read_csv(prices_file_path, sep=';')
                df_prices['day'] = day # Ensure day column is correct
                day_data['prices'] = df_prices
                print(f"  Loaded prices for day {day} from {os.path.basename(prices_file_path)}")
            except Exception as e:
                print(f"  Error loading prices file for day {day} ({os.path.basename(prices_file_path)}): {e}")
        else:
            print(f"  Warning: Prices file not found for day {day} (pattern: {os.path.basename(prices_pattern)})")

        # Load trades file
        if trade_files:
            trades_file_path = trade_files[0] # Take the first match
            try:
                # Adjust separator if needed for round 2 files, assuming ';' for now
                df_trades = pd.read_csv(trades_file_path, sep=';')
                df_trades['day'] = day # Add day column for consistency
                day_data['trades'] = df_trades
                print(f"  Loaded trades for day {day} from {os.path.basename(trades_file_path)}")
            except Exception as e:
                print(f"  Error loading trades file for day {day} ({os.path.basename(trades_file_path)}): {e}")
        else:
            print(f"  Warning: Trades file not found for day {day} (pattern: {os.path.basename(trades_pattern)})")

        # Only add day to historical_data if we loaded at least one file for it
        if day_data:
            historical_data[day] = day_data

    if not historical_data:
        print("Warning: No historical data files were successfully loaded.")

    return historical_data

# Example Usage (optional, for testing)
if __name__ == '__main__':
    # IMPORTANT: Replace with the ACTUAL path to your Round 2 data directory
    data_dir = 'C:/Users/Admin/Downloads/round-2-island-data-bottle/round-2-island-data-bottle'

    loaded_data = load_historical_data(data_dir)

    if loaded_data:
        print("\n--- Data Loading Summary ---")
        for day, data in loaded_data.items():
            print(f"Day {day}:")
            if 'prices' in data:
                print(f"  Prices DataFrame shape: {data['prices'].shape}")
                print("  Prices Columns:", data['prices'].columns.tolist())
                # print(data['prices'].head())
            if 'trades' in data:
                print(f"  Trades DataFrame shape: {data['trades'].shape}")
                print("  Trades Columns:", data['trades'].columns.tolist())
                # print(data['trades'].head())
    else:
        print("\nNo data was loaded.") 