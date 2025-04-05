import json
import pandas as pd
from io import StringIO
import re # Import regex module

def parse_log_file(file_path):
    """
    Parses the IMC Prosperity log file, extracting data from the initial JSON
    blocks (sandboxLog, lambdaLog), the middle CSV section (Activities log),
    and the final JSON section (Trade History).

    Args:
        file_path (str): The path to the .log file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame containing the processed lambdaLog data.
            - pd.DataFrame: DataFrame containing the Activities log data.
            - pd.DataFrame: DataFrame containing the Trade History data.
    """
    all_lambda_logs = []
    activities_lines = []
    trade_history_json_lines = []
    in_activities_section = False
    in_trade_history_section = False

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        current_json_block = ""
        in_json_block = False

        for line in lines:
            line_strip = line.strip()

            # --- JSON Block Parsing (Initial sandbox/lambda logs) ---
            # Skip processing if we are already in other sections
            if not in_activities_section and not in_trade_history_section:
                if line_strip.startswith('{') and not in_json_block:
                    current_json_block = line
                    in_json_block = True
                    continue # Move to next line, start accumulating block
                elif in_json_block:
                    current_json_block += line
                    if line_strip.endswith('}'):
                        try:
                            log_entry = json.loads(current_json_block)
                            if 'lambdaLog' in log_entry:
                                lambda_log_str = log_entry['lambdaLog'].strip()
                                if lambda_log_str.startswith('[['):
                                    try:
                                        lambda_data = json.loads(lambda_log_str)
                                        all_lambda_logs.append(lambda_data)
                                    except json.JSONDecodeError as e:
                                        print(f"Warning: Could not decode lambdaLog JSON: {e}")
                                        print(f"Problematic lambdaLog string: {lambda_log_str[:200]}...")

                            current_json_block = ""
                            in_json_block = False
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON block: {e}")
                            current_json_block = ""
                            in_json_block = False
                        continue # Finished processing this JSON block

            # --- Section Transitions ---
            if line_strip.startswith("Activities log:"):
                in_activities_section = True
                in_trade_history_section = False # Ensure we're not in trade history
                in_json_block = False # Ensure we're not mid-JSON block
                # We expect the header on the *next* line
                continue # Skip the "Activities log:" line itself

            if line_strip.startswith("Trade History:"):
                in_activities_section = False # Stop activities parsing
                in_trade_history_section = True
                in_json_block = False # Ensure we're not mid-JSON block
                continue # Skip the "Trade History:" line itself

            # --- Data Collection by Section ---
            if in_activities_section:
                # The first line added will be the header
                activities_lines.append(line)

            elif in_trade_history_section:
                # Collect lines forming the trade history JSON array
                trade_history_json_lines.append(line)


        # --- Process Collected Data ---

        # Process Lambda Logs using List of Dictionaries approach
        lambda_state_df = pd.DataFrame() # Initialize empty final DataFrame
        processed_lambda_rows = []
        if all_lambda_logs:
            try:
                # Define regex to find true values in logs
                # Example log line: "KELP - true value: 2026, position: 27/50"
                true_value_regex = re.compile(r"(\w+) - true value: (\d+),?")

                for lambda_entry in all_lambda_logs:
                    if not isinstance(lambda_entry, list) or len(lambda_entry) < 1:
                        print(f"Warning: Skipping unexpected lambda entry structure: {lambda_entry}")
                        continue

                    # Extract the main components
                    state_list = lambda_entry[0]
                    orders_list = lambda_entry[1] if len(lambda_entry) > 1 else []
                    conversions_val = lambda_entry[2] if len(lambda_entry) > 2 else None
                    trader_data_str_val = lambda_entry[3] if len(lambda_entry) > 3 else ""
                    logs_str_val = lambda_entry[4] if len(lambda_entry) > 4 else ""

                    # --- Extract Fair Values from Logs --- 
                    calculated_fair_values = {}
                    if logs_str_val: # Check if log string exists
                        for line in logs_str_val.split('\n'): # Split logs into lines
                            match = true_value_regex.search(line)
                            if match:
                                product_name = match.group(1)
                                true_value = int(match.group(2))
                                calculated_fair_values[product_name] = true_value
                    # --- End Fair Value Extraction ---

                    # Check compressed_state structure before unpacking
                    if isinstance(state_list, list) and len(state_list) == 8:
                        row_dict = {
                            'timestamp': state_list[0],
                            'trader_data_in_state': state_list[1],
                            'listings': state_list[2],
                            'order_depths': state_list[3],
                            'own_trades': state_list[4],
                            'market_trades': state_list[5],
                            'positions': state_list[6],
                            'observations': state_list[7],
                            'compressed_orders_list': orders_list,
                            'conversions': conversions_val,
                            'trader_data_str': trader_data_str_val,
                            'logs_str': logs_str_val,
                            'calculated_fair_values': calculated_fair_values # Add the new column
                        }
                        processed_lambda_rows.append(row_dict)
                    else:
                        print(f"Warning: Unexpected compressed_state structure found: {state_list}")
                        # Optionally append a row with Nones or skip
                        # processed_lambda_rows.append({ # Example of adding row with Nones
                        #     'timestamp': None, 'trader_data_in_state': None, 'listings': None, 
                        #     'order_depths': None, 'own_trades': None, 'market_trades': None, 
                        #     'positions': None, 'observations': None, 'compressed_orders_list': orders_list,
                        #     'conversions': conversions_val, 'trader_data_str': trader_data_str_val, 
                        #     'logs_str': logs_str_val
                        # }) 

                # Create DataFrame from the list of dictionaries
                if processed_lambda_rows:
                    lambda_state_df = pd.DataFrame(processed_lambda_rows)
                else:
                    print("Warning: No valid lambda log rows were processed.")
                    lambda_state_df = pd.DataFrame() # Ensure it's an empty DF

            except Exception as e:
                print(f"Error processing lambda logs into structured DataFrame: {e}")
                lambda_state_df = pd.DataFrame() # Keep empty DataFrame on error

        # Process Activities Log
        activities_df = pd.DataFrame() # Initialize empty
        if activities_lines:
            # The first line is the header
            header = activities_lines[0]
            csv_data = "".join(activities_lines[1:]) # Join the rest for data
            full_csv = header + csv_data
            try:
                activities_df = pd.read_csv(StringIO(full_csv), sep=';')
            except Exception as e:
                print(f"Error parsing activities log CSV: {e}")
                # Keep empty DataFrame on error

        # Process Trade History
        trade_history_df = pd.DataFrame() # Initialize empty
        if trade_history_json_lines:
            trade_history_json_string = "".join(trade_history_json_lines)
            try:
                trade_history_data = json.loads(trade_history_json_string)
                trade_history_df = pd.DataFrame(trade_history_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing trade history JSON: {e}")
            except Exception as e:
                print(f"Error converting trade history to DataFrame: {e}")
                # Keep empty DataFrame on error

        # Return the processed lambda DataFrame along with the others
        return lambda_state_df, activities_df, trade_history_df

    except FileNotFoundError:
        print(f"Error: Log file not found at {file_path}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Example Usage (optional, for testing)
if __name__ == '__main__':
    # Replace with the actual path to your log file
    log_file_path = 'C:/Users/Admin/projects/prosperity-poj/strategy/tutorial/prototype.log' # Adjust path as needed

    lambda_state_df, activities_df, trade_history_df = parse_log_file(log_file_path)

    print("--- Lambda State Data (Processed) ---")
    if not lambda_state_df.empty:
        # Print head with selected columns for readability
        print(lambda_state_df[['timestamp', 'positions', 'listings', 'compressed_orders_list', 'trader_data_str']].head())
        print(f"Shape: {lambda_state_df.shape}")
        # Example: Accessing nested data from the first row (if exists)
        if not lambda_state_df.empty:
            print("\nExample accessing first row data:")
            first_row = lambda_state_df.iloc[0]
            print(f"  Timestamp: {first_row['timestamp']}")
            print(f"  Positions: {first_row['positions']}")
            # print(f"  First Listing: {first_row['listings'][0] if first_row['listings'] else 'N/A'}")
            # print(f"  Kelp Order Depth Keys: {first_row['order_depths'].get('KELP', 'N/A')}")
            print(f"  Raw Trader Data String: {str(first_row['trader_data_str'])[:100]}...") # Show snippet
    else:
        print("No Lambda Log data processed or error during processing.")

    print("\n--- Activities Log Data ---")
    if not activities_df.empty:
        print(activities_df.head())
        print(f"Shape: {activities_df.shape}")
        if 'profit_and_loss' in activities_df.columns:
            print("\nFound 'profit_and_loss' column.")
            # print(activities_df['profit_and_loss'].describe()) # Optional: uncomment for stats
        else:
            print("\n'profit_and_loss' column not found in Activities Log.")
    else:
        print("No Activities Log data processed or error during processing.")

    print("\n--- Trade History Data ---")
    if not trade_history_df.empty:
        print(trade_history_df.head())
        print(f"Shape: {trade_history_df.shape}")
    else:
        print("No Trade History data processed or error during processing.") 