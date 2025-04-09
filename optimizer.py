import optuna
import subprocess
import os
import re
import json
import pandas as pd
from io import StringIO
import shutil
import time
import random
import copy # Import copy for deep copying parameters

# --- Configuration ---

# --- !! IMPORTANT: SET THESE PATHS CORRECTLY !! ---
BACKTESTER_SCRIPT_PATH = "prosperity3bt" # Assumes it's runnable via 'python -m prosperity3bt'
ORIGINAL_STRATEGY_PATH = "strategy/round1/round1_fallback.py" # <<< CHANGED to fallback
TEMP_DIR = "temp_optimizer_rsi_fallback" # Directory for temporary files
# --- !! END OF IMPORTANT PATHS !! ---

# Optuna Settings
N_TRIALS = 100 # Number of optimization trials to run
STUDY_NAME = "squid_ink_rsi_fallback_optimization" # <<< CHANGED Study Name

# --- Log Parsing Helper (Simplified from dashboard/log_parser) ---

def parse_backtest_log_for_pnl(log_file_path):
    """Parses the backtester log file to extract the final total PnL from the summary."""
    total_pnl = -float('inf') # Default to very low PnL
    in_summary_section = False
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line_strip = line.strip()

            # Check if we are entering the summary section
            if "Profit summary:" in line_strip:
                in_summary_section = True
                continue # Move to the next line looking for the total

            # Once in the summary section, look for the total profit line
            if in_summary_section:
                if line_strip.startswith("Total profit:"):
                    # Extract the number after the colon
                    try:
                        # Use regex to find the number (handles integers and decimals)
                        match = re.search(r':\s*([-+]?\d*\.?\d+)', line_strip)
                        if match:
                            total_pnl = float(match.group(1))
                            print(f"Trial Log: {os.path.basename(log_file_path)} -> Found Total PnL: {total_pnl:.2f}")
                            # Found the total, no need to read further lines
                            return total_pnl
                        else:
                            print(f"Warning: Could not extract numeric PnL from line: {line_strip}")
                            return -float('inf') # Failed to extract number
                    except ValueError:
                        print(f"Warning: Could not convert PnL value to float from line: {line_strip}")
                        return -float('inf') # Conversion error
                # Optional: Add a check to stop searching if we move past the summary section
                # (This might be overly complex if the format is consistent)

        # If loop finishes without finding the total profit line after the summary header
        if total_pnl == -float('inf'):
             print(f"Warning: 'Total profit:' line not found in summary section of log: {log_file_path}")
        return total_pnl # Return -inf if not found

    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file_path}")
        return -float('inf') # Return very low PnL if log file doesn't exist
    except Exception as e:
        print(f"Error parsing log file {log_file_path} for summary PnL: {e}")
        import traceback
        traceback.print_exc()
        return -float('inf') # Return very low PnL on other errors


# --- Parameter Modification Helper ---

# Define Product classes globally for use in modify_strategy_params and main block
class Product:
    KELP = "KELP" # KELP is no longer in PARAMS in fallback
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"

# --- Load BASE_PARAMS from the original strategy file --- 
# Needs to reflect the structure of round1_fallback.py's PARAMS
BASE_PARAMS = {}
try:
    with open(ORIGINAL_STRATEGY_PATH, 'r') as f_base:
        content = f_base.read()
        match = re.search(r"PARAMS\s*=\s*({.*?^})", content, re.DOTALL | re.MULTILINE)
        if match:
            params_str = match.group(1)
            exec(f"BASE_PARAMS = {params_str}", globals(), locals())
            print("Successfully loaded BASE_PARAMS from fallback strategy file.")
        else:
            print("WARNING: Could not automatically extract BASE_PARAMS from fallback strategy file. Using fallback defaults.")
            # Define fallback defaults matching fallback structure
            BASE_PARAMS = {
                Product.SQUID_INK: {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_trade_size": 5},
                Product.RAINFOREST_RESIN: {"fair_value": 10000},
            }
except Exception as e:
    print(f"ERROR loading BASE_PARAMS from {ORIGINAL_STRATEGY_PATH}: {e}. Using fallback defaults.")
    # Define fallback defaults on error
    BASE_PARAMS = {
        Product.SQUID_INK: {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_trade_size": 5},
        Product.RAINFOREST_RESIN: {"fair_value": 10000},
    }

def modify_strategy_params(original_file_path, temp_file_path, new_params):
    """Reads the original strategy, replaces the PARAMS dict, writes to temp file."""
    try:
        with open(original_file_path, 'r') as f:
            lines = f.readlines()

        # Convert the new params dict to its string representation
        params_for_json = {}
        for key, value in new_params.items():
             # Use direct Product attributes for keys
             if key == Product.RAINFOREST_RESIN: key_str = 'Product.RAINFOREST_RESIN'
             elif key == Product.SQUID_INK: key_str = 'Product.SQUID_INK'
             else:
                  print(f"Warning: Unrecognized key in new_params: {key}")
                  key_str = repr(key) # Fallback to representation
             params_for_json[key_str] = value

        new_params_str_json = json.dumps(params_for_json, indent=4)
        # Replace keys correctly
        new_params_str = new_params_str_json.replace('"Product.RAINFOREST_RESIN"', 'Product.RAINFOREST_RESIN')
        new_params_str = new_params_str.replace('"Product.SQUID_INK"', 'Product.SQUID_INK')
        # Add boolean replacements if needed (not needed for current fallback PARAMS)
        # new_params_str = new_params_str.replace(': false', ': False')
        # new_params_str = new_params_str.replace(': true', ': True')

        # Prepare the lines for the new dictionary
        indent = ''
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith("PARAMS = {"):
                 indent = line[:line.find("PARAMS")]
                 break
        new_params_lines = [f"{indent}PARAMS = {new_params_str}\n"]

        output_lines = []
        in_params_dict = False
        params_replaced = False
        brace_count = 0

        for line in lines:
            stripped_line = line.strip()

            if stripped_line.startswith("PARAMS = {"):
                if not params_replaced:
                    output_lines.extend(new_params_lines)
                    in_params_dict = True
                    brace_count = stripped_line.count('{') - stripped_line.count('}')
                    if brace_count <= 0:
                         in_params_dict = False
                    params_replaced = True
                continue

            if in_params_dict:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    in_params_dict = False
                continue

            output_lines.append(line)

        if not params_replaced:
             raise ValueError("Could not find 'PARAMS = {' line to replace in strategy file.")

        with open(temp_file_path, 'w') as f:
            f.writelines(output_lines)
        return True

    except Exception as e:
        print(f"Error modifying strategy parameters: {e}")
        import traceback
        traceback.print_exc()
        return False


# --- Optuna Objective Function ---

def objective(trial: optuna.Trial):
    """Function for Optuna to optimize."""

    # 1. Suggest Parameters (Only for SQUID_INK RSI)
    trial_id = trial.number
    # Start with a deep copy of the base parameters from fallback file
    suggested_params = copy.deepcopy(BASE_PARAMS)

    if trial_id == 0:
        print("*** Trial 0: Testing original unmodified fallback strategy file ***")
        # No changes needed, suggested_params is already BASE_PARAMS
    else:
        # Suggest parameters ONLY for SQUID_INK RSI
        squid_rsi_params = {
            "rsi_period": trial.suggest_int("squid_rsi_period", 5, 40), # Expanded range slightly
            "rsi_overbought": trial.suggest_int("squid_rsi_overbought", 60, 90),
            "rsi_oversold": trial.suggest_int("squid_rsi_oversold", 10, 40),
            "rsi_trade_size": trial.suggest_int("squid_rsi_trade_size", 1, 20) # Expanded range
        }
        # Update only the SQUID_INK part of the parameters
        suggested_params[Product.SQUID_INK] = squid_rsi_params
        # Ensure RAINFOREST_RESIN parameters remain untouched

    # --- Input validation for RSI levels ---
    # Ensure oversold < overbought
    if suggested_params[Product.SQUID_INK]['rsi_oversold'] >= suggested_params[Product.SQUID_INK]['rsi_overbought']:
        print(f"Trial {trial_id}: Oversold level ({suggested_params[Product.SQUID_INK]['rsi_oversold']}) >= Overbought level ({suggested_params[Product.SQUID_INK]['rsi_overbought']}). Pruning.")
        raise optuna.TrialPruned("Oversold level cannot be greater than or equal to overbought level.")
    # --- End Input Validation --- 

    # 2. Prepare for Backtest
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Copy datamodel.py (No change needed here)
    datamodel_src_path = "datamodel.py"
    datamodel_dest_path = os.path.join(TEMP_DIR, "datamodel.py")
    try:
        if os.path.exists(datamodel_src_path):
            shutil.copy2(datamodel_src_path, datamodel_dest_path)
        else:
            print(f"Error: {datamodel_src_path} not found. Cannot run backtest.")
            raise optuna.TrialPruned()
    except Exception as e:
        print(f"Error copying datamodel.py: {e}")
        raise optuna.TrialPruned()

    # Unique names for temp files (No change needed here)
    timestamp = int(time.time())
    random_suffix = random.randint(1000,9999)
    temp_strategy_filename = f"temp_strategy_{trial_id}_{timestamp}_{random_suffix}.py"
    temp_log_filename = f"temp_backtest_{trial_id}_{timestamp}_{random_suffix}.log"
    temp_strategy_path = os.path.join(TEMP_DIR, temp_strategy_filename)
    temp_log_path = os.path.join(TEMP_DIR, temp_log_filename)

    # 3. Modify Strategy File (Uses the suggested_params which now only modifies SQUID_INK)
    if not modify_strategy_params(ORIGINAL_STRATEGY_PATH, temp_strategy_path, suggested_params):
         print(f"Trial {trial_id}: Failed to modify strategy file. Pruning.")
         if os.path.exists(temp_strategy_path):
              try: os.remove(temp_strategy_path)
              except OSError: pass
         raise optuna.TrialPruned()
    print(f"Trial {trial_id}: Using parameters -> SQUID_INK: {suggested_params[Product.SQUID_INK]}, RAINFOREST_RESIN: {suggested_params[Product.RAINFOREST_RESIN]}")

    # 4. Run Backtester (Command remains the same)
    temp_strategy_path_safe = temp_strategy_path.replace("\\", "/")
    temp_log_path_safe = temp_log_path.replace("\\", "/")

    backtest_command = [
        "python", "-m", BACKTESTER_SCRIPT_PATH,
        temp_strategy_path_safe,
        "1", # Run on Round 1 days
        "--out", temp_log_path_safe
    ]
    print(f"\nTrial {trial_id}: Running backtest -> {' '.join(backtest_command)}")

    final_pnl = -float('inf')
    backtest_failed = False
    try:
        result = subprocess.run(
            backtest_command, capture_output=True, text=True, check=False,
            timeout=300, encoding='utf-8', errors='replace'
        )

        if result.returncode != 0:
             backtest_failed = True
             print(f"Trial {trial_id}: Backtester failed with return code {result.returncode}.")
             if result.stdout: print("Stdout (last 500 chars):", result.stdout[-500:])
             else: print("Stdout: Not available")
             if result.stderr: print("Stderr (last 500 chars):", result.stderr[-500:])
             else: print("Stderr: Not available")
        else:
            # Parse stdout for total PnL (No change needed here)
            stdout_lines = result.stdout.splitlines()
            in_summary_section = False
            found_pnl = False
            for line in stdout_lines:
                line_strip = line.strip()
                if "Profit summary:" in line_strip:
                    in_summary_section = True
                    continue
                if in_summary_section and line_strip.startswith("Total profit:"):
                    try:
                        match = re.search(r':\s*([-+]?\d*\.?\d+)', line_strip)
                        if match:
                            final_pnl = float(match.group(1))
                            print(f"Trial {trial_id}: Parsed Total PnL from stdout: {final_pnl:.2f}")
                            found_pnl = True
                            break
                        else:
                            print(f"Warning: Could not extract numeric PnL from stdout line: {line_strip}")
                            break
                    except ValueError:
                        print(f"Warning: Could not convert PnL value to float from stdout line: {line_strip}")
                        break

            if not found_pnl:
                 print(f"Warning: 'Total profit:' line not found in stdout summary for Trial {trial_id}.")
                 print(f"--- Stdout for Trial {trial_id} ---:")
                 print(result.stdout)
                 print("--- End Stdout ---:")

    except subprocess.TimeoutExpired:
        print(f"Trial {trial_id}: Backtester timed out.")
        backtest_failed = True # Treat timeout as failure for cleanup
    except Exception as e:
        print(f"Trial {trial_id}: Error during backtest execution or parsing: {e}")
        import traceback
        traceback.print_exc()
        backtest_failed = True # Treat error as failure
    finally:
        # Cleanup Temporary Files (Logic remains the same)
        keep_strategy_file = False
        keep_log_file = False

        if trial_id < 5: # Keep first 5
            print(f"Trial {trial_id}: Keeping files: {temp_strategy_path}, {temp_log_path}")
            keep_strategy_file = True
            keep_log_file = True
        elif backtest_failed:
            print(f"Trial {trial_id}: Keeping failed strategy/log files: {temp_strategy_path}, {temp_log_path}")
            keep_strategy_file = True
            keep_log_file = True # Keep log for failed runs too

        if os.path.exists(temp_strategy_path) and not keep_strategy_file:
            try: os.remove(temp_strategy_path)
            except OSError as e: print(f"Error removing temp strategy file {temp_strategy_path}: {e}")
        if os.path.exists(temp_log_path) and not keep_log_file:
             try: os.remove(temp_log_path)
             except OSError as e: print(f"Error removing temp log file {temp_log_path}: {e}")

    # Handle cases where PnL couldn't be determined
    if final_pnl == -float('inf'):
         print(f"Trial {trial_id}: Could not determine PnL. Pruning.")
         raise optuna.TrialPruned()

    return final_pnl

# --- Main Execution ---
if __name__ == "__main__":

    print(f"Starting Optuna optimization for SQUID_INK RSI in {ORIGINAL_STRATEGY_PATH}...")
    storage_name = f"sqlite:///{STUDY_NAME}.db"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=None)
    except KeyboardInterrupt:
         print("Optimization stopped manually.")
    except Exception as e:
         print(f"An error occurred during optimization: {e}")
         import traceback
         traceback.print_exc()

    print("\nOptimization Finished!")
    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")

    try:
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
             print("  No trials completed successfully.")
        else:
            print(f"  Best trial:")
            trial = study.best_trial
            print(f"    Value (Max PnL): {trial.value:.2f}")
            print(f"    Params (SQUID_INK RSI only): ")
            # Extract just the suggested RSI params
            best_squid_params = {}
            for key, value in trial.params.items():
                if key.startswith("squid_rsi_"): # Check prefix for RSI params
                     best_squid_params[key.replace("squid_", "")] = value # Store without prefix
            print(f"      {best_squid_params}")

            # Generate the best PARAMS dictionary string for the fallback strategy
            print("\n--- Best PARAMS dictionary for round1_fallback.py ---")
            # Use BASE_PARAMS as the starting point
            best_params_dict = copy.deepcopy(BASE_PARAMS)

            # Update only the SQUID_INK part with the best trial results
            best_params_dict[Product.SQUID_INK] = {
                "rsi_period": trial.params.get("squid_rsi_period"),
                "rsi_overbought": trial.params.get("squid_rsi_overbought"),
                "rsi_oversold": trial.params.get("squid_rsi_oversold"),
                "rsi_trade_size": trial.params.get("squid_rsi_trade_size"),
            }
            # Remove None values if any param wasn't suggested (shouldn't happen)
            best_params_dict[Product.SQUID_INK] = {k: v for k, v in best_params_dict[Product.SQUID_INK].items() if v is not None}

            # Convert the *full* best dict to string
            params_for_json_best = {}
            for key, value in best_params_dict.items():
                 if key == Product.RAINFOREST_RESIN: key_str = 'Product.RAINFOREST_RESIN'
                 elif key == Product.SQUID_INK: key_str = 'Product.SQUID_INK'
                 else: key_str = repr(key)
                 params_for_json_best[key_str] = value

            best_params_str_json = json.dumps(params_for_json_best, indent=4)
            best_params_str = best_params_str_json.replace('"Product.RAINFOREST_RESIN"', 'Product.RAINFOREST_RESIN')
            best_params_str = best_params_str.replace('"Product.SQUID_INK"', 'Product.SQUID_INK')
            # No boolean replacement needed for fallback PARAMS

            print(f"PARAMS = {best_params_str}")
            print("--- End Best PARAMS ---")

    except ValueError:
         print("  Best trial not available (perhaps all trials failed or were pruned).")

    # Optional: Clean up the temp directory
    # Consider keeping it for debugging if needed
    # if os.path.exists(TEMP_DIR):
    #    try:
    #        print(f"Removing temporary directory: {TEMP_DIR}")
    #        shutil.rmtree(TEMP_DIR)
    #    except OSError as e:
    #        print(f"Error removing temp directory {TEMP_DIR}: {e}")
