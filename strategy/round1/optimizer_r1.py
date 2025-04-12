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
import math

# --- Configuration ---

# --- !! IMPORTANT: SET THESE PATHS CORRECTLY !! ---
BACKTESTER_SCRIPT_PATH = "prosperity3bt" # Assumes it's runnable via 'python -m prosperity3bt'
# <<< Strategy being optimized (R1 Fallback)
ORIGINAL_STRATEGY_PATH = "strategy/round1/round1_fallback.py"
# <<< Temp dir reflects strategy + evaluation data
TEMP_DIR = "temp_optimizer_squid_rsi_fallback_vs_r1"
# <<< ADDED back Custom data path for R2 testing
CUSTOM_DATA_PATH = 'C:/Users/Admin/projects/prosperity-poj/strategy/round1/resources'
# --- !! END OF IMPORTANT PATHS !! ---

# Optuna Settings
N_TRIALS = 125 # Number of optimization trials to run (adjust as needed)
# <<< CHANGED Study Name to reflect R1 strategy tested on R2 data
STUDY_NAME = "squid_rsi_fallback_r1_vs_r1_opt"

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
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    # Remove Round 2 Products if not in fallback PARAMS
    # CROISSANT = "CROISSANT"
    # JAM = "JAM"
    # DJEMBE = "DJEMBE"
    # PICNIC_BASKET1 = "PICNIC_BASKET1"
    # PICNIC_BASKET2 = "PICNIC_BASKET2"

# --- Load BASE_PARAMS from the original strategy file ---
# Needs to reflect the structure of round1_fallback.py's PARAMS
BASE_PARAMS = {}
try:
    # Use the updated ORIGINAL_STRATEGY_PATH
    with open(ORIGINAL_STRATEGY_PATH, 'r') as f_base:
        content = f_base.read()
        # Update regex if needed, assume PARAMS structure is similar
        match = re.search(r"PARAMS\s*=\s*({.*?^})", content, re.DOTALL | re.MULTILINE)
        if match:
            params_str = match.group(1)
            # Ensure Product class definition includes all relevant products for exec
            exec_globals = {"Product": Product, "np": pd.NA, "math": math} # Add modules if needed
            exec_locals = {}
            exec(f"BASE_PARAMS = {params_str}", exec_globals, exec_locals)
            BASE_PARAMS = exec_locals['BASE_PARAMS']
            print(f"Successfully loaded BASE_PARAMS from {ORIGINAL_STRATEGY_PATH}.")
        else:
            print(f"WARNING: Could not automatically extract BASE_PARAMS from {ORIGINAL_STRATEGY_PATH}. Using fallback defaults.")
            # Fallback defaults for RSI Strategy in fallback file
            BASE_PARAMS = {
                Product.SQUID_INK: {
                    "rsi_window": 106,
                    "rsi_overbought": 52,
                    "rsi_oversold": 41,
                },
                Product.RAINFOREST_RESIN: {"fair_value": 10000},
                # KELP does not use PARAMS in the fallback script
            }
except Exception as e:
    print(f"ERROR loading BASE_PARAMS from {ORIGINAL_STRATEGY_PATH}: {e}. Using fallback defaults.")
    # Fallback defaults for RSI Strategy on error
    BASE_PARAMS = {
        Product.SQUID_INK: {
            "rsi_window": 106,
            "rsi_overbought": 52,
            "rsi_oversold": 41,
        },
        Product.RAINFOREST_RESIN: {"fair_value": 10000},
    }

def modify_strategy_params(original_file_path, temp_file_path, new_params):
    """Reads the original strategy, replaces the PARAMS dict, writes to temp file."""
    try:
        with open(original_file_path, 'r') as f:
            lines = f.readlines()

        # Convert the new params dict to its string representation
        # Important: Handle all keys present in the original BASE_PARAMS structure
        params_for_json = {}
        for key, value in new_params.items():
             # Update mapping based on fallback Product usage
             if key == Product.RAINFOREST_RESIN: key_str = 'Product.RAINFOREST_RESIN'
             elif key == Product.SQUID_INK: key_str = 'Product.SQUID_INK'
             elif key == Product.KELP: key_str = 'Product.KELP' # Keep if present in original PARAMS, even if unused by strategy class
             else:
                  print(f"Warning: Unrecognized product key in new_params: {key}")
                  key_str = repr(key) # Fallback to representation
             params_for_json[key_str] = value

        # Manually format the string for better readability and correctness
        # This avoids issues with json.dumps and Product attribute formatting
        formatted_params_lines = ["PARAMS = {\n"]
        param_items = list(params_for_json.items()) # Use list for index access
        for i, (key_str, value_dict) in enumerate(param_items):
            formatted_params_lines.append(f"    {key_str}: {{\n")
            value_items = list(value_dict.items())
            for j, (param_name, param_value) in enumerate(value_items):
                formatted_params_lines.append(f'        "{param_name}": {json.dumps(param_value)}')
                if j < len(value_items) - 1: # Add comma if not last item in inner dict
                    formatted_params_lines.append(",\n")
                else:
                    formatted_params_lines.append("\n")
            formatted_params_lines.append("    }")
            if i < len(param_items) - 1: # Add comma if not last item in outer dict
                 formatted_params_lines.append(",\n")
            else:
                 formatted_params_lines.append("\n")
        formatted_params_lines.append("}\n")
        new_params_lines = formatted_params_lines

        output_lines = []
        in_params_dict = False
        params_replaced = False
        brace_count = 0

        for line in lines:
            stripped_line = line.strip()

            if stripped_line.startswith("PARAMS = {"):
                if not params_replaced:
                    # Find indentation
                    indent = line[:line.find("PARAMS")]
                    # Apply indentation to the generated lines
                    indented_new_params_lines = [f"{indent}{l}" for l in new_params_lines]
                    output_lines.extend(indented_new_params_lines)

                    in_params_dict = True
                    brace_count = stripped_line.count('{') - stripped_line.count('}')
                    # Handle single-line PARAMS definition if necessary
                    if brace_count <= 0 and stripped_line.endswith("}"):
                         in_params_dict = False
                    params_replaced = True
                continue # Skip the original PARAMS start line

            if in_params_dict:
                # This logic correctly handles multi-line dictionaries
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    in_params_dict = False
                continue # Skip the original PARAMS lines

            output_lines.append(line) # Add lines outside the PARAMS dict

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

    # 1. Suggest Parameters (Now ONLY for SQUID_INK RSI)
    trial_id = trial.number
    # Start with a deep copy of the base parameters
    suggested_params = copy.deepcopy(BASE_PARAMS)

    # Retrieve non-optimized base params if needed (Resin fair_value)
    # base_squid_params = BASE_PARAMS.get(Product.SQUID_INK, {}) # Not needed for RSI opt

    if trial_id == 0:
        print("*** Trial 0: Testing original unmodified strategy file ***")
        # Run with BASE_PARAMS extracted from the file
        # The suggested_params already holds the base values
    else:
        # Suggest ONLY the RSI parameters for SQUID_INK
        rsi_window = trial.suggest_int("squid_rsi_window", 5, 250) # Wider range for window
        rsi_overbought = trial.suggest_float("squid_rsi_ob", 50.0, 95.0, step=1.0) # OB usually > 50
        rsi_oversold = trial.suggest_float("squid_rsi_os", 5.0, 50.0, step=1.0) # OS usually < 50

        squid_rsi_params = {
            # Use suggested RSI params
            "rsi_window": rsi_window,
            "rsi_overbought": rsi_overbought,
            "rsi_oversold": rsi_oversold,
        
        }
        # Update only the SQUID_INK part of the parameters
        # Ensure SQUID_INK key exists before assignment
        if Product.SQUID_INK not in suggested_params:
             suggested_params[Product.SQUID_INK] = {}
        suggested_params[Product.SQUID_INK].update(squid_rsi_params)
        # Ensure other product parameters (like RAINFOREST_RESIN) remain untouched

    # --- Input validation --- # Validate ONLY suggested RSI params
    squid_params = suggested_params[Product.SQUID_INK]
    if squid_params["rsi_window"] < 2:
        print(f"Trial {trial_id}: Invalid RSI window {squid_params['rsi_window']} < 2. Pruning.")
        raise optuna.TrialPruned()
    if squid_params["rsi_oversold"] >= squid_params["rsi_overbought"]:
        print(f"Trial {trial_id}: Invalid RSI thresholds OS {squid_params['rsi_oversold']} >= OB {squid_params['rsi_overbought']}. Pruning.")
        raise optuna.TrialPruned()
    # REMOVED validation for Dual Trigger params
    # if squid_params["take_profit_offset"] <= 0 ...
    # --- End Input Validation ---

    # 2. Prepare for Backtest
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Copy datamodel.py (Ensure it's available)
    datamodel_src_path = "datamodel.py"
    datamodel_dest_path = os.path.join(TEMP_DIR, "datamodel.py")
    try:
        if os.path.exists(datamodel_src_path):
            shutil.copy2(datamodel_src_path, datamodel_dest_path)
        else:
            # Attempt to find datamodel.py relative to the ORIGINAL_STRATEGY_PATH
            strategy_dir = os.path.dirname(ORIGINAL_STRATEGY_PATH)
            project_root = os.path.dirname(strategy_dir) # Go up one more level
            potential_dm_path = os.path.join(project_root, "datamodel.py")
            if os.path.exists(potential_dm_path):
                 shutil.copy2(potential_dm_path, datamodel_dest_path)
                 print(f"Found and copied datamodel.py from: {potential_dm_path}")
            else:
                 print(f"Error: datamodel.py not found in current dir or project root ({project_root}). Cannot run backtest.")
                 raise optuna.TrialPruned()
    except Exception as e:
        print(f"Error copying datamodel.py: {e}")
        raise optuna.TrialPruned()

    # Unique names for temp files
    timestamp = int(time.time())
    random_suffix = random.randint(1000,9999)
    temp_strategy_filename = f"temp_strategy_{trial_id}_{timestamp}_{random_suffix}.py"
    temp_log_filename = f"temp_backtest_{trial_id}_{timestamp}_{random_suffix}.log" # Log still useful for failures
    temp_strategy_path = os.path.join(TEMP_DIR, temp_strategy_filename)
    temp_log_path = os.path.join(TEMP_DIR, temp_log_filename) # Keep log path

    # 3. Modify Strategy File
    if not modify_strategy_params(ORIGINAL_STRATEGY_PATH, temp_strategy_path, suggested_params):
         print(f"Trial {trial_id}: Failed to modify strategy file. Pruning.")
         if os.path.exists(temp_strategy_path):
              try: os.remove(temp_strategy_path)
              except OSError: pass
         raise optuna.TrialPruned()
    # Print only the SQUID_INK params being tested for brevity
    print(f"Trial {trial_id}: Testing SQUID_INK RSI params -> {suggested_params.get(Product.SQUID_INK, 'Params Missing!')}")

    # 4. Run Backtester (Updated Command for Round 2 evaluation)
    temp_strategy_path_safe = temp_strategy_path.replace("\\", "/")
    custom_data_path_safe = CUSTOM_DATA_PATH.replace("\\", "/") # Use the defined path

    backtest_command = [
        "python", "-m", BACKTESTER_SCRIPT_PATH,
        temp_strategy_path_safe, # Use the temp file with trial params
        "1", # <<< Run on Round 1 days
        "--match-trades", "worse", # <<< ADDED match trades flag
        "--data", custom_data_path_safe, # <<< ADDED custom data path flag
        "--no-out", # <<< ADDED: Direct backtester log output
    ]
    print(f"\nTrial {trial_id}: Running backtest -> {' '.join(backtest_command)}")

    final_pnl = -float('inf')
    backtest_failed = False
    try:
        # Increased timeout slightly
        result = subprocess.run(
            backtest_command, capture_output=True, text=True, check=False,
            timeout=360, encoding='utf-8', errors='replace'
        )

        # Write stdout/stderr to log file for debugging failed runs
        if result.returncode != 0:
             backtest_failed = True
             print(f"Trial {trial_id}: Backtester failed with return code {result.returncode}.")
             try:
                 with open(temp_log_path, 'w', encoding='utf-8') as log_f:
                     log_f.write("--- STDOUT ---\n")
                     log_f.write(result.stdout if result.stdout else "N/A")
                     log_f.write("\n--- STDERR ---\n")
                     log_f.write(result.stderr if result.stderr else "N/A")
                 print(f"  -> Wrote stdout/stderr to {temp_log_path} (will be deleted)") # Indicate deletion
             except Exception as log_e:
                 print(f"  -> Error writing stdout/stderr to log file: {log_e}")

        # Parse stdout for total PnL (Existing logic should work)
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

        if not found_pnl and not backtest_failed:
             print(f"Warning: 'Total profit:' line not found in stdout summary for successful Trial {trial_id}.")
             print(f"--- Stdout for Trial {trial_id} ---:")
             print(result.stdout)
             print("--- End Stdout ---:")
             # Consider pruning if PnL not found even on success
             # raise optuna.TrialPruned()

    except subprocess.TimeoutExpired:
        print(f"Trial {trial_id}: Backtester timed out.")
        backtest_failed = True # Treat timeout as failure for cleanup
        # Optionally write timeout info to log
        try:
            with open(temp_log_path, 'w', encoding='utf-8') as log_f:
                log_f.write("Backtester timed out.")
            print(f"  -> Wrote timeout info to {temp_log_path} (will be deleted)") # Indicate deletion
        except Exception as log_e:
             print(f"  -> Error writing timeout info to log file: {log_e}")

    except Exception as e:
        print(f"Trial {trial_id}: Error during backtest execution or parsing: {e}")
        import traceback
        traceback.print_exc()
        backtest_failed = True # Treat error as failure
        # Optionally write error info to log
        try:
            with open(temp_log_path, 'w', encoding='utf-8') as log_f:
                log_f.write(f"Error during execution/parsing:\n{traceback.format_exc()}")
            print(f"  -> Wrote error info to {temp_log_path} (will be deleted)") # Indicate deletion
        except Exception as log_e:
             print(f"  -> Error writing error info to log file: {log_e}")

    finally:
        # Cleanup Temporary Files
        # Decide if strategy file should be kept (e.g., for initial trials or failures)
        keep_strategy_file = (trial_id < 5 or backtest_failed)
        if keep_strategy_file:
            print(f"Trial {trial_id}: Keeping strategy file: {temp_strategy_filename}")
            if backtest_failed and os.path.exists(temp_log_path):
                 print(f"Trial {trial_id}: Backtest failed, log content was written to {temp_log_filename} (will be deleted).")

        # Always try to remove the log file
        if os.path.exists(temp_log_path):
            try:
                os.remove(temp_log_path)
                # Optional: print(f"Trial {trial_id}: Removed log file: {temp_log_filename}")
            except OSError as e:
                print(f"Error removing log file {temp_log_filename}: {e}")

        # Remove strategy file unless marked to keep
        if os.path.exists(temp_strategy_path) and not keep_strategy_file:
            try:
                os.remove(temp_strategy_path)
            except OSError as e:
                print(f"Error removing temp strategy file {temp_strategy_filename}: {e}")

    # Handle cases where PnL couldn't be determined
    if final_pnl == -float('inf'):
         print(f"Trial {trial_id}: Could not determine PnL. Pruning.")
         raise optuna.TrialPruned()

    return final_pnl

# --- Main Execution ---
if __name__ == "__main__":

    print(f"Starting Optuna optimization for SQUID_INK RSI in {ORIGINAL_STRATEGY_PATH} (evaluated on R1 data)...") # Updated print
    storage_name = f"sqlite:///{STUDY_NAME}.db"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5) # Adjust pruner settings if needed
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
            print(f"    Params (SQUID_INK Optimized - RSI Only): ") # Updated description
            # Extract just the suggested RSI params that were OPTIMIZED
            best_squid_params = {}
            param_names = [
                "squid_rsi_window", "squid_rsi_ob", "squid_rsi_os"
                # REMOVED Dual Trigger params
                # "squid_tp_offset", ...
            ]
            for name in param_names:
                 # Map Optuna name back to strategy param name
                 if name == "squid_rsi_window": param_key = "rsi_window"
                 elif name == "squid_rsi_ob": param_key = "rsi_overbought"
                 elif name == "squid_rsi_os": param_key = "rsi_oversold"
                 else: continue # Should not happen with the filtered list
                 best_squid_params[param_key] = trial.params.get(name)

            # Print the extracted best parameters for user review
            print(f"      { {k: v for k, v in best_squid_params.items() if v is not None} }")

            # Generate the best PARAMS dictionary string for the strategy file
            print(f"\n--- Best PARAMS dictionary for {os.path.basename(ORIGINAL_STRATEGY_PATH)} ---")
            # Use BASE_PARAMS as the starting point
            best_params_dict = copy.deepcopy(BASE_PARAMS)

            # Update only the SQUID_INK part with the best trial results
            squid_update = {}
            # Update OPTIMIZED RSI params from best trial
            squid_update["rsi_window"] = best_squid_params.get('rsi_window')
            squid_update["rsi_overbought"] = best_squid_params.get('rsi_overbought')
            squid_update["rsi_oversold"] = best_squid_params.get('rsi_oversold')
            # REMOVED Dual Trigger params from update
            # squid_update["take_profit_offset"] = ...

            # Check if all OPTIMIZED values were found
            optimized_values_found = all(v is not None for k, v in squid_update.items())

            if optimized_values_found:
                # Ensure SQUID_INK exists before updating
                if Product.SQUID_INK not in best_params_dict:
                    best_params_dict[Product.SQUID_INK] = {}
                best_params_dict[Product.SQUID_INK].update(squid_update)
            else:
                 print("Warning: Could not retrieve all best optimized RSI parameters from Optuna trial.")

            # Convert the *full* best dict to string using the improved formatting logic
            params_for_json_best = {}
            for key, value in best_params_dict.items():
                 if key == Product.RAINFOREST_RESIN: key_str = 'Product.RAINFOREST_RESIN'
                 elif key == Product.SQUID_INK: key_str = 'Product.SQUID_INK'
                 elif key == Product.KELP: key_str = 'Product.KELP' # Keep if present
                 else: key_str = repr(key)
                 params_for_json_best[key_str] = value

            # Manually format the output string like before
            formatted_params_lines_best = ["PARAMS = {\n"]
            param_items_best = list(params_for_json_best.items())
            for i, (key_str, value_dict) in enumerate(param_items_best):
                formatted_params_lines_best.append(f"    {key_str}: {{\n")
                value_items_best = list(value_dict.items())
                for j, (param_name, param_value) in enumerate(value_items_best):
                    formatted_params_lines_best.append(f'        "{param_name}": {json.dumps(param_value)}')
                    if j < len(value_items_best) - 1:
                        formatted_params_lines_best.append(",\n")
                    else:
                        formatted_params_lines_best.append("\n")
                formatted_params_lines_best.append("    }")
                if i < len(param_items_best) - 1:
                    formatted_params_lines_best.append(",\n")
                else:
                    formatted_params_lines_best.append("\n")
            formatted_params_lines_best.append("}\n")

            print("".join(formatted_params_lines_best)) # Print the formatted string
            print("--- End Best PARAMS ---")

    except ValueError as e:
         print(f"  Best trial not available (perhaps all trials failed or were pruned): {e}")

  