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
import argparse # <<< Added argparse
import functools # <<< Added functools

# --- Configuration ---

# --- !! IMPORTANT: SET THESE PATHS CORRECTLY !! ---
BACKTESTER_SCRIPT_PATH = "prosperity3bt" # Assumes it's runnable via 'python -m prosperity3bt'
# <<< Strategy being optimized (R2 with Djembes RSI)
ORIGINAL_STRATEGY_PATH = "strategy/round2/round2_omer1.py"
# <<< Temp dir reflects R2 strategy + evaluation data
TEMP_DIR = "temp_optimizer_picnic_spread_r2"
# <<< ADDED back Custom data path for R2 testing
#CUSTOM_DATA_PATH = 'C:/Users/Admin/projects/prosperity-poj/strategy/round2/resources'
# --- !! END OF IMPORTANT PATHS !! ---

# Optuna Settings
N_TRIALS = 125 # Number of optimization trials to run (adjust as needed)
# <<< Study Name will be set dynamically based on target product
# STUDY_NAME = "picnic_spread_zscore_r2_opt_B2" # <-- REMOVED Static Name

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
                            # Optional: Keep minimal logging here if needed
                            # print(f"Trial Log: {os.path.basename(log_file_path)} -> Found Total PnL: {total_pnl:.2f}")
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
    # Round 1
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    # Round 2
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES" # Ensure DJEMBES is defined
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

# --- Load BASE_PARAMS from the original strategy file ---
# Needs to reflect the structure of round2_omer1.py's PARAMS
BASE_PARAMS = {}
# <<< Define position limits for validation (get from round2_omer1 or COMPETITION_INFO)
POSITION_LIMITS = {
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 100,
    # Add others if needed, but mainly for spread targets
}
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
            # Basic validation of expected keys
            if Product.PICNIC_BASKET1 not in BASE_PARAMS or Product.PICNIC_BASKET2 not in BASE_PARAMS:
                print("Warning: BASE_PARAMS missing Picnic Basket keys. Check extraction/original file.")
            else:
                # Validate loaded limits match expected if possible
                if Product.PICNIC_BASKET1 in POSITION_LIMITS and 'target_position' in BASE_PARAMS[Product.PICNIC_BASKET1]:
                    pass # No limit defined in PARAMS, only used for validation
                if Product.PICNIC_BASKET2 in POSITION_LIMITS and 'target_position' in BASE_PARAMS[Product.PICNIC_BASKET2]:
                     pass # No limit defined in PARAMS, only used for validation
        else:
            print(f"WARNING: Could not automatically extract BASE_PARAMS from {ORIGINAL_STRATEGY_PATH}.")
            # Provide minimal fallback if needed, but ideally loading works
            BASE_PARAMS = {
                Product.SQUID_INK: { # SQUID_INK defaults from round2_omer1
                    "rsi_window": 106, "rsi_overbought": 52, "rsi_oversold": 41,
                },
                Product.RAINFOREST_RESIN: {"fair_value": 10000},
                Product.PICNIC_BASKET1: { "default_spread_mean": 48.7, "spread_std_window": 50, "zscore_threshold": 1.5, "target_position": 10 },
                Product.PICNIC_BASKET2: { "default_spread_mean": 30.2, "spread_std_window": 50, "zscore_threshold": 1.5, "target_position": 15 }
            }
except Exception as e:
    print(f"ERROR loading BASE_PARAMS from {ORIGINAL_STRATEGY_PATH}: {e}. Using fallback defaults.")
    # Fallback defaults for R2 strategy on error
    BASE_PARAMS = {
        Product.SQUID_INK: { # SQUID_INK defaults
            "rsi_window": 106, "rsi_overbought": 52, "rsi_oversold": 41,
        },
        Product.RAINFOREST_RESIN: {"fair_value": 10000},
        Product.PICNIC_BASKET1: { "default_spread_mean": 48.7, "spread_std_window": 50, "zscore_threshold": 1.5, "target_position": 10 },
        Product.PICNIC_BASKET2: { "default_spread_mean": 30.2, "spread_std_window": 50, "zscore_threshold": 1.5, "target_position": 15 }
    }

def modify_strategy_params(original_file_path, temp_file_path, new_params):
    """Reads the original strategy, replaces the PARAMS dict, writes to temp file."""
    try:
        with open(original_file_path, 'r') as f:
            lines = f.readlines()

        # Convert the new params dict to its string representation
        params_for_json = {}
        for key, value in new_params.items():
             # Use Product.<NAME> representation for keys
             if key == Product.RAINFOREST_RESIN: key_str = 'Product.RAINFOREST_RESIN'
             elif key == Product.SQUID_INK: key_str = 'Product.SQUID_INK'
             elif key == Product.KELP: key_str = 'Product.KELP' # Keep if present
             elif key == Product.DJEMBES: key_str = 'Product.DJEMBES' # Keep if present
             # Add other R2 products if they appear in PARAMS
             elif key == Product.CROISSANTS: key_str = 'Product.CROISSANT'
             elif key == Product.JAMS: key_str = 'Product.JAMS'
             elif key == Product.PICNIC_BASKET1: key_str = 'Product.PICNIC_BASKET1'
             elif key == Product.PICNIC_BASKET2: key_str = 'Product.PICNIC_BASKET2'
             else:
                  # Safely handle potential non-product keys if BASE_PARAMS structure changes
                  if isinstance(key, str) and key.startswith("Product."):
                      key_str = key # Assume it's already formatted if loaded strangely
                  else:
                      print(f"Warning: Unrecognized product key type in new_params: {key} ({type(key)})")
                      key_str = repr(key) # Fallback to representation
             params_for_json[key_str] = value

        # Manually format the string (same logic as before)
        formatted_params_lines = ["PARAMS = {\n"]
        param_items = list(params_for_json.items())
        for i, (key_str, value_dict) in enumerate(param_items):
            formatted_params_lines.append(f"    {key_str}: {{\n")
            value_items = list(value_dict.items())
            for j, (param_name, param_value) in enumerate(value_items):
                # Ensure complex objects (like np.nan or math.inf if used) are handled if necessary
                try:
                    # json.dumps handles basic types (int, float, str, bool, None)
                    # For more complex types, might need a custom encoder or str()
                    value_repr = json.dumps(param_value)
                except TypeError:
                    value_repr = repr(param_value) # Fallback for unhandled types
                formatted_params_lines.append(f'        "{param_name}": {value_repr}')
                if j < len(value_items) - 1: formatted_params_lines.append(",\n")
                else: formatted_params_lines.append("\n")
            formatted_params_lines.append("    }")
            if i < len(param_items) - 1: formatted_params_lines.append(",\n")
            else: formatted_params_lines.append("\n")
        formatted_params_lines.append("}\n")
        new_params_lines = formatted_params_lines

        # Replacement logic remains the same
        output_lines = []
        in_params_dict = False
        params_replaced = False
        brace_count = 0
        for line in lines:
            stripped_line = line.strip()
            # Make the check more robust, handle potential extra spaces
            if stripped_line.startswith("PARAMS") and "=" in stripped_line and "{" in stripped_line:
                if not params_replaced:
                    indent = line[:line.find("PARAMS")] # Assumes PARAMS is at start of indent block
                    indented_new_params_lines = [f"{indent}{l}" for l in new_params_lines]
                    output_lines.extend(indented_new_params_lines)
                    in_params_dict = True
                    # Correct brace counting needed for multi-line dicts
                    brace_count = line.count('{') - line.count('}')
                    # Check if the definition ends on the same line
                    if brace_count <= 0 and stripped_line.endswith("}"):
                         in_params_dict = False # Handled single-line definition
                    params_replaced = True
                    continue # Skip the original PARAMS line(s)
            if in_params_dict:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                     in_params_dict = False # Found the closing brace
                continue # Skip the original PARAMS content
            # Only append lines if not part of the old PARAMS dict
            output_lines.append(line)

        if not params_replaced: raise ValueError("Could not find 'PARAMS = {' line to replace.")
        with open(temp_file_path, 'w') as f: f.writelines(output_lines)
        return True

    except Exception as e:
        print(f"Error modifying strategy parameters: {e}"); import traceback; traceback.print_exc(); return False


# --- Optuna Objective Function ---

def objective(trial: optuna.Trial, target_product: str):
    """Function for Optuna to optimize. Now takes target_product."""

    # 1. Suggest Parameters (Conditional based on target_product)
    trial_id = trial.number
    suggested_params = copy.deepcopy(BASE_PARAMS) # Start with base

    if target_product == Product.PICNIC_BASKET1:
        # --- Suggest Spread Params for PICNIC_BASKET1 --- 
        b1_spread_std_window = trial.suggest_int("b1_spread_std_window", 2, 200)
        b1_zscore_threshold = trial.suggest_float("b1_zscore_threshold", 1, 30, step=1)
        b1_target_position = trial.suggest_int("b1_target_position", 1, POSITION_LIMITS[Product.PICNIC_BASKET1])
        b1_close_threshold = trial.suggest_float("b1_close_threshold", 0.0, 2.0, step=0.05)

        # Update suggested_params dictionary for B1
        if Product.PICNIC_BASKET1 in suggested_params:
            suggested_params[Product.PICNIC_BASKET1].update({
                "spread_std_window": b1_spread_std_window,
                "zscore_threshold": b1_zscore_threshold,
                "target_position": b1_target_position,
                "close_threshold": b1_close_threshold
            })
        else:
            print(f"Warning: {Product.PICNIC_BASKET1} not in BASE_PARAMS, cannot update.")
            raise optuna.TrialPruned(f"{Product.PICNIC_BASKET1} missing from BASE_PARAMS.")

    elif target_product == Product.PICNIC_BASKET2:
        # --- Suggest Spread Params for PICNIC_BASKET2 --- 
        b2_spread_std_window = trial.suggest_int("b2_spread_std_window", 2, 200)
        b2_zscore_threshold = trial.suggest_float("b2_zscore_threshold", 1, 30, step=1)
        b2_target_position = trial.suggest_int("b2_target_position", 1, POSITION_LIMITS[Product.PICNIC_BASKET2])
        b2_close_threshold = trial.suggest_float("b2_close_threshold", 0.0, 2.0, step=0.05)

        # Update suggested_params dictionary for B2
        if Product.PICNIC_BASKET2 in suggested_params:
            suggested_params[Product.PICNIC_BASKET2].update({
                "spread_std_window": b2_spread_std_window,
                "zscore_threshold": b2_zscore_threshold,
                "target_position": b2_target_position,
                "close_threshold": b2_close_threshold
            })
        else:
            print(f"Warning: {Product.PICNIC_BASKET2} not in BASE_PARAMS, cannot update.")
            raise optuna.TrialPruned(f"{Product.PICNIC_BASKET2} missing from BASE_PARAMS.")
    
    elif target_product == Product.SQUID_INK:
        # --- Suggest RSI Params for SQUID_INK --- 
        rsi_window = trial.suggest_int("rsi_window", 10, 200) # Example range
        # Suggest thresholds ensuring overbought > oversold
        rsi_oversold = trial.suggest_int("rsi_oversold", 1, 49) # e.g., 1 to 49
        rsi_overbought = trial.suggest_int("rsi_overbought", rsi_oversold + 1, 99) # e.g., oversold+1 to 99

        # Update suggested_params dictionary for SQUID_INK
        if Product.SQUID_INK in suggested_params:
            suggested_params[Product.SQUID_INK].update({
                "rsi_window": rsi_window,
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold
            })
        else:
             print(f"Warning: {Product.SQUID_INK} not in BASE_PARAMS, cannot update.")
             raise optuna.TrialPruned(f"{Product.SQUID_INK} missing from BASE_PARAMS.")

    else:
        # Should not happen with argparse choices
        raise ValueError(f"Invalid target_product specified: {target_product}")

    # Keep other parameters (e.g., RESIN) untouched from BASE_PARAMS

    # --- Input validation (Conditional) --- 
    if target_product == Product.PICNIC_BASKET1:
        # Basket 1 Validation
        b1_params = suggested_params.get(Product.PICNIC_BASKET1, {})
        if b1_params.get("spread_std_window", 1) < 2:
            print(f"Trial {trial_id}: Invalid B1 window {b1_params.get('spread_std_window')} < 2. Pruning.")
            raise optuna.TrialPruned()
        if b1_params.get("zscore_threshold", 0) <= 0:
            print(f"Trial {trial_id}: Invalid B1 z-score threshold {b1_params.get('zscore_threshold')} <= 0. Pruning.")
            raise optuna.TrialPruned()
        if not (0 < b1_params.get("target_position", 0) <= POSITION_LIMITS[Product.PICNIC_BASKET1]):
            print(f"Trial {trial_id}: Invalid B1 target position {b1_params.get('target_position')} (Limit: {POSITION_LIMITS[Product.PICNIC_BASKET1]}). Pruning.")
            raise optuna.TrialPruned()
        if b1_params.get("close_threshold", -1) < 0:
            print(f"Trial {trial_id}: Invalid B1 close threshold {b1_params.get('close_threshold')} < 0. Pruning.")
            raise optuna.TrialPruned()
    elif target_product == Product.PICNIC_BASKET2:
        # Basket 2 Validation
        b2_params = suggested_params.get(Product.PICNIC_BASKET2, {})
        if b2_params.get("spread_std_window", 1) < 2:
            print(f"Trial {trial_id}: Invalid B2 window {b2_params.get('spread_std_window')} < 2. Pruning.")
            raise optuna.TrialPruned()
        if b2_params.get("zscore_threshold", 0) <= 0:
            print(f"Trial {trial_id}: Invalid B2 z-score threshold {b2_params.get('zscore_threshold')} <= 0. Pruning.")
            raise optuna.TrialPruned()
        if not (0 < b2_params.get("target_position", 0) <= POSITION_LIMITS[Product.PICNIC_BASKET2]):
             print(f"Trial {trial_id}: Invalid B2 target position {b2_params.get('target_position')} (Limit: {POSITION_LIMITS[Product.PICNIC_BASKET2]}). Pruning.")
             raise optuna.TrialPruned()
        if b2_params.get("close_threshold", -1) < 0:
            print(f"Trial {trial_id}: Invalid B2 close threshold {b2_params.get('close_threshold')} < 0. Pruning.")
            raise optuna.TrialPruned()
    elif target_product == Product.SQUID_INK:
        # Squid Ink Validation
        sq_params = suggested_params.get(Product.SQUID_INK, {})
        if sq_params.get("rsi_window", 0) < 2:
             print(f"Trial {trial_id}: Invalid SQ window {sq_params.get('rsi_window')} < 2. Pruning.")
             raise optuna.TrialPruned()
        # Check if thresholds exist and overbought > oversold (already enforced by suggestion, but good practice)
        if not (sq_params.get("rsi_overbought", -1) > sq_params.get("rsi_oversold", -1) > 0):
             print(f"Trial {trial_id}: Invalid SQ thresholds OB={sq_params.get('rsi_overbought')} OS={sq_params.get('rsi_oversold')}. Pruning.")
             raise optuna.TrialPruned()
            
    # --- End Input Validation ---

    # 2. Prepare for Backtest (Logic remains largely the same)
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
    datamodel_src_path = "datamodel.py"; datamodel_dest_path = os.path.join(TEMP_DIR, "datamodel.py")
    try:
        # Try copying from strategy dir first, then project root
        strategy_dir = os.path.dirname(ORIGINAL_STRATEGY_PATH)
        potential_dm_path_strat = os.path.join(strategy_dir, "datamodel.py")
        project_root = os.path.abspath(os.path.join(strategy_dir, '..', '..')) # Adjust levels if needed
        potential_dm_path_root = os.path.join(project_root, "datamodel.py")

        if os.path.exists(potential_dm_path_strat):
            shutil.copy2(potential_dm_path_strat, datamodel_dest_path)
            # print(f"Copied datamodel.py from strategy directory: {potential_dm_path_strat}")
        elif os.path.exists(potential_dm_path_root):
            shutil.copy2(potential_dm_path_root, datamodel_dest_path)
            # print(f"Copied datamodel.py from project root: {potential_dm_path_root}")
        elif os.path.exists(datamodel_src_path): # Check current dir as last resort
             shutil.copy2(datamodel_src_path, datamodel_dest_path)
             # print(f"Copied datamodel.py from current directory: {datamodel_src_path}")
        else:
            print(f"Error: datamodel.py not found in strategy dir, project root, or current dir. Cannot run.")
            raise optuna.TrialPruned("datamodel.py not found")
    except Exception as e:
        print(f"Error copying datamodel.py: {e}"); raise optuna.TrialPruned()

    timestamp = int(time.time()); random_suffix = random.randint(1000,9999)
    temp_strategy_filename = f"temp_strategy_{trial_id}_{timestamp}_{random_suffix}.py"
    temp_log_filename = f"temp_backtest_{trial_id}_{timestamp}_{random_suffix}.log"
    temp_strategy_path = os.path.join(TEMP_DIR, temp_strategy_filename)
    temp_log_path = os.path.join(TEMP_DIR, temp_log_filename)

    # 3. Modify Strategy File
    if not modify_strategy_params(ORIGINAL_STRATEGY_PATH, temp_strategy_path, suggested_params):
        print(f"Trial {trial_id}: Failed to modify strategy file. Pruning.")
        if os.path.exists(temp_strategy_path):
            try: os.remove(temp_strategy_path)
            except OSError: pass
        raise optuna.TrialPruned()
    # Print only the Spread params being tested
    print(f"Trial {trial_id}: Testing Params -> {target_product}: {suggested_params.get(target_product, 'N/A')}") # Shows params for target product

    # 4. Run Backtester (Command should be correct for R2)
    temp_strategy_path_safe = temp_strategy_path.replace("\\", "/")
    
    backtester_command_parts = []
    if BACKTESTER_SCRIPT_PATH.endswith(".py"):
        backtester_command_parts = ["python", BACKTESTER_SCRIPT_PATH]
    else: # Assume it's a module or direct command
        backtester_command_parts = ["python", "-m", BACKTESTER_SCRIPT_PATH]

    
    # Include data path in command if available
    backtest_command = backtester_command_parts + [temp_strategy_path_safe, "2", "--match-trades", "worse", "--no-out"]
    

    print(f"\nTrial {trial_id}: Running backtest -> {' '.join(backtest_command)}")

    # Execution and PnL Parsing Logic (remains the same)
    final_pnl = -float('inf'); backtest_failed = False
    try:
        # Increased timeout slightly
        result = subprocess.run(backtest_command, capture_output=True, text=True, check=False, timeout=420, encoding='utf-8', errors='replace')
        if result.returncode != 0:
             backtest_failed = True; print(f"Trial {trial_id}: Backtester failed ({result.returncode}).")
             try:
                 # Write more context on failure to the correct log path
                 log_content = f"Return Code: {result.returncode}\n\nSTDOUT:\n{result.stdout or 'N/A'}\n\nSTDERR:\n{result.stderr or 'N/A'}"
                 with open(temp_log_path, 'w', encoding='utf-8') as log_f: log_f.write(log_content)
                 print(f"  -> Wrote failure details to {temp_log_path}")
             except Exception as log_e: print(f"  -> Error writing failure log: {log_e}")

        stdout_lines = result.stdout.splitlines(); in_summary_section = False; found_pnl = False
        for line in stdout_lines:
            line_strip = line.strip()
            if "Profit summary:" in line_strip: in_summary_section = True; continue
            if in_summary_section and line_strip.startswith("Total profit:"):
                try:
                    match = re.search(r':\s*([-+]?\d*\.?\d+)', line_strip)
                    if match: final_pnl = float(match.group(1)); print(f"Trial {trial_id}: Parsed PnL from stdout: {final_pnl:.2f}"); found_pnl = True; break
                    else: print(f"Warning: Could not parse PnL: {line_strip}"); break
                except ValueError: print(f"Warning: Could not convert PnL: {line_strip}"); break
        if not found_pnl and not backtest_failed:
             print(f"Warning: Total profit not found in stdout for successful Trial {trial_id}.")
             # Optionally try parsing from log file as a fallback if stdout fails
             # final_pnl = parse_backtest_log_for_pnl(temp_log_path) # Use correct log path if uncommented

    except subprocess.TimeoutExpired:
        print(f"Trial {trial_id}: Backtester timed out.")
        backtest_failed = True
        try:
            # Write timeout to the correct log path
            with open(temp_log_path, 'w', encoding='utf-8') as log_f:
                log_f.write("Timeout.")
            print(f"  -> Wrote timeout log: {temp_log_path}")
        except Exception as log_e:
            print(f"  -> Error writing timeout log: {log_e}")
    except Exception as e:
        print(f"Trial {trial_id}: Error during backtest execution: {e}")
        import traceback
        traceback.print_exc()
        backtest_failed = True
        try:
            # Write error details to the correct log path
            with open(temp_log_path, 'w', encoding='utf-8') as log_f:
                log_f.write(f"Error during subprocess run:\n{traceback.format_exc()}")
            print(f"  -> Wrote execution error log: {temp_log_path}")
        except Exception as log_e:
            print(f"  -> Error writing execution error log: {log_e}")
    finally:
        # Determine if the strategy file should be kept (e.g., first few or failures)
        keep_strategy_file = (trial_id < 5 or backtest_failed or final_pnl == -float('inf'))
        if keep_strategy_file and os.path.exists(temp_strategy_path):
            print(f"Trial {trial_id}: Keeping strategy file: {temp_strategy_filename}")
        elif os.path.exists(temp_strategy_path):
            try: os.remove(temp_strategy_path)
            except OSError as e: print(f"Error removing temp strategy file {temp_strategy_filename}: {e}")

        # Keep log file ONLY if backtest failed or PnL was not found, otherwise remove
        if os.path.exists(temp_log_path):
             if backtest_failed or final_pnl == -float('inf'):
                 print(f"Trial {trial_id}: Keeping log file due to failure/no PnL: {temp_log_filename}")
             else:
                 try: os.remove(temp_log_path)
                 except OSError as e: print(f"Error removing log file {temp_log_filename}: {e}")


    if final_pnl == -float('inf'):
        print(f"Trial {trial_id}: Could not determine PnL. Pruning.")
        raise optuna.TrialPruned()

    print(f"Trial {trial_id} completed with PnL: {final_pnl:.2f}")
    return final_pnl

# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Optimize strategy parameters for a specific product.') # Generalized description
    parser.add_argument('--product', # Changed from --basket 
                        type=str, 
                        required=True, 
                        choices=[Product.PICNIC_BASKET1, Product.PICNIC_BASKET2, Product.SQUID_INK], # Added SQUID_INK
                        help='The product to optimize (e.g., PICNIC_BASKET1, SQUID_INK)')
    args = parser.parse_args()
    target_product = args.product # Changed variable name
    # --- End Argument Parsing ---

    # --- Dynamic Study Name --- 
    # (Using product name directly in study name)
    STUDY_NAME = f"r2_opt_{target_product}"
    print(f"Target Product: {target_product}")
    print(f"Study Name: {STUDY_NAME}")
    # --- End Dynamic Study Name ---

    print(f"Starting Optuna optimization for {target_product} in {ORIGINAL_STRATEGY_PATH} (evaluated on R2 data)...")
    storage_name = f"sqlite:///{STUDY_NAME}.db"
    # Adjust pruner settings if needed
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(study_name=STUDY_NAME, storage=storage_name, direction="maximize", load_if_exists=True, pruner=pruner)

    try:
        # Create partial function for objective with target_product pre-filled
        objective_partial = functools.partial(objective, target_product=target_product)

        study.optimize(objective_partial, n_trials=N_TRIALS, timeout=None, n_jobs=1) # n_jobs=1 for sequential runs
    except KeyboardInterrupt:
        print("Optimization stopped manually.")
    except Exception as e:
        print(f"An error occurred during optimization: {e}"); import traceback; traceback.print_exc()

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
            print(f"    Params (Optimized for {target_product}): ") # Dynamic description

            # Extract the best parameters for the target product
            best_params = {
                target_product: {}
            }

            # Define param maps separately for clarity
            param_map_b1 = {
                 "b1_spread_std_window": "spread_std_window",
                 "b1_zscore_threshold": "zscore_threshold",
                 "b1_target_position": "target_position",
                 "b1_close_threshold": "close_threshold",
            }
            param_map_b2 = {
                 "b2_spread_std_window": "spread_std_window",
                 "b2_zscore_threshold": "zscore_threshold",
                 "b2_target_position": "target_position",
                 "b2_close_threshold": "close_threshold",
            }
            param_map_squid = {
                "rsi_window": "rsi_window",
                "rsi_overbought": "rsi_overbought",
                "rsi_oversold": "rsi_oversold",
            }

            # Select the correct map based on target_product
            if target_product == Product.PICNIC_BASKET1:
                current_param_map = param_map_b1
                optuna_prefix = "b1_"
            elif target_product == Product.PICNIC_BASKET2:
                current_param_map = param_map_b2
                optuna_prefix = "b2_"
            elif target_product == Product.SQUID_INK:
                current_param_map = param_map_squid
                optuna_prefix = ""
            else:
                current_param_map = {}
                optuna_prefix = ""
            
            for optuna_key_base, param_key in current_param_map.items():
                # Construct the actual key used in trial.params (might have prefix)
                optuna_name = optuna_prefix + optuna_key_base # Reconstruct optuna key if prefixes were used
                # Or use the map keys directly if no prefix was added during suggestion
                # e.g., if target_product == SQUID_INK, use optuna_key_base directly
                if target_product == Product.SQUID_INK:
                    optuna_name = optuna_key_base # No prefix for squid ink params

                if optuna_name in trial.params:
                     best_params[target_product][param_key] = trial.params[optuna_name]
            
            print(f"      {target_product}: {best_params[target_product]}")

            # Generate the best PARAMS dictionary string
            print(f"\n--- Best PARAMS dictionary for {os.path.basename(ORIGINAL_STRATEGY_PATH)} (with {target_product} updated) ---")
            # Use BASE_PARAMS as the starting point
            best_params_dict = copy.deepcopy(BASE_PARAMS)

            # Update ONLY the target product parts with the best trial results
            if target_product in best_params_dict and best_params.get(target_product):
                best_params_dict[target_product].update(best_params[target_product])
            else:
                 print(f"Warning: Could not update best params for {target_product}")

            # Convert the *full* best dict to string using the improved formatting logic
            params_for_json_best = {}
            for key, value in best_params_dict.items():
                 # Reuse the key formatting logic from modify_strategy_params
                 if key == Product.RAINFOREST_RESIN: key_str = 'Product.RAINFOREST_RESIN'
                 elif key == Product.SQUID_INK: key_str = 'Product.SQUID_INK'
                 elif key == Product.KELP: key_str = 'Product.KELP'
                 elif key == Product.DJEMBES: key_str = 'Product.DJEMBES'
                 elif key == Product.CROISSANTS: key_str = 'Product.CROISSANTS '
                 elif key == Product.JAMS: key_str = 'Product.JAMS'
                 elif key == Product.PICNIC_BASKET1: key_str = 'Product.PICNIC_BASKET1'
                 # Keep B2 in the output PARAMS string, just using its BASE_PARAMS values
                 elif key == Product.PICNIC_BASKET2: key_str = 'Product.PICNIC_BASKET2' 
                 else:
                      if isinstance(key, str) and key.startswith("Product."): key_str = key
                      else: key_str = repr(key)
                 params_for_json_best[key_str] = value

            # Manually format the output string like before
            formatted_params_lines_best = ["PARAMS = {\n"]
            param_items_best = list(params_for_json_best.items())
            for i, (key_str, value_dict) in enumerate(param_items_best):
                # Ensure value_dict is actually a dictionary before iterating
                if not isinstance(value_dict, dict):
                    print(f"Warning: Expected dict for key {key_str}, got {type(value_dict)}. Skipping.")
                    continue
                formatted_params_lines_best.append(f"    {key_str}: {{\n")
                value_items_best = list(value_dict.items())
                for j, (param_name, param_value) in enumerate(value_items_best):
                    try: value_repr = json.dumps(param_value)
                    except TypeError: value_repr = repr(param_value) # Fallback
                    formatted_params_lines_best.append(f'        "{param_name}": {value_repr}')
                    if j < len(value_items_best) - 1: formatted_params_lines_best.append(",\n")
                    else: formatted_params_lines_best.append("\n")
                formatted_params_lines_best.append("    }")
                if i < len(param_items_best) - 1: formatted_params_lines_best.append(",\n")
                else: formatted_params_lines_best.append("\n")
            formatted_params_lines_best.append("}\n")

            print("".join(formatted_params_lines_best)) # Print the formatted string
            print("--- End Best PARAMS ---")

    except ValueError as e:
         print(f"  Best trial not available (perhaps all trials failed or were pruned): {e}")
    except Exception as e:
         print(f"An error occurred displaying results: {e}")
         import traceback
         traceback.print_exc()



  