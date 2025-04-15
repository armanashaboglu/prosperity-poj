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
# <<< Strategy being optimized (Update to the file with global ARB_PARAMS) >>>
ORIGINAL_STRATEGY_PATH = "strategy/round2/round2_original.py" # <<< CORRECTED Path
# <<< Temp dir reflects R2 strategy + evaluation data >>>
TEMP_DIR = "temp_optimizer_original_r2" # Updated temp dir name
# <<< ADDED back Custom data path for R2 testing >>>
# CUSTOM_DATA_PATH = 'C:/Users/Admin/projects/prosperity-poj/strategy/round2/resources'
# --- !! END OF IMPORTANT PATHS !! ---

# Optuna Settings
N_TRIALS = 200 # Adjusted trials
# STUDY_NAME set dynamically based on target product


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
    DJEMBES = "DJEMBES"
    BASKET1 = "PICNIC_BASKET1" # <<< Corrected name
    BASKET2 = "PICNIC_BASKET2" # <<< Corrected name
    # Remove B1B2_DEVIATION if not present in round2_original.py
    # B1B2_DEVIATION = "B1B2_DEVIATION"

# --- Load BASE_PARAMS and BASE_ARB_PARAMS from the original strategy file ---
BASE_PARAMS = {}
BASE_ARB_PARAMS = {}
# Remove BASE_ZSCORE_ARB_PARAMS if not present in round2_original.py
# BASE_ZSCORE_ARB_PARAMS = {}

try:
    # Use the updated ORIGINAL_STRATEGY_PATH
    with open(ORIGINAL_STRATEGY_PATH, 'r') as f_base:
        content = f_base.read()

        # Extract PARAMS (for SQUID_INK RSI strategy)
        match_params = re.search(r"^PARAMS\s*=\s*({.*?^})", content, re.DOTALL | re.MULTILINE)
        if match_params:
            params_str = match_params.group(1)
            exec_globals = {"Product": Product}
            exec_locals = {}
            exec(f"BASE_PARAMS = {params_str}", exec_globals, exec_locals)
            BASE_PARAMS = exec_locals['BASE_PARAMS']
            print(f"Successfully loaded BASE_PARAMS from {ORIGINAL_STRATEGY_PATH}.")
            # Basic validation
            if Product.SQUID_INK not in BASE_PARAMS:
                 print("Warning: BASE_PARAMS missing SQUID_INK key.")
            if Product.RAINFOREST_RESIN not in BASE_PARAMS:
                 print("Warning: BASE_PARAMS missing RAINFOREST_RESIN key.")
        else:
            print(f"WARNING: Could not automatically extract BASE_PARAMS from {ORIGINAL_STRATEGY_PATH}.")
            BASE_PARAMS = { # Fallback based on round2_original.py
                Product.SQUID_INK: { "rsi_window": 96, "rsi_overbought": 56, "rsi_oversold": 39 },
                Product.RAINFOREST_RESIN: {"fair_value": 10000},
            }

        # Extract ARB_PARAMS
        match_arb = re.search(r"^ARB_PARAMS\s*=\s*({.*?^})", content, re.DOTALL | re.MULTILINE)
        if match_arb:
            arb_params_str = match_arb.group(1)
            exec_locals_arb = {}
            exec(f"BASE_ARB_PARAMS = {arb_params_str}", {}, exec_locals_arb)
            BASE_ARB_PARAMS = exec_locals_arb['BASE_ARB_PARAMS']
            print(f"Successfully loaded BASE_ARB_PARAMS from {ORIGINAL_STRATEGY_PATH}.")
        else:
            print(f"WARNING: Could not automatically extract ARB_PARAMS from {ORIGINAL_STRATEGY_PATH}. Using fallback.")
            BASE_ARB_PARAMS = { # Fallback based on round2_original.py
                "diff_threshold_b1": 200, "diff_threshold_b2": 120,
                "diff_threshold_b1_b2": 60, "max_arb_lot": 5,
                "close_threshold_b1_b2": 18.5
            }

        # Remove ZSCORE_ARB_PARAMS extraction if not needed for round2_original.py
        # match_zscore = re.search(r"^ZSCORE_ARB_PARAMS\s*=\s*({.*?^})", ...)

except Exception as e:
    print(f"ERROR loading parameters from {ORIGINAL_STRATEGY_PATH}: {e}. Using fallback defaults.")
    # Fallback defaults
    BASE_PARAMS = {
        Product.SQUID_INK: { "rsi_window": 96, "rsi_overbought": 56, "rsi_oversold": 39 },
        Product.RAINFOREST_RESIN: {"fair_value": 10000},
    }
    BASE_ARB_PARAMS = {
        "diff_threshold_b1": 200, "diff_threshold_b2": 120,
        "diff_threshold_b1_b2": 60, "max_arb_lot": 5,
        "close_threshold_b1_b2": 18.5
    }

def modify_strategy_params(original_file_path, temp_file_path, new_params, new_arb_params): # Removed new_zscore_arb_params
    """Reads the original strategy, replaces PARAMS and ARB_PARAMS dicts, writes to temp file."""
    try:
        with open(original_file_path, 'r') as f:
            lines = f.readlines()

        # --- Prepare PARAMS string ---
        if new_params:
            params_for_json = {}
            for key, value in new_params.items():
                # Use Product.<NAME> representation for keys
                key_str = None
                if key == Product.RAINFOREST_RESIN or key == "RAINFOREST_RESIN": key_str = 'Product.RAINFOREST_RESIN'
                elif key == Product.SQUID_INK or key == "SQUID_INK": key_str = 'Product.SQUID_INK'
                elif key == Product.KELP or key == "KELP": key_str = 'Product.KELP' # Add others if PARAMS includes them
                elif isinstance(key, str) and key.startswith("Product."): key_str = key

                if key_str: params_for_json[key_str] = value

            formatted_params_lines = ["PARAMS = {\n"]
            param_items = list(params_for_json.items())
            for i, (key_str, value_dict) in enumerate(param_items):
                if not isinstance(value_dict, dict): continue
                formatted_params_lines.append(f"    {key_str}: {{\n")
                value_items = list(value_dict.items())
                for j, (param_name, param_value) in enumerate(value_items):
                    value_repr = json.dumps(param_value)
                    formatted_params_lines.append(f'        "{param_name}": {value_repr}')
                    if j < len(value_items) - 1: formatted_params_lines.append(",\n")
                    else: formatted_params_lines.append("\n")
                formatted_params_lines.append("    }")
                if i < len(param_items) - 1: formatted_params_lines.append(",\n")
                else: formatted_params_lines.append("\n")
            formatted_params_lines.append("}\n")
            new_params_lines = formatted_params_lines
        else:
            new_params_lines = []

        # --- Prepare ARB_PARAMS string ---
        if new_arb_params:
            formatted_arb_lines = ["ARB_PARAMS = {\n"]
            arb_items = list(new_arb_params.items())
            for i, (key, value) in enumerate(arb_items):
                key_repr = f'"{key}"'
                value_repr = json.dumps(value)
                formatted_arb_lines.append(f"    {key_repr}: {value_repr}")
                if i < len(arb_items) - 1: formatted_arb_lines.append(",\n")
                else: formatted_arb_lines.append("\n")
            formatted_arb_lines.append("}\n")
            new_arb_lines = formatted_arb_lines
        else:
            new_arb_lines = []

        # --- Replacement Logic ---
        output_lines = []
        in_params_dict = False; params_replaced = (not new_params_lines)
        in_arb_params_dict = False; arb_params_replaced = (not new_arb_lines)
        brace_count = 0

        for line in lines:
            stripped_line = line.strip()

            # Handle PARAMS replacement
            if stripped_line.startswith("PARAMS") and "=" in stripped_line and "{" in stripped_line and not in_params_dict and not params_replaced:
                indent = line[:line.find("PARAMS")]
                indented_new_params_lines = [f"{indent}{l}" for l in new_params_lines]
                output_lines.extend(indented_new_params_lines)
                in_params_dict = True
                brace_count = line.count('{') - line.count('}')
                if brace_count <= 0 and stripped_line.endswith("}"): in_params_dict = False
                params_replaced = True
                continue
            if in_params_dict:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0: in_params_dict = False
                continue

            # Handle ARB_PARAMS replacement
            if stripped_line.startswith("ARB_PARAMS") and "=" in stripped_line and "{" in stripped_line and not in_arb_params_dict and not arb_params_replaced:
                indent = line[:line.find("ARB_PARAMS")]
                indented_new_arb_lines = [f"{indent}{l}" for l in new_arb_lines]
                output_lines.extend(indented_new_arb_lines)
                in_arb_params_dict = True
                brace_count = line.count('{') - line.count('}')
                if brace_count <= 0 and stripped_line.endswith("}"): in_arb_params_dict = False
                arb_params_replaced = True
                continue
            if in_arb_params_dict:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0: in_arb_params_dict = False
                continue

            # Append line if not part of a replaced dict
            output_lines.append(line)

        if not params_replaced and new_params_lines: raise ValueError("Could not find 'PARAMS = {' line to replace.")
        if not arb_params_replaced and new_arb_lines: raise ValueError("Could not find 'ARB_PARAMS = {' line to replace.")

        with open(temp_file_path, 'w') as f: f.writelines(output_lines)
        return True

    except Exception as e:
        print(f"Error modifying strategy parameters: {e}"); import traceback; traceback.print_exc(); return False


# --- Optuna Objective Function ---

def objective(trial: optuna.Trial, target_product: str):
    """Function for Optuna to optimize. Now supports ARBITRAGE target."""

    trial_id = trial.number
    # Start with base params
    suggested_params = copy.deepcopy(BASE_PARAMS)
    suggested_arb_params = copy.deepcopy(BASE_ARB_PARAMS)
    # Remove zscore params if not relevant
    # suggested_zscore_arb_params = copy.deepcopy(BASE_ZSCORE_ARB_PARAMS)

    # --- Suggest Parameters based on target_product ---
    if target_product == Product.SQUID_INK:
        # --- Suggest RSI Params for SQUID_INK ---
        rsi_window = trial.suggest_int("rsi_window", 10, 200)
        rsi_oversold = trial.suggest_int("rsi_oversold", 1, 49)
        rsi_overbought = trial.suggest_int("rsi_overbought", rsi_oversold + 1, 99)

        if Product.SQUID_INK in suggested_params:
            suggested_params[Product.SQUID_INK].update({
                "rsi_window": rsi_window,
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold
            })
        else:
             print(f"Warning: {Product.SQUID_INK} not in BASE_PARAMS, cannot update.")
             suggested_params = None # Don't modify PARAMS if SQUID_INK is missing

        # When optimizing SQUID_INK, don't modify ARB params
        suggested_arb_params = None

    elif target_product == "ARBITRAGE":
        # --- Suggest Arbitrage Params ---
        # Diff thresholds
        diff_b1 = trial.suggest_int("diff_threshold_b1", 50, 300, step=5)
        diff_b2 = trial.suggest_int("diff_threshold_b2", 30, 200, step=5)
        diff_b1_b2_entry = trial.suggest_int("diff_threshold_b1_b2", 10, 250, step=5)
        # Close threshold for B1 vs B2 - suggest around 0, allowing slight positive/negative mean reversion targets
        diff_b1_b2_close = trial.suggest_float("close_threshold_b1_b2", -20.0, 50.0, step=0.5)
        # Max lot size
        max_lot = trial.suggest_int("max_arb_lot", 1, 15)

        # Update the separate arb params dictionary
        suggested_arb_params.update({
            "diff_threshold_b1": diff_b1,
            "diff_threshold_b2": diff_b2,
            "diff_threshold_b1_b2": diff_b1_b2_entry, # Entry threshold
            "close_threshold_b1_b2": diff_b1_b2_close, # New close threshold
            "max_arb_lot": max_lot
        })
        # Set suggested_params (for RSI etc.) to None for this case
        suggested_params = None

    else:
        # Check if it's a known Product enum value we just don't optimize for this run
        known_unoptimized = [Product.KELP, Product.RAINFOREST_RESIN, Product.CROISSANTS, Product.JAMS, Product.DJEMBES, Product.BASKET1, Product.BASKET2]
        if target_product in known_unoptimized:
            print(f"Warning: {target_product} is known but not the target for this optimization run. Running with base params.")
            # Use all base params when not optimizing a specific part
            suggested_params = BASE_PARAMS
            suggested_arb_params = BASE_ARB_PARAMS
        else:
            # If it's not SQUID_INK or ARBITRAGE, raise error
            raise ValueError(f"Invalid target_product specified for optimization: {target_product}")

    # --- Input validation (Optional but Recommended) ---
    if target_product == Product.SQUID_INK and suggested_params: # Check suggested_params exists
        sq_params = suggested_params.get(Product.SQUID_INK, {})
        if sq_params.get("rsi_window", 0) < 2:
             print(f"Trial {trial_id}: Invalid SQ window {sq_params.get('rsi_window')} < 2. Pruning.")
             raise optuna.TrialPruned()
        if not (0 < sq_params.get("rsi_oversold", -1) < sq_params.get("rsi_overbought", -1) < 100):
             print(f"Trial {trial_id}: Invalid SQ thresholds OB={sq_params.get('rsi_overbought')} OS={sq_params.get('rsi_oversold')}. Pruning.")
             raise optuna.TrialPruned()

    elif target_product == "ARBITRAGE":
        arb_p = suggested_arb_params
        if not (arb_p["diff_threshold_b1"] > 0 and arb_p["diff_threshold_b2"] > 0 and arb_p["diff_threshold_b1_b2"] > 0):
            print(f"Trial {trial_id}: Invalid Arb Thresholds <= 0. Pruning.")
            raise optuna.TrialPruned()
        if arb_p["max_arb_lot"] <= 0:
            print(f"Trial {trial_id}: Invalid max_arb_lot <= 0. Pruning.")
            raise optuna.TrialPruned()

    # --- End Input Validation ---

    # 2. Prepare for Backtest
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
    datamodel_src_path = "datamodel.py"; datamodel_dest_path = os.path.join(TEMP_DIR, "datamodel.py")
    try:
        strategy_dir = os.path.dirname(ORIGINAL_STRATEGY_PATH)
        potential_dm_path_strat = os.path.join(strategy_dir, "datamodel.py")
        project_root = os.path.abspath(os.path.join(strategy_dir, '..', '..')) # Adjust levels if needed
        potential_dm_path_root = os.path.join(project_root, "datamodel.py")

        if os.path.exists(potential_dm_path_strat):
            shutil.copy2(potential_dm_path_strat, datamodel_dest_path)
        elif os.path.exists(potential_dm_path_root):
            shutil.copy2(potential_dm_path_root, datamodel_dest_path)
        elif os.path.exists(datamodel_src_path): # Check current dir as last resort
             shutil.copy2(datamodel_src_path, datamodel_dest_path)
        else:
            print(f"Error: datamodel.py not found. Cannot run.")
            raise optuna.TrialPruned("datamodel.py not found")
    except Exception as e:
        print(f"Error copying datamodel.py: {e}"); raise optuna.TrialPruned()

    timestamp = int(time.time()); random_suffix = random.randint(1000,9999)
    temp_strategy_filename = f"temp_strategy_{trial_id}_{timestamp}_{random_suffix}.py"
    temp_log_filename = f"temp_backtest_{trial_id}_{timestamp}_{random_suffix}.log"
    temp_strategy_path = os.path.join(TEMP_DIR, temp_strategy_filename)
    temp_log_path = os.path.join(TEMP_DIR, temp_log_filename)

    # 3. Modify Strategy File - Pass only relevant params
    if not modify_strategy_params(ORIGINAL_STRATEGY_PATH, temp_strategy_path, suggested_params, suggested_arb_params):
        print(f"Trial {trial_id}: Failed to modify strategy file. Pruning.")
        if os.path.exists(temp_strategy_path):
            try: os.remove(temp_strategy_path)
            except OSError: pass
        raise optuna.TrialPruned()

    # Print only the params being tested
    if target_product == "ARBITRAGE":
        print(f"\nTrial {trial_id}: Testing Params -> ARBITRAGE: {suggested_arb_params}")
    elif target_product == Product.SQUID_INK and suggested_params:
        print(f"\nTrial {trial_id}: Testing Params -> SQUID_INK: {suggested_params.get(Product.SQUID_INK, 'N/A')}")
    else:
        print(f"\nTrial {trial_id}: Testing Params -> {target_product}: (Using Base Params)")


    # 4. Run Backtester
    temp_strategy_path_safe = temp_strategy_path.replace("\\", "/")
    backtester_command_parts = ["python", "-m", BACKTESTER_SCRIPT_PATH] if not BACKTESTER_SCRIPT_PATH.endswith(".py") else ["python", BACKTESTER_SCRIPT_PATH]
    backtest_command = backtester_command_parts + [temp_strategy_path_safe, "2", "--match-trades", "worse", "--no-out"]
    # Add --data path if needed
    # if 'CUSTOM_DATA_PATH' in globals() and CUSTOM_DATA_PATH and os.path.isdir(CUSTOM_DATA_PATH): ...

    print(f"Trial {trial_id}: Running backtest -> {' '.join(backtest_command)}")

    # 5. Execution and PnL Parsing Logic
    final_pnl = -float('inf'); backtest_failed = False
    try:
        result = subprocess.run(backtest_command, capture_output=True, text=True, check=False, timeout=420, encoding='utf-8', errors='replace')
        if result.returncode != 0:
             backtest_failed = True; print(f"Trial {trial_id}: Backtester failed ({result.returncode}).")
             # Log failure details
             try:
                 log_content = f"Return Code: {result.returncode}\n\nSTDOUT:\n{result.stdout or 'N/A'}\n\nSTDERR:\n{result.stderr or 'N/A'}"
                 with open(temp_log_path, 'w', encoding='utf-8') as log_f: log_f.write(log_content)
                 print(f"  -> Wrote failure details to {temp_log_path}")
             except Exception as log_e: print(f"  -> Error writing failure log: {log_e}")

        # Parse PnL from stdout
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
             # Optionally try parsing from log file as a fallback
             # final_pnl = parse_backtest_log_for_pnl(temp_log_path)

    except subprocess.TimeoutExpired:
        print(f"Trial {trial_id}: Backtester timed out.")
        backtest_failed = True
        # Log timeout
        try:
            with open(temp_log_path, 'w', encoding='utf-8') as log_f: log_f.write("Timeout.")
            print(f"  -> Wrote timeout log: {temp_log_path}")
        except Exception as log_e: print(f"  -> Error writing timeout log: {log_e}")
    except Exception as e:
        print(f"Trial {trial_id}: Error during backtest execution: {e}")
        backtest_failed = True
        # Log error
        try:
            with open(temp_log_path, 'w', encoding='utf-8') as log_f: log_f.write(f"Error during subprocess run:\n{traceback.format_exc()}")
            print(f"  -> Wrote execution error log: {temp_log_path}")
        except Exception as log_e: print(f"  -> Error writing execution error log: {log_e}")
    finally:
        # Cleanup temp files
        keep_strategy_file = (trial_id < 5 or backtest_failed or final_pnl == -float('inf'))
        if keep_strategy_file and os.path.exists(temp_strategy_path):
            print(f"Trial {trial_id}: Keeping strategy file: {temp_strategy_filename}")
        elif os.path.exists(temp_strategy_path):
            try: os.remove(temp_strategy_path)
            except OSError as e: print(f"Error removing temp strategy file {temp_strategy_filename}: {e}")

        if os.path.exists(temp_log_path):
             if backtest_failed or final_pnl == -float('inf'):
                 print(f"Trial {trial_id}: Keeping log file: {temp_log_filename}")
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
    parser = argparse.ArgumentParser(description='Optimize strategy parameters for a specific product/strategy.')
    parser.add_argument('--product',
                        type=str,
                        required=True,
                        # <<< ADDED "ARBITRAGE" to choices >>>
                        choices=[Product.SQUID_INK, "ARBITRAGE"], # Only allow optimizing SQUID or ARB
                        help='The product/strategy to optimize (e.g., SQUID_INK, ARBITRAGE)')
    args = parser.parse_args()
    target_product = args.product
    # --- End Argument Parsing ---

    # --- Dynamic Study Name ---
    study_suffix = target_product
    STUDY_NAME = f"r2_original_opt_{study_suffix}" # Reflects the script being optimized
    print(f"Target Product/Strategy: {target_product}")
    print(f"Study Name: {STUDY_NAME}")
    # --- End Dynamic Study Name ---

    print(f"Starting Optuna optimization for {target_product} in {ORIGINAL_STRATEGY_PATH} (evaluated on R2 data)...")
    storage_name = f"sqlite:///{STUDY_NAME}.db"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(study_name=STUDY_NAME, storage=storage_name, direction="maximize", load_if_exists=True, pruner=pruner)

    try:
        objective_partial = functools.partial(objective, target_product=target_product)
        study.optimize(objective_partial, n_trials=N_TRIALS, timeout=None, n_jobs=1)
    except KeyboardInterrupt:
        print("Optimization stopped manually.")
    except Exception as e:
        print(f"An error occurred during optimization: {e}"); import traceback; traceback.print_exc()

    # --- Results Display ---
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
            print(f"    Params (Optimized for {target_product}): ")

            # Extract the best parameters for the target product/strategy
            best_params_target_only = {}
            best_full_params = copy.deepcopy(BASE_PARAMS) # For individual strategies like SQUID
            best_full_arb_params = copy.deepcopy(BASE_ARB_PARAMS) # For arbitrage

            # Define param maps
            param_map_squid = {
                "rsi_window": "rsi_window",
                "rsi_overbought": "rsi_overbought",
                "rsi_oversold": "rsi_oversold",
            }
            param_map_arbitrage = { # Include the new close threshold
                "diff_threshold_b1": "diff_threshold_b1",
                "diff_threshold_b2": "diff_threshold_b2",
                "diff_threshold_b1_b2": "diff_threshold_b1_b2", # Entry threshold
                "close_threshold_b1_b2": "close_threshold_b1_b2", # Close threshold
                "max_arb_lot": "max_arb_lot",
            }

            # Select the correct map and target dictionary
            if target_product == Product.SQUID_INK:
                current_param_map = param_map_squid
                target_dict_to_update = best_full_params.get(Product.SQUID_INK, {})
                dict_to_print = best_full_params # Print full PARAMS dict
                dict_name = "PARAMS"
            elif target_product == "ARBITRAGE":
                current_param_map = param_map_arbitrage
                target_dict_to_update = best_full_arb_params # Update the arb dict directly
                dict_to_print = best_full_arb_params # Print full ARB_PARAMS dict
                dict_name = "ARB_PARAMS"
            else: # Should not happen due to argparse choices
                current_param_map = {}
                target_dict_to_update = {}
                dict_to_print = {}
                dict_name = "UNKNOWN"

            # Populate best_params_target_only using the map
            for optuna_key, param_key in current_param_map.items():
                if optuna_key in trial.params:
                     best_params_target_only[param_key] = trial.params[optuna_key]

            print(f"      {target_product}: {best_params_target_only}")

            # Update the *correct* full dictionary with the best target params
            if target_product == "ARBITRAGE":
                best_full_arb_params.update(best_params_target_only)
                # Update dict_to_print for final output
                dict_to_print = best_full_arb_params
            elif target_product == Product.SQUID_INK:
                if Product.SQUID_INK in best_full_params and isinstance(best_full_params[Product.SQUID_INK], dict):
                     best_full_params[Product.SQUID_INK].update(best_params_target_only)
                     # Update dict_to_print for final output
                     dict_to_print = best_full_params
                else:
                     print(f"Warning: Cannot update best params for {target_product}, base structure incorrect.")

            # Generate the best dictionary string
            print(f"\n--- Best {dict_name} dictionary for {os.path.basename(ORIGINAL_STRATEGY_PATH)} --- ")

            formatted_lines_best = [f"{dict_name} = {{\n"]
            if dict_name == "PARAMS":
                 params_for_json_best = {}
                 for key, value in dict_to_print.items():
                     # Reuse key formatting
                     key_str = repr(key) # Fallback
                     if key == Product.RAINFOREST_RESIN: key_str = 'Product.RAINFOREST_RESIN'
                     elif key == Product.SQUID_INK: key_str = 'Product.SQUID_INK'
                     elif key == Product.KELP: key_str = 'Product.KELP' # Add if needed
                     elif isinstance(key, str) and key.startswith("Product."): key_str = key

                     params_for_json_best[key_str] = value

                 param_items_best = list(params_for_json_best.items())
                 for i, (key_str, value_dict) in enumerate(param_items_best):
                    if not isinstance(value_dict, dict): continue
                    formatted_lines_best.append(f"    {key_str}: {{\n")
                    value_items_best = list(value_dict.items())
                    for j, (param_name, param_value) in enumerate(value_items_best):
                        try: value_repr = json.dumps(param_value)
                        except TypeError: value_repr = repr(param_value)
                        formatted_lines_best.append(f'        "{param_name}": {value_repr}')
                        if j < len(value_items_best) - 1: formatted_lines_best.append(",\n")
                        else: formatted_lines_best.append("\n")
                    formatted_lines_best.append("    }")
                    if i < len(param_items_best) - 1: formatted_lines_best.append(",\n")
                    else: formatted_lines_best.append("\n")

            elif dict_name == "ARB_PARAMS":
                 arb_items_best = list(dict_to_print.items())
                 for i, (key, value) in enumerate(arb_items_best):
                    key_repr = f'"{key}"'
                    value_repr = json.dumps(value)
                    formatted_lines_best.append(f"    {key_repr}: {value_repr}")
                    if i < len(arb_items_best) - 1: formatted_lines_best.append(",\n")
                    else: formatted_lines_best.append("\n")

            formatted_lines_best.append("}\n")
            print("".join(formatted_lines_best))
            print(f"--- End Best {dict_name} --- ")

    except ValueError as e:
         print(f"  Best trial not available (perhaps all trials failed or were pruned): {e}")
    except Exception as e:
         print(f"An error occurred displaying results: {e}")
         import traceback
         traceback.print_exc()

# --- (End of File) ---



  