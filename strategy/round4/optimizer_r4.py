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
from datetime import datetime
import traceback # Make sure this is imported at the top

# --- Configuration ---

# --- !! IMPORTANT: SET THESE PATHS CORRECTLY !! ---
BACKTESTER_SCRIPT_PATH = "prosperity3bt" # Assumes it's runnable via 'python -m prosperity3bt'
# Default strategy path - will be overridden by command line argument if provided
DEFAULT_STRATEGY_PATH = "strategy/round3/round3_submission.py"
# Temp dir for optimization
TEMP_DIR = "temp_optimizer_r4"

# Optuna Settings
N_TRIALS = 125 # Adjusted trials
# STUDY_NAME set dynamically based on target product

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
    B1B2_DEVIATION = "B1B2_DEVIATION"  # Added this attribute
    # Round 3
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    # Add short form voucher names to match round3_volsmile.py
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

# --- Load BASE_PARAMS and BASE_ARB_PARAMS from the original strategy file ---
BASE_PARAMS = {}
BASE_ARB_PARAMS = {}
# --- Added for Voucher Slope Params ---
BASE_VOUCHER_SLOPE_PARAMS = {} 
# --- Added for Voucher 10500 Strategy Params ---
BASE_VOUCHER_10500_PARAMS = {}
# Remove BASE_ZSCORE_ARB_PARAMS if not present in round2_original.py
# BASE_ZSCORE_ARB_PARAMS = {}

try:
    # Use the updated ORIGINAL_STRATEGY_PATH
    with open(DEFAULT_STRATEGY_PATH, 'r', encoding='utf-8') as f_base:
        content = f_base.read()

        # Extract PARAMS
        match_params = re.search(r"^PARAMS\s*=\s*({.*?^})", content, re.DOTALL | re.MULTILINE)
        if match_params:
            params_str = match_params.group(1)
            # Define Product
            exec_globals = {"Product": Product} 
            exec_locals = {}
            try:
                exec(f"TEMP_PARAMS = {params_str}", exec_globals, exec_locals)
                loaded_params = exec_locals['TEMP_PARAMS']
                # Separate the loaded params
                BASE_PARAMS = {} # Reset base params
                
                for key, value in loaded_params.items():
                    if key in [Product.SQUID_INK, Product.VOLCANIC_ROCK]:
                        BASE_PARAMS[key] = value

                print(f"Successfully loaded BASE_PARAMS from {DEFAULT_STRATEGY_PATH}.")
                # Basic validation
                if not BASE_PARAMS: 
                    print("Warning: BASE_PARAMS seems empty after loading.")

            except Exception as exec_e:
                print(f"Error executing PARAMS string from file: {exec_e}")
                # Fallback required if exec fails
                raise ValueError("Failed to parse PARAMS from strategy file.")

        else:
            print(f"WARNING: Could not automatically extract PARAMS from {DEFAULT_STRATEGY_PATH}.")
            # Fallback based on previous known structure
            BASE_PARAMS = { 
                Product.SQUID_INK: { "rsi_window": 96, "rsi_overbought": 56, "rsi_oversold": 39 },
                Product.VOLCANIC_ROCK: { "rsi_window": 106, "rsi_overbought": 60, "rsi_oversold": 40 }
            }

        # Extract ARB_PARAMS (for B1B2_DEVIATION strategy)
        match_arb = re.search(r"^ARB_PARAMS\s*=\s*({.*?^})", content, re.DOTALL | re.MULTILINE)
        if match_arb:
            arb_params_str = match_arb.group(1)
            exec_locals_arb = {}
            exec(f"BASE_ARB_PARAMS = {arb_params_str}", {}, exec_locals_arb)
            BASE_ARB_PARAMS = exec_locals_arb['BASE_ARB_PARAMS']
            print(f"Successfully loaded BASE_ARB_PARAMS from {DEFAULT_STRATEGY_PATH}.")
        else:
            print(f"WARNING: Could not automatically extract ARB_PARAMS from {DEFAULT_STRATEGY_PATH}. Using fallback.")
            BASE_ARB_PARAMS = { # Fallback for B1B2_DEVIATION strategy
                "diff_threshold_b1": 200, "diff_threshold_b2": 120,
                "diff_threshold_b1_b2": 60, "max_arb_lot": 5,
                "close_threshold_b1_b2": 18.5,
                "zscore_threshold_entry_long": -2.0,
                "zscore_threshold_entry_short": 2.0,
                "zscore_threshold_exit": 0.5,
                "ewma_period_mean": 100,
                "ewma_period_std": 200
            }

except Exception as e:
    print(f"ERROR loading parameters from {DEFAULT_STRATEGY_PATH}: {e}. Using fallback defaults.")
    # Fallback defaults
    BASE_PARAMS = {
        Product.SQUID_INK: { "rsi_window": 96, "rsi_overbought": 56, "rsi_oversold": 39 },
        Product.VOLCANIC_ROCK: { "rsi_window": 106, "rsi_overbought": 60, "rsi_oversold": 40 }
    }
    BASE_ARB_PARAMS = { # For B1B2_DEVIATION strategy
        "diff_threshold_b1": 200, "diff_threshold_b2": 120,
        "diff_threshold_b1_b2": 60, "max_arb_lot": 5,
        "close_threshold_b1_b2": 18.5,
        "zscore_threshold_entry_long": -2.0,
        "zscore_threshold_entry_short": 2.0,
        "zscore_threshold_exit": 0.5,
        "ewma_period_mean": 100,
        "ewma_period_std": 200
    }

def get_new_params_lines(params):
    """Format the PARAMS dictionary into a list of lines."""
    formatted_params_lines = ["PARAMS = {\n"]
    param_items = list(params.items())
    for i, (key_str, value_dict) in enumerate(param_items):
        if not isinstance(value_dict, dict): continue # Skip if value isn't a dict
        
        # Handle keys like Product.SQUID_INK - make sure to use proper Product reference
        if key_str == Product.SQUID_INK:
            formatted_key = "Product.SQUID_INK"
        elif key_str == Product.VOLCANIC_ROCK:
            formatted_key = "Product.VOLCANIC_ROCK"
        elif key_str == Product.B1B2_DEVIATION:
            formatted_key = "Product.B1B2_DEVIATION"
        else:
            # Fall back to string representation for other keys
            formatted_key = repr(key_str)
        
        formatted_params_lines.append(f"    {formatted_key}: {{\n")
        value_items = list(value_dict.items())
        for j, (param_name, param_value) in enumerate(value_items):
            # Use repr() for Python syntax, not json.dumps()
            value_repr = repr(param_value)
            formatted_params_lines.append(f'        "{param_name}": {value_repr}')
            if j < len(value_items) - 1: 
                formatted_params_lines.append(",\n")
            else: 
                formatted_params_lines.append("\n")
        formatted_params_lines.append("    }")
        if i < len(param_items) - 1: 
            formatted_params_lines.append(",\n")  # Always add comma between dictionary entries
        else: 
            formatted_params_lines.append("\n")
    formatted_params_lines.append("}\n")
    return formatted_params_lines

def get_new_arb_params_lines(arb_params):
    """Format the ARB_PARAMS dictionary into a list of lines."""
    formatted_arb_lines = ["ARB_PARAMS = {\n"]
    arb_items = list(arb_params.items())
    for i, (key, value) in enumerate(arb_items):
        key_repr = f'"{key}"'
        value_repr = json.dumps(value)
        formatted_arb_lines.append(f"    {key_repr}: {value_repr}")
        if i < len(arb_items) - 1: 
            formatted_arb_lines.append(",\n")
        else: 
            formatted_arb_lines.append("\n")
    formatted_arb_lines.append("}\n")
    return formatted_arb_lines

def validate_python_syntax(file_path):
    """Validate that the generated file has correct Python syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        compile(content, file_path, 'exec')
        return True
    except SyntaxError as e:
        print(f"Syntax error in generated file {file_path}: {e}")
        print(f"Error on line {e.lineno}, column {e.offset}: {e.text}")
        return False

def modify_strategy_params(file_path, temp_file_path=None, new_params=None, new_arb_params=None):
    """Modify PARAMS and ARB_PARAMS dictionaries in the strategy file using regex replacement."""

    # Copy original to temp or create backup
    if temp_file_path:
        try:
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            shutil.copy2(file_path, temp_file_path)
            print(f"Copied original strategy to {temp_file_path}")
            working_file = temp_file_path
        except Exception as e:
            print(f"Error copying strategy file to temp location: {e}")
            return False # Indicate failure
    else:
        # Backup logic
        working_file = file_path
        try:
            backup_folder = os.path.join(os.path.dirname(file_path), "backups")
            os.makedirs(backup_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_folder, f"{os.path.basename(file_path).replace('.py','')}_{timestamp}.py")
            shutil.copy2(file_path, backup_path)
            print(f"Created backup at {backup_path}")
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}")

    # Read the entire content
    try:
        with open(working_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading strategy file {working_file}: {e}")
        return False

    params_modified = False
    arb_params_modified = False

    # Replace PARAMS if new_params are provided
    if new_params is not None:
        # Regex to find the entire PARAMS = { ... } block, handling potential variations
        params_pattern = re.compile(r"(^PARAMS\s*=\s*\{.*?^\})", re.DOTALL | re.MULTILINE)
        new_params_str = "".join(get_new_params_lines(new_params))

        if params_pattern.search(content):
            content = params_pattern.sub(new_params_str, content, count=1)
            params_modified = True
            print("Successfully replaced PARAMS block.")
        else:
            print("Warning: Could not find PARAMS block to replace.")
            # Depending on requirements, you might want to return False or raise an error here
            # If PARAMS *must* exist for the script to function.

    # Replace or Add ARB_PARAMS if new_arb_params are provided
    if new_arb_params is not None:
        # Regex to find the entire ARB_PARAMS = { ... } block
        arb_params_pattern = re.compile(r"(^ARB_PARAMS\s*=\s*\{.*?^\})", re.DOTALL | re.MULTILINE)
        new_arb_params_str = "".join(get_new_arb_params_lines(new_arb_params))

        if arb_params_pattern.search(content):
            content = arb_params_pattern.sub(new_arb_params_str, content, count=1)
            arb_params_modified = True
            print("Successfully replaced ARB_PARAMS block.")
        else:
            # If ARB_PARAMS block not found, try inserting after the PARAMS block
            # Find the end of the (potentially updated) PARAMS block
            params_pattern_for_insertion = re.compile(r"(^PARAMS\s*=\s*\{.*?^\})", re.DOTALL | re.MULTILINE)
            params_match = params_pattern_for_insertion.search(content)
            if params_match:
                insert_point = params_match.end()
                # Ensure there's a newline before inserting
                content = content[:insert_point].rstrip() + "\n\n" + new_arb_params_str + content[insert_point:]
                arb_params_modified = True
                print("Inserted ARB_PARAMS block after PARAMS.")
            else:
                print("Warning: Could not find PARAMS block to insert ARB_PARAMS after.")
                # Consider returning False or raising an error if this is critical

    # Write the updated content back
    try:
        with open(working_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {working_file} with new parameters.")
    except Exception as e:
        print(f"Error writing updated strategy file {working_file}: {e}")
        return False # Indicate failure

    # Return True if any modification happened successfully
    return params_modified or arb_params_modified

# Let's add a debug function at the top of the file after imports

def debug_file_operations(message, file_path):
    """Print debug info about file operations."""
    print(f"DEBUG: {message}")
    print(f"  Path: {file_path}")
    print(f"  Exists: {os.path.exists(file_path)}")
    print(f"  Is absolute: {os.path.isabs(file_path)}")
    if os.path.exists(file_path):
        print(f"  Size: {os.path.getsize(file_path)} bytes")
        print(f"  Last modified: {datetime.fromtimestamp(os.path.getmtime(file_path))}")
    else:
        # Check if directory exists
        dir_path = os.path.dirname(file_path)
        print(f"  Parent dir exists: {os.path.exists(dir_path)}")
        if os.path.exists(dir_path):
            print(f"  Parent dir contents: {os.listdir(dir_path)}")

# --- Optuna Objective Function ---

def objective(trial: optuna.Trial, target_product: str):
    """Function for Optuna to optimize. Supports SQUID_INK, VOLCANIC_ROCK, and B1B2_DEVIATION."""

    trial_id = trial.number
    # Start with base params for all strategies
    suggested_base_params = copy.deepcopy(BASE_PARAMS)
    suggested_arb_params = None  # Only set this when optimizing B1B2_DEVIATION

    # --- Suggest Parameters based on target_product ---
    if target_product == Product.SQUID_INK:
        # --- Suggest RSI Params for SQUID_INK ---
        rsi_window = trial.suggest_int("rsi_window", 10, 200)
        rsi_oversold = trial.suggest_int("rsi_oversold", 1, 49)
        rsi_overbought = trial.suggest_int("rsi_overbought", rsi_oversold + 1, 99)

        if Product.SQUID_INK in suggested_base_params:
            suggested_base_params[Product.SQUID_INK].update({
                "rsi_window": rsi_window,
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold
            })
        else:
             print(f"Warning: {Product.SQUID_INK} not in BASE_PARAMS, cannot update.")

    elif target_product == Product.VOLCANIC_ROCK:
        # --- Suggest RSI Params for VOLCANIC_ROCK ---
        rsi_window = trial.suggest_int("rsi_window", 11, 200)
        rsi_oversold = trial.suggest_int("rsi_oversold", 1, 49)
        rsi_overbought = trial.suggest_int("rsi_overbought", rsi_oversold + 1, 99)

        if Product.VOLCANIC_ROCK in suggested_base_params:
            suggested_base_params[Product.VOLCANIC_ROCK].update({
                "rsi_window": rsi_window,
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold
            })
        else:
             print(f"Warning: {Product.VOLCANIC_ROCK} not in BASE_PARAMS, cannot update.")

    elif target_product == "B1B2_DEVIATION":
        # --- Suggest B1B2_DEVIATION Params ---
        # Diff thresholds
        zscore_threshold_entry_long = -trial.suggest_float("zscore_threshold_entry_long", 0.5, 6, step=0.1)
        zscore_threshold_entry_short = trial.suggest_float("zscore_threshold_entry_short", 0.5, 6, step=0.1)
        zscore_threshold_exit = trial.suggest_float("zscore_threshold_exit", 0.1, 1.5, step=0.1)
        # EWMA periods
        ewma_period_mean = trial.suggest_int("ewma_period_mean", 10, 2000, step=10)
        ewma_period_std = trial.suggest_int("ewma_period_std", 20, 3000, step=10)
        # Max lot size
        max_lot = trial.suggest_int("max_arb_lot", 1, 15)

        # Update the separate arb params dictionary
        suggested_arb_params = copy.deepcopy(BASE_ARB_PARAMS)  # Initialize here only
        suggested_arb_params.update({
            "max_arb_lot": max_lot,
            "zscore_threshold_entry_long": zscore_threshold_entry_long, # Long entry threshold
            "zscore_threshold_entry_short": zscore_threshold_entry_short, # Short entry threshold
            "zscore_threshold_exit": zscore_threshold_exit, # Exit threshold
            "ewma_period_mean": ewma_period_mean, # EWMA period for mean
            "ewma_period_std": ewma_period_std # EWMA period for std
        })
    elif target_product == "VOLATILITY_SMILE":
        # --- Suggest VOLATILITY_SMILE Strategy Params ---
        # Suggest EWMA spans and rolling window
        short_ewma_span = trial.suggest_int("short_ewma_span", 10, 50)
        long_ewma_span = trial.suggest_int("long_ewma_span", 50, 200)
        rolling_window = trial.suggest_int("rolling_window", 30, 100)
        
        # Suggest Z-score thresholds - only use global thresholds
        zscore_upper_threshold = trial.suggest_float("zscore_upper_threshold", 0.5, 3.0, step=0.1)
        zscore_lower_threshold = -trial.suggest_float("zscore_lower_threshold", 0.5, 3.0, step=0.1)
        
        # Suggest trade size parameter
        trade_size = trial.suggest_int("trade_size", 5, 30)
        
        # Create a suggested params structure for our strategy
        suggested_volatility_params = {
            "short_ewma_span": short_ewma_span,
            "long_ewma_span": long_ewma_span,
            "rolling_window": rolling_window,
            "zscore_upper_threshold": zscore_upper_threshold,
            "zscore_lower_threshold": zscore_lower_threshold,
            "trade_size": trade_size,
            "day": 1  # Set day to 1 for optimization
        }
    else:
        raise ValueError(f"Invalid target_product specified for optimization: {target_product}")

    # --- Input validation ---
    if target_product == Product.SQUID_INK and suggested_base_params: 
        sq_params = suggested_base_params.get(Product.SQUID_INK, {})
        if sq_params.get("rsi_window", 0) < 2:
             print(f"Trial {trial_id}: Invalid SQ window {sq_params.get('rsi_window')} < 2. Pruning.")
             raise optuna.TrialPruned()
        if not (0 < sq_params.get("rsi_oversold", -1) < sq_params.get("rsi_overbought", -1) < 100):
             print(f"Trial {trial_id}: Invalid SQ thresholds OB={sq_params.get('rsi_overbought')} OS={sq_params.get('rsi_oversold')}. Pruning.")
             raise optuna.TrialPruned()

    elif target_product == Product.VOLCANIC_ROCK and suggested_base_params: 
        vr_params = suggested_base_params.get(Product.VOLCANIC_ROCK, {})
        if vr_params.get("rsi_window", 0) < 2:
             print(f"Trial {trial_id}: Invalid VR window {vr_params.get('rsi_window')} < 2. Pruning.")
             raise optuna.TrialPruned()
        if not (0 < vr_params.get("rsi_oversold", -1) < vr_params.get("rsi_overbought", -1) < 100):
             print(f"Trial {trial_id}: Invalid VR thresholds OB={vr_params.get('rsi_overbought')} OS={vr_params.get('rsi_oversold')}. Pruning.")
             raise optuna.TrialPruned()

    elif target_product == "B1B2_DEVIATION":
        arb_p = suggested_arb_params
        if not (arb_p["diff_threshold_b1"] > 0 and arb_p["diff_threshold_b2"] > 0 and arb_p["diff_threshold_b1_b2"] > 0):
            print(f"Trial {trial_id}: Invalid B1B2 Thresholds <= 0. Pruning.")
            raise optuna.TrialPruned()
        if arb_p["max_arb_lot"] <= 0:
            print(f"Trial {trial_id}: Invalid max_arb_lot <= 0. Pruning.")
            raise optuna.TrialPruned()

    # 2. Prepare for Backtest
    if not os.path.exists(TEMP_DIR): 
        os.makedirs(TEMP_DIR)
        print(f"Created directory: {TEMP_DIR}")
    
    # Ensure datamodel.py exists in temp dir
    datamodel_src_path = "datamodel.py"
    datamodel_dest_path = os.path.join(TEMP_DIR, "datamodel.py")
    
    try:
        strategy_dir = os.path.dirname(DEFAULT_STRATEGY_PATH)
        potential_dm_path_strat = os.path.join(strategy_dir, "datamodel.py")
        project_root = os.path.abspath(os.path.join(strategy_dir, '..', '..')) # Adjust levels if needed
        potential_dm_path_root = os.path.join(project_root, "datamodel.py")

        if os.path.exists(potential_dm_path_strat):
            shutil.copy2(potential_dm_path_strat, datamodel_dest_path)
            print(f"Copied datamodel.py from strategy dir to {datamodel_dest_path}")
        elif os.path.exists(potential_dm_path_root):
            shutil.copy2(potential_dm_path_root, datamodel_dest_path)
            print(f"Copied datamodel.py from project root to {datamodel_dest_path}")
        elif os.path.exists(datamodel_src_path): # Check current dir as last resort
             shutil.copy2(datamodel_src_path, datamodel_dest_path)
             print(f"Copied datamodel.py from current dir to {datamodel_dest_path}")
        else:
            print(f"Error: datamodel.py not found. Cannot run.")
            raise optuna.TrialPruned("datamodel.py not found")
    except Exception as e:
        print(f"Error copying datamodel.py: {e}")
        traceback.print_exc() # Add traceback for more detailed error info
        raise optuna.TrialPruned()

    # Generate unique filename for this trial
    timestamp = int(time.time())
    random_suffix = random.randint(1000,9999)
    temp_strategy_filename = f"temp_strategy_{trial_id}_{timestamp}_{random_suffix}.py"
    temp_log_filename = f"temp_backtest_{trial_id}_{timestamp}_{random_suffix}.log"
    
    # Use absolute paths to avoid any confusion
    temp_dir_abs = os.path.abspath(TEMP_DIR)
    temp_strategy_path = os.path.join(temp_dir_abs, temp_strategy_filename)
    temp_log_path = os.path.join(temp_dir_abs, temp_log_filename)
    
    print(f"Generated temp strategy path: {temp_strategy_path}")
    debug_file_operations("Temp directory check", temp_dir_abs)

    # 3. Modify Strategy File
    if target_product == "VOLATILITY_SMILE":
        # For VOLATILITY_SMILE, directly copy the original file first, then modify
        try:
            # Make sure target directory exists
            os.makedirs(os.path.dirname(temp_strategy_path), exist_ok=True)
            # Copy the file
            shutil.copy2(DEFAULT_STRATEGY_PATH, temp_strategy_path)
            print(f"Copied original strategy to {temp_strategy_path}")
        except Exception as e:
            print(f"Error copying strategy file to temp location: {e}")
            raise optuna.TrialPruned()
    else:
        # For other target products, use the standard modify_strategy_params function
        try:
            modified = modify_strategy_params(DEFAULT_STRATEGY_PATH, temp_strategy_path, suggested_base_params, suggested_arb_params)
            if not modified:
                print(f"Trial {trial_id}: Failed to modify strategy file. Pruning.")
                if os.path.exists(temp_strategy_path):
                    try: os.remove(temp_strategy_path)
                    except OSError: pass
                raise optuna.TrialPruned()
        except Exception as e:
            print(f"Trial {trial_id}: Exception modifying strategy file: {e}")
            traceback.print_exc()
            raise optuna.TrialPruned()

    # Verify the temp file exists before proceeding
    debug_file_operations("Checking strategy file before running backtester", temp_strategy_path)
    if not os.path.exists(temp_strategy_path):
        print(f"Trial {trial_id}: Expected temp file {temp_strategy_path} does not exist. Pruning.")
        raise optuna.TrialPruned()

    # --- Add this block for verification ---
    try:
        with open(temp_strategy_path, 'r', encoding='utf-8') as f_verify:
            verify_content = f_verify.read()
            
        # Extract the relevant PARAMS or ARB_PARAMS block from the temp file
        params_to_verify = None
        if target_product == "B1B2_DEVIATION":
            match_verify = re.search(r"^ARB_PARAMS\s*=\s*({.*?^})", verify_content, re.DOTALL | re.MULTILINE)
            if match_verify:
                try:
                    # Execute to get the dictionary
                    exec_locals_verify = {}
                    exec(f"VERIFIED_PARAMS = {match_verify.group(1)}", {}, exec_locals_verify)
                    params_to_verify = exec_locals_verify['VERIFIED_PARAMS']
                except Exception as exec_e:
                    print(f"Trial {trial_id}: Error executing verified ARB_PARAMS: {exec_e}")
        elif target_product in [Product.SQUID_INK, Product.VOLCANIC_ROCK]: # Check if target is one of the base products
             match_verify = re.search(r"^PARAMS\s*=\s*({.*?^})", verify_content, re.DOTALL | re.MULTILINE)
             if match_verify:
                 try:
                     # Define Product for exec context
                     exec_globals_verify = {"Product": Product}
                     exec_locals_verify = {}
                     exec(f"VERIFIED_PARAMS = {match_verify.group(1)}", exec_globals_verify, exec_locals_verify)
                     # Get the specific product's dict
                     params_to_verify = exec_locals_verify['VERIFIED_PARAMS'].get(target_product)
                 except Exception as exec_e:
                     print(f"Trial {trial_id}: Error executing verified PARAMS: {exec_e}")
        # Add case for VOLATILITY_SMILE if needed, although its params are modified differently
        # elif target_product == "VOLATILITY_SMILE":
        #     # Extract volatility params using regex if needed for verification
        #     pass 

        if params_to_verify:
            print(f"Trial {trial_id}: Verified params in {temp_strategy_filename}: {params_to_verify}")
            # Optional: Add a comparison here against suggested_arb_params or suggested_base_params[target_product]
            # verify_target_params = suggested_arb_params if target_product == "B1B2_DEVIATION" else suggested_base_params.get(target_product)
            # if verify_target_params and params_to_verify != verify_target_params:
            #     print(f"Trial {trial_id}: WARNING! Verified params {params_to_verify} DON'T MATCH suggested params {verify_target_params}!")
            #     # Decide if you want to prune or raise an error here
        else:
            # Only print warning if we expected to verify PARAMS or ARB_PARAMS
            if target_product != "VOLATILITY_SMILE": 
                 print(f"Trial {trial_id}: Warning: Could not read back/verify parameters from temp file for {target_product}.")
            
    except Exception as verify_e:
        print(f"Trial {trial_id}: Error verifying parameters in temp file: {verify_e}")
        traceback.print_exc() # Print full traceback for verification errors
    # --- End verification block ---

    # Add validation step to check for syntax errors
    if not validate_python_syntax(temp_strategy_path):
        print(f"Trial {trial_id}: Generated file has syntax errors. Pruning.")
        if os.path.exists(temp_strategy_path):
            try:
                # Keep the file for debugging
                print(f"Trial {trial_id}: Keeping strategy file with syntax errors for debugging: {temp_strategy_path}")
            except Exception as e:
                print(f"Warning: Error handling syntax validation: {e}")
        raise optuna.TrialPruned()
        
    # --- Force clean import by using a fixed filename ---
    fixed_temp_basename = "current_trial_strategy.py"
    fixed_temp_path = os.path.join(temp_dir_abs, fixed_temp_basename)
    
    # Print only the params being tested (Moved here, before potential rename error)
    if target_product == "B1B2_DEVIATION":
        print(f"\nTrial {trial_id}: Testing Params -> B1B2_DEVIATION: {suggested_arb_params}")
    elif target_product == Product.SQUID_INK and suggested_base_params:
        print(f"\nTrial {trial_id}: Testing Params -> SQUID_INK: {suggested_base_params.get(Product.SQUID_INK, 'N/A')}")
    elif target_product == Product.VOLCANIC_ROCK and suggested_base_params:
        print(f"\nTrial {trial_id}: Testing Params -> VOLCANIC_ROCK: {suggested_base_params.get(Product.VOLCANIC_ROCK, 'N/A')}")
    elif target_product == "VOLATILITY_SMILE":
        print(f"\nTrial {trial_id}: Testing Params -> VOLATILITY_SMILE: {suggested_volatility_params}")
    else:
        print(f"\nTrial {trial_id}: Testing Params -> {target_product}: (Using Base Params)")

    # 5. Execution and PnL Parsing Logic
    final_pnl = -float('inf'); backtest_failed = False
    try:
        # Ensure any previous fixed file is removed
        if os.path.exists(fixed_temp_path):
            os.remove(fixed_temp_path)
            
        # Rename the unique file to the fixed name
        os.rename(temp_strategy_path, fixed_temp_path)
        print(f"Trial {trial_id}: Renamed {temp_strategy_filename} to {fixed_temp_basename} for backtest.")
        
        # --- Use the FIXED path relative to TEMP_DIR for the backtester ---
        backtester_module_path_arg = f"{TEMP_DIR}/{fixed_temp_basename}"
        
        # Construct and run the backtester command using the fixed path
        backtest_command = ["python", "-m", BACKTESTER_SCRIPT_PATH] if not BACKTESTER_SCRIPT_PATH.endswith(".py") else ["python", BACKTESTER_SCRIPT_PATH]
        
        backtest_command.append(backtester_module_path_arg) # Use the fixed relative path arg
        backtest_command.append("4")
        backtest_command.append("--match-trades")
        backtest_command.append("worse")
        backtest_command.append("--no-out")
        
        print(f"Trial {trial_id}: Running backtest -> {' '.join(backtest_command)}") # Prints the fixed relative path
        
        # --- Execute Backtest ---
        result = subprocess.run(backtest_command, capture_output=True, text=True, check=False, timeout=420, encoding='utf-8', errors='replace')
        
        # --- PnL Parsing (Existing Code) ---
        if result.returncode != 0:
            backtest_failed = True; print(f"Trial {trial_id}: Backtester failed ({result.returncode}).")
            # Log failure details (existing code)
            try:
                log_content = f"Return Code: {result.returncode}\n\nSTDOUT:\n{result.stdout or 'N/A'}\n\nSTDERR:\n{result.stderr or 'N/A'}"
                with open(temp_log_path, 'w', encoding='utf-8') as log_f: log_f.write(log_content)
                print(f"  -> Wrote failure details to {temp_log_path}")
            except Exception as log_e: print(f"  -> Error writing failure log: {log_e}")
        
        # Print last lines for debug (existing code)
        stdout_lines = result.stdout.splitlines()
        print(f"Trial {trial_id}: Last 10 lines of output:")
        for line in stdout_lines[-10:]:
            print(f"  > {line}")
            
        # Parse PnL (existing code with comma handling)
        found_pnl = False
        profit_patterns = [
            r'^Total profit:\s*([-+]?[\d,]*\.?\d+)', 
            r'PnL:\s*([-+]?[\d,]*\.?\d+)',
            r'Profit:\s*([-+]?[\d,]*\.?\d+)',
            r'P&L:\s*([-+]?[\d,]*\.?\d+)',
            r'Net:\s*([-+]?[\d,]*\.?\d+)',
            r'Result:\s*([-+]?[\d,]*\.?\d+)'
        ]
        for line in reversed(stdout_lines):
            line_strip = line.strip()
            for pattern in profit_patterns:
                match = re.search(pattern, line_strip)
                if match:
                    try:
                        profit_str = match.group(1)
                        profit_str_no_commas = profit_str.replace(',', '')
                        final_pnl = float(profit_str_no_commas)
                        print(f"Trial {trial_id}: Parsed PnL from stdout using pattern '{pattern}': {final_pnl:.2f} (Original: '{profit_str}')")
                        found_pnl = True
                        break
                    except ValueError:
                        print(f"Warning: Could not convert PnL value '{profit_str_no_commas}' from: {line_strip}")
            if found_pnl: break

    except subprocess.TimeoutExpired:
        print(f"Trial {trial_id}: Backtester timed out.")
        backtest_failed = True
        # Log timeout
        try:
            with open(temp_log_path, 'w', encoding='utf-8') as log_f: log_f.write("Timeout.")
            print(f"  -> Wrote timeout log: {temp_log_path}")
        except Exception as log_e: print(f"  -> Error writing timeout log: {log_e}")
        # Ensure pnl is set for pruning
        final_pnl = -float('inf') 
        pass # Ensure block is not empty if original handling was just logging
    except Exception as e: # Catch errors during setup/rename/run
         print(f"Trial {trial_id}: Error during file renaming or backtest execution: {e}")
         traceback.print_exc()
         backtest_failed = True # Mark as failed
         final_pnl = -float('inf') # Ensure pruning
    finally:
        # --- Cleanup: Ensure the fixed file is removed --- 
        if os.path.exists(fixed_temp_path):
            try:
                os.remove(fixed_temp_path)
                # print(f"Trial {trial_id}: Cleaned up {fixed_temp_basename}")
            except OSError as e:
                print(f"Error removing fixed temp strategy file {fixed_temp_basename}: {e}")
        
        # Keep the uniquely named log file based on original logic (existing code)
        if os.path.exists(temp_log_path):
             if backtest_failed or final_pnl == -float('inf'):
                 print(f"Trial {trial_id}: Keeping log file: {temp_log_filename}")
             else:
                 try: os.remove(temp_log_path)
                 except OSError as e: print(f"Error removing log file {temp_log_filename}: {e}")
                 
        # We no longer need to keep the uniquely named python file as it was renamed and the fixed one is deleted.
        # The original unique file path `temp_strategy_path` no longer points to the file after the rename.
        # If you needed to keep the *specific* strategy code for a failed/early trial,
        # you would need to rename `fixed_temp_path` *back* to `temp_strategy_path` 
        # before the `finally` block concludes, or copy it.

    if final_pnl == -float('inf'):
        print(f"Trial {trial_id}: Could not determine PnL or setup failed. Pruning.")
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
                        choices=[Product.SQUID_INK, Product.VOLCANIC_ROCK, "B1B2_DEVIATION", "VOLATILITY_SMILE"],
                        help='The product/strategy to optimize (e.g., SQUID_INK, VOLCANIC_ROCK, B1B2_DEVIATION, VOLATILITY_SMILE)')
    parser.add_argument('--strategy-file',
                        type=str,
                        default=DEFAULT_STRATEGY_PATH,
                        help=f'Path to the strategy file to optimize (default: {DEFAULT_STRATEGY_PATH})')
    args = parser.parse_args()
    target_product = args.product
    
    # Update the strategy path based on command line argument - avoid global
    # Directly modify the module-level variable
    import sys
    current_module = sys.modules[__name__]
    setattr(current_module, 'DEFAULT_STRATEGY_PATH', args.strategy_file)
    print(f"Using strategy file: {DEFAULT_STRATEGY_PATH}")
    
    # --- End Argument Parsing ---

    # --- Dynamic Study Name ---
    study_suffix = target_product
    strategy_filename = os.path.basename(DEFAULT_STRATEGY_PATH).replace('.py', '')
    STUDY_NAME = f"{strategy_filename}_{target_product}_opt" # Reflects the script being optimized
    print(f"Target Product/Strategy: {target_product}")
    print(f"Study Name: {STUDY_NAME}")
    # --- End Dynamic Study Name ---

    print(f"Starting Optuna optimization for {target_product} in {DEFAULT_STRATEGY_PATH} (evaluated on R4 data)...")
    storage_name = f"sqlite:///{STUDY_NAME}.db"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    
    try:
        # Set up parallel optimization if TPE sampler is used
        study = optuna.create_study(
            storage=storage_name,
            study_name=STUDY_NAME,
            direction="maximize",
            load_if_exists=True,
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(multivariate=True, seed=42)
        )
        
        # Define objective that depends on target_product
        objective_with_target = functools.partial(objective, target_product=target_product)
        
        # Optimize with configured trials
        study.optimize(objective_with_target, n_trials=N_TRIALS, timeout=None, show_progress_bar=True)
        
        # After optimization completes, display the best parameters
        print(f"\n--- Optimization completed. ---")
        print(f"Number of completed trials: {len(study.trials)}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value (PnL): {study.best_value:.2f}")
    except KeyboardInterrupt:
        print("Optimization stopped manually. Displaying results from trials completed so far...")
    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        import traceback
        traceback.print_exc()
        print("Attempting to display results from trials completed so far...")
    
    # Display results regardless of how optimization ended
    try:
        # Get the best trial and extract params
        trial = study.best_trial
        print(f"Best parameters found for {target_product}:")
        
        # Get the base params structure to build on top of
        best_full_params = copy.deepcopy(BASE_PARAMS)
        best_full_arb_params = copy.deepcopy(BASE_ARB_PARAMS)
        
        # Dictionary mapping optuna parameter names to strategy parameter names
        # This will vary by target_product, set a sensible default first
        best_params_target_only = {}
        
        if target_product == "VOLATILITY_SMILE":
            param_map = {
                "short_ewma_span": "short_ewma_span",
                "long_ewma_span": "long_ewma_span",
                "rolling_window": "rolling_window",
                "zscore_upper_threshold": "zscore_upper_threshold",
                "zscore_lower_threshold": "zscore_lower_threshold",
                "trade_size": "trade_size"
            }
            target_dict_to_update = {}
            dict_name = "VOLATILITY_PARAMS"
        elif target_product == Product.SQUID_INK:
            param_map = {
                "rsi_window": "rsi_window",
                "rsi_overbought": "rsi_overbought",
                "rsi_oversold": "rsi_oversold"
            }
            target_dict_to_update = best_full_params.get(Product.SQUID_INK, {})
            dict_name = "PARAMS"
        elif target_product == Product.VOLCANIC_ROCK:
            param_map = {
                "rsi_window": "rsi_window",
                "rsi_overbought": "rsi_overbought",
                "rsi_oversold": "rsi_oversold"
            }
            target_dict_to_update = best_full_params.get(Product.VOLCANIC_ROCK, {})
            dict_name = "PARAMS"
        elif target_product == "B1B2_DEVIATION":
            param_map = {
                "zscore_threshold_entry_long": "zscore_threshold_entry_long",
                "zscore_threshold_entry_short": "zscore_threshold_entry_short",
                "zscore_threshold_exit": "zscore_threshold_exit",
                "ewma_period_mean": "ewma_period_mean",
                "ewma_period_std": "ewma_period_std",
            }
            target_dict_to_update = best_full_arb_params
            dict_name = "ARB_PARAMS"
        else:
            print(f"Warning: Unsupported target product for mapping: {target_product}")
            param_map = {}
            current_param_map = {}
            target_dict_to_update = {}
            dict_to_print = {}
            dict_name = "UNKNOWN"

        # Populate best_params_target_only using the map
        for optuna_key, param_key in param_map.items():
            if optuna_key in trial.params:
                 best_params_target_only[param_key] = trial.params[optuna_key]

        print(f"      {target_product}: {best_params_target_only}")

        # Update the *correct* full dictionary with the best target params
        if target_product == "B1B2_DEVIATION":
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
        elif target_product == Product.VOLCANIC_ROCK:
            if Product.VOLCANIC_ROCK in best_full_params and isinstance(best_full_params[Product.VOLCANIC_ROCK], dict):
                 best_full_params[Product.VOLCANIC_ROCK].update(best_params_target_only)
                 # Update dict_to_print for final output
                 dict_to_print = best_full_params
            else:
                 print(f"Warning: Cannot update best params for {target_product}, base structure incorrect.")
        elif target_product == "VOLATILITY_SMILE":
            # Create a clean representation of the best parameters for display
            best_volatility_params = {
                "short_ewma_span": trial.params.get("short_ewma_span"),
                "long_ewma_span": trial.params.get("long_ewma_span"),
                "rolling_window": trial.params.get("rolling_window"),
                "zscore_upper_threshold": trial.params.get("zscore_upper_threshold"),
                "zscore_lower_threshold": trial.params.get("zscore_lower_threshold"),
                "trade_size": trial.params.get("trade_size"),
                "day": 2
            }
            # Print the best parameters in a clean format
            print("\n--- Best VOLATILITY_SMILE Parameters ---")
            print(f"short_ewma_span: {best_volatility_params['short_ewma_span']}")
            print(f"long_ewma_span: {best_volatility_params['long_ewma_span']}")
            print(f"rolling_window: {best_volatility_params['rolling_window']}")
            print(f"zscore_upper_threshold: {best_volatility_params['zscore_upper_threshold']}")
            print(f"zscore_lower_threshold: {best_volatility_params['zscore_lower_threshold']}")
            print(f"trade_size: {best_volatility_params['trade_size']}")
            print(f"day: {best_volatility_params['day']}")
            
            print("\n--- Copy-Paste Version ---")
            print("self.short_ewma_span = " + str(best_volatility_params['short_ewma_span']))
            print("self.long_ewma_span = " + str(best_volatility_params['long_ewma_span']))
            print("self.rolling_window = " + str(best_volatility_params['rolling_window']))
            print("self.zscore_upper_threshold = " + str(best_volatility_params['zscore_upper_threshold']))
            print("self.zscore_lower_threshold = " + str(best_volatility_params['zscore_lower_threshold']))
            print("self.trade_size = " + str(best_volatility_params['trade_size']))
            print("self.day = " + str(best_volatility_params['day']))
            print("--- End VOLATILITY_SMILE Parameters ---")
            
            # Not updating dict_to_print for volatility smile since it uses a different structure
            dict_to_print = {}
        else:
            print(f"Warning: Cannot update best params for {target_product}, base structure incorrect.")

        # Generate the best dictionary string
        print(f"\n--- Best {dict_name} dictionary for {os.path.basename(DEFAULT_STRATEGY_PATH)} --- ")

        formatted_lines_best = [f"{dict_name} = {{\n"]
        if dict_name == "PARAMS":
             params_for_json_best = {}
             # Ensure all relevant keys are present in the final dict_to_print
             # Add keys from BASE_PARAMS
             for key, value in BASE_PARAMS.items():
                 if key not in dict_to_print: dict_to_print[key] = value

             for key, value in dict_to_print.items():
                 # Reuse key formatting - needs robust handling for Product keys vs String keys
                 key_str = repr(key) # Fallback
                 if key == Product.SQUID_INK: key_str = 'Product.SQUID_INK'
                 elif key == Product.VOLCANIC_ROCK: key_str = 'Product.VOLCANIC_ROCK'
                 elif key == Product.B1B2_DEVIATION: key_str = 'Product.B1B2_DEVIATION'
                 elif isinstance(key, str) and key.startswith("Product."): key_str = key # Handle Product.<n> case

                 params_for_json_best[key_str] = value

             param_items_best = list(params_for_json_best.items())
             for i, (key_str, value_dict) in enumerate(param_items_best):
                if not isinstance(value_dict, dict): continue
                formatted_lines_best.append(f"    {key_str}: {{\n")
                value_items_best = list(value_dict.items())
                for j, (param_name, param_value) in enumerate(value_items_best):
                    try: value_repr = repr(param_value)
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



  