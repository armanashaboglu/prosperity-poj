import optuna # type: ignore
import subprocess
import os
import json
import re
import shutil
import time
import random
from enum import Enum

# Configuration
BACKTESTER_SCRIPT_PATH = "prosperity3bt"
STRATEGY_PATH = "strategy/round5/round5_submission.py"
TEMP_DIR = "temp_optimizer_volatility"
N_TRIALS = 125

class Product(str, Enum):
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

def modify_volatility_params(file_path, temp_file_path, params):
    """Modifies VolatilitySmileStrategy parameters in the strategy file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        modified_lines = []
        # --- Revised State Variables ---
        found_class = False # Flag: Have we seen 'class VolatilitySmileStrategy:'?
        in_init_method = False # Flag: Are we currently inside the target __init__?
        # ---

        params_to_update = { # Parameter names MUST match the start of the line in the code
            "self.short_ewma_span = ": params.get('short_ewma_span'),
            "self.long_ewma_span = ": params.get('long_ewma_span'),
            "self.rolling_window = ": params.get('rolling_window'),
            "self.zscore_upper_threshold = ": params.get('zscore_upper_threshold'),
            "self.zscore_lower_threshold = ": params.get('zscore_lower_threshold'),
            "self.threshold_adjustment_rate = ": params.get('threshold_adjustment_rate')
        }
        updated_params_count = 0
        processed_init = False # Flag to ensure we only process the first __init__ after the class def

        for line in lines:
            stripped_line = line.strip()

            # --- Revised Detection Logic ---
            # Look for the class definition
            if stripped_line.startswith("class VolatilitySmileStrategy:"):
                found_class = True
                in_init_method = False # Reset init flag if class is somehow redefined
                processed_init = False # Allow processing the next __init__ found

            # If we found the class, look for its __init__ method
            elif found_class and not processed_init and stripped_line.startswith("def __init__(self"):
                in_init_method = True
                processed_init = True # Mark this __init__ as processed
                modified_lines.append(line) # Keep the init definition line
                continue # Skip param check for the def line itself

            # Heuristic: Assume __init__ ends when the next method definition starts at the same indentation level or less
            elif in_init_method and stripped_line.startswith("def ") and (line.find("def ") <= lines[lines.index(line)-1].find("self.")): # Basic check for exiting indentation block
                 in_init_method = False
                 # Don't reset found_class here, allow finding other methods within the same class if needed later

            # --- Parameter Replacement Logic (Only when inside the detected init) ---
            if in_init_method:
                param_updated_this_line = False
                for key, value in params_to_update.items():
                    # Use startswith on the stripped line for robustness
                    if value is not None and stripped_line.startswith(key):
                        indentation = line[:len(line) - len(line.lstrip())]
                        new_line = f"{indentation}{key}{value}\n"
                        modified_lines.append(new_line)
                        param_updated_this_line = True
                        updated_params_count += 1
                        # We found the parameter for this line, move to the next line
                        break

                if not param_updated_this_line:
                    # If no parameter matched the start of this line within __init__, keep the original line
                    modified_lines.append(line)

            else:
                # Keep lines outside the target class/init method
                modified_lines.append(line)
            # --- End Revised Logic ---

        # Write modified content
        with open(temp_file_path, 'w', encoding='utf-8') as file:
            file.writelines(modified_lines)

        # Verification (still useful)
        num_expected_updates = len([v for v in params.values() if v is not None])
        if updated_params_count != num_expected_updates:
             print(f"Warning: Expected to update {num_expected_updates} params, but only updated {updated_params_count}. Check strategy file structure around VolatilitySmileStrategy.__init__.")

        return True

    except Exception as e:
        print(f"Error modifying strategy file: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_backtest(strategy_path, log_path=None):
    """Run backtest for Round 5 (Days 2, 3, 4) and return the PnL"""
    # --- UPDATED: Run across relevant days for Round 5 ---
    cmd = [
        "python", "-m", BACKTESTER_SCRIPT_PATH,
        strategy_path, "5-4", # Run Day 2 of Round 5
         "--match-trades", "worse", "--no-out"
    ]
    print(f"Running command: {' '.join(cmd)}") # Log the command being run

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300 # Add a timeout (e.g., 5 minutes)
        )

        if result.returncode != 0:
            print(f"Backtest failed with return code {result.returncode}:")
            print(f"Stderr: {result.stderr[-500:]}") # Print last 500 chars of stderr
            print(f"Stdout: {result.stdout[-500:]}") # Print last 500 chars of stdout
            return None # Indicate failure clearly

        # Parse the output (same logic as before)
        stdout_lines = result.stdout.splitlines()
        # print(f"STDOUT (last 20 lines): \n" + "\n".join(stdout_lines[-20:])) # More debug output if needed

        for line in reversed(stdout_lines):
            line_strip = line.strip()
            if line_strip.startswith("Total profit:"):
                try:
                    match = re.search(r':\s*([-+]?\d*\.?\d+)', line_strip)
                    if match:
                        final_pnl = float(match.group(1))
                        print(f"Parsed PnL: {final_pnl:.2f}")
                        return final_pnl
                except Exception as e:
                    print(f"Error parsing PnL line '{line_strip}': {e}")
                    break # Stop searching if parsing fails

        print("Could not find 'Total profit:' line in backtest output.")
        # Add more parsing attempts if needed based on actual backtester output
        return None # Return None if PnL not found

    except subprocess.TimeoutExpired:
        print("Backtest timed out.")
        return None
    except Exception as e:
        print(f"Error running backtest: {e}")
        return None

def objective(trial):
    """Objective function for Optuna optimization - VolatilitySmileStrategy"""

    # --- Define VolatilitySmile parameters to optimize ---
    params = {
        'short_ewma_span': trial.suggest_int('short_ewma_span', 5, 80),
        'long_ewma_span': trial.suggest_int('long_ewma_span', 10, 350),
        'rolling_window': trial.suggest_int('rolling_window', 10, 200),
        'zscore_upper_threshold': trial.suggest_float('zscore_upper_threshold', 0.1, 3.5, step=0.1),
        'zscore_lower_threshold': trial.suggest_float('zscore_lower_threshold', -3.5, -0.1, step=0.1),
        'threshold_adjustment_rate': trial.suggest_float('threshold_adjustment_rate', 0.0, 0.05)
    }
    print(f"\n--- Trial {trial.number} ---")
    print(f"Suggested Params: {params}")

    # --- Add Constraints ---
    if params['long_ewma_span'] <= params['short_ewma_span']:
        print(f"Pruning trial: long_ewma_span ({params['long_ewma_span']}) <= short_ewma_span ({params['short_ewma_span']})")
        raise optuna.TrialPruned("Constraint: long_ewma_span > short_ewma_span")
    if params['zscore_lower_threshold'] >= 0:
         print(f"Pruning trial: zscore_lower_threshold ({params['zscore_lower_threshold']}) must be negative.")
         raise optuna.TrialPruned("Constraint: zscore_lower_threshold < 0")
    if params['zscore_upper_threshold'] <= 0:
         print(f"Pruning trial: zscore_upper_threshold ({params['zscore_upper_threshold']}) must be positive.")
         raise optuna.TrialPruned("Constraint: zscore_upper_threshold > 0")


    # Make sure temp directory exists
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Unique filename for this trial
    trial_id = f"{trial.number}_{int(time.time())}_{random.randint(1000, 9999)}"
    temp_strategy_path = os.path.join(TEMP_DIR, f"temp_volatility_{trial_id}.py")
    temp_log_path = os.path.join(TEMP_DIR, f"backtest_log_{trial_id}.txt") # Log path (optional)

    # Prepare strategy file for this trial
    try:
        if os.path.exists("datamodel.py"):
             shutil.copy2("datamodel.py", os.path.join(TEMP_DIR, "datamodel.py"))

        if not modify_volatility_params(STRATEGY_PATH, temp_strategy_path, params):
            print(f"Trial {trial.number}: Failed to modify strategy file")
            raise optuna.TrialPruned("Failed to modify strategy file")

        # +++ START VERIFICATION STEP +++
        print(f"Verifying parameters in temporary file: {temp_strategy_path}")
        param_lines_found = []
        try:
            with open(temp_strategy_path, 'r', encoding='utf-8') as temp_file:
                lines = temp_file.readlines()
            in_init = False
            for line in lines:
                stripped_line = line.strip()
                if "class VolatilitySmileStrategy:" in line:
                    in_init = False # Reset detector
                elif "def __init__(self)" in line and "VolatilitySmileStrategy" in "".join(lines[lines.index(line)-2:lines.index(line)]):
                    in_init = True
                elif in_init and stripped_line.startswith("def "):
                    in_init = False # Exited init

                if in_init:
                    for param_prefix in [
                        "self.short_ewma_span =",
                        "self.long_ewma_span =",
                        "self.rolling_window =",
                        "self.zscore_upper_threshold =",
                        "self.zscore_lower_threshold =",
                        "self.threshold_adjustment_rate ="
                    ]:
                        if stripped_line.startswith(param_prefix):
                            param_lines_found.append(stripped_line)
                            break # Found one param line, move to next line
            print("--- Found Parameter Lines in Temp File: ---")
            for found_line in param_lines_found:
                print(f"  {found_line}")
            print("--- End Verification ---")
            if len(param_lines_found) != len(params):
                 print(f"Warning: Verification found {len(param_lines_found)} param lines, expected {len(params)}. Modification might be incomplete.")

        except Exception as e:
            print(f"Error during verification read: {e}")
        # +++ END VERIFICATION STEP +++


    except Exception as e:
        print(f"Trial {trial.number}: Error in file preparation or verification: {e}")
        raise optuna.TrialPruned("Error in file preparation/verification")

    # Run backtest
    pnl = run_backtest(temp_strategy_path, temp_log_path)

    # Clean up
    try:
        if os.path.exists(temp_strategy_path): os.remove(temp_strategy_path)
        if os.path.exists(os.path.join(TEMP_DIR, "datamodel.py")): os.remove(os.path.join(TEMP_DIR, "datamodel.py"))
    except Exception as e:
        print(f"Warning: Failed to clean up some temp files: {e}")

    if pnl is None:
        print(f"Trial {trial.number}: Backtest failed or PnL not found. Pruning.")
        raise optuna.TrialPruned("Backtest failed or PnL not found")

    print(f"Trial {trial.number} completed - PnL: {pnl:.2f}")
    return pnl

def main():
    # --- UPDATED: Study name ---
    study = optuna.create_study(direction='maximize', study_name='volatility_smile_optimization')
    # --- Consider adding storage for resuming studies ---
    # storage_name = "sqlite:///volatility_opt.db"
    # study = optuna.create_study(direction='maximize', study_name='volatility_smile_optimization', storage=storage_name, load_if_exists=True)

    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=3600*2) # Add timeout (e.g., 2 hours)
    except KeyboardInterrupt:
         print("Optimization stopped manually.")
    except Exception as e:
         print(f"An error occurred during optimization: {e}")


    # Print results if trials were completed
    if study.trials:
        print("\n======= OPTIMIZATION RESULTS =======")
        try:
            best_trial = study.best_trial
            print(f"Best trial: {best_trial.number}")
            print(f"Best PnL: {best_trial.value:.2f}")
            print("Best parameters:")
            for param, value in best_trial.params.items():
                # Don't try to print trade_size if it wasn't optimized
                print(f"  {param}: {value}")

            # Save the best strategy
            best_params = best_trial.params
            # --- UPDATED: Output file name ---
            final_strategy_path = "strategy/round5/round5_submission_optimized_volatility.py"
            # --- UPDATED: Call correct modification function ---
            if modify_volatility_params(STRATEGY_PATH, final_strategy_path, best_params):
                print(f"\nFinal optimized strategy saved to {final_strategy_path}")
            else:
                print("\nFailed to save final optimized strategy")

        except ValueError:
             print("No completed trials found. Could not determine best parameters.")


        # Save all results
        study_results = []
        for trial in study.trials:
            # Only include completed trials in results summary
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = {
                    'trial': trial.number,
                    'pnl': trial.value,
                    **trial.params,
                    'state': trial.state.name
                }
                # Remove trade_size if present in older pruned/failed trials being logged
                result.pop('trade_size', None)
                # Remove threshold_adjustment_rate if loading old results without it
                result.pop('threshold_adjustment_rate', None)
                study_results.append(result)
            elif trial.state != optuna.trial.TrialState.PRUNED: # Log other states too if needed
                 result = {
                    'trial': trial.number,
                    'pnl': None, # No value if not complete
                     **trial.params,
                     'state': trial.state.name
                 }
                 # Remove trade_size if present in older pruned/failed trials being logged
                 result.pop('trade_size', None)
                 # Remove threshold_adjustment_rate if loading old results without it
                 result.pop('threshold_adjustment_rate', None)
                 result.update(trial.params) # Ensure all current params are there
                 result['state'] = trial.state.name
                 study_results.append(result)


        if study_results:
            import pandas as pd # type: ignore
            results_df = pd.DataFrame(study_results)
            # --- UPDATED: Results file name ---
            results_filename = "volatility_smile_optimization_results.csv"
            results_df.to_csv(results_filename, index=False)
            print(f"Optimization results saved to {results_filename}")
        else:
            print("No completed trials to save.")

    else:
        print("No trials were run.")


if __name__ == "__main__":
    # Create the temp directory if it doesn't exist at the start
    if not os.path.exists(TEMP_DIR):
        try:
            os.makedirs(TEMP_DIR)
        except OSError as e:
            print(f"Error creating temporary directory {TEMP_DIR}: {e}")
            exit(1) # Exit if we can't create the temp dir

    main()

    # Optional: Clean up the temp directory at the very end
    # try:
    #     if os.path.exists(TEMP_DIR):
    #         shutil.rmtree(TEMP_DIR)
    #         print(f"Removed temporary directory: {TEMP_DIR}")
    # except Exception as e:
    #     print(f"Warning: Could not remove temporary directory {TEMP_DIR}: {e}")



  