import optuna
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
STRATEGY_PATH = "strategy/round3/round3_submission.py"
TEMP_DIR = "temp_optimizer_b1b2_deviation"
N_TRIALS = 125

class Product(str, Enum):
    B1B2_DEVIATION = "B1B2_DEVIATION"

# Function to modify B1B2_DEVIATION parameters
def modify_b1b2_deviation_params(file_path, temp_file_path, params):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Create a pattern to find the B1B2_DEVIATION parameters in the PARAMS dictionary
        pattern = r'(Product\.B1B2_DEVIATION:\s*\{[^\}]*\})'
        
        # Extract the current parameters section
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            print("Could not find B1B2_DEVIATION parameters in the file")
            return False
            
        current_params = match.group(1)
        
        # Create the replacement parameters string
        new_params = f'''Product.B1B2_DEVIATION: {{
        "deviation_mean": 0, # From notebook analysis (or optimize)
        "deviation_std_window": {params.get("deviation_std_window", 97)}, # Rolling window for std dev calc
        "zscore_threshold_entry": {params.get("zscore_threshold_entry", 11.5)}, # Z-score to enter
        "zscore_threshold_exit": {params.get("zscore_threshold_exit", 0.1)}, # Z-score to exit towards mean
        "target_deviation_spread_size": {params.get("target_deviation_spread_size", 60)} # Target units of deviation spread to hold
    }}'''
        
        # Replace the parameters in the content
        modified_content = content.replace(current_params, new_params)
        
        # Write modified content to the temp file
        with open(temp_file_path, 'w', encoding='utf-8') as file:
            file.write(modified_content)
        
        return True
    
    except Exception as e:
        print(f"Error modifying strategy file: {e}")
        return False

def run_backtest(strategy_path, log_path=None):
    """Run backtest and return the PnL"""
    cmd = [
        "python", "-m", BACKTESTER_SCRIPT_PATH,
        strategy_path, "4", "--match-trades", "worse", "--no-out" 
    ]
    
    try:
        # Add encoding='utf-8' and errors='replace' to handle encoding issues
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            encoding='utf-8',  # Explicitly use UTF-8
            errors='replace'   # Replace any characters that can't be decoded
        )
        
        if result.returncode != 0:
            print(f"Backtest failed: {result.stderr}")
            return None
            
        # Parse the output to find the PnL
        stdout_lines = result.stdout.splitlines()
        
        # Debug: Print a sample of the output to understand format
        print(f"Output sample (last 10 lines):")
        for line in stdout_lines[-10:]:
            print(f"  > {line}")
        
        # Iterate backwards through the lines to find the final profit summary
        for line in reversed(stdout_lines):
            line_strip = line.strip()
            # Look for the target line
            if line_strip.startswith("Total profit:"):
                try:
                    match = re.search(r':\s*([-+]?\d*\.?\d+)', line_strip)
                    if match:
                        final_pnl = float(match.group(1))
                        print(f"Parsed PnL: {final_pnl:.2f}")
                        return final_pnl
                except Exception as e:
                    print(f"Error parsing PnL: {e}")
                    break
        
        # Alternative patterns if "Total profit:" is not found
        for line in reversed(stdout_lines):
            line_strip = line.strip()
            for pattern in ["PnL:", "Profit:", "P&L:", "Net:", "Result:"]:
                if pattern in line_strip:
                    try:
                        match = re.search(f'{pattern}\s*([-+]?\d*\.?\d+)', line_strip)
                        if match:
                            final_pnl = float(match.group(1))
                            print(f"Parsed PnL using alternative pattern '{pattern}': {final_pnl:.2f}")
                            return final_pnl
                    except Exception:
                        pass
        
        print("Could not find PnL in backtest output after trying multiple patterns")
        return None
    
    except Exception as e:
        print(f"Error running backtest: {e}")
        return None

def objective(trial):
    """Objective function for Optuna optimization"""
    
    # Define parameters to optimize (updated for current strategy)
    params = {
        'deviation_std_window': trial.suggest_int('deviation_std_window', 50, 150),
        'zscore_threshold_entry': trial.suggest_float('zscore_threshold_entry', 5.0, 15.0),
        'zscore_threshold_exit': trial.suggest_float('zscore_threshold_exit', 0.01, 1.0),
        'target_deviation_spread_size': trial.suggest_int('target_deviation_spread_size', 10, 100),
    }
    
    # Make sure temp directory exists
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    # Generate unique filenames for this trial
    trial_id = f"{trial.number}_{int(time.time())}_{random.randint(1000, 9999)}"
    temp_strategy_path = os.path.join(TEMP_DIR, f"temp_strategy_{trial_id}.py")
    temp_log_path = os.path.join(TEMP_DIR, f"backtest_log_{trial_id}.txt")
    
    # Copy necessary files to temp dir
    try:
        # Copy datamodel.py if needed
        if os.path.exists("datamodel.py"):
            shutil.copy2("datamodel.py", os.path.join(TEMP_DIR, "datamodel.py"))
        
        # Modify strategy with parameters
        if not modify_b1b2_deviation_params(STRATEGY_PATH, temp_strategy_path, params):
            print(f"Trial {trial.number}: Failed to modify strategy file")
            raise optuna.TrialPruned()
    
    except Exception as e:
        print(f"Trial {trial.number}: Error in file preparation: {e}")
        raise optuna.TrialPruned()
    
    # Run backtest and get PnL
    pnl = run_backtest(temp_strategy_path, temp_log_path)
    
    # Clean up temporary files
    try:
        if os.path.exists(temp_strategy_path):
            os.remove(temp_strategy_path)
        if os.path.exists(temp_log_path):
            os.remove(temp_log_path)
    except Exception as e:
        print(f"Warning: Failed to clean up temp files: {e}")
    
    if pnl is None:
        raise optuna.TrialPruned()
    
    # Log the results
    print(f"Trial {trial.number} completed - Parameters: {params}, PnL: {pnl}")
    
    return pnl  # Return PnL as the objective value to maximize

def main():
    # Create study and optimize
    study = optuna.create_study(direction='maximize', study_name='b1b2_deviation_optimization')
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Print results
    print("\n======= OPTIMIZATION RESULTS =======")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best PnL: {study.best_trial.value}")
    print("Best B1B2_DEVIATION parameters:")
    for param, value in study.best_trial.params.items():
        print(f"  {param}: {value}")
    
    # Create the final strategy with the best parameters
    best_params = study.best_trial.params
    final_strategy_path = "strategy/round3/round3_submission_optimized.py"
    
    if modify_b1b2_deviation_params(STRATEGY_PATH, final_strategy_path, best_params):
        print(f"\nFinal optimized strategy saved to {final_strategy_path}")
    else:
        print("\nFailed to save final optimized strategy")
    
    # Save study results
    try:
        study_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = {
                    'trial': trial.number,
                    'pnl': trial.value,
                    **trial.params
                }
                study_results.append(result)
        
        # Save results to CSV
        import pandas as pd
        results_df = pd.DataFrame(study_results)
        results_df.to_csv("b1b2_deviation_optimization_results.csv", index=False)
        print("Optimization results saved to b1b2_deviation_optimization_results.csv")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main() 