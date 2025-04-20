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
STRATEGY_PATH = "strategy/round4/round4_macarons.py"
TEMP_DIR = "temp_optimizer_macarons_rsi"
N_TRIALS = 125

class Product(str, Enum):
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

# Function to modify only RSI parameters
def modify_rsi_params(file_path, temp_file_path, rsi_params):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        modified_lines = []
        in_init_method = False
        rsi_params_updated = False
        
        for line in lines:
            # Check if we're in the __init__ method of MacaronsStrategy
            if "def __init__(self)" in line:
                in_init_method = True
                modified_lines.append(line)
                continue
                
            # Exit the __init__ method when we see a line that seems to be a method definition
            if in_init_method and re.match(r'^\s+def [a-zA-Z_]', line):
                in_init_method = False
            
            # If we're in __init__ and encounter an RSI parameter line, update it
            if in_init_method and "self.rsi_period = " in line and not rsi_params_updated:
                modified_lines.append(f"        self.rsi_period = {rsi_params.get('rsi_period', 96)}\n")
                modified_lines.append(f"        # Corrected RSI levels: oversold < overbought\n")
                modified_lines.append(f"        self.rsi_oversold = {rsi_params.get('rsi_oversold', 39)}\n")
                modified_lines.append(f"        self.rsi_overbought = {rsi_params.get('rsi_overbought', 56)}\n")
                rsi_params_updated = True
                
                # Skip the next two lines which should be the original RSI oversold/overbought lines
                skip_counter = 2
                continue
            
            # Skip lines we've replaced
            if in_init_method and rsi_params_updated and skip_counter > 0:
                skip_counter -= 1
                continue
                
            # If we're in _apply_rsi_strategy and find trade_size_pct usage, replace it
            if "_apply_rsi_strategy" in line or (in_init_method and "self.trade_size_pct = " in line):
                if "self.trade_size_pct = " in line:
                    modified_lines.append(f"        self.trade_size_pct = {rsi_params.get('trade_size_pct', 0.7)}\n")
                    continue
            
            # Add all other lines as they are
            modified_lines.append(line)
                
        # Write modified content to the temp file
        with open(temp_file_path, 'w', encoding='utf-8') as file:
            file.writelines(modified_lines)
        
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
            
        # Parse the output to find the PnL by searching in reverse for the final summary
        stdout_lines = result.stdout.splitlines()
        found_pnl = False
        
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
                        found_pnl = True
                        return final_pnl
                except Exception as e:
                    print(f"Error parsing PnL: {e}")
                    break
        
        if not found_pnl:
            # Alternative patterns to try if "Total profit:" is not found
            for line in reversed(stdout_lines):
                line_strip = line.strip()
                # Try other possible formats
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
    """Objective function for Optuna optimization - RSI parameters only"""
    
    # Define RSI parameters to optimize
    rsi_params = {
        'rsi_period': trial.suggest_int('rsi_period', 5, 150),
        'rsi_oversold': trial.suggest_int('rsi_oversold', 25, 45),
        'rsi_overbought': trial.suggest_int('rsi_overbought', 55, 75),
    }
    
    # Input validation
    if rsi_params['rsi_period'] < 5:
        raise ValueError("RSI period must be at least 5")
    if rsi_params['rsi_oversold'] >= rsi_params['rsi_overbought']:
        raise ValueError("RSI oversold must be less than overbought")
    
    # Make sure temp directory exists
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    # Generate unique filenames for this trial
    trial_id = f"{trial.number}_{int(time.time())}_{random.randint(1000, 9999)}"
    temp_strategy_path = os.path.join(TEMP_DIR, f"temp_macarons_rsi_{trial_id}.py")
    temp_log_path = os.path.join(TEMP_DIR, f"backtest_log_{trial_id}.txt")
    
    # Copy necessary files to temp dir
    try:
        # Copy datamodel.py if needed
        if os.path.exists("datamodel.py"):
            shutil.copy2("datamodel.py", os.path.join(TEMP_DIR, "datamodel.py"))
        
        # Modify strategy with RSI parameters
        if not modify_rsi_params(STRATEGY_PATH, temp_strategy_path, rsi_params):
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
    print(f"Trial {trial.number} completed - Parameters: {rsi_params}, PnL: {pnl}")
    
    return pnl  # Return PnL as the objective value to maximize

def main():
    # Create study and optimize
    study = optuna.create_study(direction='maximize', study_name='macarons_rsi_optimization')
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Print results
    print("\n======= OPTIMIZATION RESULTS =======")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best PnL: {study.best_trial.value}")
    print("Best RSI parameters:")
    for param, value in study.best_trial.params.items():
        print(f"  {param}: {value}")
    
    # Create the final strategy with the best parameters
    best_params = study.best_trial.params
    final_strategy_path = "strategy/round4/round4_macarons_optimized.py"
    
    if modify_rsi_params(STRATEGY_PATH, final_strategy_path, best_params):
        print(f"\nFinal optimized strategy saved to {final_strategy_path}")
    else:
        print("\nFailed to save final optimized strategy")
    
    # Optionally save study results
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
    results_df.to_csv("macarons_rsi_optimization_results.csv", index=False)
    print("Optimization results saved to macarons_rsi_optimization_results.csv")

if __name__ == "__main__":
    main()



  