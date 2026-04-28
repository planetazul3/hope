#!/usr/bin/env python3
"""
Advanced Bayesian Optimization Backtester
-----------------------------------------
Replaces naive grid search with Optuna (Tree-structured Parzen Estimator).
Intelligently explores the hyperparameter space, aggressively prunes zero-trade 
regions, and converges on maximum profitability parameters.

Requirements: pip install optuna plotly pandas
"""

import os
import re
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import optuna

# Configure professional institutional logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("OptunaBacktest")

# Silence Optuna's default verbose trial logging in favor of our own progress bar
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Compile regex patterns once globally for performance
REGEX_TRADES = re.compile(r"Total Trades:\s+(\d+)")
REGEX_WINS = re.compile(r"Wins:\s+(\d+)")
REGEX_WIN_RATE = re.compile(r"Win Rate:\s+([\d.]+)")
REGEX_PROFIT = re.compile(r"Total Profit:\s+([\-?\d.]+)")

def run_rust_backtest(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Executes the Rust backtest binary with dynamically injected parameters."""
    env = os.environ.copy()
    env["BACKTEST_MODE"] = "1"
    
    # Map optimization space to Rust environment variables
    mapping = {
        "threshold": "DERIV_THRESHOLD",
        "trend_length": "STRATEGY_MIN_TREND_LENGTH",
        "momentum_reward": "STRATEGY_MOMENTUM_REWARD",
        "volatility_penalty": "STRATEGY_VOLATILITY_PENALTY"
    }
    
    for key, env_var in mapping.items():
        if key in params:
            env[env_var] = str(params[key])

    # Enforce --release mode for high-throughput execution
    cmd = ["cargo", "run", "--bin", "backtest", "--locked", "--offline", "--release"]
    
    try:
        # Check=True ensures we catch execution panics
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        output = result.stdout
        
        trades_match = REGEX_TRADES.search(output)
        trades = int(trades_match.group(1)) if trades_match else 0
        
        if trades == 0:
            return {"trades": 0, "profit": 0.0, "win_rate": 0.0, "wins": 0}

        return {
            "trades": trades,
            "wins": int(REGEX_WINS.search(output).group(1)),
            "win_rate": float(REGEX_WIN_RATE.search(output).group(1)),
            "profit": float(REGEX_PROFIT.search(output).group(1))
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Cargo execution failed: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during backtest IPC: {e}")
        return None

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function. 
    Defines the search space and returns the metric to maximize (profit).
    """
    # Define continuous and discrete search spaces
    params = {
        "threshold": trial.suggest_float("threshold", 0.45, 0.65, step=0.01),
        "trend_length": trial.suggest_int("trend_length", 2, 12),
        "momentum_reward": trial.suggest_float("momentum_reward", 0.0, 0.05, step=0.01),
        "volatility_penalty": trial.suggest_float("volatility_penalty", 0.0, 0.10, step=0.01)
    }
    
    results = run_rust_backtest(params)
    
    # Aggressively prune unviable parameters (zero/low trades) to save compute
    if not results or results["trades"] < 5:
        raise optuna.TrialPruned("Insufficient trades generated.")
        
    # Prune negative mathematical expectancy
    if results["win_rate"] < 35.0:
        raise optuna.TrialPruned("Win rate below statistical viability.")

    # Attach rich metadata to the trial for post-analysis
    trial.set_user_attr("trades", results["trades"])
    trial.set_user_attr("win_rate", results["win_rate"])
    trial.set_user_attr("wins", results["wins"])

    return results["profit"]

def main():
    logger.info("Initializing Intelligent Bayesian Optimization Pipeline...")
    
    output_dir = Path("data/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TPESampler (Tree-structured Parzen Estimator) uses probability density to focus on promising areas
    study = optuna.create_study(
        study_name="hope_strategy_optimization",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20) 
    )
    
    n_trials = 3
    logger.info(f"Commencing adaptive search for {n_trials} trials...")
    
    try:
        # Execute optimization with built-in progress bar
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted safely. Exporting current state...")

    # Data Consolidation
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    logger.info("-" * 40)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("-" * 40)
    logger.info(f"Total Trials Executed: {len(study.trials)}")
    logger.info(f"Pruned (Dead Zones):   {len(pruned_trials)}")
    logger.info(f"Completed (Viable):    {len(complete_trials)}")

    if len(complete_trials) == 0:
        logger.error("No valid parameters found. Consider widening the search space.")
        return

    # Extract best configuration
    best_trial = study.best_trial
    logger.info("🏆 OPTIMAL PARAMETERS FOUND:")
    logger.info(f"   Net Profit: ${best_trial.value:.2f}")
    logger.info(f"   Win Rate:   {best_trial.user_attrs.get('win_rate', 0.0):.2f}%")
    logger.info(f"   Total Exec: {best_trial.user_attrs.get('trades', 0)} trades")
    
    logger.info("⚙️  TARGET CONFIGURATION:")
    for key, value in best_trial.params.items():
        logger.info(f"   {key.upper()} = {value}")

    # Export structured DataFrame
    df = study.trials_dataframe()
    csv_path = output_dir / "optuna_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"💾 Full analytical dataset saved to: {csv_path}")

    # Export Interactive HTML Visualizations
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
        logger.info("Generating Plotly interactive visualizations...")
        
        fig_hist = plot_optimization_history(study)
        fig_hist.write_html(output_dir / "optimization_history.html")
        
        fig_import = plot_param_importances(study)
        fig_import.write_html(output_dir / "param_importances.html")
        
        fig_contour = plot_contour(study)
        fig_contour.write_html(output_dir / "contour_plot.html")
        
        logger.info(f"📊 Visualizations exported successfully to: {output_dir}")
    except Exception as e:
        logger.warning(f"Visualization skipped (likely due to NumPy/Matplotlib version mismatch): {e}")

if __name__ == "__main__":
    main()