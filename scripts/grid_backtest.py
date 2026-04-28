#!/usr/bin/env python3
"""
Professional Bayesian Optimisation Runner for Hope Backtesting
---------------------------------------------------------------
Wraps the Rust backtest binary inside an Optuna study.
Features:
  - Full argparse CLI for search space, trials, output, and timeouts.
  - Graceful shutdown on SIGINT/SIGTERM – study state is saved.
  - Sub‑process timeouts prevent zombie/hung backtests.
  - SQLite storage enables resuming interrupted studies.
  - Timestamped output directory with CSV results and HTML plots.
  - Dry‑run mode to validate the Rust binary before launching trials.

Usage:
  python grid_backtest.py --trials 100 --output-dir ./my_results
"""

import os
import re
import sys
import signal
import argparse
import subprocess
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler

# ---------------------------------------------------------------------------
# Logging setup – structured, with optional file output.
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("HopeOptuna")

# Mute Optuna’s internal trial logs – we report only curated information.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Compiled regex patterns (extract metrics from Rust binary output)
# ---------------------------------------------------------------------------
RE_TRADES   = re.compile(r"Total Trades:\s+(\d+)")
RE_WINS     = re.compile(r"Wins:\s+(\d+)")
RE_WIN_RATE = re.compile(r"Win Rate:\s+([\d.]+)")
RE_PROFIT   = re.compile(r"Total Profit:\s+(-?[\d.]+)")


class BacktestError(Exception):
    """Raised when the backtest binary fails in an unrecoverable way."""


# ---------------------------------------------------------------------------
# Backtest runner – isolated subprocess with timeout and error handling
# ---------------------------------------------------------------------------
def run_rust_backtest(
    params: Dict[str, Any],
    timeout_seconds: int = 120,
) -> Optional[Dict[str, Any]]:
    """
    Execute the Rust backtest binary with the given parameters injected
    via environment variables.

    Returns a dictionary with metrics or None on failure.
    """
    env = os.environ.copy()
    env["BACKTEST_MODE"] = "1"

    # Map Optuna trial parameters to the environment variables the Rust
    # binary expects.
    param_to_env = {
        "threshold":           "DERIV_THRESHOLD",
        "trend_length":        "STRATEGY_MIN_TREND_LENGTH",
        "momentum_reward":     "STRATEGY_MOMENTUM_REWARD",
        "volatility_penalty":  "STRATEGY_VOLATILITY_PENALTY",
    }
    for param_name, env_var in param_to_env.items():
        if param_name in params:
            env[env_var] = str(params[param_name])

    # Build the command. We use the compiled binary directly rather than
    # `cargo run` to prevent zombie processes if the timeout is reached
    # (Cargo might not forward SIGKILL to the child process).
    cmd = ["./target/release/backtest"]

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_seconds,
        )
        output = proc.stdout
    except subprocess.TimeoutExpired:
        logger.error("Backtest subprocess timed out after %d s.", timeout_seconds)
        return None
    except subprocess.CalledProcessError as exc:
        logger.error("Backtest binary failed (exit code %d): %s", exc.returncode, exc.stderr.strip())
        return None
    except FileNotFoundError:
        logger.critical(
            "Backtest binary `./target/release/backtest` not found. Make sure it compiled successfully."
        )
        sys.exit(1)

    # Parse output
    trades_match = RE_TRADES.search(output)
    trades = int(trades_match.group(1)) if trades_match else 0

    if trades == 0:
        return {"trades": 0, "profit": 0.0, "win_rate": 0.0, "wins": 0}

    try:
        return {
            "trades":   trades,
            "wins":     int(RE_WINS.search(output).group(1)),
            "win_rate": float(RE_WIN_RATE.search(output).group(1)),
            "profit":   float(RE_PROFIT.search(output).group(1)),
        }
    except (AttributeError, ValueError):
        logger.error("Could not parse backtest output. Raw output:\n%s", output)
        return None


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
class Objective:
    """Callable objective with a reference to the backtest timeout."""

    def __init__(self, timeout: int):
        self.timeout = timeout

    def __call__(self, trial: optuna.Trial) -> float:
        # Parameter space – matches original logic, now configurable via CLI
        # but bounds are taken from the search_space dict passed at study creation.
        # Here we extract them from the trial’s study user attrs.
        search_space = trial.study.user_attrs.get("search_space", {})
        params = {}
        for name, spec in search_space.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], step=spec.get("step", None)
                )
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"]
                )

        results = run_rust_backtest(params, timeout_seconds=self.timeout)

        if not results or results["trades"] < 5:
            raise optuna.TrialPruned("Insufficient trades generated.")

        if results["win_rate"] < 35.0:
            raise optuna.TrialPruned("Win rate below statistical viability.")

        # Attach extra info to the trial for later analysis
        trial.set_user_attr("trades", results["trades"])
        trial.set_user_attr("win_rate", results["win_rate"])
        trial.set_user_attr("wins", results["wins"])

        return results["profit"]


# ---------------------------------------------------------------------------
# Signal handling – ensures graceful shutdown
# ---------------------------------------------------------------------------
_study = None
_shutdown_requested = False

def _shutdown_handler(signum, frame):
    global _shutdown_requested
    logger.warning("Received signal %s – requesting orderly shutdown.", signum)
    _shutdown_requested = True
    if _study is not None:
        _study.stop()   # tells Optuna to stop after the current trial


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimise Hope trading strategy hyper‑parameters with Optuna.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Parameter space definition
    parser.add_argument(
        "--threshold-low", type=float, default=0.45,
        help="Lower bound for DERIV_THRESHOLD (default: 0.45)"
    )
    parser.add_argument(
        "--threshold-high", type=float, default=0.65,
        help="Upper bound for DERIV_THRESHOLD (default: 0.65)"
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.01,
        help="Step size for DERIV_THRESHOLD (default: 0.01)"
    )
    parser.add_argument(
        "--trend-length-low", type=int, default=2,
        help="Min STRATEGY_MIN_TREND_LENGTH (default: 2)"
    )
    parser.add_argument(
        "--trend-length-high", type=int, default=12,
        help="Max STRATEGY_MIN_TREND_LENGTH (default: 12)"
    )
    parser.add_argument(
        "--momentum-reward-low", type=float, default=0.0,
        help="Lower bound for STRATEGY_MOMENTUM_REWARD (default: 0.0)"
    )
    parser.add_argument(
        "--momentum-reward-high", type=float, default=0.05,
        help="Upper bound for STRATEGY_MOMENTUM_REWARD (default: 0.05)"
    )
    parser.add_argument(
        "--momentum-reward-step", type=float, default=0.01,
        help="Step size for STRATEGY_MOMENTUM_REWARD (default: 0.01)"
    )
    parser.add_argument(
        "--volatility-penalty-low", type=float, default=0.0,
        help="Lower bound for STRATEGY_VOLATILITY_PENALTY (default: 0.0)"
    )
    parser.add_argument(
        "--volatility-penalty-high", type=float, default=0.10,
        help="Upper bound for STRATEGY_VOLATILITY_PENALTY (default: 0.10)"
    )
    parser.add_argument(
        "--volatility-penalty-step", type=float, default=0.01,
        help="Step size for STRATEGY_VOLATILITY_PENALTY (default: 0.01)"
    )

    # Study control
    parser.add_argument(
        "-n", "--trials", type=int, default=100,
        help="Number of trials to run (default: 100)"
    )
    parser.add_argument(
        "--study-name", type=str, default="hope_strategy_optimization",
        help="Optuna study name (default: hope_strategy_optimization)"
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help=(
            "Database URL for Optuna persistence; e.g., sqlite:///my_study.db. "
            "If not set, a new in‑memory study is created (state lost on exit)."
        )
    )
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="Timeout in seconds for each backtest invocation (default: 180)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the sampler (default: 42)"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=Path("backtest_optimization"),
        help="Base directory for results; a timestamped subdirectory is created inside."
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip generation of Plotly HTML visualisations."
    )

    # Pre‑flight checks
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Test that the backtest binary can be executed, then exit."
    )

    return parser


# ---------------------------------------------------------------------------
# Pre‑flight check – ensures binary is available without launching a trial
# ---------------------------------------------------------------------------
def dry_run_check(timeout: int) -> None:
    logger.info("Performing dry‑run of the backtest binary …")
    dummy_params = {
        "threshold": 0.5,
        "trend_length": 5,
        "momentum_reward": 0.02,
        "volatility_penalty": 0.02,
    }
    result = run_rust_backtest(dummy_params, timeout_seconds=timeout)
    if result is None:
        logger.critical("Dry‑run failed – cannot communicate with the backtest binary.")
        sys.exit(2)
    logger.info("Dry‑run successful: %s", result)
    print("Dry‑run OK. Binary and parsing are functional.")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    global _study, _shutdown_requested

    parser = build_argument_parser()
    args = parser.parse_args()

    # Set up signal handlers early
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    logger.info("Compiling backtest binary ...")
    try:
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "backtest", "--locked", "--offline"],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as exc:
        logger.critical("Failed to compile the Rust backtest binary:\n%s", exc.stderr)
        sys.exit(1)
    except FileNotFoundError:
        logger.critical("`cargo` not found. Make sure Rust is installed and on PATH.")
        sys.exit(1)

    # --- Dry‑run mode ---
    if args.dry_run:
        dry_run_check(args.timeout)
        return

    # --- Output directory (timestamped sub‑directory) ---
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Add file logging to the run directory
    file_handler = logging.FileHandler(run_dir / "optimisation.log")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    logger.info("=== Hope Strategy Optimisation ===")
    logger.info("Run directory: %s", run_dir)
    logger.info("Trials: %d | Timeout per trial: %d s", args.trials, args.timeout)

    # --- Persist study state if a storage URL is provided ---
    storage = args.storage
    if storage is None:
        # Use SQLite inside the run directory for automatic checkpointing
        db_path = run_dir / "optuna_study.db"
        storage = f"sqlite:///{db_path}"
        logger.info("Using checkpointing storage: %s", storage)

    # --- Build search space definition from CLI ---
    search_space = {
        "threshold": {
            "type": "float",
            "low": args.threshold_low,
            "high": args.threshold_high,
            "step": args.threshold_step,
        },
        "trend_length": {
            "type": "int",
            "low": args.trend_length_low,
            "high": args.trend_length_high,
        },
        "momentum_reward": {
            "type": "float",
            "low": args.momentum_reward_low,
            "high": args.momentum_reward_high,
            "step": args.momentum_reward_step,
        },
        "volatility_penalty": {
            "type": "float",
            "low": args.volatility_penalty_low,
            "high": args.volatility_penalty_high,
            "step": args.volatility_penalty_step,
        },
    }

    # --- Create or load the study ---
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=args.seed, n_startup_trials=20),
        load_if_exists=True,
    )
    study.set_user_attr("search_space", search_space)

    # Allow the signal handler to stop the study
    _study = study

    objective = Objective(timeout=args.timeout)

    # --- Run optimisation with graceful interruption ---
    try:
        study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    except KeyboardInterrupt:
        logger.warning("Optimisation interrupted by user (KeyboardInterrupt).")
    except optuna.exceptions.OptunaError as exc:
        logger.error("Optuna runtime error: %s", exc)
    finally:
        _study = None

    # --- Post‑analysis ---
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == TrialState.FAIL]

    logger.info("--- Optimisation ended ---")
    logger.info("Completed: %d / Pruned: %d / Failed: %d", len(completed), len(pruned), len(failed))

    if not completed:
        logger.warning("No successful trials. Check logs and widen the search space if necessary.")
        # Save empty DataFrames for consistency
        pd.DataFrame().to_csv(run_dir / "optuna_results.csv", index=False)
        return

    best = study.best_trial
    logger.info("Best trial (#%d): Profit = %.2f | Win Rate = %.2f%% | Trades = %d",
                best.number, best.value,
                best.user_attrs.get("win_rate", 0.0),
                best.user_attrs.get("trades", 0))
    logger.info("Best parameters:")
    for key, val in best.params.items():
        logger.info("    %s = %s", key, val)

    # --- Export CSV with all trial data ---
    df = study.trials_dataframe()
    csv_path = run_dir / "optuna_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Trial data exported to %s", csv_path)

    # --- Interactive HTML plots (optional) ---
    if not args.no_html:
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_contour,
            )

            logger.info("Generating HTML visualisations …")

            fig_hist = plot_optimization_history(study)
            fig_hist.write_html(run_dir / "optimization_history.html")

            fig_import = plot_param_importances(study)
            fig_import.write_html(run_dir / "param_importances.html")

            # Contour plots require at least two parameters; handle gracefully
            try:
                fig_contour = plot_contour(study)
                fig_contour.write_html(run_dir / "contour_plot.html")
            except ValueError:
                logger.info("Not enough parameters for contour plot – skipped.")

            logger.info("Visualisations saved under %s", run_dir)
        except ImportError:
            logger.warning(
                "Plotly or other visualisation dependencies missing – HTML plots skipped."
            )
        except Exception as exc:
            logger.exception("Unexpected error while creating visualisations: %s", exc)

    logger.info("All results written to %s", run_dir)


if __name__ == "__main__":
    main()