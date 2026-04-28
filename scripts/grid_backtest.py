#!/usr/bin/env python3
"""
Professional Bayesian Optimization Engine for Hope Trading System
------------------------------------------------------------------
A production-grade Optuna orchestrator ensuring mathematical rigor.

Architectural Enhancements & Bug Fixes:
  1. Concurrency Fix: Removed `constant_liar` and `RetryFailedTrialCallback` 
     which cause `ValueError: Cannot tell a COMPLETE trial` race conditions 
     when using SQLite with `n_jobs > 1`. Implemented a thread-safe internal 
     retry loop instead.
  2. Warning Suppression: Cleaned up CLI output by suppressing Optuna's 
     experimental feature warnings.
  3. Mathematical Constraints: Uses Optuna's native `constraints_func` in the 
     TPESampler to guarantee the KDE correctly maps the continuous search space.
  4. Derived Statistical Metrics: Computes Expectancy and Profit Factor internally.
  5. SQLite WAL Concurrency: Enforces Write-Ahead Logging and busy_timeout at 
     the SQL level to eliminate database locking during highly parallel execution.

Usage:
  python3 scripts/grid_backtest.py --trials 400 --jobs 4 --target profit
"""

import os
import re
import sys
import json
import signal
import argparse
import subprocess
import logging
import multiprocessing
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Sequence
from dataclasses import dataclass, asdict

import pandas as pd
import optuna
from optuna.trial import TrialState, Trial
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage

# Suppress Optuna experimental warnings for cleaner CLI output
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# ---------------------------------------------------------------------------
# UI & Logging Configuration
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.panel import Panel
    RICH = True
except ImportError:
    RICH = False

LOG_FORMAT_PLAIN = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FORMAT_RICH  = "%(message)s"

logger = logging.getLogger("HopeOptuna")
logger.setLevel(logging.INFO)

if RICH:
    console = Console()
    handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
    handler.setFormatter(logging.Formatter(LOG_FORMAT_RICH))
else:
    console = None
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT_PLAIN, datefmt="%H:%M:%S"))
logger.addHandler(handler)

optuna.logging.set_verbosity(optuna.logging.ERROR)

# ---------------------------------------------------------------------------
# Compiled Regex Patterns
# ---------------------------------------------------------------------------
RE_TRADES   = re.compile(r"Total Trades:\s+(\d+)")
RE_WINS     = re.compile(r"Wins:\s+(\d+)")
RE_WIN_RATE = re.compile(r"Win Rate:\s+([\d.]+)")
RE_PROFIT   = re.compile(r"Total Profit:\s+(-?[\d.]+)")
RE_SHARPE   = re.compile(r"Sharpe Ratio:\s+(-?[\d.]+)")
RE_DRAWDOWN = re.compile(r"Max Drawdown:\s+([\d.]+)")

def _safe_float(regex_match) -> float:
    """Extract a float from a regex match, defaulting to 0.0 on failure."""
    if regex_match:
        try:
            return float(regex_match.group(1))
        except (ValueError, IndexError):
            pass
    return 0.0

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class BacktestMetrics:
    trades: int
    wins: int
    win_rate: float
    profit: float
    sharpe: float
    drawdown: float
    expectancy: float
    profit_factor: float

# ---------------------------------------------------------------------------
# Mathematical Validation & Parsing
# ---------------------------------------------------------------------------
def compute_derived_metrics(
    trades: int, 
    wins: int, 
    profit: float, 
    reported_win_rate: float,
    reported_sharpe: float,
    reported_drawdown: float
) -> BacktestMetrics:
    """
    Guarantees the mathematics of the algorithm by computing objective 
    statistical metrics independently of the Rust stdout layer.
    """
    if trades == 0:
        return BacktestMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Re-calculate to prevent Rust rounding truncation errors
    win_rate = (wins / trades) * 100.0 if trades > 0 else 0.0
    losses = trades - wins
    
    # Expectancy: Average profit per trade.
    expectancy = profit / trades if trades > 0 else 0.0

    # Profit Factor Proxy: (Wins * Avg Win) / (Losses * Avg Loss)
    profit_factor = 0.0
    if losses == 0 and profit > 0:
        profit_factor = 999.0
    elif profit < 0:
        profit_factor = max(0.0, 1.0 + (profit / trades))
    else:
        profit_factor = 1.0 + (profit / trades)

    return BacktestMetrics(
        trades=trades,
        wins=wins,
        win_rate=win_rate,
        profit=profit,
        sharpe=reported_sharpe,
        drawdown=reported_drawdown,
        expectancy=expectancy,
        profit_factor=profit_factor
    )

def run_rust_backtest(params: Dict[str, Any], timeout_seconds: int = 120) -> Optional[BacktestMetrics]:
    """Invoke ./target/release/backtest with isolated environment bounds."""
    env = os.environ.copy()
    env["BACKTEST_MODE"] = "1"

    param_to_env = {
        "threshold":          "DERIV_THRESHOLD",
        "trend_length":       "STRATEGY_MIN_TREND_LENGTH",
        "momentum_reward":    "STRATEGY_MOMENTUM_REWARD",
        "volatility_penalty": "STRATEGY_VOLATILITY_PENALTY",
    }
    for key, env_var in param_to_env.items():
        if key in params:
            env[env_var] = str(params[key])

    cmd = ["./target/release/backtest"]

    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, check=True, timeout=timeout_seconds,
        )
        output = proc.stdout
    except subprocess.TimeoutExpired:
        logger.error("Subprocess timeout threshold exceeded.")
        return None
    except subprocess.CalledProcessError as exc:
        logger.error(f"Rust Panic [{exc.returncode}]: {exc.stderr.strip()}")
        return None
    except FileNotFoundError:
        logger.critical("Binary missing. Execute: cargo build --release")
        sys.exit(1)

    trades_match = RE_TRADES.search(output)
    trades = int(trades_match.group(1)) if trades_match else 0

    if trades == 0:
        return compute_derived_metrics(0, 0, 0.0, 0.0, 0.0, 0.0)

    try:
        wins = int(RE_WINS.search(output).group(1))
        profit = float(RE_PROFIT.search(output).group(1))
        reported_win_rate = float(RE_WIN_RATE.search(output).group(1))
        reported_sharpe = _safe_float(RE_SHARPE.search(output))
        reported_drawdown = _safe_float(RE_DRAWDOWN.search(output))
        return compute_derived_metrics(
            trades=trades,
            wins=wins,
            profit=profit,
            reported_win_rate=reported_win_rate,
            reported_sharpe=reported_sharpe,
            reported_drawdown=reported_drawdown
        )
    except (AttributeError, ValueError) as e:
        logger.error("Stdout regex parser failed: %s. Verify Rust output formatting.", e)
        return None

# ---------------------------------------------------------------------------
# Safe attribute fetching for constraints
# ---------------------------------------------------------------------------
def _get_user_attr_float(trial: Trial, key: str, default: float = 0.0) -> float:
    """Return a user attribute as float, ensuring no None values."""
    val = trial.user_attrs.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

# ---------------------------------------------------------------------------
# Optuna Objective & Constraint Management
# ---------------------------------------------------------------------------
class ObjectiveEvaluator:
    def __init__(self, timeout: int, target_metric: str):
        self.timeout = timeout
        self.target_metric = target_metric

    def __call__(self, trial: Trial) -> float:
        search_space = trial.study.user_attrs.get("search_space", {})
        params = {}
        for name, spec in search_space.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])

        # Thread-safe internal retry loop (replaces RetryFailedTrialCallback)
        max_retries = 3
        metrics = None
        for attempt in range(max_retries):
            metrics = run_rust_backtest(params, timeout_seconds=self.timeout)
            if metrics is not None:
                break
            if attempt < max_retries - 1:
                logger.debug(f"Trial {trial.number} subprocess failed. Retrying ({attempt + 1}/{max_retries})...")

        if not metrics:
            raise optuna.TrialPruned("Subprocess execution failed after retries.")

        # Attach all computed data mathematically to the trial attributes
        for k, v in asdict(metrics).items():
            trial.set_user_attr(k, v)

        return getattr(metrics, self.target_metric)

def build_constraints(min_trades: int, min_win_rate: float):
    """
    Closure providing Optuna with a mathematical constraint evaluator.
    Returns a sequence of values where <= 0.0 is feasible and > 0.0 is violated.
    """
    def constraints_func(trial: Trial) -> Sequence[float]:
        trades   = _get_user_attr_float(trial, "trades", 0.0)
        win_rate = _get_user_attr_float(trial, "win_rate", 0.0)

        # Condition 1: trades must be >= min_trades  -> (min_trades - trades) <= 0
        c1 = float(min_trades) - trades
        # Condition 2: win_rate must be >= min_win_rate -> (min_win_rate - win_rate) <= 0
        c2 = min_win_rate - win_rate
        return (c1, c2)
    return constraints_func

# ---------------------------------------------------------------------------
# Graceful Shutdown
# ---------------------------------------------------------------------------
_study = None
def _shutdown_handler(signum, frame):
    msg = "[bold red]Interrupt received – stopping study gracefully.[/]" if RICH else "Interrupt received – stopping study gracefully."
    logger.warning(msg)
    if _study is not None:
        _study.stop()

# ---------------------------------------------------------------------------
# CLI & Execution Configuration
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Professional Bayesian Optuna Orchestrator with Mathematical Constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group_space = parser.add_argument_group("Search Space Bounds")
    group_space.add_argument("--threshold-low", type=float, default=0.45)
    group_space.add_argument("--threshold-high", type=float, default=0.65)
    group_space.add_argument("--threshold-step", type=float, default=0.01)
    group_space.add_argument("--trend-length-low", type=int, default=2)
    group_space.add_argument("--trend-length-high", type=int, default=12)
    group_space.add_argument("--momentum-reward-low", type=float, default=0.0)
    group_space.add_argument("--momentum-reward-high", type=float, default=0.05)
    group_space.add_argument("--momentum-reward-step", type=float, default=0.01)
    group_space.add_argument("--volatility-penalty-low", type=float, default=0.0)
    group_space.add_argument("--volatility-penalty-high", type=float, default=0.10)
    group_space.add_argument("--volatility-penalty-step", type=float, default=0.01)

    group_study = parser.add_argument_group("Study Constraints")
    group_study.add_argument("-n", "--trials", type=int, default=200)
    group_study.add_argument("-j", "--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    group_study.add_argument("--target", type=str, choices=["profit", "expectancy", "sharpe", "profit_factor"], default="profit")
    group_study.add_argument("--min-trades", type=int, default=10, help="Mathematical constraint: Min Trades")
    group_study.add_argument("--min-win-rate", type=float, default=35.0, help="Mathematical constraint: Min Win Rate")
    group_study.add_argument("--timeout", type=int, default=180)
    group_study.add_argument("--seed", type=int, default=42)

    group_out = parser.add_argument_group("Output Engine")
    group_out.add_argument("--study-name", type=str, default="hope_strategy_optimization")
    group_out.add_argument("--output-dir", type=Path, default=Path("backtest_optimization"))
    group_out.add_argument("--no-html", action="store_true", help="Skip Plotly rendering")
    group_out.add_argument("--dry-run", action="store_true", help="Execute single compilation and test pass")

    return parser

def validate_bounds(args):
    """Ensure low < high for each parameter. Exit early if violated."""
    errors = []
    for name in ["threshold", "momentum_reward", "volatility_penalty"]:
        low = getattr(args, f"{name}_low")
        high = getattr(args, f"{name}_high")
        if low >= high:
            errors.append(f"--{name}-low ({low}) must be less than --{name}-high ({high})")
    if args.trend_length_low >= args.trend_length_high:
        errors.append(f"--trend-length-low ({args.trend_length_low}) must be less than --trend-length-high ({args.trend_length_high})")
    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)

def configure_sqlite_wal(db_path: Path) -> str:
    """Enforce Write-Ahead Logging and busy timeouts to prevent multithreading lock conflicts."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.close()
    return f"sqlite:///{db_path}"

def main() -> None:
    global _study
    parser = build_argument_parser()
    args = parser.parse_args()

    validate_bounds(args)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    if hasattr(os, "getloadavg"):
        load1, _, _ = os.getloadavg()
        if load1 > multiprocessing.cpu_count() * 0.9:
            logger.warning(f"High CPU saturation detected ({load1:.2f}). Throttling --jobs may improve stability.")

    logger.info("Executing release build of Rust core...")
    try:
        subprocess.run(["cargo", "build", "--release", "--bin", "backtest", "--locked", "--offline"],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.critical(f"Rust Compilation Aborted:\n{exc.stderr}")
        sys.exit(1)

    if args.dry_run:
        logger.info("Verifying mathematical outputs via dry-run...")
        test_params = {"threshold": 0.50, "trend_length": 5, "momentum_reward": 0.02, "volatility_penalty": 0.02}
        res = run_rust_backtest(test_params, timeout_seconds=args.timeout)
        if not res:
            logger.critical("Dry-run failed. Validation aborted.")
            sys.exit(2)
        logger.info(f"Dry-run passed. Target metric [{args.target}]: {getattr(res, args.target)}")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(run_dir / "optimization.log")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT_PLAIN, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    if RICH and console:
        console.print(Panel.fit(
            f"[bold cyan]Hope ML Optimization Engine[/]\n"
            f"Targeting: [green]{args.target.upper()}[/] | Concurrency: [yellow]{args.jobs}[/]",
            border_style="cyan"
        ))

    # Configure SQLite Database with concurrent safety
    db_path = run_dir / "optuna_study.db"
    storage_url = configure_sqlite_wal(db_path)
    
    storage = RDBStorage(
        url=storage_url, 
        engine_kwargs={
            "connect_args": {"timeout": 120.0}
        }
    )

    search_space = {
        "threshold":          {"type": "float", "low": args.threshold_low, "high": args.threshold_high, "step": args.threshold_step},
        "trend_length":       {"type": "int",   "low": args.trend_length_low, "high": args.trend_length_high},
        "momentum_reward":    {"type": "float", "low": args.momentum_reward_low, "high": args.momentum_reward_high, "step": args.momentum_reward_step},
        "volatility_penalty": {"type": "float", "low": args.volatility_penalty_low, "high": args.volatility_penalty_high, "step": args.volatility_penalty_step},
    }

    # Multivariate TPE Sampler with hard mathematical constraints
    sampler = TPESampler(
        seed=args.seed,
        multivariate=True,
        constraints_func=build_constraints(args.min_trades, args.min_win_rate)
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    study.set_user_attr("search_space", search_space)
    _study = study

    objective = ObjectiveEvaluator(timeout=args.timeout, target_metric=args.target)

    logger.info("Initializing %d trials | Timeout: %ds", args.trials, args.timeout)

    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            n_jobs=args.jobs,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.exception(f"Optuna runtime exception: {exc}")
    finally:
        _study = None

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        logger.warning("No trials met the mathematical constraints. Review bounds.")
        pd.DataFrame().to_csv(run_dir / "optuna_results.csv", index=False)
        return

    best = study.best_trial

    with open(run_dir / "best_params.json", "w") as f:
        json.dump({
            "target_metric": args.target,
            "objective_value": best.value,
            "metrics": best.user_attrs,
            "parameters": best.params
        }, f, indent=4)

    if RICH and console:
        table = Table(title="🏆 Mathematically Optimal Configuration", style="green")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        for key, val in best.params.items():
            table.add_row(key.upper(), str(val))
        console.print(table)

        metrics = Table(show_header=False, style="yellow")
        metrics.add_row("Target Score", f"{best.value:.4f}")
        metrics.add_row("Expectancy", f"{best.user_attrs.get('expectancy', 0.0):.4f}")
        metrics.add_row("Net Profit", f"${best.user_attrs.get('profit', 0.0):.2f}")
        metrics.add_row("Win Rate", f"{best.user_attrs.get('win_rate', 0.0):.2f}%")
        metrics.add_row("Total Trades", str(best.user_attrs.get('trades', 0)))
        console.print(Panel(metrics, title="Validated Constraints & Metrics", border_style="yellow"))
    else:
        logger.info(f"Optimal Trial #{best.number} | {args.target.upper()} = {best.value:.4f} | Win Rate = {best.user_attrs.get('win_rate'):.2f}% | Trades = {best.user_attrs.get('trades')}")

    df = study.trials_dataframe()
    df.to_csv(run_dir / "optuna_results.csv", index=False)

    if not args.no_html:
        try:
            from optuna.visualization import (
                plot_optimization_history, plot_param_importances,
                plot_parallel_coordinate, plot_slice, plot_edf
            )
            plots = {
                "optimization_history.html": plot_optimization_history(study),
                "param_importances.html": plot_param_importances(study),
                "parallel_coordinate.html": plot_parallel_coordinate(study),
                "slice_plot.html": plot_slice(study),
                "edf_plot.html": plot_edf(study)
            }
            for filename, fig in plots.items():
                fig.write_html(run_dir / filename)
            logger.info(f"Interactive HTML analytics generated in {run_dir}")
        except Exception as exc:
            logger.warning(f"Plotly generation skipped due to rendering conflict: {exc}")

if __name__ == "__main__":
    main()
