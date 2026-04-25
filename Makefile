# Commands
PYTHON ?= python3
CARGO  ?= cargo

.PHONY: help fmt check test verify run backtest export collect consolidate setup clean

# Default target
help:
	@echo "Hope Trading Engine - Development Commands"
	@echo "=========================================="
	@echo "fmt         : Format source code"
	@echo "check       : Run cargo check (offline)"
	@echo "test        : Run unit tests (offline)"
	@echo "verify      : Format, check, and test everything"
	@echo "run         : Start the trading engine"
	@echo "backtest    : Run strategy backtesting on data/ticks.csv"
	@echo "export      : Export ticks from SQLite to data/ticks.csv"
	@echo "collect     : Collect historical ticks from Deriv API"
	@echo "consolidate : Generate an audit snapshot of the project"
	@echo "setup       : Install Python dependencies from requirements.txt"
	@echo "clean       : Remove audit logs and temporary artifacts"

fmt:
	$(CARGO) fmt

check:
	$(CARGO) check --offline

test:
	$(CARGO) test --offline

verify: fmt check test

run:
	$(CARGO) run --offline

backtest:
	$(CARGO) run --bin backtest --offline

train:
	@echo "NOTICE: Invoking scripts/train_fixed.py directly on a local machine is prohibited by project policy (see AGENTS.md)."
	@echo "Upload data/ticks.csv to Google Colab or Kaggle and execute notebooks/train_transformer.ipynb in a cloud GPU environment."
	@echo "All training scripts contain runtime guards that abort execution if a cloud environment is not detected."

export:
	$(PYTHON) scripts/export_db.py

collect:
	$(PYTHON) scripts/tick_collector.py --hours 24

consolidate:
	$(PYTHON) consolidate_project_sources.py

setup:
	$(PYTHON) -m pip install -r requirements.txt

clean:
	@echo "Removing logs and snapshots only."
	rm -f tick_audit.log *_merged_sources.txt project_snapshot.txt
	$(CARGO) clean

clean-all:
	@echo "WARNING: permanently deleting locally collected data/ticks.csv"
	@echo "You will need to re-run make collect and make export to restore the dataset."
	rm -f tick_audit.log *_merged_sources.txt data/ticks.csv project_snapshot.txt
	$(CARGO) clean
