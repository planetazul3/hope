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
	@echo "Please upload data/ticks.csv and execute the training notebook exclusively in a cloud environment (e.g. Google Colab/Kaggle)."

export:
	$(PYTHON) scripts/export_db.py

collect:
	$(PYTHON) scripts/tick_collector.py --hours 24

consolidate:
	$(PYTHON) consolidate_project_sources.py

setup:
	$(PYTHON) -m pip install -r requirements.txt

clean:
	rm -f tick_audit.log *_merged_sources.txt data/ticks.csv project_snapshot.txt
	$(CARGO) clean
