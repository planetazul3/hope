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
	cargo fmt

check:
	cargo check --offline

test:
	cargo test --offline

verify: fmt check test

run:
	cargo run --bin hope --offline

backtest:
	cargo run --bin backtest --offline

export:
	python3 scripts/export_db.py

collect:
	python3 scripts/tick_collector.py --hours 24

consolidate:
	python3 consolidate_project_sources.py

setup:
	pip install -r requirements.txt

clean:
	rm -f tick_audit.log *_merged_sources.txt data/ticks.csv
	cargo clean
