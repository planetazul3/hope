.PHONY: fmt check test run consolidate

fmt:
	cargo fmt

check:
	cargo check --offline

test:
	cargo test --offline

run:
	cargo run

consolidate:
	python3 consolidate_project_sources.py
