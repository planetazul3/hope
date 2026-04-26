# ADR 0015: Professional-Grade Data Engineering Infrastructure

## Status

Accepted

## Context

The previous tick collection and export pipeline consisted of basic procedural scripts. They lacked robustness in several key areas:
- **No Symbol Discovery**: Users had to know symbol codes beforehand.
- **Single-Symbol Storage**: Databases were symbol-agnostic, making it difficult to store and research multiple assets.
- **Resource Management**: Scripts did not handle OS signals gracefully, often leaving database connections in a "dirty" state or hanging on exit.
- **Data Integrity**: There was no automated detection of missing ticks (gaps) or duplicate data during ingestion.
- **Export Inefficiency**: Large datasets were processed entirely in memory, and the export process had to be restarted from scratch even for small updates.

## Decision

We have overhauled the data engineering gear into a professional-grade asynchronous service.

1.  **Architecture**: Transitioned to a modular, class-based design in `tick_collector.py` using `asyncio`.
2.  **Discovery**: Implemented a `list` mode that queries the Deriv API for active synthetic indices, providing a professional interface for asset discovery.
3.  **Hybrid Ingestion**: Added a `both` mode that performs a historical backfill followed by a seamless transition to a live WebSocket subscription.
4.  **Storage Engine**: Optimized the SQLite backend:
    - Enabled **Write-Ahead Logging (WAL)** and `synchronous=NORMAL` to support concurrent high-frequency writes and reads.
    - Implemented a multi-symbol schema with composite unique constraints and indexing.
5.  **Robustness**: Implemented `interruptible_sleep` and explicit signal handlers for `SIGINT`/`SIGTERM`, ensuring immediate and safe script termination.
6.  **Data Auditing**: Integrated real-time gap analysis (detecting non-sequential ticks) and summary statistics/histograms into the pipeline.
7.  **High-Performance Exports**: Rebuilt `export_db.py` to use `pandas` chunking and O(1) file seeking for incremental updates to existing CSV/Parquet files.

## Consequences

-   **Scalability**: The system can now manage dozens of symbols in a single database without collisions.
-   **Stability**: Operational reliability is significantly improved through exponential backoff reconnections and graceful shutdowns.
-   **Fidelity**: Data gaps are identified at the point of ingestion, preventing "garbage in, garbage out" scenarios in ML training.
-   **Efficiency**: Incremental exports reduce the time required to update training datasets from minutes to seconds.
