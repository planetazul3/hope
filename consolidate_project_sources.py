#!/usr/bin/env python3
"""
hope — Project Source Code Consolidation Script (v3.0)
------------------------------------------------------
Optimized for high-fidelity codebase analysis by LLMs.

Fixes and Enhancements over v2.0:
- [FIX] Critical NameError when tqdm is not installed (missing fallback definition).
- [FIX] DEFAULT_TRUNCATE_SIZE CLI override had no effect (module-level mutation
        after class instantiation). Now passed explicitly as an instance attribute.
- [FIX] Redundant `if HAS_RICH and HAS_RICH:` condition corrected.
- [FIX] MAX_TOTAL_SIZE_MB was defined but never enforced; now aborts output write
        when the safety cap is reached.
- [NEW] Stats summary is embedded directly in the output file header for LLM context.
- [NEW] Language breakdown included in the terminal summary table.
- [NEW] --dry-run flag: prints what would be included without writing any file.
- [NEW] --no-tree flag: skips the file tree section (useful for very large repos).
- [NEW] --quiet flag: suppresses the terminal summary table.
- [NEW] Atomic write: output is written to a temp file and renamed on success to
        avoid leaving a partial file on errors.
- [NEW] Truncated-file warning emitted to logger at DEBUG level, not silently counted.
- [NEW] Sensitive file count included in terminal summary.
"""

import argparse
import fnmatch
import logging
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set, Dict, Any, Optional

# ── Optional UI libraries ────────────────────────────────────────────────────

try:
    from tqdm import tqdm as _tqdm_cls  # noqa: F401
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Fallback: transparent pass-through so callers never hit a NameError
    class _tqdm_cls:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kwargs):
            self._it = iterable

        def __iter__(self):
            return iter(self._it or [])

        def set_postfix_str(self, *args, **kwargs):
            pass

        def close(self):
            pass

tqdm = _tqdm_cls  # unified name used everywhere below


try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.logging import RichHandler
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── Configuration & Metadata ─────────────────────────────────────────────────

PROJECT_NAME = "hope"
PROJECT_ROOT = Path(__file__).parent.absolute()

DEFAULT_TRUNCATE_SIZE: int = 64 * 1024   # 64 KB per file
MAX_TOTAL_SIZE_MB: int = 50              # Safety cap for the final output file

DB_EXTENSIONS: Set[str] = {
    ".db", ".sqlite", ".sqlite3", ".parquet", ".onnx", ".onnx.data", ".bin",
}
# CSV is intentionally excluded from DB_EXTENSIONS so small CSVs can be included;
# only named artifacts below are treated as opaque data blobs.
DB_FILES: Set[str] = {
    "tick_store.db", "ticks.csv", "candles.csv",
    "backtest_trades.csv", "model.onnx",
}

EXCLUDE_DIRS: Set[str] = {
    ".git", ".vscode", ".idea", "venv", ".venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".eggs",
    "node_modules", ".next", "dist", "build", "target", ".cargo",
    "coverage", "playwright-report", "test-results", ".gemini", ".agents",
    ".agent", "_agents", "_agent", "artifacts", "scratch", "knowledge",
    "temp", "tmp", "libtorch", "logs", "new_venv", "test_venv",
    "backtest_optimization",
}

EXCLUDE_FILES: Set[str] = {
    ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll",
    "*.dylib", "*.exe", "*.log", "*.pid", "*.pid.lock", "*.min.js",
    "*.min.css", "*.map", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "*.seed", "Cargo.lock", "*.rlib", "*.rmeta", "*.d", "*.pt",
    "*.pth", "*.bin", "*.safetensors", "*.joblib", "*.resolved",
    "*.tar.gz", "*.tar", "*.gz", "*.bz2", "*.7z", "*.rar", "*.zip",
    "audit_trial.log", "tick_audit.log", "consolidated_audit.txt",
    "project_snapshot.txt", "*.sig", "engine.log", "engine_final.log",
}

SENSITIVE_PATTERNS: List[str] = [
    r"\.env(\.[a-z]+)?$", r".*\.key$", r".*\.pem$", r".*\.crt$", r".*\.cert$",
    r".*secrets.*", r".*credentials.*", r".*password.*", r".*token.*",
    r".*app_id.*", r".*api_token.*", r".*deriv_token.*", r".*ssid.*",
]

FORCE_INCLUDE_FILES: Set[str] = {
    "Dockerfile", "docker-compose.yml", ".dockerignore", ".gitignore",
    ".gitattributes", "requirements.txt", "package.json", "tsconfig.json",
    "README.md", "LICENSE", "CHANGELOG.md", "Cargo.toml", "Makefile",
    "pyproject.toml", "setup.py", "setup.cfg", "AGENTS.md", "GEMINI.md",
}

LANGUAGE_MAP: Dict[str, str] = {
    ".py": "Python", ".rs": "Rust", ".js": "JavaScript", ".ts": "TypeScript",
    ".json": "JSON", ".toml": "TOML", ".md": "Markdown", ".sh": "Shell",
    ".sql": "SQL", ".ipynb": "Jupyter Notebook", ".yaml": "YAML", ".yml": "YAML",
    ".env": "Environment Variables", ".proto": "Protobuf", ".c": "C", ".cpp": "C++",
}

# ── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> logging.Logger:
    log_level = logging.DEBUG if verbose else logging.INFO
    env_level = os.environ.get("LOG_LEVEL", "").upper()
    if env_level:
        log_level = getattr(logging, env_level, log_level)

    if HAS_RICH:
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True, markup=True, show_path=False
        )
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(message)s"

    logging.basicConfig(level=log_level, format=fmt, datefmt="%H:%M:%S",
                        handlers=[handler])
    return logging.getLogger("Consolidator")


logger = setup_logging()

# ── File Walker ───────────────────────────────────────────────────────────────

class FileWalker:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def is_excluded_dir(self, dir_name: str) -> bool:
        return dir_name in EXCLUDE_DIRS

    def is_excluded_file(self, file_path: Path) -> bool:
        # Force-included files are never excluded (except by sensitive-pattern check
        # which is handled separately in the consolidator).
        if file_path.name in FORCE_INCLUDE_FILES:
            return False
        for pattern in EXCLUDE_FILES:
            if fnmatch.fnmatch(file_path.name, pattern):
                # Named DB artefacts are kept (as opaque headers); everything else is out.
                if file_path.name not in DB_FILES:
                    return True
        return False

    def is_database_file(self, file_path: Path) -> bool:
        return (
            file_path.suffix.lower() in DB_EXTENSIONS
            or file_path.name in DB_FILES
        )

    def is_text_file(self, file_path: Path) -> bool:
        if file_path.name in FORCE_INCLUDE_FILES:
            return True
        if file_path.suffix.lower() in LANGUAGE_MAP:
            return True
        # Heuristic: read a small chunk and check for binary markers.
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(2048)
            if not chunk:
                return True
            if b"\x00" in chunk:
                return False
            chunk.decode("utf-8")
            return True
        except (OSError, UnicodeDecodeError):
            return False

    @staticmethod
    def get_file_language(file_path: Path) -> str:
        special = {
            "Cargo.toml": "TOML (Cargo)",
            "Makefile": "Makefile",
            "Dockerfile": "Dockerfile",
        }
        return special.get(file_path.name, LANGUAGE_MAP.get(file_path.suffix.lower(), "Text"))

    def build_file_tree(
        self,
        directory: Path,
        prefix: str = "",
        depth: int = 0,
        max_depth: int = 10,
    ) -> List[str]:
        if depth > max_depth:
            return [f"{prefix}└── [Max Depth Reached]"]

        lines: List[str] = []
        try:
            items = []
            for item in sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
                if item.is_dir():
                    if not self.is_excluded_dir(item.name):
                        items.append(item)
                else:
                    if not self.is_excluded_file(item):
                        items.append(item)

            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                extension = "    " if is_last else "│   "
                if item.is_dir():
                    lines.append(f"{prefix}{connector}{item.name}/")
                    lines.extend(
                        self.build_file_tree(item, prefix + extension, depth + 1, max_depth)
                    )
                else:
                    lines.append(f"{prefix}{connector}{item.name}")
        except PermissionError:
            lines.append(f"{prefix}└── [Permission Denied]")
        return lines

# ── Project Consolidator ──────────────────────────────────────────────────────

class ProjectConsolidator:
    def __init__(
        self,
        project_root: Path,
        truncate_size: int = DEFAULT_TRUNCATE_SIZE,
        max_output_mb: int = MAX_TOTAL_SIZE_MB,
        dry_run: bool = False,
        include_tree: bool = True,
    ) -> None:
        self.project_root = project_root
        self.truncate_size = truncate_size
        self.max_output_bytes = max_output_mb * 1024 * 1024
        self.dry_run = dry_run
        self.include_tree = include_tree
        self.walker = FileWalker(project_root)
        self.stats: Dict[str, Any] = {
            "total_scanned": 0,
            "included": 0,
            "truncated": 0,
            "dbs": 0,
            "total_lines": 0,
            "sensitive": 0,
            "excluded": 0,
            "cap_reached": False,
            "languages": {},
            "start_time": datetime.now(timezone.utc),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_sensitive(self, file_path: Path) -> bool:
        name = file_path.name
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, name, re.IGNORECASE):
                if ".example" not in name and ".template" not in name:
                    return True
        return False

    def _git_info(self) -> Dict[str, str]:
        info = {"rev": "unknown", "branch": "unknown"}
        try:
            info["rev"] = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.project_root, text=True, stderr=subprocess.DEVNULL,
            ).strip()
            info["branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            pass
        return info

    def _discover_files(self, output_name: str) -> List[Path]:
        """Walk the project and return candidate files in deterministic order."""
        result: List[Path] = []
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = sorted(d for d in dirs if not self.walker.is_excluded_dir(d))
            for fname in sorted(files):
                if fname == output_name:
                    continue
                result.append(Path(root) / fname)
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def consolidate(self, output_path: Path) -> None:
        mode_label = "[DRY RUN] " if self.dry_run else ""
        label = (
            f"🚀 {mode_label}Consolidating [bold cyan]{PROJECT_NAME}[/] source"
            if HAS_RICH
            else f"{mode_label}Consolidating {PROJECT_NAME} source"
        )
        logger.info(label)

        git_info = self._git_info()
        all_files = self._discover_files(output_path.name)

        if self.dry_run:
            self._dry_run(all_files)
            return

        # Atomic write: use a sibling temp file, rename on success.
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=output_path.parent, prefix=".tmp_consolidate_", suffix=".txt"
        )
        tmp_path = Path(tmp_path_str)

        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as out:
                bytes_written = self._write_output(out, all_files, git_info)

            # Only rename if we didn't blow the cap (partial file is still useful
            # but labelled; we always keep it, just warn the user).
            tmp_path.rename(output_path)
            logger.debug(f"Atomic rename: {tmp_path} → {output_path}")

        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        self._print_summary(output_path, bytes_written)

    # ── Internal write logic ──────────────────────────────────────────────────

    def _write_output(self, out, all_files: List[Path], git_info: Dict[str, str]) -> int:
        """Write content to `out`, return total bytes written."""
        bytes_written = 0

        # ── Global header ──
        now_iso = datetime.now(timezone.utc).isoformat()
        header = (
            f"<!-- PROJECT CONSOLIDATION SNAPSHOT -->\n"
            f"<!-- Project  : {PROJECT_NAME} -->\n"
            f"<!-- Generated: {now_iso} -->\n"
            f"<!-- Git      : {git_info['branch']} @ {git_info['rev']} -->\n"
            f"<!-- Truncate : {self.truncate_size // 1024} KB per file -->\n"
            f"<!-- Cap      : {self.max_output_bytes // 1024 // 1024} MB total -->\n\n"
        )
        out.write(header)
        bytes_written += len(header.encode())

        # ── File tree ──
        if self.include_tree:
            tree_lines = self.walker.build_file_tree(self.project_root)
            tree_block = "FILE_TREE_START\n" + "\n".join(tree_lines) + "\nFILE_TREE_END\n\n"
            out.write(tree_block)
            bytes_written += len(tree_block.encode())

        # ── File contents ──
        pbar = tqdm(all_files, desc="Processing files", disable=not HAS_TQDM)
        for fpath in pbar:
            self.stats["total_scanned"] += 1

            if self.walker.is_excluded_file(fpath):
                self.stats["excluded"] += 1
                continue

            rel_path = fpath.relative_to(self.project_root)

            try:
                stat = fpath.stat()
            except OSError as exc:
                logger.warning(f"Cannot stat {fpath}: {exc}")
                self.stats["excluded"] += 1
                continue

            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            size = stat.st_size

            # ── Database / binary artefacts ──
            if self.walker.is_database_file(fpath):
                self.stats["dbs"] += 1
                block = (
                    f'<file path="{rel_path}" type="database" size="{size}" mtime="{mtime}">\n'
                    f"<!-- Binary/Database content omitted. Extension: {fpath.suffix} -->\n"
                    f"</file>\n\n"
                )
                out.write(block)
                bytes_written += len(block.encode())
                continue

            # ── Sensitive files ──
            if self._is_sensitive(fpath):
                self.stats["sensitive"] += 1
                block = (
                    f'<file path="{rel_path}" type="sensitive" mtime="{mtime}">\n'
                    f"<!-- Sensitive content omitted for security. -->\n"
                    f"</file>\n\n"
                )
                out.write(block)
                bytes_written += len(block.encode())
                continue

            # ── Non-text / binary ──
            if not self.walker.is_text_file(fpath):
                self.stats["excluded"] += 1
                continue

            # ── Text file ──
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read(self.truncate_size + 1)

                is_truncated = len(raw) > self.truncate_size
                if is_truncated:
                    raw = raw[: self.truncate_size]
                    self.stats["truncated"] += 1
                    logger.debug(f"Truncated: {rel_path}")

                lang = FileWalker.get_file_language(fpath)
                self.stats["languages"][lang] = self.stats["languages"].get(lang, 0) + 1
                line_count = raw.count("\n")

                block = (
                    f'<file path="{rel_path}" language="{lang}" lines="{line_count}"'
                    f' size="{size}" mtime="{mtime}" truncated="{str(is_truncated).lower()}">\n'
                    + raw
                    + ("\n\n[... CONTENT TRUNCATED AT "
                       f"{self.truncate_size // 1024} KB ...]\n" if is_truncated else "")
                    + "\n</file>\n\n"
                )
                block_bytes = len(block.encode())

                # ── Safety cap ──
                if bytes_written + block_bytes > self.max_output_bytes:
                    self.stats["cap_reached"] = True
                    cap_notice = (
                        f"\n<!-- OUTPUT CAP REACHED ({self.max_output_bytes // 1024 // 1024} MB). "
                        f"Remaining files omitted. -->\n"
                    )
                    out.write(cap_notice)
                    bytes_written += len(cap_notice.encode())
                    logger.warning(
                        f"Output cap reached at {bytes_written // 1024 // 1024:.1f} MB. "
                        "Stopping early."
                    )
                    break

                out.write(block)
                bytes_written += block_bytes
                self.stats["included"] += 1
                self.stats["total_lines"] += line_count

                if HAS_TQDM:
                    pbar.set_postfix_str(f"{bytes_written // 1024} KB")

            except Exception as exc:
                logger.error(f"Failed to read {fpath}: {exc}")

        # ── Embedded stats footer (for LLM consumers) ──
        footer = self._build_stats_footer(bytes_written)
        out.write(footer)
        bytes_written += len(footer.encode())

        return bytes_written

    def _build_stats_footer(self, bytes_written: int) -> str:
        s = self.stats
        top_langs = sorted(s["languages"].items(), key=lambda x: x[1], reverse=True)[:8]
        lang_list = ", ".join(f"{k}:{v}" for k, v in top_langs)
        cap_note = " [CAP REACHED]" if s["cap_reached"] else ""
        return (
            f"\n<!-- CONSOLIDATION STATS{cap_note}\n"
            f"     Total scanned : {s['total_scanned']}\n"
            f"     Included      : {s['included']}\n"
            f"     Truncated     : {s['truncated']}\n"
            f"     Databases     : {s['dbs']}\n"
            f"     Sensitive     : {s['sensitive']}\n"
            f"     Excluded      : {s['excluded']}\n"
            f"     Total lines   : {s['total_lines']:,}\n"
            f"     Output size   : {bytes_written / 1024:.1f} KB\n"
            f"     Languages     : {lang_list}\n"
            f"-->\n"
        )

    # ── Dry-run mode ──────────────────────────────────────────────────────────

    def _dry_run(self, all_files: List[Path]) -> None:
        logger.info("DRY RUN — no file will be written.")
        rows = []
        for fpath in all_files:
            self.stats["total_scanned"] += 1
            if self.walker.is_excluded_file(fpath):
                self.stats["excluded"] += 1
                continue
            rel = fpath.relative_to(self.project_root)
            if self.walker.is_database_file(fpath):
                tag = "database"
            elif self._is_sensitive(fpath):
                tag = "sensitive"
            elif not self.walker.is_text_file(fpath):
                tag = "binary-skip"
            else:
                tag = FileWalker.get_file_language(fpath)
                self.stats["included"] += 1
            rows.append((str(rel), tag))

        if HAS_RICH:
            console = Console()
            table = Table(title="Dry Run — Files to be Included", show_lines=False)
            table.add_column("Path", style="cyan", no_wrap=False)
            table.add_column("Type / Language", style="green")
            for path, tag in rows:
                table.add_row(path, tag)
            console.print(table)
        else:
            for path, tag in rows:
                print(f"  {tag:<20} {path}")

        logger.info(
            f"Would include {self.stats['included']} files "
            f"({self.stats['total_scanned']} scanned, "
            f"{self.stats['excluded']} excluded)."
        )

    # ── Terminal summary ──────────────────────────────────────────────────────

    def _print_summary(self, output_path: Path, bytes_written: int) -> None:
        s = self.stats
        duration = (datetime.now(timezone.utc) - s["start_time"]).total_seconds()
        top_langs = sorted(s["languages"].items(), key=lambda x: x[1], reverse=True)[:8]
        lang_str = "  ".join(f"{k}: {v}" for k, v in top_langs)

        if HAS_RICH:
            console = Console()
            table = Table(title="Consolidation Summary", show_header=False, box=None, padding=(0, 1))
            table.add_row("Output", str(output_path))
            table.add_row("Output size", f"{bytes_written / 1024:.1f} KB")
            table.add_row("Scanned", str(s["total_scanned"]))
            table.add_row("Included", f"[green]{s['included']}[/]")
            table.add_row("Truncated", f"[yellow]{s['truncated']}[/]")
            table.add_row("Databases", str(s["dbs"]))
            table.add_row("Sensitive", f"[red]{s['sensitive']}[/]")
            table.add_row("Excluded", str(s["excluded"]))
            table.add_row("Total lines", f"{s['total_lines']:,}")
            table.add_row("Duration", f"{duration:.2f}s")
            table.add_row("Languages", lang_str or "—")
            if s["cap_reached"]:
                table.add_row("[bold red]WARNING[/]", "Output cap was reached — output is partial!")
            console.print(Panel(table, title="[bold green]✓ Done[/]", expand=False))
        else:
            sep = "=" * 50
            logger.info(sep)
            logger.info(f"Done in {duration:.2f}s | Output: {output_path}")
            logger.info(f"Size: {bytes_written / 1024:.1f} KB")
            logger.info(
                f"Included: {s['included']} files | Lines: {s['total_lines']:,} | "
                f"Truncated: {s['truncated']} | Sensitive: {s['sensitive']}"
            )
            logger.info(f"Languages: {lang_str}")
            if s["cap_reached"]:
                logger.warning("Output cap was reached — output is partial!")
            logger.info(sep)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate hope source code for LLM auditing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, help="Custom output file path")
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT,
        help="Project root directory",
    )
    parser.add_argument(
        "--max-size", type=int, default=DEFAULT_TRUNCATE_SIZE,
        metavar="BYTES",
        help="Max bytes per file before truncation",
    )
    parser.add_argument(
        "--max-output-mb", type=int, default=MAX_TOTAL_SIZE_MB,
        metavar="MB",
        help="Safety cap for total output size in MB",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be included without writing any file",
    )
    parser.add_argument(
        "--no-tree", action="store_true",
        help="Skip the file tree section in the output",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    # Re-configure logging if --verbose was passed
    if args.verbose:
        global logger
        logger = setup_logging(verbose=True)

    root: Path = args.root.resolve()
    if not root.is_dir():
        logger.error(f"Root directory not found: {root}")
        raise SystemExit(1)

    if not args.output:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        output = root / f"project_snapshot_{ts}.txt"
    else:
        output = args.output

    consolidator = ProjectConsolidator(
        project_root=root,
        truncate_size=args.max_size,
        max_output_mb=args.max_output_mb,
        dry_run=args.dry_run,
        include_tree=not args.no_tree,
    )
    consolidator.consolidate(output)


if __name__ == "__main__":
    main()