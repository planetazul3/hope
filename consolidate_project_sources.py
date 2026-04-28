#!/usr/bin/env python3
"""
hope — Project Source Code Consolidation Script (v2.0)
------------------------------------------------------
Optimized for high-fidelity codebase analysis by LLMs.

Enhancements in v2.0:
- Integrated 'tqdm' progress tracking for large codebase walks.
- Detailed 'Project Statistics' header (token-friendly summary).
- Enhanced XML-style tagging with metadata (lines, size, mtime).
- Improved binary detection and database header generation.
- Dynamic truncation logic with clear boundaries.
- Graceful degradation using 'rich' for professional terminal UI.
"""

import argparse
import fnmatch
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set, Dict, Any

# Attempt to use professional UI libraries if available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.logging import RichHandler
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ═══════════════════════════════════════════════════════════════
# Configuration & Metadata
# ═══════════════════════════════════════════════════════════════

PROJECT_NAME = "hope"
PROJECT_ROOT = Path(__file__).parent.absolute()

# Default settings
DEFAULT_TRUNCATE_SIZE: int = 64 * 1024  # 64 KB
MAX_TOTAL_SIZE_MB: int = 50             # Safety cap for the final output

# Categorization
DB_EXTENSIONS: Set[str] = {".db", ".sqlite", ".sqlite3", ".parquet", ".csv", ".onnx", ".onnx.data", ".bin"}
DB_FILES: Set[str] = {"tick_store.db", "ticks.csv", "candles.csv", "backtest_trades.csv", "model.onnx"}

EXCLUDE_DIRS: Set[str] = {
    ".git", ".vscode", ".idea", "venv", ".venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".eggs",
    "node_modules", ".next", "dist", "build", "target", ".cargo",
    "coverage", "playwright-report", "test-results", ".gemini", ".agents",
    ".agent", "_agents", "_agent", "artifacts", "scratch", "knowledge",
    "temp", "tmp", "libtorch", "logs", "new_venv", "test_venv", "backtest_optimization",
}

EXCLUDE_FILES: Set[str] = {
    ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll",
    "*.dylib", "*.exe", "*.log", "*.pid", "*.pid.lock", "*.min.js",
    "*.min.css", "*.map", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "*.seed", "Cargo.lock", "*.rlib", "*.rmeta", "*.d", "*.pt",
    "*.pth", "*.bin", "*.safetensors", "*.joblib", "*.resolved",
    "*.tar.gz", "*.tar", "*.gz", "*.bz2", "*.7z", "*.rar", "*.zip",
    "audit_trial.log", "tick_audit.log", "consolidated_audit.txt",
    "project_snapshot.txt", "*.sig", "engine.log", "engine_final.log"
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

LANGUAGE_MAP = {
    ".py": "Python", ".rs": "Rust", ".js": "JavaScript", ".ts": "TypeScript",
    ".json": "JSON", ".toml": "TOML", ".md": "Markdown", ".sh": "Shell",
    ".sql": "SQL", ".ipynb": "Jupyter Notebook", ".yaml": "YAML", ".yml": "YAML",
    ".env": "Environment Variables", ".proto": "Protobuf", ".c": "C", ".cpp": "C++",
}

# ═══════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════

def setup_logging():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    if HAS_RICH:
        handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(message)s"
    
    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[handler]
    )
    return logging.getLogger("Consolidator")

logger = setup_logging()

# ═══════════════════════════════════════════════════════════════
# Core Logic
# ═══════════════════════════════════════════════════════════════

class FileWalker:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def is_excluded_dir(self, dir_name: str) -> bool:
        return dir_name in EXCLUDE_DIRS

    def is_excluded_file(self, file_path: Path) -> bool:
        if file_path.name in FORCE_INCLUDE_FILES:
            return False
        
        # Check patterns
        for pattern in EXCLUDE_FILES:
            if fnmatch.fnmatch(file_path.name, pattern):
                # Don't exclude if it's a database file we want to track
                if file_path.name not in DB_FILES:
                    return True
        return False

    def is_database_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in DB_EXTENSIONS or file_path.name in DB_FILES

    def is_text_file(self, file_path: Path) -> bool:
        if file_path.name in FORCE_INCLUDE_FILES or file_path.suffix.lower() in LANGUAGE_MAP:
            return True
        
        # Heuristic check for binary content
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(2048)
                if not chunk: return True
                # Check for null bytes or high non-ascii density
                null_count = chunk.count(b'\0')
                if null_count > 0: return False
                
                # Check if it can be decoded as utf-8
                try:
                    chunk.decode('utf-8')
                    return True
                except UnicodeDecodeError:
                    return False
        except OSError:
            return False

    @staticmethod
    def get_file_language(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if file_path.name == "Cargo.toml": return "TOML (Cargo)"
        if file_path.name == "Makefile": return "Makefile"
        if file_path.name == "Dockerfile": return "Dockerfile"
        return LANGUAGE_MAP.get(ext, "Text")

    def build_file_tree(self, directory: Path, prefix: str = "", depth: int = 0, max_depth: int = 10) -> List[str]:
        if depth > max_depth:
            return [f"{prefix}└── [Max Depth Reached]"]
            
        tree_lines = []
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
                if item.is_dir():
                    tree_lines.append(f"{prefix}{connector}{item.name}/")
                    tree_lines.extend(self.build_file_tree(item, prefix + ("    " if is_last else "│   "), depth + 1, max_depth))
                else:
                    tree_lines.append(f"{prefix}{connector}{item.name}")
        except PermissionError:
            tree_lines.append(f"{prefix}└── [Permission Denied]")
        return tree_lines

class ProjectConsolidator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.walker = FileWalker(project_root)
        self.stats = {
            "total_scanned": 0,
            "included": 0,
            "truncated": 0,
            "dbs": 0,
            "total_lines": 0,
            "sensitive": 0,
            "excluded": 0,
            "languages": {},
            "start_time": datetime.now(timezone.utc)
        }

    def is_sensitive_file(self, file_path: Path) -> bool:
        file_str = file_path.name
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, file_str, re.IGNORECASE):
                if ".example" not in file_str and ".template" not in file_str:
                    return True
        return False

    def _get_git_info(self) -> Dict[str, str]:
        info = {"rev": "unknown", "branch": "unknown"}
        try:
            info["rev"] = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], 
                cwd=self.project_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
            info["branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                cwd=self.project_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            pass
        return info

    def consolidate(self, output_path: Path):
        msg = f"🚀 Consolidating [bold cyan]{PROJECT_NAME}[/] source" if HAS_RICH else f"Consolidating {PROJECT_NAME} source"
        logger.info(msg)
        
        git_info = self._get_git_info()
        all_files = []
        
        # Phase 1: Discovery
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not self.walker.is_excluded_dir(d)]
            for file in files:
                fpath = Path(root) / file
                if fpath.name == output_path.name: continue
                all_files.append(fpath)
        
        # Phase 2: Processing
        with open(output_path, "w", encoding="utf-8") as out:
            # Global Header
            out.write("<!-- PROJECT CONSOLIDATION SNAPSHOT -->\n")
            out.write(f"<!-- Generated: {datetime.now(timezone.utc).isoformat()} -->\n")
            out.write(f"<!-- Git: {git_info['branch']} @ {git_info['rev']} -->\n\n")
            
            # File Tree
            out.write("FILE_TREE_START\n")
            tree = self.walker.build_file_tree(self.project_root)
            for line in tree:
                out.write(line + "\n")
            out.write("FILE_TREE_END\n\n")

            pbar = tqdm(all_files, desc="Processing files", disable=not HAS_TQDM)
            for fpath in pbar:
                self.stats["total_scanned"] += 1
                
                if self.walker.is_excluded_file(fpath):
                    self.stats["excluded"] += 1
                    continue
                
                rel_path = fpath.relative_to(self.project_root)
                
                # Metadata
                mtime = datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc).isoformat()
                size = fpath.stat().st_size
                
                # Handle databases/binary data
                if self.walker.is_database_file(fpath):
                    self.stats["dbs"] += 1
                    out.write(f'<file path="{rel_path}" type="database" size="{size}" mtime="{mtime}">\n')
                    out.write(f"<!-- Binary/Database content omitted. Type: {fpath.suffix} -->\n")
                    out.write("</file>\n\n")
                    continue

                # Handle sensitive files
                if self.is_sensitive_file(fpath):
                    self.stats["sensitive"] += 1
                    out.write(f'<file path="{rel_path}" type="sensitive" mtime="{mtime}">\n')
                    out.write("<!-- Sensitive content omitted for security. -->\n")
                    out.write("</file>\n\n")
                    continue

                # Handle regular text files
                if not self.walker.is_text_file(fpath):
                    self.stats["excluded"] += 1
                    continue

                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(DEFAULT_TRUNCATE_SIZE + 1024)
                        is_truncated = len(content) > DEFAULT_TRUNCATE_SIZE
                        if is_truncated:
                            content = content[:DEFAULT_TRUNCATE_SIZE]
                            self.stats["truncated"] += 1
                        
                        lang = FileWalker.get_file_language(fpath)
                        self.stats["languages"][lang] = self.stats["languages"].get(lang, 0) + 1
                        
                        line_count = content.count("\n")
                        out.write(f'<file path="{rel_path}" language="{lang}" lines="{line_count}" size="{size}" mtime="{mtime}" truncated="{str(is_truncated).lower()}">\n')
                        out.write(content)
                        if is_truncated:
                            out.write("\n\n[... CONTENT TRUNCATED AT 64KB ...]\n")
                        out.write("\n</file>\n\n")
                        
                        self.stats["included"] += 1
                        self.stats["total_lines"] += line_count
                        
                except Exception as e:
                    logger.error(f"Failed to read {fpath}: {e}")

        self._print_summary(output_path)

    def _print_summary(self, output_path: Path):
        duration = (datetime.now(timezone.utc) - self.stats["start_time"]).total_seconds()
        
        if HAS_RICH and HAS_RICH:
            console = Console()
            table = Table(title="Consolidation Summary", show_header=False, box=None)
            table.add_row("Output File", str(output_path))
            table.add_row("File Size", f"{output_path.stat().st_size / 1024 / 1024:.2f} MB")
            table.add_row("Total Files Scanned", str(self.stats["total_scanned"]))
            table.add_row("Files Included", f"[green]{self.stats['included']}[/]")
            table.add_row("Excluded", str(self.stats["excluded"]))
            table.add_row("Databases (Headers)", str(self.stats["dbs"]))
            table.add_row("Total Lines", f"{self.stats['total_lines']:,}")
            table.add_row("Duration", f"{duration:.2f}s")
            console.print(Panel(table, title="[bold green]Success[/]", expand=False))
        else:
            logger.info("=" * 40)
            logger.info(f"Consolidation complete in {duration:.2f}s")
            logger.info(f"Output: {output_path} ({output_path.stat().st_size:,} bytes)")
            logger.info(f"Included: {self.stats['included']} files, {self.stats['total_lines']:,} lines")
            logger.info("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate project source code for LLM auditing.")
    parser.add_argument("--output", type=Path, help="Custom output path")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root directory")
    parser.add_argument("--max-size", type=int, default=DEFAULT_TRUNCATE_SIZE, help="Max file size in bytes before truncation")
    args = parser.parse_args()
    
    DEFAULT_TRUNCATE_SIZE = args.max_size
    
    root = args.root
    if not args.output:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        args.output = root / f"project_snapshot_{ts}.txt"
    
    consolidator = ProjectConsolidator(root)
    consolidator.consolidate(args.output)
