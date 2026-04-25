#!/usr/bin/env python3
"""
hope — Project Source Code Consolidation Script
Adapted for the Rust trading-system project structure.

Consolidates all project source code into a single auditable text file.

Features:
- Excludes binaries, compiled files, dependencies, caches
- Handles databases (.db, .csv) with descriptive headers instead of content
- Truncates files larger than 50 KB to optimize AI processing
- Includes metadata and file headers for every source file
- Handles sensitive files (existence check without content)
- Generates comprehensive project snapshot for auditing
- Restore detailed console summary and logging
"""

import argparse
import fnmatch
import logging
import mimetypes
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project root — dynamically determined
PROJECT_ROOT = Path(__file__).parent.absolute()

# Project metadata
PROJECT_NAME = "hope"
PROJECT_DESCRIPTION = "Deterministic Deriv trading engine"

# Output file pattern for gitignore
OUTPUT_FILE_PATTERN = "*_merged_sources*.txt"
OUTPUT_FILE_REGEX = re.compile(fnmatch.translate(OUTPUT_FILE_PATTERN))

# Files that must NEVER be overwritten or included in the consolidation
PROTECTED_FILES: Set[str] = set()

# ═══════════════════════════════════════════════════════════════
# Exclusion and Truncation configuration
# ═══════════════════════════════════════════════════════════════

# Default truncation size (50 KB)
DEFAULT_TRUNCATE_SIZE: int = 50 * 1024

# Database and large data extensions (content ignored, header added)
DB_EXTENSIONS: Set[str] = {".db", ".sqlite", ".sqlite3", ".parquet", ".csv"}
DB_FILES: Set[str] = {"tick_store.db", "ticks.csv", "candles.csv", "backtest_trades.csv"}

EXCLUDE_DIRS: Set[str] = {
    ".git", ".vscode", ".idea", "venv", ".venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".eggs",
    "node_modules", ".next", "dist", "build", "target", ".cargo",
    "coverage", "playwright-report", "test-results", ".gemini", ".agents",
    ".agent", "_agents", "_agent", "artifacts", "scratch", "knowledge",
    "temp", "tmp", "libtorch", "logs", "new_venv",
}

EXCLUDE_FILES: Set[str] = {
    ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll",
    "*.dylib", "*.exe", "*.log", "*.pid", "*.pid.lock", "*.min.js",
    "*.min.css", "*.map", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "*.seed", "Cargo.lock", "*.rlib", "*.rmeta", "*.d", "*.onnx", "*.pt",
    "*.pth", "*.bin", "*.safetensors", "*.joblib", "*.resolved",
    "*.tar.gz", "*.tar", "*.gz", "*.bz2", "*.7z", "*.rar", "*.zip",
    "audit_trial.log", "tick_audit.log", "consolidated_audit.txt",
}

EXCLUDE_EXTENSIONS: Set[str] = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe", ".o", ".a",
    ".lib", ".obj", ".class", ".jar", ".war", ".rlib", ".rmeta", ".d",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mp3", ".wav",
    ".ogg", ".flac", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".woff", ".woff2", ".ttf",
    ".eot", ".otf", ".lock", ".map", ".onnx", ".pt", ".pth", ".safetensors",
    ".joblib",
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
}

class FileWalker:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def is_excluded_dir(self, dir_name: str) -> bool:
        return dir_name in EXCLUDE_DIRS

    def is_excluded_file(self, file_path: Path) -> bool:
        if file_path.name in FORCE_INCLUDE_FILES:
            return False
        if file_path.suffix.lower() in EXCLUDE_EXTENSIONS and file_path.suffix.lower() not in DB_EXTENSIONS:
            return True
        for pattern in EXCLUDE_FILES:
            if fnmatch.fnmatch(file_path.name, pattern) and file_path.name not in DB_FILES:
                return True
        return False

    def is_database_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in DB_EXTENSIONS or file_path.name in DB_FILES

    def is_text_file(self, file_path: Path) -> bool:
        if file_path.name in FORCE_INCLUDE_FILES:
            return True
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type.startswith("text"):
            return True
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' not in chunk
        except OSError:
            return False

    @staticmethod
    def get_file_language(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if file_path.name == "Cargo.toml":
            return "TOML (Cargo)"
        return LANGUAGE_MAP.get(ext, "Text")

    def build_file_tree(self, directory: Path, prefix: str = "") -> List[str]:
        tree_lines = []
        try:
            items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            items = [item for item in items if not self.is_excluded_dir(item.name)]
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                if item.is_dir():
                    tree_lines.append(f"{prefix}{connector}{item.name}/")
                    tree_lines.extend(self.build_file_tree(item, prefix + ("    " if is_last else "│   ")))
                elif not self.is_excluded_file(item):
                    tree_lines.append(f"{prefix}{connector}{item.name}")
        except PermissionError:
            pass
        return tree_lines

class ReportGenerator:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def write_header(self, out, git_info: Dict[str, str]) -> None:
        out.write("=" * 80 + "\n")
        out.write("PROJECT SOURCE CODE CONSOLIDATION (AUDIT READY)\n")
        out.write("=" * 80 + "\n\n")
        out.write(f"Project:          {PROJECT_NAME}\n")
        out.write(f"Consolidation:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Git Commit:       {git_info.get('commit', 'unknown')}\n")
        out.write("\nNote: Files > 50 KB are truncated. Databases are listed as headers only.\n\n")

    def write_db_header(self, out, file_path: Path, file_stat: os.stat_result) -> None:
        rel_path = file_path.relative_to(self.project_root)
        out.write("\n" + "#" * 80 + "\n")
        out.write(f"DATABASE FILE DETECTED: {rel_path}\n")
        out.write("#" * 80 + "\n")
        out.write("Type:      Data/Database (Content Ignored)\n")
        out.write(f"Size:      {file_stat.st_size:,} bytes\n")
        content_desc = "Ticks/Market Data" if "tick" in rel_path.name.lower() else "General Data"
        if "candle" in rel_path.name.lower():
            content_desc = "Candle/OHLC Data"
        out.write(f"Estimated Content: {content_desc}\n")
        out.write("-" * 80 + "\n\n")

    def write_regular_file(self, out, file_path: Path, content: str, truncated: bool) -> None:
        rel_path = file_path.relative_to(self.project_root)
        lang = FileWalker.get_file_language(file_path)
        out.write("\n" + "-" * 80 + "\n")
        out.write(f"FILE: {rel_path} ({lang})\n")
        if truncated:
            out.write("WARNING: Content truncated due to size limit (> 50 KB)\n")
        out.write("-" * 80 + "\n\n")
        out.write(content)
        if truncated:
            out.write("\n\n[... CONTENT TRUNCATED ...]\n")
        out.write("\n")

class ProjectConsolidator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.walker = FileWalker(project_root)
        self.generator = ReportGenerator(project_root)
        self.stats = {
            "total_scanned": 0,
            "included": 0,
            "truncated": 0,
            "dbs": 0,
            "total_lines": 0,
            "sensitive": 0,
            "excluded": 0,
            "languages": {}
        }

    def is_sensitive_file(self, file_path: Path) -> bool:
        file_str = str(file_path)
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, file_str, re.IGNORECASE):
                return True
        return False

    def consolidate(self, output_path: Path):
        git_info = self._get_git_info()
        logger.info("Starting project consolidation...")
        logger.info(f"Project: {self.project_root}")
        logger.info(f"Output:  {output_path}")

        with open(output_path, "w", encoding="utf-8") as out:
            self.generator.write_header(out, git_info)
            tree = self.walker.build_file_tree(self.project_root)
            out.write("PROJECT STRUCTURE\n" + "-" * 80 + "\n")
            for line in tree:
                out.write(line + "\n")
            out.write("\n")
            
            for root, dirs, files in os.walk(self.project_root):
                dirs[:] = [d for d in dirs if not self.walker.is_excluded_dir(d)]
                for file in sorted(files):
                    fpath = Path(root) / file
                    if fpath.name == Path(__file__).name or OUTPUT_FILE_REGEX.match(fpath.name):
                        continue
                    
                    self.stats["total_scanned"] += 1
                    
                    if self.walker.is_excluded_file(fpath):
                        self.stats["excluded"] += 1
                        continue
                    
                    stat = fpath.stat()
                    if self.walker.is_database_file(fpath):
                        self.generator.write_db_header(out, fpath, stat)
                        self.stats["dbs"] += 1
                        logger.info(f"💾 Database: {fpath.relative_to(self.project_root)}")
                        continue
                    
                    if self.is_sensitive_file(fpath):
                        self.stats["sensitive"] += 1
                        # Include as header only
                        out.write("\n" + "-" * 80 + "\n")
                        out.write(f"FILE: {fpath.relative_to(self.project_root)} (SENSITIVE)\n")
                        out.write("-" * 80 + "\n")
                        out.write("NOTE: Content not included for security.\n\n")
                        logger.info(f"🔒 Sensitive: {fpath.relative_to(self.project_root)}")
                        continue

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
                            
                            self.generator.write_regular_file(out, fpath, content, is_truncated)
                            self.stats["included"] += 1
                            self.stats["total_lines"] += content.count("\n")
                            
                            lang = FileWalker.get_file_language(fpath)
                            self.stats["languages"][lang] = self.stats["languages"].get(lang, 0) + 1
                            logger.debug(f"✓ Included: {fpath.relative_to(self.project_root)}")
                    except Exception as e:
                        logger.error(f"Failed to read {fpath}: {e}")

        self._print_summary(output_path)

    def _get_git_info(self) -> Dict[str, str]:
        try:
            rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
            return {"commit": rev}
        except Exception:
            return {}

    def _print_summary(self, output_path: Path):
        logger.info("Consolidation complete!")
        logger.info(f"Output file: {output_path}")
        logger.info("=" * 80)
        logger.info("CONSOLIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Output File:        {output_path}")
        logger.info(f"File Size:          {output_path.stat().st_size:,} bytes")
        logger.info(f"Total Files:        {self.stats['total_scanned']}")
        logger.info(f"Included:           {self.stats['included']}")
        logger.info(f"Databases:          {self.stats['dbs']}")
        logger.info(f"Excluded:           {self.stats['excluded']}")
        logger.info(f"Sensitive:          {self.stats['sensitive']}")
        logger.info(f"Total Lines:        {self.stats['total_lines']:,}")
        logger.info("\nTop Languages:")
        sorted_langs = sorted(self.stats["languages"].items(), key=lambda x: x[1], reverse=True)
        for lang, count in sorted_langs:
            logger.info(f"  {lang:25s} {count:4d} files")
        logger.info("\n✅ Consolidation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    
    root = PROJECT_ROOT
    logger.info("=" * 80)
    logger.info(f"{PROJECT_NAME.upper()} — PROJECT SOURCE CONSOLIDATION TOOL")
    logger.info("=" * 80)
    
    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        args.output = root / f"{root.name}_{ts}_merged_sources.txt"
    
    ProjectConsolidator(root).consolidate(args.output)
