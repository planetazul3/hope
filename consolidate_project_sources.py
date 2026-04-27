#!/usr/bin/env python3
"""
hope — Project Source Code Consolidation Script
Adapted for the Rust trading-system project structure.

Consolidates all project source code into a single auditable text file 
optimized for LLM analysis.

Features:
- XML-style tagging for clear file boundaries (<file path="...">...</file>)
- Robust exclusion of binaries, compiled files, dependencies, caches
- Database and large data handling (header only, content ignored)
- Automatic truncation of extremely large files to prevent context overflow
- Respects LOG_LEVEL environment variable
- Generates a predictable 'project_snapshot.txt' by default
"""

import argparse
import fnmatch
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Set

# ═══════════════════════════════════════════════════════════════
# Configuration & Metadata
# ═══════════════════════════════════════════════════════════════

PROJECT_NAME = "hope"
PROJECT_ROOT = Path(__file__).parent.absolute()

# Default truncation size (50 KB)
DEFAULT_TRUNCATE_SIZE: int = 50 * 1024

# Database and large data extensions (content ignored, header added)
DB_EXTENSIONS: Set[str] = {".db", ".sqlite", ".sqlite3", ".parquet", ".csv", ".onnx", ".onnx.data"}
DB_FILES: Set[str] = {"tick_store.db", "ticks.csv", "candles.csv", "backtest_trades.csv", "model.onnx"}

EXCLUDE_DIRS: Set[str] = {
    ".git", ".vscode", ".idea", "venv", ".venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".eggs",
    "node_modules", ".next", "dist", "build", "target", ".cargo",
    "coverage", "playwright-report", "test-results", ".gemini", ".agents",
    ".agent", "_agents", "_agent", "artifacts", "scratch", "knowledge",
    "temp", "tmp", "libtorch", "logs", "new_venv", "test_venv",
}

EXCLUDE_FILES: Set[str] = {
    ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll",
    "*.dylib", "*.exe", "*.log", "*.pid", "*.pid.lock", "*.min.js",
    "*.min.css", "*.map", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "*.seed", "Cargo.lock", "*.rlib", "*.rmeta", "*.d", "*.onnx", "*.pt",
    "*.pth", "*.bin", "*.safetensors", "*.joblib", "*.resolved",
    "*.tar.gz", "*.tar", "*.gz", "*.bz2", "*.7z", "*.rar", "*.zip",
    "audit_trial.log", "tick_audit.log", "consolidated_audit.txt",
    "project_snapshot.txt",
}

EXCLUDE_EXTENSIONS: Set[str] = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe", ".o", ".a",
    ".lib", ".obj", ".class", ".jar", ".war", ".rlib", ".rmeta", ".d",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mp3", ".wav",
    ".ogg", ".flac", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".woff", ".woff2", ".ttf",
    ".eot", ".otf", ".lock", ".map",
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
    ".env": "Environment Variables",
}

# ═══════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════

def setup_logging():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)

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
        
        # Check extensions
        if file_path.suffix.lower() in EXCLUDE_EXTENSIONS:
            # Special case: some "database" extensions might be in exclude but we want to list them as headers
            if file_path.suffix.lower() not in DB_EXTENSIONS:
                return True
                
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
                chunk = f.read(1024)
                return b'\0' not in chunk
        except OSError:
            return False

    @staticmethod
    def get_file_language(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if file_path.name == "Cargo.toml":
            return "TOML (Cargo)"
        if file_path.name == "Makefile":
            return "Makefile"
        return LANGUAGE_MAP.get(ext, "Text")

    def build_file_tree(self, directory: Path, prefix: str = "") -> List[str]:
        tree_lines = []
        try:
            # Filter and sort items
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
                    tree_lines.extend(self.build_file_tree(item, prefix + ("    " if is_last else "│   ")))
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
            "languages": {}
        }

    def is_sensitive_file(self, file_path: Path) -> bool:
        file_str = file_path.name
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, file_str, re.IGNORECASE):
                # Don't mark example files as sensitive
                if ".example" not in file_str:
                    return True
        return False

    def _get_git_info(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], 
                cwd=self.project_root, 
                text=True, 
                stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return "unknown"

    def consolidate(self, output_path: Path):
        logger.info(f"🚀 Consolidating {PROJECT_NAME} at {self.project_root}")
        
        git_rev = self._get_git_info()
        
        with open(output_path, "w", encoding="utf-8") as out:
            # Write global header
            out.write("<!-- PROJECT CONSOLIDATION SNAPSHOT -->\n")
            out.write(f"<!-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n")
            out.write(f"<!-- Git Revision: {git_rev} -->\n\n")
            
            # Write file tree
            out.write("FILE_TREE_START\n")
            tree = self.walker.build_file_tree(self.project_root)
            for line in tree:
                out.write(line + "\n")
            out.write("FILE_TREE_END\n\n")

            # Walk and include files
            for root, dirs, files in os.walk(self.project_root):
                # Prune excluded directories
                dirs[:] = [d for d in dirs if not self.walker.is_excluded_dir(d)]
                
                for file in sorted(files):
                    fpath = Path(root) / file
                    
                    # Skip the output file
                    if fpath.name == output_path.name:
                        continue
                    
                    self.stats["total_scanned"] += 1
                    
                    if self.walker.is_excluded_file(fpath):
                        self.stats["excluded"] += 1
                        continue
                    
                    rel_path = fpath.relative_to(self.project_root)
                    
                    # Handle databases/binary data
                    if self.walker.is_database_file(fpath):
                        self.stats["dbs"] += 1
                        stat = fpath.stat()
                        out.write(f'<file path="{rel_path}" type="database" size="{stat.st_size}">\n')
                        out.write(f"<!-- Database content omitted. Type: {fpath.suffix} -->\n")
                        out.write("</file>\n\n")
                        logger.info(f"💾 {rel_path} (Database)")
                        continue

                    # Handle sensitive files
                    if self.is_sensitive_file(fpath):
                        self.stats["sensitive"] += 1
                        out.write(f'<file path="{rel_path}" type="sensitive">\n')
                        out.write("<!-- Sensitive content omitted for security. -->\n")
                        out.write("</file>\n\n")
                        logger.info(f"🔒 {rel_path} (Sensitive)")
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
                            
                            out.write(f'<file path="{rel_path}" language="{lang}" truncated="{"true" if is_truncated else "false"}">\n')
                            out.write(content)
                            if is_truncated:
                                out.write("\n\n[... CONTENT TRUNCATED ...]\n")
                            out.write("\n</file>\n\n")
                            
                            self.stats["included"] += 1
                            self.stats["total_lines"] += content.count("\n")
                            logger.debug(f"✓ {rel_path}")
                            
                    except Exception as e:
                        logger.error(f"Failed to read {fpath}: {e}")

        self._print_summary(output_path)

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
        if self.stats.get("truncated", 0) > 0:
            logger.info(f"Truncated:          {self.stats['truncated']}")
        
        logger.info("\nTop Languages:")
        sorted_langs = sorted(self.stats["languages"].items(), key=lambda x: x[1], reverse=True)
        for lang, count in sorted_langs:
            logger.info(f"  {lang:25s} {count:4d} files")
        logger.info("\n✅ Consolidation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate project source code for auditing.")
    parser.add_argument("--output", type=Path, help="Custom output path")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root directory")
    args = parser.parse_args()
    
    root = args.root
    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        args.output = root / f"{root.name}_{ts}_merged_sources.txt"
    
    consolidator = ProjectConsolidator(root)
    consolidator.consolidate(args.output)
