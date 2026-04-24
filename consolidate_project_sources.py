#!/usr/bin/env python3
"""
hope — Project Source Code Consolidation Script
Adapted for the Rust trading-system project structure.

Consolidates all project source code into a single auditable text file.

Features:
- Excludes binaries, compiled files, dependencies, caches
- Excludes large binary ML/data files (.db, .onnx, .pt)
- Includes metadata and file headers for every source file
- Handles sensitive files (existence check without content)
- Generates comprehensive project snapshot for auditing
- Version and timestamp-based output filename
- Rust-aware: excludes target/, Cargo build artifacts
- Project-aware: skips model artifacts and local databases while including auditable notebooks
"""

import argparse
import fnmatch
import logging
import mimetypes
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

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
# Exclusion configuration — tailored for hope
# ═══════════════════════════════════════════════════════════════

EXCLUDE_DIRS: Set[str] = {
    # Version control
    ".git",
    # IDE / editor
    ".vscode",
    ".idea",
    # Python runtime
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".eggs",
    # Node (just in case)
    "node_modules",
    ".next",
    # Build output
    "dist",
    "build",
    "target",
    ".cargo",
    # Test / coverage artifacts
    "coverage",
    "playwright-report",
    "test-results",
    # Gemini / agent artifacts
    ".gemini",
    ".agents",
    ".agent",
    "_agents",
    "_agent",
    "artifacts",
    "scratch",
    "knowledge",     # Knowledge items
    # Temp
    "temp",
    "tmp",
    "libtorch",
    "logs",
}

EXCLUDE_FILES: Set[str] = {
    ".DS_Store",
    "Thumbs.db",
    # Python bytecode
    "*.pyc",
    "*.pyo",
    "*.pyd",
    # Shared libraries / executables
    "*.so",
    "*.dll",
    "*.dylib",
    "*.exe",
    # Logs and PIDs
    "*.log",
    "*.pid",
    "*.pid.lock",
    # JS / CSS minified
    "*.min.js",
    "*.min.css",
    "*.map",
    # Lock files
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "*.seed",
    # Rust-specific (kept for safety)
    "Cargo.lock",
    "*.rlib",
    "*.rmeta",
    "*.d",
    # ML / binary model files
    "*.onnx",
    "*.pt",
    "*.pth",
    "*.bin",
    "*.safetensors",
    "*.joblib",
    "backtest_trades.csv",
    "*.resolved",
    "*.tar.gz",         # Archives
    "*.tar",
    "*.gz",
    "*.bz2",
    "*.7z",
    "*.rar",
    "*.zip",
    "audit_trial.log",
    "tick_audit.log",
    "consolidated_audit.txt",
}

EXCLUDE_EXTENSIONS: Set[str] = {
    # Compiled / binary
    ".pyc", ".pyo", ".pyd",
    ".so", ".dll", ".dylib",
    ".exe", ".o", ".a", ".lib", ".obj",
    ".class", ".jar", ".war",
    # Rust compiled artifacts
    ".rlib", ".rmeta", ".d",
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    # Video / Audio
    ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".mp3", ".wav", ".ogg", ".flac",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Lock / source maps
    ".lock", ".map",
    # ML model / serialization files
    ".onnx", ".pt", ".pth", ".safetensors",
    ".joblib",
    ".db",
    ".parquet",
}

# Sensitive file patterns (check existence, don't include content)
SENSITIVE_PATTERNS: List[str] = [
    r"\.env(\.[a-z]+)?$",
    r".*\.key$",
    r".*\.pem$",
    r".*\.crt$",
    r".*\.cert$",
    r".*secrets.*",
    r".*credentials.*",
    r".*password.*",
    r".*token.*",
    r".*app_id.*",
    r".*api_token.*",
    r".*deriv_token.*",
    r".*ssid.*",
]

# Files to always include even if binary check might fail
FORCE_INCLUDE_FILES: Set[str] = {
    "Dockerfile",
    "docker-compose.yml",
    ".dockerignore",
    ".gitignore",
    ".gitattributes",
    "requirements.txt",
    "package.json",
    "tsconfig.json",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "Cargo.toml",       # Rust manifest (kept for safety)
    "Makefile",         # Core project execution instructions
    "pyproject.toml",   # Python project metadata
    "setup.py",
    "setup.cfg",
}

NOTEBOOK_SCRIPT_REFERENCES: Dict[str, str] = {
    "notebooks/train_transformer.ipynb": "scripts/train.py",
}

# Maximum file size to include (5 MB — conservative for this project)
MAX_FILE_SIZE: int = 5_000_000


class GitInfoProvider:
    """Provides git repository information."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def get_git_info(self) -> Dict[str, str]:
        """Get current git commit information."""
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                stderr=subprocess.PIPE,
                text=True,
            ).strip()

            commit_date = subprocess.check_output(
                ["git", "log", "-1", "--format=%cd", "--date=iso"],
                cwd=self.project_root,
                stderr=subprocess.PIPE,
                text=True,
            ).strip()

            branch = subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=self.project_root,
                stderr=subprocess.PIPE,
                text=True,
            ).strip()

            return {
                "commit": commit_hash[:8],
                "date": commit_date,
                "branch": branch,
            }
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip() if e.stderr else "No error details"
            logger.warning(f"Git command failed: {stderr}")
            return {"commit": "unknown", "date": "unknown", "branch": "unknown"}
        except FileNotFoundError:
            logger.warning("Git executable not found.")
            return {"commit": "unknown", "date": "unknown", "branch": "unknown"}


TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".config",
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".css", ".scss", ".html", ".xml", ".sql",
    ".sh", ".bash", ".zsh",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".lua", ".pl", ".r", ".m",
    ".vim", ".el", ".clj", ".ex", ".exs",
    ".Dockerfile", ".gitignore", ".dockerignore",
    ".ipynb",
}

LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "React JSX",
    ".tsx": "React TSX",
    ".css": "CSS",
    ".scss": "SCSS",
    ".html": "HTML",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".md": "Markdown",
    ".sql": "SQL",
    ".sh": "Shell",
    ".bash": "Bash",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C Header",
    ".hpp": "C++ Header",
    ".ipynb": "Jupyter Notebook",
    ".ini": "INI Config",
    ".cfg": "Config",
}


class FileWalker:
    """Handles walking the file system and applying exclusion logic."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def is_excluded_dir(self, dir_name: str) -> bool:
        """Check if directory should be excluded."""
        return dir_name in EXCLUDE_DIRS

    def is_excluded_file(
        self,
        file_path: Path,
        file_size: Optional[int] = None,
    ) -> bool:
        """Check if file should be excluded."""
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        # Force include certain files FIRST
        if file_path.name in FORCE_INCLUDE_FILES:
            return False

        # Check extension
        if file_path.suffix.lower() in EXCLUDE_EXTENSIONS:
            return True

        # Check filename patterns (glob-based)
        for pattern in EXCLUDE_FILES:
            if fnmatch.fnmatch(file_path.name, pattern):
                return True

        # Check file size
        if file_size is None:
            try:
                file_size = os.path.getsize(str(file_path))
            except OSError:
                try:
                    file_size = file_path.stat().st_size
                except OSError as e:
                    logger.error(f"Error accessing file {file_path}: {e}")
                    return True

        if file_size >= MAX_FILE_SIZE:
            logger.warning(
                "File %s exceeds size limit (%s bytes), excluding",
                file_path.relative_to(self.project_root),
                f"{file_size:,}",
            )
            return True

        # Check if binary
        if not self.is_text_file(file_path):
            return True

        return False

    def is_text_file(self, file_path: Path) -> bool:
        """Check if file is text (not binary)."""
        if file_path.name in FORCE_INCLUDE_FILES:
            return True

        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type.startswith("text"):
            return True

        if file_path.suffix.lower() in TEXT_EXTENSIONS:
            return True

        try:
            with open(file_path, encoding="utf-8") as f:
                f.read(512)
            return True
        except (OSError, UnicodeDecodeError):
            return False

    @staticmethod
    def get_file_language(file_path: "Path | str") -> str:
        """Detect file language/type."""
        if not isinstance(file_path, Path):
            name = str(file_path)
            ext = Path(name).suffix.lower()
        else:
            ext = file_path.suffix.lower()
            name = file_path.name

        if name == "Dockerfile":
            return "Docker"
        if name == "Cargo.toml":
            return "TOML (Cargo)"
        if name == "settings.json":
            return "JSON (Bot Config)"
        if name == "pytest.ini":
            return "INI (pytest)"

        return LANGUAGE_MAP.get(ext, "Text")

    def build_file_tree(self, directory: Path, prefix: str = "") -> List[str]:
        """Build a visual tree structure of the project."""
        tree_lines = []

        try:
            items = sorted(
                directory.iterdir(), key=lambda x: (not x.is_dir(), x.name)
            )
            items = [
                item for item in items if not self.is_excluded_dir(item.name)
            ]

            for i, item in enumerate(items):
                is_last_item = i == len(items) - 1
                connector = "└── " if is_last_item else "├── "
                extension = "    " if is_last_item else "│   "

                if item.is_dir():
                    tree_lines.append(f"{prefix}{connector}{item.name}/")
                    sub_tree = self.build_file_tree(item, prefix + extension)
                    tree_lines.extend(sub_tree)
                elif not self.is_excluded_file(item):
                    tree_lines.append(f"{prefix}{connector}{item.name}")
        except PermissionError as e:
            logger.warning(f"Permission denied accessing {directory}: {e}")

        return tree_lines


class ReportGenerator:
    """Generates the final consolidated text report."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def write_header(
        self, out, timestamp: datetime, git_info: Dict[str, str]
    ) -> None:
        """Write file header."""
        out.write("=" * 80 + "\n")
        out.write("PROJECT SOURCE CODE CONSOLIDATION\n")
        out.write("=" * 80 + "\n\n")

        out.write(f"Project:          {PROJECT_NAME}\n")
        out.write(f"Description:      {PROJECT_DESCRIPTION}\n")
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        out.write(f"Consolidation:    {time_str}\n")
        out.write(f"Git Commit:       {git_info['commit']}\n")
        out.write(f"Git Branch:       {git_info['branch']}\n")
        out.write(f"Commit Date:      {git_info['date']}\n")
        out.write(f"Project Root:     {self.project_root}\n")

        out.write("\n" + "=" * 80 + "\n")
        out.write("PURPOSE\n")
        out.write("=" * 80 + "\n\n")
        out.write(
            "This file contains a complete consolidation of the project "
            "source code,\nconfiguration files, notebooks, and "
            "documentation for auditing and reproduction purposes.\n"
        )
        out.write("\n")
        out.write("Exclusions:\n")
        out.write("  - Binary files (images, compiled code, executables)\n")
        out.write("  - Dependencies and build outputs (venv, target, __pycache__)\n")
        out.write("  - ML model files (*.onnx, *.pt, *.pth)\n")
        out.write("  - Database files (data/tick_store.db)\n")
        out.write("  - Archives (*.tar.gz, *.zip)\n")
        out.write("  - Runtime logs (tick_audit.log, *.log)\n")
        out.write("  - Large files (> 5 MB)\n")
        out.write("\n")
        out.write(
            "Sensitive files are listed with metadata but "
            "content is not included.\n"
        )
        out.write("\n")

    def write_file_tree(self, out, tree_lines: List[str]) -> None:
        """Write project file tree."""
        out.write("=" * 80 + "\n")
        out.write("PROJECT STRUCTURE\n")
        out.write("=" * 80 + "\n\n")

        out.write(f"{self.project_root.name}/\n")
        for line in tree_lines:
            out.write(line + "\n")

        out.write("\n")

    def write_source_files_header(self, out) -> None:
        """Writes the header for the source files section."""
        out.write("=" * 80 + "\n")
        out.write("SOURCE FILES\n")
        out.write("=" * 80 + "\n\n")

    def write_sensitive_file(
        self,
        out,
        file_path: Path,
        file_stat: os.stat_result,
        info: Dict,
        language: str,
    ) -> None:
        """Write sensitive file metadata without content."""
        rel_path = file_path.relative_to(self.project_root)

        out.write("\n" + "-" * 80 + "\n")
        out.write(f"FILE: {rel_path}\n")
        out.write("-" * 80 + "\n")
        out.write("Type:      SENSITIVE (content not included)\n")
        out.write(f"Location:  {rel_path}\n")
        out.write(f"Size:      {file_stat.st_size} bytes\n")
        out.write(f"Language:  {language}\n")

        if "keys" in info:
            out.write("\nEnvironment Variables:\n")
            for key in info["keys"]:
                out.write(f"  {key}\n")

        out.write("\nNOTE: This is a sensitive file. ")
        out.write("Content is not included for security.\n")
        out.write(
            "      The file exists and should be configured separately.\n"
        )
        out.write("\n")

        logger.info(f"🔒 Sensitive: {rel_path}")

    def write_regular_file(
        self,
        out,
        file_path: Path,
        file_stat: os.stat_result,
        content: str,
        line_count: int,
        language: str,
    ) -> None:
        """Write regular file with content."""
        rel_path = file_path.relative_to(self.project_root)

        out.write("\n" + "-" * 80 + "\n")
        out.write(f"FILE: {rel_path}\n")
        out.write("-" * 80 + "\n")
        out.write(f"Location:   {rel_path}\n")
        out.write(f"Language:   {language}\n")
        out.write(f"Lines:      {line_count}\n")
        out.write(f"Size:       {file_stat.st_size} bytes\n")
        out.write("-" * 80 + "\n\n")

        out.write(content)

        if not content.endswith("\n"):
            out.write("\n")

        out.write("\n")

        logger.debug(f"✓ Included: {rel_path} ({line_count} lines)")

    def write_error(self, out, rel_path: Path, error: Exception) -> None:
        """Writes an error message for a file that couldn't be read."""
        out.write(f"\nERROR: Unable to read file: {error}\n\n")
        logger.error(f"✗ Error reading {rel_path}: {error}")

    def write_statistics(self, out, timestamp: datetime, stats: Dict) -> None:
        """Write consolidation statistics with a visual summary table."""
        out.write("=" * 80 + "\n")
        out.write("CONSOLIDATION STATISTICS\n")
        out.write("=" * 80 + "\n\n")

        out.write(f"Completion Time:   {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Project Name:      {PROJECT_NAME}\n")
        out.write("-" * 40 + "\n")
        
        # Table format
        table_fmt = "| {:<25} | {:>10} |\n"
        out.write(table_fmt.format("Metric", "Value"))
        out.write("|" + "-" * 27 + "|" + "-" * 12 + "|\n")
        out.write(table_fmt.format("Total Files Scanned", stats['total_files']))
        out.write(table_fmt.format("Files Included", stats['included_files']))
        out.write(table_fmt.format("Files Excluded", stats['excluded_files']))
        out.write(table_fmt.format("Sensitive Files", stats['sensitive_files']))
        out.write(table_fmt.format("Total Lines of Code", f"{stats['total_lines']:,}"))
        out.write("-" * 40 + "\n\n")

        out.write("Language Distribution:\n")
        sorted_langs = sorted(
            stats["languages"].items(), key=lambda x: x[1], reverse=True
        )
        for lang, count in sorted_langs:
            out.write(f"  {lang:25s} {count:4d} files\n")

        out.write("\n")
        out.write("=" * 80 + "\n")
        out.write("END OF CONSOLIDATION\n")
        out.write("=" * 80 + "\n")


class ProjectConsolidator:
    """Consolidates project source code into a single auditable file."""

    def __init__(self, project_root: Path, list_env_keys: bool = True) -> None:
        self.project_root = project_root
        self.list_env_keys = list_env_keys
        self.stats: Dict[str, int | Dict[str, int]] = {
            "total_files": 0,
            "included_files": 0,
            "excluded_files": 0,
            "sensitive_files": 0,
            "total_lines": 0,
            "languages": {},
        }
        self.file_tree: List[str] = []
        self._file_stats_cache: Dict[Path, os.stat_result] = {}
        self._output_file: Optional[Path] = None
        self.report_generator = ReportGenerator(self.project_root)
        self.file_walker = FileWalker(self.project_root)
        self.git_info_provider = GitInfoProvider(self.project_root)

    def _get_file_stat(self, file_path: Path) -> Optional[os.stat_result]:
        """Get file stat with caching."""
        if file_path not in self._file_stats_cache:
            try:
                self._file_stats_cache[file_path] = file_path.stat()
            except OSError as e:
                logger.error(f"Error accessing {file_path}: {e}")
                return None
        return self._file_stats_cache[file_path]

    @staticmethod
    def is_sensitive_file(file_path: "Path | str") -> bool:
        """Check if file contains sensitive information."""
        file_str = str(file_path)
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, file_str, re.IGNORECASE):
                return True
        return False

    def analyze_sensitive_file(
        self,
        file_path: Path,
        list_env_keys: bool = True,
    ) -> Dict[str, "str | int | List[str] | bool"]:
        """Analyze sensitive file without exposing content."""
        info = {
            "exists": True,
            "size": file_path.stat().st_size,
            "type": self.file_walker.get_file_language(file_path),
        }

        if file_path.name.startswith(".env") and self.list_env_keys:
            try:
                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()

                keys = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key = line.split("=")[0].strip()
                        if len(key) > 8:
                            redacted = f"{key[:4]}...{key[-2:]}={{Exists}}"
                        else:
                            redacted = f"{key}={{Exists}}"
                        keys.append(redacted)

                info["keys"] = keys
            except OSError as e:
                logger.error(f"Error reading sensitive file {file_path}: {e}")
                info["keys"] = ["<Unable to read>"]

        return info

    def consolidate(self, output_file: Path) -> None:
        """Consolidate all project files into output file."""
        logger.info("Starting project consolidation...")
        logger.info(f"Project: {self.project_root}")
        logger.info(f"Output:  {output_file}")

        git_info = self.git_info_provider.get_git_info()
        timestamp = datetime.now()

        try:
            try:
                self._output_file = output_file.resolve()
            except Exception:
                self._output_file = output_file

            with open(output_file, "w", encoding="utf-8") as out:
                self.report_generator.write_header(out, timestamp, git_info)

                tree_lines = self.file_walker.build_file_tree(
                    self.project_root
                )
                self.report_generator.write_file_tree(out, tree_lines)

                self._process_files(out)

                self.report_generator.write_statistics(
                    out, timestamp, self.stats
                )

            logger.info("Consolidation complete!")
            logger.info(f"Output file: {output_file}")
            logger.info(f"Total files processed: {self.stats['total_files']}")
            logger.info(f"Files included: {self.stats['included_files']}")
            logger.info(f"Total lines: {self.stats['total_lines']:,}")

        except OSError as e:
            logger.error(f"Error writing to output file {output_file}: {e}")
            raise

    def _is_output_file(self, file_path: Path) -> bool:
        """Check if a file is the output file, a previously generated report,
        or a protected file that must never be modified."""
        # Skip this script itself
        if file_path.name == Path(__file__).name:
            return True
        # Skip protected files (e.g. llms.txt = Deriv API reference)
        if file_path.name in PROTECTED_FILES:
            return True
        # Skip the current output file
        try:
            if self._output_file and file_path.resolve() == self._output_file:
                return True
        except Exception:
            pass
        # Skip previously generated consolidated reports
        if OUTPUT_FILE_REGEX.match(file_path.name):
            return True
        return False

    def _process_ipynb(self, file_path: Path) -> str:
        """Extract code cells from Jupyter Notebook."""
        import json
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            
            output = []
            output.append(f"# Jupyter Notebook: {file_path.name}")
            rel_path = file_path.relative_to(self.project_root).as_posix()
            related_script = NOTEBOOK_SCRIPT_REFERENCES.get(rel_path)
            if related_script:
                output.append(
                    f"# Related local training script: {related_script}"
                )
            output.append(f"# Total Cells: {len(nb.get('cells', []))}")
            output.append("-" * 40)
            
            for i, cell in enumerate(nb.get("cells", [])):
                cell_type = cell.get("cell_type", "unknown")
                if cell_type == "code":
                    source = "".join(cell.get("source", []))
                    if source.strip():
                        output.append(f"\n[Code Cell #{i+1}]")
                        output.append(source)
                elif cell_type == "markdown":
                    source = "".join(cell.get("source", []))
                    if source.strip():
                        # Redact long markdown for brevity if desired, 
                        # but usually it's fine.
                        output.append(f"\n[Markdown Cell #{i+1}]")
                        # Prepend with comment for clarity in consolidated view
                        markdown_lines = ["# " + line for line in source.split("\n")]
                        output.append("\n".join(markdown_lines))
            
            return "\n".join(output)
        except Exception as e:
            return f"ERROR extracting Jupyter Notebook {file_path.name}: {str(e)}"

    def _process_files(self, out) -> None:
        """Process and write all files."""
        self.report_generator.write_source_files_header(out)

        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = sorted(
                d for d in dirs if not self.file_walker.is_excluded_dir(d)
            )

            root_path = Path(root)

            for file in sorted(files):
                file_path = root_path / file

                # Skip the consolidation script itself and output files
                if self._is_output_file(file_path):
                    continue

                self.stats["total_files"] += 1

                file_stat = self._get_file_stat(file_path)
                if file_stat is None:
                    self.stats["excluded_files"] += 1
                    continue

                file_size = file_stat.st_size

                if self.file_walker.is_excluded_file(file_path, file_size):
                    self.stats["excluded_files"] += 1
                    continue

                if self.is_sensitive_file(file_path):
                    info = self.analyze_sensitive_file(
                        file_path, self.list_env_keys
                    )
                    language = self.file_walker.get_file_language(file_path)
                    self.report_generator.write_sensitive_file(
                        out, file_path, file_stat, info, language
                    )
                    self.stats["sensitive_files"] += 1
                    continue

                # Write regular file
                try:
                    if file_path.suffix.lower() == ".ipynb":
                        content = self._process_ipynb(file_path)
                    else:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()

                    lines = content.split("\n")
                    line_count = len(lines)
                    self.stats["total_lines"] += line_count

                    language = self.file_walker.get_file_language(file_path)
                    lang_stats = self.stats["languages"]
                    lang_stats[language] = lang_stats.get(language, 0) + 1

                    self.report_generator.write_regular_file(
                        out,
                        file_path,
                        file_stat,
                        content,
                        line_count,
                        language,
                    )
                    self.stats["included_files"] += 1
                except (OSError, UnicodeDecodeError, ValueError) as e:
                    self.report_generator.write_error(
                        out, file_path.relative_to(self.project_root), e
                    )

        logger.info(f"Processed {self.stats['total_files']} files")


def ensure_gitignore_entry(update_gitignore: bool = True) -> None:
    """Ensure the output file pattern is in .gitignore."""
    if not update_gitignore:
        logger.debug("Skipping .gitignore update (disabled by user)")
        return

    gitignore_path = PROJECT_ROOT / ".gitignore"
    entries_to_add = [OUTPUT_FILE_PATTERN]

    try:
        if gitignore_path.exists():
            with open(gitignore_path, encoding="utf-8") as f:
                content = f.read()

            missing = [e for e in entries_to_add if e not in content]
            if not missing:
                logger.debug(".gitignore already contains all output patterns")
                return

            with open(gitignore_path, "a", encoding="utf-8") as f:
                if not content.endswith("\n"):
                    f.write("\n")
                f.write("\n# Exclude consolidated source snapshots\n")
                for entry in missing:
                    f.write(f"{entry}\n")

            logger.info(f"Added {missing} to .gitignore")
        else:
            with open(gitignore_path, "w", encoding="utf-8") as f:
                f.write("# Exclude consolidated source snapshots\n")
                for entry in entries_to_add:
                    f.write(f"{entry}\n")

            logger.info("Created .gitignore with output patterns")

    except OSError as e:
        logger.warning(f"Could not update .gitignore: {e}")


def detect_project_root() -> Path:
    """Detect the project root directory by looking for common markers."""
    current = Path(__file__).parent.absolute()

    root_markers = {
        ".git",
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "Cargo.toml",
        "go.mod",
        ".gitignore",
        "Makefile",
    }

    while current.parent != current:
        for marker in root_markers:
            if (current / marker).exists():
                logger.debug(f"Detected project root via {marker}: {current}")
                return current
        current = current.parent

    fallback = Path(__file__).parent.absolute()
    logger.debug(
        f"No project root markers found, using script directory: {fallback}"
    )
    return fallback


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} — Consolidate project source code into a single file for auditing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run with defaults → timestamped snapshot
  %(prog)s --output snapshot.txt    # Specify custom output file
  %(prog)s --verbose                # Enable verbose logging
  %(prog)s --project-root /path     # Specify project root directory
  %(prog)s --no-update-gitignore    # Skip .gitignore update
        """,
    )

    parser.add_argument(
        "--output",
        type=Path,
        help=f"Output file path (default: {OUTPUT_FILE_PATTERN})",
        metavar="FILE",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory (default: auto-detect)",
        metavar="DIR",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--no-update-gitignore",
        action="store_true",
        help="Don't automatically update .gitignore",
    )

    parser.add_argument(
        "--no-list-env-keys",
        action="store_true",
        help="Don't list .env file keys (more secure)",
    )

    parser.add_argument(
        "--max-file-size",
        type=int,
        default=MAX_FILE_SIZE,
        help=(
            "Maximum file size to include in bytes "
            f"(default: {MAX_FILE_SIZE:,})"
        ),
        metavar="BYTES",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s 2.0 ({PROJECT_NAME})",
    )

    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.project_root:
        project_root = args.project_root.absolute()
    else:
        project_root = detect_project_root()

    if not project_root.exists():
        logger.error(f"Project root does not exist: {project_root}")
        return 1

    if args.output:
        output_path = args.output
        # Guard: never allow overwriting protected files
        if output_path.name in PROTECTED_FILES:
            logger.error(
                f"'{output_path.name}' is a protected file and cannot be used "
                "as output. Choose a different filename."
            )
            return 1
    else:
        timestamp_tag = datetime.now()
        project_name = project_root.name.replace(" ", "_").lower()
        date_str = timestamp_tag.strftime("%Y%m%d_%H%M")
        output_filename = f"{project_name}_{date_str}_merged_sources.txt"
        output_path = project_root / output_filename

    logger.info("=" * 80)
    logger.info(f"{PROJECT_NAME.upper()} — PROJECT SOURCE CONSOLIDATION TOOL")
    logger.info("=" * 80)
    logger.info(f"Protected files (never overwritten): {PROTECTED_FILES}")

    global MAX_FILE_SIZE
    if args.max_file_size != MAX_FILE_SIZE:
        MAX_FILE_SIZE = args.max_file_size
        logger.info(f"Using custom max file size: {MAX_FILE_SIZE:,} bytes")

    ensure_gitignore_entry(update_gitignore=not args.no_update_gitignore)

    consolidator = ProjectConsolidator(
        project_root, list_env_keys=not args.no_list_env_keys
    )

    try:
        consolidator.consolidate(output_path)

        logger.info("=" * 80)
        logger.info("CONSOLIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Output File:        {output_path}")
        logger.info(
            f"File Size:          {output_path.stat().st_size:,} bytes"
        )
        logger.info(f"Total Files:        {consolidator.stats['total_files']}")
        logger.info(
            f"Included:           {consolidator.stats['included_files']}"
        )
        logger.info(
            f"Excluded:           {consolidator.stats['excluded_files']}"
        )
        logger.info(
            f"Sensitive:          {consolidator.stats['sensitive_files']}"
        )
        logger.info(
            f"Total Lines:        {consolidator.stats['total_lines']:,}"
        )

        logger.info("\nTop Languages:")
        sorted_langs = sorted(
            consolidator.stats["languages"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        for lang, count in sorted_langs:
            logger.info(f"  {lang:25s} {count:4d} files")

        logger.info("\n✅ Consolidation completed successfully!")
        return 0

    except Exception as e:
        logger.exception("Error during consolidation: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
