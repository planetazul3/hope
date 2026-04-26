import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "consolidate_project_sources.py"


class ConsolidateProjectSourcesTest(unittest.TestCase):
    def test_includes_notebook_and_train_script_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "snapshot.txt"

            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT_PATH),
                    "--root",
                    str(REPO_ROOT),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
            )

            snapshot = output_path.read_text(encoding="utf-8")

            self.assertIn('<file path="notebooks/colab_training.ipynb"', snapshot)
            self.assertIn(
                '<file path="scripts/hope_ml/common.py"',
                snapshot,
            )


if __name__ == "__main__":
    unittest.main()
