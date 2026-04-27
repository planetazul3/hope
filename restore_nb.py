import json
import os

def make_notebook(title, cells, env='colab'):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    if env == 'colab':
        nb["metadata"]["colab"] = {"provenance": []}
    return nb

def make_markdown_cell(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content if isinstance(content, list) else [content]
    }

def make_code_cell(content):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content if isinstance(content, list) else [content]
    }

# --- Colab Notebook ---
colab_cells = [
    make_markdown_cell([
        "# Hope ML: Cloud Training Pipeline (Colab)\n",
        "\n",
        "This notebook trains the Canonical Causal Transformer model with TS2Vec pre-training and exports the ONNX artifact.\n",
        "Requires `ticks.csv` to be uploaded or mounted from Google Drive."
    ]),
    make_code_cell([
        "from google.colab import drive\n",
        "import os\n",
        "\n",
        "# 1. Mount Drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# 2. Install Dependencies\n",
        "!pip install torch==2.11.0 pandas==3.0.2 numpy==2.4.4 scikit-learn==1.8.0 tqdm==4.67.3 cryptography==42.0.5 onnx==1.21.0 onnxruntime==1.20.1\n",
        "\n",
        "# 3. Set up Path and Run Training\n",
        "import sys\n",
        "sys.path.append('/content/drive/MyDrive/hope/scripts')\n",
        "\n",
        "import train_fixed\n",
        "train_fixed.main()\n"
    ])
]

# --- Kaggle Notebook ---
kaggle_cells = [
    make_markdown_cell([
        "# Hope ML: Cloud Training Pipeline (Kaggle)\n",
        "\n",
        "This notebook trains the Canonical Causal Transformer model with TS2Vec pre-training and exports the ONNX artifact.\n",
        "Requires `ticks.csv` as an input dataset."
    ]),
    make_code_cell([
        "!rm -rf hope\n",
        "!git clone https://github.com/planetazul3/hope.git\n",
        "%cd hope\n",
        "!git pull origin main\n"
    ]),
    make_code_cell([
        "!pip install -r requirements.txt\n"
    ]),
    make_code_cell([
        "import os, sys\n",
        "import torch\n",
        "from kaggle_secrets import UserSecretsClient\n",
        "\n",
        "assert torch.cuda.is_available(), \"CRITICAL: GPU not detected. Aborting to save quota.\"\n",
        "\n",
        "try:\n",
        "    user_secrets = UserSecretsClient()\n",
        "    os.environ[\"MODEL_SIGNING_KEY\"] = user_secrets.get_secret(\"MODEL_SIGNING_KEY\")\n",
        "    print(\"Ed25519 Signing Key loaded securely.\")\n",
        "except Exception as e:\n",
        "    print(\"WARNING: MODEL_SIGNING_KEY not found in Kaggle Secrets. Model signing will fail or skip.\")\n"
    ]),
    make_code_cell([
        "# Ensure local scripts can be imported\n",
        "repo_root = '/kaggle/working/hope'\n",
        "sys.path.append(os.path.join(repo_root, 'scripts'))\n",
        "\n",
        "import train_fixed\n",
        "# Pass Kaggle's read-only dataset path directly\n",
        "train_fixed.main(csv_path=\"/kaggle/input/ticks-csv/ticks.csv\")\n"
    ]),
    make_code_cell([
        "# Move artifacts to /kaggle/working so they can be downloaded\n",
        "!cp model.onnx model.onnx.sig /kaggle/working/\n",
        "print(\"Artifacts successfully moved to working directory for download.\")\n"
    ])
]

if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    
    colab_nb = make_notebook("Hope Colab Training", colab_cells, env='colab')
    with open("notebooks/colab_training.ipynb", "w") as f:
        json.dump(colab_nb, f, indent=1)
    print("Restored notebooks/colab_training.ipynb")
    
    kaggle_nb = make_notebook("Hope Kaggle Training", kaggle_cells, env='kaggle')
    with open("notebooks/kaggle_training.ipynb", "w") as f:
        json.dump(kaggle_nb, f, indent=1)
    print("Restored notebooks/kaggle_training.ipynb")
    
    # Remove deprecated notebook if it exists
    deprecated_nb = "notebooks/train_transformer.ipynb"
    if os.path.exists(deprecated_nb):
        os.remove(deprecated_nb)
        print(f"Removed deprecated notebook: {deprecated_nb}")
