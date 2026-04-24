# Feature Envy Detection using Graph Neural Networks

A reproducible pipeline for detecting and refactoring Feature Envy code smells
using Graph Neural Networks, based on the paper:

> "Efficient Feature Envy Detection and Refactoring Based on Graph Neural Network"
> Yu et al., Automated Software Engineering (2025)
> DOI: https://doi.org/10.1007/s10515-024-00476-3

**Course:** DSCI 644.02 — Software Engineering for Data Science
**Team:** Group 4

---

## What is Feature Envy?

Feature Envy is a code smell where a method interacts more with an external
class than its own class — suggesting the method belongs elsewhere. This
pipeline automates:

1. **Detection** — identifying which methods have feature envy
2. **Refactoring recommendation** — suggesting which class the method should
   move to (Move Method refactoring)

---

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourname/feature-envy-gnn
cd feature-envy-gnn
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install PyTorch
```bash
pip install torch==2.2.2
```

### 4. Install PyTorch Geometric

**Mac (Python 3.12+):**
```bash
pip install torch-geometric
```

**Linux/Windows with CUDA 11.8:**
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
```

**Linux/Windows with CUDA 12.1:**
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

### 5. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 6. Verify installation
```bash
python -c "
import torch
import torch_geometric
print(f'PyTorch:           {torch.__version__}')
print(f'PyTorch Geometric: {torch_geometric.__version__}')
print(f'CUDA available:    {torch.cuda.is_available()}')
print('All good!')
"
```

> **Note:** On Mac with Python 3.12, `torch-scatter`, `torch-sparse`,
> and `torch-cluster` are bundled into `torch-geometric` and do not
> need to be installed separately. NumPy must be below version 2.0
> for compatibility with PyTorch 2.2.2 — this is handled automatically
> by `requirements.txt`.

---

## Data

### Download

The dataset is sourced from:
> Sharma, T., Kessentini, M.: Qscored — A large dataset of code smells
> and quality metrics. MSR 2021.
> DOI: https://doi.org/10.5281/zenodo.4468361

Download the dataset and place files under `data/raw/` following
the structure shown above.

### File Descriptions

| File | Description | Used For |
|------|-------------|----------|
| `ground_truth.csv` | Method IDs, labels, source and target class | Labels, refactoring targets |
| `metrics.csv` | 7 code metrics per method (CC, PC, LOC, etc.) | Node features X |
| `method-invocate-method.csv` | Method call relationships | Edge list A |
| `method.csv` | Method identifiers and metadata | Node list |
| `classinfo.csv` | Class identifiers and metadata | Class mapping |
| `methodInfo.csv` | Rich method metadata | Semantic info |
| `method_tokens.pkl` | Tokenized method names | SFFL (future) |
| `class_tokens.pkl` | Tokenized class names | SFFL (future) |

### Dataset Statistics

| Project | Classes | Methods | Smelly | Smell Rate |
|---------|---------|---------|--------|------------|
| ActiveMQ | 2,907 | 15,482 | 585 | 3.78% |
| Alluxio | 1,501 | 7,019 | 367 | 5.23% |
| BinNavi | 3,331 | 10,852 | 996 | 9.18% |
| Kafka | 1,819 | 9,818 | 446 | 4.54% |
| Realm-java | 372 | 2,466 | 260 | 10.54% |

---

## Running the Pipeline

The pipeline consists of five sequential steps. Each step must be
completed before the next can begin.

---

### Step 1 — Data Exploration (optional)

Before preprocessing, explore the raw data to understand column names,
check for nulls, and verify ID alignment across files.

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Key checks performed:
- Column names in each CSV file
- Null values across all files
- Label distribution (positive rate per project)
- ID consistency across ground_truth.csv, metrics.csv,
  and method-invocate-method.csv

---

### Step 2 — Preprocess

Converts raw CSVs into PyTorch Geometric graph objects.
Must be run once before training.

```bash
# Process a single project
python scripts/preprocess.py --project activemq

# Process all projects
python scripts/preprocess.py --all
```

What this does:
1. **Method index** — maps every method ID to a consistent integer
   index 0 to N-1, saved as `method_to_idx.pt`
2. **Feature matrix X** — extracts 7 code metrics per method,
   normalizes with StandardScaler, saved as `X.pt`
3. **Graph construction** — builds edge index from method call
   relationships, zero edges dropped across all projects
4. **Label vector** — binary labels from `ground_truth.csv`,
   source and target class IDs stored separately
5. **Train/val/test splits** — stratified 60/20/20 splits across
   3 random seeds, saved as separate `graph.pt` files per seed

Expected output per project:

---

### Step 3 — Heuristic Baseline

Runs a deterministic rule-based detector to establish a performance
floor before any learned model is introduced.

```bash
# Run on a single project
python scripts/baseline.py --project activemq

# Run on all projects with optimal threshold
python scripts/baseline.py --all --threshold 0.6
```

Detection rule:
