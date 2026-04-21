### File Descriptions

| File | Description | Used For |
|------|-------------|----------|
| `method.csv` | Method identifiers and metadata | Node list |
| `metrics.csv` | Code metrics per method (LOC, CC, PC, etc.) | Node features X |
| `method-invocate-method.csv` | Method call relationships | Edge list A |
| `ground_truth.csv` | Labels and refactoring targets | Labels y, targets |
| `classinfo.csv` | Class identifiers and metadata | Refactoring mapping |
| `methodInfo.csv` | Rich method metadata and names | Semantic embeddings |
| `method_tokens.pkl` | Tokenized method names | SFFL semantic input |
| `class_tokens.pkl` | Tokenized class names | SFFL semantic input |

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

### Step 1 — Explore the data (optional)
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Step 2 — Preprocess

Process a single project:
```bash
python scripts/preprocess.py --project activemq
```

Process all projects:
```bash
python scripts/preprocess.py --all
```

### Step 3 — Train
```bash
python scripts/train.py --project activemq --config configs/default.yaml
```

### Step 4 — Evaluate
```bash
python scripts/evaluate.py --project activemq
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_preprocessor.py -v
```

---

## Results

*To be updated as experiments are completed.*

| Project | Precision1 | Recall1 | F1-Score1 | Refactoring Acc |
|---------|------------|---------|-----------|-----------------|
| ActiveMQ | - | - | - | - |
| Alluxio | - | - | - | - |
| BinNavi | - | - | - | - |
| Kafka | - | - | - | - |
| Realm-java | - | - | - | - |
| **Average** | **-** | **-** | **-** | **-** |

*Results reported as mean ± std across 5 random seeds.*

---

## Reproducing Results

All experiments use fixed random seeds for reproducibility:

```bash
# Runs experiments across all 5 seeds and reports mean +- std
python scripts/train.py --project activemq --seeds 1 2 3 4 5
```

---

## Key Design Decisions

**Why GNNs?** Feature envy is a relational smell — it's defined by how a method
relates to other methods and classes. GNNs operate directly on graph structure,
capturing interaction patterns that simple metrics cannot.

**Why GraphSMOTE?** Smelly methods are rare (3–10% of all methods). Without
imbalance handling, the model collapses to predicting all-negative. GraphSMOTE
generates synthetic smelly nodes in the graph's embedding space to balance
training.

**Why stratified splits?** With low positive rates, random splits can produce
training sets with almost no positive examples. Stratified splitting preserves
the positive rate across train/val/test.

---

## References

1. Yu et al. (2025). Efficient Feature Envy Detection and Refactoring Based on
   Graph Neural Network. *Automated Software Engineering*, 32(7).
   https://doi.org/10.1007/s10515-024-00476-3

2. Sharma & Kessentini (2021). Qscored: A large dataset of code smells and
   quality metrics. *MSR 2021*.
   https://doi.org/10.5281/zenodo.4468361

3. Hamilton et al. (2017). Inductive Representation Learning on Large Graphs
   (GraphSAGE). *NeurIPS 2017*.

4. Zhao et al. (2021). GraphSMOTE: Imbalanced Node Classification on Graphs
   with Graph Neural Networks. *WSDM 2021*.

---

## Team

**Group 4 — DSCI 644.02, Software Engineering for Data Science**

---

## Phase Reports

| Phase | Description | Report |
|-------|-------------|--------|
| Phase 1 | Problem definition, dataset strategy, RQs | [Phase 1 Report](reports/phase1_report.pdf) |
| Phase 2 | Baseline pipeline implementation | [Phase 2 Report](reports/phase2_report.pdf) |
| Phase 3 | Optimized solution and results | [Phase 3 Report](reports/phase3_report.pdf) |