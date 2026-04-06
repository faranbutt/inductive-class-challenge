# 🎗️ The THInC Challenge: Tumor Histopathology Inductive Classification

## Welcome to the Tumor Diagnosis Challenge 🚀

This competition bridges **Biomedical Engineering** and **Graph Machine Learning**. Your task is to build a model that can diagnose the type of a cell (Tumor, Stroma, Immune, etc.) based on its features and its spatial neighbors in a tissue biopsy. Standard AI often fails in clinics because it memorizes training data, so our challenge forces models to predict cell types inductively, meaning they must generalize to completely unseen patients.

---

## 🎯 The Task: Inductive Node Classification

You are provided with **cell graphs** constructed from H&E stained histology images.

- **Training Phase:** You receive full graphs (nodes, edges, and labels) from a set of patients (e.g., Patient A, Patient B).
- **Testing Phase:** You must predict the cell types for **completely unseen patients** (e.g., Patient C).

### 🔍 Why "Inductive"?

Unlike standard transductive tasks (like Cora), the test nodes belong to **entirely new graphs** (new tissue slides) that were not present during training. Your model must learn **general rules** about tissue organization, not just memorize a specific graph structure.

---

## 📂 The Dataset (NuCLS-Based)

The data is derived from the [NuCLS dataset](https://nucls.grand-challenge.org/) (breast cancer).

### 🏗️ Graph Construction Pipeline

The graph was built using the following inductive pipeline to ensure biological realism:
```
[ Histology Image ]  -->  [ Cell Detection ]  -->  [ Graph Construction ]
       🖼️                       📍                        🕸️
   (Raw Pixels)            (Centroids x,y)           (Nodes + Edges)
                                                            |
                                                    (k-NN Neighbors)
```

| Component | Description |
|-----------|-------------|
| **Node Definition** | Raw bounding boxes from NuCLS were converted into centroids `(x, y)`. Each node represents a single cell nucleus. |
| **Edge Construction (k-NN)** | For every cell, we computed its **5 nearest spatial neighbors** within the same tissue image. Edges represent the local tissue microenvironment (e.g., cell–cell interactions). Edges strictly connect cells within the same image — no edges between different patients. |
| **Inductive Split** | Dataset was split by **Image ID**, not by random cells. Training graph = 80% of tissue images. Test graph = remaining 20% (completely unseen patients). |
| **Feature Normalization** | Node features `(x, y, width, height)` are standardized (zero mean, unit variance) for stable GNN training. |

---

## 1. Graph Components

- **Nodes:** Individual cells (nuclei)
- **Edges:** Spatial proximity via k-Nearest Neighbors (`k=5`). Physically close cells are connected.

### Node Features (X)

| Feature | Description |
|---------|-------------|
| `x`, `y` | Normalized spatial coordinates |
| `width`, `height` | Morphological features of the nucleus |

### Labels (Y)

| Label | Cell Type | Description |
|-------|-----------|-------------|
| `0` | Tumor | Malignant cells |
| `1` | Stromal | Connective tissue |
| `2` | Lymphocyte | Immune cells |
| `3` | Macrophage | Immune cells |

---

## 2. File Structure (`data/public/`)

| File | Description | Columns |
|------|-------------|---------|
| `train.csv` | Training nodes with labels | `id, x, y, width, height, label` |
| `edge_list.csv` | Edges for the training graph | `source, target` |
| `test_nodes.csv` | Unseen test nodes (no labels) | `id, x, y, width, height` |
| `test_edges.csv` | Edges for the test graph | `source, target` |
| `sample_submission.csv` | Example submission format | — |

---

## 📝 Submission Format

Create a single CSV file named strictly using your team name: `<team_name>.csv` (e.g., `emmanuel_owusu.csv`).

```csv
id,y_pred
41269,0
41270,2
...
```

**Requirements:**
- Header must be exactly: `id,y_pred`
- One row per test node
- Labels must be integers in `[0–3]`

---

## 🚀 How to Participate

### 1️⃣ Clone the Repository & Install Dependencies
```bash
git clone https://github.com/emmakowu3579-ui/inductive-class-challenge.git
cd inductive-class-challenge
pip install -r starter_code/requirements.txt
```

### 2️⃣ Run the Baseline Model

A simple PyTorch GCN baseline is provided in the `starter_code/` directory.
```bash
python starter_code/baseline.py
```

This will:
- Train a basic GCN on the training graph
- Generate a submission file at `submissions/baseline_submission.csv`


### 3️⃣ Encrypt Your Submission

To preserve privacy, you **must encrypt** your CSV file before uploading. Do not upload raw CSV files.

```bash
# Usage: python starter_code/encrypt.py <path_to_your_team_csv>
python starter_code/encrypt.py submissions/<team_name>.csv

# Example:
# python starter_code/encrypt.py submissions/emmanuel_owusu.csv
```



### 4️⃣ Submit via GitHub

> ⚠️ **IMPORTANT:** Do **NOT** commit the raw `.csv` file. Ensure you are only committing the `.enc` file with your team name.

```bash
git add submissions/<team_name>.csv.enc
git commit -m "Add encrypted submission for <team_name>"
git push origin <your-branch-name>

# Example:
# git add submissions/emmanuel_owusu.csv.enc
# git commit -m "Add encrypted submission for emmanuel_owusu"
# git push origin main
```
Then open a **Pull Request** against the `main` branch on GitHub.

---

## 🤖 Instant Grading

Once your PR is opened:

- ✅ An **Auto-Grader Bot** runs automatically (it decrypts your file securely)
- 📊 Your **Macro F1-Score** is computed
- 💬 The score is posted as a **comment on your PR**

If the submission is valid, the PR will be merged by an admin and 🎉 your name appears on the **Leaderboard**.

---

## 📏 Rules & Restrictions

| Rule | Detail |
|------|--------|
| **Evaluation Metric** | Macro F1-Score |
| **Inductive Setting** | No access to test labels during training. No memorization of node IDs or embeddings. |
| **Message Passing** | Allowed only on the training graph during training. Test edges may be used only at inference time. |
| **External Data** | ❌ Strictly forbidden |
| **Runtime Constraint** | Training must finish in **< 5 minutes** on Google Colab (CPU/GPU) |
| **Libraries** | Any standard GNN library (PyTorch, PyTorch Geometric, DGL, etc.) |

---

## 🏆 Leaderboard
[📈 View Leaderboard](LEADERBOARD.md)
