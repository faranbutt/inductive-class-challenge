import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
import random
import matplotlib.pyplot as plt

print("Starting Baseline Tumor-GNN...")

SEED = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 1. CONFIGURATION
TRAIN_PATH = 'data/public/train.csv'
TEST_PATH = 'data/public/test_nodes.csv'
EDGE_PATH = 'data/public/edge_list.csv'
TEST_EDGE_PATH = 'data/public/test_edges.csv'
OUTPUT_PATH = 'submission.csv'

# 2. LOAD DATA
print("   - Loading CSVs...")
try:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    edge_list = pd.read_csv(EDGE_PATH)
    test_edges = pd.read_csv(TEST_EDGE_PATH)
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("   (Make sure you are running this from the repo root!)")
    sys.exit(1)

# 3. PREPARE GRAPH
print("   - Building Adjacency Matrix...")

all_nodes = pd.concat([train_df, test_df], ignore_index=True)
id_map = {row_id: i for i, row_id in enumerate(all_nodes['id'].values)}
num_nodes = len(all_nodes)

all_edges = pd.concat([edge_list, test_edges])

src = [id_map[i] for i in all_edges['source']]
dst = [id_map[i] for i in all_edges['target']]

src.extend(range(num_nodes))
dst.extend(range(num_nodes))

indices = torch.tensor([src, dst], dtype=torch.long)
values = torch.ones(len(src), dtype=torch.float32)

row_sum = torch.zeros(num_nodes)
row_sum.index_add_(0, indices[0], values)
deg_inv_sqrt = row_sum.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
values = values * deg_inv_sqrt[indices[0]] * deg_inv_sqrt[indices[1]]

adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

# 4. PREPARE FEATURES
feat_cols = [c for c in train_df.columns if c not in ['id', 'label', 'y_pred']]

degree_counts = torch.zeros(num_nodes, dtype=torch.float32)
degree_counts.index_add_(0, indices[0], torch.ones(len(src), dtype=torch.float32))
all_nodes['degree'] = degree_counts.numpy()

feat_cols = feat_cols + ['degree']
print(f"   - Using features: {feat_cols}")

features = torch.tensor(all_nodes[feat_cols].values, dtype=torch.float32)

train_indices = [id_map[i] for i in train_df['id']]
train_mask = torch.tensor(train_indices, dtype=torch.long)
train_labels = torch.tensor(train_df['label'].values, dtype=torch.long)

# 5. DEFINE MODEL
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.W1 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(in_feats, h_feats)))
        self.W2 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h_feats, num_classes)))

    def forward(self, adj, x):
        h = torch.spmm(adj, x)
        h = torch.mm(h, self.W1)
        h = F.relu(h)
        h = torch.spmm(adj, h)
        h = torch.mm(h, self.W2)
        return h

# 6. TRAIN
model = GCN(features.shape[1], 32, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

class_counts = np.bincount(train_df['label'].values, minlength=4)
class_weights = class_counts.sum() / (4.0 * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

print(f"   - Class counts: {class_counts.tolist()}")
print(f"   - Class weights: {class_weights.tolist()}")

print("   - Training...")
losses = []

model.train()
for e in range(201):
    logits = model(adj, features)
    loss = F.cross_entropy(logits[train_mask], train_labels, weight=class_weights)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if e % 50 == 0:
        print(f"     Epoch {e}, Loss: {loss.item():.4f}")

plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)
print("   - Saved loss curve to loss_curve.png")

# 7. PREDICT & SAVE
print("   - Generating Predictions...")
model.eval()
with torch.no_grad():
    logits = model(adj, features)
    test_indices = [id_map[i] for i in test_df['id']]
    test_logits = logits[test_indices]
    test_preds = test_logits.argmax(1)

submission = pd.DataFrame({
    'id': test_df['id'],
    'y_pred': test_preds.numpy()
})

submission.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved submission to {OUTPUT_PATH}")
print(f"   (Rows: {len(submission)})")
