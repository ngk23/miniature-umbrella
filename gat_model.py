import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np

def generate_graph_data(num_nodes):
    x = torch.rand((num_nodes, 3), dtype=torch.float)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), dtype=torch.long)
    return x, edge_index

def generate_labels(num_samples):
    return np.random.randint(0, 2, num_samples)
num_nodes = 1000
x, edge_index = generate_graph_data(num_nodes)
labels = generate_labels(num_nodes)
data = Data(x=x, edge_index=edge_index, y=torch.tensor(labels, dtype=torch.long))
train_mask, test_mask = train_test_split(np.arange(num_nodes), test_size=0.2, random_state=42)

class GATModel(torch.nn.Module):
    def __init__(self, hidden_units=64, dropout_rate=0.5):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(3, hidden_units, heads=4, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_units * 4, hidden_units, heads=4, dropout=dropout_rate)
        self.conv3 = GATConv(hidden_units * 4, 2, heads=1, concat=False, dropout=dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GATModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
best_acc = 0
patience = 10
patience_counter = 0

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[test_mask].eq(data.y[test_mask]).sum().item())
    acc = correct / len(test_mask)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')
    
    if acc > best_acc:
        best_acc = acc
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print('Early stopping triggered')
        break 