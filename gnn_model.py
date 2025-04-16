import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np
from itertools import product

# Generate synthetic graph data for demonstration purposes
def generate_graph_data(num_nodes):
    # Randomly generate node features and edge indices
    x = torch.rand((num_nodes, 3), dtype=torch.float)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), dtype=torch.long)
    return x, edge_index

# Generate synthetic labels for demonstration purposes
def generate_labels(num_samples):
    return np.random.randint(0, 2, num_samples)

# Prepare graph data
num_nodes = 1000
x, edge_index = generate_graph_data(num_nodes)
labels = generate_labels(num_nodes)

# Create a PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index, y=torch.tensor(labels, dtype=torch.long))

# Split data into training and testing sets
train_mask, test_mask = train_test_split(np.arange(num_nodes), test_size=0.2, random_state=42)

# Define an enhanced GNN model with GraphSAGE layers
class EnhancedGNNModel(torch.nn.Module):
    def __init__(self, hidden_units=64, dropout_rate=0.5):
        super(EnhancedGNNModel, self).__init__()
        self.conv1 = SAGEConv(3, hidden_units)
        self.conv2 = SAGEConv(hidden_units, hidden_units)
        self.conv3 = SAGEConv(hidden_units, hidden_units)
        self.conv4 = SAGEConv(hidden_units, 2)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize the enhanced model, optimizer, and loss function
model = EnhancedGNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Implement a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Implement early stopping
best_acc = 0
patience = 10
patience_counter = 0

# Train the model
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluate the model
def test():
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[test_mask].eq(data.y[test_mask]).sum().item())
    acc = correct / len(test_mask)
    return acc

# Training loop with learning rate scheduling and early stopping
for epoch in range(1, 101):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')
    
    # Step the optimizer before the scheduler
    optimizer.step()
    scheduler.step()
    
    # Check for early stopping
    if acc > best_acc:
        best_acc = acc
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print('Early stopping triggered')
        break

# Define a function to perform hyperparameter tuning
def hyperparameter_tuning():
    best_acc = 0
    best_params = None
    learning_rates = [0.01, 0.005]
    hidden_units = [64, 128]
    dropout_rates = [0.5, 0.6]

    for lr, hu, dr in product(learning_rates, hidden_units, dropout_rates):
        model = EnhancedGNNModel(hidden_units=hu, dropout_rate=dr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        for epoch in range(1, 51):
            loss = train()
            acc = test()
            scheduler.step()

            if acc > best_acc:
                best_acc = acc
                best_params = (lr, hu, dr)

    print(f'Best Accuracy: {best_acc:.4f} with params: Learning Rate={best_params[0]}, Hidden Units={best_params[1]}, Dropout Rate={best_params[2]}')

# Define an ensemble of GNN models
class EnsembleGNNModel(torch.nn.Module):
    def __init__(self, num_models=3):
        super(EnsembleGNNModel, self).__init__()
        self.models = torch.nn.ModuleList([EnhancedGNNModel() for _ in range(num_models)])

    def forward(self, data):
        outputs = [model(data) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# Initialize the ensemble model
ensemble_model = EnsembleGNNModel()
optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=0.01)

# Train and evaluate the ensemble model
for epoch in range(1, 101):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

    if acc > best_acc:
        best_acc = acc
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print('Early stopping triggered')
        break

# Perform hyperparameter tuning
hyperparameter_tuning()

# Define a GAT model
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

# Initialize the GAT model
model = GATModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop with learning rate scheduling and early stopping
for epoch in range(1, 101):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')
    
    # Step the optimizer before the scheduler
    optimizer.step()
    scheduler.step()
    
    # Check for early stopping
    if acc > best_acc:
        best_acc = acc
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print('Early stopping triggered')
        break 