import matplotlib.pyplot as plt

# Sample accuracy data for demonstration purposes
# Replace these with actual accuracy data from your model runs
epochs = list(range(1, 12))
gnn_accuracies = [0.4750, 0.4800, 0.4650, 0.4650, 0.4600, 0.4550, 0.4550, 0.4650, 0.4650, 0.4550, 0.5000]
gat_accuracies = [0.4650, 0.4650, 0.4650, 0.4650, 0.4650, 0.4650, 0.4650, 0.4650, 0.4650, 0.4650, 0.4650]

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.plot(epochs, gnn_accuracies, label='GNN Model', marker='o')
plt.plot(epochs, gat_accuracies, label='GAT Model', marker='x')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Model Accuracy over Epochs')
plt.legend()
plt.grid(True)

# Show the plot
plt.show() 