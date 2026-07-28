"""
Test PyTorch Geometric GPU functionality
Tests graph neural network operations on CUDA device
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
import time

print("=" * 60)
print("PyTorch Geometric GPU Test")
print("=" * 60)

# Check GPU availability
print(f"\n1. GPU Setup:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
else:
    print("   ERROR: CUDA not available!")
    exit(1)

# Create a sample graph (Zachary's Karate Club-style network)
print(f"\n2. Creating sample graph data:")
num_nodes = 1000
num_edges = 5000
num_features = 64
num_classes = 7

# Random edges (undirected graph)
edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
x = torch.randn(num_nodes, num_features)
y = torch.randint(0, num_classes, (num_nodes,))

data = Data(x=x, edge_index=edge_index, y=y)
print(f"   Nodes: {data.num_nodes}")
print(f"   Edges: {data.num_edges}")
print(f"   Features: {data.num_features}")

# Define a simple GNN model
class TestGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

print(f"\n3. Testing GCN (Graph Convolutional Network):")
model = TestGNN(num_features, 128, num_classes)
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Move to GPU
device = torch.device('cuda')
model = model.to(device)
data = data.to(device)
print(f"   Model device: {next(model.parameters()).device}")
print(f"   Data device: {data.x.device}")

# Forward pass on GPU
print(f"\n4. Running forward pass on GPU:")
model.eval()
with torch.no_grad():
    start = time.time()
    out = model(data.x, data.edge_index)
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start

print(f"   Output shape: {out.shape}")
print(f"   Output device: {out.device}")
print(f"   GPU forward pass time: {gpu_time*1000:.2f} ms")
print(f"   ✓ Forward pass successful!")

# Test training step
print(f"\n5. Testing training step (backpropagation):")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_mask = torch.rand(num_nodes) > 0.8
train_mask = train_mask.to(device)

start = time.time()
optimizer.zero_grad()
out = model(data.x, data.edge_index)
loss = F.nll_loss(out[train_mask], data.y[train_mask])
loss.backward()
optimizer.step()
torch.cuda.synchronize()
train_time = time.time() - start

print(f"   Loss: {loss.item():.4f}")
print(f"   Training step time: {train_time*1000:.2f} ms")
print(f"   ✓ Training step successful!")

# Test other GNN architectures
print(f"\n6. Testing other GNN architectures:")

# GAT (Graph Attention Network)
class TestGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

gat_model = TestGAT(num_features, 32, num_classes).to(device)
with torch.no_grad():
    gat_out = gat_model(data.x, data.edge_index)
print(f"   ✓ GAT (Graph Attention Network) works!")

# GraphSAGE
class TestSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

sage_model = TestSAGE(num_features, 128, num_classes).to(device)
with torch.no_grad():
    sage_out = sage_model(data.x, data.edge_index)
print(f"   ✓ GraphSAGE works!")

# Memory usage
print(f"\n7. GPU Memory Usage:")
print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Performance comparison (GPU vs CPU)
print(f"\n8. GPU vs CPU Performance Comparison:")
model_cpu = TestGNN(num_features, 128, num_classes).cpu()
data_cpu = data.cpu()

model_cpu.eval()
with torch.no_grad():
    start = time.time()
    out_cpu = model_cpu(data_cpu.x, data_cpu.edge_index)
    cpu_time = time.time() - start

speedup = cpu_time / gpu_time
print(f"   CPU time: {cpu_time*1000:.2f} ms")
print(f"   GPU time: {gpu_time*1000:.2f} ms")
print(f"   Speedup: {speedup:.2f}x")

print(f"\n" + "=" * 60)
print(f"✓ All PyTorch Geometric GPU tests passed!")
print(f"=" * 60)
