import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# DATASET
def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
# PRUNABLE LAYER
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))
    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)
# MODEL
class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# SPARSITY LOSS
def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            loss += torch.sum(gates)
    return loss
# EVALUATION
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
# SPARSITY %
def calculate_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            pruned += torch.sum(gates < threshold).item()
    return (pruned / total) * 100
# PLOT
def plot_gates(model):
    all_gates = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())
    plt.hist(all_gates, bins=50)
    plt.title("Gate Distribution")
    plt.xlabel("Gate Values")
    plt.ylabel("Frequency")
    plt.savefig("gate_distribution.png")
    plt.show()
# TRAIN
def train_model(lambda_sparse):
    train_loader, test_loader = get_dataloaders()
    model = PrunableNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(3):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) + lambda_sparse * sparsity_loss(model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Lambda {lambda_sparse}] Epoch {epoch+1} Loss: {total_loss:.2f}")
    acc = evaluate(model, test_loader)
    sparsity = calculate_sparsity(model)
    print(f"Lambda: {lambda_sparse} | Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")
    return model, acc, sparsity
# MAIN
if __name__ == "__main__":
    print("Training started...\n")
    lambdas = [1e-5, 1e-4, 1e-3]
    best_model = None
    best_acc = 0
    results = []
    for lam in lambdas:
        model, acc, sp = train_model(lam)
        results.append((lam, acc, sp))
        if acc > best_acc:
            best_acc = acc
            best_model = model
    print("\nFinal Results:")
    for r in results:
        print(r)
    plot_gates(best_model)
    print("\nTraining completed!")
