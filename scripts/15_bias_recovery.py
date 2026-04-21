"""
Exp 15: sort(W_{:,i} + b) recovery on a bias-having MLP (MNIST).
Calls lib.attack.attack_layer (simulate_query + round_then_sort).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from lib.attack import attack_layer, simulate_query, round_then_sort

torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.fc2 = nn.Linear(128, 64, bias=True)
        self.fc3 = nn.Linear(64, 10, bias=True)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,),(0.3081,))])
trainset = torchvision.datasets.MNIST(
    root=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
    train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(
    root=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
    train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=256, shuffle=False)

model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
for epoch in range(10):
    for x, y in trainloader:
        opt.zero_grad()
        criterion(model(x.view(x.size(0),-1)), y).backward()
        opt.step()

model.eval()
correct = sum((model(x.view(x.size(0),-1)).argmax(1)==y).sum().item() for x, y in testloader)
print("Trained MLP test accuracy: %.1f%%" % (correct/len(testset)*100))

p = 256
Delta = 1.0 / (2 * p)

def quantize(w):
    return np.round(w * p) / p

print("\n=== Attack on bias-having MLP ===")
print("Precision p=%d, gamma=%.4f, Delta=%.4f" % (p, 1.0/p, Delta))
print()

layers = [
    ("fc1", model.fc1.weight.detach().numpy(), model.fc1.bias.detach().numpy()),
    ("fc2", model.fc2.weight.detach().numpy(), model.fc2.bias.detach().numpy()),
]

for name, W, b in layers:
    Wq = quantize(W)
    bq = quantize(b)

    true_sort_b = np.sort(bq)
    true_spectra = [np.sort(Wq[:,i] + bq) for i in range(Wq.shape[1])]

    max_err_layer, rec_spectra = attack_layer(W, b, p)

    gamma = 1.0 / p
    rec_b = round_then_sort(simulate_query(bq, p), gamma)

    b_error = np.max(np.abs(rec_b - true_sort_b))
    col_errors = [np.max(np.abs(rec_spectra[i] - true_spectra[i])) for i in range(len(true_spectra))]
    max_err = max(col_errors)

    # Check bias inseparability
    n_inseparable = sum(
        not np.allclose(rec_spectra[i], np.sort(Wq[:,i]) + np.sort(bq))
        for i in range(len(true_spectra)))

    print("Layer %s: shape %s" % (name, str(W.shape)))
    print("  Bias nonzero: %d/%d" % ((bq != 0).sum(), len(bq)))
    print("  sort(b) recovery error: %.2e (exact: %s)" % (b_error, b_error==0))
    print("  sort(W[:,i]+b) max error: %.2e (exact: %s)" % (max_err, max_err==0))
    print("  Columns where sort(W+b) != sort(W)+sort(b): %d/%d" % (n_inseparable, len(true_spectra)))
    print()

print("Done.")
