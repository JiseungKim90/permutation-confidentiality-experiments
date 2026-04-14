"""Shared model definitions used across all experiments."""
import torch
import torch.nn as nn


# ---- MLP models (Section 4.1, Section 5) ----

class SmallMLP(nn.Module):
    """3-layer MLP: 64 -> 128 -> 128 -> 10. Used in MLP attack (Table 3)."""
    def __init__(self, in_dim=64, hidden=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class WideMLP(nn.Module):
    """2-layer MLP: 64 -> 512 -> 10. Used in fingerprinting (Table 5)."""
    def __init__(self, in_dim=64, hidden=512, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class DeepMLP(nn.Module):
    """5-layer MLP: 64 -> 64 -> 64 -> 64 -> 64 -> 10. Used in fingerprinting (Table 5)."""
    def __init__(self, in_dim=64, hidden=64, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


# ---- MNIST MLP (Section 5.4 tradeoff, Appendix B KD) ----

class MNISTNet(nn.Module):
    """784 -> 128 -> 64 -> 10. Target model for MNIST experiments."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TinyNet(nn.Module):
    """784 -> 32 -> 16 -> 10. KD student (Appendix B)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class WideNet(nn.Module):
    """784 -> 256 -> 10. KD student (Appendix B)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ---- ResNet-20 for CIFAR-10 (Section 4.1, Appendix A) ----

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class _ResNetCIFAR(nn.Module):
    """Generic He et al. CIFAR ResNet with `n` BasicBlocks per stage.

    Total conv layers: 6n + 1 (without counting shortcut convs at stage transitions).
    n=3 -> ResNet-20, n=5 -> ResNet-32, n=7 -> ResNet-44, n=9 -> ResNet-56.
    """
    def __init__(self, n):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, n, 1)
        self.layer2 = self._make_layer(16, 32, n, 2)
        self.layer3 = self._make_layer(32, 64, n, 2)
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, in_ch, out_ch, n, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.mean(dim=[2, 3])
        return self.fc(out)


class ResNet20(_ResNetCIFAR):
    def __init__(self):
        super().__init__(n=3)


class ResNet56(_ResNetCIFAR):
    """CIFAR ResNet with 9 BasicBlocks per stage (55 conv + 1 fc)."""
    def __init__(self):
        super().__init__(n=9)


# ---- Training config for ResNet-20 (single source of truth) ----
CIFAR_TRAIN_CONFIG = {
    "architecture": "ResNet-20",
    "dataset": "CIFAR-10",
    "n_models": 20,
    "n_epochs": 10,
    "optimizer": "SGD",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "lr_schedule": "MultiStepLR",
    "milestones": [5, 8],
    "lr_gamma": 0.1,
    "batch_size": 128,
}
