"""Model definitions and layer extraction.

ResNet56, MNISTNet and get_conv_layers are copied from the prior paper's public
artifact (repo JiseungKim90/permutation-confidentiality-experiments, lib/models.py
and lib/attack.py) without any change of semantics.
"""
import numpy as np
import torch
import torch.nn as nn

from .checkpoint import load_verified_checkpoint


# ---------------- copied from lib/models.py of the artifact ----------------
class MNISTNet(nn.Module):
    """784 -> 128 -> 64 -> 10."""
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


class ResNet56(_ResNetCIFAR):
    def __init__(self):
        super().__init__(n=9)


# ---------------- copied from lib/attack.py of the artifact ----------------
def get_conv_layers(model):
    """Extract all Conv2d layers as (name, W_mat, b_vec, has_bias, C_out, d)."""
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            W = mod.weight.detach().numpy()
            C_out = W.shape[0]
            d = int(np.prod(W.shape[1:]))
            W_mat = W.reshape(C_out, d)
            b = mod.bias.detach().numpy() if mod.bias is not None else np.zeros(C_out)
            layers.append((name, W_mat, b, mod.bias is not None, C_out, d))
    return layers


# ---------------- helpers ----------------
def load_resnet56(path, expected_sha256):
    model = ResNet56()
    ckpt = load_verified_checkpoint(path, expected_sha256)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, ckpt


def bn_for_conv(name):
    """Name of the BatchNorm2d that follows the given conv in ResNet56."""
    if name == "conv1":
        return "bn1"
    if name.endswith(".conv1"):
        return name[: -len(".conv1")] + ".bn1"
    if name.endswith(".conv2"):
        return name[: -len(".conv2")] + ".bn2"
    if name.endswith(".shortcut.0"):
        return name[: -len(".shortcut.0")] + ".shortcut.1"
    return None


def fold_bn(model, name, W_mat, b_vec):
    """Fold the BatchNorm following conv `name` into (W, b):
    W' = W gamma / sigma (row-wise), b' = beta - gamma mu / sigma,
    sigma = sqrt(running_var + eps)."""
    mods = dict(model.named_modules())
    bn_name = bn_for_conv(name)
    if bn_name is None or bn_name not in mods:
        return None
    bn = mods[bn_name]
    gamma = bn.weight.detach().numpy().astype(np.float64)
    beta = bn.bias.detach().numpy().astype(np.float64)
    mu = bn.running_mean.detach().numpy().astype(np.float64)
    var = bn.running_var.detach().numpy().astype(np.float64)
    sigma = np.sqrt(var + bn.eps)
    scale = gamma / sigma
    Wf = W_mat.astype(np.float64) * scale[:, None]
    bf = beta - gamma * mu / sigma + b_vec.astype(np.float64) * scale
    return Wf, bf


def resnet56_main_path_names():
    """The 55 main-path convolutions of ResNet-56 in network order
    (shortcut convs and the final fc excluded)."""
    names = ["conv1"]
    for stage in (1, 2, 3):
        for blk in range(9):
            names.append("layer%d.%d.conv1" % (stage, blk))
            names.append("layer%d.%d.conv2" % (stage, blk))
    return names


def env_info():
    import os
    import platform
    import sys
    info = {
        "host": platform.node() or "unknown",
        "kernel": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "longdouble_mantissa_bits": int(np.finfo(np.longdouble).nmant) + 1,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        info["cores"] = os.cpu_count()
    except Exception:
        pass
    return info
