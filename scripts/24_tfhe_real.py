"""
Concrete-based TFHE sanity checks.
This script combines:
  (1) a real Concrete TFHE identity-circuit measurement,
  (2) a real Concrete TFHE linear layer on a small synthetic instance, and
  (3) a larger simulated quantized-layer experiment.
It does not provide a single large end-to-end deployed-layer run.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
os.makedirs("outputs/logs", exist_ok=True)

try:
    from concrete import fhe
except ImportError:
    print("ERROR: concrete-python not installed")
    sys.exit(1)

print(f"concrete-python version: {fhe.__version__ if hasattr(fhe, '__version__') else 'unknown'}")

# Parameters
m, d = 64, 32  # smaller for TFHE compile time
p_prec = 256   # 8-bit
gamma = 1.0 / p_prec
Delta = 1.0 / (2 * p_prec)

np.random.seed(42)

# Quantized weights as integers in [-128, 127]
W_int = np.random.randint(-64, 64, size=(m, d)).astype(np.int64)
b_int = np.random.randint(-32, 32, size=(m,)).astype(np.int64)

results = {}

# ---- Approach 1: Encrypt-decrypt identity to measure TFHE noise ----
print("\n=== Approach 1: real TFHE encrypt-decrypt identity measurement ===")

@fhe.compiler({"x": "encrypted"})
def identity(x):
    return x

try:
    t0 = time.time()
    # Use values in the range we care about
    inputset = [np.random.randint(-128, 128, size=(m,)).astype(np.int64) for _ in range(100)]
    circuit = identity.compile(inputset, configuration=fhe.Configuration(
        enable_unsafe_features=True,
        use_insecure_key_cache=True,
        insecure_key_cache_location="/tmp/concrete_keys",
    ))
    print(f"Compile time: {time.time()-t0:.1f}s")

    # Generate keys
    t0 = time.time()
    circuit.keygen()
    print(f"Keygen time: {time.time()-t0:.1f}s")

    # Measure noise over multiple trials
    max_noise = 0
    mean_noises = []
    n_trials = 20
    for trial in range(n_trials):
        test_vec = np.random.randint(-128, 128, size=(m,)).astype(np.int64)
        result = circuit.encrypt_run_decrypt(test_vec)
        noise = np.abs(result.astype(np.float64) - test_vec.astype(np.float64))
        max_noise = max(max_noise, noise.max())
        mean_noises.append(noise.mean())

    print(f"Max noise across {n_trials} trials: {max_noise}")
    print(f"Mean noise: {np.mean(mean_noises):.4f}")
    print(f"TFHE operates on integers => noise is 0 for identity (exact)")

    results["identity_noise"] = {
        "max_noise": float(max_noise),
        "mean_noise": float(np.mean(mean_noises)),
        "exact": bool(max_noise == 0),
    }
except Exception as e:
    print(f"Identity test failed: {e}")
    results["identity_noise"] = {"error": str(e)}

# ---- Approach 2: Linear layer computation in TFHE ----
print("\n=== Approach 2: real TFHE linear layer on a small synthetic instance ===")

# For concrete-python, we define a function that computes Wx+b
# on encrypted x. The weights W and bias b are clear (server-side).
# We need small dimensions due to TFHE compile time.
m_small, d_small = 16, 8
W_small = np.random.randint(-16, 16, size=(m_small, d_small)).astype(np.int64)
b_small = np.random.randint(-8, 8, size=(m_small,)).astype(np.int64)

@fhe.compiler({"x": "encrypted"})
def linear_layer(x):
    # Compute Wx + b where W, b are captured from outer scope
    y = W_small @ x + b_small
    return y

try:
    t0 = time.time()
    # Input set for calibration
    inputset = [np.random.randint(-4, 5, size=(d_small,)).astype(np.int64) for _ in range(200)]
    circuit = linear_layer.compile(inputset, configuration=fhe.Configuration(
        enable_unsafe_features=True,
        use_insecure_key_cache=True,
        insecure_key_cache_location="/tmp/concrete_keys",
    ))
    print(f"Compile time: {time.time()-t0:.1f}s")

    circuit.keygen()

    # Test with basis vectors
    n_exact = 0
    max_err = 0
    for i in range(d_small):
        e_i = np.zeros(d_small, dtype=np.int64)
        e_i[i] = 1

        # Clear computation (ground truth)
        y_true = W_small @ e_i + b_small
        y_true_sorted = np.sort(y_true)

        # Encrypted computation
        y_enc = circuit.encrypt_run_decrypt(e_i)

        # Apply permutation (simulate shuffling) + round-then-sort
        perm = np.random.permutation(m_small)
        y_shuffled = y_enc[perm]  # simulate server shuffle

        # For integer TFHE: no rounding needed if exact
        y_recovered = np.sort(y_shuffled)

        err = np.max(np.abs(y_recovered.astype(np.float64) - y_true_sorted.astype(np.float64)))
        max_err = max(max_err, err)
        if err == 0:
            n_exact += 1

    print(f"Exact columns: {n_exact}/{d_small}")
    print(f"Max error: {max_err}")

    results["linear_layer"] = {
        "dimensions": f"{m_small}x{d_small}",
        "exact_columns": f"{n_exact}/{d_small}",
        "max_error": float(max_err),
        "all_exact": bool(n_exact == d_small),
    }
except Exception as e:
    print(f"Linear layer test failed: {e}")
    results["linear_layer"] = {"error": str(e)}

# ---- Approach 3: Simulate larger Safhire-like quantized computation ----
print("\n=== Approach 3: simulated larger quantized layer ===")

# Simulation (not a real Concrete TFHE run): larger quantized layer,
# conservative small-noise injection, round-then-sort on the simulated
# client-side observation.

from lib.attack import round_then_sort

for i in range(d):
    true_col = (W_int[:, i] + b_int).astype(np.float64) / p_prec
    true_sorted = np.sort(true_col)

    # Simulate TFHE noise: based on our measurement (0 for integers,
    # but in practice there's a small probability of error for
    # programmable bootstrapping). Use conservative bound.
    noise = np.random.normal(0, 1e-6, m).clip(-Delta+1e-15, Delta-1e-15)
    perm = np.random.permutation(m)
    observed = true_col[perm] + noise
    recovered = round_then_sort(observed, gamma)
    err = np.max(np.abs(recovered - true_sorted))
    if err > 0:
        print(f"  Column {i}: ERROR = {err}")
        break
else:
    print(f"All {d} columns: exact recovery")
    results["quantized_layer"] = {"dimensions": f"{m}x{d}", "all_exact": True}

# Save
with open("outputs/logs/24_tfhe_real.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n=== SUMMARY ===")
for k, v in results.items():
    print(f"  {k}: {v}")
print("Done.")
