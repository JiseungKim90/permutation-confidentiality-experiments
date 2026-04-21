"""
Single-round TFHE transcript check on actual quantized ResNet-20 layers.

Verifies that round-then-sort recovers the sorted spectrum from TFHE-encrypted
transcripts, using real trained weights rather than synthetic matrices.
Not full end-to-end encrypted inference; server-side permutation is applied
after each layer's linear map under encryption.
"""
import argparse
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from lib.attack import get_conv_layers, round_then_sort
from lib.models import ResNet20

os.makedirs("outputs/logs", exist_ok=True)

try:
    from concrete import fhe
except ImportError:
    print("ERROR: concrete-python not installed")
    sys.exit(1)


DEFAULT_TEACHER_PATH = os.path.join("models", "resnet20_seed0.pt")
DEFAULT_PREC = 256
DEFAULT_LAYERS = ["conv1", "layer2.0.shortcut.0", "layer3.0.shortcut.0"]
DEFAULT_PERM_SEED = 20260410


def load_model(path):
    model = ResNet20()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, ckpt


def make_linear_circuit(W_int, b_int, perm):
    P = np.eye(len(perm), dtype=np.int64)[perm]

    @fhe.compiler({"x": "encrypted"})
    def linear_perm_layer(x):
        y = W_int @ x + b_int
        return P @ y

    inputset = [
        np.zeros(W_int.shape[1], dtype=np.int64),
        *[
            np.random.randint(0, 2, size=(W_int.shape[1],)).astype(np.int64)
            for _ in range(64)
        ],
    ]
    circuit = linear_perm_layer.compile(
        inputset,
        configuration=fhe.Configuration(
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            insecure_key_cache_location="/tmp/concrete_keys",
        ),
    )
    circuit.keygen()
    return circuit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-path", type=str, default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--precision", type=int, default=DEFAULT_PREC)
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--perm-seed", type=int, default=DEFAULT_PERM_SEED)
    parser.add_argument("--output-json", type=str, default=os.path.join("outputs", "logs", "25_tfhe_resnet_transcript.json"))
    args = parser.parse_args()

    p_prec = args.precision
    gamma = 1.0 / p_prec
    teacher_path = args.teacher_path
    layers_to_check = args.layers
    perm_seed = args.perm_seed

    print(f"concrete-python version: {fhe.__version__ if hasattr(fhe, '__version__') else 'unknown'}")
    print("=== TFHE transcript check on trained ResNet-20 layers ===")
    print(f"teacher_path={teacher_path}")
    print(f"precision p={p_prec}")
    print(f"layers={layers_to_check}")

    model, ckpt = load_model(teacher_path)
    conv_layers = {name: (W_mat, b_vec, c_out, d)
                   for name, W_mat, b_vec, _has_bias, c_out, d in get_conv_layers(model)}

    rng = np.random.default_rng(perm_seed)
    results = {
        "teacher_path": teacher_path,
        "teacher_test_accuracy": ckpt.get("test_accuracy"),
        "precision": p_prec,
        "gamma": gamma,
        "perm_seed": perm_seed,
        "layers": {},
    }

    for layer_name in layers_to_check:
        if layer_name not in conv_layers:
            print(f"SKIP {layer_name}: not found")
            continue

        W_mat, b_vec, c_out, d = conv_layers[layer_name]
        W_int = np.round(W_mat * p_prec).astype(np.int64)
        b_int = np.round(b_vec * p_prec).astype(np.int64)

        print(f"\n=== Layer {layer_name} ({c_out}x{d}) ===")

        queries = [("zero", None, np.zeros(d, dtype=np.int64))]
        for i in range(d):
            e_i = np.zeros(d, dtype=np.int64)
            e_i[i] = 1
            queries.append(("basis", i, e_i))

        layer_results = {}
        for perm_mode in ("fixed", "fresh"):
            if perm_mode == "fixed":
                perm = rng.permutation(c_out).astype(np.int64)
                t0 = time.time()
                circuit = make_linear_circuit(W_int, b_int, perm)
                compile_and_keygen_time = time.time() - t0
            else:
                perm = None
                circuit = None
                compile_and_keygen_time = 0.0

            max_err = 0.0
            exact = True
            transcript = []

            t1 = time.time()
            for qnum, (qtype, qidx, x) in enumerate(queries):
                if perm_mode == "fresh":
                    perm_q = rng.permutation(c_out).astype(np.int64)
                    circuit_q = make_linear_circuit(W_int, b_int, perm_q)
                    y_dec = circuit_q.encrypt_run_decrypt(x)
                else:
                    perm_q = perm
                    y_dec = circuit.encrypt_run_decrypt(x)

                if qtype == "zero":
                    y_true = b_int.astype(np.float64) / p_prec
                else:
                    y_true = (W_int[:, qidx] + b_int).astype(np.float64) / p_prec

                y_obs = y_dec.astype(np.float64) / p_prec
                recovered = round_then_sort(y_obs, gamma)
                true_sorted = np.sort(y_true)
                err = float(np.max(np.abs(recovered - true_sorted)))
                max_err = max(max_err, err)
                exact = exact and (err == 0.0)

                if qtype == "zero":
                    label = "zero"
                else:
                    label = f"e_{qidx}"
                if qnum < 3:
                    transcript.append({
                        "query": label,
                        "perm_mode": perm_mode,
                        "max_error": err,
                        "observed_first4": y_obs[:4].tolist(),
                    })

            query_time = time.time() - t1
            print(f"  [{perm_mode}] Queries: {len(queries)}")
            print(f"  [{perm_mode}] Compile+keygen time: {compile_and_keygen_time:.2f}s")
            print(f"  [{perm_mode}] Query time: {query_time:.2f}s")
            print(f"  [{perm_mode}] Max recovery error: {max_err:.2e}")
            print(f"  [{perm_mode}] All exact: {exact}")

            layer_results[perm_mode] = {
                "dimensions": f"{c_out}x{d}",
                "queries": len(queries),
                "compile_and_keygen_time_sec": compile_and_keygen_time,
                "query_time_sec": query_time,
                "max_recovery_error": max_err,
                "all_exact": exact,
                "perm": perm.tolist() if perm is not None else "fresh-per-query",
                "transcript_sample": transcript,
            }

        results["layers"][layer_name] = layer_results

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    for layer_name, layer_result in results["layers"].items():
        print(f"  {layer_name}: {layer_result}")
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
