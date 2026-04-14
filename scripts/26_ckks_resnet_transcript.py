"""
One-round CKKS transcript checks on actual quantized layers extracted from
a trained ResNet-20 checkpoint.

The experiment mirrors the TFHE transcript check:
  1. load a trained checkpoint;
  2. quantize selected layers to 1/p;
  3. evaluate zero/basis queries under CKKS encryption;
  4. apply a server-side output permutation after decryption-side observation;
  5. run round-then-sort and compare against the exact sorted spectra.

Unlike TFHE, CKKS is approximate, so we also report the maximum decrypted
noise relative to the rounding threshold.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

import tenseal as ts

from lib.attack import get_conv_layers, round_then_sort
from lib.models import ResNet20

os.makedirs("outputs/logs", exist_ok=True)


TEACHER_PATH = os.path.join("models", "resnet20_seed0.pt")
P_PREC = 256
GAMMA = 1.0 / P_PREC
DELTA = 1.0 / (2 * P_PREC)
LAYERS_TO_CHECK = ["conv1", "layer2.0.shortcut.0", "layer3.0.shortcut.0"]
PERM_SEED = 20260410


def load_model(path):
    model = ResNet20()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, ckpt


def make_ctx():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 30, 30, 40],
    )
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    return ctx


def encrypted_linear_eval(ctx, W_q, b_q, x):
    enc_x = ts.ckks_vector(ctx, x.tolist())
    out = []
    for row_idx in range(W_q.shape[0]):
        val = enc_x.dot(W_q[row_idx, :].tolist()).decrypt()[0] + float(b_q[row_idx])
        out.append(val)
    return np.array(out, dtype=np.float64)


def main():
    print("=== CKKS transcript check on trained ResNet-20 layers ===")
    print(f"teacher_path={TEACHER_PATH}")
    print(f"precision p={P_PREC}")
    print(f"layers={LAYERS_TO_CHECK}")

    model, ckpt = load_model(TEACHER_PATH)
    conv_layers = {name: (W_mat, b_vec, c_out, d)
                   for name, W_mat, b_vec, _has_bias, c_out, d in get_conv_layers(model)}

    rng = np.random.default_rng(PERM_SEED)
    t0 = time.time()
    ctx = make_ctx()
    ctx_time = time.time() - t0
    print(f"CKKS context setup time: {ctx_time:.2f}s")

    results = {
        "teacher_path": TEACHER_PATH,
        "teacher_test_accuracy": ckpt.get("test_accuracy"),
        "precision": P_PREC,
        "gamma": GAMMA,
        "delta": DELTA,
        "perm_seed": PERM_SEED,
        "context_setup_time_sec": ctx_time,
        "layers": {},
    }

    for layer_name in LAYERS_TO_CHECK:
        if layer_name not in conv_layers:
            print(f"SKIP {layer_name}: not found")
            continue

        W_mat, b_vec, c_out, d = conv_layers[layer_name]
        W_q = np.round(W_mat * P_PREC) / P_PREC
        b_q = np.round(b_vec * P_PREC) / P_PREC

        print(f"\n=== Layer {layer_name} ({c_out}x{d}) ===")

        queries = [("zero", None, np.zeros(d, dtype=np.float64))]
        for i in range(d):
            e_i = np.zeros(d, dtype=np.float64)
            e_i[i] = 1.0
            queries.append(("basis", i, e_i))

        layer_results = {}
        for perm_mode in ("fixed", "fresh"):
            perm = rng.permutation(c_out) if perm_mode == "fixed" else None
            max_noise = 0.0
            max_err = 0.0
            exact = True
            transcript = []

            t1 = time.time()
            for qnum, (qtype, qidx, x) in enumerate(queries):
                y_dec = encrypted_linear_eval(ctx, W_q, b_q, x)

                if qtype == "zero":
                    y_true = b_q.astype(np.float64)
                else:
                    y_true = (W_q[:, qidx] + b_q).astype(np.float64)

                perm_q = perm if perm is not None else rng.permutation(c_out)
                y_obs = y_dec[perm_q]
                y_true_perm = y_true[perm_q]

                noise = y_obs - y_true_perm
                max_noise = max(max_noise, float(np.max(np.abs(noise))))

                recovered = round_then_sort(y_obs, GAMMA)
                true_sorted = np.sort(y_true)
                err = float(np.max(np.abs(recovered - true_sorted)))
                max_err = max(max_err, err)
                exact = exact and (err == 0.0)

                if qnum < 3:
                    transcript.append({
                        "query": "zero" if qtype == "zero" else f"e_{qidx}",
                        "perm_mode": perm_mode,
                        "max_noise": float(np.max(np.abs(noise))),
                        "max_error": err,
                        "observed_first4": y_obs[:4].tolist(),
                    })

            query_time = time.time() - t1
            print(f"  [{perm_mode}] Queries: {len(queries)}")
            print(f"  [{perm_mode}] Query time: {query_time:.2f}s")
            print(f"  [{perm_mode}] Max decrypt noise: {max_noise:.2e}")
            print(f"  [{perm_mode}] Noise/Delta: {max_noise/DELTA:.2e}")
            print(f"  [{perm_mode}] Max recovery error: {max_err:.2e}")
            print(f"  [{perm_mode}] All exact: {exact}")

            layer_results[perm_mode] = {
                "dimensions": f"{c_out}x{d}",
                "queries": len(queries),
                "query_time_sec": query_time,
                "max_decrypt_noise": max_noise,
                "noise_to_delta_ratio": float(max_noise / DELTA),
                "max_recovery_error": max_err,
                "all_exact": exact,
                "perm": perm.tolist() if perm is not None else "fresh-per-query",
                "transcript_sample": transcript,
            }

        results["layers"][layer_name] = layer_results

    out_path = os.path.join("outputs", "logs", "26_ckks_resnet_transcript.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    for layer_name, layer_result in results["layers"].items():
        print(f"  {layer_name}: {layer_result}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
