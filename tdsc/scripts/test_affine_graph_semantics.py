"""Exact algebra regressions for joint permutation actions on affine graphs.

Only synthetic integer tensors are used. There is no model loader, service,
recovery procedure, or protocol implementation in this test.
"""

import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path


IDENTITY = (0, 1, 2)
S3 = tuple(itertools.permutations(IDENTITY))
C3 = (IDENTITY, (1, 2, 0), (2, 0, 1))


def apply(p, vector):
    return tuple(vector[index] for index in p)


def inverse(p):
    return tuple(p.index(index) for index in range(len(p)))


def compose(p, q):
    # Matrix convention: apply(P Q, v) = apply(P, apply(Q, v)).
    return tuple(q[index] for index in p)


def matvec(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def add(*vectors):
    return tuple(sum(values) for values in zip(*vectors))


def transform_matrix(matrix, output_frame, input_frame):
    return tuple(tuple(matrix[i][j] for j in input_frame) for i in output_frame)


def transform(model, frames):
    first, skip, second, bias1, bias2 = model
    p, q = frames
    return (transform_matrix(first, p, IDENTITY),
            transform_matrix(skip, q, IDENTITY),
            transform_matrix(second, q, p), apply(p, bias1), apply(q, bias2))


def direct_response(model, frames, x, z):
    # Independent evaluation: change vectors' coordinates before applying the
    # original matrices, instead of evaluating the transformed coefficients.
    first, skip, second, bias1, bias2 = model
    p, q = frames
    return (apply(p, add(matvec(first, x), bias1)),
            apply(q, add(matvec(skip, x), matvec(second, apply(inverse(p), z)), bias2)))


def coefficient_response(model, x, z):
    first, skip, second, bias1, bias2 = model
    return (add(matvec(first, x), bias1),
            add(matvec(skip, x), matvec(second, z), bias2))


def main():
    seed = 20261002
    rng = random.Random(seed)
    models = []
    for _ in range(12):
        matrices = tuple(tuple(tuple(rng.randrange(-2, 3) for _ in range(3))
                               for _ in range(3)) for _ in range(3))
        biases = tuple(tuple(rng.randrange(-2, 3) for _ in range(3)) for _ in range(2))
        models.append(matrices + biases)
    zero = (0, 0, 0)
    zero_matrix = (zero, zero, zero)
    models.extend(((zero_matrix, zero_matrix, zero_matrix, zero, zero),
                   (zero_matrix, zero_matrix, zero_matrix, (0, 1, 2), (2, 1, 0))))
    groups = {
        "independent_S3": tuple(itertools.product(S3, repeat=2)),
        "shared_S3": tuple((p, p) for p in S3),
        "shared_C3": tuple((p, p) for p in C3),
        "labeled_output": tuple((p, IDENTITY) for p in S3),
    }
    points = ((zero, zero), ((1, 0, 0), (0, 1, 0)),
              ((0, 0, 1), (1, 0, 0)), ((-1, 2, 1), (2, -1, 0)))
    direct_checks = law_checks = coupling_checks = 0
    for group in groups.values():
        for model in models:
            original_law = Counter(transform(model, frame) for frame in group)
            for frame in group:
                coefficients = transform(model, frame)
                for x, z in points:
                    assert direct_response(model, frame, x, z) == coefficient_response(coefficients, x, z)
                    direct_checks += 1
                relabeled = transform(model, frame)
                assert Counter(transform(relabeled, p) for p in group) == original_law
                law_checks += 1
                frame_inverse = tuple(inverse(p) for p in frame)
                for p in group:
                    coupled = tuple(compose(a, b) for a, b in zip(p, frame_inverse))
                    assert coupled in group
                    assert transform(relabeled, coupled) == transform(model, p)
                    coupling_checks += 1
    # A 3-cycle exposes an inverse-direction bug which transpositions hide.
    frame = ((1, 2, 0), IDENTITY)
    model = models[0]
    x, z = points[-1]
    wrong = apply(frame[1], add(matvec(model[1], x),
                              matvec(model[2], apply(frame[0], z)), model[4]))
    assert wrong != direct_response(model, frame, x, z)[1]
    # Correlated frames cannot be replaced by independent marginals.
    constant_model = models[-1]
    joint = Counter(transform(constant_model, p) for p in groups["shared_C3"])
    mixed = transform(constant_model, (C3[1], IDENTITY))
    assert joint != Counter(transform(mixed, p) for p in groups["shared_C3"])
    # The bias at a port remains observable when every incoming edge is zero.
    assert direct_response(models[-1], (IDENTITY, IDENTITY), zero, zero) != direct_response(models[-2], (IDENTITY, IDENTITY), zero, zero)
    print(json.dumps({
        "success": True, "seed": seed, "arithmetic": "exact integers",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "models": len(models), "joint_groups": list(groups),
        "direct_response_checks": direct_checks, "uniform_law_checks": law_checks,
        "coupling_checks": coupling_checks, "negative_controls": 3,
        "scope": "Finite algebra regressions, not a proof or a recovery experiment."
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
