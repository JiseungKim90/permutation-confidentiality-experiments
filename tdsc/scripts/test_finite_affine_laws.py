"""Exact small-model checks for finite distributions of affine response maps.

This is a mathematical regression, not model recovery. It enumerates binary
coefficients of a two-port affine graph, including correlated and biased
frame laws, and compares coefficient-law and response-law partitions.
"""
import hashlib
import itertools
import json
import platform
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def weighted_law(values, weights):
    mass = defaultdict(Fraction)
    total = sum(weights)
    for value, weight in zip(values, weights):
        if weight:
            mass[value] += Fraction(weight, total)
    return tuple(sorted((value, prob.numerator, prob.denominator)
                        for value, prob in mass.items()))


def coefficient_tuple(model, p1, p2):
    # y1 = P1 b1; y2 = P2 (B P1^{-1} z1 + b2).
    b1 = model[:2]
    matrix = (model[2:4], model[4:6])
    b2 = model[6:8]
    return (b1[p1[0]], b1[p1[1]],
            matrix[p2[0]][p1[0]], matrix[p2[0]][p1[1]],
            matrix[p2[1]][p1[0]], matrix[p2[1]][p1[1]],
            b2[p2[0]], b2[p2[1]])


def evaluate(coefficients, point):
    v = coefficients
    return (v[0], v[1], v[2]*point[0] + v[3]*point[1] + v[6],
            v[4]*point[0] + v[5]*point[1] + v[7])


def main():
    frames = list(itertools.product(((0, 1), (1, 0)), repeat=2))
    priors = ((1, 1, 1, 1), (3, 1, 1, 3), (1, 0, 0, 1),
              (0, 1, 1, 0), (1, 0, 0, 0), (1, 2, 3, 4))
    # With binary coefficients, b + 4*a + 16*c determines (a,c,b).
    # This point therefore separates every distinct map in this finite test.
    point = (4, 16)
    coefficient_to_reply = defaultdict(set)
    reply_to_coefficient = defaultdict(set)
    zero_reply_laws = set()
    systems = 0
    for model in itertools.product((0, 1), repeat=8):
        maps = [coefficient_tuple(model, p1, p2) for p1, p2 in frames]
        for weights in priors:
            coefficient_law = weighted_law(maps, weights)
            reply_law = weighted_law([evaluate(value, point) for value in maps], weights)
            coefficient_to_reply[coefficient_law].add(reply_law)
            reply_to_coefficient[reply_law].add(coefficient_law)
            zero_reply_laws.add(weighted_law([evaluate(value, (0, 0)) for value in maps], weights))
            systems += 1
    failures = sum(len(values) != 1 for values in coefficient_to_reply.values())
    failures += sum(len(values) != 1 for values in reply_to_coefficient.values())
    assert failures == 0
    assert len(zero_reply_laws) < len(coefficient_to_reply)
    # Same orbit is insufficient for a biased sampler. Reversing the bias
    # together with the constant vector, however, preserves its exact law.
    forward = ((0, 1), (1, 0))
    reverse = tuple(reversed(forward))
    assert weighted_law(forward, (3, 1)) != weighted_law(reverse, (3, 1))
    assert weighted_law(forward, (3, 1)) == weighted_law(reverse, (1, 3))
    assert weighted_law(forward, (1, 1)) == weighted_law(reverse, (1, 1))
    print(json.dumps({
        "schema": "p050-finite-affine-law-check-v1", "success": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "arithmetic": "exact integers and fractions",
        "coefficient_models": 256, "joint_frames": 4, "frame_priors": len(priors),
        "systems": systems, "coefficient_law_classes": len(coefficient_to_reply),
        "separating_response_law_classes": len(reply_to_coefficient),
        "zero_message_response_law_classes": len(zero_reply_laws),
        "partition_disagreements": failures, "biased_sampler_checks": 3,
        "scope": "Finite regression of the proof argument, not a proof of the open-domain proposition."
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
