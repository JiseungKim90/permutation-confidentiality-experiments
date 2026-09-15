#!/usr/bin/env python3
"""Regression witness for the observed-port bias quantifier in Theorem 7."""

import json
import numpy as np


def transcript(weight, bias1, bias2, message):
    return (int(weight * message + bias1), int(bias2))


def main():
    # E={(0,1)} and the trivial frame group.  Port 2 has no incoming edge but
    # remains observed, so its bias must be quantified independently of E.
    edge = (3, 5)
    model_m = transcript(edge[0], edge[1], 7, 11)
    model_n = transcript(edge[0], edge[1], 9, 11)
    edge_only_condition = True
    all_observed_port_biases = bool(np.array_equal([5, 7], [5, 9]))
    assert edge_only_condition
    assert not all_observed_port_biases
    assert model_m != model_n
    print(json.dumps({
        "edge_only_condition": edge_only_condition,
        "all_observed_port_biases_equal": all_observed_port_biases,
        "transcript_M": model_m,
        "transcript_N": model_n,
        "distinguishable": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
