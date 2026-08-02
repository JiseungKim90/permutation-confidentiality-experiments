"""Cross-check manuscript claims against canonical experiment JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT.parent
OUT = ROOT / "outputs"


def load(relative: str) -> dict:
    return json.loads((OUT / relative).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    interpolation = load("interpolation_recovery.json")
    lifecycle = load("session_lifecycle_recovery.json")
    equivalence = load("resnet_permutation_equivalence.json")
    audit = load("offset_query_audit.json")
    tfhe = load("logs/49_tfhe_fixed_session_later_layer.json")
    fresh4 = load("fresh_session_orbit_recovery_input15.json")
    fresh5 = load("fresh_session_orbit_recovery_input31.json")
    fresh8 = load("fresh_session_orbit_recovery_input255.json")
    fresh_tfhe = load(
        "logs/58_tfhe_state_machine_fresh_session_orbit_recovery.json"
    )
    protection_small = load("logs/54_tfhe_protected_activation_input31.json")
    protection = load("logs/54_tfhe_protected_activation_input255.json")
    private_embeddings = load(
        "data_side/private_embedding_orbits_exact_fullvocab_20260802.json"
    )
    stip_margin = load(
        "data_side/stip_margin_phase_exact_rows_20260802.json"
    )
    stip_trained = load(
        "data_side/stip_independent_embedding_frequency_f025_3seed_20260802.json"
    )
    stip_fraction_sweep = load(
        "data_side/stip_independent_embedding_frequency_phase_sweep_20260802.json"
    )
    gelo_base = load("data_side/gelo_rowspace_gpt2_20260802.json")
    gelo_large = load("data_side/gelo_rowspace_gpt2_512x32_20260802.json")
    gelo_medium = load("data_side/gelo_rowspace_gpt2medium_20260802.json")
    gelo_official = load("data_side/gelo_official_transcript_gpt2_20260802.json")
    tex = (PAPER_ROOT / "tdsc_main.tex").read_text(encoding="utf-8")
    marked_tex = (PAPER_ROOT / "tdsc_main_marked.tex").read_text(
        encoding="utf-8"
    )
    ref_bib = (PAPER_ROOT / "ref.bib").read_text(encoding="utf-8")
    journal_ref_bib = (PAPER_ROOT / "journal_ref.bib").read_text(
        encoding="utf-8"
    )
    checks: list[str] = []

    stress_8 = next(
        row for row in interpolation["collision_stress"] if row["shape"] == [8, 8]
    )
    require(stress_8["interpolation_exact"] == stress_8["trials"] == 100,
            "8x8 interpolation is not 100/100 exact")
    require(stress_8["interpolation_wrong"] == 0, "interpolation has wrong outputs")
    require(stress_8["practical_exact"] == 99 and stress_8["practical_ambiguous"] == 1,
            "practical-solver split changed")
    checks.append("collision recovery")

    trained = interpolation["trained_resnet20_first_conv_bn"]
    require(trained["bias_anchored"]["queries"] == 433,
            "bias query count changed")
    require(trained["bias_free"]["queries"] == 417,
            "bias-free query count changed")
    require(trained["bias_anchored"]["status"] ==
            trained["bias_free"]["status"] == "exact",
            "trained first-layer recovery is not exact")
    checks.append("trained first layer")

    fixed = lifecycle["fixed_session_repeated_layer_queries"]
    require(fixed["all_exact"] and fixed["layer_count"] == 21,
            "fixed-session layer result changed")
    require(fixed["total_queries"] == 5712 and fixed["global_max_lattice_error"] == 0,
            "fixed-session query/error result changed")
    require(lifecycle["fresh_input_per_request"]["all_equal"],
            "fresh-input transcript equivalence failed")
    checks.append("permutation lifetime")

    expected_fresh = [
        (fresh4, 1, 3, 9),
        (fresh5, 4, 5, 15),
        (fresh8, 12, 15, 45),
    ]
    for record, certified, exact_layers, exact_trials in expected_fresh:
        summary = record["summary"]
        require(summary["layer_count"] == 21,
                "fresh-session layer denominator changed")
        require(summary["certified_layer_count"] == certified,
                "fresh-session certified-layer count changed")
        require(summary["all_trials_exact_layer_count"] == exact_layers,
                "fresh-session exact-layer count changed")
        require(summary["total_exact_trials"] == exact_trials and
                summary["total_trials"] == 63,
                "fresh-session trial count changed")
    checks.append("fresh-session orbit recovery")

    classifier_expectations = [(fresh4, 0), (fresh5, 0), (fresh8, 3)]
    for record, exact_trials in classifier_expectations:
        classifier = record["final_classifier"]
        require(classifier["shape"] == [10, 64],
                "final-classifier shape changed")
        require(len(classifier["trials"]) == 3 and
                sum(trial["status"] == "exact_orbit"
                    for trial in classifier["trials"]) == exact_trials,
                "final-classifier exact-trial count changed")
        require(all(trial["queries"] == 1152
                    for trial in classifier["trials"]),
                "final-classifier query count changed")
    require(fresh8["final_classifier"]["certified_interval_separation"],
            "8-bit final-classifier gap certificate changed")
    checks.append("full final-classifier orbit recovery")

    require(fresh_tfhe["status"] == "exact_orbit" and
            fresh_tfhe["shape"] == [32, 16] and
            fresh_tfhe["queries"] == 267,
            "fresh-session TFHE result changed")
    require(fresh_tfhe["sampled_latent_column_count"] == 16 and
            fresh_tfhe["recovered_distinct_column_count"] == 16 and
            fresh_tfhe["no_false_columns"] and
            fresh_tfhe["bias_recovered_exactly"],
            "fresh-session TFHE recovery evidence changed")
    require(fresh_tfhe["fresh_hidden_input_permutation_each_query"] and
            fresh_tfhe["fresh_hidden_output_permutation_each_query"] and
            not fresh_tfhe["same_round_replay"],
            "fresh-session TFHE threat-model flags changed")
    require(fresh_tfhe["unique_session_id_each_query"] and
            fresh_tfhe["monotonic_round_counter_enforced"] and
            fresh_tfhe["stale_and_duplicate_requests_rejected"] and
            fresh_tfhe["target_round"] == 5,
            "fresh-session state-machine flags changed")
    state = fresh_tfhe["state_machine"]
    require(state["unique_sessions_opened"] ==
            state["accepted_target_requests"] ==
            state["unique_session_round_pairs"] == 267,
            "fresh-session session/round counts changed")
    require(state["rejected_stale_or_duplicate_requests"] == 267 and
            state["all_target_rounds_used_once"],
            "fresh-session replay-rejection evidence changed")
    checks.append("state-machine-compliant fresh-session TFHE")

    functional = equivalence["functional_equivalence"]
    difference = equivalence["parameter_difference"]
    require(functional["examples"] == 10000 and
            functional["prediction_agreement_percent"] == 100.0,
            "functional-equivalence result changed")
    require(functional["original_accuracy_percent"] ==
            functional["equivalent_accuracy_percent"] == 82.84,
            "clone accuracy changed")
    require(difference["changed_coordinates"] == 273894 and
            difference["floating_coordinates"] == 274042,
            "parameter-change counts changed")
    checks.append("functional equivalence")

    attack = audit["attack"]
    require(attack["queries"] == 433 and
            attack["weight_status"] == attack["bias_status"] == "exact",
            "dense-offset recovery changed")
    require(attack["range_audit_accept_percent"] ==
            attack["sparsity_audit_accept_percent"] == 100.0,
            "dense-offset audit acceptance changed")
    require([attack["attack_query_minimum"], attack["attack_query_maximum"]] == [22, 250],
            "dense-offset input range changed")
    checks.append("audit bypass")

    require(tfhe["status"] == "exact" and tfhe["queries"] == 17,
            "later-layer TFHE result changed")
    require(tfhe["max_weight_lattice_error"] == tfhe["max_bias_lattice_error"] == 0,
            "later-layer TFHE error changed")
    checks.append("later-layer TFHE")

    require(protection["baseline_affine"]["exact"],
            "protected-activation baseline is not exact")
    protected = protection["protected_affine_relu"]
    require(protected["status"] == "compile_unsupported",
            "8-bit protected-ReLU compilation status changed")
    require(
        "18-bit value" in protected["error"] and
        "16-bit table lookups" in protected["error"],
        "protected-ReLU width boundary changed",
    )
    checks.append("protected-activation feasibility")
    require(protection_small["protected_affine_relu"]["status"] == "exact",
            "five-bit protected-ReLU result changed")
    require(protection_small["protected_affine_relu"]["exact"],
            "five-bit protected-ReLU is not exact")
    require(protection_small["online_latency_ratio"] > 3000,
            "five-bit protected-ReLU latency ratio changed")
    checks.append("protected-activation reduced-range cost")

    pairs = {row["variant"]: row for row in private_embeddings["pairs"]}
    private_expected = {
        "gpt2-wte": (50257, 0.9643233778379131, 0.9730385816901128),
        "gpt2-pos0": (50257, 1.0, 1.0),
        "bert-wte": (30522, 0.999148155428871, 0.999148155428871),
        "bert-pos0": (30522, 1.0, 1.0),
    }
    require(set(pairs) == set(private_expected),
            "private-embedding variants changed")
    for variant, (n, top1, top5) in private_expected.items():
        row = pairs[variant]
        require(row["index_kind"] == "flat" and row["n"] == n,
                f"{variant} search scope changed")
        require(abs(row["certified_top1"] - top1) < 1e-12 and
                abs(row["top5_or_ann_target"] - top5) < 1e-12,
                f"{variant} recovery rate changed")
        require(row["ann_missed_known_target"] == 0,
                f"{variant} exact-index condition changed")
    require(all(adapter["embedding_frozen_by_artifact"] and
                not adapter["embedding_state_keys"]
                for adapter in private_embeddings["adapters"]),
            "LoRA embedding-artifact boundary changed")
    checks.append("private embedding orbit recovery")

    margin_pairs = {row["variant"]: row for row in stip_margin["pairs"]}
    margin_expected = {
        "gpt2-wte": (50257, 36966, 0.7355393278548262,
                     0.9732962572377977),
        "gpt2-pos0": (50257, 48999, 0.974968661082038,
                      1.0),
        "bert-wte": (30522, 30495, 0.999115392176135,
                     0.999379136360658),
        "bert-pos0": (30522, 30518, 0.999868946989057,
                      1.0),
    }
    require(set(margin_pairs) == set(margin_expected),
            "STIP margin variants changed")
    for variant, (n, certified_rows, certified_rate, partial75) in (
        margin_expected.items()
    ):
        row = margin_pairs[variant]
        phase75 = next(
            item for item in row["controlled_partial_row_update"]["phase"]
            if item["updated_fraction"] == 0.75
        )
        require(row["shape"][0] == n and
                row["certificate"]["certified_rows"] == certified_rows,
                f"{variant} certified row count changed")
        require(abs(row["certificate"]["certified_fraction"] -
                    certified_rate) < 1e-12,
                f"{variant} certified rate changed")
        require(abs(phase75["mean_semantic_top1"] - partial75) < 1e-12,
                f"{variant} partial-update boundary changed")
        require(not row["controls"]["token_id_reindexing_changed_result"],
                f"{variant} ID-reindexing control changed")
        require(row["controls"]["private_signature_exact_collision_rows"] == 0,
                f"{variant} private signature collision changed")
    checks.append("STIP margin and alignment boundary")

    require(stip_trained["runs"] == 3 and
            stip_trained["seeds"] == [20260802, 20260803, 20260804],
            "trained STIP seed campaign changed")
    require(stip_trained["train_fraction_of_active"] == 0.25 and
            stip_trained["steps"] == 200 and
            stip_trained["selection"] == "frequency",
            "trained STIP protocol changed")
    require(stip_trained["pooled_changed_rows"] == 13285 and
            all(row["changed_rows"] == row["selected_rows"] and
                row["changed_unselected_rows"] == 0
                for row in stip_trained["seed_results"]),
            "trained STIP row-isolation evidence changed")
    require(abs(stip_trained["selected_top1"]["mean"] -
                0.9685372124405767) < 1e-12 and
            abs(stip_trained["selected_top1"]["min"] -
                0.9675033617212012) < 1e-12,
            "trained STIP Top-1 result changed")
    require(abs(stip_trained["selected_top5"]["mean"] -
                0.9718518382823594) < 1e-12 and
            abs(stip_trained["selected_certificate"]["mean"] -
                0.3220785045528037) < 1e-12,
            "trained STIP Top-5/certificate result changed")
    require(stip_trained["validation_loss_after"]["mean"] <
            stip_trained["validation_loss_before"]["mean"],
            "trained STIP validation loss did not improve")
    checks.append("independently trained partial-embedding recovery")

    require(stip_fraction_sweep["fractions"] == [0.5, 0.75, 1.0] and
            stip_fraction_sweep["runs"] == 3,
            "trained STIP fraction sweep changed")
    require(stip_fraction_sweep["all_selected_rows_changed"] and
            stip_fraction_sweep["all_unselected_rows_unchanged"],
            "trained STIP fraction row isolation changed")
    require(abs(stip_fraction_sweep["selected_top1_min"] -
                0.9670642958961663) < 1e-12 and
            abs(stip_fraction_sweep["selected_top1_max"] -
                0.9682646860229575) < 1e-12 and
            stip_fraction_sweep["no_failure_transition_above_95_percent"],
            "trained STIP fraction boundary changed")
    checks.append("trained partial-embedding fraction boundary")

    gelo_expected = [
        (gelo_base, "gpt2", 128, 16),
        (gelo_large, "gpt2", 512, 32),
        (gelo_medium, "gpt2-medium", 128, 16),
    ]
    for record, model, candidates, seq_len in gelo_expected:
        require(record["model"] == model and
                record["candidate_count"] == candidates and
                record["seq_len"] == seq_len,
                "GELO campaign identity changed")
        require(len(record["results"]) == 24 and record["trials"] == 20,
                "GELO setting count changed")
        require(sum(row["exact_sets"] for row in record["results"]) == 480,
                "GELO exact-set count changed")
        require(min(row["mean_recall"] for row in record["results"]) == 1.0 and
                min(row["min_gap"] for row in record["results"]) > 0,
                "GELO ranking separation changed")
    checks.append("GELO candidate presence")
    require(
        gelo_official["official_commit"]
        == "786668f20936ae794d4e39483f401492e82d257e",
        "official GELO commit changed",
    )
    require(
        gelo_official["capture_point"]
        == "GeloObfuscatedLinear._project remote client input",
        "official GELO capture point changed",
    )
    require(len(gelo_official["results"]) == 12 and
            gelo_official["trials"] == 20 and
            sum(row["exact_sets"] for row in gelo_official["results"]) == 240,
            "official GELO replay count changed")
    require(min(row["mean_recall"] for row in gelo_official["results"]) == 1.0 and
            min(row["min_gap"] for row in gelo_official["results"]) > 0,
            "official GELO replay separation changed")
    checks.append("official GELO wrapper replay")

    require("\\def\\JOURNALMARKS{1}" in marked_tex,
            "marked manuscript does not enable journal marks")
    require(tex.count("\\begin{jnewblock}") ==
            tex.count("\\end{jnewblock}") == 10,
            "journal-only block coverage changed")
    require(tex.count("\\jfloatcolor") == 10,
            "journal-only float-color count changed")
    journal_float_captions = [
        "Target-specific transcript map",
        "Fresh-output-permutation row recovery",
        "Pretrained ImageNet first-layer recovery",
        "Collision stress test over",
        "Fresh-session orbit recovery on 21",
        "Deterministic and fresh-session recovery",
        "Private embedding recovery against",
        "GELO candidate-presence recovery",
        "Safhire defense boundary",
    ]
    for caption in journal_float_captions:
        pattern = (
            r"\\begin\{(?:table|table\*|figure\*)\}\[t\]"
            r"\s*\\jfloatcolor(?:(?!\\end\{(?:table|table\*|figure\*)\}).)*?"
            r"\\caption\{" + re.escape(caption)
        )
        require(re.search(pattern, tex, flags=re.DOTALL) is not None,
                f"journal-only float is not explicitly marked: {caption}")
    inherited_table = re.search(
        r"\\begin\{table\}\[t\](.*?)"
        r"\\caption\{Sorted-spectrum evidence inherited",
        tex,
        flags=re.DOTALL,
    )
    require(inherited_table is not None and
            "\\jfloatcolor" not in inherited_table.group(1),
            "preliminary-version table was incorrectly marked as new")
    checks.append("journal-only red marking coverage")

    primary_citations = {
        "Safhire": "BCD+25",
        "STIP": "YZL24",
        "GELO": "BelikovFedotovGELO26",
    }
    for system, key in primary_citations.items():
        require(f"\\cite{{{key}}}" in tex,
                f"primary citation missing for {system}: {key}")
    require(tex.count("\\cite{BelikovFedotovGELO26}") >= 5,
            "GELO primary citation is not visible across framing and analysis")
    require("\\cite{BelikovFedotovGELOArtifact26}" in tex,
            "GELO official code artifact citation missing")
    require("Safhire preprint" in tex and "GELO preprint" in tex,
            "preprint status missing from abstract")
    require("Version 1, 1 September 2025" in ref_bib and
            "https://arxiv.org/abs/2509.01253" in ref_bib,
            "Safhire preprint metadata changed")
    require("Version 3, 19 June 2026" in journal_ref_bib and
            "https://arxiv.org/abs/2603.05035" in journal_ref_bib,
            "GELO preprint metadata changed")
    require("@misc{BelikovFedotovGELOArtifact26" in journal_ref_bib and
            "786668f20936ae794d4e39483f401492e82d257e" in journal_ref_bib and
            "https://github.com/noskill/gelo" in journal_ref_bib,
            "GELO official artifact metadata changed")
    checks.append("primary-system citations and preprint status")

    normalized_tex = re.sub(r"\s+", " ", tex)
    required_snippets = [
        "433 bias-anchored or 417 bias-free queries",
        "5,712 total",
        "$[0,15]$ (4 bit) & 1 & 3/21",
        "$[0,31]$ (5 bit) & 4 & 5/21",
        "$[0,255]$ (8 bit) & 12 & 15/21",
        "complete $10\\times64$ final dense classifier",
        "Each of the 267 queries opens a unique session",
        "rejects an attempted duplicate after every query",
        "($99.95\\%$)",
        "100\\% prediction agreement on all 10,000 test images",
        "22--250",
        "4.038 seconds in total",
        "15.1 ms/query",
        "requires 18 bits",
        "supports table lookups only through 16 bits",
        "58.77 seconds per vector",
        "$3{,}122\\times$ ratio",
        "few-coefficient first-round idea",
        "not a remote exploit or a complete Julia Safhire deployment",
        "complete public-base vocabulary by exhaustive inner-product search",
        "36,966 of 50,257",
        "30,495 of 30,522",
        "three separate partial-embedding checkpoints",
        "all 13,285 selected rows change",
        "96.85\\% mean Top-1 recovery",
        "96.75\\% worst",
        "32.21\\% sufficient certificate",
        "one-seed boundary sweep at 50\\%, 75\\%, and 100\\%",
        "8,886--17,772 rows",
        "96.71--96.83\\% Top-1",
        "Margin-certified private recovery",
        "Unaligned-dictionary nonidentifiability",
        "The three attacks follow one proof pattern, but not one universal security game.",
        "GELO explicitly evaluates text-presence detection as a privacy goal",
        "A preliminary version appeared at ESORICS 2026",
        "all 1,680 trials are exact",
        "All 240 source sets are recovered",
        "authors' public code artifact",
    ]
    missing = [
        snippet for snippet in required_snippets if snippet not in normalized_tex
    ]
    require(not missing, f"manuscript snippets missing: {missing}")
    checks.append("manuscript claims and scope")

    result = {
        "audit": "manuscript-to-artifact consistency",
        "status": "pass",
        "checks": checks,
        "canonical_inputs": [
            "interpolation_recovery.json",
            "session_lifecycle_recovery.json",
            "resnet_permutation_equivalence.json",
            "offset_query_audit.json",
            "logs/49_tfhe_fixed_session_later_layer.json",
            "fresh_session_orbit_recovery_input15.json",
            "fresh_session_orbit_recovery_input31.json",
            "fresh_session_orbit_recovery_input255.json",
            "logs/58_tfhe_state_machine_fresh_session_orbit_recovery.json",
            "logs/54_tfhe_protected_activation_input255.json",
            "logs/54_tfhe_protected_activation_input31.json",
            "data_side/private_embedding_orbits_exact_fullvocab_20260802.json",
            "data_side/stip_margin_phase_exact_rows_20260802.json",
            "data_side/stip_independent_embedding_frequency_f025_3seed_20260802.json",
            "data_side/stip_independent_embedding_frequency_phase_sweep_20260802.json",
            "data_side/gelo_rowspace_gpt2_20260802.json",
            "data_side/gelo_rowspace_gpt2_512x32_20260802.json",
            "data_side/gelo_rowspace_gpt2medium_20260802.json",
            "data_side/gelo_official_transcript_gpt2_20260802.json",
            "tdsc_main.tex",
            "tdsc_main_marked.tex",
            "ref.bib",
            "journal_ref.bib",
        ],
    }
    destination = OUT / "artifact_consistency_audit.json"
    destination.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
