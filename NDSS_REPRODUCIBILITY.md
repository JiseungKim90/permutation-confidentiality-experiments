# Artifact snapshot and canonical execution

The canonical executable copy is on lab614 at /home/user/research-vault/projects/P050/ndss-2027. This Dropbox folder is a paper-facing snapshot; run and retain experiments in the canonical server directory.

## Scope

- lib/stip_orbit.py implements the complete scale/permutation canonicalizer and exact cosine search.
- scripts/63_stip_final_full_prompt.py runs the main one-transcript public/private-baseline campaign.
- scripts/64_stip_tokenizer_private_boundary.py runs the tokenizer-extension and independent-BPE boundary studies.
- scripts/65_stip_noise_defense.py runs the utility/recovery noise sweep.
- scripts/66_stip_chosen_prompt_codebook.py runs the conditional private-embedding chosen-prompt attack.
- tests/test_stip_orbit.py checks transformation invariance and exact synthetic decoding.
- ../results/raw retains all JSON values cited in the manuscript.

## Canonical environment

- Host: user-PowerEdge-T440, Intel Xeon Silver 4208, 16 hardware threads.
- Python 3.10.12; PyTorch 2.7.1+cu126; CPU execution for reported campaigns.
- Official STIP repository pinned to d8e8b876280979efd996ace5d9ed4b3a50bb271f.
- Final STIP PDF SHA-256: 8e7ac8f1e6e728de8328abad35e66c012dff82a7d059cf8456e73aa65a6c0a77.

## Minimum validation

From the canonical server artifact directory:

    PYTHONPATH=. python3 -m pytest -q tests/test_stip_orbit.py
    python3 -m py_compile lib/stip_orbit.py scripts/63_stip_final_full_prompt.py scripts/64_stip_tokenizer_private_boundary.py scripts/65_stip_noise_defense.py scripts/66_stip_chosen_prompt_codebook.py

The validated unit-test result is 2 passed.

## Claim hierarchy

1. Main C0: known public/frozen GPT-2 124M and 355M embeddings, one cloud transcript, no chosen query, 100% exact 32-token prompts at both sizes.
2. Conditional C1: privately fine-tuned embedding, known tokenizer and authorized labeled input/output samples, 77.1% exact 32-token prompts at 4,096 queries.
3. Boundary C2: a completely unaligned tokenizer blocks plaintext labels; independent-BPE transcripts retain 97.1% unique pseudotoken signatures at 1e-6 quantization.

Do not describe C1 as assumption-free or C2 as plaintext recovery. Do not describe these experiments as an exploit of an official final TEE binary: the public repository does not contain the final alpha/TEE path.
