# Quartz eight-layer LoRA verification harness

This harness tests the experimental **Q/V LoRA trained in Qwen2.5-0.5B
internal transformer layers 16–23**. It is intentionally more sceptical
than the original two-example demonstration. It accepts or rejects
concrete, reproducible claims rather than awarding a vague "AI learned"
badge.

The default adapter is the **four-step Samsung S26** checkpoint made on
2026-10-28, published as a small, synthetic-sentiment-trained sample
in verification/sample/. The base GGUF and trained adapter have pinned
SHA-256s. The 380 MB original model is NOT included; supply the
byte-identical frozen GGUF with --model /path/to/qwen05.gguf or set
QUARTZ_GGUF=/path/to/qwen05.gguf. Neither base GGUF nor checkpoint
is written or replaced by the audit/evaluation commands.
See verification/sample/README.md for its origins and limitations.

## Entry points

Run on the owner's PC from the isolated Quartz repository:

    python3 verification/verify.py --mode audit
    python3 verification/verify.py --mode verify
    python3 verification/verify.py --mode verify --repeat-evaluation
    python3 verification/verify.py --mode full --repeat-train --timeout-seconds 220
    python3 -m unittest discover -s verification -p 'test_*.py' -v

Modes:

- **audit** — SHA256 verification of the frozen GGUF and input adapter,
  strict safetensors structural and numerical validation, exactly 32
  trainable adapter tensors, learned Q and V weights in all 8 layers,
  identity metadata and fixture contamination checks. Outputs
  PASS_AUDIT_ONLY, never a claim of behavioural generalisation.
- **verify** — all audit gates, actual Quartz tokenizer single-token
  checks, a six-example independent holdout, a six-example *incorrect
  label* negative control, negation/compositional challenge prompts,
  unrelated-completion regression diagnostics, native wrong-base-model
  and truncated-file rejection, and model/adapter integrity after run.
  A PASS means the known checkpoint passed this narrow controlled suite.
- **full** — everything above plus the full Rust unit/regression suite
  and another real-GGUF two-layer numerical derivative check for Q.A,
  Q.B, V.A and V.B across a transformer layer boundary. With
  --repeat-train, train **twice** at the same seed and compare safetensors
  SHA256 byte-for-byte. With --repeat-evaluation, compare per-example
  losses across two independent model processes.

Result status is explicit: PASS when required checks pass with no WARN
diagnostics, PASS_WITH_WARNINGS when required checks pass but optional
challenge/label-preference diagnostics raise a warning,
PASS_AUDIT_ONLY for structural audit, INCOMPLETE when a mandatory
negative control was disabled, and FAIL for any failed required gate.
Exit codes are 0 for completed PASS variants, 2 for FAIL, and
3 for INCOMPLETE. Neither warning nor an audit-only pass implies
production-readiness.

A report is written to a new, never overwritten path:
verification/runs/<UTC timestamp>-<unique id>/report.json and report.md.
Native stdout, stderr, exit code and duration are preserved in each
named check's log. The JSON also records the runner script SHA-256 and
the pinned fixture manifest SHA-256 so uncommitted harness versions
can be distinguished from a subsequent Git commit. A nonzero process exit, timeout, malformed/misleading
metric, data overlap or broken integrity check causes FAIL, not a
silent PASS.

## Acceptance conditions for behavioural verification

1. Pinned base and checkpoint identity, valid contiguous F32
   safetensors payload (no NaN/Inf, unexpected tensors or layer names),
   all Q.B and V.B norms nonzero.
2. At least six *distinct* unseen prompts, with at least two examples
   of each label, and no exact normalized prompt reuse across original
   train, internal heldout, fresh, challenge or unrelated fixtures.
3. Targets must encode as exactly ONE token under the actual GGUF
   tokenizer. Engine output must include every sample and a summary
   mathematically consistent with per-example data.
4. Minimum 25% mean heldout cross-entropy reduction AND at least 80%
   of individual prompts improved. This threshold is an experimental
   smoke-test acceptance condition, not a statistical confidence bound.
5. Deliberately inverted true labels must NOT also improve: their mean
   loss must get worse, demonstrating target-dependent behaviour.
6. Wrong-GGUF-hash and deliberately truncated checkpoint must be
   rejected by the *native* safetensors loader for the expected reason.
7. The model and checkpoint must retain their original SHA256 after
   tests. These operations are read-only except inside the new report
   directory.

Negation, composition and unrelated-prompt performance are **diagnostic**:
the harness shows regressions as WARN, never hides them, but they do not
claim reliable natural-language reasoning. They must be reviewed before
using the adapter in a production app.

## Samsung S26 companion

The file verification/phone_verify.sh is a lightweight Android shell
runner for an **independent native saved-adapter evaluation**. It
requires the already installed Quartz ARM64 binary, frozen GGUF,
verified four-step adapter and holdout_fresh.jsonl on local storage.
It checks their pinned SHA256 values, memory/battery/thermal thresholds,
runs the true Quartz eval-qv-suffix command with two CPU threads and
checks all six sample results before checking the model and adapter
hashes again. It writes a structured JSON report and full log under:

    /sdcard/Download/quartz-lora/verification-s26-<UTC>.json
    /sdcard/Download/quartz-lora/verification-s26-<UTC>.log

It never trains, downloads, merges a model, modifies another app or
requests external access. It does not replace the PC negative-controls
or independent numerical derivative checks.

## Interpretation

This is **LoRA applied to real internal Q/V projections in eight
connected blocks**, with a frozen GGUF. It is NOT full-parameter
transformer weight training. The small sentiment tasks are heavily
templated, can reward task-specific shortcuts, and cannot establish
broad capability or general learning. CE improvements on the control
suite are evidence for correctness of the local training/evaluation
pipeline under the tested workload.

A future production-release gate requires larger preregistered tasks,
label balance across prompt templates, repeated independent seeds,
full-sequence teacher-forced targets, runtime safety and adapter-aware
inference integration. CI should treat WARN results as review-required
even if the narrow PASS conditions are satisfied.

No internet access, cloud inference, GPU training or changes to ARC are
required for the default audit/verify modes.


Cross-device comparison:
  python3 verification/compare_phone.py --pc PATH/report.json     --phone PATH/phone-s26.json

The comparator requires the same frozen model and adapter SHA256s,
matching sample counts, normal device safety, and PC/S26 mean CE values
within 0.0001. It is local numerical consistency, not cryptographic
hardware identity attestation.

See RESULTS_2026-10-28.md for the first actual PC and phone findings,
including the exact counterexample and negation failures.
