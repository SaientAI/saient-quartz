#!/usr/bin/env python3
"""Quartz multi-layer Q/V LoRA verification harness.

Uses only Python stdlib. A result marked PASS requires independent evaluation,
negative controls, provenance and structural checks; an audit-only result
cannot establish learning or generalisation.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path(os.environ.get("QUARTZ_GGUF", str(ROOT / "models/qwen05.gguf")))
DEFAULT_ADAPTER = ROOT / "verification/sample/adapter-qwen25-qv16-23-r2-step4-s26.safetensors"
DEFAULT_TRAIN = ROOT / "experiments/android-lora-smoke/multi-3-samples.jsonl"
DEFAULT_FRESH = ROOT / "verification/fixtures/holdout_fresh.jsonl"
DEFAULT_CHALLENGE = ROOT / "verification/fixtures/challenge_negation.jsonl"
DEFAULT_UNRELATED = ROOT / "verification/fixtures/unrelated_regression.jsonl"
KNOWN_MODEL_SHA = "6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653"
KNOWN_4STEP_SHA = "adfb838b23efefadf9423df0d3b6574920b378173c6ef026a31b5ec4a56c710d"
SAMPLE = re.compile(
    r"^MULTI_EVAL_SAMPLE:\s+(\d+)\s+base=([+-]?[\d.eE]+)\s+adapted=([+-]?[\d.eE]+)\s+elapsed_ms=(\d+)$",
    re.MULTILINE,
)
SUMMARY = re.compile(
    r"^MULTI_EVAL_RESULT:\s+(IMPROVED|NOT_IMPROVED)\s+unseen=(\d+)\s+layers=(\d+)"
    r"\s+base=([+-]?[\d.eE]+)\s+adapted=([+-]?[\d.eE]+)\s+total_ms=(\d+)\s+frozen=true$",
    re.MULTILINE,
)
MODEL_METRICS = ("model_identity", "adapter_identity", "adapter_structure", "dataset_isolation")


class VerificationError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def no_duplicate_keys(items):
    d = {}
    for k, v in items:
        if k in d:
            raise VerificationError("duplicate JSON key: " + k)
        d[k] = v
    return d


def read_jsonl(path: Path, required_split=None) -> list[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=no_duplicate_keys)
            except (ValueError, VerificationError) as e:
                raise VerificationError(f"{path}:{idx}: {e}") from e
            if not isinstance(row, dict) or set(row) != {"prompt", "target", "split"}:
                raise VerificationError(f"{path}:{idx}: JSONL record must be a prompt/target/split object")
            if not all(isinstance(row[k], str) for k in row):
                raise VerificationError(f"{path}:{idx}: JSONL values must be strings")
            if not row["prompt"] or not row["target"]:
                raise VerificationError(f"{path}:{idx}: missing prompt/target")
            if len(row["prompt"]) > 1024 or len(row["target"]) > 128:
                raise VerificationError(f"{path}:{idx}: sample exceeds trainer limits")
            if row["split"] not in ("train", "heldout"):
                raise VerificationError(f"{path}:{idx}: invalid split")
            if required_split is not None and row["split"] != required_split:
                raise VerificationError(f"{path}:{idx}: wrong split")
            data.append(row)
    if not data or len(data) > 6:
        raise VerificationError(f"{path}: dataset must have 1..6 rows")
    return data


def normalized_prompt(text: str) -> str:
    return " ".join(text.lower().split())


def check_isolation(train: list[dict], *evaluations: list[dict]):
    if len([r for r in train if r["split"] == "train"]) < 2:
        raise VerificationError("training split is too small")
    if not any(r["split"] == "heldout" for r in train):
        raise VerificationError("training file lacks internal heldout")
    train_all = [normalized_prompt(x["prompt"]) for x in train]
    if len(set(train_all)) != len(train_all):
        raise VerificationError("duplicate prompts in training+internal heldout")
    seen = set(train_all)
    counts = []
    for rows in evaluations:
        names = [normalized_prompt(x["prompt"]) for x in rows]
        if len(set(names)) != len(names):
            raise VerificationError("duplicate prompts within external evaluation")
        if seen.intersection(names):
            raise VerificationError("evaluation prompt overlaps another fixture (leakage)")
        seen.update(names)
        counts.append(len(names))
    return {"initial": len(train), "external_counts": counts, "all_unique": len(seen)}


def inverted_labels(rows: list[dict]) -> list[dict]:
    flipped = []
    for row in rows:
        target = row["target"]
        if target not in (" positive", " negative"):
            raise VerificationError("counterfactual requires binary sentiment labels")
        flipped.append({
            "prompt": row["prompt"],
            "target": " negative" if target == " positive" else " positive",
            "split": "heldout",
        })
    return flipped


def adapter_detail(
    path: Path, expected_base: str, layers: int,
    hidden: int, q_dim: int, v_dim: int
) -> dict:
    size = path.stat().st_size
    if not (128 <= size <= 8_000_000):
        raise VerificationError("adapter length outside expected bounds")
    with path.open("rb") as f:
        length_data = f.read(8)
        if len(length_data) != 8:
            raise VerificationError("truncated adapter")
        hdr_len = struct.unpack("<Q", length_data)[0]
        if not (2 <= hdr_len <= 2_000_000) or hdr_len > size - 8:
            raise VerificationError("invalid safetensors header length")
        header = json.loads(f.read(hdr_len), object_pairs_hook=no_duplicate_keys)
        body = f.read()
    if len(body) != size - 8 - hdr_len:
        raise VerificationError("safetensors data length mismatch")
    meta = header.get("__metadata__")
    if not isinstance(meta, dict):
        raise VerificationError("missing safetensors metadata")
    if meta.get("format") != "quartz-multi-qv-v0":
        raise VerificationError("unexpected adapter format")
    if meta.get("target") != "suffix_causal_qv":
        raise VerificationError("unexpected adapter target")
    if meta.get("base_gguf_sha256") != expected_base:
        raise VerificationError("adapter bound to wrong GGUF")
    try:
        first = int(meta["first_layer"])
        layer_count = int(meta["layer_count"])
        rank = int(meta["rank"])
        alpha = float(meta["alpha"])
        steps = int(meta["steps"])
    except (KeyError, ValueError, TypeError) as exc:
        raise VerificationError("invalid adapter scalar metadata") from exc
    if first < 0 or layers != layer_count or rank not in range(1, 9):
        raise VerificationError("adapter layer count/rank mismatch")
    if not (math.isfinite(alpha) and alpha > 0 and 1 <= steps <= 8):
        raise VerificationError("adapter alpha/update count invalid")
    if first + layers != 24:
        raise VerificationError("adapter is not a suffix of Qwen2.5-0.5B")
    specs = []
    for layer in range(first, first + layers):
        specs += [
            (f"lora.{layer}.q.A", [rank, hidden], "A", layer),
            (f"lora.{layer}.q.B", [q_dim, rank], "qB", layer),
            (f"lora.{layer}.v.A", [rank, hidden], "A", layer),
            (f"lora.{layer}.v.B", [v_dim, rank], "vB", layer),
        ]
    if set(header) != {"__metadata__"} | {v[0] for v in specs}:
        raise VerificationError("missing/extra safetensors tensor name")
    start = 0
    norms = {}
    for name, dims, label, layer in specs:
        desc = header[name]
        if desc.get("dtype") != "F32" or desc.get("shape") != dims:
            raise VerificationError(name + " has wrong dtype or dimensions")
        offsets = desc.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise VerificationError(name + " has invalid offsets")
        end_expected = start + math.prod(dims) * 4
        if offsets != [start, end_expected] or end_expected > len(body):
            raise VerificationError(name + " overlaps, skips or truncates data")
        floats = struct.iter_unpack("<f", memoryview(body)[start:end_expected])
        norm_squared = 0.0
        nonzero = 0
        for (val,) in floats:
            if not math.isfinite(val):
                raise VerificationError(name + " contains NaN/Inf")
            norm_squared += float(val) ** 2
            if val != 0.0:
                nonzero += 1
        if nonzero == 0:
            raise VerificationError(name + " has all-zero weights")
        if label in ("qB", "vB"):
            norm = math.sqrt(norm_squared)
            if norm <= 1e-8:
                raise VerificationError(name + " did not learn any nontrivial update")
            norms[name] = round(norm, 8)
        start = end_expected
    if start != len(body):
        raise VerificationError("safetensors payload has unexpected trailing bytes")
    return {
        "sha256": sha256_file(path),
        "size_bytes": size,
        "layer_range": [first, first + layers - 1],
        "tensor_count": len(specs),
        "rank": rank,
        "alpha": alpha,
        "steps": steps,
        "trainable_f32_weights": start // 4,
        "q_v_b_norms": norms,
    }


def parse_eval(output: str, expected: int, layers: int):
    records = [(int(a), float(b), float(c), int(ms))
               for a, b, c, ms in SAMPLE.findall(output)]
    summary = SUMMARY.findall(output)
    if len(summary) != 1:
        raise VerificationError("missing or ambiguous MULTI_EVAL_RESULT")
    status, n_text, layer_text, base_text, adapted_text, ms = summary[0]
    if int(n_text) != expected or int(layer_text) != layers:
        raise VerificationError("eval summary count/layer mismatch")
    if len(records) != expected or [r[0] for r in records] != list(range(1, expected+1)):
        raise VerificationError("missing, duplicate or reordered per-sample evaluation")
    if not all(math.isfinite(v) and v >= 0
               for r in records for v in r[1:3]):
        raise VerificationError("invalid CE metric (NaN, negative or infinity)")
    mean_b = sum(r[1] for r in records) / expected
    mean_a = sum(r[2] for r in records) / expected
    if abs(mean_b-float(base_text)) > 5e-5 or abs(mean_a-float(adapted_text)) > 5e-5:
        raise VerificationError("reported eval average does not match individual rows")
    if status != ("IMPROVED" if mean_a < mean_b else "NOT_IMPROVED"):
        raise VerificationError("eval status conflicts with observed losses")
    return {
        "base_mean": round(mean_b, 7),
        "adapted_mean": round(mean_a, 7),
        "relative_improvement": round((mean_b-mean_a)/max(mean_b,1e-9), 7),
        "improved_samples": sum(1 for _, a, b, _ in records if b < a),
        "count": expected,
        "engine_ms": int(ms),
        "examples": [{"index": i, "base": a, "adapted": b, "delta": round(b-a, 7)}
                     for i, a, b, _ in records],
    }


def run_program(argv, cwd: Path, timeout_s: int, log: Path):
    start = time.monotonic()
    env = dict(os.environ)
    env.update({"RAYON_NUM_THREADS": "2", "CARGO_BUILD_JOBS": "2",
                "CARGO_INCREMENTAL": "0", "OMP_NUM_THREADS": "2"})
    proc = subprocess.Popen(
        [str(x) for x in argv], cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, errors="replace",
        env=env, start_new_session=True, preexec_fn=lambda: os.nice(15)
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
    duration = round(time.monotonic() - start, 3)
    log.write_text(
        "command: " + " ".join(str(x) for x in argv) +
        f"\nexit: {proc.returncode}, timed_out: {timed_out}, elapsed_s: {duration}\n\nSTDOUT:\n" +
        stdout + "\nSTDERR:\n" + stderr, encoding="utf-8"
    )
    return proc.returncode, stdout, stderr, timed_out, duration


def save_json(path: Path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def report_markdown(report):
    lines = [
        "# Quartz Q/V LoRA verification",
        "",
        f"**Result:** {report['result']}",
        f"**Mode:** {report['mode']}",
        f"**Time:** {report['timestamp_utc']}",
        f"**Git revision:** {report['git_revision']}",
        "",
        "## Gates",
        "",
        "| Gate | Status | Evidence |",
        "|---|---|---|",
    ]
    for g in report["gates"]:
        desc = str(g["detail"]).replace("|","/").replace("\n"," ")
        lines.append(f"| {g['name']} | {g['status']} | {desc[:320]} |")
    lines += ["", "## Interpretation", "",
        "PASS validates this particular adapter, frozen model and controlled",
        "fixtures. It does NOT establish general language understanding or",
        "full-parameter training. Challenge and unrelated-prompt results are",
        "reported separately, never hidden by a favourable aggregate.",
        "",
        "Logs, adversarial fixture and machine-readable report are in this run folder.",
    ]
    return "\n".join(lines) + "\n"


def git_rev():
    p = subprocess.run(["git","rev-parse","--short=12","HEAD"],
                       cwd=ROOT, text=True,capture_output=True)
    return p.stdout.strip() if p.returncode == 0 else "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit","verify","full"),default="verify")
    parser.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    parser.add_argument("--adapter",type=Path,default=DEFAULT_ADAPTER)
    parser.add_argument("--binary",type=Path,default=ROOT/"target/release/quartz")
    parser.add_argument("--fixture-manifest",type=Path,
                        default=ROOT/"verification/manifest.json")
    parser.add_argument("--train",type=Path,default=DEFAULT_TRAIN)
    parser.add_argument("--fresh",type=Path,default=DEFAULT_FRESH)
    parser.add_argument("--challenge",type=Path,default=DEFAULT_CHALLENGE)
    parser.add_argument("--unrelated",type=Path,default=DEFAULT_UNRELATED)
    parser.add_argument("--layers",type=int,default=8)
    parser.add_argument("--hidden",type=int,default=896)
    parser.add_argument("--q-dim",type=int,default=896)
    parser.add_argument("--v-dim",type=int,default=128)
    parser.add_argument("--expected-model-sha",default=KNOWN_MODEL_SHA)
    parser.add_argument("--expected-adapter-sha",default=KNOWN_4STEP_SHA)
    parser.add_argument("--min-fresh",type=int,default=6)
    parser.add_argument("--timeout-seconds",type=int,default=150)
    parser.add_argument("--output-root",type=Path,default=ROOT/"verification/runs")
    parser.add_argument("--skip-challenge",action="store_true")
    parser.add_argument("--skip-unrelated",action="store_true")
    parser.add_argument("--repeat-evaluation",action="store_true")
    parser.add_argument("--skip-incorrect-label-control",action="store_true")
    parser.add_argument("--repeat-train",action="store_true",
                        help="In full mode, run two new deterministic two-layer train jobs.")
    a=parser.parse_args()
    if not (1 <= a.layers <= 8 and 1 <= a.min_fresh <= 6 and 10 <= a.timeout_seconds <= 600):
        parser.error("invalid size/time budget")
    timestamp = dt.datetime.now(dt.timezone.utc)
    root = a.output_root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    run_dir = root / (timestamp.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(mode=0o700)
    data = {
        "schema_version": 1, "timestamp_utc": timestamp.isoformat(),
        "mode": a.mode, "git_revision": git_rev(),
        "result": "INCOMPLETE", "gates":[], "metrics":{}, "files":{},
        "model": str(a.model), "adapter": str(a.adapter),
        "binary": str(a.binary), "run_directory": str(run_dir),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "limitations": ["next-token-only", "sentiment-fixtures", "frozen-gguf",
                        "no-full-SFT", "non-independent-model-benchmark"],
    }
    failed = False
    incomplete = False
    def gate(name,status,detail):
        nonlocal failed,incomplete
        data["gates"].append({"name":name,"status":status,"detail":detail})
        print(f"[{status}] {name}: {detail}",flush=True)
        if status=="FAIL":failed=True
        if name=="negative_control_opposite_labels" and status=="SKIP":
            incomplete=True
    def result_status():
        if failed:return "FAIL"
        if incomplete:return "INCOMPLETE"
        if a.mode=="audit":return "PASS_AUDIT_ONLY"
        if any(g["status"]=="WARN" for g in data["gates"]):
            return "PASS_WITH_WARNINGS"
        return "PASS"
    def flush():
        data["result"] = result_status()
        save_json(run_dir/"report.json",data)
        (run_dir/"report.md").write_text(report_markdown(data),encoding="utf-8")
    def command(name,cmd,expected_success=True):
        rc,out,err,to,secs=run_program(cmd,ROOT,a.timeout_seconds,run_dir/(name+".log"))
        data["metrics"][name+"_duration_s"]=secs
        if to:gate(name,"FAIL",f"timeout at {secs}s")
        elif bool(rc==0)!=bool(expected_success):
            gate(name,"FAIL",f"unexpected exit {rc}; see {name}.log")
        return rc,out,err,to
    try:
        for path in (a.model,a.adapter,a.train,a.fresh,a.challenge,
                     a.unrelated,a.fixture_manifest):
            if not path.is_file():
                raise VerificationError("required file missing: "+str(path))
        model_hash_before=sha256_file(a.model)
        data["files"]["model_sha256"]=model_hash_before
        if model_hash_before != a.expected_model_sha:
            gate("model_identity","FAIL","actual model SHA does not match expected pinned base")
        else:
            gate("model_identity","PASS","frozen base SHA256 verified")
        actual_adapter_hash=sha256_file(a.adapter)
        data["files"]["adapter_sha256"]=actual_adapter_hash
        if a.expected_adapter_sha and actual_adapter_hash != a.expected_adapter_sha:
            gate("adapter_identity","FAIL","checkpoint SHA256 does not match pinned phone artifact")
        else:
            gate("adapter_identity","PASS",f"checkpoint SHA256 {actual_adapter_hash[:16]}...")
        try:
            detail=adapter_detail(a.adapter,model_hash_before,a.layers,a.hidden,a.q_dim,a.v_dim)
            data["metrics"]["adapter"]=detail
            gate("adapter_structure","PASS",f"{detail['tensor_count']} tensors; "
                 f"{detail['layer_range']}; all Q/V B norms nonzero")
        except (ValueError,KeyError,OSError,VerificationError) as e:
            gate("adapter_structure","FAIL",str(e))
        train=read_jsonl(a.train)
        fresh=read_jsonl(a.fresh,"heldout")
        challenge=read_jsonl(a.challenge,"heldout")
        unrelated=read_jsonl(a.unrelated,"heldout")
        if len(fresh)<a.min_fresh:
            gate("evaluation_sample_count","FAIL",
                 f"need {a.min_fresh} fresh; only {len(fresh)}")
        else:
            gate("evaluation_sample_count","PASS",f"{len(fresh)} distinct prompts")
        labels=collections.Counter(r["target"] for r in fresh)
        if len(labels)<2 or min(labels.values())<2:
            gate("balanced_labels","FAIL",f"unbalanced external labels: {dict(labels)}")
        else:
            gate("balanced_labels","PASS",dict(labels))
        isolation=check_isolation(train,fresh,challenge,unrelated)
        data["metrics"]["fixture_isolation"]=isolation
        gate("dataset_isolation","PASS",f"{isolation['all_unique']} unique prompts; no literal overlap")
        hashes = {
            name:sha256_file(path) for name,path in [
                ("train",a.train),("fresh",a.fresh),
                ("challenge",a.challenge),("unrelated",a.unrelated)]
        }
        data["files"]["fixture_sha256"]=hashes
        data["files"]["fixture_manifest_sha256"]=sha256_file(a.fixture_manifest)
        manifest=json.loads(a.fixture_manifest.read_text(encoding="utf-8"),
                            object_pairs_hook=no_duplicate_keys)
        if (manifest.get("schema_version")!=1
            or manifest.get("model_sha256")!=model_hash_before
            or manifest.get("adapter_sha256")!=actual_adapter_hash
            or manifest.get("expected_layers")!=a.layers
            or manifest.get("fixtures")!=hashes
            or len(fresh)<manifest.get("limits",{}).get("min_fresh",6)):
            gate("pinned_fixture_manifest","FAIL",
                 "changed fixture, model or adapter identity relative to the pinned manifest")
        else:
            gate("pinned_fixture_manifest","PASS",
                 f"four fixture SHA256s pinned to manifest {data['files']['fixture_manifest_sha256'][:16]}...")
        print("Audit completed; report path:",run_dir/"report.json",flush=True)

        if failed or a.mode=="audit":
            flush()
            return 2 if failed else 0
        if not a.binary.is_file() or not os.access(a.binary,os.X_OK):
            gate("native_executable","FAIL","binary missing or not executable")
            flush();return 2
        data["files"]["binary_sha256"]=sha256_file(a.binary)

        # Direct GGUF tokenizer validation for unique labels. The CLI is the
        # training source of truth for whether targets are single tokens.
        for target in sorted({r["target"] for rows in (train,fresh,challenge,unrelated) for r in rows}):
            code,out,err,to=command(
                "token_"+hashlib.sha256(target.encode()).hexdigest()[:9],
                [a.binary,a.model,"--tokenize",target],
            )
            if code or to:
                continue
            token_match=re.search(r"^ids\s+:\s+\[([0-9,\s]*)\]$",out,re.MULTILINE)
            count=len(token_match.group(1).split(",")) if token_match and token_match.group(1).strip() else 0
            gate("single_token_"+target.strip(),"PASS" if count==1 else "FAIL",
                 f"GGUF encodes target {target!r} as {count} tokens")
        if failed:flush();return 2

        def evaluate(name,fixture:Path,expected:int):
            rc,out,err,to=command(
                name,[a.binary,"eval-qv-suffix",a.model,fixture,a.adapter,
                      "--layers",str(a.layers),"--max-tokens","16"])
            if rc or to:return None
            try:
                metrics=parse_eval(out,expected,a.layers)
                data["metrics"][name]=metrics
                gate(name+"_schema","PASS",f"{expected} coherent independently reported sample losses")
                return metrics
            except VerificationError as e:
                gate(name+"_schema","FAIL",str(e))
                return None

        fresh_metrics=evaluate("fresh_eval",a.fresh,len(fresh))
        if fresh_metrics:
            good=(fresh_metrics["relative_improvement"]>=0.25
                  and fresh_metrics["improved_samples"] >= math.ceil(0.8*len(fresh))
                  and fresh_metrics["base_mean"]>fresh_metrics["adapted_mean"])
            gate("genuine_fresh_generalisation","PASS" if good else "FAIL",
                 f"improved {fresh_metrics['improved_samples']}/{len(fresh)}, "
                 f"relative CE reduction {fresh_metrics['relative_improvement']:.1%}; "
                 f"limited to narrow heldout sentiment set")

        if not a.skip_incorrect_label_control:
            inverted=inverted_labels(fresh)
            control_path=run_dir/"incorrect_labels_control.jsonl"
            control_path.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in inverted),encoding="utf-8")
            bad_metrics=evaluate("wrong_label_control",control_path,len(inverted))
            if bad_metrics and fresh_metrics:
                actual=(fresh_metrics["base_mean"]-fresh_metrics["adapted_mean"])
                wrong=(bad_metrics["base_mean"]-bad_metrics["adapted_mean"])
                passed=actual>0 and wrong<0
                gate("negative_control_opposite_labels","PASS" if passed else "FAIL",
                     f"CE improvement true labels {actual:.5f}; incorrect labels {wrong:.5f} "
                     "(expected incorrect-label degradation)")
                # A strong *relative* target preference needs the correct-vs-
                # incorrect log odds to improve, not merely both probabilities
                # rising as mass moves from the rest of the vocabulary.
                shifts=[]
                for right,wrong_row in zip(fresh_metrics["examples"],bad_metrics["examples"]):
                    original_margin=wrong_row["base"]-right["base"]
                    adapted_margin=wrong_row["adapted"]-right["adapted"]
                    shifts.append(adapted_margin-original_margin)
                better=sum(1 for shift in shifts if shift>0.01)
                required=math.ceil(0.8*len(shifts))
                data["metrics"]["label_preference_margin"]={"shifts":shifts,"improved":better,"total":len(shifts)}
                gate("target_preference_margin","PASS" if better>=required else "WARN",
                     f"correct-vs-incorrect log-odds margin improved {better}/{len(shifts)}; "
                     f"requires {required} for a clean diagnostic")
        else:
            gate("negative_control_opposite_labels","SKIP","explicitly disabled")

        if a.repeat_evaluation:
            repeat=evaluate("fresh_repeat",a.fresh,len(fresh))
            if repeat and fresh_metrics:
                deltas=[
                    abs(x["adapted"]-y["adapted"])
                    for x,y in zip(fresh_metrics["examples"],repeat["examples"])
                ]
                passed=max(deltas)<=1e-5
                gate("repeated_process_determinism","PASS" if passed else "FAIL",
                     f"max independent process CE drift {max(deltas):.9f}")
        else:
            gate("repeated_process_determinism","SKIP","pass --repeat-evaluation to enable")

        if not a.skip_challenge:
            challenge_metrics=evaluate("negation_challenge",a.challenge,len(challenge))
            if challenge_metrics:
                improved=challenge_metrics["improved_samples"]
                gate("negation_challenge","WARN" if improved<len(challenge) else "PASS",
                     f"diagnostic only: {improved}/{len(challenge)} improve; not proof of negation reasoning")
        if not a.skip_unrelated:
            unrelated_metrics=evaluate("unrelated_regression",a.unrelated,len(unrelated))
            if unrelated_metrics:
                degraded=[v for v in unrelated_metrics["examples"] if v["delta"]>0.5]
                gate("unrelated_capability_regression","WARN" if degraded else "PASS",
                     f"diagnostic only: {len(degraded)}/{len(unrelated)} unrelated completions "
                     "show >0.5 CE regression")

        # These mutations must be rejected by the actual native Rust loader.
        with a.adapter.open("rb") as f:
            original=f.read()
        header_len=struct.unpack_from("<Q",original,0)[0]
        header=json.loads(original[8:8+header_len],object_pairs_hook=no_duplicate_keys)
        header["__metadata__"]["base_gguf_sha256"]="0"*64
        changed=json.dumps(header,separators=(",",":"),sort_keys=True).encode()
        mutated=run_dir/"wrong_base.safetensors"
        mutated.write_bytes(struct.pack("<Q",len(changed))+changed+original[8+header_len:])
        rc,out,err,to=command("wrong_base_rejection",
              [a.binary,"eval-qv-suffix",a.model,a.fresh,mutated,
               "--layers",str(a.layers)],expected_success=False)
        if rc and not to:
            ok="adapter model identity or format mismatch" in (out+err)
            gate("wrong_base_rejection_reason","PASS" if ok else "FAIL",
                 "native loader rejected mismatched frozen GGUF identity"
                 if ok else "unexpected rejection reason")

        trunc=run_dir/"truncated.safetensors"
        trunc.write_bytes(original[:-4])
        rc,out,err,to=command("truncated_rejection",
              [a.binary,"eval-qv-suffix",a.model,a.fresh,trunc,
               "--layers",str(a.layers)],expected_success=False)
        if rc and not to:
            ok="adapter tensor data invalid" in (out+err)
            gate("truncated_rejection_reason","PASS" if ok else "FAIL",
                 "native loader rejected truncated tensor payload"
                 if ok else "unexpected rejection reason")

        if a.mode=="full":
            if a.layers<2:
                gate("cross_layer_gradient_prerequisite","FAIL",
                     "full verification requires at least two adapted layers")
                flush();return 2
            rc,out,err,to=command("rust_regression",
                ["cargo","test","--bin","quartz","--offline","--quiet","--","--test-threads=2"])
            if rc==0 and not to:
                test_result=re.search(r"test result:\s+ok.\s+(\d+) passed;\s+(\d+) failed;\s+(\d+) ignored",out)
                gate("rust_regression_summary","PASS" if test_result and int(test_result.group(2))==0 else "FAIL",
                     test_result.group(0) if test_result else "no test summary")

            if a.repeat_train:
                hashes=[]
                for i in range(2):
                    output=run_dir/f"deterministic_run_{i}.safetensors"
                    rc,out,err,to=command(f"train_repeat_{i}",
                        [a.binary,"train-qv-suffix",a.model,a.train,output,
                         "--layers",str(a.layers),"--steps","1","--rank","2","--lr","0.01",
                         "--max-tokens","16"])
                    if rc==0 and not to and "MULTI_RESULT: PASS" in out:
                        hashes.append(sha256_file(output))
                good=len(hashes)==2 and hashes[0]==hashes[1]
                gate("deterministic_training_repeat","PASS" if good else "FAIL",
                     f"same-seed {a.layers}-layer artifact SHA256s {hashes}")
            else:
                gate("deterministic_training_repeat","SKIP",
                     "full mode: add --repeat-train for two deterministic train runs")

            # Separate from repeat-train: repeat the real-model gradient audit
            # on a fresh full-suffix checkpoint, validating dQ/dV across layers.
            path=run_dir/"numerical_gradient_output.safetensors"
            rc,out,err,to=command("real_model_numerical_gradient",
                [a.binary,"train-qv-suffix",a.model,a.train,path,
                 "--layers",str(a.layers),"--steps","1","--rank","2","--lr","0.01",
                 "--max-tokens","16","--gradient-check"])
            if rc==0 and not to:
                checks=re.findall(r"^MULTI_GRADCHECK: layer=(\d+)\s+group=(Q\.A|Q\.B|V\.A|V\.B)\s+index=\d+\s+analytic=[^\s]+\s+numeric=[^\s]+\s+rel_err=([^\s]+)",out,re.MULTILINE)
                groups={(int(x),y) for x,y,_ in checks}
                good=("MULTI_GRADCHECK: PASS" in out and len(groups)==8 and
                      "MULTI_RESULT: PASS" in out)
                gate("real_model_gradient_numerical","PASS" if good else "FAIL",
                     f"{len(groups)}/8 first/last Q/V groups matched finite differences")

        if sha256_file(a.model)!=model_hash_before:
            gate("postrun_model_unchanged","FAIL","model SHA256 changed during verification!")
        else:
            gate("postrun_model_unchanged","PASS","frozen model unchanged")
        if sha256_file(a.adapter)!=actual_adapter_hash:
            gate("postrun_adapter_unchanged","FAIL","input adapter changed during verification!")
        else:
            gate("postrun_adapter_unchanged","PASS","input adapter untouched")
        flush()
        print(f"VERIFICATION_RESULT: {data['result']}",flush=True)
        print("RESULT_JSON:",run_dir/"report.json",flush=True)
        return 2 if failed else 3 if incomplete else 0
    except (VerificationError,ValueError,KeyError,TypeError,AttributeError,
            OSError,subprocess.SubprocessError) as e:
        gate("fatal","FAIL",str(e))
        flush()
        return 2


if __name__=="__main__":
    sys.exit(main())
