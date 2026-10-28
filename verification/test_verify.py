#!/usr/bin/env python3
"""Unit tests for the verification harness itself (no model required)."""
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify as v


def fake_adapter(path, corrupted=None):
    hidden, qdim, vdim, rank = 4, 4, 2, 2
    hdr = {"__metadata__": {
        "format":"quartz-multi-qv-v0",
        "target":"suffix_causal_qv",
        "base_gguf_sha256":"abc123",
        "first_layer":"22",
        "layer_count":"2",
        "rank":"2",
        "alpha":"4",
        "steps":"1",
    }}
    raw = bytearray()
    for layer in (22,23):
        for name, dims in (
            (f"lora.{layer}.q.A", [rank,hidden]),
            (f"lora.{layer}.q.B", [qdim,rank]),
            (f"lora.{layer}.v.A", [rank,hidden]),
            (f"lora.{layer}.v.B", [vdim,rank]),
        ):
            start = len(raw)
            vals = [0.002*(i+1) for i in range(math.prod(dims))]
            raw.extend(struct.pack("<"+"f"*len(vals),*vals))
            hdr[name] = {"shape":dims,"dtype":"F32","data_offsets":[start,len(raw)]}
    encoded=json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q",len(encoded))+encoded+raw)
    if corrupted=="nan":
        data=bytearray(path.read_bytes())
        start=8+len(encoded)
        data[start:start+4] = struct.pack("<f",float("nan"))
        path.write_bytes(data)
    if corrupted=="missing":
        data=path.read_bytes()
        path.write_bytes(data[:-3])


class HarnessSelfTests(unittest.TestCase):
    def test_genuine_adapter_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"fake.safetensors"
            fake_adapter(path)
            info=v.adapter_detail(path,"abc123",2,4,4,2)
            self.assertEqual(info["tensor_count"],8)
            self.assertEqual(info["layer_range"],[22,23])
            self.assertEqual(info["rank"],2)

    def test_wrong_model_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"fake.safetensors"
            fake_adapter(path)
            with self.assertRaises(v.VerificationError):
                v.adapter_detail(path,"not-abc123",2,4,4,2)

    def test_nan_and_truncated_payload_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            for corruption in ("nan","missing"):
                path=Path(td)/(corruption+".safetensors")
                fake_adapter(path,corrupted=corruption)
                with self.assertRaises(v.VerificationError):
                    v.adapter_detail(path,"abc123",2,4,4,2)

    def test_no_hidden_overlap_between_splits(self):
        t=[
            {"prompt":"Review: great\nSentiment:","target":" positive","split":"train"},
            {"prompt":"Review: bad\nSentiment:","target":" negative","split":"train"},
            {"prompt":"Review: okay\nSentiment:","target":" positive","split":"heldout"},
        ]
        e=[{"prompt":"Review: amazing\nSentiment:","target":" positive","split":"heldout"}]
        v.check_isolation(t,e)
        with self.assertRaises(v.VerificationError):
            v.check_isolation(t,e,[{"prompt":" Review:  AMAZING \nSentiment:",
                "target":" positive","split":"heldout"}])
        with self.assertRaises(v.VerificationError):
            v.check_isolation(t,[t[0]])

    def test_inverse_labels_explicit(self):
        a=[{"prompt":"x","target":" positive","split":"heldout"},
           {"prompt":"y","target":" negative","split":"heldout"}]
        b=v.inverted_labels(a)
        self.assertEqual([x["target"] for x in b],[" negative"," positive"])
        self.assertEqual([x["prompt"] for x in b],["x","y"])

    def test_eval_parser_detects_mean_fraud(self):
        logs=(
            "MULTI_EVAL_SAMPLE: 1 base=1.0000000 adapted=0.3000000 elapsed_ms=100\n"
            "MULTI_EVAL_SAMPLE: 2 base=2.0000000 adapted=0.4000000 elapsed_ms=150\n"
            "MULTI_EVAL_RESULT: IMPROVED unseen=2 layers=8 "
            "base=1.5000000 adapted=0.3500000 total_ms=200 frozen=true\n"
        )
        out=v.parse_eval(logs,2,8)
        self.assertEqual(out["count"],2)
        self.assertEqual(out["improved_samples"],2)
        with self.assertRaises(v.VerificationError):
            v.parse_eval(logs.replace("adapted=0.3500000","adapted=0.1000000"),2,8)
        with self.assertRaises(v.VerificationError):
            v.parse_eval(logs.replace("unseen=2","unseen=3"),2,8)
        with self.assertRaises(v.VerificationError):
            v.parse_eval(logs.replace("IMPROVED","NOT_IMPROVED"),2,8)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"bad.jsonl"
            p.write_text('{"prompt":"a","prompt":"b","target":" positive","split":"heldout"}\n')
            with self.assertRaises(v.VerificationError):
                v.read_jsonl(p,"heldout")


if __name__=="__main__":
    unittest.main()
