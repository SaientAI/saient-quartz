#!/usr/bin/env python3
"""Tests for the cross-device verification report and pinned fixtures."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import compare_phone


class CrossDeviceTests(unittest.TestCase):
    def test_pinned_manifests_match_current_fixtures(self):
        m=json.loads((HERE/"manifest.json").read_text())
        names={
            "train":ROOT/"experiments/android-lora-smoke/multi-3-samples.jsonl",
            "fresh":HERE/"fixtures/holdout_fresh.jsonl",
            "challenge":HERE/"fixtures/challenge_negation.jsonl",
            "unrelated":HERE/"fixtures/unrelated_regression.jsonl",
        }
        self.assertEqual(
            {name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in names.items()},
            m["fixtures"]
        )

    def example_reports(self):
        model=compare_phone.MODEL_SHA
        adapter=compare_phone.ADAPTER_SHA
        pc={
            "result":"PASS_WITH_WARNINGS",
            "files":{
                "model_sha256":model,"adapter_sha256":adapter,
                "fixture_sha256":{"fresh":"289724c68c514311966143efe6c244070dd52ee08cb9a746c6182fe87ebd041d"},
            },
            "metrics":{"fresh_eval":{"count":6,"improved_samples":5,
                "base_mean":0.6072431,"adapted_mean":0.3144017}},
        }
        phone={
            "status":"PASS",
            "model_sha256_before":model,"model_sha256_after":model,
            "adapter_sha256_before":adapter,"adapter_sha256_after":adapter,
            "samples":6,"improved_samples":5,
            "base_mean":"0.6072459","adapted_mean":"0.3144062",
            "thermal_before":0,"thermal_after":0,
            "battery_before":50,"battery_after":50,
        }
        return pc,phone

    def test_cross_device_consistency_rejects_drift_and_wrong_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            p=tmp/"pc.json";f=tmp/"phone.json";out=tmp/"out.json"
            pc,phone=self.example_reports()
            p.write_text(json.dumps(pc));f.write_text(json.dumps(phone))
            cmd=[sys.executable,str(HERE/"compare_phone.py"),"--pc",str(p),
                 "--phone",str(f),"--output",str(out)]
            r=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(json.loads(out.read_text())["result"],"PASS")
            phone["adapted_mean"]="0.415"
            f.write_text(json.dumps(phone))
            r=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(r.returncode,2)
            phone["adapted_mean"]="0.3144062"
            phone["adapter_sha256_after"]="0"*64
            f.write_text(json.dumps(phone))
            r=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(r.returncode,2)


if __name__=="__main__":
    unittest.main()
