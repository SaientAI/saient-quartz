#!/usr/bin/env python3
"""Compare PC and S26 independent Quartz verification reports.

Not cryptographic device attestation: this verifies internal consistency,
artifact hashes and numerical reproducibility across the two trusted local
runtimes, not an unforgeable identity statement about the phone.
"""
import argparse
import json
from pathlib import Path
import sys

MODEL_SHA = "6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653"
ADAPTER_SHA = "adfb838b23efefadf9423df0d3b6574920b378173c6ef026a31b5ec4a56c710d"
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pc",type=Path,required=True)
    p.add_argument("--phone",type=Path,required=True)
    p.add_argument("--tolerance",type=float,default=0.0001)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    try:
        pc=json.loads(a.pc.read_text())
        phone=json.loads(a.phone.read_text())
        metrics=pc["metrics"]["fresh_eval"]
        checks={}
        checks["pc_report_complete"]=pc["result"] in ("PASS","PASS_WITH_WARNINGS")
        checks["s26_report_complete"]=phone.get("status")=="PASS"
        checks["model_and_adapter_identity"]=(
            pc["files"]["model_sha256"]==MODEL_SHA
            and pc["files"]["adapter_sha256"]==ADAPTER_SHA
            and phone.get("model_sha256_before")==MODEL_SHA
            and phone.get("model_sha256_after")==MODEL_SHA
            and phone.get("adapter_sha256_before")==ADAPTER_SHA
            and phone.get("adapter_sha256_after")==ADAPTER_SHA
        )
        checks["same_six_examples_and_improvement_count"]=(
            metrics["count"]==6
            and phone.get("samples")==6
            and metrics["improved_samples"]==int(phone.get("improved_samples",-1))
            and pc["files"]["fixture_sha256"]["fresh"]==
                "289724c68c514311966143efe6c244070dd52ee08cb9a746c6182fe87ebd041d"
        )
        base_drift=abs(metrics["base_mean"]-float(phone["base_mean"]))
        adapted_drift=abs(metrics["adapted_mean"]-float(phone["adapted_mean"]))
        checks["cross_device_loss_parity"]=(
            base_drift<=a.tolerance and adapted_drift<=a.tolerance
        )
        checks["s26_thermal_and_battery"]=(
            int(phone.get("thermal_before",999))<=2
            and int(phone.get("thermal_after",999))<=2
            and int(phone.get("battery_before",0))>=20
            and int(phone.get("battery_after",0))>=15
        )
        result={
            "schema_version":1,
            "result":"PASS" if all(checks.values()) else "FAIL",
            "checks":checks,
            "base_loss_absolute_difference":round(base_drift,9),
            "adapted_loss_absolute_difference":round(adapted_drift,9),
            "tolerance":a.tolerance,
            "pc_report_path":str(a.pc),
            "s26_report_path":str(a.phone),
            "limitations":"Local two-runtime consistency, not authenticated hardware attestation",
        }
        out=a.output or a.pc.parent/"cross_device.json"
        out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(json.dumps(result,indent=2,sort_keys=True))
        return 0 if result["result"]=="PASS" else 2
    except (ValueError,TypeError,KeyError,OSError) as e:
        print("VERIFICATION_COMPARISON_ERROR:",e,file=sys.stderr)
        return 2

if __name__=="__main__":
    sys.exit(main())
