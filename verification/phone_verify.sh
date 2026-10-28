#!/system/bin/sh
# Quartz S26 native evidence collector: read-only, no install/network/training.
set -eu
ROOT=/sdcard/Download/quartz-lora
if [ "$#" -ge 1 ]; then BIN=$1; else BIN=/data/local/tmp/quartz-lora/quartz-multi8; fi
if [ "$#" -ge 2 ]; then MODEL=$2; else MODEL="$ROOT/qwen05.gguf"; fi
if [ "$#" -ge 3 ]; then ADAPTER=$3; else ADAPTER="$ROOT/adapter-qwen25-qv16-23-r2-step4-s26.safetensors"; fi
if [ "$#" -ge 4 ]; then FRESH=$4; else FRESH="$ROOT/holdout_fresh.jsonl"; fi
MODEL_SHA=6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653
ADAPTER_SHA=adfb838b23efefadf9423df0d3b6574920b378173c6ef026a31b5ec4a56c710d
FRESH_SHA=289724c68c514311966143efe6c244070dd52ee08cb9a746c6182fe87ebd041d
BIN_SHA=1cf7855d19e17df14b874debad00c2be7cb99aa62204b595daee4459e395a260

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
report="$ROOT/verification-s26-$timestamp.json"
log="$ROOT/verification-s26-$timestamp.log"
status=FAIL
reason=not_started
model_before=
model_after=
adapter_before=
adapter_after=
battery_start=0
battery_end=0
thermal_start=999
thermal_end=999
available_kb=0
base_mean=
adapted_mean=
samples=0
improved=0
write_report() {
    printf '{"schema_version":1,"device":"SM-S948B","mode":"independent_eval","timestamp_utc":"%s","status":"%s","reason":"%s","model_sha256_before":"%s","model_sha256_after":"%s","adapter_sha256_before":"%s","adapter_sha256_after":"%s","samples":%s,"improved_samples":%s,"base_mean":"%s","adapted_mean":"%s","battery_before":%s,"battery_after":%s,"thermal_before":%s,"thermal_after":%s,"available_kb":%s}\n' \
        "$timestamp" "$status" "$reason" "$model_before" "$model_after" "$adapter_before" "$adapter_after" \
        "$samples" "$improved" "$base_mean" "$adapted_mean" "$battery_start" "$battery_end" "$thermal_start" "$thermal_end" "$available_kb" > "$report"
    echo "PHONE_VERIFICATION_RESULT=$status"
    echo "PHONE_VERIFICATION_REPORT=$report"
    echo "PHONE_VERIFICATION_LOG=$log"
}
trap write_report EXIT
if [ ! -x "$BIN" ] || [ ! -r "$MODEL" ] || [ ! -r "$ADAPTER" ] || [ ! -r "$FRESH" ]; then
    reason=missing_input; exit 2
fi
reason=hash_mismatch
printf '%s  %s\n' "$MODEL_SHA" "$MODEL" | sha256sum -c - >>"$log" 2>&1
printf '%s  %s\n' "$ADAPTER_SHA" "$ADAPTER" | sha256sum -c - >>"$log" 2>&1
printf '%s  %s\n' "$FRESH_SHA" "$FRESH" | sha256sum -c - >>"$log" 2>&1
printf '%s  %s\n' "$BIN_SHA" "$BIN" | sha256sum -c - >>"$log" 2>&1
model_before=$MODEL_SHA
adapter_before=$ADAPTER_SHA
available_kb=$(awk '/^MemAvailable:/ {print $2;exit}' /proc/meminfo)
battery_start=$(dumpsys battery | sed -n 's/^[[:space:]]*level: \([0-9]*\).*/\1/p' | head -n 1)
thermal_start=$(dumpsys thermalservice | sed -n 's/^Thermal Status: \([0-9]*\).*/\1/p' | head -n 1)
if [ -z "$battery_start" ]; then battery_start=0; fi
if [ -z "$thermal_start" ]; then thermal_start=999; fi
if [ "$available_kb" -lt 1048576 ] || [ "$battery_start" -lt 20 ] || [ "$thermal_start" -gt 2 ]; then
    reason=unsafe_device_conditions; exit 3
fi

reason=native_evaluation_failed
echo "VERIFICATION_START $timestamp" >> "$log"
RAYON_NUM_THREADS=2 nice -n 10 "$BIN" eval-qv-suffix \
    "$MODEL" "$FRESH" "$ADAPTER" --layers 8 --max-tokens 16 >> "$log" 2>&1
summary=$(grep '^MULTI_EVAL_RESULT: ' "$log" | tail -n 1)
samples=$(grep -c '^MULTI_EVAL_SAMPLE: ' "$log" || :)
case "$summary" in
    'MULTI_EVAL_RESULT: IMPROVED unseen=6 layers=8 '* ) : ;;
    *) reason=bad_summary; exit 4 ;;
esac
if [ "$samples" -ne 6 ]; then reason=missing_samples; exit 4; fi
counts=$(awk '
/^MULTI_EVAL_SAMPLE:/ {
 split($3,orig,"="); split($4,adapt,"=");
 n++;
 if ((adapt[2]+0) < (orig[2]+0)) good++;
}
END{printf "%d %d\n",n,good}
' "$log")
n=$(echo "$counts" | awk '{print $1}')
improved=$(echo "$counts" | awk '{print $2}')
if [ "$n" -ne 6 ] || [ "$improved" -lt 5 ]; then
    reason=insufficient_per_sample_improvement; exit 4
fi
base_mean=$(echo "$summary" | sed -n 's/.* base=\([0-9.]*\) adapted=.*/\1/p')
adapted_mean=$(echo "$summary" | sed -n 's/.* adapted=\([0-9.]*\) total_ms=.*/\1/p')
if [ -z "$base_mean" ] || [ -z "$adapted_mean" ]; then reason=invalid_ce_values; exit 4; fi
if ! awk -v b="$base_mean" -v a="$adapted_mean" 'BEGIN{exit !(a < b && a >= 0 && b > 0)}'; then
    reason=loss_did_not_improve; exit 4
fi

reason=postrun_integrity
model_after=$(sha256sum "$MODEL" | awk '{print $1}')
adapter_after=$(sha256sum "$ADAPTER" | awk '{print $1}')
battery_end=$(dumpsys battery | sed -n 's/^[[:space:]]*level: \([0-9]*\).*/\1/p' | head -n 1)
thermal_end=$(dumpsys thermalservice | sed -n 's/^Thermal Status: \([0-9]*\).*/\1/p' | head -n 1)
if [ -z "$battery_end" ]; then battery_end=0; fi
if [ -z "$thermal_end" ]; then thermal_end=999; fi
if [ "$model_after" != "$MODEL_SHA" ] || [ "$adapter_after" != "$ADAPTER_SHA" ]; then exit 5; fi
if [ "$thermal_end" -gt 2 ]; then reason=thermal_warning; exit 5; fi
status=PASS
reason=verified_independent_native_evaluation
