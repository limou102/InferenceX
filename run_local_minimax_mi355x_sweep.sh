#!/usr/bin/env bash
# =============================================================================
# Sweep over environment variable combinations and run benchmarks.
# For each parameter combination, this script invokes run_local_minimax_mi355x.sh,
# which in turn launches benchmarks/single_node/minimaxm2.5_fp8_mi355x.sh to
# start a fresh vLLM server, run the client load test, then tear everything down.
#
# Usage:
#   enter docker container of image vllm/vllm-openai-rocm:v0.19.0
#   bash run_local_minimax_mi355x_sweep.sh
# =============================================================================
set -x
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/sweep-logs}"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

# Server arguments
EP_ENABLEDS=(1)        # 0: disable expert parallel; 1: enable expert parallel
ISLS=(1024 8192)
OSLS=(1024)
TPS=(2 4)

# Client arguments
CONCS=(32 64 128 256)

kill_vllm() {
    ps xu | grep -i "vllm"       | grep -v "grep" | grep -v "kill_vllm_server" | awk '{print $2}' | xargs -r kill -9 1>/dev/null 2>&1 || true
    ps xu | grep -i "kimi"       | grep -v "grep" | awk '{print $2}' | xargs -r kill -9 1>/dev/null 2>&1 || true
    ps xu | grep -i "import main"| grep "python"  | awk '{print $2}' | xargs -r kill -9 1>/dev/null 2>&1 || true
    ps xu | grep "tail"          | grep "\-n"     | awk '{print $2}' | xargs -r kill -9 1>/dev/null 2>&1 || true
    sleep 3
}

# ---------------------------------------------------------------------------
# Enumerate & run
# ---------------------------------------------------------------------------
total=$(( ${#EP_ENABLEDS[@]} * ${#ISLS[@]} * ${#OSLS[@]} * ${#TPS[@]} * ${#CONCS[@]} ))
echo "[sweep] Total benchmark runs: $total"

run_idx=0
for ep_en in "${EP_ENABLEDS[@]}"; do
for isl  in "${ISLS[@]}";        do
for osl  in "${OSLS[@]}";        do
for tp   in "${TPS[@]}";         do
for conc in "${CONCS[@]}";       do
    run_idx=$((run_idx + 1))
    tag="isl${isl}_osl${osl}_tp${tp}_ep${ep_en}_conc${conc}"
    log_file="${LOG_DIR}/${tag}.log"
    server_log_file="${LOG_DIR}/server_${tag}.log"

    echo ""
    echo "======================================================="
    echo " [${run_idx}/${total}] ${tag}"
    echo "======================================================="
    echo "  ISL=${isl}  OSL=${osl}"
    echo "  TP=${tp}  EP_ENABLED=${ep_en}  CONC=${conc}"
    echo "  Client/driver Log: ${log_file}"
    echo "  Server Log       : ${server_log_file}"
    echo "======================================================="

    kill_vllm
    # Wipe any stale server.log left by a previous run so we don't misattribute it.
    rm -f /workspace/server.log

    export ISL="$isl"
    export OSL="$osl"
    export TP_SIZE="$tp"
    export EP_ENABLED="$ep_en"
    export CONC="$conc"
    export RUN_EVAL=false

    if bash "${SCRIPT_DIR}/run_local_minimax_mi355x.sh" > "$log_file" 2>&1; then
        echo "  [${run_idx}/${total}] PASSED"
    else
        echo "  [${run_idx}/${total}] FAILED  (see ${log_file})"
    fi

    # Archive vLLM server log for this case before the next iteration overwrites it.
    if [ -f /workspace/server.log ]; then
        cp -f /workspace/server.log "$server_log_file" \
            || echo "  [${run_idx}/${total}] WARN: failed to copy server.log"
    else
        echo "  [${run_idx}/${total}] WARN: /workspace/server.log not found"
    fi

    kill_vllm
    rm -f "${SCRIPT_DIR}"/gpucore.* "${SCRIPT_DIR}"/core.* 2>/dev/null || true
    sleep 3
done
done
done
done
done

echo ""
echo "======================================================="
echo " Sweep complete. ${run_idx}/${total} benchmark runs executed."
echo " Logs directory: ${LOG_DIR}"
echo "======================================================="
ls -lh "$LOG_DIR"/*.log || true
