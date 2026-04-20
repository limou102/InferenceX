#!/usr/bin/env bash

ps xu | grep -i "vllm" | grep -v "grep" | grep -v "kill_vllm_server" | awk '{print $2}' | xargs -r kill -9 1>/dev/null 2>&1 || true
ps xu | grep -i "kimi" | grep -v "grep" | awk '{print $2}' | xargs -r kill -9 1>/dev/null 2>&1 || true
ps xu | grep -i "import main" | grep "python" | awk '{print $2}' | xargs -r kill -9 1>/dev/null 2>&1 || true
ps xu | grep "tail" | grep "\-n" | awk '{print $2}' | xargs -r kill -9 1>/dev/null 2>&1 || true

exit 0
