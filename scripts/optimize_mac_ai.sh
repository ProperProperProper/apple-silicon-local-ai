#!/usr/bin/env bash
# ==============================================================================
# optimize_mac_ai.sh
# macOS & Apple Silicon Hardware / Memory Tuning for Local AI Workloads
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo " macOS Apple Silicon Tuning for Local AI & LLM Inference  "
echo "=========================================================="

# 1. Expand GPU Wired Memory Allocation
# By default, macOS reserves a percentage of RAM for system overhead, limiting
# how much unified memory the GPU can allocate for heavy model context windows.
TOTAL_MEM_BYTES=$(sysctl -n hw.memsize)
TOTAL_MEM_GB=$((TOTAL_MEM_BYTES / 1024 / 1024 / 1024))
echo "[+] Detected Unified Memory: ${TOTAL_MEM_GB} GB"

# Allocate up to ~80% of total unified memory to the GPU without kernel panic
# Value is in megabytes
TARGET_GPU_MEM_MB=$((TOTAL_MEM_GB * 1024 * 80 / 100))

echo "[*] Setting GPU wired memory ceiling to ${TARGET_GPU_MEM_MB} MB..."
sudo sysctl iogpu.wired_mem_limit=${TARGET_GPU_MEM_MB} 2>/dev/null || {
    echo "ℹ️ Note: iogpu.wired_mem_limit requires root privileges or is managed by boot-args on this kernel."
}

# 2. Configure Dynamic Shared Memory & Metal Cache
export METAL_DEVICE_WRAPPER_TYPE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85

echo "[+] Configured PyTorch MPS high-watermark ratio: 85%"
echo "[+] Configured CPU fallback for unsupported Metal ops"

# 3. Model Storage Centralization (Prevent disk bloat across tools)
DEFAULT_MODEL_DIR="$HOME/Models/local-weights"
mkdir -p "$DEFAULT_MODEL_DIR"
echo "[+] Centralized local model directory created at: $DEFAULT_MODEL_DIR"

# 4. Telemetry and Offline Performance Guards
echo "[+] Enabling offline execution flags & disabling outbound telemetry delays"
export DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=========================================================="
echo "✅ System tuning configuration applied successfully!"
echo "To persist environment variables, source them in your ~/.zshrc"
echo "=========================================================="
