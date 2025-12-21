#!/bin/bash
# Train all models for steering awareness experiments
# Designed for 8x A100-40GB setup

set -e

# Load environment
source .env

OUTPUT_BASE="./outputs"
mkdir -p "$OUTPUT_BASE"

# Log file
LOG_FILE="$OUTPUT_BASE/training_$(date +%Y%m%d_%H%M%S).log"
echo "Training log: $LOG_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

train_model() {
    local shortname=$1
    local gpu=$2

    log "Starting training: $shortname on GPU $gpu"

    CUDA_VISIBLE_DEVICES=$gpu python experiments/run_training.py \
        --model "$shortname" \
        --output "$OUTPUT_BASE" \
        --epochs 4 \
        --learning-rate 1e-4 \
        --batch-size 4 \
        2>&1 | tee -a "$LOG_FILE"

    log "Completed: $shortname"
}

log "=== STEERING AWARENESS TRAINING ==="
log "GPUs: 8x A100-40GB"
log "Models: llama-8b, llama-70b, deepseek, gemma, qwen-7b, qwen-32b, gpt-oss"

# Phase 1: Smaller models in parallel (each on 1 GPU)
log "=== Phase 1: Smaller Models (parallel) ==="

# Run 6 smaller models in parallel across GPUs 0-5
train_model "llama-8b" 0 &
train_model "deepseek" 1 &
train_model "gemma" 2 &
train_model "qwen-7b" 3 &
train_model "gpt-oss" 4 &

# Wait for smaller models to complete
wait
log "Phase 1 complete"

# Phase 2: Larger models (need more VRAM)
log "=== Phase 2: Larger Models ==="

# Qwen 32B with 4-bit quantization (1 GPU should work)
train_model "qwen-32b" 0

# Llama 70B with 4-bit quantization (uses device_map=auto)
log "Starting Llama 70B (4-bit quantized, multi-GPU)"
python experiments/run_training.py \
    --model llama-70b \
    --output "$OUTPUT_BASE" \
    --epochs 4 \
    --learning-rate 1e-4 \
    --batch-size 4 \
    2>&1 | tee -a "$LOG_FILE"

log "=== ALL TRAINING COMPLETE ==="
log "Outputs saved to: $OUTPUT_BASE"

# List completed adapters
echo ""
echo "Trained adapters:"
find "$OUTPUT_BASE" -name "adapter_config.json" -exec dirname {} \;
