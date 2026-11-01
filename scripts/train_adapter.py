#!/usr/bin/env python3
"""
train_adapter.py
Local LoRA fine-tuning runner on Apple Silicon using Apple MLX.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapter on Apple Silicon")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-Instruct-4bit", help="Base model HuggingFace / MLX repo")
    parser.add_argument("--data", type=str, default="adapters/datasets/sample_domain.jsonl", help="Path to training data directory or jsonl")
    parser.add_argument("--output", type=str, default="adapters/weights/custom-adapter", help="Directory to save adapter weights")
    parser.add_argument("--iters", type=int, default=600, help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lora-layers", type=int, default=16, help="Number of layers to adapt")
    return parser.parse_args()

def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting local LoRA training on Apple Silicon...")
    print(f"[*] Base model: {args.model}")
    print(f"[*] Dataset:    {args.data}")
    print(f"[*] Output:     {args.output}")

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", args.model,
        "--train",
        "--data", str(Path(args.data).parent),
        "--adapter-path", str(output_path),
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--lora-layers", str(args.lora_layers)
    ]

    try:
        subprocess.run(cmd, check=True)
        print("[+] Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[-] Training encountered an error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
