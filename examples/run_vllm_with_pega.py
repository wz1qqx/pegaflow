#!/usr/bin/env python3
"""
Start a vLLM server with PegaFlow KV cache connector.

Usage:
    python run_vllm_with_pega.py --model <model_path>
    python run_vllm_with_pega.py --model meta-llama/Llama-3.1-8B --port 8000
"""
import argparse
import json
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start vLLM server with PegaFlow KV cache connector"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure PegaFlow KV connector
    kv_transfer_config = {
        "kv_connector": "PegaKVConnector",
        "kv_role": "kv_both",  # Both scheduler and worker roles
        "kv_connector_module_path": "pegaflow.connector",
    }

    # Build vllm serve command
    cmd = [
        "vllm", "serve",
        args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--kv-transfer-config", json.dumps(kv_transfer_config),
    ]

    print(f"Starting vLLM server with PegaFlow...")
    print(f"Model: {args.model}")
    print(f"Endpoint: http://{args.host}:{args.port}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"\nCommand: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
