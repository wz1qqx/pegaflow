#!/usr/bin/env python3
"""
Benchmark vLLM server using ShareGPT dataset.

Usage:
    python share_gpt_bench.py --model meta-llama/Llama-3.1-8B
    python share_gpt_bench.py --model Qwen/Qwen2.5-7B --num-prompts 200
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


# Default ShareGPT dataset URL
DEFAULT_DATASET_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def download_dataset(url: str, output_path: Path):
    """Download ShareGPT dataset if not exists."""
    if output_path.exists():
        print(f"Dataset already exists at: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ShareGPT dataset from {url}...")

    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
            with open(output_path, 'wb') as f:
                f.write(data)
        print(f"✓ Dataset downloaded to: {output_path}\n")
    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        print("\nYou can manually download the dataset and specify it with --dataset-path")
        sys.exit(1)


def run_benchmark(args, result_file: Path) -> dict:
    """Run vllm bench serve with ShareGPT dataset and return the results."""
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--host", args.host,
        "--port", str(args.port),
        "--model", args.model,
        "--dataset-name", "sharegpt",
        "--num-prompts", str(args.num_prompts),
        "--request-rate", str(args.request_rate),
        "--seed", str(args.seed),
        "--save-result",
        "--result-filename", result_file.name,
        "--result-dir", str(result_file.parent),
        "--label", args.label,
    ]

    # Add dataset path if provided
    if args.dataset_path:
        cmd.extend(["--dataset-path", args.dataset_path])

    print(f"Running ShareGPT benchmark: {args.label}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Benchmark failed with return code {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Benchmark failed: {args.label}")

    print(f"✓ Benchmark complete -> {result_file}")

    # Load and return the results
    with open(result_file, 'r') as f:
        return json.load(f)


def print_results(results: dict, args):
    """Print benchmark results summary."""
    print("\n" + "=" * 80)
    print("SHAREGPT BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        "Model: {model} | Prompts: {num_prompts} | Rate: {request_rate} req/s | Seed: {seed}".format(
            model=args.model,
            num_prompts=args.num_prompts,
            request_rate=args.request_rate,
            seed=args.seed,
        )
    )

    metrics = [
        ("mean_ttft_ms", "TTFT mean (ms)"),
        ("p99_ttft_ms", "TTFT p99 (ms)"),
        ("request_throughput", "Request throughput (req/s)"),
        ("output_throughput", "Output throughput (tok/s)"),
        ("duration", "Duration (s)"),
    ]

    print("-" * 80)
    for key, label in metrics:
        value = results.get(key)
        if value is not None:
            if isinstance(value, float):
                formatted = f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
            else:
                formatted = str(value)
            print(f"{label:<35} {formatted:>15}")

    print("-" * 80)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM server with ShareGPT dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to ShareGPT dataset JSON file (if not provided, will auto-download to bench_results/)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to test (default: 100)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=10.0,
        help="Request rate in requests/sec (default: 10.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="sharegpt",
        help="Label for this benchmark run (default: sharegpt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/bench_results",
        help="Directory to save benchmark results (default: examples/bench_results)",
    )

    args = parser.parse_args()

    # Create output directory with timestamp
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"sharegpt_bench_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Handle dataset download
    if not args.dataset_path:
        dataset_path = output_dir / "sharegpt_dataset.json"
        download_dataset(DEFAULT_DATASET_URL, dataset_path)
        args.dataset_path = str(dataset_path)

    print("\n" + "=" * 70)
    print("SHAREGPT BENCHMARK")
    print("=" * 70)
    print(f"Model:           {args.model}")
    print(f"Server:          http://{args.host}:{args.port}")
    print(f"Num Prompts:     {args.num_prompts}")
    print(f"Request Rate:    {args.request_rate} req/s")
    print(f"Random Seed:     {args.seed}")
    print(f"Dataset Path:    {args.dataset_path}")
    print(f"Results Dir:     {run_dir}")
    print("=" * 70 + "\n")

    try:
        # Run benchmark
        result_file = run_dir / f"{args.label}.json"
        results = run_benchmark(args, result_file)

        # Print summary
        print_results(results, args)

        print(f"\n✓ Results saved to: {run_dir}")
        print(f"  Result file: {result_file}\n")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
