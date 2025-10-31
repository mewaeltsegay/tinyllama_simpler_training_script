#!/usr/bin/env python3
"""Distributed training launcher for multi-GPU training."""

import os
import sys
import argparse
import subprocess
import socket
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.distributed import auto_detect_distributed_config, find_free_port


def find_free_port():
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def launch_distributed_training(args):
    """Launch distributed training with torch.distributed.launch."""
    
    # Auto-detect configuration if not specified
    if args.nproc_per_node is None:
        config = auto_detect_distributed_config()
        if config["can_use_distributed"]:
            args.nproc_per_node = config["num_gpus"]
            print(f"Auto-detected {args.nproc_per_node} GPUs for distributed training")
        else:
            print("Multi-GPU training not available, falling back to single GPU")
            args.nproc_per_node = 1
    
    # Find free port if not specified
    if args.master_port is None:
        args.master_port = find_free_port()
        print(f"Using port {args.master_port} for distributed training")
    
    # Build command for torch.distributed.launch
    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
    ]
    
    # Add additional torchrun arguments
    if args.nnodes > 1:
        cmd.extend([
            f"--nnodes={args.nnodes}",
            f"--node_rank={args.node_rank}"
        ])
    
    if args.use_env:
        cmd.append("--use_env")
    
    # Add training script and its arguments
    cmd.append(args.training_script)
    cmd.extend(args.script_args)
    
    print(f"Launching distributed training with command:")
    print(" ".join(cmd))
    print()
    
    # Set environment variables
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    
    # Launch training
    try:
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Distributed training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return 1


def main():
    """Main function for distributed training launcher."""
    parser = argparse.ArgumentParser(
        description="Launch distributed training for Tigrinya TinyLlama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect GPUs and launch distributed training
  python scripts/launch_distributed.py train.py --config configs/production_h100.json
  
  # Specify number of GPUs
  python scripts/launch_distributed.py --nproc_per_node 4 train.py --config configs/production_h100.json
  
  # Multi-node training (run on each node)
  python scripts/launch_distributed.py --nnodes 2 --node_rank 0 --master_addr 192.168.1.100 train.py --config configs/production_h100.json
        """
    )
    
    # Distributed training arguments
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=None,
        help="Number of processes per node (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes for multi-node training"
    )
    
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Rank of the current node"
    )
    
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Master node address"
    )
    
    parser.add_argument(
        "--master_port",
        type=int,
        default=None,
        help="Master node port (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--use_env",
        action="store_true",
        help="Use environment variables for distributed training"
    )
    
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="CUDA_VISIBLE_DEVICES setting"
    )
    
    # Training script and arguments
    parser.add_argument(
        "training_script",
        help="Path to training script (e.g., train.py)"
    )
    
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to training script"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.training_script):
        print(f"Error: Training script not found: {args.training_script}")
        return 1
    
    if args.nproc_per_node is not None and args.nproc_per_node < 1:
        print("Error: nproc_per_node must be >= 1")
        return 1
    
    if args.nnodes < 1:
        print("Error: nnodes must be >= 1")
        return 1
    
    if args.node_rank < 0 or args.node_rank >= args.nnodes:
        print(f"Error: node_rank must be between 0 and {args.nnodes - 1}")
        return 1
    
    # Show configuration
    print("Distributed Training Configuration:")
    print(f"  Training script: {args.training_script}")
    print(f"  Script arguments: {' '.join(args.script_args) if args.script_args else 'None'}")
    print(f"  Processes per node: {args.nproc_per_node or 'auto-detect'}")
    print(f"  Number of nodes: {args.nnodes}")
    print(f"  Node rank: {args.node_rank}")
    print(f"  Master address: {args.master_addr}")
    print(f"  Master port: {args.master_port or 'auto-detect'}")
    print(f"  CUDA_VISIBLE_DEVICES: {args.cuda_visible_devices or 'default'}")
    print()
    
    # Launch distributed training
    return launch_distributed_training(args)


if __name__ == "__main__":
    sys.exit(main())