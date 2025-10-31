#!/usr/bin/env python3
"""
Script to find optimal batch size for H100 training by gradually increasing
batch size until we hit memory limits.
"""

import json
import torch
import subprocess
import sys
from pathlib import Path

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        }
    return None

def test_batch_size(config_path, batch_size, max_length=512):
    """Test a specific batch size configuration."""
    print(f"\nTesting batch_size={batch_size}, max_length={max_length}")
    
    # Load base config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update batch size and sequence length
    config['training']['batch_size'] = batch_size
    config['data']['max_length'] = max_length
    config['training']['max_steps'] = 5  # Just test a few steps
    
    # Save test config
    test_config_path = 'configs/test_batch_size.json'
    with open(test_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Run training for a few steps
        result = subprocess.run([
            sys.executable, 'train.py', 
            '--config', test_config_path,
            '--quiet'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            memory_info = get_gpu_memory()
            if memory_info:
                print(f"✅ SUCCESS - Memory used: {memory_info['allocated']:.1f}GB / {memory_info['max_memory']:.1f}GB")
                return True, memory_info['allocated']
            return True, 0
        else:
            print(f"❌ FAILED - {result.stderr.split('RuntimeError:')[-1].strip() if 'RuntimeError:' in result.stderr else 'Unknown error'}")
            return False, 0
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT")
        return False, 0
    except Exception as e:
        print(f"❌ ERROR - {e}")
        return False, 0

def find_optimal_config():
    """Find optimal batch size and sequence length."""
    base_config = 'configs/h100_no_compile.json'
    
    print("🔍 Finding optimal H100 configuration...")
    print(f"GPU Memory Available: {get_gpu_memory()['max_memory']:.1f}GB")
    
    # Test different configurations
    configs_to_test = [
        # (batch_size, max_length)
        (8, 512),
        (16, 512),
        (32, 512),
        (48, 512),
        (64, 512),
        (96, 512),
        (128, 512),
        (32, 1024),
        (48, 1024),
        (64, 1024),
        (96, 1024),
        (128, 1024),
        (64, 2048),
        (96, 2048),
        (128, 2048),
    ]
    
    successful_configs = []
    
    for batch_size, max_length in configs_to_test:
        success, memory_used = test_batch_size(base_config, batch_size, max_length)
        if success:
            successful_configs.append({
                'batch_size': batch_size,
                'max_length': max_length,
                'memory_used': memory_used,
                'throughput_estimate': batch_size * max_length
            })
        else:
            # If we hit memory limit, stop testing larger configs with same max_length
            if any(c['max_length'] == max_length for c in successful_configs):
                continue
    
    if successful_configs:
        # Find best config (highest throughput that uses <90% memory)
        max_memory = get_gpu_memory()['max_memory']
        safe_configs = [c for c in successful_configs if c['memory_used'] < max_memory * 0.9]
        
        if safe_configs:
            best_config = max(safe_configs, key=lambda x: x['throughput_estimate'])
            print(f"\n🎯 RECOMMENDED CONFIG:")
            print(f"   batch_size: {best_config['batch_size']}")
            print(f"   max_length: {best_config['max_length']}")
            print(f"   Memory usage: {best_config['memory_used']:.1f}GB ({best_config['memory_used']/max_memory*100:.1f}%)")
            print(f"   Estimated throughput: {best_config['throughput_estimate']} tokens/batch")
            
            # Create optimized config
            with open('configs/h100_no_compile.json', 'r') as f:
                config = json.load(f)
            
            config['training']['batch_size'] = best_config['batch_size']
            config['data']['max_length'] = best_config['max_length']
            config['training']['mixed_precision'] = 'bf16'  # Enable bf16 for better performance
            config['hardware']['dataloader_workers'] = 8
            config['hardware']['pin_memory'] = True
            
            with open('configs/h100_optimized_auto.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\n💾 Saved optimized config to: configs/h100_optimized_auto.json")
        else:
            print("\n⚠️  All configs use >90% memory. Consider using gradient accumulation.")
    else:
        print("\n❌ No successful configurations found.")
    
    # Cleanup
    Path('configs/test_batch_size.json').unlink(missing_ok=True)

if __name__ == '__main__':
    find_optimal_config()