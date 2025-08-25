#!/usr/bin/env python3
"""
GPU memory cleanup utility
"""

import torch
import gc
import os

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        print("Cleaning up GPU memory...")
        
        # Clear Python garbage collection
        gc.collect()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Reset CUDA memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Show current memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory allocated: {memory_allocated:.2f} GB")
            print(f"GPU Memory reserved: {memory_reserved:.2f} GB")
            
        print("GPU memory cleanup completed!")
    else:
        print("No CUDA device available")

def kill_python_processes():
    """Kill all Python processes (nuclear option)"""
    print("WARNING: This will kill all Python processes!")
    choice = input("Continue? (y/N): ").lower()
    if choice == 'y':
        os.system("pkill -f python")
        print("All Python processes killed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU Memory Cleanup")
    parser.add_argument("--nuclear", action="store_true", help="Kill all Python processes")
    args = parser.parse_args()
    
    if args.nuclear:
        kill_python_processes()
    else:
        cleanup_gpu_memory()