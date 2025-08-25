import torch, gc


def cleanup_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # make sure all CUDA ops are finished

    free_mem, total_mem = torch.cuda.mem_get_info()
    print(
        f"VRAM freed ✅ | Free: {free_mem/1024**3:.2f} GB / Total: {total_mem/1024**3:.2f} GB"
    )


cleanup_vram()
