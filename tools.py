import torch

def get_memory_usage():
    gpu_usage = []
    id = 0
    while True:
        try:
            free, total = torch.cuda.mem_get_info(id)
            gpu_usage.append((total - free, total))
            id += 1
        except:
            break
    return gpu_usage
