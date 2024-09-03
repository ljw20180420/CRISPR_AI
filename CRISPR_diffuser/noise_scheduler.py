from config import args, device
import torch

def linear_noise_scheduler(t: torch.Tensor):
    t = t.to(device)
    return (
        args.noise_timesteps / (args.noise_timesteps - t).maximum(torch.tensor(torch.finfo(torch.float32).tiny, device=device))
    ).log()

def cosine_noise_scheduler(t: torch.Tensor):
    t = t.to(device)
    return (
        torch.cos(torch.tensor(args.cosine_factor / (1 + args.cosine_factor) * torch.pi / 2, device=device)) /
        torch.cos((t / args.noise_timesteps + args.cosine_factor) / (1 + args.cosine_factor) * torch.pi / 2).maximum(torch.tensor(torch.finfo(torch.float32).tiny, device=device))
    ).log()

def exp_noise_scheduler(t: torch.Tensor):
    t = t.to(device)
    return args.exp_scale * (args.exp_base ** (t / args.noise_timesteps) - 1)

def uniform_noise_scheduler(t: torch.Tensor):
    t = t.to(device)
    return args.uniform_scale * t / args.noise_timesteps

noise_scheduler = linear_noise_scheduler if args.noise_scheduler == "linear" else cosine_noise_scheduler if args.noise_scheduler == "cosine" else exp_noise_scheduler if args.noise_scheduler == "uniform" else uniform_noise_scheduler
    