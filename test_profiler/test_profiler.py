#!/usr/bin/env python

import os
import pathlib
import torch
import time
import shutil
import numpy as np

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

for direc in ["time", "space"]:
    if os.path.exists(direc):
        shutil.rmtree(direc)

prof = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=0, warmup=1, active=2, repeat=1, skip_first=0, skip_first_wait=0
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("time", use_gzip=True),
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
)

os.makedirs("space", exist_ok=True)

prof.start()
a1 = torch.zeros(1000000)
a2 = torch.zeros(1000000, device="cuda")
time.sleep(1)

prof.step()
b1 = torch.zeros(1000000)
b2 = torch.zeros(1000000, device="cuda")
time.sleep(1)

prof.step()
c1 = torch.zeros(1000000)
c2 = torch.zeros(1000000, device="cuda")
time.sleep(1)

prof.stop()

prof.export_memory_timeline("space/cpu.json", device="cpu")
prof.export_memory_timeline("space/cuda:0.json", device="cuda:0")


# prof.start()

# # warmup

# prof.step()

# with torch.profiler.record_function("active1"):
#     time.sleep(1)

# # prof.stop()

# # time.sleep(1)

# # prof.start()
# prof.step()

# with torch.profiler.record_function("active2"):
#     time.sleep(1)

# prof.step()

# # warmup

# prof.step()

# with torch.profiler.record_function("active1"):
#     time.sleep(1)

# prof.step()

# with torch.profiler.record_function("active2"):
#     time.sleep(1)

# prof.stop()
