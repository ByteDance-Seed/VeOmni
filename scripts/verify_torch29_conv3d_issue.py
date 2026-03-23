# Copied from https://github.com/pytorch/pytorch/issues/166643#issue-3571291598
# Check it to see if using the suggested cudnn version with torch 2.9 has fixed the conv3d memory issue
#
# Test results (torch 2.9.1+cu129) on A100:
#
# With nvidia-cudnn-cu12==9.15.1.9 (forced version):
#   Peak memory before: 1.2622 GB
#   Peak memory after:  1.6773 GB
#   Memory usage:       0.4151 GB
#   Time for 100 ops:   0.8583 s (0.0086 s/op)
#
# With nvidia-cudnn-cu12==9.10.2.21 (torch2.9.1+cu129 default version):
#   Peak memory before: 1.2622 GB
#   Peak memory after:  8.5054 GB
#   Memory usage:       7.2432 GB
#   Time for 100 ops:   2.3023 s (0.0230 s/op)
#
# Conclusion: cudnn 9.15.1.9 fixes the conv3d memory issue (~17x less memory, ~2.7x faster).

import time

import torch
import torch.nn.functional as F


print(f"PyTorch version: {torch.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.reset_peak_memory_stats(device)
torch.cuda.empty_cache()

input_shape = (1, 96, 6, 626, 626)
weight_shape = (96, 96, 3, 3, 3)
bias_shape = (96,)

input_tensor = torch.randn(input_shape, device=device).type(torch.bfloat16)
weight = torch.randn(weight_shape, device=device).type(torch.bfloat16)
bias = torch.randn(bias_shape, device=device).type(torch.bfloat16)

torch.cuda.synchronize()
mem_before_op_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
print(f"Peak memory allocated before operations: {mem_before_op_gb:.4f} GB")

torch.cuda.synchronize()
start_time = time.time()

for _ in range(100):
    conv3d_output = F.conv3d(
        input=input_tensor, weight=weight, bias=bias, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)
    )

torch.cuda.synchronize()
elapsed_time = time.time() - start_time

mem_after_op_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
print(f"Peak memory allocated after operations:  {mem_after_op_gb:.4f} GB")
print(f"Memory usage for all operations: {(mem_after_op_gb - mem_before_op_gb):.4f} GB")
print(f"Time for 100 conv3d operations: {elapsed_time:.4f} s")
print(f"Average time per conv3d operation: {elapsed_time / 100:.4f} s")
