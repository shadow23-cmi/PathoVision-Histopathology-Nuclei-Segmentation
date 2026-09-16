import torch
import torch_directml
import time

dim  = 4096*3
print("DirectML devices:", torch_directml.device_count())

for i in range(torch_directml.device_count()):
    device = torch_directml.device(i)

    print(f"\n=== Device {i} ===")
    print("Device:", device)

    try:
        x = torch.randn(dim, dim, device=device)
        y = torch.randn(dim, dim, device=device)

        # Warmup
        for _ in range(3):
            z = torch.matmul(x, y)

        start = time.perf_counter()
        z = torch.matmul(x, y)
        elapsed = time.perf_counter() - start

        print("Result device:", z.device)
        print("Result shape:", z.shape)
        print(f"Time: {elapsed:.4f}s")
        print("✓ DirectML operation succeeded")

    except Exception as e:
        print("✗ FAILED")
        print(repr(e))