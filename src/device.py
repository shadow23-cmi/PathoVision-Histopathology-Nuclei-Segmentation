import torch
import torch_directml


def get_device():
    """
    Return the DirectML device that successfully executed
    the user's GPU benchmark.
    """
    return torch_directml.device(0)


def print_device_info():
    print("PyTorch version:", torch.__version__)
    print("DirectML device count:", torch_directml.device_count())

    for i in range(torch_directml.device_count()):
        print(f"Device {i}:", torch_directml.device(i))

    device = get_device()
    print("Selected device:", device)


if __name__ == "__main__":
    print_device_info()