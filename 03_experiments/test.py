import torch

print("CUDA verfügbar:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Setze Threads für PyTorch Operationen
torch.set_num_threads(16)          # max Threads für Matrix-Operationen
torch.set_num_interop_threads(16)