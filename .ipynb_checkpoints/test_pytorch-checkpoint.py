import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name())

import torch

x = torch.cuda.FloatTensor([1.0, 2.0, 3.0])
print(x)
print("Tensor device:", x.device)
