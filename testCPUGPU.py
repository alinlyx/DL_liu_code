import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.tensor([1.0, 2.0, 3.0])
    x = x.to(device)
    y = torch.tensor([4.0, 5.0, 6.0])
    y = y.to(device)
    z = x + y
    print(z)
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU")
