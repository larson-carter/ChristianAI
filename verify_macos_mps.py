import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available and will be used.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA backend is available and will be used.")
else:
    device = torch.device("cpu")
    print("No GPU accelerators available. Using CPU.")
