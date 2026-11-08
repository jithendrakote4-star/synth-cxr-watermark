import torch, pywt, diffusers, transformers
print("torch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("pywt ok:", pywt.__version__)