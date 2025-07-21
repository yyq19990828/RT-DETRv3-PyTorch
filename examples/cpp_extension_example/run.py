import torch
import lltm

# Example usage
batch_size, features = 16, 32
X = torch.randn(batch_size, features, device='cuda')
h = torch.randn(batch_size, features, device='cuda')
C = torch.randn(batch_size, features, device='cuda')
W = torch.randn(3 * features, features, device='cuda')
b = torch.randn(3 * features, device='cuda')

# Forward pass
new_h, new_C = lltm.forward(X, W, b, h, C)

print("Forward pass successful!")
print("Output h shape:", new_h.shape)
print("Output C shape:", new_C.shape)
