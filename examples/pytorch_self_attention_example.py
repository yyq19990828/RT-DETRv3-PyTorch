import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value):
    """
    Computes scaled dot-product attention.

    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.

    Returns:
        torch.Tensor: The output of the attention mechanism.
        torch.Tensor: The attention weights.
    """
    # Matmul + scale
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1)**0.5)
    attn_weights = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights

# Example usage:
# (batch_size, seq_len, d_k)
query = torch.randn(2, 5, 16)
key = torch.randn(2, 5, 16)
value = torch.randn(2, 5, 32) # d_v can be different

output, attn_weights = scaled_dot_product_attention(query, key, value)
print("Output shape:", output.shape)
print("Attention weights shape:", attn_weights.shape)
