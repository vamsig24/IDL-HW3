import torch
from torch import nn
import math

'''
TODO: Implement these Modules.

This file contains two ways of giving a transformer positional information:

1. PositionalEncoding: Fixed sinusoidal encoding added to the token embeddings
   - Added once, before the first layer
   - Encodes absolute position

2. RotaryPositionalEncoding: Rotates queries and keys inside each attention operation
   - Applied per head, after the q/k projections and the reshape into heads
   - Encodes relative position

Specification for RotaryPositionalEncoding:
- Applied to queries and keys only, never to values
- Pairs dimension i with dimension i + head_dim//2 (the "split-half" convention
  used by Llama and HuggingFace), not with dimension i+1
- The angle for position t and pair i is t / (base ** (2i / head_dim))
- cos and sin are registered as buffers of shape (1, 1, max_len, head_dim),
  so that they broadcast over batch and head
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """     
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """
        # TODO: Implement create_pe_table
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        # Register as buffer to save with model state
        self.register_buffer('pe', pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length 
        """
        # TODO: Implement forward
        # Step 1: Get sequence length from input tensor
        seq_len = x.size(1)
        # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")
        # Step 3: Add positional encodings to input
        return x + self.pe[:, :seq_len]

## -------------------------------------------------------------------------------------------------

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE).
    Rotates queries and keys so that attention scores depend on relative position.
    """
    def __init__(self, head_dim, max_len, base=10000.0):
        """
        Initialize the RotaryPositionalEncoding.
        Args:
            head_dim (int): The dimension of each attention head. Must be even.
            max_len  (int): The maximum length of the input sequence.
            base   (float): The base of the geometric progression of wavelengths.
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, but got {head_dim}")
        self.head_dim = head_dim
        self.create_rope_table(head_dim, max_len, base)

    def create_rope_table(self, head_dim, max_len, base):
        """
        Create the rotation tables.
        Args:
            head_dim (int): The dimension of each attention head.
            max_len  (int): The maximum length of the input sequence.
            base   (float): The base of the geometric progression of wavelengths.
        """
        # TODO: Implement create_rope_table
        # Step 1: Compute the inverse frequencies for the head_dim//2 pairs
        # Step 2: Take the outer product with the positions to get the angles
        # Step 3: Widen the angles to head_dim so each pair shares an angle, then take cos and sin
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        positions = torch.arange(max_len, dtype=torch.float)
        angles = torch.outer(positions, inv_freq)       # (max_len, head_dim//2)
        angles = torch.cat([angles, angles], dim=-1)    # (max_len, head_dim)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        # Register as buffers to save with model state
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the two halves of the last dimension into each other.
        Given x = [x_0 ... x_{d/2-1}, x_{d/2} ... x_{d-1}], return
        [-x_{d/2} ... -x_{d-1}, x_0 ... x_{d/2-1}].
        Args:
            x (torch.Tensor): Input tensor, shape (..., head_dim)
        Returns:
            torch.Tensor: Rotated tensor of the same shape
        """
        # TODO: Implement rotate_half
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """
        Forward pass for the RotaryPositionalEncoding.
        Args:
            q (torch.Tensor): Queries, shape (batch_size, num_heads, seq_len, head_dim)
            k (torch.Tensor): Keys, shape (batch_size, num_heads, seq_len, head_dim)
            offset     (int): Index of the first position. Defaults to 0.
        Returns:
            q (torch.Tensor): Rotated queries of the same shape
            k (torch.Tensor): Rotated keys of the same shape
        Errors:
            - ValueError: If sequence length exceeds maximum length
        """
        # TODO: Implement forward
        # Step 1: Get sequence length from the query tensor
        seq_len = q.size(2)
        # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
        if offset + seq_len > self.cos.size(2):
            raise ValueError(f"Sequence length {offset + seq_len} exceeds the maximum length {self.cos.size(2)}")
        # Step 3: Slice the cos and sin tables to the positions in use
        cos = self.cos[:, :, offset:offset + seq_len].to(q.dtype)
        sin = self.sin[:, :, offset:offset + seq_len].to(q.dtype)
        # Step 4: Rotate q and k, where rotating x is  x * cos + self.rotate_half(x) * sin
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        return q, k
