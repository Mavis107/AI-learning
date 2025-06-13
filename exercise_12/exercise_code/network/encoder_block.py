from torch import nn
import torch
from ..network import MultiHeadAttention
from ..network import FeedForwardNeuralNetwork

class EncoderBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()

        # Initialize the Encoder Block 
        # Multi-Head Self-Attention layer
        self.multi_head = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        # Layer Normalization 
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        # Feed forward neural network layer 
        self.ffn = FeedForwardNeuralNetwork(d_model, d_ff)
        # Layer Normalization
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        


    def forward(self,
                inputs: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs to the Encoder Block
            pad_mask: Optional Padding Mask

        Shape:
            - inputs: (batch_size, sequence_length, d_model)
            - pad_mask: (batch_size, sequence_length, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """
        
        # Implement the forward pass of the encoder block 
        # Pass on the padding mask    
        outputs = self.multi_head(inputs, inputs, inputs, pad_mask)+inputs
        # Add + Norm
        outputs = self.layer_norm1(outputs)
        intermediate_outputs = outputs
        outputs = self.ffn(outputs)
        # Add + Norm
        outputs = self.layer_norm2(outputs + intermediate_outputs)

        return outputs