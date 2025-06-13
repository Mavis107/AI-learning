from torch import nn
import torch

from ..network import MultiHeadAttention
from ..network import FeedForwardNeuralNetwork
from ..util.transformer_util import create_causal_mask

class DecoderBlock(nn.Module):

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

        # Initialize the Decoder Block 
        # Causal Multi-Head Self-Attention layer 
        self.causal_multi_head = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        # Layer Normalization 
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        # Multi-Head Cross-Attention layer         
        self.cross_multi_head = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        # Layer Normalization 
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        # Feed forward neural network layer        
        self.ffn = FeedForwardNeuralNetwork(d_model, d_ff)
        # Layer Normalization 
        self.layer_norm3 = torch.nn.LayerNorm(d_model)


    def forward(self,
                inputs: torch.Tensor,
                context: torch.Tensor,
                causal_mask: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs from the Decoder
            context: Context from the Encoder
            causal_mask: Mask used for Causal Self Attention
            pad_mask: Optional Padding Mask used for Cross Attention

        Shape: 
            - inputs: (batch_size, sequence_length_decoder, d_model)
            - context: (batch_size, sequence_length_encoder, d_model)
            - causal_mask: (batch_size, sequence_length_decoder, sequence_length_decoder)
            - pad_mask: (batch_size, sequence_length_decoder, sequence_length_encoder)
            - outputs: (batch_size, sequence_length_decoder, d_model)
        """

        # Implement the forward pass of the decoder block            
        outputs = self.causal_multi_head(inputs, inputs, inputs, causal_mask)+ inputs
        # The residual connections
        outputs = self.layer_norm1(outputs)
        # intermediate_outputs = outputs
        # Pass on the padding mask 
        outputs = self.cross_multi_head(outputs, context, context, pad_mask)+ outputs
        # The residual connections
        outputs = self.layer_norm2(outputs)
        intermediate_outputs = outputs
        outputs = self.ffn(outputs)
        # The residual connections
        outputs = self.layer_norm3(outputs + intermediate_outputs)

        return outputs