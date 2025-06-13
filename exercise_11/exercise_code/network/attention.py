from torch import nn
import torch
from ..network import SCORE_SAVER

class ScaledDotAttention(nn.Module):

    def __init__(self,
                 d_k):
        """

        Args:
            d_k: Dimension of Keys and Queries
            dropout: Dropout probability
        """
        super().__init__()
        self.d_k = d_k
        # Initialize the softmax layer (torch.nn implementation)
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        """
        Computes the scaled dot attention given query, key and value inputs. Stores the scores in SCORE_SAVER for
        visualization

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs

        Shape:
            - q: (*, sequence_length_queries, d_model)
            - k: (*, sequence_length_keys, d_model)
            - v: (*, sequence_length_keys, d_model)
            - outputs: (*, sequence_length_queries, d_v)
        """
        scores = None
        outputs = None

        # Hint 2:                                                              #
        #       - torch.transpose(x, dim_1, dim_2) swaps the dimensions dim_1  #
        #         and dim_2 of the tensor x!                                   #
        #       - Later we will insert more dimensions into *, so how could    #
        #         index these dimensions to always get the right ones?         #
        #       - Also dont forget to scale the scores as discussed!           #


        # Calculate the dot product of queries and keys
        attention_weights = torch.matmul(q, torch.transpose(k,-1,-2)) # shape: (*, sequence_length_queries, sequence_length_keys)
        # Scale the scores by the square root of d_k
        attention_weights = attention_weights / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        # Apply softmax to normalize the scores
        scores = self.softmax(attention_weights)    # shape: (*, sequence_length_queries, sequence_length_keys)
        # Apply the attention weights to the values
        outputs = torch.matmul(scores,v)            # shape: (*, sequence_length_queries, d_model)

        SCORE_SAVER.save(scores)

        return outputs
