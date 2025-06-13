from torch import nn
import torch    

from ..network import ScaledDotAttention

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            dropout: Dropout probability
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        # Hints 3:                                                             #
        #       - Instead of initializing several weight layers for each head, #
        #         you can create one large weight matrix. This speed up        #
        #         the forward pass, since we dont have to loop through all     #
        #         heads!                                                       #
        #       - All linear layers should only be a weight without a bias!    #

        # Initialize all weight layers as linear layers 
        self.weights_q = torch.nn.Linear(d_model, n_heads * d_k, bias=False)
        self.weights_k = torch.nn.Linear(d_model, n_heads * d_k, bias=False)
        self.weights_v = torch.nn.Linear(d_model, n_heads * d_v, bias=False)
        # Initialize the ScaledDotAttention 
        self.attention = ScaledDotAttention(d_model)
        # Initialize the projection layer as a linear layer             
        self.project = torch.nn.Linear(n_heads * d_v, d_model, bias=False)


    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs

        Shape:
            - q: (batch_size, sequence_length_queries, d_model)
            - k: (batch_size, sequence_length_keys, d_model)
            - v: (batch_size, sequence_length_keys, d_model)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """

        # You will need these here!
        batch_size, sequence_length_queries, _ = q.size()
        _, sequence_length_keys, _ = k.size()

        outputs = None

        # Hints 3:                                                             #
        #       - It helps to write down which dimensions you want to have on  #
        #         paper!                                                       #
        #       - Above the todo, we have already extracted the batch_size and #
        #         the sequence lengths for you!                                #
        #       - Use reshape() to split or combine dimensions                 #
        #       - Use transpose() again to swap dimensions                     #                            


        #  Pass q,k and v through the linear layer                      
        q_out = self.weights_q(q)
        # Split the last dimensions into n_heads and d_k od d_v  
        q_out = torch.reshape(q_out, (batch_size, sequence_length_queries, self.n_heads, -1))
        # Swap the dimensions so that the shape matches the required
        q_out = torch.transpose(q_out, -2, -3)  # change (batch_size, sequence_length_queries, n_heads, d_k) to (batch_size, n_heads, sequence_length_queries, d_k)
        k_out = self.weights_k(k)
        k_out = torch.reshape(k_out,(batch_size, sequence_length_keys, self.n_heads, -1))
        k_out = torch.transpose(k_out, -2, -3)
        v_out = self.weights_v(v)
        v_out = torch.reshape(v_out,(batch_size, sequence_length_keys, self.n_heads, -1))
        v_out = torch.transpose(v_out, -2, -3)
        # Pass the ScaledDotAttention 
        attention = self.attention(q_out, k_out, v_out)
        attention = torch.softmax(q_out@torch.transpose(k_out,-1,-2)/self.d_k**0.5,-1)@v_out
        # Swap the dimensions of the output back
        attention = torch.transpose(attention, -2, -3)
        # Combine the last two dimensions
        attention = torch.reshape(attention, (-1, sequence_length_queries, self.n_heads * self.d_v))
        # Pass the outputs through the projection layer 
        outputs = self.project(attention)

        return outputs
    