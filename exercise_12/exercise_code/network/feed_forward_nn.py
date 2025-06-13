from torch import nn
import torch

class FeedForwardNeuralNetwork(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        # Initialize the feed forward network   
        self.linear_1 = torch.nn.Linear(d_model, d_ff)
        self.relu = torch.nn.ReLU() 
        self.linear_2 = torch.nn.Linear(d_ff, d_model)
        # Initialize the dropout layer
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            inputs: Inputs to the Feed Forward Network

        Shape:
            - inputs: (batch_size, sequence_length_queries, d_model)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """
        # Implement forward pass of feed forward layer
        outputs = self.linear_1(inputs)
        outputs = self.relu(outputs)
        outputs = self.linear_2(outputs)
        # Pass the output through a dropout layer as a final step 
        outputs = self.dropout(outputs)

        return outputs