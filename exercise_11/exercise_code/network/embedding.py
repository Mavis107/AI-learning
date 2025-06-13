from torch import nn
import torch

def positional_encoding(d_model: int,
                        max_length: int) -> torch.Tensor:
    """
    Computes the positional encoding matrix
    Args:
        d_model: Dimension of Embedding
        max_length: Maximums sequence length

    Shape:
        - output: (max_length, d_model)
    """

    # Generates an array of even indices from 0 to d_model (exclusive), representing every other dimension
    i = torch.arange(0, d_model, 2) / d_model
    # Creates a column vector of positions from 0 to max_length - 1, where each row represents a position in the sequence
    # The extra dimension is added with [:, None] to enable broadcasting when multiplying with angle frequencies
    pos = torch.arange(0, max_length)[:, None]
    # Computes the scaling factor for the frequencies
    angle_freq = torch.exp(i * (-torch.log(torch.Tensor([10000]))))
    # Initialize the positional encoding
    output = torch.zeros((max_length, d_model))
    # Take the sine of the even indices
    output[:, 0::2] = torch.sin(pos * angle_freq)
    # Take the cosine of the odd indices
    output[:, 1::2] = torch.cos(pos * angle_freq)

    return output
    


class Embedding(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 max_length: int):
        """

        Args:
            vocab_size: Number of elements in the vocabulary
            d_model: Dimension of Embedding
            max_length: Maximum sequence length
        """
        super().__init__()

        # Initialize the embedding layer (torch.nn implementation)
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        # Initialize the positional encoding layer
        self.pos_encoding = positional_encoding(d_model, max_length)


        # We will convert it into a torch parameter module for you! You can treat it like a normal tensor though!
        if self.pos_encoding is not None:
            self.pos_encoding = nn.Parameter(data=self.pos_encoding, requires_grad=False)

    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """
        The forward function takes in tensors of token ids and transforms them into vector embeddings. 
        It then adds the positional encoding to the embeddings, and if configured, performs dropout on the layer!

        Args:
            inputs: Batched Sequence of Token Ids

        Shape:
            - inputs: (batch_size, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """

        outputs = None

        # Use fancy indexing to extract the positional encodings until position sequence_length
        sequence_length = inputs.shape[-1]
        pos_encoding = 0
        if self.pos_encoding is not None:
            pos_encoding = self.pos_encoding[:sequence_length]


        # Compute the outputs of the embedding layer                 
        outputs = self.embedding(inputs)
        # Add the positional encoding to the output
        outputs += pos_encoding


        return outputs