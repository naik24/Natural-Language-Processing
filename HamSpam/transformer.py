import torch
import tiktoken
import numpy as np
import math
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from torch.nn import MultiheadAttention

class InputEmbeddings(nn.Module):
  """
  Embeds integer token indices into dense embedding vectors.

  This module takes integer token indices as input and maps them to dense
  embedding vectors using an embedding lookup table. It also handles padding
  by assigning a special embedding vector to the padding token.

  Args:
    vocabSize: The size of the vocabulary (number of unique tokens).
    embeddingDim: The dimension of the embedding vectors.
  """
  def __init__(self, vocabSize, embeddingDim):
    super(InputEmbeddings, self).__init__()
    self.embedder = nn.Embedding(vocabSize, embeddingDim, padding_idx = vocabSize - 1)

  def forward(self, tokens):
    """
    Embeds a sequence of integer tokens.

    Args:
      tokens: A LongTensor of shape (batch_size, sequence_length) containing
              integer token indices.

    Returns:
      A FloatTensor of shape (batch_size, sequence_length, embeddingDim)
      containing the embedded token vectors.
    """

    return self.embedder(tokens)
  
class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings.

    This module implements the positional encoding technique described in the
    "Attention is All You Need" paper. It adds sinusoidal encodings to the
    embedding vectors, allowing the model to learn the relative positions of
    tokens in a sequence.

    Args:
      embeddingDim: The dimension of the embedding vectors.
      maxLen: The maximum sequence length that the model can handle.
    """

    def __init__(self, embeddingDim, maxLen):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(maxLen, embeddingDim)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embeddingDim, 2).float() * -(math.log(10000.0) / embeddingDim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Adds positional encoding to the input embeddings.

        Args:
          x: A FloatTensor of shape (batch_size, sequence_length, embeddingDim)
              containing the input embeddings.

        Returns:
          A FloatTensor of the same shape as the input with positional encoding
          added.
        """

        return x + self.pe[:, :x.size(1)]
    
class MultiHeadAttention(nn.Module):
  """
  Implements a multi-head attention layer.

  This module performs multi-head attention as described in the "Attention is All You Need" paper.
  It allows the model to attend to different parts of the input sequence simultaneously, capturing
  relationships between tokens from different positions.

  Args:
    embeddingDim: The dimension of the embedding vectors.
    heads: The number of attention heads (default: 8).
  """
    
  def __init__(self, embeddingDim, heads = 8):
    super(MultiHeadAttention, self).__init__()
    self.q = nn.Linear(embeddingDim, embeddingDim)
    self.k = nn.Linear(embeddingDim, embeddingDim)
    self.v = nn.Linear(embeddingDim, embeddingDim)
    self.mha = nn.MultiheadAttention(embeddingDim, heads, batch_first = True)
    self.layernorm = nn.LayerNorm(embeddingDim)

  def forward(self, x): # here x is input encodings
    """
    Performs multi-head attention on the input sequence.

    Args:
      x: A FloatTensor of shape (batch_size, sequence_length, embeddingDim)
          containing the input sequence.

    Returns:
      A FloatTensor of the same shape as the input with the attention weights
      applied.
    """

    Q = self.q(x)
    K = self.k(x)
    V = self.v(x)

    attnOutput, attnOutputWeights = self.mha(Q, K, V)
    x = attnOutput + x
    x = self.layernorm(x)

    return x
  
class FeedForwardNetwork(nn.Module):
  """
    A feed-forward neural network used as part of the Transformer architecture. 
    It consists of two linear layers with a ReLU activation function in between, 
    followed by an optional layer normalization.

    Args:
        embeddingDim (int): The dimensionality of the input and output vectors.
    
    Attributes:
        l1 (nn.Linear): First linear layer that maps from embeddingDim to 2048 dimensions.
        relu (nn.ReLU): ReLU activation function.
        l2 (nn.Linear): Second linear layer that maps back to the embeddingDim size.
        layernorm (nn.LayerNorm): Layer normalization applied after the feed-forward network.
  """

  def __init__(self, embeddingDim):
    """
    Initializes the feed-forward network layers.
    
    Args:
        embeddingDim (int): The input/output dimension for the linear layers.
    """

    super(FeedForwardNetwork, self).__init__()
    self.l1 = nn.Linear(embeddingDim, 2048)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(2048, embeddingDim)
    self.layernorm = nn.LayerNorm(embeddingDim)

  def forward(self, x):
    """
    Defines the forward pass of the feed-forward network.
    
    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, embeddingDim).

    Returns:
        torch.Tensor: The output tensor of the same shape as the input, after applying the linear layers and ReLU activation.
    """

    x = self.l1(x)
    x = self.relu(x)
    x = self.l2(x)
    return x
  
class TransformerEncoder(nn.Module):
  """
  A Transformer encoder layer. It consists of an input embedding layer, positional encoding, 
  multi-head self-attention, and a feed-forward network, with residual connections and layer normalization.

  Args:
      vocabSize (int): The size of the input vocabulary.
      embeddingDim (int): The dimensionality of token embeddings.
      maxLen (int): The maximum length of input sequences.

  Attributes:
      embedder (InputEmbeddings): A layer that transforms input token indices into dense embeddings.
      positionalEncoder (PositionalEncoding): A layer that adds positional information to the input embeddings.
      multiheadattention (MultiHeadAttention): Multi-head self-attention mechanism.
      ffn (FeedForwardNetwork): Feed-forward network applied after attention.
      layernorm (nn.LayerNorm): Layer normalization applied after the feed-forward network and residual connection.
  """

  def __init__(self, vocabSize, embeddingDim, maxLen):
    super(TransformerEncoder, self).__init__()
    self.embedder = InputEmbeddings(vocabSize, embeddingDim)
    self.positionalEncoder = PositionalEncoding(embeddingDim, maxLen)
    self.multiheadattention = MultiHeadAttention(embeddingDim, 8)
    self.ffn = FeedForwardNetwork(embeddingDim)
    self.layernorm = nn.LayerNorm(embeddingDim)

  def forward(self, x):
    """
    Defines the forward pass of the Transformer encoder.
    
    Args:
        x (torch.Tensor): The input tensor of token indices with shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: The output tensor of shape (batch_size, sequence_length, embeddingDim) after applying 
                      the embedding layer, positional encoding, multi-head attention, and feed-forward network.
    """
    x = self.embedder(x)
    x = self.positionalEncoder(x)

    mhaOutput = self.multiheadattention(x)

    ffnOutput = self.ffn(mhaOutput)
    x = ffnOutput + mhaOutput
    x = self.layernorm(x)
    
    return x