import torch
import torch.nn as nn

from transformer import TransformerEncoder

class HamSpamClassifier(nn.Module):
  """
  A binary classification model built on a Transformer encoder for text classification.
  The model is designed to classify text data into two categories: ham or spam.

  Args:
      vocabSize (int): Size of the vocabulary (number of unique tokens in the dataset).
      embeddingDim (int): Dimensionality of the token embeddings.
      maxLen (int): Maximum length of the input sequence.

  Attributes:
      transformer (TransformerEncoder): The transformer encoder responsible for processing the input sequence.
      classifier (nn.Sequential): A linear layer for binary classification. It outputs a single score that can be
                                    interpreted as the probability of the input being spam.
  """

  def __init__(self, vocabSize, embeddingDim, maxLen):
    """
    Initializes the HamSpamClassifier with the given vocabulary size, embedding dimension, and maximum sequence length.
    
    Args:
        vocabSize (int): The number of unique tokens in the dataset.
        embeddingDim (int): The size of each embedding vector.
        maxLen (int): The maximum sequence length for the input data.
    """
    super(HamSpamClassifier, self).__init__()
    self.transformer = TransformerEncoder(vocabSize, embeddingDim, maxLen)
    self.classifier = nn.Sequential(
        nn.Linear(embeddingDim, 1)
    )
  
  def forward(self, x):
    """
    Defines the forward pass of the model. First, the input sequence is passed through the transformer encoder.
    Then, the output of the encoder is averaged across the sequence, and finally passed through a linear layer
    to predict a binary class (ham or spam).
    
    Args:
        x (torch.Tensor): The input tensor of token indices with shape (batch_size, sequence_length).
    
    Returns:
        torch.Tensor: A tensor of shape (batch_size, 1) representing the predicted probability of the input
                      being in the spam class.
    """
    
    x = self.transformer(x)
    x = torch.mean(x, dim = 1)
    x = self.classifier(x)

    return x