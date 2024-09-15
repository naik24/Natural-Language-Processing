import torch
import tiktoken
import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

class HamSpamDataset(Dataset):
  """
  Prepares a dataset for ham/spam email classification. It loads email data from a CSV file, tokenizes the text, 
  and returns the corresponding tokenized sequences and labels (0 for ham, 1 for spam).

  Args:
      filename (str): The path to the CSV file containing the dataset. The file must have 'Message' and 'Category' columns.
      
  Attributes:
      dataframe (pd.DataFrame): A DataFrame containing the email data.
      emails (pd.Series): A series containing the email messages.
      category (pd.Series): A series containing the labels ('ham' or 'spam') for the emails.
      encoder (tiktoken.Encoding): The tokenizer used for encoding the email text (GPT-2 tokenizer in this case).
      tokenizedEmails (list of torch.Tensor): A list of tokenized emails as tensors.
      paddedEmails (torch.Tensor): A tensor containing tokenized emails with padding applied to the sequences.
  """

  def __init__(self, filename):
    """
    Initializes the dataset by loading the email data, tokenizing the messages, and padding the sequences.

    Args:
        filename (str): The path to the CSV file with the 'Message' and 'Category' columns.
    """

    # reading and processing dataset
    self.filename = filename
    self.dataframe = pd.read_csv(self.filename)
    self.emails = self.dataframe['Message']
    self.category = self.dataframe['Category']

    # initializing the tokenizer
    self.encoder = tiktoken.get_encoding('gpt2')

    # tokenizing and padding emails
    self.tokenizedEmails = [torch.tensor(self.encoder.encode(x)) for x in self.emails]
    self.paddedEmails = pad_sequence(self.tokenizedEmails, batch_first = True, padding_value = self.getVocabSize() - 1)
    

  def getVocabSize(self):
    """
    Returns the vocabulary size of the tokenizer.

    Returns:
        int: The size of the vocabulary used by the GPT-2 tokenizer, plus 1 for padding.
    """

    return self.encoder.n_vocab + 1

  def getMaxLen(self):
    """
    Returns the length of the longest sequence in the dataset (before padding).

    Returns:
        int: The length of the longest tokenized email.
    """

    return max([len(x) for x in self.tokenizedEmails])

  def __len__(self):
    """
    Returns the number of samples in the dataset.

    Returns:
        int: The number of emails in the dataset.
    """

    return len(self.dataframe)

  def __getitem__(self, idx):
    """
    Returns the tokenized and padded sequence and its corresponding label at the given index.

    Args:
        idx (int): The index of the sample to retrieve.

    Returns:
        tuple: A tuple containing the tokenized email (torch.Tensor) and the label (int, 0 for ham and 1 for spam).
    """
    
    tokens = self.paddedEmails[idx]
    
    category = self.category.iloc[idx]
    if category == 'spam':
      category = 1
    else:
      category = 0
    
    return tokens, category
  
if __name__ == "__main__":
  # initializing the dataset
  dataset = HamSpamDataset('email.csv')
  
  # train and test splits
  trainSize = int(0.8 * len(dataset))
  testSize = len(dataset) - trainSize

  # preparing train and test datasets
  trainDataset, testDataset = torch.utils.data.random_split(dataset, [trainSize, testSize])

  # preparing data loaders
  trainDataLoader = DataLoader(trainDataset, batch_size = 32, shuffle = True)
  testDataLoader = DataLoader(testDataset, batch_size = 32, shuffle = True)

  print("Dataset Prepared Successfully!")