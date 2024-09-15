import torch
import torch.nn as nn

from dataset import HamSpamDataset
from transformer import TransformerEncoder
from model import HamSpamClassifier
from torch.utils.data import DataLoader, Dataset

def train(dataset: str, epochs = 5, learning_rate = 1e-2):
    """
    Trains the HamSpamClassifier model for ham/spam classification using a given dataset.

    Args:
        dataset (str): Path to the dataset file (CSV).
        epochs (int, optional): Number of training epochs. Default is 5.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-2.

    The function performs the following steps:
    1. Loads and preprocesses the dataset.
    2. Splits the dataset into training and testing sets.
    3. Initializes the HamSpamClassifier model.
    4. Trains the model using Binary Cross-Entropy Loss with Logits.
    5. Saves the trained model to a file (`hamspam_model.pth`).
    6. Evaluates the model on the test set and prints the accuracy.
    """
    
    # initializing the dataset
    dataset = HamSpamDataset('email.csv')

    # train and test splits
    trainSize = int(0.8 * len(dataset))
    testSize = len(dataset) - trainSize

    # preparing data loaders
    trainDataset, testDataset = torch.utils.data.random_split(dataset, [trainSize, testSize])
    trainDataLoader = DataLoader(trainDataset, batch_size = 32, shuffle = True)
    testDataLoader = DataLoader(testDataset, batch_size = 32, shuffle = True)

    # dimensions for the model
    VOCAB_SIZE = dataset.getVocabSize()
    EMBEDDING_DIM = 512
    MAXLEN = dataset.getMaxLen()

    # initializing the model
    model = HamSpamClassifier(VOCAB_SIZE, EMBEDDING_DIM, MAXLEN)

    # training parameters
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    epochs = epochs

    # training loop
    model.train()
    for epoch in range(epochs):
        for batch, (tokens, category) in enumerate(trainDataLoader):

            category = category.float()

            output = model(tokens)
            output = output.squeeze()

            loss = loss_fn(output, category)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Epoch: {epoch} | Batch: {batch} | Loss: {loss}")
        print("\n\n")

    # saving the model
    torch.save(model.state_dict(), 'hamspam_model.pth')
    print("Model Saved Successfully!")

    # evaluation
    modelBest = HamSpamClassifier(VOCAB_SIZE, EMBEDDING_DIM, MAXLEN)
    modelBest.load_state_dict(torch.load('hamspam_model.pth'))

    modelBest.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for tokens, category in testDataLoader:
            outputs = modelBest(tokens)
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            total += category.size(0)
            correct += (predicted == category).sum().item()
        print('Test Accuracy: {} %'.format(100 * correct / total))


if __name__ == "__main__":
    train('email.csv', 5, 0.01)