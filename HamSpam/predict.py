import torch

from dataset import HamSpamDataset
from model import HamSpamClassifier

def predict_email(email, model, dataset):
  """Predicts whether a given email is spam or ham.

  Args:
    email: The email text to be classified.
    model: The trained HamSpamClassifier model.
    dataset: The HamSpamDataset instance used to preprocess the email.

  Returns:
    "spam" if the email is predicted to be spam, "ham" otherwise.
  """

  # Tokenize and pad the email
  tokens = torch.tensor(dataset.encoder.encode(email))
  padded_tokens = torch.nn.functional.pad(tokens, (0, dataset.getMaxLen() - len(tokens)), "constant", dataset.getVocabSize() - 1)
  padded_tokens = padded_tokens.unsqueeze(0)  # Add batch dimension

  # Make prediction
  with torch.no_grad():
    output = model(padded_tokens)
    predicted = torch.round(torch.sigmoid(output.squeeze()))

  # Return prediction
  if predicted.item() == 1:
    return "spam"
  else:
    return "ham"

def main(text: str):
    """Main function to demonstrate the email classification process.

    Args:
        text: The email text to be classified.
    """

    dataset = HamSpamDataset('email.csv')

    EMBEDDING_DIM = 512
    VOCAB_SIZE = dataset.getVocabSize()
    MAXLEN = dataset.getMaxLen()

    modelBest = HamSpamClassifier(VOCAB_SIZE, EMBEDDING_DIM, MAXLEN)
    modelBest.load_state_dict(torch.load('hamspam_model.pth'))

    email_text = text
    prediction = predict_email(email_text, modelBest, dataset)
    print(f"Email classified as: {prediction}")

if __name__ == "__main__":
  sampleText = "Congrats! Nokia 3650 video camera phone is your Call 09066382422 Calls cost 150ppm Ave call 3mins vary from mobiles 16+ Close 300603 post BCM4284 Ldn WC1N3XX"
  main(sampleText)