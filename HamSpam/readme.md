# HamSpam

Spam emails, commonly known as junk emails, are unsolicited messages sent in bulk, often for advertising, phishing, or spreading malicious software. They clutter inboxes and can lead to harmful outcomes if not filtered properly. Spam detection is a crucial task in maintaining email security and user experience. Traditionally, spam filters have relied on rule-based approaches or simple machine learning classifiers like Naive Bayes or Support Vector Machines (SVM). However, with the growing complexity of spam messages, more advanced models are required to effectively classify emails as spam or ham (non-spam).

Transformers have revolutionized the field of Natural Language Processing (NLP) by introducing a more powerful architecture for sequence-to-sequence tasks, outperforming previous models like Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks. Unlike RNNs or LSTMs, which process input sequentially, transformers use self-attention mechanisms to capture relationships between tokens in a sentence without considering their sequential order.

## Code Analysis

Model Structure:

- The model consists of two primary components:

    - TransformerEncoder: This handles the token embedding and self-attention-based representation of emails. It includes token embedding, positional encoding, multi-head attention, and feed-forward layers with layer normalization.

    - HamSpamClassifier: This class extends the TransformerEncoder to output classification probabilities for spam and ham. After the transformer processes the input, a linear classifier converts the final embedding into a binary classification output (spam or ham).

Dataset Processing:

- The HamSpamDataset class handles loading and processing of the dataset. It tokenizes the email content using the tiktoken library and pads the sequences to ensure uniform input sizes.

- The dataset is split into training and testing sets (80%-20% split), and DataLoader is used to feed the data into the model in mini-batches during training.

Training Loop:

- The training process runs for a specified number of epochs. For each batch, the model computes the binary cross-entropy loss between the predicted outputs and the actual labels (spam/ham). The optimizer then adjusts the model’s weights accordingly.

- A checkpoint of the best-performing model is saved as `hamspam_model.pth`.

Evaluation:

- After training, the model is evaluated on the test dataset. The test accuracy is computed by applying a sigmoid activation followed by rounding to classify the model's predictions into spam or ham categories.

## Training Results

The model was trained on a dataset of email messages and achieved `97%` accuracy on the test set. This high level of performance demonstrates the effectiveness of the transformer-based approach for classifying emails, even when compared to more traditional spam filters.

## Demo

1. Clone this repository
2. Run `python train.py` to train the model