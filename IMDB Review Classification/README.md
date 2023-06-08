# IMDB Reviews Sentiment Classification

In this project, I aim to understand the NLP task of Text Classification. Text classification also known as text tagging or text categorization is the process of categorizing text into organized groups. By using Natural Language Processing (NLP), text classifiers can automatically analyze text and then assign a set of pre-defined tags or categories based on its content.

The project is divided into three sections - **Text Classification from scratch**, **Text Classification with Transformer**, **Text Classification with RNN**

## Text Classification from scratch
🗣️ We start with importing the required libraries for the task. Then we download the dataset from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz.

🗣️ We use the Keras function tf.keras.utils.text_dataset_from_directory to load the dataset and create training, validation and test sets.

🗣️ Text Vectorization is an important step in NLP projects. It converts raw text into vectors which can be processed by our model.

🗣️ We then build the model using Keras layers.

🗣️ Next steps is to compile the model and fit it on the training data.

🗣️ Final step is to evaluate the model on the test set.

## Text Classification with Transformer
🗣️ After importing the required libraries, we implement the transformer block which will be used in our model.

🗣️ We also implement the token and position embedding layer.

🗣️ For preparing the data, we directly load the data into our training and validation sets from keras.datasets.

🗣️ Next, we build our classifier model starting off with embedding our data and passing the embedded data through the transformer block.

🗣️ Lastly, we compile, fit and evaluate our model on the training set.

## Text Classification with RNN
🗣️ In this section, we use the library tensorflow_datasets to load our data.

🗣️ After creating training and test sets, we vectorize the data.

🗣️ After vectorization, we build the model with the bidirectional LSTM layer in our network.

🗣️ Finally, we compile, train and fit the model on our training set.
