# NLP_BiLSTM
Implement a BiLSTM for Classification 

## Intro
The authors goal in "Interpretable Neural Predictions with Differentiable Binary Variables" (Bastings et al. 2016) is to improve on text classification models by providing a _rationale_ as to why a model predicted text as representing either positive or negative sentiment. The idea is that by providing a rationale, we can better understand how an algorithm used the underlying text to create the resulting prediction (e.g., positive or negative). The authors define a rationale as a "short, yet sufficient" piece of text that the model used in order to classify the text as either positive or negative.

The goal of this project is to replicate the first piece of this paper, which is classifying a movie review from the Stanford Sentiment Treebank (SST2) as either a positive or negative movie review. We follow the model outlined in appendix B.2, which inputs movie reviews, applies an embedding to the words in the review, and then classifies the text using a Bidirectional Long Short Term Memory (BiLSTM) model. We find our model is 78% accurate in classifying movie reviews as either positive or negative.  The next phase of this project is to 1) move from 2-part classification to 5-part classification, 2) add in a validation step where we choose the model weights from highest performing model, 3) add an attention layer, and 4) add to the model such that it can perform rationale extraction.

## Methods
We run a BiLSTM model for two-class classification on movie reviews. In particular, we follow the model pictured below. 

<img src="{{ site.url }}{{ site.baseurl }}/assets/bilstm.png">

The first phase of our model is the input text. Our initial text input comes from the Stanford Sentiment Treebank, which includes 9,613 reviews representing 6,920 training examples, 872 validation examples, and 1,821 testing examples. For our initial analysis, we focus only on the training and testing examples. We start by preprocessing our text. We lowercase the reviews and then remove contractions, punctuation, and special characters. Each word in a review makes up what is our "X" matrix. These are the inputs to our model. The next step is to associate an initial weight matrix to these words and then optimize those weights for an appropriate classification.

The second phase of our model applies word embeddings to each word in a review. For this, we use GloVe word embeddings. We use the the GloVe model trained on 6 billion tokens from Wikpedia 2014 and Gigaword 5, which returns a 300 dimension representation of each word. These word embeddings are a vector representation of how a respective word co-occurs with other words. These embeddings act as our initial weight, which we are attempting to optimize over. 

Our model architecture takes as input our text. We apply dropout to our input text. We then apply an initial weighting matrix (i.e. the GloVe embeddings). At this point, we create our BiLSTM model. This model reads the text in from left to right, and then from right to left. As the model reads in the text from either direction, it takes the initial embedding matrix of 300 dimensions and then creates a new weighting matrix of a size we determine. With these new weights, we then apply a linear layer which concatenates the weights from either direction. This represents our entire BiLSTM model.

In order to get our predictions (and improve on them), we next create a training loop. Within the training loop, we first run our model to get our predictions. We then apply Cross Entropy to these predictions to properly classify our results as positive or negative and obtain the resulting loss. Cross Entropy takes the linear layer and applies a logged soft max in order to predict the sentiment of the review (e.g., positive or negative) and this function provides the specific loss from our model. In order for our linear layer to learn, we take the loss and back propogate to get the gradients from our model. The paper recommends using the Adam optimizer with a learning rate of 0.0002, hidden size of 150, batch size of 25, dropout of 0.5, and weight decay of 10e-6. Due to the fact that we are using a different GloVe embeding and our focusing on 2-class classification, we keep the dropuout of 0.5 but adjust our model parameters to a hidden size of 100. We use the Adam optimizer with a learning rate of 0.0002 and weight decay of 10e-6. We run through our training loop using a batch size of 10, due to the smaller nature of our dataset. Our training loop runs through the training data 10 times.   



## Results 


## Data Sources
SST2: https://github.com/clairett/pytorch-sentiment-classification/tree/master/data
GloVe: https://nlp.stanford.edu/projects/glove/
