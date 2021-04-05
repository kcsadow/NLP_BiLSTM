# NLP_BiLSTM
Implement a BiLSTM for Classification 

## Intro
In "Interpretable Neural Predictions with Differentiable Binary Variables", Bastings et al. (2020) offer an alternative approach to rationale extraction. They, along with others (Lei et al. 2016), claim that text classification models can be improved using _rationales_ for why models predict text as representing either positive or negative sentiment. The idea is that a rationale - a "short, yet sufficent" piece of text - can help us better understand how a model's algorithm used the data to create its predictions.

The goal of our project was to partially replicate the first part of this paper, that is, classifying movie reviews from the Stanford Sentiment Treebank (SST2) as being either positive or negative. (Bastings et al. (2020) used five classes rather than two, and they utilize a GloVe model trained on a larger dataset.) We followed the model outlined in appendix B.2, which takes in movie reviews, applies an embedding to the words in the review, and then classifies the text using a Bidirectional Long Short Term Memory (BiLSTM) model. We found our model to be approximately 78% accurate on a test dataset. The details of our methods and results will be discussed in the following sections. 

For future work, we would like to consider 1) shifting from two-part classification to five-part classification; 2) adding a validation step to optimize model weights; 3) adding an attention layer; and 4) adding rational extraction to our base model.

## Methods
We run a BiLSTM model for two-part classification on movie reviews. In particular, we follow the model pictured below. 

![BiLSTM](https://kcsadow.github.com/NLP_BiLSTM/assets/bilistm.png)

Before running the model itself, our first step is to import the text of the reviews. Our data comes from the SST2, which includes 9,613 reviews that have already been split into 6,920 training examples, 872 validation examples, and 1,821 testing examples. For our initial analyses, we focus only on the training and testing examples. We preprocess our text by putting the reviews in lowercase and then removing contractions, punctuation, and special characters. Each word in a review is an element of our model's input matrix.

We now need to link these words to a weight matrix and optimize those weights for classification. Our second step is therefore to apply word embeddings to each word in a review. Following Bastings et al. (2020), we utilized GloVe word embeddings. We chose the GloVe model trained on 6 billion tokens from Wikpedia 2014 and Gigaword 5, which returns a 300-dimension vector representation for each word. These word embeddings indicate how a given word co-occurs with other words. These embeddings act as the initial weights that we are attempting to optimize over. 

Our third step is to set up our model architecture. Our model takes as input our preprocessed text. We introduce dropout and apply our initial weighting matrix (i.e. the GloVe embeddings). At this point, we create our BiLSTM model. This model reads the input text from left to right, then from right to left. As the model reads in the text from either direction, it takes the initial embedding matrix of 300 dimensions and then generates a new weighting matrix (or hidden state) based on dimension hyperparameters we provide the model with in advance. Lastly, we apply a fully-connected, linear layer which concatenates these new weights from both directions.

We next set up a training loop for our BiLSTM model. Within the training loop, we first generate our model's predictions, then update our model paramters with PyTorch's implementation of cross-entropy Loss. This implementation takes the output of our model's linear layer and applies a logged softmax to predict the sentiment of a given review; it then calculates the loss with which we run backprop. Our training loop includes 10 epochs, so our model runs on the entire training data 10 times. We save the weights from the final iteration and apply those to our test examples to test the accuracy of our model on unseen data.   

In terms of hyperparameters, Basting et al. 2020 uses the Adam optimizer with a learning rate of 0.0002, a hidden size of 150, a batch size of 25, a dropout probability of 0.5, and a weight decay of 10e-6. Due to our use of a smaller GloVe embeding and our focus on two-part classification, this set of hyperparameters induces a greater loss that would be expected. Hence, we made the following modifications: we used a hidden size of 100, a batch size of 10, and a learning rate of 0.001. (We assessed these hyperparamters manually. A possible future improvement would be to implement a grid search or cross-validation for hyperparamters.) 

## Results 

After implementing and running our model, we find that it is indeed learning new embedding weights to better classify the reviews as being of positive or negative sentiment. This is evidenced by the clear decrease in the training loss after each run through the training dataset. 

| Epoch: 01 | Train Loss: 384.848
| Epoch: 02 | Train Loss: 233.480
| Epoch: 03 | Train Loss: 124.787
| Epoch: 04 | Train Loss: 63.286
| Epoch: 05 | Train Loss: 38.415
| Epoch: 06 | Train Loss: 28.379
| Epoch: 07 | Train Loss: 28.656
| Epoch: 08 | Train Loss: 23.972
| Epoch: 09 | Train Loss: 21.096
| Epoch: 10 | Train Loss: 17.085

This trend is also present when using the hyperparameters from the paper, albeit with different performance:

| Epoch: 01 | Train Loss: 166.863
| Epoch: 02 | Train Loss: 132.821
| Epoch: 03 | Train Loss: 117.455
| Epoch: 04 | Train Loss: 107.327
| Epoch: 05 | Train Loss: 95.827
| Epoch: 06 | Train Loss: 86.891
| Epoch: 07 | Train Loss: 82.442
| Epoch: 08 | Train Loss: 72.420
| Epoch: 09 | Train Loss: 64.861
| Epoch: 10 | Train Loss: 60.507

We find our model does okay at classifying unseen data as either positive or negative. When testing the final model weights on the unseen test data, we get an accuracy rate of approximately 78%. This suggest that our model does better at properly predicting the sentiment of a movie review relative to chance. 


## Data Sources
SST2: https://github.com/clairett/pytorch-sentiment-classification/tree/master/data

GloVe: https://nlp.stanford.edu/projects/glove/

Bastings et. al: https://www.aclweb.org/anthology/P19-1284v2.pdf
