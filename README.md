# NLP_BiLSTM
Implement a BiLSTM for Classification 

## Intro
The authors goal in "Interpretable Neural Predictions with Differentiable Binary Variables" (Bastings et al. 2016) is to improve on text classification models by providing a _rationale_ as to why a model predicted text as representing either positive or negative sentiment. The idea is that by providing a rationale, we can better understand how an algorithm used the underlying text to create the resulting prediction (e.g., positive or negative). The authors define a rationale as a "short, yet sufficient" piece of text that the model used in order to classify the text as either positive or negative.

The goal of this project is to replicate the first piece of this paper, which is classifying a movie review from the Stanford Sentiment Treebank (SST2) as either a positive or negative movie review. We follow the model outlined in appendix B.2, which inputs movie reviews, applies an embedding to the words in the review, and then classifies the text using a Bidirectional Long Short Term Memory (BiLSTM) model. We find our model is 78% accurate in classifying movie reviews as either positive or negative.  The next phase of this project is to update the model in order to replicate the rationale extractor.

## Methods


## Results 
