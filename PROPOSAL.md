# Project Proposal: Climate Change Tweet Sentiment Classification

## What is Sentiment Analysis?

**Sentiment analysis** is a natural language processing (NLP) technique used to determine whether a piece of text expresses a positive, negative, or neutral opinion. It's widely used in areas like customer feedback, product reviews, social media monitoring, and political opinion mining. In this project, sentiment analysis is applied to identify public stance toward climate change based on tweets.

This task is a form of **multi-class sentiment classification**, where the goal is to predict one of four sentiment categories that reflect a user's attitude or opinion about climate change.

## Dataset
This project will use the **Climate Change Tweet Dataset**, which contains 43,943 tweets related to climate change, annotated by three independent reviewers.

Each tweet has been classified into one of the following sentiment categories:

- `2 (News)`: Links to factual news about climate change  
- `1 (Pro)`: Supports belief in man-made climate change  
- `0 (Neutral)`: Neither supports nor refutes belief in man-made climate change  
- `-1 (Anti)`: Does not believe in man-made climate change  

All included tweets have full agreement among the reviewers, making this a high-quality labeled dataset.  
Source: [Climate Change Tweets Dataset on Kaggle](https://www.kaggle.com/datasets/sbhatti/Climate-Change-Tweet-Dataset)

## What is BERT?

**BERT (Bidirectional Encoder Representations from Transformers)** is a powerful pre-trained language model developed by Google. It is designed to understand the context of words in a sentence by looking at both the left and right sides of a word (bidirectional attention). 

BERT can be fine-tuned for various downstream NLP tasks like sentiment analysis, question answering, and named entity recognition. In this project, BERT will be fine-tuned specifically for multi-class sentiment classification of climate-related tweets.

## Modeling Approach
My goal is to build a **multi-class classification model** that can predict the sentiment label of a tweet related to climate change.

Modeling workflow:
- Preprocessing tweets 
- Tokenization using a transformer model like `DistiBERT`
- Fine-tuning a Hugging Face transformer (`BertForSequenceClassification`)
- Comparing performance against a baseline model (TF-IDF + Logistic Regression)
- Evaluating results using metrics like accuracy, F1-score, and confusion matrix

## GitHub Repository
🔗 https://github.com/PranithaChilvari/LING539-Sentiment-Classification.git

## LiveSite
🔗 
