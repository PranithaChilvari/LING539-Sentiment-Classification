# LING539-Sentiment-Classification

## 🧠 What is Sentiment Analysis?

Sentiment analysis is a natural language processing (NLP) task that focuses on identifying emotions or opinions expressed in text. It is commonly used to determine whether a text expresses a **positive**, **negative**, or **neutral** attitude. In this project, we take it further by classifying tweets into four distinct categories, offering a more nuanced understanding of public opinion on climate change.


## 🤖 What is BERT?

**BERT** (Bidirectional Encoder Representations from Transformers) is a state-of-the-art language model developed by Google. Unlike earlier models that read text either left-to-right or right-to-left, BERT reads text **in both directions at once**, allowing it to better understand the context of each word.

BERT is pre-trained on a massive amount of text and can be fine-tuned on smaller, task-specific datasets like ours. This makes it ideal for sentiment classification, where subtle differences in language matter.

## 🧑🏻‍💻 Code:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import html
import torch
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


pd.set_option('display.max_colwidth', 200)
```

### 1. Loading the Dataset

Loading the dataset using the simple pandas.read_csv




```python
df = pd.read_csv('/content/twitter_sentiment_data.csv')
```

## 2. Exploratory Data Analysis

EDA helps understand the data's structure, patterns, and quality before modeling. It identifies issues like missing values and reveals insights for better decision-making.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 43943 entries, 0 to 43942
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   sentiment  43943 non-null  int64 
     1   message    43943 non-null  object
     2   tweetid    43943 non-null  int64 
    dtypes: int64(2), object(1)
    memory usage: 1.0+ MB
    


```python
df.describe()
```



Checking the unique values in the 'sentiment' column

*   2 (News): factual news
*   1 (Pro): supports man-made climate change
*   0 (Neutral): neither supports nor refutes
*   -1 (Anti): does not believe in man-made climate change


```python
df['sentiment'].unique()
```




    array([-1,  1,  2,  0])



Checking for any duplicate elements


```python
df.duplicated().sum()
```




    np.int64(0)



Checking the distribution of the sentiment classes


```python
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df, x='sentiment', order=df['sentiment'].value_counts().index)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.5, int(height), ha="center")

plt.show()
```


    
![png](output_11_0.png)
    


### 3. Data preprocessing
Data preprocessing is a crucial step for NLP tasks. In this step, I clean the tweet text by removing URLs, mentions, hashtags, and any non-alphabetic characters. I also convert all text to lowercase and strip any extra spaces. This ensures that the text is ready for tokenization and model input.


```python
df['message'].head(5)
```




```python
def clean_text(text):
    text = text.lower()  # Lowercase
    text = html.unescape(text)  # Decode HTML entities like &amp;
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'rt\s+', '', text)  # Remove 'RT'
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters, numbers, emojis
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text
```


```python
df['clean_tweet'] = df['message'].apply(clean_text)
```


```python
df['clean_tweet'].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>clean_tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>climate change is an interesting hustle as it was global warming but the planet stopped warming for yes while the suv boom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>watch right here as travels the world to tackle climate change htt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fabulous leonardo s film on change is brilliant do watch via</td>
    </tr>
    <tr>
      <th>3</th>
      <td>just watched this amazing documentary by leonardodicaprio on climate change we all think this</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pranita biswasi a lutheran from odisha gives testimony on effects of climate change natural disasters on the po</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>



### 4. Encoding the Labels
Since machine learning models require numerical inputs, I encode the sentiment labels using LabelEncoder. This transforms the four sentiment labels into numeric values (0, 1, 2, 3) so they can be used as target labels for training the model.


```python
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])  # -1 to 2 mapped to 0 to 3
```

### 5. Train-Test Split
To evaluate the model’s performance effectively, I split the dataset into training and testing sets using an 80/20 split. The training set is used to train the model, and the test set is used to evaluate the model’s ability to generalize to unseen data. I ensure the split maintains the class distribution by using stratify.


```python
X = df['clean_tweet']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

```

### 6. Baseline Model - TF-IDF + Logistic Regression
As a baseline model, I use a combination of TF-IDF (Term Frequency-Inverse Document Frequency) and Logistic Regression. First, I vectorize the tweet text using TfidfVectorizer, which transforms the text into numerical feature vectors. Then, I train a Logistic Regression model on the transformed features and evaluate its performance on the test set. I calculate metrics such as accuracy, F1-score, and present a confusion matrix to assess its classification performance.


```python

# Encode sentiment labels (if not already done)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded_sentiment'] = le.fit_transform(df['sentiment'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_tweet'],
    df['encoded_sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=df['encoded_sentiment']
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Predict
y_pred = lr_model.predict(X_test_tfidf)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' accounts for class imbalance

print("Accuracy:", accuracy)
print("F1 Score (Weighted):", f1)
#print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Optional: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - TF-IDF + Logistic Regression")
plt.show()
```

    Accuracy: 0.688474229150074
    F1 Score (Weighted): 0.6715470583136712
    


    
![png](output_22_1.png)
    


### 7. Fine-tune Transformer Model (DistilBERT)
Next, I fine-tune a pre-trained DistilBERT model, which is a lighter and faster version of BERT (Bidirectional Encoder Representations from Transformers). DistilBERT is capable of handling complex language patterns and context, making it a great choice for sentiment analysis tasks. I tokenize the tweet text using the DistilBertTokenizerFast, and then fine-tune the model using the Trainer API from Hugging Face's transformers library. The model is trained to predict sentiment classes based on the input tweet text.


```python
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Wrap our train/test data into Hugging Face Datasets format
train_ds = Dataset.from_pandas(pd.DataFrame({'text': X_train, 'label': y_train}))
test_ds = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))

# Tokenize: Convert text to input IDs and attention masks
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Format the datasets for PyTorch
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

```

```python
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=4
)

```

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro')
    }

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    metric_for_best_model='accuracy'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

```


```python
trainer.train()
trainer.evaluate()
```

    <div>

      <progress value='3297' max='3297' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [3297/3297 08:10, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>500</td>
      <td>0.838500</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>0.683900</td>
    </tr>
    <tr>
      <td>1500</td>
      <td>0.517100</td>
    </tr>
    <tr>
      <td>2000</td>
      <td>0.463000</td>
    </tr>
    <tr>
      <td>2500</td>
      <td>0.347200</td>
    </tr>
    <tr>
      <td>3000</td>
      <td>0.266100</td>
    </tr>
  </tbody>
</table><p>




<div>

  <progress value='138' max='138' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [138/138 00:09]
</div>






    {'eval_loss': 0.7461682558059692,
     'eval_accuracy': 0.7659574468085106,
     'eval_f1_macro': 0.7145001166627605,
     'eval_runtime': 9.1891,
     'eval_samples_per_second': 956.457,
     'eval_steps_per_second': 15.018,
     'epoch': 3.0}



### 8. Model Evaluation Summary
To evaluate the effectiveness of different modeling approaches for multiclass sentiment classification on the Climate Change Tweet Dataset, I compared a traditional machine learning model (TF-IDF + Logistic Regression) with a fine-tuned transformer-based model (DistilBERT).

#### Baseline Model: TF-IDF + Logistic Regression
Accuracy: 0.6887

F1 Score (Weighted): 0.6715

The baseline model provides a decent performance using simple text vectorization and a linear classifier. However, it struggles to fully capture contextual nuances in the tweets, especially for more subjective sentiment classes like Pro and Anti.

#### Transformer Model: DistilBERT (Fine-tuned)
Eval Accuracy: 0.7659

Eval F1 Score (Macro): 0.7145

Eval Loss: 0.7461

Epochs: 3

The DistilBERT model significantly outperforms the baseline, achieving over 76% accuracy and a macro F1 score of ~0.71, indicating better balanced performance across all sentiment classes. As a contextual language model, DistilBERT is better able to understand subtle patterns in language and provides a more robust solution to the multiclass sentiment classification task.


```python

```
