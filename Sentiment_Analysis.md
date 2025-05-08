```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import html
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

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





  <div id="df-3b2c8e75-3889-44e2-81d2-c52017e6f54e" class="colab-df-container">
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
      <th>sentiment</th>
      <th>tweetid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>43943.000000</td>
      <td>4.394300e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.853924</td>
      <td>8.367966e+17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.853543</td>
      <td>8.568506e+16</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.000000</td>
      <td>5.926334e+17</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>7.970376e+17</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>8.402301e+17</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>9.020003e+17</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000000</td>
      <td>9.667024e+17</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3b2c8e75-3889-44e2-81d2-c52017e6f54e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3b2c8e75-3889-44e2-81d2-c52017e6f54e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3b2c8e75-3889-44e2-81d2-c52017e6f54e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-ff592861-8e45-4aad-b4c1-dd1206f6c714">
      <button class="colab-df-quickchart" onclick="quickchart('df-ff592861-8e45-4aad-b4c1-dd1206f6c714')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-ff592861-8e45-4aad-b4c1-dd1206f6c714 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




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
    


## Data preprocessing
Data preprocessing is a crucial step for NLP tasks. In this step, I clean the tweet text by removing URLs, mentions, hashtags, and any non-alphabetic characters. I also convert all text to lowercase and strip any extra spaces. This ensures that the text is ready for tokenization and model input.


```python
df['message'].head(5)
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@tiniebeany climate change is an interesting hustle as it was global warming but the planet stopped warming for 15 yes while the suv boom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RT @NatGeoChannel: Watch #BeforeTheFlood right here, as @LeoDiCaprio travels the world to tackle climate change https://t.co/LkDehj3tNn httÃ¢â‚¬Â¦</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fabulous! Leonardo #DiCaprio's film on #climate change is brilliant!!! Do watch. https://t.co/7rV6BrmxjW via @youtube</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RT @Mick_Fanning: Just watched this amazing documentary by leonardodicaprio on climate change. We all think thisÃ¢â‚¬Â¦ https://t.co/kNSTE8K8im</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RT @cnalive: Pranita Biswasi, a Lutheran from Odisha, gives testimony on effects of climate change &amp;amp; natural disasters on the poÃ¢â‚¬Â¦</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>




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



### Encoding the Labels
Since machine learning models require numerical inputs, I encode the sentiment labels using LabelEncoder. This transforms the four sentiment labels into numeric values (0, 1, 2, 3) so they can be used as target labels for training the model.


```python
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])  # -1 to 2 mapped to 0 to 3
```

### Train-Test Split
To evaluate the model’s performance effectively, I split the dataset into training and testing sets using an 80/20 split. The training set is used to train the model, and the test set is used to evaluate the model’s ability to generalize to unseen data. I ensure the split maintains the class distribution by using stratify.


```python
from sklearn.model_selection import train_test_split

X = df['clean_tweet']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

```

### Baseline Model - TF-IDF + Logistic Regression
As a baseline model, I use a combination of TF-IDF (Term Frequency-Inverse Document Frequency) and Logistic Regression. First, I vectorize the tweet text using TfidfVectorizer, which transforms the text into numerical feature vectors. Then, I train a Logistic Regression model on the transformed features and evaluate its performance on the test set. I calculate metrics such as accuracy, F1-score, and present a confusion matrix to assess its classification performance.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

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
    


### Fine-tune Transformer Model (DistilBERT)
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

    /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    


    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/483 [00:00<?, ?B/s]



    Map:   0%|          | 0/35154 [00:00<?, ? examples/s]



    Map:   0%|          | 0/8789 [00:00<?, ? examples/s]



```python
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=4
)

```

    Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
    WARNING:huggingface_hub.file_download:Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
    


    model.safetensors:   0%|          | 0.00/268M [00:00<?, ?B/s]


    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    


```python
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding
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

    [34m[1mwandb[0m: [33mWARNING[0m The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.
    


    <IPython.core.display.Javascript object>


    [34m[1mwandb[0m: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
    [34m[1mwandb[0m: You can find your API key in your browser here: https://wandb.ai/authorize?ref=models
    wandb: Paste an API key from your profile and hit enter:

     ··········
    

    [34m[1mwandb[0m: [33mWARNING[0m If you're specifying your api key in code, ensure this code is not shared publicly.
    [34m[1mwandb[0m: [33mWARNING[0m Consider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.
    [34m[1mwandb[0m: No netrc file found, creating one.
    [34m[1mwandb[0m: Appending key for api.wandb.ai to your netrc file: /root/.netrc
    [34m[1mwandb[0m: Currently logged in as: [33mpranithachilvari1234[0m ([33mpranithachilvari1234-university-of-arizona[0m) to [32mhttps://api.wandb.ai[0m. Use [1m`wandb login --relogin`[0m to force relogin
    


Tracking run with wandb version 0.19.10



Run data is saved locally in <code>/content/wandb/run-20250508_031821-asftvu9x</code>



Syncing run <strong><a href='https://wandb.ai/pranithachilvari1234-university-of-arizona/huggingface/runs/asftvu9x' target="_blank">./results</a></strong> to <a href='https://wandb.ai/pranithachilvari1234-university-of-arizona/huggingface' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/pranithachilvari1234-university-of-arizona/huggingface' target="_blank">https://wandb.ai/pranithachilvari1234-university-of-arizona/huggingface</a>



View run at <a href='https://wandb.ai/pranithachilvari1234-university-of-arizona/huggingface/runs/asftvu9x' target="_blank">https://wandb.ai/pranithachilvari1234-university-of-arizona/huggingface/runs/asftvu9x</a>




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



### Model Evaluation Summary
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
