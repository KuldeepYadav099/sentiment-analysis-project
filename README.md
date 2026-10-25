# Sentiment Analysis Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Decision%20Tree-4CAF50)
![NLP](https://img.shields.io/badge/NLP-Text%20Analytics-FF9800)
![Status](https://img.shields.io/badge/Status-Completed-success)

</div>

A complete beginner-friendly machine learning and NLP project for analyzing customer reviews and predicting whether they are positive, negative, or neutral.

This project uses a dataset of product reviews, cleans the text, converts it into numerical features, and trains a Decision Tree Classifier to classify sentiment automatically.

## Project Preview

> Add your project screenshot here.
>
> Example:
>
> ![Sentiment Analysis Project Screenshot](assets/screenshot.png)

If you want to include a live screenshot later, place the image in an `assets/` folder and replace the placeholder above with your real image.

## Table of Contents

- [About the Project](#about-the-project)
- [Objectives](#objectives)
- [Features](#features)
- [Project Workflow](#project-workflow)
- [How the Code Works](#how-the-code-works)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Run the Project](#run-the-project)
- [Sample Output](#sample-output)
- [Future Improvements](#future-improvements)
- [Author](#author)

## About the Project

Sentiment analysis is a natural language processing task where a model evaluates text and identifies the emotion or opinion behind it. In this project, customer review text is analyzed to determine whether the user opinion is positive, negative, or neutral.

This project is useful for understanding:
- text preprocessing techniques
- stop-word removal
- stemming and normalization
- TF-IDF vectorization
- machine learning model training
- evaluation and visualization

## Objectives

The main goal of this project is to:
- process review text from a dataset
- convert textual data into machine-readable features
- classify sentiment using a trained model
- evaluate the model with standard metrics
- visualize patterns from customer opinions

## Features

- Reads review data from CSV format
- Cleans review text by removing stop words and punctuation
- Applies stemming to reduce words to root form
- Maps product ratings to sentiment labels
- Converts review text into TF-IDF vectors
- Splits data into training and testing sets
- Trains a Decision Tree classifier
- Evaluates using accuracy, precision, and recall
- Creates plots for sentiment distribution and confusion matrix
- Generates a word cloud from positive reviews

## Project Workflow

```text
Review Data
   ↓
Text Cleaning
   ↓
Sentiment Labeling
   ↓
TF-IDF Vectorization
   ↓
Train-Test Split
   ↓
Decision Tree Model
   ↓
Prediction & Evaluation
   ↓
Visualization
```

## How the Code Works

### 1. Importing libraries

```python
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
```

These libraries are used for:
- `pandas` for data handling
- `nltk` for stop words and stemming
- `re` for text cleaning
- `TfidfVectorizer` for text-to-numeric conversion
- `DecisionTreeClassifier` for training the sentiment model
- `matplotlib`, `seaborn`, and `WordCloud` for visual analysis

### 2. Download NLTK resources

```python
nltk.download('stopwords')
```

This downloads the English stopwords list used to remove common useless words like "the", "is", "and", and "on".

### 3. Load dataset

```python
file_path = '/home/virendra/Documents/flipkart_data.csv'
df = pd.read_csv(file_path)
```

The project expects a CSV file containing review content and ratings. The key columns are:
- `review` — product review text
- `rating` — numeric rating given by the customer

### 4. Text preprocessing

```python
def simple_tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    words = simple_tokenize(text)
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)
```

This step:
- converts text to lowercase
- removes punctuation and non-letter characters
- removes stop words
- applies stemming

Example:
- "amazing product" → "amaz product"
- "loved it" → "love"

This helps the model learn meaningful patterns instead of noise.

### 5. Map ratings to sentiment labels

```python
def sentiment_label(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 2
    else:
        return 1
```

The project labels sentiment as:
- `0` → Negative
- `1` → Positive
- `2` → Neutral

### 6. Train/Test split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

The dataset is split so the model learns from training data and is tested on unseen data.

### 7. TF-IDF Vectorization

```python
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_cleaned['review'])
y = df_cleaned['sentiment']
```

TF-IDF converts the cleaned review text into numerical values so a machine learning model can understand it.

### 8. Train the Decision Tree model

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

The model learns from the converted review features and their corresponding sentiment labels.

### 9. Predict and evaluate

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
```

This calculates how well the model is performing.

### 10. Confusion matrix and chart generation

```python
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

The confusion matrix shows how well the model distinguishes between positive, negative, and neutral sentiment.

## Technology Stack

- Python
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- WordCloud

## Installation

1. Clone the repository

```bash
git clone https://github.com/KuldeepYadav099/sentiment-analysis-project.git
cd sentiment-analysis-project
```

2. Install dependencies

```bash
pip install pandas nltk scikit-learn matplotlib seaborn wordcloud
```

3. Download NLTK stopwords

```bash
python -m nltk.downloader stopwords
```

4. Add your dataset

Update the dataset path in `project.py`:

```python
file_path = '/home/virendra/Documents/flipkart_data.csv'
```

Replace it with your local file path or a project dataset location.

## Run the Project

```bash
python project.py
```

After execution, the script will:
- clean the review text
- train the model
- print evaluation metrics
- show charts and visualizations

## Sample Output

```text
Accuracy: 0.82
Precision: 0.81
Recall: 0.80
```

The project also displays:
- sentiment distribution chart
- positive word cloud
- confusion matrix heatmap

## Project Structure

```text
sentiment-analysis-project/
├── project.py
├── README.md
├── assets/
│   └── screenshot.png
└── dataset/
    └── flipkart_data.csv
```

## Results Summary

| Metric | Value |
|--------|-------|
| Model | Decision Tree Classifier |
| Feature Extraction | TF-IDF |
| Accuracy | Depends on dataset |
| Precision | Depends on dataset |
| Recall | Depends on dataset |
| Sentiment Labels | Negative, Neutral, Positive |

## Why This Project Matters

This project demonstrates how basic machine learning can be applied to real-world text data. It is a great example for:
- beginner NLP projects
- sentiment detection tasks
- academic assignments
- portfolio projects
- understanding classification pipelines

## Future Improvements

- Add a Streamlit web app for interactive sentiment prediction
- Try multiple models such as Naive Bayes, Random Forest, and Logistic Regression
- Add support for multilingual review datasets
- Save the trained model for later reuse
- Add a graphical dashboard for better visualization
- Improve preprocessing with lemmatization and text normalization

## Author

Kuldeep Yadav

## License

This project is created for educational and learning purposes.

---

If you want, I can also help you create:
- a more premium GitHub-style README with custom badges and a banner
- a version specifically for college submission
- a portfolio version with a more impressive project summary
- a README that matches popular GitHub project designs
