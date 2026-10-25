# Sentiment Analysis Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Decision%20Tree-green)
![NLP](https://img.shields.io/badge/NLP-Text%20Preprocessing-orange)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

</div>

A machine learning project that analyzes customer reviews and classifies them as positive, negative, or neutral based on their text content and rating.

This project is a beginner-friendly NLP-based sentiment analysis pipeline built using Python, Pandas, NLTK, Scikit-learn, and visual libraries.

## Project Demo / Screenshot

> Add your project screenshot here.
>
> Example:
>
> ![Sentiment Analysis Dashboard](assets/screenshot.png)

If you want to show a real screenshot later, place the image inside an `assets/` folder and update the path above.

## Overview

Sentiment analysis is the process of determining whether a piece of text expresses a positive, negative, or neutral opinion. In this project, we use product review data to train a model that learns patterns in customer feedback and predicts sentiment automatically.

This project helps in understanding:
- how review text is cleaned and prepared for analysis
- how sentiment is mapped from rating values
- how text is converted into numeric form using TF-IDF
- how a machine learning model is trained and evaluated
- how to visualize prediction performance and review patterns

## Features

- Loads review data from a CSV file
- Cleans text by removing stop words and applying stemming
- Maps ratings to sentiment labels
- Converts text to TF-IDF vectors
- Trains a Decision Tree Classifier
- Evaluates the model with accuracy, precision, and recall
- Displays charts for sentiment distribution and confusion matrix
- Generates a word cloud for positive reviews

## What is inside this project?

The repository contains the main implementation file:

- `project.py` — contains the full sentiment analysis workflow
- `README.md` — project documentation and explanation
- `dataset/flipkart_data.csv` — review dataset (add this file if required)

## How the project works

The workflow is simple and follows a classic NLP pipeline:

1. Load the review dataset
2. Clean the review text
3. Map review ratings to sentiment labels
4. Convert text into numerical features using TF-IDF
5. Split data into training and testing sets
6. Train a Decision Tree classifier
7. Predict sentiment for unseen reviews
8. Measure performance and visualize results

## Step-by-step code explanation

### 1. Import required libraries

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

These libraries help us:
- load and process data (`pandas`)
- clean text (`nltk`, `re`)
- convert text to vectors (`TfidfVectorizer`)
- train a model (`DecisionTreeClassifier`)
- evaluate model quality (`accuracy_score`, `precision_score`, `recall_score`)
- build visual outputs (`matplotlib`, `seaborn`, `WordCloud`)

### 2. Download NLTK stopwords

```python
nltk.download('stopwords')
```

This downloads a built-in list of common English words such as "the", "is", "and", etc., which are often removed during text cleaning.

### 3. Load the dataset

```python
file_path = '/home/virendra/Documents/flipkart_data.csv'
df = pd.read_csv(file_path)
```

The project expects a CSV file with at least two important columns:
- `review` → the text written by the customer
- `rating` → the star rating of the product

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

This step does the following:
- converts text to lowercase
- removes punctuation and non-letter characters
- removes common stop words
- reduces words to their root form using stemming

Example:
- "better" → "better"
- "loved" → "love"
- "amazing" → "amaz"

This makes the model more consistent when analyzing text.

### 5. Label the sentiment

```python
def sentiment_label(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 2
    else:
        return 1
```

This maps the rating values into sentiment classes:
- `0` → Negative
- `1` → Positive
- `2` → Neutral

### 6. Apply preprocessing and sentiment labels

```python
df_cleaned = preprocess_reviews_stopwords(df)
```

This updates the review column with cleaned text and adds a new `sentiment` column based on the rating.

### 7. Visualize sentiment distribution

```python
sentiment_counts = df_cleaned['sentiment'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color=['red', 'green', 'blue'])
plt.title('Sentiment Distribution (0: Negative, 1: Positive, 2: Neutral)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```

This chart helps understand how many reviews are positive, negative, or neutral.

### 8. Generate a word cloud for positive reviews

```python
positive_reviews = df_cleaned[df_cleaned['sentiment'] == 1]['review']
positive_text = ' '.join(positive_reviews)
wordcloud = WordCloud(width=800, height=400).generate(positive_text)
plt.figure(figsize=(8,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Reviews')
plt.show()
```

The word cloud highlights the words that appear most often in positive reviews.

### 9. Convert text to TF-IDF features

```python
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_cleaned['review'])
y = df_cleaned['sentiment']
```

TF-IDF converts text into a matrix of numbers that machine learning algorithms can understand.

### 10. Train/Test split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

This splits the data so the model can be trained on one set and evaluated on another.

### 11. Train the model

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

A Decision Tree classifier learns patterns from the training reviews and uses them to classify future reviews.

### 12. Evaluate the model

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
```

This gives a numerical measure of how well the model performs.

### 13. Confusion matrix

```python
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

A confusion matrix shows how many predictions were correct or incorrect across each sentiment class.

## Tech Stack

- Python
- Pandas
- NLTK
- scikit-learn
- Matplotlib
- Seaborn
- WordCloud

## Installation

1. Clone the repository

```bash
git clone https://github.com/KuldeepYadav099/sentiment-analysis-project.git
cd sentiment-analysis-project
```

2. Install the required packages

```bash
pip install pandas nltk scikit-learn matplotlib seaborn wordcloud
```

3. Download NLTK resources

```bash
python -m nltk.downloader stopwords
```

4. Add your dataset file

The script currently expects a dataset path like:

```python
file_path = '/home/virendra/Documents/flipkart_data.csv'
```

Update it to point to your own CSV file, for example:

```python
file_path = 'dataset/flipkart_data.csv'
```

## Run the project

```bash
python project.py
```

When the script runs, it will:
- preprocess the review text
- create sentiment labels
- train the model
- print evaluation metrics
- show sentiment plots and the confusion matrix

## Project output

The model prints metrics like:

```python
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
```

Typical output includes:
- Accuracy value
- Precision value
- Recall value
- Graphical charts for analysis

## Why this project is useful

This project is a good starting point for learning:
- text preprocessing
- feature extraction from reviews
- supervised machine learning
- model evaluation
- data visualization in NLP

It is especially useful for students and beginners who want to build their first NLP or sentiment analysis project.

## Future improvements

- Add a Flask or Streamlit web app
- Use more advanced models like Naive Bayes, Logistic Regression, or Random Forest
- Add real-time single-review prediction
- Save the trained model using joblib or pickle
- Add support for larger datasets and better preprocessing

## Author

Kuldeep Yadav

## License

This project is for educational and learning purposes.

---

If you want, I can also make this README even more advanced by adding:
- a project architecture diagram
- a results table
- a professional college-project version
- a GitHub-style premium README with badges and better formatting
