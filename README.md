# Sentiment Analysis Project

<div align="center">

<img src="https://images.unsplash.com/photo-1555949963-aa79dcee981c?auto=format&fit=crop&w=1200&q=80" alt="Sentiment Analysis" width="100%" />

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-FF6F61?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine%20Learning-Decision%20Tree-4CAF50?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

A machine learning and natural language processing project that detects the sentiment of customer reviews using text classification techniques.

This project reads product review data, cleans the text, converts it into features using TF-IDF, and trains a Decision Tree classifier to classify reviews as positive, negative, or neutral.

## Overview

Sentiment analysis is the process of identifying emotions, opinions, and attitudes from text. In this project, we apply that concept to customer reviews to determine whether feedback is positive, negative, or neutral.

The repository demonstrates a complete workflow for:
- loading review data
- preprocessing raw text
- mapping ratings to sentiment labels
- turning text into numeric vectors
- training a machine learning classifier
- evaluating the model
- visualizing the results

## Why This Project?

This project is a strong beginner-friendly example of how NLP and machine learning can be applied to real-world data. It is useful for:
- learning text preprocessing
- understanding sentiment classification
- exploring feature extraction with TF-IDF
- building a first machine learning project
- presenting a project for academic or portfolio purposes

## Project Highlights

- Text preprocessing with stop-word removal and stemming
- Rating-to-sentiment conversion logic
- TF-IDF feature generation
- Decision Tree classification model
- Accuracy, precision, and recall evaluation
- Visual analysis using plotting and word cloud generation

## Project Structure

```text
sentiment-analysis-project/
├── project.py
├── README.md
├── assets/
│   └── screenshot.png
├── dataset/
│   └── flipkart_data.csv
└── requirements.txt   (optional if added later)
```

## What Is Inside This Project?

- `project.py` — contains the complete sentiment analysis pipeline
- `README.md` — project documentation and explanation
- `dataset/flipkart_data.csv` — review dataset used for model training
- `assets/` — optional folder for screenshots or project images

## How It Works

The workflow is simple and follows a standard NLP machine learning pipeline:

1. Load the review dataset
2. Clean the review text
3. Assign sentiment labels based on ratings
4. Convert text to numerical vectors using TF-IDF
5. Split data into training and testing sets
6. Train a Decision Tree classifier
7. Predict sentiment for the test data
8. Evaluate the model
9. Visualize the outcome

## Step-by-Step Code Explanation

### 1. Importing the required libraries

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

These libraries are responsible for:
- loading and managing data (`pandas`)
- cleaning text (`nltk`, `re`)
- converting text into vectors (`TfidfVectorizer`)
- building the predictive model (`DecisionTreeClassifier`)
- evaluating performance (`accuracy_score`, `precision_score`, `recall_score`)
- generating charts and word clouds (`matplotlib`, `seaborn`, `WordCloud`)

### 2. Download NLTK stopwords

```python
nltk.download('stopwords')
```

This ensures the English stopword list is available for text cleaning.

### 3. Read the review dataset

```python
file_path = '/home/virendra/Documents/flipkart_data.csv'
df = pd.read_csv(file_path)
```

The dataset is expected to contain columns like:
- `review` — customer review text
- `rating` — numeric rating value

### 4. Text cleaning and preprocessing

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
- converts everything to lowercase
- removes punctuation and numbers
- removes common stop words
- applies stemming to normalize words

Example:
- `loved` → `love`
- `amazing` → `amaz`
- `very good product` → `good product`

### 5. Sentiment labeling based on rating

```python
def sentiment_label(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 2
    else:
        return 1
```

This maps the score into classes:
- `0` → Negative
- `1` → Positive
- `2` → Neutral

### 6. Vectorization using TF-IDF

```python
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_cleaned['review'])
y = df_cleaned['sentiment']
```

TF-IDF converts the text review into a numeric matrix which can be understood by machine learning algorithms.

### 7. Train-test split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

This divides the dataset so the model is trained on one subset and evaluated on another unseen subset.

### 8. Train the Decision Tree model

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

The model learns patterns from the review features and their sentiment labels.

### 9. Prediction and evaluation

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
```

The model output includes evaluation metrics such as:
- Accuracy
- Precision
- Recall

### 10. Visualizations

```python
sentiment_counts = df_cleaned['sentiment'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color=['red', 'green', 'blue'])
plt.title('Sentiment Distribution (0: Negative, 1: Positive, 2: Neutral)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```

This creates a chart showing the count of each sentiment class.

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

This creates a word cloud that highlights the most common words in positive reviews.

```python
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

This helps understand the model’s classification quality for each sentiment class.

## Tech Stack

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

2. Install the dependencies

```bash
pip install pandas nltk scikit-learn matplotlib seaborn wordcloud
```

3. Download NLTK resources

```bash
python -m nltk.downloader stopwords
```

4. Add your dataset

The script currently uses a local absolute path:

```python
file_path = '/home/virendra/Documents/flipkart_data.csv'
```

Update it to the correct dataset file path, for example:

```python
file_path = 'dataset/flipkart_data.csv'
```

## Run the Project

```bash
python project.py
```

The project will:
- clean the text
- build sentiment labels
- train the classifier
- print performance metrics
- display charts and visualizations

## Sample Output

```text
Accuracy: 0.82
Precision: 0.81
Recall: 0.80
```

The output may vary depending on your dataset and model behavior.

## Results and Interpretation

This project is a demonstration of how a simple machine learning pipeline can classify sentiment in review text.

A typical result would show:
- how many reviews are positive, negative, or neutral
- which words appear most often in positive feedback
- how accurately the model predicts sentiment on unseen reviews

## Project Purpose

This project is especially useful for:
- academic assignment work
- NLP learning projects
- machine learning portfolio work
- understanding review classification methods

## Future Enhancements

- Add a Streamlit web application for live sentiment prediction
- Compare multiple models such as Naive Bayes, Logistic Regression, and Random Forest
- Add lemmatization for better text normalization
- Save the trained model using `pickle` or `joblib`
- Add a dashboard showing sentiment distribution in real time
- Support multilingual datasets and broader NLP preprocessing

## Author

Kuldeep Yadav

## License

This project is intended for educational and learning purposes.

---

If you want, I can also make one final version that is even more polished for a college submission or portfolio by adding:
- a detailed abstract
- a problem statement
- methodology section
- result interpretation section
- project conclusion
