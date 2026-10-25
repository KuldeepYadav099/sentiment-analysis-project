# Sentiment Analysis Project

A simple sentiment analysis project that classifies product reviews into positive, negative, or neutral categories using a machine learning model.

This project reads a dataset of reviews, cleans the text, converts it into numerical features using TF-IDF, and trains a Decision Tree Classifier to predict sentiment.

## Project Overview

This project is built to:
- load customer review data from a CSV file
- clean and normalize the review text
- label the sentiment based on the product rating
- visualize the distribution of sentiments
- build a word cloud for positive reviews
- train a text classification model
- evaluate the model using accuracy, precision, and recall

## Tech Stack

- Python
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- WordCloud

## Project Structure

```text
sentiment-analysis-project/
├── project.py
├── README.md
└── dataset/
    └── flipkart_data.csv
```

## What is in this project?

The project contains a single main Python file:

- `project.py` – contains all the logic for data loading, preprocessing, sentiment labeling, visualizations, model training, and evaluation.

## How it works

### 1. Data loading
The script loads a CSV file containing review text and rating values.

```python
file_path = '/home/virendra/Documents/flipkart_data.csv'
df = pd.read_csv(file_path)
```

This dataset is expected to have at least these columns:
- `review` – the product review text
- `rating` – the rating given by the customer

### 2. Text preprocessing
Before training the model, the review text is cleaned:
- converts to lowercase
- removes stop words
- performs stemming using PorterStemmer

```python
def clean_text(text):
    words = simple_tokenize(text)
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)
```

This helps reduce noise and makes similar words more consistent.

### 3. Sentiment labeling
The rating is converted into a sentiment label:
- 0 = Negative
- 1 = Positive
- 2 = Neutral

```python
def sentiment_label(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 2
    else:
        return 1
```

### 4. TF-IDF Vectorization
The cleaned review text is transformed into numeric form using TF-IDF so the machine learning model can understand it.

```python
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_cleaned['review'])
y = df_cleaned['sentiment']
```

### 5. Train/Test split
The dataset is split into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 6. Model training
A Decision Tree Classifier is used to train the sentiment model.

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

### 7. Prediction and evaluation
The trained model predicts the sentiment of test reviews and then calculates:
- Accuracy
- Precision
- Recall
- Confusion Matrix

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
```

## Data Visualization Included

The project also creates visual outputs such as:
- sentiment distribution bar chart
- positive review word cloud
- confusion matrix heatmap

These help in understanding the dataset and checking model performance visually.

## Example of the code used in the project

```python
# Preprocessing Setup
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Clean review text
def clean_text(text):
    words = simple_tokenize(text)
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Label sentiment
if rating <= 2:
    return 0
elif rating == 3:
    return 2
else:
    return 1
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/KuldeepYadav099/sentiment-analysis-project.git
cd sentiment-analysis-project
```

2. Install required packages
```bash
pip install pandas nltk scikit-learn matplotlib seaborn wordcloud
```

3. Download NLTK stopwords
```bash
python -m nltk.downloader stopwords
```

4. Place your dataset file in the project and update the path in `project.py`:
```python
file_path = '/home/virendra/Documents/flipkart_data.csv'
```

Replace it with your local file path, for example:
```python
file_path = 'dataset/flipkart_data.csv'
```

## How to run

```bash
python project.py
```

After running, the script will:
- load the data
- clean the reviews
- train the model
- print evaluation metrics
- display charts for sentiment distribution and confusion matrix

## Model summary

This project uses a classic machine learning pipeline:

Text data -> Clean text -> TF-IDF -> Decision Tree -> Sentiment prediction

This is a good beginner-friendly project for learning:
- text preprocessing
- sentiment analysis
- feature extraction
- model evaluation
- data visualization

## Important note

This project currently uses an absolute dataset path, which may only work on the original machine where the CSV file exists. For easier use on another computer, update the `file_path` to a relative path or add the dataset file inside the repository.

## Future improvements

- Add a web UI using Streamlit or Flask
- Use more advanced models like Logistic Regression, Random Forest, or Naive Bayes
- Add a real-time input feature for single review prediction
- Save the trained model for later use
- Add support for multilingual reviews

## License

This project is for educational and learning purposes.

## Author

Kuldeep Yadav

---

If you want, I can also help you create a more polished version of this README with:
- a project banner image
- a screenshot section
- badges
- a complete GitHub-style project description
- a more professional academic presentation version
