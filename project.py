# Importing Libraries and Dataset

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

# Download NLTK resources once
nltk.download('stopwords')

# Load the dataset
file_path = '/home/virendra/Documents/flipkart_data.csv'
df = pd.read_csv(file_path)

#Preprocessing Setup
ps = PorterStemmer()    #i have used this to perform word stemming
stop_words = set(stopwords.words('english'))

def simple_tokenize(text):
    # Tokenize by extracting words only (letters )
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def clean_text(text):
    words = simple_tokenize(text)
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def preprocess_reviews_stopwords(df):
    df['review'] = df['review'].astype(str).apply(clean_text)
    # Label sentiment: 0=negative(1,2), 2=neutral(3), 1=positive(4,5)
    def sentiment_label(rating):
        if rating <= 2:
            return 0
        elif rating == 3:
            return 2
        else:
            return 1
    df['sentiment'] = df['rating'].apply(sentiment_label)
    return df

df_cleaned = preprocess_reviews_stopwords(df)

# Visualize Sentiment Distribution
sentiment_counts = df_cleaned['sentiment'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color=['red', 'green', 'blue'])
plt.title('Sentiment Distribution (0: Negative, 1: Positive, 2: Neutral)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(ticks=[0,1,2], labels=['Negative', 'Positive', 'Neutral'], rotation=0)
plt.show()

# Word Cloud of Positive Reviews
positive_reviews = df_cleaned[df_cleaned['sentiment'] == 1]['review']
positive_text = ' '.join(positive_reviews)
wordcloud = WordCloud(width=800, height=400).generate(positive_text)
plt.figure(figsize=(8,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Reviews')
plt.show()

# Vectorization of cleaned review text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_cleaned['review'])
y = df_cleaned['sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("Accuracy:", accuracy)

# Precision and Recall
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
print("Precision:", precision)
print("Recall:", recall)
