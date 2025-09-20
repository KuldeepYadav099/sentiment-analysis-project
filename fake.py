import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

nltk.download('stopwords')

df = pd.read_csv('/home/virendra/Documents/flipkart_data.csv')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def simple_tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def clean_text(text):
    words = simple_tokenize(text)
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

df['review'] = df['review'].astype(str).apply(clean_text)

def sentiment_label(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 2
    else:
        return 1

df['sentiment'] = df['rating'].apply(sentiment_label)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
