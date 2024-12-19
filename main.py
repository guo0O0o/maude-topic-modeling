import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)


# Load the TSV file into a DataFrame
df = pd.read_csv('data/incidents_train.csv', index_col=0)

# Group reports by product type and concatenate
grouped_reports = df.groupby('product-category')['text'].apply(' '.join)

# Process grouped texts
grouped_cleaned = grouped_reports.apply(preprocess_text)

# Vectorize and apply topic modeling
vectorizer = TfidfVectorizer(max_features=1000)
'''
X_grouped = vectorizer.fit_transform(grouped_cleaned)
n_topics = 5  # Adjust the number of topics based on your data
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
with open('lda_model.pkl', 'wb') as file:
    pickle.dump(lda, file)'''
with open('lda_model.pkl', 'rb') as file:
    lda = pickle.load(file)
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx + 1}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
