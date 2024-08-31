from sklearn.feature_extraction.text import TfidfVectorizer

# Combine title and content into a single text feature for TF-IDF
df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the text data
tfidf_matrix = vectorizer.fit_transform(df['text'])

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
from keybert import KeyBERT

# Initialize KeyBERT
kw_model = KeyBERT()

# Extract keywords for each article
df['keywords'] = df['text'].apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 2), stop_words='english'))
import yake

# Initialize YAKE
yake_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=10)

# Extract keywords for each article
df['yake_keywords'] = df['text'].apply(lambda x: yake_extractor.extract_keywords(x))
from sklearn.metrics.pairwise import cosine_similarity

# Compute TF-IDF vectors for titles and content separately
title_tfidf = vectorizer.fit_transform(df['title'])
content_tfidf = vectorizer.fit_transform(df['content'])

# Calculate cosine similarity between title and content vectors
df['similarity'] = [cosine_similarity(title_tfidf[i], content_tfidf[i])[0][0] for i in range(len(df))]
from bertopic import BERTopic

# Initialize BERTopic
topic_model = BERTopic()

# Fit the model on the text data
topics, probs = topic_model.fit_transform(df['text'])

# Add topics to the DataFrame
df['topic'] = topics

# Get topic info
topic_info = topic_model.get_topic_info()
print(topic_info.head())
import matplotlib.pyplot as plt

# Assuming 'published_at' is in datetime format
df['published_at'] = pd.to_datetime(df['published_at'])

# Group by date and topic to count occurrences
topic_trends = df.groupby([df['published_at'].dt.date, 'topic']).size().unstack().fillna(0)

# Plotting the trends
topic_trends.plot(kind='line', figsize=(15, 8), colormap='tab20')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.title('Topic Trends Over Time')
plt.show()
# Example: Clustering articles around an event using topic modeling
event_cluster = df[df['topic'] == specific_topic_id]  # Replace with the topic ID related to the event
# Group by domain and find the earliest reporting date for each event
earliest_reporting = df.groupby('domain')['published_at'].min()
print(earliest_reporting.sort_values().head())
import mlflow
import mlflow.sklearn

# Start a new MLFlow run
with mlflow.start_run():
    mlflow.log_param("num_topics", len(topic_info))
    mlflow.log_metric("topic_coherence", topic_coherence_score)  # Assume you have a metric for coherence
    mlflow.sklearn.log_model(topic_model, "BERTopic_Model")

    # End the run
    mlflow.end_run()
\  