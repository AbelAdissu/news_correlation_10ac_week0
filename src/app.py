import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# Title of the app
st.title('News Analysis Dashboard')

# Sidebar for uploading CSV
st.sidebar.header('Upload Your Data')
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write("Data Preview:")
    st.sidebar.write(df.head())
else:
    st.sidebar.write("Awaiting CSV file to be uploaded.")

# Show data overview
if uploaded_file is not None:
    st.subheader('Dataset Overview')
    st.write(df)

    # EDA Section
    st.subheader('Exploratory Data Analysis (EDA)')

    # Top 10 Websites with the Largest Count of News Articles
    top_websites = df['source_name'].value_counts().head(10)
    st.write("Top 10 Websites with the Largest Count of News Articles")
    st.bar_chart(top_websites)

    # Countries with the Highest Number of News Media Organisations
    if 'country' in df.columns:
        top_countries = df['country'].value_counts().head(10)
        st.write("Countries with the Highest Number of News Media Organisations")
        st.bar_chart(top_countries)

    # Distribution of Sentiments
    if 'title_sentiment' in df.columns:
        st.write("Distribution of Sentiments Across Articles")
        sentiment_counts = df['title_sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

    # Word Cloud for Titles
    st.write("Word Cloud for News Titles")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['title'].dropna()))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # TF-IDF Analysis
    st.subheader('TF-IDF Keyword Extraction')
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['title'].dropna())
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    st.write("Top 20 TF-IDF Keywords")
    st.write(tfidf_df.sum().sort_values(ascending=False).head(20))

    # Topic Modeling using Hugging Face's BERTopic
    st.subheader('Topic Modeling')
    if st.button('Run Topic Modeling'):
        topic_modeler = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        topics = df['title'].apply(lambda x: topic_modeler(x, candidate_labels=['Politics', 'World News', 'Technology', 'Business/Finance', 'Science', 'Health', 'Sports', 'Entertainment', 'Environment', 'Crime', 'Education', 'Weather', 'Other'])['labels'][0])
        df['predicted_topic'] = topics
        st.write("Topic Distribution")
        st.bar_chart(df['predicted_topic'].value_counts())
        st.write("Sample of Predicted Topics")
        st.write(df[['title', 'predicted_topic']].head(10))

    # Scatter Plot for Global Ranking vs. News Reporting Frequency
    st.subheader('Scatter Plot: Global Ranking vs. News Reporting Frequency')
    if 'GlobalRank' in df.columns:
        df_grouped = df.groupby('source_name').agg({'GlobalRank': 'mean', 'title': 'count'}).reset_index()
        scatter_fig = px.scatter(df_grouped, x='title', y='GlobalRank', hover_name='source_name', size='title', color='GlobalRank')
        scatter_fig.update_layout(title='Global Ranking vs. News Reporting Frequency', xaxis_title='Number of Reports', yaxis_title='Global Ranking (Lower is Better)')
        st.plotly_chart(scatter_fig)

    # Sentiment Analysis on User Input
    st.subheader('Sentiment Analysis')
    user_input = st.text_input("Enter a news headline or content:")
    if st.button('Analyze Sentiment'):
        sentiment_analyzer = pipeline('sentiment-analysis')
        sentiment = sentiment_analyzer(user_input)[0]
        st.write(f"Sentiment: {sentiment['label']} with a score of {sentiment['score']:.2f}")

# Footer
st.write("Developed by [ABEL ADISSU]")

