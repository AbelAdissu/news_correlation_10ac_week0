import os
import sys
import pandas as pd
import spacy
from custom_exeption import NewsAnalysisException
from logger import logging
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")

@dataclass
class FeatureEngineeringConfig:
    processed_data_path: str = os.path.join("artifacts", "merged_data.csv")
    transformed_data_path: str = os.path.join("artifacts", "transformed_data.csv")

class FeatureEngineering:
    def __init__(self):
        self.config = FeatureEngineeringConfig()
    
    def remove_stopwords(self, text):
        doc = nlp(text)
        filtered_text = [token.text for token in doc if not token.is_stop]
        return ' '.join(filtered_text)

    def stem_words(self, text):
        doc = nlp(text)
        stemmed_words = [token.lemma_ for token in doc]  # Using lemma_ for lemmatization instead of stemming
        return ' '.join(stemmed_words)

    def calculate_tfidf(self, df, column_name):
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(df[column_name])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        return tfidf_df

    def encode_labels(self, df, column_name):
        le = LabelEncoder()
        df[column_name] = le.fit_transform(df[column_name])
        return df

    def initiate_feature_engineering(self):
        logging.info("Started the feature engineering process.")
        try:
            # Load the dataset
            logging.info(f"Loading dataset from {self.config.processed_data_path}")
            if not os.path.exists(self.config.processed_data_path):
                logging.error(f"File not found: {self.config.processed_data_path}")
                return

            df = pd.read_csv(self.config.processed_data_path)
            logging.info(f"Dataset loaded successfully with shape: {df.shape}")

            # Apply text processing
            logging.info("Starting text preprocessing (stopwords removal and lemmatization).")
            df['cleaned_content'] = df['content'].apply(self.remove_stopwords)
            df['cleaned_content'] = df['cleaned_content'].apply(self.stem_words)
            logging.info("Text preprocessing completed.")

            # Calculate TF-IDF for content
            logging.info("Calculating TF-IDF features.")
            tfidf_df = self.calculate_tfidf(df, 'cleaned_content')
            logging.info(f"TF-IDF calculation completed with shape: {tfidf_df.shape}")

            # Encode categorical labels (e.g., title_sentiment)
            logging.info("Encoding categorical labels (e.g., title_sentiment).")
            df = self.encode_labels(df, 'title_sentiment')
            logging.info("Label encoding completed.")

            # Combine the TF-IDF features with the original dataframe
            logging.info("Combining TF-IDF features with the original dataframe.")
            df = pd.concat([df, tfidf_df], axis=1)
            logging.info(f"Dataframe combined with shape: {df.shape}")

            # Save the processed data
            transformed_data_path = self.config.transformed_data_path
            logging.info(f"Saving transformed data to {transformed_data_path}")
            os.makedirs(os.path.dirname(transformed_data_path), exist_ok=True)
            df.to_csv(transformed_data_path, index=False)
            logging.info(f"Transformed data saved successfully at {transformed_data_path}")

        except Exception as e:
            logging.error("An error occurred during the feature engineering process.")
            raise NewsAnalysisException(e, sys)

if __name__ == "__main__":
    feature_engineering = FeatureEngineering()
    feature_engineering.initiate_feature_engineering()
