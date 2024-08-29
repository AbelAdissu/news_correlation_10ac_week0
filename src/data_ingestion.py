import os
import sys
from custom_exeption import NewsAnalysisException
from logger import logging
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import re
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "merged_data.csv")
    stratified_kfold_path: str = os.path.join("artifacts", "stratified_kfold.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Data ingestion config object

    def remove_duplicates(self, df):
        logging.info("Removing duplicate articles from the dataset")
        df.drop_duplicates(subset=['url'], keep='first', inplace=True)
        logging.info(f"Dataset size after removing duplicates: {df.shape}")
        return df

    def extract_domain(self, url):
        pattern = r'https?://(?:www\.)?([^/]+)'
        matches = re.findall(pattern, url)
        return matches[0] if matches else None

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load the datasets
            df_rating = pd.read_csv(r"C:\Users\user\3D Objects\news_correlation_10ac_week0\data_sets\data_rating.csv\rating.csv")
            df_domain_location = pd.read_csv(r"C:\Users\user\3D Objects\news_correlation_10ac_week0\data_sets\domain_location\domains_location.csv")
            df_traffic = pd.read_csv(r"C:\Users\user\3D Objects\news_correlation_10ac_week0\data_sets\traffic_data\traffic.csv")
            
            # Rename columns for consistency
            df_domain_location.rename(columns={"SourceCommonName":"domain"}, inplace=True)
            df_traffic.rename(columns={"Domain":"domain"}, inplace=True)

            # Extract domain from the URL in df_rating
            df_rating['domain'] = df_rating['url'].apply(self.extract_domain)

            # Merge the datasets
            df_merge = pd.merge(df_rating, df_traffic, on="domain", how="inner")
            df_merge = pd.merge(df_merge, df_domain_location, on="domain", how="inner")
            logging.info("Merged all three datasets")

            # Remove duplicates
            df_merge = self.remove_duplicates(df_merge)

            # Reset the index to ensure default integer indexing
            df_merge.reset_index(drop=True, inplace=True)

            # Save the merged data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df_merge.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Saved the merged data at {self.ingestion_config.raw_data_path}")

            # Stratified K-Fold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            df_merge['kfold'] = -1  # Initialize kfold column

            for fold, (train_index, test_index) in enumerate(skf.split(df_merge, df_merge['title_sentiment'])):
                df_merge.loc[test_index, 'kfold'] = fold
            logging.info("Performed Stratified K-Fold splitting")

            # Save the Stratified K-Fold data
            df_merge.to_csv(self.ingestion_config.stratified_kfold_path, index=False)
            logging.info(f"Saved the Stratified K-Fold data at {self.ingestion_config.stratified_kfold_path}")

            # Optionally save train and test splits for the first fold
            train_df = df_merge[df_merge['kfold'] != 0].reset_index(drop=True)
            test_df = df_merge[df_merge['kfold'] == 0].reset_index(drop=True)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise NewsAnalysisException(e, sys)

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()
