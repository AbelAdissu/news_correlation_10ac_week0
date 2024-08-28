# Global News Analysis Project

## Overview

The Global News Analysis Project is part of the 10 Academy Intensive Training Week-0 Challenge. The project focuses on analyzing a global news dataset to extract insights, perform sentiment analysis, and categorize news articles based on their content. The project incorporates various Data Science and MLOps techniques, including data ingestion, feature engineering, classification, and model evaluation.

## Project Structure


GlobalNewsAnalysis/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       ├── flake8_check.yml
│       ├── unittests.yml
│       └── docstring_tests.yml
├── data_sets/
│   ├── data_rating.csv
│   ├── domain_location/
│   └── traffic_data/
├── notebooks/
│   ├── parse_slack_data.ipynb
│   └── news_data_EDA.ipynb
├── src/
│   ├── config.py
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── custom_exception.py
│   ├── logger.py
│   ├── __init__.py
│   ├── utils.py
│   └── loader.py
├── tests/
│   ├── __init__.py
│   └── test_feature_engineering.py
├── logs/
│   └── (log files)
├── artifacts/
│   ├── (processed files)
├── .gitignore
├── .flake8
├── .pre-commit-config.yaml
├── setup.cfg
├── Makefile
├── pyproject.toml
├── requirements.txt
├── style_guide.md
├── README.md
└── view_tree.py




## Objectives

The primary objectives of this project are:

1. **Data Ingestion:** Collect and clean data from various sources, including domain information, traffic data, and article ratings.
2. **Exploratory Data Analysis (EDA):** Analyze the data to uncover patterns, trends, and insights.
3. **Feature Engineering:** Develop and select features for model training.
4. **Classification and Prediction:** Categorize news articles into predefined categories such as Breaking News, Politics, World News, and more.
5. **Sentiment Analysis:** Analyze the sentiment of news headlines and content.
6. **MLOps:** Implement MLOps practices, including model versioning, CI/CD, and Dockerization.
7. **Model Evaluation:** Evaluate the performance of the models and fine-tune them for better accuracy.

## Installation

### Prerequisites

- Python 3.8+
- pip
- Git




