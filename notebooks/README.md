# Exploratory Data Analysis (EDA) for Global News Dataset

## Overview

This Jupyter Notebook provides an Exploratory Data Analysis (EDA) for the Global News Dataset as part of the 10 Academy Intensive Training Week-0 Challenge. The analysis focuses on understanding the metadata, content, and structure of the dataset to extract meaningful insights.

## Dataset Description

The dataset consists of global news articles with the following key attributes:

- **Article ID**: Unique identifier for each article.
- **Source Name**: The name of the news source.
- **Author**: The author of the article.
- **Title**: The title of the article.
- **Description**: A brief description or summary of the article.
- **Content**: The full content of the article (truncated to 200 characters).
- **Published At**: The date and time when the article was published.
- **URL**: The direct link to the article.
- **URL to Image**: Link to the relevant image for the article.
- **Category**: The category under which the article falls.
- **Title Sentiment**: The sentiment of the title, categorized as Positive, Negative, or Neutral.

## Objectives

The EDA aims to achieve the following:

1. **Content Metadata Analysis**:
   - Compare the content metadata across different news websites.
   - Analyze the distribution of content lengths across various sites.
   - Examine the number of words in article titles across different sites.

2. **Sentiment Analysis**:
   - Investigate the sentiment distribution across different domains.
   - Compare the impact of using mean, median, and variance for sentiment analysis.

3. **Geographical Analysis**:
   - Identify the countries with the highest number of news media organizations.
   - Explore the countries that have the most articles written about them.

4. **Traffic Analysis**:
   - Determine the websites with the highest number of visitors.
   - Analyze the correlation between website traffic and the volume of news articles.

## Structure of the Notebook

1. **Data Loading**:
   - Load the Global News Dataset, Domain Location data, and Website Traffic data.
   
2. **Data Cleaning**:
   - Handle missing values and outliers.
   - Ensure data consistency across different datasets.

3. **Content Metadata Analysis**:
   - Compare the length of article content across top domains.
   - Analyze the distribution of title word counts.

4. **Sentiment Analysis**:
   - Group articles by domain and analyze sentiment distribution.
   - Compare sentiment statistics using mean, median, and variance.

5. **Geographical Analysis**:
   - Determine the top countries with the most news media organizations.
   - Analyze the content of articles about specific countries and regions.

6. **Traffic Analysis**:
   - Identify websites with the highest visitor traffic.
   - Correlate traffic data with the volume of articles and sentiment distribution.

## Requirements

The analysis is performed using Python and the following libraries:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib` and `seaborn`: For data visualization.
- `nltk`: For natural language processing, particularly in analyzing text data.

