# Sentiment-Analysis
# Sentiment Analysis of Movie Reviews

This project is a machine learning model designed to predict the sentiment of a movie review as either **positive** or **negative**. The model uses Logistic Regression to perform binary classification on a dataset of 50,000 movie reviews from IMDB.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Sentiment analysis is a natural language processing (NLP) technique that determines whether a given piece of text conveys a positive, negative, or neutral sentiment. In this project, we focus on **binary sentiment classification** (positive or negative) using a dataset of IMDB movie reviews.

The project can help businesses, especially in the entertainment industry, understand customer opinions based on their reviews.

## Features
- Classifies movie reviews as **positive** or **negative**.
- Utilizes **Logistic Regression** for binary classification.
- Text preprocessing techniques include tokenization, stop-word removal, and TF-IDF vectorization.
- Performance evaluation through accuracy and confusion matrix.

## Dataset

The dataset used for this project consists of 50,000 movie reviews from the IMDB dataset. The two columns in the dataset are:
- **Review**: The movie review (text).
- **Sentiment**: The sentiment label (either `positive` or `negative`).

You can download the dataset from [this link](https://ai.stanford.edu/~amaas/data/sentiment/).

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SyedSubhan12/Sentiment-Analysis.git
   cd Sentiment-Analysis
