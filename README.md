# CreditWise Loan System

An end-to-end supervised machine learning pipeline that predicts loan approval outcomes based on applicant data, using classification models trained and evaluated on real-world style data.

## Overview

This project walks through the full ML workflow — from raw data to evaluated models — for a binary classification problem: **will a loan application be approved or not?**

## What's inside

- `credit_wise.ipynb` — the full notebook: EDA, preprocessing, model training, and evaluation
- `loan_approval_data.csv` — the dataset used for training and testing

## Workflow

1. **Handling missing data** — identified and treated null/missing values in the dataset
2. **Exploratory Data Analysis (EDA)** — examined distributions, relationships, and patterns in the data before modeling
3. **Feature encoding** — converted categorical variables into a numeric format models can use
4. **Feature scaling** — normalized numeric features for distance-based models like KNN
5. **Model training** — trained three different classification models:
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - Naive Bayes
6. **Model evaluation** — compared models using:
   - Precision
   - Recall
   - F1 Score

## Why three models?

Rather than relying on a single algorithm, this project compares KNN, Logistic Regression, and Naive Bayes to understand trade-offs between them — e.g., how a distance-based model (KNN) performs versus a probabilistic model (Naive Bayes) versus a linear model (Logistic Regression) on the same loan approval task.

## Tech stack

- Python
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- matplotlib / seaborn (for EDA visualizations)

## How to run

1. Clone this repo
2. Install dependencies:
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
3. Open `credit_wise.ipynb` in Jupyter and run all cells
