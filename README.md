# Fake Job Detection Using NLP

A Natural Language Processing (NLP) project that classifies job postings as **Real** or **Fake** using TF-IDF Vectorization and Logistic Regression.


## Overview

This project analyzes job descriptions and uses machine learning to detect potentially fraudulent job postings.

The workflow includes:

- Text preprocessing
- Stopword removal
- TF-IDF Vectorization
- Logistic Regression
- Model evaluation using Accuracy, Precision, Recall, and F1 Score


## Dataset

The dataset contains **3000 job postings** with **25 features**, including:

- Job Title
- Job Description
- Requirements
- Company Information
- Employment Type
- Industry
- Location
- Salary Range
- Fraud Indicator 


 0 - Real Job 
 1 - Fake Job 


## Tools Used

- Python
- Pandas
- NLTK
- Regular Expressions 
- Scikit-Learn
- Matplotlib
- Seaborn



### 1. Text Cleaning

- Converted all text to lowercase
- Removed special characters and punctuation using Regex
- Removed English stopwords using NLTK

### 2. TF-IDF Vectorization

Converted text data into numerical features and the final TF-IDF matrix contained:

3000 Documents

17 Text Features


### 3. Train-Test Split

```text
Training Data: 80%
Testing Data: 20%
```

Result:

```text
Training Samples: 
