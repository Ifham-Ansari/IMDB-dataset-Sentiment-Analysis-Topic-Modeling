
# Sentiment Analysis and Topic Modeling on IMDB 50k Reviews

## Overview

This repository contains multiple projects focused on sentiment analysis and topic modeling using machine learning (ML), deep learning (DL), and natural language processing (NLP) techniques. The IMDB 50k reviews dataset is utilized to explore, preprocess, analyze, and derive meaningful insights about movie reviews.

## Repository Structure

The repository is organized as follows:

1. **`sentiment_ML.ipynb`**  
   - **Description:**  
     Performs sentiment analysis on the IMDB 50k reviews dataset using:  
       - Naive Bayes (NB)  
       - Logistic Regression (LR)  
     - **Key Features:**  
       - Comparison of metrics like accuracy, precision, recall, and F1 score for both models.  
       - Insights into model performance for classification tasks.
   **colab link:** https://colab.research.google.com/github/Ifham-Ansari/IMDB-dataset-Sentiment-Analysis-Topic-Modeling/blob/main/sentiment_ML.ipynb

2. **`sentiment_classification_DL.ipynb`**  
   - **Description:**  
     Applies deep learning to sentiment classification using:  
       - Long Short-Term Memory (LSTM) networks.  
     - **Key Features:**  
       - Implementation of LSTM for textual data.  
       - Analysis of its performance metrics.  
    **colab link:** https://colab.research.google.com/github/Ifham-Ansari/IMDB-dataset-Sentiment-Analysis-Topic-Modeling/blob/main/sentiment_classification_DL.ipynb
      
3. **`Topic_Modeling_BERT.ipynb`**  
   - **Description:**  
     Performs topic modeling on the IMDB 50k reviews dataset using BERTopic.  
     - **Key Features:**  
       - Topic extraction to discover themes in reviews.  
       - Insights into common discussion points such as acting, screenplay, and music.
   **colab link:** https://colab.research.google.com/github/Ifham-Ansari/IMDB-dataset-Sentiment-Analysis-Topic-Modeling/blob/main/Topic_Modeling_BERT.ipynb  

4. **`index.ipynb`**  
   - **Description:**  
     A comprehensive file combining all the analyses and comparisons in one place.  
     - **Key Features:**  
       - Exploratory Data Analysis (EDA):  
         - Visualizations of demographic and textual data.  
         - Identification of the best and worst-reviewed movies by genre.  
       - Sentiment Analysis Comparison:  
         - Metrics comparison for Naive Bayes (NB), Logistic Regression (LR), and LSTM models.  
       - Topic Modeling with BERTopic:  
         - Extracting and visualizing key topics in the reviews.  
       - Preprocessed data and model training insights.
   **colab link:** https://colab.research.google.com/github/Ifham-Ansari/IMDB-dataset-Sentiment-Analysis-Topic-Modeling/blob/main/index.ipynb

---

## Installation

To use the notebooks, clone this repository and set up the required environment.

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
pip install -r requirements.txt
```

> Note: Ensure you have Python 3.8+ installed.

## Usage

1. Open the desired Jupyter Notebook (`.ipynb`) using Jupyter Notebook or JupyterLab.
2. Run the cells step by step to reproduce the results.

---

## Dataset

- **Dataset Used:** [IMDB 50k Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- The dataset contains two columns: `review` and `sentiment`.  
  - `review`: Text of the movie review.  
  - `sentiment`: Binary sentiment labels (`positive` or `negative`).  

---

## Results

### Sentiment Analysis

- **Machine Learning Models:**  
  - Naive Bayes (NB): `Metric scores`  
  - Logistic Regression (LR): `Metric scores`  

- **Deep Learning Model:**  
  - LSTM: `Metric scores`  

### Topic Modeling

- **BERTopic**: Extracted themes such as acting, screenplay, and music.

### Visualizations

- **Best and Worst-Reviewed Movies by Genre**  
- **Demographic and Textual Data Analysis**

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

---

## Acknowledgments

- Dataset: [Kaggle IMDB 50k Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Libraries: Scikit-learn, TensorFlow, BERTopic, Pandas, Matplotlib, and more.

---
