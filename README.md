# Spam/Ham Email Classification

This project implements a complete pipeline for spam email classification using a Multinomial Naive Bayes algorithm built from scratch. The pipeline includes data loading and cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and interactive functions for real-time prediction. The project demonstrates an end-to-end machine learning workflow in Python, utilizing libraries like `pandas`, `numpy`, `matplotlib`, and `nltk`.

## Project Overview

The core of this project is a `NaiveBayesClassifier` class that implements the spam detection logic with Laplace smoothing. The model is trained on a dataset of emails, which are preprocessed to clean text, handle special characters, and convert email bodies into numerical feature vectors using a Bag-of-Words approach.

The Jupyter Notebook is structured to guide the user from initial data exploration to final model application, making it a comprehensive demonstration of text classification principles.

## Features

-   **Data Loading & Preprocessing**: The `format_data` function loads CSV files, combines 'Subject' and 'Message' fields, and engineers new features like punctuation ratios.
-   **Exploratory Data Analysis (EDA)**: Visualizations of class distribution (spam vs. ham), and word clouds to highlight the most frequent terms in each category.
-   **Custom Text Cleaning**: A robust `process_email` function that lowercases text, removes HTML, normalizes URLs and numbers, performs stemming, and removes stopwords.
-   **Feature Engineering**: Converts cleaned text data into numerical Bag-of-Words vectors based on a filtered vocabulary of the 5,000 most common words.
-   **From-Scratch Naive Bayes Classifier**: A custom-built `NaiveBayesClassifier` class that handles model fitting, prediction, and evaluation without relying on external ML libraries like Scikit-learn for the core algorithm.
-   **Interactive Functions**:
    1.  **Predict New Email**: A function that prompts the user for a new email subject and message and provides a real-time spam/ham prediction with confidence scores.
    2.  **Evaluate on CSV**: A function that evaluates the trained model on an external CSV file (e.g., `val.csv`), providing a full performance report, including a confusion matrix.

## Usage

To run the spam classifier, execute the Jupyter Notebook `Spam-Ham-Detection.ipynb`.

1.  **Prerequisites**: Ensure you have Python and Jupyter Notebook installed. The required libraries can be installed via pip:
    ```bash
    pip install pandas numpy matplotlib seaborn wordcloud nltk
    ```
2.  **Download NLTK Data**: The notebook will automatically download the necessary `nltk` packages (`punkt`, `stopwords`).
3.  **Dataset**: Make sure `train.csv` and `val.csv` are in the same directory as the notebook.
4.  **Run the Notebook**: Open and run all cells sequentially.
    -   The notebook will first perform EDA and data preprocessing.
    -   It will then train the `NaiveBayesClassifier`.
    -   Finally, it will launch the interactive functions, where you can:
        -   Test a new email by following the prompts for `function_1_predict_new_email`.
        -   Evaluate the model on `val.csv` by entering the filename at the prompt for `function_2_evaluate_csv_file`.

## Model Performance

The Naive Bayes classifier was trained on `train.csv` and evaluated on both the training set (seen data) and a separate validation set (`val.csv`, unseen data). The model demonstrated strong and consistent performance.

| Dataset          | Accuracy | Precision | Recall   | F1-Score |
| ---------------- | :------: | :-------: | :------: | :------: |
| **Training Set** |  97.83%  |  97.83%   |  97.83%  |  97.83%  |
| **Validation Set**|  97.47%  |  97.47%   |  97.47%  |  97.47%  |

*(Note: Precision, Recall, and F1-Score are the weighted averages for both classes.)*

### Analysis

-   The model achieves very high accuracy **(97.83%)** on the data it was trained on.
-   More importantly, it maintains a high accuracy of **97.47%** on the unseen validation data.
-   The minimal drop in performance (~0.36%) between the training and validation sets indicates that the model generalizes well and is not significantly overfitted.
-   High precision and recall scores across both datasets show that the classifier is effective at correctly identifying spam while also avoiding the misclassification of legitimate (ham) emails.

## Conclusion

This project successfully demonstrates the implementation of a Naive Bayes classifier for spam detection from scratch. The end-to-end pipeline, from data cleaning to interactive prediction, highlights a practical approach to solving a common NLP problem.

**Key Accomplishments:**
-   An effective, from-scratch implementation of the Multinomial Naive Bayes algorithm.
-   A robust data preprocessing pipeline tailored for email text.
-   High classification accuracy and strong generalization to unseen data.
-   User-friendly interactive functions for real-world testing and evaluation.

**Future Improvements:**
-   **Advanced Models**: Explore more complex models like LSTMs or Transformers that can capture word context and semantics, which may improve performance on more sophisticated spam.
-   **Feature Engineering**: Incorporate metadata features (e.g., sender information, email headers) or n-grams to enhance the feature set.
-   **Handling Modern Spam**: Develop techniques to handle image-based spam or text obfuscation (e.g., `v1agra`).
