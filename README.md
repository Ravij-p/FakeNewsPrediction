## Fake News Detection Project using Logistic Regression

This project aims to classify news articles as either real or fake using a machine learning approach. The dataset used in this project is sourced from a Kaggle competition on fake news detection. The steps involved in the project include data preprocessing, feature extraction, model training, and evaluation.

### Dataset

The dataset used can be found [here](https://www.kaggle.com/c/fake-news/data?select=train.csv).

### Steps Involved

1. **Importing Libraries**
   - Import necessary libraries such as `numpy`, `pandas`, `re`, `nltk`, `sklearn`.
   - Download NLTK stopwords.

2. **Data Loading**
   - Load the dataset using `pandas`.

3. **Data Preprocessing**
   - Handle missing values by filling them with empty strings.
   - Combine the `author` and `title` columns to create a new feature `content`.
   - Define a stemming function to preprocess the text by removing non-alphabet characters, converting to lowercase, removing stopwords, and applying stemming.

4. **Text Vectorization**
   - Use `TfidfVectorizer` to convert the text data into numerical features.

5. **Splitting the Dataset**
   - Split the dataset into training and testing sets.

6. **Model Training**
   - Train a logistic regression model on the training set.

7. **Model Evaluation**
   - Evaluate the model's performance on both training and testing sets using accuracy scores.

8. **Prediction and Verification**
   - Predict the labels for a subset of the test data and compare them with the actual labels to verify the model's performance.

### Results

- The logistic regression model achieved an accuracy of approximately 98.66% on the training data.
- The model achieved an accuracy of approximately 97.91% on the testing data.

### Example Predictions

The model's predictions on a few test samples are compared with the actual labels to demonstrate its performance.

### Libraries and Tools Used

- **Numpy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **NLTK**: For natural language processing tasks such as stopword removal.
- **Scikit-learn**: For machine learning tasks including text vectorization, model training, and evaluation.

### How to Run the Project

1. Download the dataset from the provided link.
2. Install the necessary libraries if not already installed:
   ```bash
   pip install numpy pandas nltk scikit-learn
   ```
3. Ensure you have the NLTK stopwords downloaded:
   ```python
   import nltk
   nltk.download('stopwords')
   ```
4. Follow the steps outlined above to preprocess the data, train the model, and evaluate its performance.

By following these steps, you can replicate the process of building a fake news detection model and apply similar techniques to other text classification problems.
