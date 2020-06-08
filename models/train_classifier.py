import sys
import pandas as pd
import re
from sqlalchemy import create_engine

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('messages', engine)

    # Split the dataset (target and features)
    X = df['message']  # target variable
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)  # all others are features (categories to predict)

    return X, Y, Y.columns
    # pass


def tokenize(text):
    # Normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize
    tokens = word_tokenize(text)
    # Set stopword idiom
    stop_words = stopwords.words("english")
    # Reduce words to their root form and remove stop words
    clean_text = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]

    return clean_text


def model_report(prediction, y_test):
    '''
     Function to generate classification report on the model
     prediction: predicted model
     y_test: features test
     Output: prints a classification model report by categories
    '''
    for i, col in enumerate(y_test):
        print("Feature: {}".format(col))
        print("------------------------------------------------------")
        print(classification_report(y_test[col], prediction[:, i]))


def build_model():
    '''
    Build pipeline model
    :return: GridSearchCV
    '''
    # pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # pipeline parameters
    parameters = {  # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_df': (0.5, 1.0),
        # 'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__learning_rate': [1, 2]}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test):
    '''

    :param model: model to evaluate
    :param X_test: X test dataset
    :param Y_test: Y test dataset
    :Output: calls model_report function
    '''
    model_ = model.predict(X_test)
    model_report(model_, Y_test)


def save_model(model, model_filepath):
    '''
    Saves the model into a pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Launch the train classifier process
    :return:
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
