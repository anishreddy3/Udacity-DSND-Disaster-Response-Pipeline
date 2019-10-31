import sys
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    
    '''
    This functions reads data from the database and returns X and y
    
    Args:
    database_filepath : path to the database
    
    Out:
    X : input messages for the machine learning model
    y : output labels for the input messages
    
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM "+database_filepath[5:-3], engine)
    X = df['message']

    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    category_names = y.columns.values
    return X, y, category_names


def tokenize(text):
    '''
    A function to normalize, tokenize and stem the given string
    
    Input:
    text : A string containing a message that needs to be processed
    
    Output:
    clean_tok : A list of normalized and stemmed tokens from the given string
    '''
    # Normalize text : converting it to lowercase and removing punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words('english')]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tok = []
    for tok in tokens:
        clean_token = lemmatizer.lemmatize(tok).strip()
        clean_tok.append(clean_token)
    
    return clean_tok


def build_model():
    '''
    A function that consists of the Machine Learning Pipeline
    
    Input:
    N/A
    
    Output:
    model : grid search model
    
    '''
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer = tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__min_samples_split': [10, 15],
        'clf__estimator__n_estimators': [10, 20]
    }

    model = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1)
    
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    A function to display the evaluation results of our model
    
    Input:
    model : Our model fitted on training data
    X_test : Test training data
    Y_test : True labels for test data
    category_names : columns of the target 
    
    Output:
    Display Classification accuracy
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names = category_names))


def save_model(model, model_filepath):
    ''' 
    A function to save the fitted model
    
    Input:
    model : Fitted model
    model_filepath : path to save the model as a pickle file
    
    Output:
    N/A
    
    '''
    with open(model_filepath, 'wb') as f:
    
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()