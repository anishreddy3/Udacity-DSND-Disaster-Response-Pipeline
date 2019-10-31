import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # dictionary for mapping out 'related' column
    rel_dict = {0: 'not_related', 1: 'related'}
    
    # get the related column
    rel = df['related'].map(rel_dict)
    rel_counts = rel.value_counts()
    rel_names = list(rel_counts.index)
    
    # get other columns other than related
    cat = df.iloc[:,4:]
    cat_mean = cat.mean().sort_values(ascending = False)[1:6]
    cat_names = list(cat_mean.index)
    
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x = rel_names,
                    y = rel_counts
                )
            ],
            'layout': {
                'title' : 'Distribution of whether the message is relevant',
                'yaxis' : {
                    'title' : 'Count of respective title'
                },
                'xaxis' : {
                    'title' : 'Message relevance'
                }
            }
        },
        {
            'data' : [
                Bar(
                    x = cat_names,
                    y = cat_mean
                )
            ],
            'layout': {
                'title' : 'Top 5 categories of messages',
                'yaxis' : {
                    'title' : 'Mean of the category '
                },
                'xaxis' : {
                    'title' : 'Category'
                }
            }
        }
            
              
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()