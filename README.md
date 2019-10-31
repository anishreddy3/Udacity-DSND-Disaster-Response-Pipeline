# Disaster Response Pipeline Project

## Table of Contents:

1. Installation
2. Summary
3. File Descriptions
4. Instructions
5. Screenshots

## Installation:

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.* 

## Summary: 

This project is designed to apply data engineering skills in building an end to end machine learning pipeline to classify disaster messages into 36 categories. A machine learning model is built based on real messages by preproceesing them using NLTK library and improved its performance by doing grid search to find the optimal parameters. The prediction results can help connect the affected individual to the right department.

This project contains a web app which acts as an API to our model for getting predictions in real time. We have some additional visualizations representing our dataset.

## File Descriptions:

We have three main files in this project:

1. `process_data.py` - It is a python script that loads in the categories.csv and messages.csv files, combine them into a single dataset, does all the required preprocessing and stores it in a SQLite database.

2. `train_classifier.py` - It is a python script that loads the dataset from the database, splits into feature and target variables. Then further splits it into train and test sets, then builds a pipeline for text processing and machine learning, after training the model we fine tune it using GridSearchCV, outputs classification report on test set and exports the model as a pickle files

3. `run.py` - It is a python script that runs our web app using Flask, Plotly is used for visualizations

## Instructions:

1. Go to the projects root directory and run the following command to run the ETL pipeline that does the preproceesing and stores in database : `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. To run the ML pipeline that trains our pipeline and saves the model, run the following command : `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command to start running our web app to get predictions and see visualizations : `python run.py`

4. Visit [](http://0.0.0.0:3001/) to see the web app

## Screenshots:

![]('pic1.png')
![]('pic2.png')
