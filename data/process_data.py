
# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    A function to load data from two csv files and merge them together
    
    Input:
    messages_filepath : location of file 1
    categories_filepath : location of file 2
    
    Output:
    df : merged dataframe from the two datasets
    
    '''
    
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on = 'id')
    
    return df


def clean_data(df):
    '''
    A function to clean the given datatset and return the cleaned version
    
    Input:
    df : A dataframe that needs to be cleaned by splitting the category columns and only having integers as values, dropping              duplicates and irrelevant data
    
    Output:
    df : A cleaned version of the dataframe according to our requirements
    
    '''
    
    
    
 
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat = ';', expand = True)
    
    # Getting category names for new columns
    row = categories[:1]
    category_colnames =  row.applymap(lambda s: s[:-2]).iloc[0, :].tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just binary values 0 or 1
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(columns = ['categories'], axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # remove entries with a related value of 2
    df = df[df['related'] != 2]
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename[5:-3], engine, index = False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()