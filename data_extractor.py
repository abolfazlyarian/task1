from flask import Flask, request, make_response, jsonify
import numpy as np
import pandas as pd
import pymongo
from transformers import AutoTokenizer
import os
import logging

app = Flask(__name__)

class mongoDBManager:
    """
    A class to manage MongoDB operations.
    """
    def __init__(self, mongodb_url: str, database_name: str):
        """
        Initializes the MongoDBManager instance.

        Args:
            mongodb_url (str): The URL of the MongoDB instance.
            database_name (str): The name of the database to connect to.
        """
        self.client = pymongo.MongoClient(mongodb_url)
        self.database = self.client[database_name]

    def create_collection(self, collection_name: str):
        """
        Creates a new collection in the connected database.

        Args:
            collection_name (str): The name of the collection to create.
        """
        if collection_name in self.database.list_collection_names():
            self.database.drop_collection(collection_name)
            logging.info("old %s collection dropped", collection_name)

        self.database.create_collection(collection_name)
        logging.info("create %s collection", collection_name)
        
    def insert_dataframe_data(self, collection_name: str, dataframe: pd.DataFrame):
        """
        Inserts data from a DataFrame into the specified collection.

        Args:
            collection_name (str): The name of the collection to insert data into.
            dataframe (pandas.DataFrame): The DataFrame containing the data.
        """
        collection = self.database[collection_name]
        if collection.find_one() == None:
            records = dataframe.to_dict(orient="records")
            collection.insert_many(records)
            logging.info('insert data to new %s collection', collection_name)
        else:
            field_names = list(collection.find_one().keys())
            field_names.remove('_id')
            if dataframe.columns == field_names:
                records = dataframe.to_dict(orient="records")
                collection.insert_many(records)
                logging.info('insert data to %s collection', collection_name)
            else:
                logging.info("inserting is wrong. your csv file is not match with %s collection. field_name = %s",collection_name, field_names)

    
class dataExtractor:
    def __init__(self) -> None:
        """
        Initializes the DataExtractor class.

        """
        super(dataExtractor, self).__init__()
        self.actions_df = None
        self.book_df = None
        self.merge_df = None
        self.feature_encodings = None
        self.data_df = None 

    def read_csv_data(self, csv_url, columns=[]):
        """
        Reads a CSV file from the given URL and returns a pandas DataFrame.

        Args:
            csv_url (str): The URL of the CSV file.
            columns (list): Optional. List of column names to assign to the DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame containing the data from the CSV file.
        """
        self.df = pd.read_csv(csv_url)
        if columns:
            self.df.columns = columns

        return self.df
    
    def pre_process_actions(self, csv_url):
        """
        Pre-processes the actions data.

        Reads the actions.csv file from the given URL, performs data cleaning and manipulation,
        and adds a 'score' column to the DataFrame.

        Args:
            csv_url (str): The URL of the actions.csv file.
        """

        # Read actions.csv file
        self.actions_df = self.read_csv_data(csv_url, columns=['account_id', 'book_id', 'creation_date']).dropna()
        self.actions_df['creation_date'] = pd.to_datetime(self.actions_df['creation_date']).dt.ceil('s')
        
        # Sort each user by the time of using books 
        self.actions_df = self.actions_df.groupby('account_id', group_keys=True).apply(lambda x: x.sort_values('creation_date', ascending=False)).reset_index(drop=True)
        
        # Remove 2021's data
        self.actions_df = self.actions_df[pd.to_datetime(self.actions_df['creation_date']).dt.year == 2022]

        # Adding score column to dataframe
        self.rating()

    def rating(self):
        """
        Calculates the rating scores for each action in the actions DataFrame.

        The rating score is assigned based on the time of action and the user's previous actions.
        """
        self.actions_df['score'] = 5
        previous_id = None
        previous_day = None
        previous_idx = None
        for index, row in self.actions_df.iterrows():
            # check that user is changed
            current_id = row['account_id']
            if previous_id != current_id:
                previous_idx = None
                first_day = pd.to_datetime(row['creation_date']).day_of_year
            
            current_day = pd.to_datetime(row['creation_date']).day_of_year
            
            # rating to user 
            if current_id == previous_id and current_day < previous_day:
                if previous_idx == None:
                    self.actions_df.at[index, 'score'] = self.actions_df.at[index, 'score'] - (first_day-current_day)//29
                else:
                    self.actions_df.at[index, 'score'] = self.actions_df.at[previous_idx, 'score'] - (first_day-current_day)//29
                previous_idx = index
            previous_id = current_id
            previous_day = current_day

    def pre_process_book(self, csv_url):
        """
        Pre-processes the book data.

        Reads the book data from the given URL and fills the missing values in the 'rating' column.

        Args:
            csv_url (str): The URL of the book data CSV file.
        """

        # Read actions.csv file
        self.book_df = self.read_csv_data(csv_url)
        rating_mean = self.book_df['rating'].mean()

        # Filling the Nan velues in rating column
        self.book_df['rating'] = self.book_df['rating'].fillna(value=rating_mean)

    def merge_tables(self, on='book_id', tok_column='categories'):
        """
        Merges the actions and book data and creates a final DataFrame.

        Args:
            on (str): Optional. The column to merge the tables on (default: 'book_id').
            tok_column (str): Optional. The column to tokenize (default: 'categories').
        """

        # inner merge two dataframe on book_id a
        self.merge_df = self.actions_df.merge(self.book_df, on=on, how='inner')
        self.merge_df['rating_count'] = self.merge_df.groupby('account_id')['book_id'].transform('count')

        # tokenizing the categories coulmn for using feature 
        self.tokenizer(column=tok_column)

        # concate the features 
        self.data_df = pd.concat([self.merge_df[['account_id', 'book_id','creation_date', 'score', 'price', 'number_of_page',
                                                'PhysicalPrice', 'rating', 'rating_count']],
                                  pd.DataFrame(self.feature_encodings['input_ids'])], axis=1)
        
        self.data_df.columns = self.data_df.columns.astype(str)

    def tokenizer(self, column='categories'):
        """
        Tokenizes the specified column using a BERT-based tokenizer.

        Args:
            column (str): Optional. The column to tokenize (default: 'categories').
        """
        tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        self.feature_encodings = tokenizer(self.merge_df[column].tolist(), truncation=True, padding=True, max_length=64)

@app.route('/update/collections', methods = ['GET'])
def update_collections():
    # Create an instance of MongoDBManager
    mongo_manager = mongoDBManager(os.environ.get("ME_CONFIG_MONGODB_URL",default="mongodb://localhost:27017/"),
                                   os.environ.get("MONGODB_NAME",default="taaghche"))

    # Create collections
    mongo_manager.create_collection("actions")
    mongo_manager.create_collection("book_data")
    mongo_manager.create_collection("merge")

    # Data extracting
    data_extractor = dataExtractor()
    data_extractor.pre_process_actions(csv_url= os.environ.get("action_url" ,default='dataset/actions.csv'))
    data_extractor.pre_process_book(csv_url= os.environ.get('book_url',default='dataset/book_data.csv'))
    data_extractor.merge_tables()

    # Insert Data to collections
    mongo_manager.insert_dataframe_data("actions", data_extractor.actions_df)
    mongo_manager.insert_dataframe_data("book_data", data_extractor.book_df)
    mongo_manager.insert_dataframe_data("merge", data_extractor.data_df)

    # Process the data and generate a response
    response_data = {'status': 'collections updated'}
    
    # Create a response object
    response = make_response(response_data, 200)
    
    # Return the response
    return response


def main(mongodb_url, database_name, action_url, book_url):

    # Create an instance of MongoDBManager
    mongo_manager = mongoDBManager(mongodb_url, database_name)

    # Create collections
    mongo_manager.create_collection("actions")
    mongo_manager.create_collection("book_data")
    mongo_manager.create_collection("merge")

    # Data extracting
    data_extractor = dataExtractor()
    data_extractor.pre_process_actions(csv_url=action_url)
    data_extractor.pre_process_book(csv_url=book_url)
    data_extractor.merge_tables()

    # Insert Data to collections
    mongo_manager.insert_dataframe_data("actions", data_extractor.actions_df)
    mongo_manager.insert_dataframe_data("book_data", data_extractor.book_df)
    mongo_manager.insert_dataframe_data("merge", data_extractor.data_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='[%(asctime)s] --> %(message)s')
    main(mongodb_url= os.environ.get("ME_CONFIG_MONGODB_URL",default="mongodb://localhost:27017/"),
         database_name= os.environ.get("MONGODB_NAME",default="taaghche"),
         action_url= os.environ.get("action_url" ,default='dataset/actions.csv'),
         book_url= os.environ.get('book_url',default='dataset/book_data.csv'))
    
    app.run(host='0.0.0.0',port=5001)
 