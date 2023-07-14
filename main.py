import numpy as np
import pandas as pd


import pymongo
import csv
import os
import time
from transformers import AutoTokenizer


class mongoDBManager:
    """
    A class to manage MongoDB operations.

    Attributes:
        mongodb_url (str): The URL of the MongoDB instance.
        database_name (str): The name of the database to connect to.
        client (pymongo.MongoClient): The MongoDB client object.
        database (pymongo.database.Database): The MongoDB database object.
    """

    def __init__(self, mongodb_url, database_name):
        """
        Initializes the MongoDBManager instance.

        Args:
            mongodb_url (str): The URL of the MongoDB instance.
            database_name (str): The name of the database to connect to.
        """
        self.client = pymongo.MongoClient(mongodb_url)
        self.database = self.client[database_name]

    def create_collection(self, collection_name):
        """
        Creates a new collection in the connected database.

        Args:
            collection_name (str): The name of the collection to create.
        """
        if collection_name in self.database.list_collection_names():
            self.database.drop_collection(collection_name)

        self.database.create_collection(collection_name)
        

    def insert_dataframe_data(self, collection_name, dataframe):
        """
        Inserts data from a DataFrame into the specified collection.

        Args:
            collection_name (str): The name of the collection to insert data into.
            dataframe (pandas.DataFrame): The DataFrame containing the data.
        """
        collection = self.database[collection_name]
        records = dataframe.to_dict(orient="records")
        collection.insert_many(records)
    


class dataExtractor:
    def __init__(self) -> None:
        super(dataExtractor, self).__init__()
        self.actions_df = None
        self.book_df = None
        self.merge_df = None
        self.feature_encodings = None
        self.data_df = None 

    def read_csv_data(self, csv_url, columns=[]):
        self.df = pd.read_csv(csv_url)
        if columns:
            self.df.columns = columns

        return self.df
    
    def pre_process_actions(self, csv_url):
        self.actions_df = self.read_csv_data(csv_url, columns=['account_id', 'book_id', 'creation_date']).dropna()
        self.actions_df['creation_date'] = pd.to_datetime(self.actions_df['creation_date']).dt.ceil('s')
        # self.actions_df['creation_date'] = pd.to_datetime(self.actions_df['creation_date']).dt.date
        self.actions_df = self.actions_df.groupby('account_id', group_keys=True).apply(lambda x: x.sort_values('creation_date', ascending=False)).reset_index(drop=True)
        self.actions_df = self.actions_df[pd.to_datetime(self.actions_df['creation_date']).dt.year == 2022]
        self.rating()

    def rating(self):
        self.actions_df['score'] = 5
        previous_id = None
        previous_day = None
        previous_idx = None
        for index, row in self.actions_df.iterrows():
            current_id = row['account_id']
            if previous_id != current_id:
                previous_idx = None
            current_day = pd.to_datetime(row['creation_date']).day_of_year
            if current_id == previous_id and current_day <= previous_day:
                if previous_idx == None:
                    self.actions_df.at[index, 'score'] = self.actions_df.at[index, 'score'] - (previous_day-current_day)//29
                else:
                    self.actions_df.at[index, 'score'] = self.actions_df.at[previous_idx, 'score'] - (previous_day-current_day)//29
                previous_idx = index
            previous_id = current_id
            previous_day = current_day

    def pre_process_book(self, csv_url):
        self.book_df = self.read_csv_data(csv_url)
        rating_mean = self.book_df['rating'].mean()
        self.book_df['rating'] = self.book_df['rating'].fillna(value=rating_mean)

    def merge_tables(self, on='book_id', tok_column='categories'):
        self.merge_df = self.actions_df.merge(self.book_df, on=on, how='inner')
        self.merge_df['rating_count'] = self.merge_df.groupby('account_id')['book_id'].transform('count')
        self.tokenizer(column=tok_column)
        self.data_df = pd.concat([self.merge_df[['account_id', 'book_id','creation_date', 'score', 'price', 'number_of_page',
                                                'PhysicalPrice', 'rating', 'rating_count']],
                                  pd.DataFrame(self.feature_encodings['input_ids'])], axis=1)
        
        self.data_df.columns = self.data_df.columns.astype(str)

    def tokenizer(self, column='categories'):
        tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        self.feature_encodings = tokenizer(self.merge_df[column].tolist(), truncation=True, padding=True, max_length=64)


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
    main(mongodb_url="mongodb://localhost:27017/",
         database_name="taaghche",
         action_url='dataset/actions.csv',
         book_url='dataset/book_data.csv')
    
 