import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import pymongo
import csv
import os
import time

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
        self.database.create_collection(collection_name)

    def insert_csv_data(self, collection_name, csv_file):
        """
        Inserts data from a CSV file into the specified collection.

        Args:
            collection_name (str): The name of the collection to insert data into.
            csv_file (str): The path to the CSV file containing the data.
        """
        collection = self.database[collection_name]
        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                collection.insert_one(row)

    def watch_directory(self, directory):
        """
        Watches the specified directory for changes and updates MongoDB collections accordingly.

        Args:
            directory (str): The path to the directory to watch for changes.
        """
        while True:
            for file in os.listdir(directory):
                if file.endswith(".csv"):
                    csv_path = os.path.join(directory, file)
                    collection_name = os.path.splitext(file)[0]
                    self.insert_csv_data(collection_name, csv_path)
                    print(f"Inserted data from {csv_path} into {collection_name}")
            time.sleep(60)


class dataExtractor:
    def __init__(self) -> None:
        super(dataExtractor, self).__init__()
        self.actions_df = None
        self.book_df = None

    def read_csv_data(self, csv_url, columns=[]):
        self.df = pd.read_csv(csv_url)
        if columns:
            self.df.columns = columns

        return self.df
    
    def pre_process_actions(self, df):
        self.df = df.dropna()

        # adfs 
        self.df['creation_date'] = pd.to_datetime(self.df['creation_date']).dt.date
        self.df = self.df.groupby('account_id', group_keys=True).apply(lambda x: x.sort_values('creation_date', ascending=False)).reset_index(drop=True)
        self.df = self.df[pd.to_datetime(self.df['creation_date']).dt.year == 2022]

        # dsaf
        self.df['score'] = 5
        previous_id = None
        previous_day = None
        previous_idx = None
        for index, row in self.df.iterrows():
            current_id = row['account_id']
            if previous_id != current_id:
                previous_idx = None
            current_day = pd.to_datetime(row['creation_date']).day_of_year
            if current_id == previous_id and current_day <= previous_day:
                if previous_idx == None:
                    self.df.at[index, 'score'] = self.df.at[index, 'score'] - (previous_day-current_day)//29
                else:
                    self.df.at[index, 'score'] = self.df.at[previous_idx, 'score'] - (previous_day-current_day)//29
                previous_idx = index
            previous_id = current_id
            previous_day = current_day

        # self.df.to_csv('dataset/action-pre.csv', index=False)
        return self.df


    def pre_process_books(self, df):
        self.df = df
        rating_mean = self.df['rating'].mean()
        self.df['rating'] = self.df['rating'].fillna(value=rating_mean)

        return self.df
    
    def merge_tables(self, left_df, right_df, on='book_id'):
        pass


class actionsExtractor(dataExtractor):
    def __init__(self, csv_url):
        super(actionsExtractor, self).__init__()
        self.actions_df = self.read_csv_data(csv_url, columns=['account_id', 'book_id', 'creation_date']).dropna()

    def pre_process(self):

        self.actions_df['creation_date'] = pd.to_datetime(self.actions_df['creation_date']).dt.date
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
    

class bookExtractor(dataExtractor):
    def __init__(self, csv_url):
        super(bookExtractor, self).__init__()
        self.book_df = self.read_csv_data(csv_url)

    def pre_process(self):
        rating_mean = self.book_df['rating'].mean()
        self.book_df['rating'] = self.book_df['rating'].fillna(value=rating_mean)




def dataExtractor1():
    action_df = pd.read_csv("dataset/actions.csv").dropna()
    action_df.columns = ['account_id', 'book_id', 'creation_date']
    action_df['creation_date'] = pd.to_datetime(action_df['creation_date']).dt.date
    
    action_df = action_df.groupby('account_id', group_keys=True).apply(lambda x: x.sort_values('creation_date', ascending=False)).reset_index(drop=True)
    action_df = action_df[pd.to_datetime(action_df['creation_date']).dt.year == 2022]

    action_df['score'] = 5
    previous_id = None
    previous_day = None
    previous_idx = None
    for index, row in action_df.iterrows():
        current_id = row['account_id']
        if previous_id != current_id:
            previous_idx = None
        current_day = pd.to_datetime(row['creation_date']).day_of_year
        if current_id == previous_id and current_day <= previous_day:
            if previous_idx == None:
                action_df.at[index, 'score'] = action_df.at[index, 'score'] - (previous_day-current_day)//29
            else:
                action_df.at[index, 'score'] = action_df.at[previous_idx, 'score'] - (previous_day-current_day)//29
            previous_idx = index
        previous_id = current_id
        previous_day = current_day

    action_df.to_csv('dataset/action-pre.csv', index=False)


def main():
    # MongoDB connection details
    mongodb_url = "mongodb://localhost:27017/"
    database_name = "mydatabase"

    # Create an instance of MongoDBManager
    mongo_manager = mongoDBManager(mongodb_url, database_name)

    # Create two collections
    mongo_manager.create_collection("actions")
    mongo_manager.create_collection("book_data")





    # Insert data from the first CSV file into table1
    mongo_manager.insert_csv_data("actions", "path/to/first.csv")

    # Insert data from the second CSV file into table2
    mongo_manager.insert_csv_data("table2", "path/to/second.csv")

    # Watch the directory for changes
    mongo_manager.watch_directory("path/to/watched_directory")


if __name__ == "__main__":
    dataext = dataExtractor()
    x = dataext.read_csv_data(csv_url='dataset/actions.csv', columns=['account_id', 'book_id', 'creation_date'])
    y = dataext.pre_process_actions(x)
    print(y)
