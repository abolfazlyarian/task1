from flask import Flask, request, make_response, jsonify
import numpy as np
import pandas as pd
import pymongo
from transformers import AutoTokenizer
import os
import logging
import time
import requests

app = Flask(__name__)

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
            logging.info("old %s collection dropped", collection_name)

        self.database.create_collection(collection_name)
        logging.info("create %s collection", collection_name)
        
    def read_collection_data(self, collection_name):
        """
        Reads all documents from the specified collection and returns them as a DataFrame.

        Args:
            collection_name (str): The name of the collection to read data from.

        Returns:
            pandas.DataFrame: The DataFrame containing the collection data.
        """
        collection = self.database[collection_name]
        cursor = collection.find({})
        data = list(cursor)
        df = pd.DataFrame(data)
        return df

def dataProvider(mongodb_url, database_name, coolection_name):
    # Create an instance of MongoDBManager
    mongo_manager = mongoDBManager(mongodb_url, database_name)
    df = pd.DataFrame()
    # Reading data from a collection
    for _ in range(3):
        try:
            df = mongo_manager.read_collection_data(coolection_name).drop(['_id'], axis=1)
            return df
        except:
            logging.info('collection not found. trying to found ...')
            time.sleep(3)
    
    if df.empty:
        # start to extraxting data with request to data_extractor web server
        logging.info("Collection not found. Initiating extraction process. Please wait...")
        requests.get('http://127.0.0.1:5001/update/collections', timeout=3)
        time.sleep(10*60)

        for _ in range(3):
            try:
                df = mongo_manager.read_collection_data(coolection_name).drop(['_id'], axis=1)
                return df
            except:
                logging.info('collection not found. trying to found ...')
                time.sleep(3)
        
        raise Exception("Collection not found. Extraction error encountered. Please perform manual extraction.")
    

@app.route('/provide', methods = ['GET'])
def provide():

    data = request.json
    try:
        # read data from database (Data_provider)
        data_df = dataProvider(mongodb_url= os.environ.get("ME_CONFIG_MONGODB_URL",default="mongodb://localhost:27017/"),
                            database_name= os.environ.get("MONGODB_NAME",default="taaghche"),
                            coolection_name=data['collection_name'])
        
        # Process the data and generate a response
        response_data = {'status': True, "data": data_df.to_json(orient='records')}
        
        # Create a response object
        response = make_response(response_data, 200)

    except Exception as e:
        # Process the data and generate a response
        response_data = {'status': False, 'msg': str(e)}

        # Create a response object
        response = make_response(response_data, 200)
    
    # Return the response
    return response

@app.route('/provideMerge', methods = ['GET'])
def provide_merge():

    data = request.json
    try:
        # read data from database (Data_provider)
        data_df = dataProvider(mongodb_url= os.environ.get("ME_CONFIG_MONGODB_URL",default="mongodb://localhost:27017/"),
                            database_name= os.environ.get("MONGODB_NAME",default="taaghche"),
                            coolection_name='book_data')
        
        # create data
        user_id = data['uid']
        book_list = data['book_list']
        data = pd.DataFrame({'account_id': [user_id] * len(book_list),
                            'book_id': book_list})
        merge_df = data.merge(data_df, on='book_id', how='inner')
        if merge_df.shape[0] == 0:
            raise Exception('There are no books from this list in the collection')
        
        # Process the data and generate a response
        response_data = {'status': True, "data": merge_df.to_json(orient='records')}
        
        # Create a response object
        response = make_response(response_data, 200)

    except Exception as e:
        # Process the data and generate a response
        response_data = {'status': False, 'msg': str(e)}

        # Create a response object
        response = make_response(response_data, 200)
    
    # Return the response
    return response

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='[%(asctime)s] --> %(message)s')
  
    app.run(host='0.0.0.0',port=5003)
 