from flask import Flask, request, make_response, jsonify
import pandas as pd 
import numpy as np
import pymongo
import time
import logging
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import xgboost as xgb
import pickle
from sklearn.metrics import ndcg_score, average_precision_score

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

        self.database.create_collection(collection_name)
        
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

    # Reading data from a collection
    for _ in range(3):
        try:
            df = mongo_manager.read_collection_data(coolection_name).drop(['_id'], axis=1)
            return df
        except:
            logging.info('collection not found. trying to found ...')
    
    if df.empty:
        logging.info('collection not found.')
    
    return pd.DataFrame()

def preProcess(df):
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
    df['rating_count'] = df.groupby('account_id')['book_id'].transform('count')
    feature_encodings = tokenizer(df['categories'].tolist(), truncation=True, padding=True, max_length=64)
    data_df = pd.concat([df[['account_id', 'book_id', 'price', 'number_of_page',
                    'PhysicalPrice', 'rating', 'rating_count']],
                    pd.DataFrame(feature_encodings['input_ids'])], axis=1)
    
    data_df.columns = data_df.columns.astype(str)
    if data_df.shape[1] < 16:
        col = list(map(str,range(data_df.shape[1]-7,9)))
        fill_mat = pd.DataFrame(np.zeros((data_df.shape[0], 16 - data_df.shape[1]), dtype=np.int16), columns=col)
        return pd.concat([data_df, fill_mat], axis=1)
    else:
        return data_df


@app.route('/eval', methods = ['POST'])
def eval():
    data = request.json

    book_df = dataProvider(mongodb_url="mongodb://localhost:27017/",
                           database_name='taaghche',
                           coolection_name='book_data')
    if book_df.empty:
        return {'result': 'book collection not found'}

    
    user_id = data['uid']
    book_list = data['book_list']
    data = pd.DataFrame({'account_id': [user_id] * len(book_list),
                        'book_id': book_list})
    merge = data.merge(book_df, on='book_id', how='inner')
    if merge.shape[0] == 0:
        return {'result': 'There are no books from this list in the collection'}
    
    pred_df = preProcess(merge)

    # recommending

    filename = 'model.pkl'
    model = pickle.load(open(filename, 'rb'))
    features = ['price', 'number_of_page', 'PhysicalPrice', 'rating', 'rating_count', '0', '1', '2', '3', '4', '5', '6', '7', '8']
    preds = np.abs(model.predict(pred_df[features]))
    prob_scores = preds

    topk_idx = np.argsort(prob_scores)[::-1]
    recommend_df = pred_df.loc[topk_idx].reset_index(drop=True)

    # check recommend
    recom = []
    for i, row in recommend_df.iterrows():
        recom.append({"rank":i+1,
                    "book":book_df[book_df["book_id"] == row["book_id"]].to_dict(orient='records'),
                    "score":float(prob_scores[topk_idx][i]) })

    # Process the data and generate a response
    response_data = {'result': recom}
    
    # Create a response object
    response = jsonify(response_data)
    response.status_code = 200
    
    # Return the response
    return response
    

@app.route('/train', methods = ['GET'])
def train():

    data_df = dataProvider(mongodb_url="mongodb://localhost:27017/",
                           database_name='taaghche',
                           coolection_name='merge')
    if data_df.empty:
        return {'result': 'merge collection not found'}

    train, test = train_test_split(data_df, test_size=0.2, random_state=42)
    train = train.sort_values('account_id').reset_index(drop=True)
    test = test.sort_values('account_id').reset_index(drop=True)
    
    train_query = train['account_id'].value_counts().sort_index()
    test_query = test['account_id'].value_counts().sort_index()

    features = [i for i in train.columns.to_list() if i not in ['account_id','book_id','creation_date','score'] ]
    target = 'score'

    model = xgb.XGBRanker(objective='rank:pairwise', n_estimators=10, random_state=0,learning_rate=0.01)
    model.fit(
        train[features],
        train[target],
        group=train_query,
        eval_metric='ndcg',
        eval_set=[(test[features], test[target])],
        eval_group=[list(test_query)],
        verbose =True
    )

    true_relevance = np.asarray([test[target]])
    scores = np.asarray([model.predict(test[features])])
    model_eval = ndcg_score(true_relevance, scores)
    

    filename = 'model.pkl'
    pickle.dump(model, open(filename, 'wb'))

    # Process the data and generate a response
    response_data = {'status': 'model trained', 'ndcg_score': model_eval}
    
    # Create a response object
    response = make_response(response_data, 200)
    
    # Return the response
    return response
 
if __name__ == '__main__':
   logging.basicConfig(level=logging.INFO,
                       format='[%(asctime)s] --> %(message)s')
   app.run(debug=True)