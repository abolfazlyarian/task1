from flask import Flask, request, make_response, jsonify
import pandas as pd 
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import xgboost as xgb
import pickle
import requests
from sklearn.metrics import ndcg_score

app = Flask(__name__)

def preProcess(df):
    # tokenizing (feature engineering)
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
    df['rating_count'] = df.groupby('account_id')['book_id'].transform('count')
    feature_encodings = tokenizer(df['categories'].tolist(), truncation=True, padding=True, max_length=64)
    data_df = pd.concat([df[['account_id', 'book_id', 'price', 'number_of_page',
                    'PhysicalPrice', 'rating', 'rating_count']],
                    pd.DataFrame(feature_encodings['input_ids'])], axis=1)
    
    # concat features
    data_df.columns = data_df.columns.astype(str)
    if data_df.shape[1] < 16:
        col = list(map(str,range(data_df.shape[1]-7,9)))
        fill_mat = pd.DataFrame(np.zeros((data_df.shape[0], 16 - data_df.shape[1]), dtype=np.int16), columns=col)
        return pd.concat([data_df, fill_mat], axis=1)
    else:
        return data_df


@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json

    res = requests.get('http://127.0.0.1:5003/provideMerge', json=data)
    
    # data parsing
    json_records = res.json()
    if not json_records['status']:
        return {'result': json_records['msg']}

    # Convert the JSON records into a DataFrame
    merge_df = pd.read_json(json_records['data'])

    pred_df = preProcess(merge_df)

    # recommending use model
    try:
        filename = 'model.pkl'
        model = pickle.load(open(filename, 'rb'))
    except:
        response_data = {'result': 'model not found. please train the model'}
        return response_data
    
    features = ['account_id', 'book_id', 'price', 'number_of_page', 'PhysicalPrice', 'rating', 'rating_count', '0', '1', '2', '3', '4', '5', '6', '7', '8']
    preds = np.abs(model.predict(pred_df[features]))
    prob_scores = preds

    topk_idx = np.argsort(prob_scores)[::-1]
    recommend_df = pred_df.loc[topk_idx].reset_index(drop=True)

    # check recommend
    recom = []
    for i, row in recommend_df.iterrows():
        recom.append({"rank":i+1,
                    "book":merge_df[merge_df["book_id"] == row["book_id"]].to_dict(orient='records'),
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
    
    res = requests.get('http://127.0.0.1:5003/provide', json={"collection_name": "merge"})
    # data parsing
    json_records = res.json()
    if not json_records['status']:
        return {'result': json_records['msg']}

    # Convert the JSON records into a DataFrame
    data_df = pd.read_json(json_records['data'])

    # split data into train and test
    train, test = train_test_split(data_df, test_size=0.2, random_state=42)
    train = train.sort_values('account_id').reset_index(drop=True)
    test = test.sort_values('account_id').reset_index(drop=True)
    
    # create query group for xgboost
    train_query = train['account_id'].value_counts().sort_index()
    test_query = test['account_id'].value_counts().sort_index()
    
    features = [i for i in train.columns.to_list() if i not in ['creation_date','score'] ]
    target = 'score'

    # train xgboost model for learning-to-rank task
    model = xgb.XGBRanker(objective='rank:pairwise', n_estimators=50, random_state=0,learning_rate=0.1)
    model.fit(
        train[features],
        train[target],
        group=train_query,
        eval_metric='ndcg',
        eval_set=[(test[features], test[target])],
        eval_group=[list(test_query)],
        verbose =True
    )

    # calculate NDCG score for similarity of predictive rank and ground truth
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
   app.run(debug=True,host='0.0.0.0',port=5002)