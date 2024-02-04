
import argparse
import os
import json
import numpy as np
import pandas as pd
import sklearn
import s3fs
# import boto3
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split 

# This is created because the scikit learn models once deployed creates a scikit-learn model server which automatically loads the model saved during training
# It invokes the model_fn function to serve the model so we should keep this in our script.

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return clf




if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # this is for tuning. we can also run jobs for randomsearch or gridsearch to find optimum hyperparameters.
    # parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--random_state', type = int, default = 0)
    

    # an alternative way to load hyperparameters via SM_HPS environment variable.
    # parser.add_argument('--sm-hps', type=json.loads, default=os.environ['SM_HPS'])
    # SM_HPS='{"batch-size": "256", "learning-rate": "0.0001","epochs": "10"}' one way to load all the hyperparameters through a dict

    # input data and model directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--train-file' , type = str , default ='train_Data.csv')
    # parser.add_argument('--test-file', type =str , default = 'test_data.csv')

    args, _ = parser.parse_known_args()

    print("SM_MODEL_DIR:", os.environ.get('SM_MODEL_DIR'))
    print("SM_CHANNEL_TRAIN:", os.environ.get('SM_CHANNEL_TRAIN'))

    print("sklearn version: " , sklearn.__version__)
    print("joblib version: ", joblib.__version__)


    print("Reading Data")

    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    # train_df = pd.read_csv('s3://loaneligibilitybucket/sagemaker/loaneligibilityalgorithm/sklearncontainer/train_Data.csv')
    # test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    y = train_df['Loan_Status']
    X = train_df.drop('Loan_Status', axis = 1)

    print(" Building training and testing datasets")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state=42)


    print('\nTraining the model')
    model = RandomForestClassifier(n_estimators=args.n_estimators,random_state=args.random_state)
    model.fit(X_train,y_train)

    # inorder to save the model for deployment we should save the model in the training script here
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print('\nmodel stored at ',model_path)

    y_pred = model.predict(X_test)

    ##model performance

    test_accuracy = accuracy_score(y_test,y_pred)
    ClassificationReport = classification_report(y_test, y_pred)
    confusionmatrix = confusion_matrix(y_test,y_pred)

    # print('Model Accuraacy' , test_accuracy)
    # print('\nclassification_Report')
    # print(ClassificationReport)
    # print('\nconfusion Matrix' )
    # print(confusionmatrix)
