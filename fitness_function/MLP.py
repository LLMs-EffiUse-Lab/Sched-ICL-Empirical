### Machine Learning based Prediction

import pandas as pd
import numpy as np
import pickle
import string

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from prediction_model.data_preprocess import *


def get_avg_accuracy(df_accuracy, model_list):
    return df_accuracy[model_list].mean()

def log_features_extraction(log):
    features = []
    # number of tokens
    features.append(len(log.split()))
    # number of unique tokens
    features.append(len(set(log.split())))
    # number of characters
    features.append(len(log))
    # number of unique characters
    features.append(len(set(log)))
    # number of digits
    features.append(sum(c.isdigit() for c in log))
    # number of letters
    features.append(sum(c.isalpha() for c in log))
    # number of punctuations
    features.append(sum(c in string.punctuation for c in log))
    # average number of characters per token
    features.append(features[2] / features[0])
    # average number of characters per unique token
    features.append(features[2] / features[1])
    # average number of digits per token
    features.append(features[4] / features[0])
    # average number of punctuations per token
    features.append(features[6] / features[0])
    # max length of token
    features.append(max(len(token) for token in log.split()))
    # min length of token
    features.append(min(len(token) for token in log.split()))
    # max punctuation length of token
    features.append(max(sum(c in string.punctuation for c in token) for token in log.split()))
    # min punctuation length of token
    features.append(min(sum(c in string.punctuation for c in token) for token in log.split()))
    # max digit length of token
    features.append(max(sum(c.isdigit() for c in token) for token in log.split()))
    # min digit length of token
    features.append(min(sum(c.isdigit() for c in token) for token in log.split()))
    return features

def get_cost_(job_data, model_list):
    cost_data = pd.DataFrame()
    for model in model_list:
        cost_column = f'{model}_cost'
        cost_data[model] = job_data[cost_column]
    return cost_data

def get_accuracy_(job_data, model_list):
    accuracy_data = pd.DataFrame()
    for model in model_list:
        cost_column = f'{model}'
        accuracy_data[model] = job_data[cost_column]
    return accuracy_data


def cross_project_individual_classifier_(data_dir, model_list, train_sys, test_sys):
    def process_row(row, model):
        x = log_features_extraction(row['content'])
        x = preprocessing.scale(x)
        y = row[model]
        return x, y

    all_data = pd.read_csv(data_dir, index_col=0)
    for model in model_list:
        all_data[model] = all_data['ref_answer'] == all_data[f'{model}_answer']

    pre_accuracy = {}
    evl_accuracy = {}
    train_data = all_data[all_data['dataset'] == train_sys]
    test_data = all_data[all_data['dataset'] == test_sys]

    for type in ['simple', 'standard', 'enhance', 'fewshot_1', 'fewshot_2', 'fewshot_4']:  #
        train_x_list, train_y_list = zip(*train_data.apply(lambda x: process_row(x, type), axis=1))
        train_x = np.array(train_x_list)
        train_y = np.array(train_y_list)

        test_x_list, test_y_list = zip(*test_data.apply(lambda x: process_row(x, type), axis=1))
        test_x = np.array(test_x_list)
        test_y = np.array(test_y_list)

        le = preprocessing.LabelEncoder()
        train_y = le.fit_transform(train_y)

        clf = XGBClassifier(n_jobs=-1, max_depth=10, n_estimators=1000)
        clf.fit(train_x, train_y)

        y_pred_accuracy = clf.predict_proba(test_x)
        y_pred_binary = clf.predict(test_x)

        pre_accuracy[type] = y_pred_accuracy[:, 1]
        accuracy = accuracy_score(test_y, y_pred_binary)
        evl_accuracy[type] = accuracy

    df_pre_accuracy = pd.DataFrame(pre_accuracy)
    # df_evl_accuracy = pd.DataFrame(evl_accuracy)
    df_cost = get_cost_(test_data, model_list)
    df_true_accuracy = get_accuracy_(test_data, model_list)

    return df_pre_accuracy, df_true_accuracy, df_cost, evl_accuracy

def individual_classifier_(data_dir, model_list, test_sys, test_size, itr):
    def process_row(row, model):
        x = log_features_extraction(row['content'])
        x = preprocessing.scale(x)
        y = row[model]
        return x, y

    all_data = pd.read_csv(data_dir, index_col=0)
    for model in model_list:
        all_data[model] = all_data['ref_answer'] == all_data[f'{model}_answer']

    job_data = all_data[all_data['dataset'] == test_sys]
    train_data, test_data = train_test_split(job_data, test_size=test_size, random_state=itr)
    pre_accuracy = {}
    evl_accuracy = {}

    for type in ['simple', 'standard', 'enhance', 'fewshot_1', 'fewshot_2', 'fewshot_4']:
        train_x_list, train_y_list = zip(*train_data.apply(lambda x: process_row(x, type), axis=1))
        train_x = np.array(train_x_list)
        train_y = np.array(train_y_list)

        test_x_list, test_y_list = zip(*test_data.apply(lambda x: process_row(x, type), axis=1))
        test_x = np.array(test_x_list)
        test_y = np.array(test_y_list)

        le = preprocessing.LabelEncoder()
        train_y = le.fit_transform(train_y)

        clf = XGBClassifier(n_jobs=-1, max_depth=10, n_estimators=1000)
        clf.fit(train_x, train_y)

        y_pred_accuracy = clf.predict_proba(test_x)
        y_pred_binary = clf.predict(test_x)

        pre_accuracy[type] = y_pred_accuracy[:, 1]
        accuracy = accuracy_score(test_y, y_pred_binary)
        evl_accuracy[type] = accuracy
    df_pre_accuracy = pd.DataFrame(pre_accuracy)
    # df_evl_accuracy = pd.DataFrame(evl_accuracy)
    df_cost = get_cost_(test_data, model_list)
    df_true_accuracy = get_accuracy_(test_data, model_list)

    # label_cost = get_cost_(train_data, model_list)

    return df_pre_accuracy, df_true_accuracy, df_cost, evl_accuracy #, label_cost