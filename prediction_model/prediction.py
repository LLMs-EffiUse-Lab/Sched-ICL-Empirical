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

from prediction_model.TCA import TCA
from prediction_model.help import *
from prediction_model.CORAL import CORAL
from prediction_model.data_preprocess import *

MSG_LEN = 1

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


def cross_project_multi_label_classifier_(data_dir, model_list, train_sys, test_sys):
    def process_row(row):
        x = log_features_extraction(row['content'])  # Assuming 'query' is the input feature
        x = preprocessing.scale(x)
        y = [row[model] for model in model_list]
        return x, y

    all_data = pd.read_csv(data_dir, index_col=0)
    for model in model_list:
        all_data[model] = all_data['ref_answer'] == all_data[f'{model}_answer']

    train_data = all_data[all_data['dataset'] == train_sys]
    train_x, train_y = zip(*train_data.apply(process_row, axis=1))

    test_data = all_data[all_data['dataset'] == test_sys]
    test_x, test_y = zip(*test_data.apply(process_row, axis=1))

    model_num = len(model_list)

    le = preprocessing.LabelEncoder()
    train_y = le.fit_transform(train_y)
    clf = MultiOutputClassifier(estimator=XGBClassifier(n_jobs=-1, max_depth=10, n_estimators=1000))
    clf.fit(train_x, train_y)

    y_pred_accuracy = clf.predict_proba(test_x)
    y_pred = clf.predict(test_x)

    complexity = [[y_pred_accuracy[j][i][1] for j in range(model_num)] for i in range(len(y_pred_accuracy[0]))]
    df_pre_accuracy = pd.DataFrame(complexity, columns=model_list)

    print(classification_report(np.array(test_y), y_pred, digits=3, target_names=model_list))
    print('Accuracy Score: ', accuracy_score(np.array(test_y), y_pred))

    df_cost = get_cost_(test_data, model_list)
    df_true_accuracy = get_accuracy_(test_data, model_list)

    confusion_matrices = multilabel_confusion_matrix(np.array(test_y), y_pred)
    label_accuracy = {}
    for i, label in enumerate(model_list):
        cm = confusion_matrices[i]
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        label_accuracy[label] = accuracy
    df_label_accuracy = pd.DataFrame.from_dict(label_accuracy, orient='index', columns=['Accuracy'])

    return df_pre_accuracy, df_true_accuracy, df_cost, df_label_accuracy

def individual_classifier_query_(data_dir, model_list, train_sys, test_sys):
    def process_row(row, model):
        x = log_features_extraction(row['query'])  # Assuming 'query' is the input feature
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
        print(f"Processing {type}...")
        # if train_data['query'].isnull().sum() > 0:
        #     print(f"Missing query in {train_sys} dataset processing {type}...")

        train_data['query'] = get_query_for_dataset_(train_sys, all_data[all_data["dataset"] == train_sys]['content'], type)
        # train_data.loc[:, 'query'] = train_data.apply(lambda x: get_whole_query_(train_sys, x['content'], type, shot), axis=1)  # slow

        for i, row in train_data.iterrows():
            if row['query'] is None:
                print(f"Missing query in {train_sys} dataset processing {type}...")


        train_x_list, train_y_list = zip(*train_data.apply(lambda x: process_row(x, type), axis=1))
        train_x = np.array(train_x_list)
        train_y = np.array(train_y_list)


        test_data['query'] = get_query_for_dataset_(test_sys, all_data[all_data["dataset"] == test_sys]['content'], type)
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

    return df_pre_accuracy, evl_accuracy

        # if domain_adaption == 'coral':
        #     coral = CORAL()
        #     Xs_new = coral.fit(np.array(train_x), np.array(test_x))
        # else:
        #     raise ValueError(f"Invalid available domain adaption type: {domain_adaption}")


def individual_classifier_content_(data_dir, model_list, train_sys, test_sys):
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

