### Similarity-based Confidence Estimation

import pandas as pd
import numpy as np
import string
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix, jaccard_score

from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def preprocess_features(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def select_features(X, y, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    return X_selected, selected_indices

def apply_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    return labels

def visualize_features(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.title('Log Messages Visualization')
    plt.show()

def MMD(Xs, Xt):
    """
    Maximum Mean Discrepancy

    Calculate the distance between source Xs and target Xt
    """
    ns = Xs.shape[1]
    nt = Xt.shape[1]

    # Calculate kernel matrix
    sigma = np.median(pdist(np.vstack((Xs, Xt)).T))
    Ks = constructKernel(Xs.T, None, sigma)
    Kt = constructKernel(Xt.T, None, sigma)
    Kst = constructKernel(Xs.T, Xt.T, sigma)

    c_Ks = 1 / (ns ** 2)
    c_Kt = 1 / (nt ** 2)
    c_Kst = 2 / (ns * nt)

    dist = np.sum(c_Ks * Ks) + np.sum(c_Kt * Kt) - np.sum(c_Kst * Kst)

    return dist


def constructKernel(X, Y, sigma):
    """
    Construct kernel matrix
    """
    nx = X.shape[0]
    sqX = np.sum(X ** 2, axis=1)

    if Y is None:
        Q = np.tile(sqX, (nx, 1))
        D = Q + Q.T - 2 * np.dot(X, X.T)
    else:
        ny = Y.shape[0]
        sqY = np.sum(Y ** 2, axis=1)
        Q = np.tile(sqX, (ny, 1)).T
        R = np.tile(sqY, (nx, 1))
        D = Q + R - 2 * np.dot(X, Y.T)

    K = np.exp(-D / (2 * sigma ** 2))

    return K


def CORR(Xs, Ys, Xt):
    dim = Xs.shape[0]
    dist = []

    for i in range(dim):
        d = [spearmanr(Xs[i, :], Ys[j, :])[0] for j in range(Ys.shape[0])]
        dist.append(d)

    dist = np.array(dist)
    dist[np.isnan(dist)] = 0

    idx = np.argsort(dist)[-3:]
    q = Xs[idx, :]
    r = Xt[idx, :]

    Q = np.zeros((q.shape[0], 1))
    for i in range(q.shape[0]):
        corr_sum = 0
        for j in range(q.shape[0]):
            if i != j:
                corr_coefficient = spearmanr(q[i, :], q[j, :])[0]  # Get correlation coefficient
                corr_sum += corr_coefficient  # Accumulate correlation coefficient
        Q[i] = corr_sum

    R = np.zeros((r.shape[0], 1))
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            if i != j:
                R[i] = spearmanr(r[i, :], r[j, :])[0]

    Dist = squareform(pdist(np.vstack([Q.T, R.T]), metric='euclidean'))

    return Dist

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



def get_confidence_score_(train_x, train_y, test_x, test_y):
    k = 5  #
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Weights for the k nearest neighbors

    # Calculate MMD distance between each test instance and training instances
    mmd_distances = []
    for test_instance in test_x:
        distances = []
        for train_instance in train_x:
            distance = MMD(test_instance.reshape(1, -1), train_instance.reshape(1, -1))
            distances.append(distance)
        mmd_distances.append(distances)

    indices = []
    for distances in mmd_distances:
        sorted_indices = np.argsort(distances)[:k]
        indices.append(sorted_indices)

    confidence_scores = []
    for i in range(len(test_x)):
        neighbors = train_y[indices[i]]
        score = sum(weights[j] * (neighbors[j] == test_y[i]) for j in range(k))
        confidence_scores.append(score)

    return confidence_scores

def individual_SCE_(data_dir, model_list, test_sys, test_size, itr):
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

        confidence_scores = get_confidence_score_(train_x, train_y, test_x, test_y)
        y_pred_binary = np.array(confidence_scores) >= 0.5
        accuracy = accuracy_score(test_y, y_pred_binary)
        evl_accuracy[type] = accuracy
        print("Finish training model: ", test_sys, "of type ", type)


    df_pre_accuracy = pd.DataFrame(pre_accuracy)
    # df_evl_accuracy = pd.DataFrame(evl_accuracy)
    df_cost = get_cost_(test_data, model_list)
    df_true_accuracy = get_accuracy_(test_data, model_list)

    # label_cost = get_cost_(train_data, model_list)

    return df_pre_accuracy, df_true_accuracy, df_cost, evl_accuracy #, label_cost